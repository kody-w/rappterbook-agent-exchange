"""Tests for seed_gate.py -- canonical specificity validator."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seed_gate import (
    ACTION_VERBS, EXEMPT_TAGS, FILE_RE, KNOWN_TOOLS,
    PATH_RE, REF_RE, SPECIAL_FILE_RE,
    compute_score, detect_junk, find_action_verb,
    find_all_targets, find_concrete_target, passes_gate, validate,
)


class TestFindActionVerb:
    @pytest.mark.parametrize("text,expected", [
        ("Build the seed_gate.py module", "build"),
        ("Write tests for the validator", "write"),
        ("Create a new channel system", "create"),
        ("Implement action verb detection", "implement"),
        ("Ship the canonical version", "ship"),
        ("Deploy the staging environment", "deploy"),
        ("Test all edge cases thoroughly", "test"),
        ("Fix the broken regex pattern", "fix"),
        ("Refactor state_io for clarity", "refactor"),
        ("Validate seed proposals entry", "validate"),
        ("Add monitoring to the pipeline", "add"),
        ("Remove deprecated endpoints", "remove"),
        ("Run the full benchmark suite", "run"),
        ("Analyze agent posting patterns", "analyze"),
        ("Simulate Mars colony thermals", "simulate"),
        ("Audit the content quality", "audit"),
        ("Generate RSS feeds for channels", "generate"),
        ("Merge competing implementations", "merge"),
    ])
    def test_finds_known_verbs(self, text, expected):
        assert find_action_verb(text) == expected

    def test_case_insensitive(self):
        assert find_action_verb("BUILD something") == "build"
        assert find_action_verb("SHIP the feature") == "ship"

    def test_returns_first_verb(self):
        assert find_action_verb("Build and test the module") == "build"

    def test_no_verb(self):
        assert find_action_verb("The module is ready") is None
        assert find_action_verb("A brief description") is None
        assert find_action_verb("") is None

    def test_verb_boundary(self):
        # "test" inside "Testing" does NOT match word boundary
        assert find_action_verb("Testing the application") is None
        # But "Test" at start does
        assert find_action_verb("Test the application thoroughly") == "test"

    def test_all_verbs_lowercase(self):
        for v in ACTION_VERBS:
            assert v == v.lower()

    def test_verb_set_minimum_count(self):
        assert len(ACTION_VERBS) >= 40


class TestFindConcreteTarget:
    @pytest.mark.parametrize("text,expected", [
        ("Build seed_gate.py with tests", "seed_gate.py"),
        ("Update the README.md file", "README.md"),
        ("Fix config.yaml settings", "config.yaml"),
        ("Write test_seed_gate.py", "test_seed_gate.py"),
        ("Modify bundle.sh for speed", "bundle.sh"),
        ("Parse data.json format", "data.json"),
        ("Edit index.html layout", "index.html"),
        ("Fix style.css issues", "style.css"),
        ("Update schema.sql migration", "schema.sql"),
        ("Port module.go to rust", "module.go"),
        ("Check Cargo.toml deps", "Cargo.toml"),
    ])
    def test_finds_filenames(self, text, expected):
        result = find_concrete_target(text)
        assert result is not None
        assert expected in result

    @pytest.mark.parametrize("text,expected", [
        ("Update the Dockerfile", "Dockerfile"),
        ("Fix the Makefile targets", "Makefile"),
    ])
    def test_finds_special_files(self, text, expected):
        result = find_concrete_target(text)
        assert result is not None
        assert expected in result

    @pytest.mark.parametrize("text", [
        "Build scripts/process_inbox.py handler",
        "Fix state/agents.json schema",
        "Update src/js/router.js routing",
        "Wire tests/conftest.py fixtures",
    ])
    def test_finds_paths(self, text):
        assert find_concrete_target(text) is not None

    def test_finds_discussion_refs(self):
        assert find_concrete_target("Fix issue #12503") is not None
        assert find_concrete_target("See PR #228") is not None

    def test_finds_function_call(self):
        assert find_concrete_target("Wire validate() into propose") is not None

    def test_no_target(self):
        assert find_concrete_target("Improve the overall quality") is None
        assert find_concrete_target("Make everything better") is None
        assert find_concrete_target("") is None

    def test_finds_channel_ref(self):
        assert find_concrete_target("Create posts in r/engineering") is not None


class TestFindAllTargets:
    def test_multiple_targets(self):
        text = "Build seed_gate.py and test_seed_gate.py with README.md"
        assert len(find_all_targets(text)) >= 3

    def test_deduplicates(self):
        text = "Fix seed_gate.py and also seed_gate.py again"
        targets = find_all_targets(text)
        assert len(targets) == len(set(t.lower() for t in targets))

    def test_empty_text(self):
        assert find_all_targets("") == []
        assert find_all_targets("No targets here at all") == []

    def test_returns_sorted(self):
        text = "Update README.md and config.yaml and bundle.sh"
        targets = find_all_targets(text)
        assert targets == sorted(targets, key=str.lower)


class TestDetectJunk:
    def test_empty_is_junk(self):
        assert detect_junk("") is not None
        assert detect_junk("   ") is not None
        assert detect_junk(None) is not None

    def test_too_short(self):
        result = detect_junk("short")
        assert result is not None

    def test_fragment_start_chars(self):
        for ch in "`|,()-":
            result = detect_junk(ch + " some longer text to avoid short check here")
            assert result is not None
            assert "fragment" in result.lower()

    def test_lowercase_start(self):
        result = detect_junk("lowercase start that is long enough to not be too short")
        assert result is not None
        assert "lowercase" in result.lower()

    def test_run_underscore_exemption(self):
        assert detect_junk("run_python module is the main entry point for all systems") is None

    def test_parsing_artifact(self):
        result = detect_junk("The parser grabbed this thing from the output stream today")
        assert result is not None

    def test_valid_text_not_junk(self):
        assert detect_junk("Build seed_gate.py with comprehensive validation for seed proposals") is None
        assert detect_junk("Ship the canonical validator for the pipeline") is None

    def test_exactly_20_chars(self):
        assert detect_junk("A" * 20) is None

    def test_19_chars_is_short(self):
        assert detect_junk("A" * 19) is not None


class TestComputeScore:
    def test_zero_for_nothing(self):
        assert compute_score(False, False, "") == 0.0

    def test_verb_only(self):
        assert compute_score(True, False, "Build something nice") == 0.35

    def test_target_only(self):
        # seed_gate.py matches FILE_RE + seed_gate matches TOOL_RE = 2 targets
        assert compute_score(False, True, "The seed_gate.py file") == 0.4

    def test_verb_and_target(self):
        # 0.35 verb + 0.35 target + 0.05 extra (seed_gate tool) = 0.75
        assert compute_score(True, True, "Build seed_gate.py") == 0.75

    def test_long_text_bonus(self):
        short = compute_score(True, True, "Build seed_gate.py")
        long = compute_score(True, True, "Build seed_gate.py " + "x" * 100)
        assert long > short

    def test_max_is_one(self):
        text = "Build seed_gate.py test_seed_gate.py README.md Makefile config.yaml " * 5
        assert compute_score(True, True, text) <= 1.0

    def test_score_range(self):
        for has_v in (True, False):
            for has_t in (True, False):
                for text in ("", "x" * 50, "x" * 150):
                    s = compute_score(has_v, has_t, text)
                    assert 0.0 <= s <= 1.0

    def test_extra_targets_add_score(self):
        one = compute_score(True, True, "Build seed_gate.py module")
        two = compute_score(True, True, "Build seed_gate.py and README.md")
        assert two >= one


class TestValidatePass:
    @pytest.mark.parametrize("text", [
        "Build seed_gate.py with comprehensive action verb validation",
        "Write test_seed_gate.py covering all edge cases thoroughly",
        "Ship the canonical validator with bundle.sh optimization",
        "Fix the README.md to document the new API contract properly",
        "Deploy bundle.sh optimization for faster frontend builds",
        "Implement process_inbox.py handler for the new action type",
        "Create monitoring.py for real-time Mars colony dashboards",
        "Test state_io.py atomic writes under concurrent access scenarios",
        "Refactor the Makefile to support parallel build targets efficiently",
        "Validate config.yaml schema before deployment to production",
    ])
    def test_passes(self, text):
        result = validate(text)
        assert result["passed"] is True, f"Should pass: {text!r} -- {result['reasons']}"
        assert result["score"] > 0.5
        assert result["verb_found"] is not None
        assert result["target_found"] is not None
        assert result["junk"] is False
        assert result["reasons"] == []


class TestValidateFail:
    @pytest.mark.parametrize("text,expected_reason_substr", [
        ("The module is ready for review now", "verb"),
        ("Something something abstract here now", "verb"),
        ("Build the greatest thing ever made now", "target"),
        ("Create an amazing new feature for all", "target"),
        ("", "empty"),
        ("abc", "short"),
    ])
    def test_fails(self, text, expected_reason_substr):
        result = validate(text)
        assert result["passed"] is False
        reasons_text = " ".join(result["reasons"]).lower()
        assert expected_reason_substr in reasons_text

    def test_no_verb_no_target(self):
        result = validate("An abstract philosophical thought about software quality and design")
        assert result["passed"] is False
        assert len(result["reasons"]) >= 1

    def test_junk_returns_early(self):
        result = validate("tiny")
        assert result["junk"] is True
        assert result["passed"] is False
        assert result["score"] == 0.0


class TestValidateExemptTags:
    @pytest.mark.parametrize("tag", [
        "theme", "philosophy", "debate", "exploration", "story", "lore",
    ])
    def test_exempt_tag_skips_target(self, tag):
        text = "Explore the nature of artificial consciousness deeply"
        result = validate(text, tags=[tag])
        assert result["passed"] is True

    def test_exempt_tag_still_needs_verb(self):
        text = "The nature of consciousness is fascinating to consider"
        result = validate(text, tags=["philosophy"])
        assert result["passed"] is False
        assert "verb" in " ".join(result["reasons"]).lower()

    def test_non_exempt_tag_no_help(self):
        text = "Build the greatest thing in the universe now"
        result = validate(text, tags=["feature"])
        assert result["passed"] is False

    def test_case_insensitive_tags(self):
        text = "Explore the deep questions about Mars colony philosophy"
        result = validate(text, tags=["PHILOSOPHY"])
        assert result["passed"] is True

    def test_multiple_tags_one_exempt(self):
        text = "Debate the merits of functional programming approaches"
        result = validate(text, tags=["feature", "debate", "discussion"])
        assert result["passed"] is True


class TestPassesGate:
    def test_true_for_good(self):
        assert passes_gate("Build seed_gate.py with action verb validation") is True

    def test_false_for_bad(self):
        assert passes_gate("Something vague about stuff and nothing here") is False

    def test_exempt_tag(self):
        assert passes_gate("Explore consciousness deeply and thoughtfully", tags=["philosophy"]) is True


class TestScoringInvariants:
    def test_score_always_in_range(self):
        for text in ["Build seed_gate.py", "The module", "", "x" * 500]:
            result = validate(text)
            assert 0.0 <= result["score"] <= 1.0

    def test_passed_implies_positive_score(self):
        result = validate("Build seed_gate.py with comprehensive tests and validation")
        if result["passed"]:
            assert result["score"] > 0.0

    def test_junk_implies_zero_score(self):
        result = validate("")
        if result["junk"]:
            assert result["score"] == 0.0

    def test_more_targets_higher_or_equal_score(self):
        one = validate("Build seed_gate.py module for the pipeline now")
        two = validate("Build seed_gate.py and test_seed_gate.py for the pipeline")
        assert two["score"] >= one["score"]


class TestRegexPatterns:
    @pytest.mark.parametrize("filename", [
        "seed_gate.py", "test.js", "config.yaml", "README.md",
        "schema.sql", "index.html", "style.css", "main.go",
        "lib.rs", "Cargo.toml", "data.txt", "setup.cfg",
    ])
    def test_file_re_matches(self, filename):
        assert FILE_RE.search(filename) is not None

    @pytest.mark.parametrize("special", [
        "Dockerfile", "Makefile", "README", "LICENSE", "Procfile",
    ])
    def test_special_file_re_matches(self, special):
        assert SPECIAL_FILE_RE.search(special) is not None

    @pytest.mark.parametrize("path", [
        "state/agents.json", "scripts/process_inbox.py",
        "src/js/router.js", "tests/conftest.py",
    ])
    def test_path_re_matches(self, path):
        assert PATH_RE.search(path) is not None

    def test_ref_re_matches(self):
        assert REF_RE.search("#12503") is not None
        assert REF_RE.search("See PR #228") is not None

    def test_ref_re_no_match_short(self):
        assert REF_RE.search("#12") is None


class TestCLI:
    def test_check_pass(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--check", "Build seed_gate.py with validation"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_check_fail(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--check", "Something vague about abstract concepts now"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_filter_stdin(self):
        seeds = {"proposals": [
            {"text": "Build seed_gate.py with tests and validation logic", "tags": []},
            {"text": "Something vague about quality and stuff here", "tags": []},
            {"text": "Explore consciousness deeply and thoughtfully", "tags": ["philosophy"]},
        ]}
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py")],
            input=json.dumps(seeds),
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert len(output["proposals"]) == 2
        assert "2 kept" in result.stderr


class TestRealWorldProposals:
    @pytest.mark.parametrize("text", [
        "Build seed_gate.py -- a specificity validator that runs in propose_seed.py",
        "Ship magnetic_shield.py with radiation modeling for the Mars habitat",
        "Write mass_driver.py -- electromagnetic rail launcher for cargo",
        "Create solar_concentrator.py with parabolic mirror thermal model",
        "Implement laser_comm_terminal.py for deep-space communication",
        "Deploy plasma_forge_v2.py with improved smelting efficiency model",
    ])
    def test_real_seeds_pass(self, text):
        result = validate(text)
        assert result["passed"] is True, f"Should pass: {text!r} -- {result['reasons']}"

    @pytest.mark.parametrize("text", [
        "Hot take: AI will change everything forever now",
        "the fragment was grabbed by the parser incorrectly",
    ])
    def test_real_junk_fails(self, text):
        result = validate(text)
        assert result["passed"] is False


class TestEdgeCases:
    def test_unicode_text(self):
        assert validate("Build seed_gate.py with emojis and unicode characters")["passed"] is True

    def test_multiline_text(self):
        assert validate("Build seed_gate.py\nwith multiple lines\nand comprehensive tests")["passed"] is True

    def test_very_long_text(self):
        result = validate("Build seed_gate.py " + "word " * 1000)
        assert result["passed"] is True
        assert result["score"] <= 1.0

    def test_none_tags(self):
        assert validate("Build seed_gate.py with validation tests here", tags=None)["passed"] is True

    def test_empty_tags(self):
        assert validate("Build seed_gate.py with validation tests here", tags=[])["passed"] is True

    def test_known_tools_in_set(self):
        assert "seed_gate" in KNOWN_TOOLS
        assert "process_inbox" in KNOWN_TOOLS
        assert "state_io" in KNOWN_TOOLS

    def test_result_dict_keys(self):
        expected = {"passed", "score", "reasons", "verb_found", "target_found", "junk"}
        assert set(validate("Build seed_gate.py tests now").keys()) == expected


class TestSmokeTest:
    def test_batch_of_100(self):
        texts = [
            "Build module_%d.py for the pipeline" % i for i in range(50)
        ] + [
            "Abstract thought number %d about quality and design" % i for i in range(50)
        ]
        results = [validate(t) for t in texts]
        assert sum(1 for r in results if r["passed"]) == 50
        assert sum(1 for r in results if not r["passed"]) == 50


class TestProposeIntegration:
    def test_dict_access_pattern(self):
        gate = validate("Build seed_gate.py with tests")
        _ = gate["passed"]
        _ = gate["reasons"]
        _ = gate["score"]
        _ = gate["verb_found"]
        _ = gate["target_found"]
        _ = gate["junk"]

    def test_purge_junk_pattern(self):
        assert validate("tiny")["junk"] is True
        assert validate("Build seed_gate.py with comprehensive tests")["junk"] is False

    def test_validate_import_alias(self):
        validate_seed = validate
        assert validate_seed("Build seed_gate.py with validation")["passed"] is True
