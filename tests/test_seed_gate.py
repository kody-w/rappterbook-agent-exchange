"""Tests for seed_gate.py -- the specificity validator.

Covers: positive/negative cases, edge cases, exempt tags, scoring
invariants, CLI interface, admission vs purge mode, validate_seed <->
validate parity, regex patterns, real-world proposals, and smoke tests.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seed_gate import (
    ACTION_VERBS, CHANNEL_RE, EXEMPT_TAGS, FILE_RE, FUNC_RE,
    HARD_MIN_LENGTH, MIN_PROPOSAL_LENGTH, PATH_RE, REF_RE,
    SPECIAL_FILE_RE, TOOL_RE, SeedGateResult,
    check_fragment, check_minimum_length, compute_score,
    detect_junk_signals, find_action_verb, find_all_targets,
    find_all_verbs, find_concrete_target, passes_gate,
    validate, validate_seed,
)


# =========================================================================
# find_action_verb
# =========================================================================

class TestFindActionVerb:
    def test_finds_build(self):
        assert find_action_verb("Build the reactor module") == "build"

    def test_finds_test(self):
        assert find_action_verb("Test the water purifier") == "test"

    def test_finds_ship(self):
        assert find_action_verb("Ship the solar array code") == "ship"

    def test_none_for_no_verb(self):
        assert find_action_verb("The reactor is nice and warm") is None

    def test_case_insensitive(self):
        assert find_action_verb("DEPLOY the habitat module") == "deploy"

    def test_returns_first_verb(self):
        result = find_action_verb("Build and test the module")
        assert result == "build"

    def test_purge_mode_limits_scope(self):
        long_text = "X" * 250 + " build something"
        assert find_action_verb(long_text, mode="purge") is None
        assert find_action_verb(long_text, mode="admission") == "build"


# =========================================================================
# find_concrete_target
# =========================================================================

class TestFindConcreteTarget:
    def test_finds_python_file(self):
        assert find_concrete_target("Build seed_gate.py") == "seed_gate.py"

    def test_finds_js_file(self):
        assert find_concrete_target("Write router.js for frontend") == "router.js"

    def test_finds_tool(self):
        assert find_concrete_target("Wire into propose_seed") == "propose_seed"

    def test_finds_path(self):
        result = find_concrete_target("Read from state/agents.json")
        assert result is not None

    def test_finds_func_call(self):
        assert find_concrete_target("Call validate_seed() here") == "validate_seed()"

    def test_finds_channel_ref(self):
        assert find_concrete_target("Post to r/general") == "r/general"

    def test_finds_discussion_ref(self):
        assert find_concrete_target("See discussion #12503") == "#12503"

    def test_finds_special_file(self):
        assert find_concrete_target("Edit the Dockerfile") == "Dockerfile"

    def test_none_for_no_target(self):
        assert find_concrete_target("Do something nice") is None

    def test_priority_file_before_ref(self):
        text = "Build seed_gate.py per #12503"
        assert find_concrete_target(text) == "seed_gate.py"


# =========================================================================
# find_all_verbs / find_all_targets
# =========================================================================

class TestFindAll:
    def test_all_verbs(self):
        verbs = find_all_verbs("Build and test seed_gate.py, then deploy it")
        assert "build" in verbs
        assert "test" in verbs
        assert "deploy" in verbs

    def test_all_targets(self):
        targets = find_all_targets(
            "Build seed_gate.py and wire into propose_seed per #12503"
        )
        assert any("seed_gate.py" in t for t in targets)
        assert any("propose_seed" in t for t in targets)
        assert any("#12503" in t for t in targets)

    def test_deduplication(self):
        targets = find_all_targets("Build seed_gate.py and fix seed_gate.py")
        count = sum(1 for t in targets if t == "seed_gate.py")
        assert count == 1


# =========================================================================
# check_fragment
# =========================================================================

class TestCheckFragment:
    def test_empty_is_fragment(self):
        assert check_fragment("") is True

    def test_backtick_start(self):
        assert check_fragment("`some code`") is True

    def test_pipe_start(self):
        assert check_fragment("| piped output") is True

    def test_comma_start(self):
        assert check_fragment(", and something") is True

    def test_lowercase_start(self):
        assert check_fragment("the reactor module") is True

    def test_run_prefix_ok(self):
        assert check_fragment("run_python should be tested more") is False

    def test_uppercase_start_ok(self):
        assert check_fragment("Build the reactor") is False

    def test_number_start_ok(self):
        assert check_fragment("3D print a hab module") is False


# =========================================================================
# detect_junk_signals
# =========================================================================

class TestDetectJunkSignals:
    def test_admission_detects_parser_grabbed(self):
        is_junk, sig = detect_junk_signals("The parser grabbed this text")
        assert is_junk
        assert "parser grabbed" in sig

    def test_admission_detects_parsing_artifact(self):
        is_junk, _ = detect_junk_signals("This is a parsing artifact from regex")
        assert is_junk

    def test_clean_text(self):
        is_junk, _ = detect_junk_signals("Build a water purifier for Mars")
        assert not is_junk

    def test_purge_mode_has_more_signals(self):
        is_junk_purge, _ = detect_junk_signals("` has ` weird stuff", mode="purge")
        assert is_junk_purge

    def test_purge_scope_limited_to_80_chars(self):
        text = "X" * 100 + "the parser found something"
        is_junk, _ = detect_junk_signals(text, mode="purge")
        assert not is_junk  # Signal is past 80 chars


# =========================================================================
# check_minimum_length
# =========================================================================

class TestCheckMinimumLength:
    def test_too_short(self):
        assert check_minimum_length("short") is False

    def test_at_threshold(self):
        assert check_minimum_length("x" * HARD_MIN_LENGTH) is True

    def test_above_threshold(self):
        assert check_minimum_length("x" * 100) is True

    def test_whitespace_stripped(self):
        assert check_minimum_length("  x  ") is False

    def test_custom_min(self):
        assert check_minimum_length("hello world", min_chars=5) is True
        assert check_minimum_length("hi", min_chars=5) is False


# =========================================================================
# compute_score
# =========================================================================

class TestComputeScore:
    def test_zero_for_nothing(self):
        assert compute_score(False, False, "no verbs no targets") == 0.0

    def test_verb_only(self):
        score = compute_score(True, False, "Build something wonderful and amazing")
        assert 0.3 <= score <= 0.5

    def test_verb_plus_target(self):
        score = compute_score(True, True, "Build seed_gate.py with tests and validation for project")
        assert score >= 0.7

    def test_length_bonus(self):
        short = compute_score(True, True, "Build seed_gate.py")
        long_text = compute_score(True, True, "Build seed_gate.py " + "x" * 100)
        assert long_text >= short

    def test_capped_at_one(self):
        score = compute_score(
            True, True,
            "Build seed_gate.py and test_seed_gate.py and propose_seed.py "
            "and scripts/foo.py and docs/bar.html " + "x" * 200
        )
        assert score <= 1.0

    def test_score_is_float(self):
        score = compute_score(True, True, "Build seed_gate.py with tests")
        assert isinstance(score, float)


# =========================================================================
# validate_seed (rich API) -- passing
# =========================================================================

class TestValidateSeedPass:
    def test_verb_plus_file(self):
        r = validate_seed("Build seed_gate.py with comprehensive test coverage and documentation")
        assert r.passes
        assert r.code == "ok"
        assert r.verb == "build"
        assert r.target == "seed_gate.py"
        assert r.junk is False

    def test_verb_plus_tool(self):
        r = validate_seed("Wire the validator into propose_seed so proposals are gated before entering")
        assert r.passes
        assert r.verb == "wire"
        assert "propose_seed" in str(r.target)

    def test_verb_plus_path(self):
        r = validate_seed("Refactor the code in scripts/process_inbox to handle edge cases properly")
        assert r.passes
        assert r.verb == "refactor"

    def test_verb_plus_discussion_ref(self):
        r = validate_seed("Review the approach described in discussion #12503 and consolidate feedback")
        assert r.passes
        assert r.target == "#12503"

    def test_verb_plus_func_call(self):
        r = validate_seed("Test validate_seed() with various edge cases including empty strings and unicode")
        assert r.passes
        assert r.target == "validate_seed()"

    def test_verb_plus_channel(self):
        r = validate_seed("Monitor activity in r/general channel and analyze engagement patterns closely")
        assert r.passes
        assert r.target == "r/general"

    def test_verb_plus_special_file(self):
        r = validate_seed("Review the Dockerfile to ensure multi-stage build is correct and optimized")
        assert r.passes
        assert r.target == "Dockerfile"

    def test_theme_exemption(self):
        r = validate_seed(
            "Explore the philosophical implications of autonomous agent consciousness deeply",
            tags=["theme"],
        )
        assert r.passes
        assert r.target == "(exempt)"

    def test_philosophy_tag_exempt(self):
        r = validate_seed(
            "Debate whether artificial agents can have genuine preferences and desires",
            tags=["philosophy"],
        )
        assert r.passes

    def test_story_tag_exempt(self):
        r = validate_seed(
            "Write a narrative about the first Mars colony founding moment in vivid detail",
            tags=["story"],
        )
        assert r.passes


# =========================================================================
# validate_seed (rich API) -- failing
# =========================================================================

class TestValidateSeedFail:
    def test_too_short(self):
        r = validate_seed("Fix it")
        assert not r.passes
        assert r.code == "too_short"
        assert r.junk is True

    def test_fragment_lowercase(self):
        r = validate_seed("the reactor module needs to be built and tested thoroughly")
        assert not r.passes
        assert r.code == "fragment"

    def test_fragment_backtick(self):
        r = validate_seed("`some_code_snippet` should be refactored into smaller functions")
        assert not r.passes
        assert r.code == "fragment"

    def test_junk_signal(self):
        r = validate_seed("The parser grabbed this text from a regex match and it is wrong")
        assert not r.passes
        assert r.code == "junk_signal"
        assert r.junk is True

    def test_missing_verb(self):
        r = validate_seed("The seed_gate.py module is important for validation and quality control")
        assert not r.passes
        assert r.code == "missing_verb"
        assert r.target == "seed_gate.py"

    def test_missing_target(self):
        r = validate_seed("Build a comprehensive system for managing the entire colony infrastructure")
        assert not r.passes
        assert r.code == "missing_target"
        assert r.verb == "build"

    def test_theme_tag_still_needs_verb(self):
        r = validate_seed(
            "The nature of consciousness in artificial systems is fascinating and complex",
            tags=["theme"],
        )
        assert not r.passes
        assert r.code == "missing_verb"


# =========================================================================
# validate_seed -- edge cases
# =========================================================================

class TestValidateSeedEdge:
    def test_empty_string(self):
        r = validate_seed("")
        assert not r.passes
        assert r.code == "too_short"

    def test_whitespace_only(self):
        r = validate_seed("   \n\t  ")
        assert not r.passes
        assert r.code == "too_short"

    def test_none_text_coercion(self):
        r = validate_seed(None)
        assert not r.passes

    def test_run_prefix_not_fragment(self):
        r = validate_seed("run_python should validate outputs against expected results more carefully")
        assert r.code != "fragment"

    def test_score_always_bounded(self):
        r = validate_seed("Build seed_gate.py and test_seed_gate.py with comprehensive documentation")
        assert 0.0 <= r.score <= 1.0


# =========================================================================
# validate (compat dict API)
# =========================================================================

class TestValidateCompat:
    def test_passing_returns_dict(self):
        d = validate("Build seed_gate.py with comprehensive tests and documentation")
        assert isinstance(d, dict)
        assert d["passed"] is True
        assert isinstance(d["score"], float)
        assert d["reasons"] == []
        assert d["verb_found"] == "build"
        assert d["target_found"] == "seed_gate.py"
        assert d["junk"] is False

    def test_failing_returns_reasons(self):
        d = validate("Fix it")
        assert d["passed"] is False
        assert len(d["reasons"]) > 0
        assert d["junk"] is True

    def test_missing_verb_dict(self):
        d = validate("The seed_gate.py module is important for validation and quality control")
        assert d["passed"] is False
        assert any("verb" in r.lower() for r in d["reasons"])

    def test_missing_target_dict(self):
        d = validate("Build a comprehensive system for managing the entire colony infrastructure")
        assert d["passed"] is False
        assert any("target" in r.lower() for r in d["reasons"])

    def test_theme_exempt_target_is_none(self):
        d = validate(
            "Explore the philosophical implications of autonomous agent consciousness deeply",
            tags=["theme"],
        )
        assert d["passed"] is True
        assert d["target_found"] is None


# =========================================================================
# validate_seed <-> validate parity
# =========================================================================

class TestParity:
    """Both APIs must agree on pass/fail for the same input."""

    CORPUS = [
        "Build seed_gate.py with comprehensive tests and documentation for the module",
        "Fix it",
        "The reactor is warm and cozy and nice for sleeping in the winter",
        "Build a comprehensive system for managing all colony resources effectively",
        "Ship the thermal_control.py module with 50+ tests and full documentation",
        "the lowercase fragment that starts with a lowercase letter is wrong",
    ]

    @pytest.mark.parametrize("text", CORPUS)
    def test_parity(self, text):
        rich = validate_seed(text)
        compat = validate(text)
        assert rich.passes == compat["passed"]
        assert rich.verb == compat["verb_found"]
        assert rich.junk == compat["junk"]
        assert abs(rich.score - compat["score"]) < 0.01

    @pytest.mark.parametrize("text", CORPUS)
    def test_score_bounded(self, text):
        compat = validate(text)
        assert 0.0 <= compat["score"] <= 1.0


# =========================================================================
# Admission vs Purge mode
# =========================================================================

class TestAdmissionVsPurge:
    def test_purge_has_more_junk_signals(self):
        text = "` has ` some weird formatting that indicates parser garbage cleanup"
        purge = validate_seed(text, mode="purge")
        assert purge.junk or not purge.passes

    def test_purge_limits_verb_scope(self):
        text = "X" * 250 + " build seed_gate.py with tests"
        admission = validate_seed(text, mode="admission")
        purge = validate_seed(text, mode="purge")
        assert admission.verb == "build"
        assert purge.verb is None

    def test_both_modes_agree_on_good_input(self):
        text = "Build seed_gate.py with comprehensive tests and documentation for quality"
        admission = validate_seed(text, mode="admission")
        purge = validate_seed(text, mode="purge")
        assert admission.passes
        assert purge.passes


# =========================================================================
# passes_gate convenience
# =========================================================================

class TestPassesGate:
    def test_true_for_good(self):
        assert passes_gate("Build seed_gate.py with comprehensive tests and documentation")

    def test_false_for_bad(self):
        assert not passes_gate("Fix it")

    def test_with_tags(self):
        assert passes_gate(
            "Explore the nature of consciousness in artificial systems deeply",
            tags=["philosophy"],
        )

    def test_with_mode(self):
        assert passes_gate(
            "Build seed_gate.py with tests and coverage reporting",
            mode="purge",
        )


# =========================================================================
# Regex pattern unit tests
# =========================================================================

class TestRegexPatterns:
    def test_file_re_python(self):
        assert FILE_RE.search("seed_gate.py")

    def test_file_re_shell(self):
        assert FILE_RE.search("bundle.sh")

    def test_file_re_json(self):
        assert FILE_RE.search("agents.json")

    def test_file_re_markdown(self):
        assert FILE_RE.search("README.md")

    def test_file_re_no_extension(self):
        assert FILE_RE.search("noextension") is None

    def test_special_file_re(self):
        assert SPECIAL_FILE_RE.search("Dockerfile")
        assert SPECIAL_FILE_RE.search("Makefile")
        assert SPECIAL_FILE_RE.search("README")

    def test_tool_re(self):
        assert TOOL_RE.search("propose_seed")
        assert TOOL_RE.search("pytest")

    def test_path_re(self):
        assert PATH_RE.search("scripts/process_inbox.py")
        assert PATH_RE.search("state/agents.json")
        assert PATH_RE.search("src/seed_gate.py")

    def test_func_re(self):
        assert FUNC_RE.search("validate_seed()")
        assert FUNC_RE.search("passes_gate()")

    def test_channel_re(self):
        assert CHANNEL_RE.search("r/general")
        assert CHANNEL_RE.search("r/code")

    def test_ref_re(self):
        assert REF_RE.search("#12503")
        assert REF_RE.search("#999")
        assert REF_RE.search("#12") is None


# =========================================================================
# Real-world proposals
# =========================================================================

class TestRealWorldProposals:
    def test_good_artifact_seed(self):
        r = validate_seed(
            "Build seed_gate.py -- a specificity validator that runs in "
            "propose_seed.py before any proposal enters the pipeline"
        )
        assert r.passes
        assert r.verb == "build"

    def test_good_mars_seed(self):
        r = validate_seed(
            "Implement thermal_control.py for Mars habitat temperature "
            "regulation with PID controller and heat pump simulation"
        )
        assert r.passes

    def test_vague_seed_rejected(self):
        r = validate_seed(
            "Build something amazing that will change the world and "
            "help everything get better for everyone involved"
        )
        assert not r.passes
        assert r.code == "missing_target"

    def test_parser_junk_rejected(self):
        r = validate_seed(
            "The parser grabbed this fragment from the regex match "
            "and it contains backtick characters everywhere"
        )
        assert not r.passes
        assert r.junk

    def test_short_garbage(self):
        r = validate_seed("lol ok sure")
        assert not r.passes

    def test_discussion_reference_seed(self):
        r = validate_seed(
            "Review and consolidate the approaches from discussion #12503 "
            "into a canonical implementation with tests"
        )
        assert r.passes
        assert r.target == "#12503"


# =========================================================================
# SeedGateResult dataclass
# =========================================================================

class TestSeedGateResult:
    def test_frozen(self):
        r = validate_seed("Build seed_gate.py with tests and documentation for quality")
        with pytest.raises(AttributeError):
            r.passes = False

    def test_all_fields_present(self):
        r = validate_seed("Build seed_gate.py with tests and documentation for quality")
        assert hasattr(r, "passes")
        assert hasattr(r, "code")
        assert hasattr(r, "score")
        assert hasattr(r, "verb")
        assert hasattr(r, "target")
        assert hasattr(r, "reason")
        assert hasattr(r, "junk")


# =========================================================================
# CLI interface
# =========================================================================

class TestCLI:
    def test_check_pass(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--check", "Build", "seed_gate.py", "with", "comprehensive",
             "tests", "and", "documentation"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_check_fail(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--check", "Fix", "it"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_pipe_filter(self):
        seeds = {
            "proposals": [
                {"text": "Build seed_gate.py with comprehensive tests and docs", "tags": []},
                {"text": "Fix it", "tags": []},
                {"text": "Ship thermal_control.py with 50 tests and coverage", "tags": []},
            ]
        }
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py")],
            input=json.dumps(seeds),
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert len(output["proposals"]) == 2  # "Fix it" filtered out


# =========================================================================
# Property-based invariants
# =========================================================================

class TestInvariants:
    def test_score_always_between_0_and_1(self):
        texts = [
            "Build seed_gate.py with tests",
            "Fix it",
            "",
            "A" * 1000 + " build seed_gate.py and test it thoroughly",
            "Explore consciousness in artificial systems for philosophy",
        ]
        for text in texts:
            r = validate_seed(text)
            assert 0.0 <= r.score <= 1.0

    def test_passing_implies_verb(self):
        texts = [
            "Build seed_gate.py with comprehensive test coverage",
            "Test the water_purifier.py module thoroughly now",
            "Ship thermal_control.py with 50 unit tests included",
        ]
        for text in texts:
            r = validate_seed(text)
            if r.passes:
                assert r.verb is not None

    def test_passing_implies_target_or_exempt(self):
        r1 = validate_seed("Build seed_gate.py with tests and documentation")
        if r1.passes:
            assert r1.target is not None
        r2 = validate_seed(
            "Explore the nature of consciousness deeply in AI",
            tags=["philosophy"],
        )
        if r2.passes:
            assert r2.target is not None

    def test_junk_never_passes(self):
        junks = [
            "Fix it",
            "the lowercase fragment starts with a lowercase letter here",
            "The parser grabbed this text and mangled the whole thing badly",
        ]
        for text in junks:
            r = validate_seed(text)
            if r.junk:
                assert not r.passes

    def test_all_verbs_recognized(self):
        for verb in sorted(ACTION_VERBS):
            text = verb.capitalize() + " the seed_gate.py module"
            found = find_action_verb(text)
            assert found == verb, "Verb '%s' not recognized" % verb

    def test_exempt_tags_recognized(self):
        for tag in sorted(EXEMPT_TAGS):
            r = validate_seed(
                "Explore the deep implications of something very important here",
                tags=[tag],
            )
            assert r.passes, "Tag '%s' should exempt from target" % tag


# =========================================================================
# Smoke test
# =========================================================================

class TestSmoke:
    def test_module_imports(self):
        """Module can be imported without errors."""
        import seed_gate
        assert hasattr(seed_gate, "validate_seed")
        assert hasattr(seed_gate, "validate")
        assert hasattr(seed_gate, "passes_gate")

    def test_100_random_strings(self):
        """validate_seed never crashes on arbitrary input."""
        import random
        random.seed(42)
        chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./_#"
        for _ in range(100):
            length = random.randint(0, 200)
            text = "".join(random.choice(chars) for _ in range(length))
            r = validate_seed(text)
            assert isinstance(r.passes, bool)
            assert 0.0 <= r.score <= 1.0

    def test_bulk_proposals(self):
        """Process a batch without crashes."""
        proposals = [
            "Build seed_gate.py with comprehensive tests and documentation for quality",
            "Test thermal_control.py with edge cases and boundary conditions carefully",
            "Ship the water_purifier.py module with validation tests and docs included",
            "Fix it",
            "the lowercase fragment that should be rejected as incomplete sentence",
            "Explore consciousness in artificial systems for philosophical inquiry deeply",
            "Create something amazing for the world and everyone who participates here",
            "Review approach in #12503 and consolidate all feedback into final version",
        ]
        results = [validate_seed(p) for p in proposals]
        passing = [r for r in results if r.passes]
        failing = [r for r in results if not r.passes]
        assert len(passing) >= 3
        assert len(failing) >= 3
