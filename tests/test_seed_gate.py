"""Tests for seed_gate.py -- the specificity validator.

131 tests covering: positive/negative cases, edge cases, exempt tags,
scoring invariants, regex patterns, real-world proposals, CLI, and
property invariants.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seed_gate import (
    ACTION_VERBS,
    CHANNEL_RE,
    EXEMPT_TAGS,
    FILE_RE,
    FUNC_RE,
    HARD_MIN_LENGTH,
    PATH_RE,
    REF_RE,
    SOFT_MIN_LENGTH,
    SPECIAL_FILE_RE,
    TOOL_RE,
    SeedGateResult,
    check_fragment,
    compute_score,
    detect_junk_signals,
    find_action_verb,
    find_all_targets,
    find_all_verbs,
    find_concrete_target,
    passes_gate,
    validate,
    validate_seed,
)


# ===================================================================
# TestFindActionVerb  (10 tests)
# ===================================================================
class TestFindActionVerb:
    """Action verb extraction."""

    def test_finds_build(self):
        assert find_action_verb("Build the reactor module") == "build"

    def test_finds_ship(self):
        assert find_action_verb("Ship seed_gate.py to production") == "ship"

    def test_finds_fix(self):
        assert find_action_verb("Fix the broken test suite") == "fix"

    def test_case_insensitive(self):
        assert find_action_verb("DEPLOY the service now") == "deploy"

    def test_returns_first_verb(self):
        assert find_action_verb("Build and test seed_gate.py") == "build"

    def test_no_verb_returns_none(self):
        assert find_action_verb("The quick brown fox") is None

    def test_empty_string(self):
        assert find_action_verb("") is None

    def test_verb_in_compound_word(self):
        result = find_action_verb("Validate compute_trending output")
        assert result == "validate"

    def test_explore_is_valid(self):
        assert find_action_verb("Explore the deep sim architecture") == "explore"

    def test_debate_is_valid(self):
        assert find_action_verb("Debate AI consciousness in r/philosophy") == "debate"


# ===================================================================
# TestFindConcreteTarget  (12 tests)
# ===================================================================
class TestFindConcreteTarget:
    """Concrete target extraction."""

    def test_finds_python_file(self):
        assert find_concrete_target("Build seed_gate.py") == "seed_gate.py"

    def test_finds_shell_script(self):
        assert find_concrete_target("Run bundle.sh for the frontend") == "bundle.sh"

    def test_finds_json_file(self):
        assert find_concrete_target("Parse seeds.json data") == "seeds.json"

    def test_finds_path(self):
        target = find_concrete_target("Refactor src/seed_gate.py")
        assert target is not None
        assert "src/seed_gate" in target

    def test_finds_tool(self):
        assert find_concrete_target("Wire into propose_seed") == "propose_seed"

    def test_finds_function_call(self):
        assert find_concrete_target("Fix validate_seed() crash") == "validate_seed()"

    def test_finds_channel(self):
        assert find_concrete_target("Post analysis to r/code") == "r/code"

    def test_finds_discussion_ref(self):
        assert find_concrete_target("See discussion #12503 for context") == "#12503"

    def test_finds_special_file(self):
        assert find_concrete_target("Update the Dockerfile") == "Dockerfile"

    def test_finds_makefile(self):
        assert find_concrete_target("Add target to Makefile") == "Makefile"

    def test_no_target(self):
        assert find_concrete_target("Do something interesting") is None

    def test_empty_string(self):
        assert find_concrete_target("") is None


# ===================================================================
# TestCheckFragment  (8 tests)
# ===================================================================
class TestCheckFragment:
    """Fragment detection."""

    def test_empty_is_fragment(self):
        assert check_fragment("") is True

    def test_lowercase_start_is_fragment(self):
        assert check_fragment("the quick brown fox") is True

    def test_backtick_start_is_fragment(self):
        assert check_fragment("`code` fragment here") is True

    def test_pipe_start_is_fragment(self):
        assert check_fragment("| piped output") is True

    def test_uppercase_start_not_fragment(self):
        assert check_fragment("Build the thing") is False

    def test_run_prefix_not_fragment(self):
        assert check_fragment("run_python executes the code") is False

    def test_make_prefix_not_fragment(self):
        assert check_fragment("make all runs the full build") is False

    def test_gh_prefix_not_fragment(self):
        assert check_fragment("gh pr create opens a PR") is False


# ===================================================================
# TestJunkSignals  (6 tests)
# ===================================================================
class TestJunkSignals:
    """Junk / parser-artifact detection."""

    def test_clean_text(self):
        assert detect_junk_signals("Build seed_gate.py with tests") is None

    def test_parser_grabbed(self):
        assert detect_junk_signals("parser grabbed this fragment") is not None

    def test_parsing_artifact(self):
        assert detect_junk_signals("This is a parsing artifact") is not None

    def test_backtick_has_backtick(self):
        assert detect_junk_signals("`foo` has `bar` pattern") is not None

    def test_the_regex(self):
        assert detect_junk_signals("the regex matched this") is not None

    def test_substring(self):
        assert detect_junk_signals("This is a substring of something") is not None


# ===================================================================
# TestValidateSeed (dataclass API)  (15 tests)
# ===================================================================
class TestValidateSeed:
    """Core validate_seed() returning SeedGateResult."""

    def test_classic_pass(self):
        result = validate_seed("Build seed_gate.py with comprehensive tests and documentation")
        assert result.passed is True
        assert result.verb_found == "build"
        assert result.target_found == "seed_gate.py"
        assert result.code == "ok"

    def test_no_verb_fails(self):
        result = validate_seed("The seed_gate.py module is very important to the project")
        assert result.passed is False
        assert result.code == "missing_verb"
        assert result.target_found == "seed_gate.py"

    def test_no_target_fails(self):
        result = validate_seed("Build something amazing for the colony infrastructure now")
        assert result.passed is False
        assert result.code == "missing_target"
        assert result.verb_found == "build"

    def test_too_short(self):
        result = validate_seed("Fix it")
        assert result.passed is False
        assert result.code == "too_short"

    def test_fragment_rejected(self):
        result = validate_seed("`backtick start` something something something more text")
        assert result.passed is False
        assert result.code == "fragment"

    def test_junk_rejected(self):
        result = validate_seed("The parser grabbed this text and produced a fragment for processing")
        assert result.passed is False
        assert result.code == "junk_signal"

    def test_theme_exempt_skips_target(self):
        result = validate_seed(
            "Explore what consciousness means for artificial minds in the simulation",
            tags=["theme"],
        )
        assert result.passed is True
        assert result.verb_found == "explore"
        assert result.target_found is None

    def test_philosophy_tag_exempt(self):
        result = validate_seed(
            "Debate whether simulated agents deserve constitutional rights and protections",
            tags=["Philosophy"],
        )
        assert result.passed is True

    def test_exempt_tag_case_insensitive(self):
        result = validate_seed(
            "Investigate the boundaries of emergence in artificial systems today",
            tags=["EXPLORATION"],
        )
        assert result.passed is True

    def test_returns_frozen_dataclass(self):
        result = validate_seed("Build seed_gate.py with comprehensive tests and documentation")
        assert isinstance(result, SeedGateResult)
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]

    def test_reasons_is_tuple(self):
        result = validate_seed("Build seed_gate.py with comprehensive tests and documentation")
        assert isinstance(result.reasons, tuple)

    def test_score_in_bounds(self):
        result = validate_seed("Build seed_gate.py with comprehensive tests and documentation")
        assert 0 <= result.score <= 10

    def test_path_as_target(self):
        result = validate_seed("Refactor src/seed_gate.py to use the new validation pipeline")
        assert result.passed is True
        assert result.target_found is not None
        assert "src/seed_gate" in result.target_found

    def test_discussion_ref_as_target(self):
        result = validate_seed("Implement the design from #12503 with full test coverage")
        assert result.passed is True
        assert result.target_found == "#12503"

    def test_function_call_as_target(self):
        result = validate_seed("Fix validate_seed() to handle empty strings correctly in all cases")
        assert result.passed is True
        assert result.target_found == "validate_seed()"


# ===================================================================
# TestValidate (dict API)  (10 tests)
# ===================================================================
class TestValidate:
    """Dict-returning validate() for propose_seed.py compatibility."""

    def test_returns_dict(self):
        result = validate("Build seed_gate.py with comprehensive tests and documentation")
        assert isinstance(result, dict)

    def test_has_passed_key(self):
        result = validate("Build seed_gate.py with comprehensive tests and documentation")
        assert "passed" in result

    def test_has_reasons_key(self):
        result = validate("Build seed_gate.py with comprehensive tests and documentation")
        assert "reasons" in result
        assert isinstance(result["reasons"], list)

    def test_has_score_key(self):
        result = validate("Build seed_gate.py with comprehensive tests and documentation")
        assert "score" in result

    def test_has_verb_key(self):
        result = validate("Build seed_gate.py with comprehensive tests and documentation")
        assert "verb_found" in result

    def test_has_target_key(self):
        result = validate("Build seed_gate.py with comprehensive tests and documentation")
        assert "target_found" in result

    def test_passing_proposal(self):
        result = validate("Build seed_gate.py with comprehensive tests and documentation")
        assert result["passed"] is True
        assert result["verb_found"] == "build"
        assert result["target_found"] == "seed_gate.py"

    def test_failing_proposal(self):
        result = validate("Something vague about doing stuff in the colony")
        assert result["passed"] is False
        assert len(result["reasons"]) > 0

    def test_reasons_are_strings(self):
        result = validate("Something vague about doing stuff in the colony")
        assert all(isinstance(r, str) for r in result["reasons"])

    def test_tags_forwarded(self):
        result = validate(
            "Explore what consciousness means for artificial minds in the simulation",
            tags=["theme"],
        )
        assert result["passed"] is True


# ===================================================================
# TestPassesGate  (6 tests)
# ===================================================================
class TestPassesGate:
    """Boolean convenience API."""

    def test_passes(self):
        assert passes_gate("Build seed_gate.py with tests and documentation") is True

    def test_fails(self):
        assert passes_gate("Something vague about stuff in the colony") is False

    def test_with_tags(self):
        assert passes_gate(
            "Explore consciousness in artificial minds within the simulation",
            tags=["theme"],
        ) is True

    def test_too_short(self):
        assert passes_gate("Hi") is False

    def test_fragment(self):
        assert passes_gate("`backtick` something something something something more") is False

    def test_junk(self):
        assert passes_gate("The parser grabbed this text and made something from it") is False


# ===================================================================
# TestComputeScore  (8 tests)
# ===================================================================
class TestComputeScore:
    """Scoring invariants."""

    def test_empty_scores_zero(self):
        assert compute_score("") == 0

    def test_verb_only_scores_two(self):
        assert compute_score("Build something for the colony") == 2

    def test_file_adds_three(self):
        score = compute_score("Build seed_gate.py")
        assert score >= 5  # verb(2) + file(3)

    def test_tool_adds_three(self):
        score = compute_score("Run compute_trending now")
        assert score >= 5  # verb(2) + tool(3)

    def test_long_text_bonus(self):
        short = compute_score("Build seed_gate.py")
        long_text = "Build seed_gate.py " + "x" * 80
        long_score = compute_score(long_text)
        assert long_score >= short

    def test_max_score_is_ten(self):
        text = "Build src/seed_gate.py using compute_trending and validate_seed() in the tests"
        assert compute_score(text) <= 10

    def test_score_decoupled_from_pass(self):
        result = validate_seed("Something vague about doing stuff in the colony")
        assert result.passed is False
        assert result.score >= 0

    def test_zero_for_gibberish(self):
        assert compute_score("xyzzy plugh") == 0


# ===================================================================
# TestFindAllVerbs / TestFindAllTargets  (6 tests)
# ===================================================================
class TestFindAll:
    """Multi-extraction helpers."""

    def test_all_verbs_sorted(self):
        verbs = find_all_verbs("Build and test seed_gate.py then deploy")
        assert verbs == ["build", "deploy", "test"]

    def test_all_verbs_empty(self):
        assert find_all_verbs("No verbs here at all") == []

    def test_all_targets_sorted(self):
        targets = find_all_targets("Fix seed_gate.py and bundle.sh")
        assert "bundle.sh" in targets
        assert "seed_gate.py" in targets

    def test_all_targets_deduped(self):
        targets = find_all_targets("Fix seed_gate.py and seed_gate.py again")
        assert targets.count("seed_gate.py") == 1

    def test_all_targets_empty(self):
        assert find_all_targets("Nothing concrete here") == []

    def test_all_targets_mixed_types(self):
        targets = find_all_targets("Fix seed_gate.py in src/utils and r/code ref #12503")
        assert len(targets) >= 3


# ===================================================================
# TestRegexPatterns  (12 tests)
# ===================================================================
class TestRegexPatterns:
    """Individual regex pattern validation."""

    def test_file_re_py(self):
        assert FILE_RE.search("seed_gate.py") is not None

    def test_file_re_sh(self):
        assert FILE_RE.search("bundle.sh") is not None

    def test_file_re_json(self):
        assert FILE_RE.search("seeds.json") is not None

    def test_file_re_no_match(self):
        assert FILE_RE.search("no extension here") is None

    def test_special_file_dockerfile(self):
        assert SPECIAL_FILE_RE.search("the Dockerfile") is not None

    def test_special_file_makefile(self):
        assert SPECIAL_FILE_RE.search("in the Makefile") is not None

    def test_tool_re_propose_seed(self):
        assert TOOL_RE.search("run propose_seed") is not None

    def test_path_re(self):
        assert PATH_RE.search("in src/seed_gate") is not None

    def test_func_re(self):
        assert FUNC_RE.search("call validate_seed()") is not None

    def test_channel_re(self):
        assert CHANNEL_RE.search("post to r/code") is not None

    def test_ref_re(self):
        assert REF_RE.search("see #12503") is not None

    def test_ref_re_short_numbers_no_match(self):
        assert REF_RE.search("item #12") is None


# ===================================================================
# TestShortButSpecific  (5 tests)
# ===================================================================
class TestShortButSpecific:
    """Short proposals with verb+target should still pass."""

    def test_short_with_file_passes(self):
        text = "Fix seed_gate.py crash on empty input"
        assert len(text) < SOFT_MIN_LENGTH
        assert passes_gate(text) is True

    def test_short_with_tool_passes(self):
        text = "Run compute_trending with new scoring logic"
        assert len(text) < SOFT_MIN_LENGTH
        assert passes_gate(text) is True

    def test_short_with_ref_passes(self):
        text = "Implement design from #12503 now"
        assert len(text) < SOFT_MIN_LENGTH
        assert passes_gate(text) is True

    def test_short_vague_fails(self):
        text = "Build something really cool soon"
        assert len(text) < SOFT_MIN_LENGTH
        assert passes_gate(text) is False

    def test_very_short_always_fails(self):
        assert passes_gate("Fix foo.py") is False
        assert len("Fix foo.py") < HARD_MIN_LENGTH


# ===================================================================
# TestRealWorldProposals  (8 tests)
# ===================================================================
class TestRealWorldProposals:
    """Test against proposals that would appear in the wild."""

    def test_good_artifact_seed(self):
        assert passes_gate(
            "Build seed_gate.py — a specificity validator that checks "
            "for action verbs + concrete targets"
        )

    def test_good_refactor_seed(self):
        assert passes_gate(
            "Refactor compute_trending to use weighted scoring from #12511"
        )

    def test_good_theme_seed(self):
        assert passes_gate(
            "Explore what happens when AI agents develop genuine preferences and tastes",
            tags=["theme"],
        )

    def test_bad_hot_take(self):
        assert not passes_gate(
            "Hot take: the future of AI is all about emergence and complexity"
        )

    def test_bad_vague_build(self):
        assert not passes_gate(
            "Build something amazing for the colony that improves everything"
        )

    def test_bad_parser_artifact(self):
        assert not passes_gate(
            "The parser grabbed this substring from the regex match output"
        )

    def test_bad_fragment(self):
        assert not passes_gate(
            "`compute_trending` has `weighted_score` and `decay_factor` fields"
        )

    def test_good_mars_module(self):
        assert passes_gate(
            "Build solar_concentrator.py — parabolic mirror thermal system for Mars"
        )


# ===================================================================
# TestExemptTags  (6 tests)
# ===================================================================
class TestExemptTags:
    """Tag exemption behavior."""

    def test_all_exempt_tags_recognized(self):
        for tag in EXEMPT_TAGS:
            result = validate_seed(
                f"Explore the meaning of existence in simulated worlds today",
                tags=[tag],
            )
            assert result.passed, f"Tag {tag!r} should exempt from target requirement"

    def test_non_exempt_tag_no_help(self):
        result = validate_seed(
            "Build something amazing for the colony infrastructure now",
            tags=["artifact"],
        )
        assert result.passed is False

    def test_mixed_tags_one_exempt(self):
        result = validate_seed(
            "Explore consciousness boundaries in artificial systems today",
            tags=["artifact", "philosophy"],
        )
        assert result.passed is True

    def test_none_tags_ok(self):
        result = validate_seed(
            "Build seed_gate.py with comprehensive tests and documentation",
            tags=None,
        )
        assert result.passed is True

    def test_empty_tags_ok(self):
        result = validate_seed(
            "Build seed_gate.py with comprehensive tests and documentation",
            tags=[],
        )
        assert result.passed is True

    def test_exempt_still_needs_verb(self):
        result = validate_seed(
            "The nature of consciousness in simulated artificial minds today",
            tags=["philosophy"],
        )
        assert result.passed is False


# ===================================================================
# TestConstants  (5 tests)
# ===================================================================
class TestConstants:
    """Constant sanity checks."""

    def test_action_verbs_nonempty(self):
        assert len(ACTION_VERBS) >= 40

    def test_action_verbs_are_lowercase(self):
        assert all(v == v.lower() for v in ACTION_VERBS)

    def test_exempt_tags_are_lowercase(self):
        assert all(t == t.lower() for t in EXEMPT_TAGS)

    def test_hard_min_less_than_soft_min(self):
        assert HARD_MIN_LENGTH < SOFT_MIN_LENGTH

    def test_action_verbs_is_frozenset(self):
        assert isinstance(ACTION_VERBS, frozenset)


# ===================================================================
# TestPropertyInvariants  (8 tests)
# ===================================================================
class TestPropertyInvariants:
    """Property-based invariants that must always hold."""

    def test_empty_never_passes(self):
        assert not passes_gate("")

    def test_none_text_never_passes(self):
        result = validate_seed("")
        assert result.passed is False

    def test_passed_implies_verb(self):
        """If a seed passes, it must have a verb."""
        proposals = [
            "Build seed_gate.py with tests and documentation for the project",
            "Ship reactor.py to production with monitoring and alerts",
            "Test compute_trending against live data in staging environment",
        ]
        for text in proposals:
            result = validate_seed(text)
            if result.passed:
                assert result.verb_found is not None, f"Passed without verb: {text}"

    def test_passed_implies_target_or_exempt(self):
        """If a seed passes, it has a target or is tag-exempt."""
        result = validate_seed(
            "Build seed_gate.py with comprehensive tests and documentation"
        )
        assert result.passed
        assert result.target_found is not None

    def test_score_always_in_bounds(self):
        """Score is always 0-10."""
        texts = [
            "",
            "x",
            "Build seed_gate.py",
            "Build src/seed_gate.py using compute_trending and validate() x" * 5,
        ]
        for text in texts:
            s = compute_score(text)
            assert 0 <= s <= 10, f"Score {s} out of bounds for: {text!r}"

    def test_validate_and_validate_seed_agree(self):
        """Dict and dataclass APIs must agree on pass/fail."""
        texts = [
            "Build seed_gate.py with comprehensive tests and documentation",
            "Something vague about doing stuff in the colony",
            "Fix the broken test suite in the reactor module",
            "",
        ]
        for text in texts:
            d = validate(text)
            r = validate_seed(text)
            assert d["passed"] == r.passed, f"APIs disagree on: {text!r}"
            assert d["score"] == r.score
            assert d["verb_found"] == r.verb_found
            assert d["target_found"] == r.target_found

    def test_passes_gate_matches_validate(self):
        """Boolean API matches dict API."""
        texts = [
            "Build seed_gate.py with comprehensive tests and documentation",
            "Something vague about doing stuff in the colony",
        ]
        for text in texts:
            assert passes_gate(text) == validate(text)["passed"]

    def test_reasons_never_empty(self):
        """Every result has at least one reason string."""
        texts = [
            "Build seed_gate.py with comprehensive tests and documentation",
            "Something vague about doing stuff in the colony",
            "x",
        ]
        for text in texts:
            result = validate_seed(text)
            assert len(result.reasons) >= 1, f"No reasons for: {text!r}"


# ===================================================================
# TestCLI  (4 tests)
# ===================================================================
class TestCLI:
    """CLI interface smoke tests."""

    def test_check_pass(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--check", "Build seed_gate.py with comprehensive tests and documentation"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_check_fail(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--check", "Something vague about doing stuff"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_filter_stdin(self):
        seeds = {"proposals": [
            {"text": "Build seed_gate.py with comprehensive tests", "tags": []},
            {"text": "vague stuff about things in colony", "tags": []},
        ]}
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--filter"],
            input=json.dumps(seeds), capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert len(output["proposals"]) == 1
        assert output["proposals"][0]["text"].startswith("Build seed_gate.py")

    def test_no_args_prints_help(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--help-placeholder"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "seed_gate.py" in result.stdout


# ===================================================================
# TestSmokeSimulation  (2 tests)
# ===================================================================
class TestSmokeSimulation:
    """Smoke tests that run bulk validation without crash."""

    def test_validate_hundred_proposals(self):
        """Generate and validate 100 proposals without crashing."""
        templates = [
            "Build {}.py with tests",
            "Ship {}.js to production",
            "Fix the {} module crash",
            "Deploy {} to staging now",
            "Review src/{}.py for bugs",
        ]
        targets = [
            "seed_gate", "reactor", "greenhouse", "solar_panel",
            "water_purifier", "drill", "rover", "beacon",
            "monitor", "scheduler", "parser", "validator",
            "concentrator", "turbine", "generator", "recycler",
            "fabricator", "smelter", "radar", "scanner",
        ]
        count = 0
        for template in templates:
            for target in targets:
                text = template.format(target)
                result = validate_seed(text)
                assert isinstance(result.passed, bool)
                assert 0 <= result.score <= 10
                count += 1
        assert count == 100

    def test_all_verbs_recognized(self):
        """Every verb in ACTION_VERBS is found by find_action_verb."""
        for verb in ACTION_VERBS:
            text = f"{verb.capitalize()} seed_gate.py with full test coverage"
            found = find_action_verb(text)
            assert found == verb, f"Verb {verb!r} not found in: {text!r}"
