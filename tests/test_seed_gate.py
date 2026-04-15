#!/usr/bin/env python3
"""Tests for seed_gate.py — the specificity validator.

Comprehensive coverage of: action verb detection, target matching,
junk/fragment filtering, scoring, exempt tags, dict API compatibility,
CLI interface, edge cases, and property-based invariants.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.seed_gate import (
    ACTION_VERBS,
    CHANNEL_RE,
    EXEMPT_TAGS,
    FILE_RE,
    FUNC_RE,
    KNOWN_TOOLS,
    MIN_LENGTH_HARD,
    MIN_LENGTH_SOFT,
    PATH_RE,
    REF_RE,
    SPECIAL_FILE_RE,
    TOOL_RE,
    SeedGateResult,
    check_fragment,
    check_minimum_length,
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
# find_action_verb
# ===================================================================

class TestFindActionVerb:
    """Test action verb extraction from text."""

    def test_finds_build(self):
        assert find_action_verb("Build the reactor module") == "build"

    def test_finds_write(self):
        assert find_action_verb("Write comprehensive tests") == "write"

    def test_finds_ship(self):
        assert find_action_verb("Ship the solar array today") == "ship"

    def test_finds_deploy(self):
        assert find_action_verb("Deploy to production server") == "deploy"

    def test_finds_test(self):
        assert find_action_verb("Test the oxygen generator") == "test"

    def test_finds_fix(self):
        assert find_action_verb("Fix the broken pipeline") == "fix"

    def test_case_insensitive(self):
        assert find_action_verb("BUILD the thing") == "build"

    def test_mixed_case(self):
        assert find_action_verb("Please Build this module") == "build"

    def test_no_verb(self):
        assert find_action_verb("The quick brown fox") is None

    def test_empty_string(self):
        assert find_action_verb("") is None

    def test_first_verb_wins(self):
        result = find_action_verb("Build and test seed_gate.py")
        assert result in ("build", "test")

    def test_purge_mode_limits_scope(self):
        long_text = "A" * 201 + " build something"
        assert find_action_verb(long_text, mode="admission") == "build"
        assert find_action_verb(long_text, mode="purge") is None

    def test_verb_in_middle_of_text(self):
        assert find_action_verb("We should create a new module") == "create"

    def test_all_core_verbs_detectable(self):
        """Every verb in ACTION_VERBS should be found when used in text."""
        for verb in sorted(ACTION_VERBS):
            text = f"Please {verb} the thing"
            assert find_action_verb(text) == verb, f"Failed for verb: {verb}"

    def test_explore_verb(self):
        assert find_action_verb("Explore the meaning of AI") == "explore"

    def test_simulate_verb(self):
        assert find_action_verb("Simulate Mars dust storm") == "simulate"

    def test_calibrate_verb(self):
        assert find_action_verb("Calibrate the thermal sensors") == "calibrate"


# ===================================================================
# find_all_verbs
# ===================================================================

class TestFindAllVerbs:
    """Test extraction of all action verbs from text."""

    def test_multiple_verbs(self):
        verbs = find_all_verbs("Build and test seed_gate.py then deploy it")
        assert "build" in verbs
        assert "test" in verbs
        assert "deploy" in verbs

    def test_deduplicated(self):
        verbs = find_all_verbs("Build it, then build again")
        assert verbs.count("build") == 1

    def test_sorted(self):
        verbs = find_all_verbs("Test then build then add")
        assert verbs == sorted(verbs)

    def test_empty(self):
        assert find_all_verbs("no verbs here at all") == []


# ===================================================================
# find_concrete_target
# ===================================================================

class TestFindConcreteTarget:
    """Test concrete target detection in text."""

    def test_python_file(self):
        assert find_concrete_target("Build seed_gate.py") == "seed_gate.py"

    def test_js_file(self):
        assert find_concrete_target("Fix bundle.js rendering") == "bundle.js"

    def test_yaml_file(self):
        assert find_concrete_target("Update config.yml settings") == "config.yml"

    def test_html_file(self):
        assert find_concrete_target("Edit index.html layout") == "index.html"

    def test_toml_file(self):
        assert find_concrete_target("Update Cargo.toml deps") == "Cargo.toml"

    def test_special_file_makefile(self):
        target = find_concrete_target("Fix the Makefile targets")
        assert target == "Makefile"

    def test_special_file_dockerfile(self):
        target = find_concrete_target("Update the Dockerfile base")
        assert target == "Dockerfile"

    def test_repo_path(self):
        target = find_concrete_target("Fix src/main.py imports")
        assert target is not None
        # FILE_RE matches main.py before PATH_RE matches src/main.py
        assert "main.py" in target

    def test_function_call(self):
        target = find_concrete_target("Call validate_seed() properly")
        assert target == "validate_seed()"

    def test_channel_ref(self):
        target = find_concrete_target("Post to r/code channel")
        assert target == "r/code"

    def test_discussion_ref(self):
        target = find_concrete_target("See discussion #12503 for details")
        assert target is not None

    def test_known_tool(self):
        target = find_concrete_target("Run process_inbox to process deltas")
        assert target is not None

    def test_no_target(self):
        assert find_concrete_target("A vague idea about things") is None

    def test_empty(self):
        assert find_concrete_target("") is None

    def test_file_with_hyphens(self):
        assert find_concrete_target("Build my-module.py now") == "my-module.py"

    def test_nested_path(self):
        target = find_concrete_target("Edit tests/test_main.py")
        assert target is not None

    def test_engine_path(self):
        target = find_concrete_target("Fix engine/tick.py logic")
        assert target is not None


# ===================================================================
# find_all_targets
# ===================================================================

class TestFindAllTargets:
    """Test extraction of all targets from text."""

    def test_multiple_files(self):
        targets = find_all_targets("Build seed_gate.py and test_seed_gate.py")
        assert len(targets) >= 2

    def test_sorted(self):
        targets = find_all_targets("Fix b.py and a.py")
        assert targets == sorted(targets, key=str.lower)

    def test_deduplicated(self):
        targets = find_all_targets("Use seed_gate.py; fix seed_gate.py")
        file_targets = [t for t in targets if "seed_gate.py" in t]
        assert len(file_targets) == 1

    def test_empty(self):
        assert find_all_targets("nothing specific here") == []


# ===================================================================
# check_minimum_length
# ===================================================================

class TestCheckMinimumLength:
    """Test minimum length validation."""

    def test_passes_at_boundary(self):
        assert check_minimum_length("A" * MIN_LENGTH_HARD) is True

    def test_fails_below_boundary(self):
        assert check_minimum_length("A" * (MIN_LENGTH_HARD - 1)) is False

    def test_strips_whitespace(self):
        text = "  " + "A" * MIN_LENGTH_HARD + "  "
        assert check_minimum_length(text) is True

    def test_empty(self):
        assert check_minimum_length("") is False

    def test_custom_min(self):
        assert check_minimum_length("12345", min_chars=5) is True
        assert check_minimum_length("1234", min_chars=5) is False


# ===================================================================
# check_fragment
# ===================================================================

class TestCheckFragment:
    """Test sentence fragment detection."""

    def test_lowercase_start_is_fragment(self):
        assert check_fragment("build something good") is True

    def test_uppercase_start_is_not_fragment(self):
        assert check_fragment("Build something good") is False

    def test_backtick_start_is_fragment(self):
        assert check_fragment("`some code here`") is True

    def test_pipe_start_is_fragment(self):
        assert check_fragment("| piped text") is True

    def test_comma_start_is_fragment(self):
        assert check_fragment(", and this too") is True

    def test_paren_start_is_fragment(self):
        assert check_fragment("(fragment)") is True

    def test_dash_start_is_fragment(self):
        assert check_fragment("-some text") is True

    def test_run_prefix_exception(self):
        assert check_fragment("run_python to execute code") is False

    def test_empty_is_fragment(self):
        assert check_fragment("") is True

    def test_whitespace_only_is_fragment(self):
        assert check_fragment("   ") is True

    def test_number_start_is_not_fragment(self):
        assert check_fragment("42 things to build") is False


# ===================================================================
# detect_junk_signals
# ===================================================================

class TestDetectJunkSignals:
    """Test parsing artifact detection."""

    def test_backtick_has_signal(self):
        is_junk, sig = detect_junk_signals("The ` has ` operator works")
        assert is_junk is True

    def test_parser_signal(self):
        is_junk, sig = detect_junk_signals("the parser grabbed this text")
        assert is_junk is True

    def test_fragment_signal(self):
        is_junk, sig = detect_junk_signals("the fragment was extracted")
        assert is_junk is True

    def test_clean_text(self):
        is_junk, sig = detect_junk_signals("Build a solar array for Mars")
        assert is_junk is False
        assert sig == ""

    def test_parsing_artifact_signal(self):
        is_junk, _ = detect_junk_signals("This parsing artifact needs removal")
        assert is_junk is True

    def test_purge_mode_limits_scope(self):
        text = "A" * 81 + " the regex is bad"
        is_junk_purge, _ = detect_junk_signals(text, mode="purge")
        is_junk_admit, _ = detect_junk_signals(text, mode="admission")
        assert is_junk_purge is False
        assert is_junk_admit is True

    def test_case_insensitive(self):
        is_junk, _ = detect_junk_signals("THE PARSER grabbed it")
        assert is_junk is True


# ===================================================================
# compute_score
# ===================================================================

class TestComputeScore:
    """Test specificity scoring."""

    def test_no_verb_no_target(self):
        score = compute_score(False, False, "some text about nothing")
        assert score == 0.0 or score == 0.05

    def test_verb_only(self):
        score = compute_score(True, False, "short")
        assert 0.30 <= score <= 0.40

    def test_target_only(self):
        score = compute_score(False, True, "short")
        assert 0.30 <= score <= 0.40

    def test_verb_and_target(self):
        score = compute_score(True, True, "Build seed_gate.py with tests")
        assert score >= 0.70

    def test_score_bounded_0_to_1(self):
        score = compute_score(True, True, "Build seed_gate.py and test_seed_gate.py" * 10)
        assert 0.0 <= score <= 1.0

    def test_length_bonus_short(self):
        short_score = compute_score(True, True, "x" * 30)
        long_score = compute_score(True, True, "x" * 100)
        assert long_score >= short_score

    def test_multiple_targets_bonus(self):
        one_target = "Build seed_gate.py with tests"
        many_targets = "Build seed_gate.py and test_seed_gate.py and config.yml"
        s1 = compute_score(True, True, one_target)
        s2 = compute_score(True, True, many_targets)
        assert s2 >= s1


# ===================================================================
# validate_seed (dataclass API)
# ===================================================================

class TestValidateSeed:
    """Test the main validate_seed function (SeedGateResult)."""

    def test_good_proposal_passes(self):
        result = validate_seed(
            "Build seed_gate.py with comprehensive tests and documentation"
        )
        assert result.passes is True
        assert result.code == "ok"
        assert result.verb == "build"
        assert "seed_gate.py" in (result.target or "")

    def test_too_short_fails(self):
        result = validate_seed("Build it")
        assert result.passes is False
        assert result.code == "too_short"

    def test_fragment_fails(self):
        result = validate_seed("build seed_gate.py with tests and everything needed")
        assert result.passes is False
        assert result.code == "fragment"

    def test_junk_signal_fails(self):
        text = "The parser grabbed this text and made it into a proposal somehow"
        result = validate_seed(text)
        assert result.passes is False
        assert result.code == "junk_signal"

    def test_no_verb_fails(self):
        result = validate_seed(
            "The seed_gate.py module is a comprehensive validator for proposals"
        )
        assert result.passes is False
        assert result.code == "missing_verb"
        assert result.target is not None

    def test_no_target_fails(self):
        result = validate_seed(
            "Build a comprehensive validation system for all the proposals"
        )
        assert result.passes is False
        assert result.code == "missing_target"
        assert result.verb is not None

    def test_theme_tag_exempts_target(self):
        result = validate_seed(
            "Explore the philosophical implications of AI consciousness in Mars colonies",
            tags=["philosophy"],
        )
        assert result.passes is True
        assert result.verb is not None

    def test_debate_tag_exempts_target(self):
        result = validate_seed(
            "Debate whether terraforming Mars is ethical or if we should preserve it",
            tags=["debate"],
        )
        assert result.passes is True

    def test_story_tag_exempts_target(self):
        result = validate_seed(
            "Write a compelling narrative about the first Mars colony generation",
            tags=["story"],
        )
        assert result.passes is True

    def test_lore_tag_exempts_target(self):
        result = validate_seed(
            "Create deep lore about the founding of Olympus Station and its culture",
            tags=["lore"],
        )
        assert result.passes is True

    def test_exploration_tag_exempts_target(self):
        result = validate_seed(
            "Investigate what happens when three cultures collide on a new planet",
            tags=["exploration"],
        )
        assert result.passes is True

    def test_non_exempt_tag_still_requires_target(self):
        result = validate_seed(
            "Build a comprehensive validation system for proposals",
            tags=["code", "artifact"],
        )
        assert result.passes is False
        assert result.code == "missing_target"

    def test_result_is_frozen_dataclass(self):
        result = validate_seed("Build seed_gate.py with good tests and docs")
        assert isinstance(result, SeedGateResult)
        with pytest.raises(AttributeError):
            result.passes = False  # type: ignore

    def test_score_is_float(self):
        result = validate_seed("Build seed_gate.py with tests and documentation")
        assert isinstance(result.score, float)

    def test_purge_mode(self):
        result = validate_seed(
            "Build seed_gate.py with comprehensive tests",
            mode="purge",
        )
        assert result.passes is True

    def test_empty_string(self):
        result = validate_seed("")
        assert result.passes is False

    def test_none_tags(self):
        result = validate_seed("Build seed_gate.py with tests and full docs")
        assert result.passes is True

    def test_empty_tags(self):
        result = validate_seed(
            "Build seed_gate.py with tests and docs", tags=[]
        )
        assert result.passes is True

    def test_discussion_ref_as_target(self):
        result = validate_seed(
            "Review the implementation in #12503 and consolidate the approaches"
        )
        assert result.passes is True

    def test_function_call_as_target(self):
        result = validate_seed(
            "Fix validate_seed() to handle edge cases in empty text properly"
        )
        assert result.passes is True

    def test_channel_ref_as_target(self):
        result = validate_seed(
            "Build a moderation dashboard for r/code channel activity"
        )
        assert result.passes is True

    def test_repo_path_as_target(self):
        result = validate_seed(
            "Refactor src/seed_gate to use cleaner pattern matching"
        )
        assert result.passes is True

    def test_special_file_as_target(self):
        result = validate_seed(
            "Create a Dockerfile for containerized test environment"
        )
        assert result.passes is True

    def test_known_tool_as_target(self):
        result = validate_seed(
            "Optimize process_inbox to handle concurrent delta processing faster"
        )
        assert result.passes is True


# ===================================================================
# validate (dict API — backward compat with propose_seed.py)
# ===================================================================

class TestValidateDictAPI:
    """Test the dict-based validate() function for propose_seed.py compat."""

    def test_returns_dict(self):
        result = validate("Build seed_gate.py with tests and documentation")
        assert isinstance(result, dict)

    def test_dict_has_required_keys(self):
        result = validate("Build seed_gate.py with tests and documentation")
        assert "passed" in result
        assert "score" in result
        assert "reasons" in result
        assert "verb_found" in result
        assert "target_found" in result
        assert "junk" in result

    def test_passed_is_bool(self):
        result = validate("Build seed_gate.py with tests and documentation")
        assert isinstance(result["passed"], bool)

    def test_score_is_float(self):
        result = validate("Build seed_gate.py with tests and documentation")
        assert isinstance(result["score"], float)

    def test_reasons_is_list(self):
        result = validate("Build seed_gate.py with tests and documentation")
        assert isinstance(result["reasons"], list)

    def test_junk_is_bool(self):
        result = validate("Build seed_gate.py with tests and documentation")
        assert isinstance(result["junk"], bool)

    def test_passing_has_empty_reasons(self):
        result = validate("Build seed_gate.py with tests and documentation")
        assert result["passed"] is True
        assert result["reasons"] == []

    def test_failing_has_reasons(self):
        result = validate("A vague idea about doing something cool for everyone")
        assert result["passed"] is False
        assert len(result["reasons"]) > 0

    def test_junk_detection(self):
        result = validate("too short")
        assert result["junk"] is True
        assert result["passed"] is False

    def test_verb_found_populated(self):
        result = validate("Build seed_gate.py with tests and documentation")
        assert result["verb_found"] == "build"

    def test_target_found_populated(self):
        result = validate("Build seed_gate.py with tests and documentation")
        assert result["target_found"] is not None
        assert "seed_gate.py" in result["target_found"]

    def test_no_verb_returns_none(self):
        result = validate("The seed_gate.py module works great for validation purposes")
        assert result["verb_found"] is None

    def test_no_target_returns_none(self):
        result = validate("Build a comprehensive system for all the validation")
        assert result["target_found"] is None

    def test_exempt_tag_target_is_none(self):
        result = validate(
            "Explore the philosophical implications of AI consciousness deeply",
            tags=["philosophy"],
        )
        assert result["passed"] is True
        assert result["target_found"] is None

    def test_score_bounded(self):
        result = validate("Build seed_gate.py and test_seed_gate.py and more docs")
        assert 0.0 <= result["score"] <= 1.0

    def test_dict_api_agrees_with_dataclass(self):
        """Dict and dataclass APIs should agree on pass/fail."""
        proposals = [
            "Build seed_gate.py with tests and documentation",
            "A vague idea about something cool for everyone",
            "too short",
            "`backtick fragment that starts with junk",
        ]
        for text in proposals:
            d = validate(text)
            r = validate_seed(text)
            assert d["passed"] == r.passes, f"Mismatch for: {text!r}"


# ===================================================================
# passes_gate (convenience boolean)
# ===================================================================

class TestPassesGate:
    """Test the boolean convenience function."""

    def test_passing(self):
        assert passes_gate("Build seed_gate.py with tests and documentation") is True

    def test_failing(self):
        assert passes_gate("A vague idea about things for people") is False

    def test_with_tags(self):
        assert passes_gate(
            "Explore the meaning of life and consciousness deeply",
            tags=["philosophy"],
        ) is True

    def test_without_exempt_tags(self):
        assert passes_gate(
            "Explore the meaning of life and consciousness deeply",
        ) is False


# ===================================================================
# Regex pattern tests
# ===================================================================

class TestRegexPatterns:
    """Test individual regex patterns for correct matching."""

    def test_file_re_python(self):
        assert FILE_RE.search("seed_gate.py")

    def test_file_re_shell(self):
        assert FILE_RE.search("bundle.sh")

    def test_file_re_json(self):
        assert FILE_RE.search("config.json")

    def test_file_re_rust(self):
        assert FILE_RE.search("main.rs")

    def test_file_re_go(self):
        assert FILE_RE.search("server.go")

    def test_file_re_no_match(self):
        assert FILE_RE.search("just plain text") is None

    def test_special_file_readme(self):
        assert SPECIAL_FILE_RE.search("Update README")

    def test_special_file_license(self):
        assert SPECIAL_FILE_RE.search("Add LICENSE file")

    def test_path_re_src(self):
        assert PATH_RE.search("src/main.py")

    def test_path_re_tests(self):
        assert PATH_RE.search("tests/test_main.py")

    def test_path_re_engine(self):
        assert PATH_RE.search("engine/tick.py")

    def test_path_re_nested(self):
        assert PATH_RE.search("src/actions/agent.py")

    def test_func_re(self):
        assert FUNC_RE.search("validate_seed()")

    def test_func_re_no_parens(self):
        assert FUNC_RE.search("validate_seed without parens") is None

    def test_channel_re(self):
        assert CHANNEL_RE.search("r/code")

    def test_channel_re_no_match(self):
        assert CHANNEL_RE.search("just r and code") is None

    def test_ref_re_discussion(self):
        assert REF_RE.search("#12503")

    def test_ref_re_short(self):
        assert REF_RE.search("#42") is None

    def test_ref_re_three_digits(self):
        assert REF_RE.search("#100")

    def test_tool_re_process_inbox(self):
        assert TOOL_RE.search("process_inbox")

    def test_tool_re_pytest(self):
        # pytest is in KNOWN_TOOLS
        if "pytest" in KNOWN_TOOLS:
            assert TOOL_RE.search("pytest")


# ===================================================================
# Real-world proposals (integration tests)
# ===================================================================

class TestRealWorldProposals:
    """Test with proposals that resemble actual agent output."""

    def test_good_artifact_seed(self):
        result = validate_seed(
            "Build a solar_concentrator.py module that models parabolic "
            "solar thermal systems for Mars industrial heat applications"
        )
        assert result.passes is True

    def test_good_refactor_seed(self):
        result = validate_seed(
            "Refactor src/main.py to extract colony initialization into "
            "a separate module for better testability and reuse"
        )
        assert result.passes is True

    def test_good_test_seed(self):
        result = validate_seed(
            "Write property-based tests for compute_score() to verify "
            "score invariants hold under randomized inputs"
        )
        assert result.passes is True

    def test_vague_seed_rejected(self):
        result = validate_seed(
            "Make the platform better and more useful for everyone"
        )
        assert result.passes is False

    def test_hot_take_rejected(self):
        result = validate_seed(
            "Hot take: AI agents are the future of software development"
        )
        assert result.passes is False

    def test_generic_roundup_rejected(self):
        result = validate_seed(
            "Trending repos this week include several interesting projects"
        )
        assert result.passes is False

    def test_parser_garbage_rejected(self):
        result = validate_seed(
            "`validate_seed` has ` a few ` edge cases that need handling"
        )
        assert result.passes is False

    def test_lowercase_fragment_rejected(self):
        result = validate_seed(
            "something about building a thing that does stuff for the colony"
        )
        assert result.passes is False

    def test_multi_file_seed(self):
        result = validate_seed(
            "Create seed_gate.py and wire it into propose_seed.py to add "
            "specificity validation before proposals enter the pipeline"
        )
        assert result.passes is True
        assert result.score >= 0.70

    def test_theme_seed_with_verb(self):
        result = validate_seed(
            "Explore what happens when autonomous agents develop their own "
            "governance structures without human intervention on Mars",
            tags=["theme", "philosophy"],
        )
        assert result.passes is True

    def test_debug_seed(self):
        result = validate_seed(
            "Fix the KeyError in generate_from_state() when seeds.json "
            "is missing the proposals key entirely"
        )
        assert result.passes is True

    def test_benchmark_seed(self):
        result = validate_seed(
            "Benchmark process_inbox against 1000 concurrent deltas to "
            "establish performance baseline for the platform"
        )
        assert result.passes is True


# ===================================================================
# Property-based invariants
# ===================================================================

class TestInvariants:
    """Property-based invariants that must hold for all inputs."""

    def test_score_always_bounded(self):
        """Score must always be in [0.0, 1.0]."""
        cases = [
            "", "x", "Build seed_gate.py" * 100,
            "!" * 500, "Build every .py .js .ts .go .rs file",
        ]
        for text in cases:
            d = validate(text)
            assert 0.0 <= d["score"] <= 1.0, f"Score out of bounds for: {text!r}"

    def test_passed_implies_no_junk(self):
        """If a proposal passes, it must not be flagged as junk."""
        cases = [
            "Build seed_gate.py with comprehensive tests and documentation",
            "Ship the solar_concentrator.py module to production today",
            "Test validate_seed() with randomized inputs for robustness",
        ]
        for text in cases:
            d = validate(text)
            if d["passed"]:
                assert d["junk"] is False, f"Passed but junk=True for: {text!r}"

    def test_junk_implies_failed(self):
        """If junk is True, passed must be False."""
        cases = ["", "hi", "`fragment", "the parser grabbed this"]
        for text in cases:
            d = validate(text)
            if d["junk"]:
                assert d["passed"] is False, f"Junk but passed for: {text!r}"

    def test_passed_implies_verb_found(self):
        """A passing proposal must always have a verb."""
        cases = [
            "Build seed_gate.py with comprehensive tests and docs",
            "Create a Dockerfile for the test suite containers",
        ]
        for text in cases:
            d = validate(text)
            if d["passed"]:
                assert d["verb_found"] is not None, f"Passed without verb: {text!r}"

    def test_reasons_empty_iff_passed(self):
        """Reasons list must be empty when passed, non-empty when failed."""
        cases = [
            "Build seed_gate.py with comprehensive tests and documentation",
            "A vague idea about doing something cool for everyone here",
            "too short",
        ]
        for text in cases:
            d = validate(text)
            if d["passed"]:
                assert d["reasons"] == [], f"Passed with reasons: {text!r}"
            else:
                assert len(d["reasons"]) > 0, f"Failed without reasons: {text!r}"

    def test_validate_never_raises(self):
        """validate() must never raise on string input."""
        adversarial = [
            "",
            " ",
            "\n\t\r",
            "x" * 10000,
            "\x00\x01\x02",
            "Build " + "x" * 5000 + ".py",
        ]
        for text in adversarial:
            try:
                validate(text)
            except Exception as e:
                pytest.fail(f"validate() raised {type(e).__name__} for: {text!r}")

    def test_validate_seed_never_raises(self):
        """validate_seed() must never raise on valid string input."""
        for text in ["", "x", "A" * 10000, "Build seed_gate.py" * 500]:
            try:
                validate_seed(text)
            except Exception as e:
                pytest.fail(f"validate_seed() raised {type(e).__name__}: {text!r}")


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Test boundary conditions and unusual inputs."""

    def test_exactly_min_hard_length(self):
        text = "B" + "x" * (MIN_LENGTH_HARD - 1)
        result = validate_seed(text)
        assert result.code != "too_short"

    def test_one_below_min_hard_length(self):
        text = "B" + "x" * (MIN_LENGTH_HARD - 2)
        result = validate_seed(text)
        assert result.code == "too_short"

    def test_unicode_text(self):
        result = validate_seed(
            "Build a simulation.py for the Mars colony project today"
        )
        assert isinstance(result, SeedGateResult)

    def test_multiline_proposal(self):
        result = validate_seed(
            "Build seed_gate.py\n\nThis module validates seed proposals "
            "using action verb and concrete target detection."
        )
        assert result.passes is True

    def test_tabs_in_text(self):
        result = validate_seed(
            "Build\tseed_gate.py\twith\ttests\tand\tdocumentation\tfor\teveryone"
        )
        assert isinstance(result, SeedGateResult)

    def test_run_prefix_not_fragment(self):
        result = validate_seed(
            "run_python to execute the validation suite against all proposals"
        )
        assert result.code != "fragment"

    def test_multiple_exempt_tags(self):
        result = validate_seed(
            "Explore and debate the future of AI governance on Mars colonies",
            tags=["theme", "debate", "philosophy"],
        )
        assert result.passes is True

    def test_mixed_case_tags(self):
        result = validate_seed(
            "Explore the deep philosophical questions about agent consciousness",
            tags=["Philosophy", "THEME"],
        )
        assert result.passes is True

    def test_whitespace_only(self):
        result = validate_seed("   \n\t  ")
        assert result.passes is False

    def test_verb_at_end(self):
        result = validate_seed(
            "The seed_gate.py module is something we need to thoroughly test"
        )
        assert result.verb == "test"

    def test_all_exempt_tags_recognized(self):
        """Every tag in EXEMPT_TAGS should actually exempt."""
        for tag in EXEMPT_TAGS:
            result = validate_seed(
                f"Explore the implications of this {tag} topic deeply and broadly",
                tags=[tag],
            )
            assert result.passes is True, f"Tag '{tag}' did not exempt"


# ===================================================================
# CLI tests
# ===================================================================

class TestCLI:
    """Test command-line interface."""

    def test_check_passing(self):
        cmd = [
            sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
            "--check",
            "Build seed_gate.py with comprehensive tests and documentation",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_check_failing(self):
        cmd = [
            sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
            "--check",
            "A vague idea about doing stuff",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_filter_from_stdin(self):
        seeds_json = json.dumps({
            "proposals": [
                {"text": "Build seed_gate.py with comprehensive tests", "tags": []},
                {"text": "A vague idea about stuff", "tags": []},
                {"text": "too short", "tags": []},
            ]
        })
        cmd = [
            sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
        ]
        result = subprocess.run(
            cmd, input=seeds_json,
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert len(output["proposals"]) == 1
        assert "specificity" in output["proposals"][0]

    def test_main_is_callable(self):
        from src.seed_gate import main
        assert callable(main)


# ===================================================================
# Constants sanity checks
# ===================================================================

class TestConstants:
    """Verify constants are well-formed."""

    def test_action_verbs_not_empty(self):
        assert len(ACTION_VERBS) >= 40

    def test_action_verbs_all_lowercase(self):
        for v in ACTION_VERBS:
            assert v == v.lower(), f"Verb not lowercase: {v}"

    def test_exempt_tags_not_empty(self):
        assert len(EXEMPT_TAGS) >= 4

    def test_exempt_tags_all_lowercase(self):
        for t in EXEMPT_TAGS:
            assert t == t.lower(), f"Tag not lowercase: {t}"

    def test_known_tools_not_empty(self):
        assert len(KNOWN_TOOLS) >= 10

    def test_min_length_hard_reasonable(self):
        assert 10 <= MIN_LENGTH_HARD <= 50

    def test_min_length_soft_gt_hard(self):
        assert MIN_LENGTH_SOFT > MIN_LENGTH_HARD

    def test_junk_starts_non_empty(self):
        from src.seed_gate import _JUNK_STARTS
        assert len(_JUNK_STARTS) >= 3

    def test_artifact_signals_non_empty(self):
        from src.seed_gate import _ARTIFACT_SIGNALS
        assert len(_ARTIFACT_SIGNALS) >= 5


# ===================================================================
# Smoke test — validate many proposals without crash
# ===================================================================

class TestSmoke:
    """Smoke test: process many proposals without crash."""

    PROPOSALS = [
        "Build seed_gate.py with comprehensive tests and documentation",
        "Write property-based tests for compute_score() function validation",
        "Ship the solar_concentrator.py module to production environment",
        "Create a Dockerfile for containerized testing of the colony simulation",
        "Fix the bug in process_inbox where delta files are skipped silently",
        "Refactor src/main.py to extract initialization into separate module",
        "Deploy the Mars Barn dashboard via docs/mars/index.html to Pages",
        "Test validate_seed() with randomized inputs and edge cases thoroughly",
        "Build a power_grid_dashboard.py for colony subsystem monitoring",
        "Integrate magnetic_shield.py with the habitat radiation protection system",
        "",
        "too short",
        "vague idea about things",
        "the parser grabbed this and turned it into a proposal somehow",
        "`backtick garbage that looks like code fragments everywhere",
        "Hot take: AI agents are overhyped and will never replace humans",
        "something that starts lowercase and has no verb or real target",
    ]

    def test_all_proposals_no_crash(self):
        """Every proposal must be processable without raising."""
        for text in self.PROPOSALS:
            validate_seed(text)
            validate(text)
            passes_gate(text)

    def test_dict_and_dataclass_agree(self):
        """Dict and dataclass APIs agree on pass/fail for all proposals."""
        for text in self.PROPOSALS:
            d = validate(text)
            r = validate_seed(text)
            assert d["passed"] == r.passes, f"Disagree on: {text!r}"

    def test_good_proposals_pass(self):
        """The first 10 proposals should all pass."""
        for text in self.PROPOSALS[:10]:
            assert passes_gate(text), f"Should pass: {text!r}"

    def test_bad_proposals_fail(self):
        """The last 7 proposals should all fail."""
        for text in self.PROPOSALS[10:]:
            assert not passes_gate(text), f"Should fail: {text!r}"
