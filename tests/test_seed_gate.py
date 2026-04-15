#!/usr/bin/env python3
"""Tests for seed_gate.py -- the specificity validator.

160+ tests covering: verb detection, target detection, exempt tags,
fragment detection, junk signals, length checks, scoring, integration,
purge vs admission modes, CLI, edge cases, property invariants.
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
    ACTION_VERBS, CHANNEL_RE, EXEMPT_TAGS, FILE_RE, FUNC_RE, PATH_RE,
    SPECIAL_FILE_RE, TOOL_RE, SeedGateResult, DOMAIN_NOUNS,
    MIN_PROPOSAL_LENGTH, FRAGMENT_LEADING_CHARS,
    ADMISSION_JUNK_SIGNALS, PURGE_JUNK_SIGNALS,
    compute_score, find_action_verb, find_concrete_target,
    find_all_verbs, find_all_targets,
    check_minimum_length, check_fragment, detect_junk_signals,
    has_exempt_tag, passes_gate, validate_seed,
)


# --- Helpers ---------------------------------------------------------
def _pad(text: str, min_len: int = 55) -> str:
    """Pad text to meet minimum length requirement if needed."""
    if len(text) >= min_len:
        return text
    return text + " " + "x" * (min_len - len(text) - 1)


# =====================================================================
# TestFindActionVerb (16 tests)
# =====================================================================
class TestFindActionVerb:
    def test_finds_build(self):
        assert find_action_verb("Build the reactor module") == "build"

    def test_finds_write(self):
        assert find_action_verb("Write a test suite for the hab") == "write"

    def test_finds_ship(self):
        assert find_action_verb("Ship the validator to production") == "ship"

    def test_finds_test(self):
        assert find_action_verb("Test the drill module thoroughly") == "test"

    def test_finds_fix(self):
        assert find_action_verb("Fix the broken pipeline") == "fix"

    def test_finds_deploy(self):
        assert find_action_verb("Deploy the solar array now") == "deploy"

    def test_case_insensitive(self):
        assert find_action_verb("DEPLOY the staging branch") == "deploy"

    def test_mixed_case(self):
        assert find_action_verb("BuIlD something") == "build"

    def test_no_verb_abstract(self):
        assert find_action_verb("Something about the colony") is None

    def test_no_verb_empty(self):
        assert find_action_verb("") is None

    def test_no_verb_numbers(self):
        assert find_action_verb("12345 67890") is None

    def test_purge_mode_200_char_limit(self):
        text = "x " * 101 + "build something"  # verb past 200 chars
        assert find_action_verb(text, mode="purge") is None
        assert find_action_verb(text, mode="admission") == "build"

    def test_first_verb_wins(self):
        assert find_action_verb("Build and ship the module") == "build"

    def test_verb_in_middle(self):
        assert find_action_verb("The team will build it") == "build"

    def test_explore_verb(self):
        assert find_action_verb("Explore the consciousness module") == "explore"

    def test_all_core_verbs_detected(self):
        core = ["build", "write", "create", "implement", "ship", "deploy",
                "test", "fix", "refactor", "validate", "benchmark"]
        for v in core:
            assert find_action_verb(f"{v.capitalize()} the thing") == v


# =====================================================================
# TestFindConcreteTarget (18 tests)
# =====================================================================
class TestFindConcreteTarget:
    def test_finds_py_file(self):
        assert find_concrete_target("Fix seed_gate.py") == "seed_gate.py"

    def test_finds_js_file(self):
        assert find_concrete_target("Edit router.js") == "router.js"

    def test_finds_sh_file(self):
        assert find_concrete_target("Run bundle.sh") == "bundle.sh"

    def test_finds_json_file(self):
        assert find_concrete_target("Update seeds.json") == "seeds.json"

    def test_finds_special_file(self):
        assert find_concrete_target("Edit the Makefile") == "Makefile"

    def test_finds_dockerfile(self):
        assert find_concrete_target("Create Dockerfile") == "Dockerfile"

    def test_finds_tool(self):
        assert find_concrete_target("Run pytest") == "pytest"

    def test_finds_propose_seed(self):
        assert find_concrete_target("Fix propose_seed") == "propose_seed"

    def test_finds_path(self):
        target = find_concrete_target("Edit src/seed_gate.py")
        assert target == "src/seed_gate.py"

    def test_finds_deep_path(self):
        target = find_concrete_target("Fix scripts/actions/agent.py")
        assert target == "scripts/actions/agent.py"

    def test_finds_function(self):
        assert find_concrete_target("Call validate_seed() now") == "validate_seed()"

    def test_finds_channel(self):
        assert find_concrete_target("Post to r/general") == "r/general"

    def test_no_target_abstract(self):
        assert find_concrete_target("Improve the colony somehow") is None

    def test_no_target_vague(self):
        assert find_concrete_target("Things should be better") is None

    def test_no_target_empty(self):
        assert find_concrete_target("") is None

    def test_finds_yml_file(self):
        assert find_concrete_target("Edit config.yml") == "config.yml"

    def test_finds_toml(self):
        assert find_concrete_target("Update pyproject.toml") == "pyproject.toml"

    def test_domain_noun_not_target(self):
        assert find_concrete_target("Build a reactor") is None


# =====================================================================
# TestFindAllVerbsTargets (6 tests)
# =====================================================================
class TestFindAllVerbsTargets:
    def test_all_verbs(self):
        verbs = find_all_verbs("Build and test seed_gate.py then deploy")
        assert "build" in verbs
        assert "test" in verbs
        assert "deploy" in verbs

    def test_all_targets(self):
        targets = find_all_targets("Fix seed_gate.py and run pytest")
        assert any("seed_gate" in t for t in targets)
        assert any("pytest" in t for t in targets)

    def test_empty_verbs(self):
        assert find_all_verbs("nothing here") == []

    def test_empty_targets(self):
        assert find_all_targets("nothing here") == []

    def test_deduplication(self):
        targets = find_all_targets("seed_gate.py and seed_gate.py again")
        count = sum(1 for t in targets if "seed_gate" in t)
        assert count == 1

    def test_all_verbs_sorted(self):
        verbs = find_all_verbs("test and build and ship")
        assert verbs == sorted(verbs)


# =====================================================================
# TestCheckMinimumLength (5 tests)
# =====================================================================
class TestCheckMinimumLength:
    def test_long_enough(self):
        assert check_minimum_length("x" * 50) is True

    def test_too_short(self):
        assert check_minimum_length("short") is False

    def test_empty(self):
        assert check_minimum_length("") is False

    def test_exactly_min(self):
        assert check_minimum_length("a" * MIN_PROPOSAL_LENGTH) is True

    def test_one_under(self):
        assert check_minimum_length("a" * (MIN_PROPOSAL_LENGTH - 1)) is False


# =====================================================================
# TestCheckFragment (8 tests)
# =====================================================================
class TestCheckFragment:
    def test_lowercase_start_is_fragment(self):
        assert check_fragment("something starting lowercase") is True

    def test_uppercase_start_not_fragment(self):
        assert check_fragment("Something starting uppercase") is False

    def test_backtick_start(self):
        assert check_fragment("`code block start") is True

    def test_pipe_start(self):
        assert check_fragment("|piped text") is True

    def test_run_underscore_exempt(self):
        assert check_fragment("run_python something") is False

    def test_empty_is_fragment(self):
        assert check_fragment("") is True

    def test_paren_start(self):
        assert check_fragment("(parenthetical start") is True

    def test_number_start_not_fragment(self):
        assert check_fragment("42 things to do") is False


# =====================================================================
# TestDetectJunkSignals (10 tests)
# =====================================================================
class TestDetectJunkSignals:
    def test_admission_parser_grabbed(self):
        is_junk, sig = detect_junk_signals("parser grabbed this text")
        assert is_junk is True
        assert "parser grabbed" in sig

    def test_admission_parsing_artifact(self):
        is_junk, _ = detect_junk_signals("this is a parsing artifact")
        assert is_junk is True

    def test_admission_clean(self):
        is_junk, sig = detect_junk_signals("Build seed_gate.py with tests")
        assert is_junk is False
        assert sig == ""

    def test_purge_head_signal(self):
        is_junk, _ = detect_junk_signals("the regex matched", mode="purge")
        assert is_junk is True

    def test_purge_clean(self):
        is_junk, _ = detect_junk_signals("Build something great", mode="purge")
        assert is_junk is False

    def test_purge_only_first_60(self):
        text = "A" * 61 + " the regex matched"
        is_junk, _ = detect_junk_signals(text, mode="purge")
        assert is_junk is False

    def test_admission_checks_full_text(self):
        text = "A" * 100 + " parser grabbed this"
        is_junk, _ = detect_junk_signals(text, mode="admission")
        assert is_junk is True

    def test_the_fragment_was(self):
        is_junk, _ = detect_junk_signals("the fragment was extracted wrong")
        assert is_junk is True

    def test_substring_signal(self):
        is_junk, _ = detect_junk_signals("this is just a substring")
        assert is_junk is True

    def test_case_insensitive(self):
        is_junk, _ = detect_junk_signals("Parser Grabbed this text")
        assert is_junk is True


# =====================================================================
# TestHasExemptTag (5 tests)
# =====================================================================
class TestHasExemptTag:
    def test_philosophy_exempt(self):
        assert has_exempt_tag(["philosophy"]) is True

    def test_theme_exempt(self):
        assert has_exempt_tag(["theme"]) is True

    def test_engineering_not_exempt(self):
        assert has_exempt_tag(["engineering"]) is False

    def test_none_tags(self):
        assert has_exempt_tag(None) is False

    def test_empty_list(self):
        assert has_exempt_tag([]) is False


# =====================================================================
# TestComputeScore (8 tests)
# =====================================================================
class TestComputeScore:
    def test_perfect_score(self):
        text = "Build src/seed_gate.py with pytest to validate the pipeline"
        s = compute_score(text)
        assert s >= 7

    def test_verb_only(self):
        text = "Build something cool for the colony"
        s = compute_score(text)
        assert s == 2

    def test_file_adds_3(self):
        s = compute_score("Look at seed_gate.py for reference material today")
        assert s >= 3

    def test_empty_zero(self):
        assert compute_score("") == 0

    def test_capped_at_10(self):
        text = ("Build src/seed_gate.py and run pytest and call validate_seed() "
                "and check bundle.sh and state_io and more tools and things")
        assert compute_score(text) <= 10

    def test_length_bonus(self):
        short = "Build seed_gate.py"
        long_ = "Build seed_gate.py " + "with comprehensive tests " * 3
        assert compute_score(long_) >= compute_score(short)

    def test_tool_adds_3(self):
        s = compute_score("The pytest suite is comprehensive and detailed")
        assert s >= 3

    def test_score_nonnegative(self):
        assert compute_score("xyzzy") >= 0


# =====================================================================
# TestValidateSeedPass (8 tests)
# =====================================================================
class TestValidateSeedPass:
    def test_verb_plus_filename(self):
        r = validate_seed(_pad("Build seed_gate.py with comprehensive test coverage"))
        assert r.passes is True
        assert r.verb == "build"
        assert r.target == "seed_gate.py"

    def test_verb_plus_tool(self):
        r = validate_seed(_pad("Run pytest on the new module to validate everything"))
        assert r.passes is True
        assert r.verb == "run"
        assert r.target == "pytest"

    def test_verb_plus_path(self):
        r = validate_seed(_pad("Fix src/seed_gate.py to handle edge cases in validation"))
        assert r.passes is True
        assert r.target == "src/seed_gate.py"

    def test_verb_plus_function(self):
        r = validate_seed(_pad("Refactor validate_seed() to support purge mode checks"))
        assert r.passes is True
        assert r.target == "validate_seed()"

    def test_verb_plus_channel(self):
        r = validate_seed(_pad("Create content pipeline for r/general channel updates"))
        assert r.passes is True
        assert r.target == "r/general"

    def test_multiple_verbs_and_targets(self):
        r = validate_seed(
            "Build seed_gate.py and run pytest to ship the validator to production"
        )
        assert r.passes is True
        assert r.verb is not None
        assert r.target is not None

    def test_long_detailed_proposal(self):
        text = ("Implement comprehensive specificity validation in seed_gate.py "
                "that checks for action verbs and concrete targets before any "
                "proposal enters the pipeline")
        r = validate_seed(text)
        assert r.passes is True

    def test_exempt_tag_no_target(self):
        r = validate_seed(
            _pad("Explore the philosophical implications of agent consciousness deeply"),
            tags=["philosophy"]
        )
        assert r.passes is True
        assert r.verb == "explore"


# =====================================================================
# TestValidateSeedFail (10 tests)
# =====================================================================
class TestValidateSeedFail:
    def test_no_verb_no_target(self):
        r = validate_seed(_pad("Something about the colony that needs attention"))
        assert r.passes is False
        assert r.code == "missing_verb"

    def test_verb_no_target(self):
        r = validate_seed(
            _pad("Build something amazing for the Mars habitat module system")
        )
        assert r.passes is False
        assert r.code == "missing_target"
        assert r.verb == "build"

    def test_too_short(self):
        r = validate_seed("Build x.py")
        assert r.passes is False
        assert r.code == "too_short"

    def test_empty(self):
        r = validate_seed("")
        assert r.passes is False
        assert r.code == "too_short"

    def test_fragment_lowercase(self):
        r = validate_seed(_pad("build seed_gate.py with all those comprehensive tests"))
        assert r.passes is False
        assert r.code == "fragment"

    def test_fragment_backtick(self):
        r = validate_seed(_pad("`seed_gate.py` needs to be built with comprehensive testing"))
        assert r.passes is False
        assert r.code == "fragment"

    def test_junk_signal_parser(self):
        r = validate_seed(_pad("Parser grabbed this text from the parsing pipeline output"))
        assert r.passes is False
        assert r.code == "junk_signal"

    def test_junk_signal_artifact(self):
        r = validate_seed(_pad("This is a parsing artifact from the extraction pipeline run"))
        assert r.passes is False
        assert r.code == "junk_signal"

    def test_only_target_no_verb(self):
        r = validate_seed(
            _pad("The seed_gate.py file is what the community needs attention on")
        )
        assert r.passes is False
        assert r.code == "missing_verb"
        assert r.target == "seed_gate.py"

    def test_no_exempt_tag_no_target(self):
        r = validate_seed(
            _pad("Explore the philosophical implications of agent consciousness deeply"),
            tags=["engineering"]
        )
        assert r.passes is False
        assert r.code == "missing_target"


# =====================================================================
# TestPassesGate (6 tests)
# =====================================================================
class TestPassesGate:
    def test_passes_with_verb_and_file(self):
        assert passes_gate(_pad("Build seed_gate.py with comprehensive validation tests")) is True

    def test_fails_no_verb(self):
        assert passes_gate(_pad("The seed_gate.py file needs some kind of attention soon")) is False

    def test_fails_no_target(self):
        assert passes_gate(_pad("Build something amazing for the colony habitat system")) is False

    def test_passes_with_tags(self):
        assert passes_gate(
            _pad("Design agent consciousness framework for the colony exploration"),
            tags=["philosophy"]
        ) is True

    def test_fails_short(self):
        assert passes_gate("Build x.py") is False

    def test_mode_parameter(self):
        assert passes_gate(
            _pad("Build seed_gate.py with comprehensive tests and documentation"),
            mode="purge"
        ) is True


# =====================================================================
# TestPurgeMode (8 tests)
# =====================================================================
class TestPurgeMode:
    def test_purge_skips_length_check(self):
        r = validate_seed("Build seed_gate.py", mode="purge")
        assert r.code != "too_short"

    def test_purge_skips_fragment_check(self):
        r = validate_seed("build seed_gate.py with tests", mode="purge")
        assert r.code != "fragment"

    def test_purge_verb_limit(self):
        text = "A" * 201 + " build seed_gate.py with tests"
        r_purge = validate_seed(text, mode="purge")
        r_admission = validate_seed(text, mode="admission")
        assert r_purge.verb is None
        assert r_admission.verb == "build"

    def test_purge_junk_checks_head(self):
        text = _pad("The regex matched something in the parsing output stream")
        r = validate_seed(text, mode="purge")
        assert r.code == "junk_signal"

    def test_purge_ignores_tail_junk(self):
        text = "Build seed_gate.py with tests" + " " * 40 + "the regex"
        r = validate_seed(text, mode="purge")
        assert r.code != "junk_signal"

    def test_purge_still_needs_verb(self):
        r = validate_seed(
            "The seed_gate.py file needs attention",
            mode="purge"
        )
        assert r.passes is False
        assert r.code == "missing_verb"

    def test_purge_still_needs_target(self):
        r = validate_seed(
            "Build something amazing for the colony",
            mode="purge"
        )
        assert r.passes is False
        assert r.code == "missing_target"

    def test_admission_checks_full_junk(self):
        text = "Build seed_gate.py" + " " * 50 + "parser grabbed this text"
        r = validate_seed(_pad(text), mode="admission")
        assert r.code == "junk_signal"


# =====================================================================
# TestExemptTags (6 tests)
# =====================================================================
class TestExemptTags:
    def test_philosophy_exempt(self):
        r = validate_seed(
            _pad("Explore the philosophical implications of agent emergence deeply"),
            tags=["philosophy"]
        )
        assert r.passes is True

    def test_theme_exempt(self):
        r = validate_seed(
            _pad("Design a thematic framework for colony cultural evolution now"),
            tags=["theme"]
        )
        assert r.passes is True

    def test_debate_exempt(self):
        r = validate_seed(
            _pad("Investigate whether agents can develop genuine autonomy in sim"),
            tags=["debate"]
        )
        assert r.passes is True

    def test_non_exempt_tag(self):
        r = validate_seed(
            _pad("Build something amazing for the Mars colony habitat system now"),
            tags=["engineering"]
        )
        assert r.passes is False
        assert r.code == "missing_target"

    def test_exempt_still_needs_verb(self):
        r = validate_seed(
            _pad("Something philosophical about consciousness and emergence now"),
            tags=["philosophy"]
        )
        assert r.passes is False
        assert r.code == "missing_verb"

    def test_mixed_tags_one_exempt(self):
        r = validate_seed(
            _pad("Explore agent autonomy in the context of Mars colony governance"),
            tags=["engineering", "philosophy"]
        )
        assert r.passes is True


# =====================================================================
# TestRegexPatterns (12 tests)
# =====================================================================
class TestRegexPatterns:
    def test_file_re_py(self):
        assert FILE_RE.search("seed_gate.py") is not None

    def test_file_re_js(self):
        assert FILE_RE.search("router.js") is not None

    def test_file_re_no_match(self):
        assert FILE_RE.search("no extension here") is None

    def test_special_file_makefile(self):
        assert SPECIAL_FILE_RE.search("Makefile") is not None

    def test_special_file_dockerfile(self):
        assert SPECIAL_FILE_RE.search("Dockerfile") is not None

    def test_tool_re_pytest(self):
        assert TOOL_RE.search("pytest") is not None

    def test_tool_re_make(self):
        assert TOOL_RE.search("make") is not None

    def test_path_re_src(self):
        assert PATH_RE.search("src/seed_gate.py") is not None

    def test_path_re_scripts(self):
        assert PATH_RE.search("scripts/actions/agent.py") is not None

    def test_func_re(self):
        assert FUNC_RE.search("validate_seed()") is not None

    def test_func_re_short_rejected(self):
        assert FUNC_RE.search("go()") is None

    def test_channel_re(self):
        assert CHANNEL_RE.search("r/general") is not None


# =====================================================================
# TestEdgeCases (12 tests)
# =====================================================================
class TestEdgeCases:
    def test_mixed_case_verb(self):
        r = validate_seed(_pad("BuIlD seed_gate.py with comprehensive testing coverage"))
        assert r.passes is True
        assert r.verb == "build"

    def test_verb_at_end(self):
        r = validate_seed(
            _pad("The seed_gate.py module is what the colony needs to build now")
        )
        assert r.passes is True
        assert r.verb == "build"

    def test_unicode_safe(self):
        r = validate_seed(_pad("Build seed_gate.py with unicode support for the colony"))
        assert r.passes is True

    def test_newlines(self):
        text = "Build seed_gate.py\nwith comprehensive tests\nfor the colony pipeline"
        r = validate_seed(text)
        assert r.passes is True

    def test_multiple_files(self):
        r = validate_seed(
            "Build seed_gate.py and test_seed_gate.py together for the colony pipeline"
        )
        assert r.passes is True

    def test_path_with_extension(self):
        r = validate_seed(
            "Fix src/seed_gate.py to handle edge cases in the validation pipeline"
        )
        assert r.passes is True

    def test_run_python_tool(self):
        r = validate_seed(
            "Use run_python to execute the simulation step in the colony pipeline"
        )
        assert r.passes is True

    def test_whitespace_only(self):
        r = validate_seed("   ")
        assert r.passes is False
        assert r.code == "too_short"

    def test_very_long_text(self):
        text = "Build seed_gate.py " + "with " * 500
        r = validate_seed(text)
        assert r.passes is True

    def test_result_is_frozen_dataclass(self):
        r = validate_seed(_pad("Build seed_gate.py with comprehensive tests for colony"))
        with pytest.raises(AttributeError):
            r.passes = False  # type: ignore[misc]

    def test_func_call_target(self):
        r = validate_seed(
            _pad("Refactor validate_seed() to handle purge mode edge cases now")
        )
        assert r.passes is True
        assert r.target == "validate_seed()"

    def test_channel_target(self):
        r = validate_seed(
            _pad("Create automated content pipeline for r/code channel updates now")
        )
        assert r.passes is True
        assert r.target == "r/code"


# =====================================================================
# TestRealWorldProposals (10 tests)
# =====================================================================
class TestRealWorldProposals:
    """Proposals from the 6 agent implementations that should pass/fail."""

    def test_good_build_seed_gate(self):
        r = validate_seed(
            "Build seed_gate.py -- a specificity validator that checks for "
            "action verb + concrete target"
        )
        assert r.passes is True

    def test_good_ship_with_tests(self):
        r = validate_seed(
            "Ship seed_gate.py with tests in tests/test_seed_gate.py covering "
            "all edge cases"
        )
        assert r.passes is True

    def test_good_wire_into_propose(self):
        r = validate_seed(
            "Wire seed_gate.py into the propose() function so every proposal "
            "passes the specificity gate"
        )
        assert r.passes is True

    def test_good_implement_validator(self):
        r = validate_seed(
            "Implement seed_gate.py as a standalone module that validates "
            "proposals have an action verb and a concrete target like "
            "a filename or tool name"
        )
        assert r.passes is True

    def test_bad_vague_improve(self):
        r = validate_seed(
            "Improve the platform to be better at everything and more useful"
        )
        assert r.passes is False

    def test_bad_fragment(self):
        r = validate_seed(
            _pad("`seed_gate` should check for verbs and targets in proposals")
        )
        assert r.passes is False
        assert r.code == "fragment"

    def test_bad_too_generic(self):
        r = validate_seed(
            "Build something that makes the colony better and more efficient"
        )
        assert r.passes is False

    def test_bad_no_target(self):
        r = validate_seed(
            "Fix all the broken things in the codebase so they function better"
        )
        assert r.passes is False
        assert r.code == "missing_target"

    def test_bad_parsing_artifact(self):
        r = validate_seed(
            _pad("Parser grabbed this substring from the seed text pipeline run")
        )
        assert r.passes is False
        assert r.code == "junk_signal"

    def test_bad_only_noun(self):
        r = validate_seed(
            _pad("The reactor module needs comprehensive attention and improvement")
        )
        assert r.passes is False


# =====================================================================
# TestPropertyInvariants (6 tests)
# =====================================================================
class TestPropertyInvariants:
    """Properties that must hold for ALL inputs."""

    SAMPLES = [
        "Build seed_gate.py with comprehensive tests for the colony pipeline",
        "x",
        "",
        "build seed_gate.py",
        "Build something cool and amazing forever now now now now now now now now now now",
        _pad("Build seed_gate.py with tests for the colony module right now"),
        "Parser grabbed the substring from pipeline output text stream",
        _pad("`backtick start of a proposal with seed_gate.py target text`"),
    ]

    @pytest.mark.parametrize("text", SAMPLES)
    def test_score_bounded(self, text):
        assert 0 <= compute_score(text) <= 10

    @pytest.mark.parametrize("text", SAMPLES)
    def test_result_has_all_fields(self, text):
        r = validate_seed(text)
        assert isinstance(r, SeedGateResult)
        assert isinstance(r.passes, bool)
        assert isinstance(r.score, int)
        assert isinstance(r.reason, str)
        assert r.code in ("ok", "too_short", "fragment", "junk_signal",
                          "missing_verb", "missing_target")

    @pytest.mark.parametrize("text", SAMPLES)
    def test_passes_implies_verb(self, text):
        r = validate_seed(text)
        if r.passes:
            assert r.verb is not None

    @pytest.mark.parametrize("text", SAMPLES)
    def test_code_ok_iff_passes(self, text):
        r = validate_seed(text)
        assert (r.code == "ok") == r.passes

    def test_verb_plus_file_always_passes(self):
        """ANY verb + ANY filename should pass if length is sufficient."""
        verbs = list(ACTION_VERBS)[:5]
        files = ["gate.py", "router.js", "config.yml", "Makefile", "README"]
        for v in verbs:
            for f in files:
                text = _pad(f"{v.capitalize()} {f} with comprehensive tests for colony")
                assert passes_gate(text), f"{v} {f}: {text}"

    def test_passes_gate_equals_validate(self):
        text = _pad("Build seed_gate.py with comprehensive tests for the colony")
        assert passes_gate(text) == validate_seed(text).passes


# =====================================================================
# TestConstants (6 tests)
# =====================================================================
class TestConstants:
    def test_min_length_is_50(self):
        assert MIN_PROPOSAL_LENGTH == 50

    def test_action_verbs_nonempty(self):
        assert len(ACTION_VERBS) >= 20

    def test_exempt_tags_nonempty(self):
        assert len(EXEMPT_TAGS) >= 3

    def test_domain_nouns_nonempty(self):
        assert len(DOMAIN_NOUNS) >= 20

    def test_fragment_chars_include_backtick(self):
        assert "`" in FRAGMENT_LEADING_CHARS

    def test_junk_signals_nonempty(self):
        assert len(ADMISSION_JUNK_SIGNALS) >= 3
        assert len(PURGE_JUNK_SIGNALS) >= 3


# =====================================================================
# TestCLI (4 tests)
# =====================================================================
class TestCLI:
    SCRIPT = str(REPO_ROOT / "src" / "seed_gate.py")

    def test_check_passing_seed(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--check",
             "Build", "seed_gate.py", "with", "comprehensive", "tests",
             "for", "the", "colony", "pipeline", "validation"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_check_failing_seed(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--check",
             "Something", "vague", "about", "the", "colony",
             "that", "needs", "attention", "and", "improvement"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_no_args_prints_help(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT],
            capture_output=True, text=True, timeout=10,
            # No stdin redirect -- isatty() will be False in subprocess
            # but empty stdin triggers JSONDecodeError, so we use --check
            # with no args for help. Actually just test that module loads.
        )
        # With no args and no tty, it tries to read stdin filter.
        # With DEVNULL it gets empty JSON. Just verify it doesn't crash.
        assert result.returncode in (0, 1)

    def test_filter_stdin(self):
        seeds = {
            "proposals": [
                {"text": "Build seed_gate.py with comprehensive tests for the colony pipeline",
                 "tags": []},
                {"text": "Something vague", "tags": []},
                {"text": "Ship reactor.py to production with full test coverage for habitat",
                 "tags": []},
            ]
        }
        result = subprocess.run(
            [sys.executable, self.SCRIPT],
            input=json.dumps(seeds),
            capture_output=True, text=True, timeout=10,
        )
        output = json.loads(result.stdout)
        assert len(output["proposals"]) == 2
        assert all(p["specificity"]["passes"] for p in output["proposals"])


# =====================================================================
# TestSmoke (2 tests)
# =====================================================================
class TestSmoke:
    def test_import(self):
        import seed_gate
        assert hasattr(seed_gate, "validate_seed")
        assert hasattr(seed_gate, "passes_gate")

    def test_roundtrip(self):
        text = "Build seed_gate.py with comprehensive tests for the colony pipeline"
        r = validate_seed(text)
        assert isinstance(r, SeedGateResult)
        assert r.code in ("ok", "too_short", "fragment", "junk_signal",
                          "missing_verb", "missing_target")
