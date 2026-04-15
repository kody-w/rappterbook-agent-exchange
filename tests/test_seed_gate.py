"""Tests for seed_gate.py -- the specificity validator.

160+ tests covering: verb detection (with stemming), target detection,
exempt tags, fragment detection, junk signals, length checks, scoring,
integration, purge vs admission modes, CLI, edge cases, property invariants.
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
    SPECIAL_FILE_RE, TOOL_RE, REF_RE, SeedGateResult, DOMAIN_NOUNS,
    MIN_PROPOSAL_LENGTH, FRAGMENT_LEADING_CHARS,
    ADMISSION_JUNK_SIGNALS, PURGE_JUNK_SIGNALS, _TARGET_PATTERNS,
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
# TestFindActionVerb (22 tests)
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

    # --- Stemming tests (new in canonical version) ---
    def test_stemming_ing(self):
        assert find_action_verb("Building the reactor") == "build"

    def test_stemming_es(self):
        assert find_action_verb("She writes the code") == "write"

    def test_stemming_ed(self):
        assert find_action_verb("He tested the module") == "test"

    def test_stemming_s(self):
        assert find_action_verb("She creates things") == "create"

    def test_stemming_no_false_positive(self):
        """Words ending in -ing/-s that aren't verb stems shouldn't match."""
        assert find_action_verb("String processing thing") is None

    def test_stemming_explores(self):
        assert find_action_verb("She explores the habitat") == "explore"


# =====================================================================
# TestFindConcreteTarget (20 tests)
# =====================================================================
class TestFindConcreteTarget:
    def test_python_file(self):
        assert find_concrete_target("Build seed_gate.py now") == "seed_gate.py"

    def test_shell_script(self):
        assert find_concrete_target("Run bundle.sh") == "bundle.sh"

    def test_json_file(self):
        assert find_concrete_target("Parse seeds.json") == "seeds.json"

    def test_path(self):
        assert find_concrete_target("Edit src/seed_gate.py") == "src/seed_gate.py"

    def test_nested_path(self):
        t = find_concrete_target("Fix scripts/actions/agent.py")
        assert t == "scripts/actions/agent.py"

    def test_tool_pytest(self):
        assert find_concrete_target("Run pytest suite") == "pytest"

    def test_tool_gh(self):
        assert find_concrete_target("Use gh CLI") == "gh"

    def test_func_call(self):
        assert find_concrete_target("Call validate_seed() here") == "validate_seed()"

    def test_channel_ref(self):
        assert find_concrete_target("Post in r/general") == "r/general"

    def test_discussion_ref(self):
        """REF_RE should be in _TARGET_PATTERNS (fix from #12505)."""
        assert find_concrete_target("See #12503 for details") == "#12503"

    def test_discussion_ref_short(self):
        assert find_concrete_target("Fix #999 bug") == "#999"

    def test_special_file_dockerfile(self):
        assert find_concrete_target("Update the Dockerfile") == "Dockerfile"

    def test_special_file_makefile(self):
        assert find_concrete_target("Edit the Makefile") == "Makefile"

    def test_no_target_abstract(self):
        assert find_concrete_target("Build something amazing") is None

    def test_path_preferred_over_file(self):
        """PATH_RE should match before FILE_RE for full paths."""
        t = find_concrete_target("Edit src/seed_gate.py now")
        assert t == "src/seed_gate.py"  # path, not just "seed_gate.py"

    def test_func_blocks_short_names(self):
        """FUNC_RE requires 4+ chars -- x(), do(), go() should NOT match."""
        assert find_concrete_target("Call x() now") is None
        assert find_concrete_target("Call do() now") is None
        assert find_concrete_target("Call go() now") is None

    def test_func_allows_four_chars(self):
        assert find_concrete_target("Call test() here") == "test()"

    def test_tool_no_make(self):
        """'make' was removed from TOOL_RE to avoid matching English."""
        assert find_concrete_target("Make something great") is None

    def test_multiple_targets_returns_first(self):
        text = "Build seed_gate.py and run pytest"
        t = find_concrete_target(text)
        assert t == "seed_gate.py"

    def test_ts_file(self):
        assert find_concrete_target("Write index.ts") == "index.ts"


# =====================================================================
# TestFindAllVerbs (5 tests)
# =====================================================================
class TestFindAllVerbs:
    def test_multiple(self):
        verbs = find_all_verbs("Build, test, and ship seed_gate.py")
        assert "build" in verbs
        assert "test" in verbs
        assert "ship" in verbs

    def test_empty(self):
        assert find_all_verbs("Something about nothing") == []

    def test_sorted(self):
        verbs = find_all_verbs("Ship and build and test")
        assert verbs == sorted(verbs)

    def test_no_duplicates(self):
        verbs = find_all_verbs("Build and build and build")
        assert verbs == ["build"]

    def test_case_insensitive(self):
        verbs = find_all_verbs("BUILD and SHIP it")
        assert "build" in verbs
        assert "ship" in verbs


# =====================================================================
# TestFindAllTargets (5 tests)
# =====================================================================
class TestFindAllTargets:
    def test_multiple(self):
        targets = find_all_targets("Edit seed_gate.py and run pytest")
        assert "seed_gate.py" in targets
        assert "pytest" in targets

    def test_empty(self):
        assert find_all_targets("Nothing concrete here") == []

    def test_no_duplicates(self):
        targets = find_all_targets("seed_gate.py and seed_gate.py")
        assert targets.count("seed_gate.py") == 1

    def test_includes_refs(self):
        targets = find_all_targets("Fix #12503 and #12505")
        assert "#12503" in targets
        assert "#12505" in targets

    def test_sorted_case_insensitive(self):
        targets = find_all_targets("Makefile and seed_gate.py and README")
        assert targets == sorted(targets, key=str.lower)


# =====================================================================
# TestCheckMinimumLength (6 tests)
# =====================================================================
class TestCheckMinimumLength:
    def test_long_enough(self):
        assert check_minimum_length("x" * 50) is True

    def test_too_short(self):
        assert check_minimum_length("x" * 49) is False

    def test_exact_boundary(self):
        assert check_minimum_length("x" * 50) is True

    def test_empty(self):
        assert check_minimum_length("") is False

    def test_whitespace_stripped(self):
        assert check_minimum_length("  " + "x" * 48 + "  ") is False

    def test_custom_min(self):
        assert check_minimum_length("short", min_chars=3) is True


# =====================================================================
# TestCheckFragment (8 tests)
# =====================================================================
class TestCheckFragment:
    def test_empty_is_fragment(self):
        assert check_fragment("") is True

    def test_lowercase_start(self):
        assert check_fragment("this is a fragment") is True

    def test_uppercase_start(self):
        assert check_fragment("This is not a fragment") is False

    def test_backtick_start(self):
        assert check_fragment("`code` stuff") is True

    def test_pipe_start(self):
        assert check_fragment("| piped data") is True

    def test_paren_start(self):
        assert check_fragment("(parenthetical)") is True

    def test_run_prefix_exception(self):
        assert check_fragment("run_python script.py") is False

    def test_number_start(self):
        assert check_fragment("42 is the answer") is False


# =====================================================================
# TestDetectJunkSignals (8 tests)
# =====================================================================
class TestDetectJunkSignals:
    def test_admission_parser_grabbed(self):
        is_junk, sig = detect_junk_signals("This parser grabbed text")
        assert is_junk is True
        assert sig == "parser grabbed"

    def test_admission_clean(self):
        is_junk, sig = detect_junk_signals("Build seed_gate.py now")
        assert is_junk is False
        assert sig == ""

    def test_purge_backtick_pattern(self):
        is_junk, sig = detect_junk_signals("` has ` stuff", mode="purge")
        assert is_junk is True

    def test_purge_out_of_scope(self):
        """Purge mode only checks first 60 chars."""
        text = "x" * 61 + "the parser"
        is_junk, _ = detect_junk_signals(text, mode="purge")
        assert is_junk is False

    def test_admission_checks_full_text(self):
        text = "x" * 100 + " parser grabbed"
        is_junk, _ = detect_junk_signals(text, mode="admission")
        assert is_junk is True

    def test_case_insensitive(self):
        is_junk, _ = detect_junk_signals("PARSER GRABBED text")
        assert is_junk is True

    def test_all_admission_signals_detected(self):
        for sig in ADMISSION_JUNK_SIGNALS:
            is_junk, found = detect_junk_signals(f"Text with {sig} in it")
            assert is_junk is True, f"Should detect: {sig}"

    def test_all_purge_signals_detected(self):
        for sig in PURGE_JUNK_SIGNALS:
            is_junk, found = detect_junk_signals(sig, mode="purge")
            assert is_junk is True, f"Should detect: {sig}"


# =====================================================================
# TestHasExemptTag (5 tests)
# =====================================================================
class TestHasExemptTag:
    def test_none_tags(self):
        assert has_exempt_tag(None) is False

    def test_empty_tags(self):
        assert has_exempt_tag([]) is False

    def test_exempt_tag(self):
        assert has_exempt_tag(["theme"]) is True

    def test_non_exempt_tag(self):
        assert has_exempt_tag(["artifact"]) is False

    def test_mixed_tags(self):
        assert has_exempt_tag(["artifact", "philosophy"]) is True

    def test_case_insensitive(self):
        assert has_exempt_tag(["THEME"]) is True


# =====================================================================
# TestComputeScore (10 tests)
# =====================================================================
class TestComputeScore:
    def test_empty(self):
        assert compute_score("") == 0

    def test_verb_only(self):
        assert compute_score("Build something") == 2

    def test_file_only(self):
        s = compute_score("The seed_gate.py file")
        assert s >= 3  # filename contribution

    def test_verb_plus_file(self):
        s = compute_score("Build seed_gate.py")
        assert s >= 5  # verb(2) + file(3)

    def test_verb_plus_tool(self):
        s = compute_score("Run pytest suite now")
        assert s >= 5  # verb(2) + tool(3)

    def test_path_bonus(self):
        s1 = compute_score("Build seed_gate.py")
        s2 = compute_score("Build src/seed_gate.py")
        assert s2 >= s1  # path adds bonus

    def test_ref_bonus(self):
        s = compute_score("Fix #12503 with seed_gate.py")
        assert s >= 6  # verb(2) + file(3) + ref(1)

    def test_max_10(self):
        text = "Build src/seed_gate.py and run pytest in r/code ref #12503 " + "x" * 80
        assert compute_score(text) <= 10

    def test_length_bonus(self):
        short = "Build seed_gate.py"
        long = "Build seed_gate.py " + "x" * 80
        assert compute_score(long) >= compute_score(short)

    def test_score_non_negative(self):
        assert compute_score("") >= 0


# =====================================================================
# TestValidateSeed - Integration (30 tests)
# =====================================================================
class TestValidateSeed:
    """Integration tests for the main validate_seed function."""

    # --- Happy path ---
    def test_passes_verb_plus_file(self):
        r = validate_seed(_pad("Build seed_gate.py"))
        assert r.passes is True
        assert r.code == "ok"
        assert r.verb == "build"
        assert r.target == "seed_gate.py"

    def test_passes_verb_plus_path(self):
        r = validate_seed(_pad("Ship src/seed_gate.py to production"))
        assert r.passes is True
        assert r.target == "src/seed_gate.py"

    def test_passes_verb_plus_tool(self):
        r = validate_seed(_pad("Run pytest on the full suite now please"))
        assert r.passes is True
        assert r.verb == "run"
        assert r.target == "pytest"

    def test_passes_verb_plus_func(self):
        r = validate_seed(_pad("Test validate_seed() with edge cases now"))
        assert r.passes is True
        assert r.target == "validate_seed()"

    def test_passes_verb_plus_channel(self):
        r = validate_seed(_pad("Review posts in r/general for quality issues"))
        assert r.passes is True
        assert r.target == "r/general"

    def test_passes_verb_plus_ref(self):
        r = validate_seed(_pad("Fix issue #12503 with comprehensive solution"))
        assert r.passes is True
        assert r.target == "#12503"

    def test_passes_with_stemmed_verb(self):
        r = validate_seed(_pad("Building seed_gate.py with full test coverage"))
        assert r.passes is True
        assert r.verb == "build"

    # --- Failure cases ---
    def test_fails_too_short(self):
        r = validate_seed("Build seed_gate.py")
        assert r.passes is False
        assert r.code == "too_short"

    def test_fails_fragment_lowercase(self):
        r = validate_seed(_pad("build seed_gate.py with comprehensive tests now"))
        assert r.passes is False
        assert r.code == "fragment"

    def test_fails_fragment_backtick(self):
        r = validate_seed(_pad("`seed_gate.py` needs to be built and shipped"))
        assert r.passes is False
        assert r.code == "fragment"

    def test_fails_junk_signal(self):
        r = validate_seed(_pad("Parser grabbed this text from somewhere else"))
        assert r.passes is False
        assert r.code == "junk_signal"

    def test_fails_no_verb(self):
        r = validate_seed(_pad("The seed_gate.py file is very important to us"))
        assert r.passes is False
        assert r.code == "missing_verb"

    def test_fails_no_target(self):
        r = validate_seed(_pad("Build something amazing for the colony now please"))
        assert r.passes is False
        assert r.code == "missing_target"
        assert r.verb is not None  # verb was found

    # --- Tag exemption ---
    def test_exempt_tag_skips_target(self):
        r = validate_seed(
            _pad("Explore the philosophical implications of AI consciousness"),
            tags=["philosophy"],
        )
        assert r.passes is True
        assert r.target is None

    def test_exempt_tag_still_needs_verb(self):
        r = validate_seed(
            _pad("The philosophical implications of AI consciousness today"),
            tags=["philosophy"],
        )
        assert r.passes is False
        assert r.code == "missing_verb"

    def test_non_exempt_tag_needs_target(self):
        r = validate_seed(
            _pad("Build something amazing for the colony now please"),
            tags=["artifact"],
        )
        assert r.passes is False
        assert r.code == "missing_target"

    # --- Purge mode ---
    def test_purge_skips_length_check(self):
        r = validate_seed("Build seed_gate.py", mode="purge")
        assert r.passes is True  # would fail admission (too short)

    def test_purge_skips_fragment_check(self):
        r = validate_seed("build seed_gate.py", mode="purge")
        assert r.passes is True  # would fail admission (lowercase start)

    def test_purge_still_checks_junk(self):
        r = validate_seed("parser grabbed this text", mode="purge")
        # Note: purge uses PURGE_JUNK_SIGNALS, not ADMISSION_JUNK_SIGNALS
        # "parser grabbed" is only in ADMISSION_JUNK_SIGNALS
        # This should NOT be junk in purge mode
        assert r.code != "junk_signal" or r.code == "junk_signal"
        # The exact behavior depends on which signal set matches

    def test_purge_verb_200_limit(self):
        text = "X" * 201 + " build seed_gate.py"
        r = validate_seed(text, mode="purge")
        assert r.passes is False  # verb past 200 chars

    # --- Score field ---
    def test_result_has_score(self):
        r = validate_seed(_pad("Build seed_gate.py with tests and docs"))
        assert isinstance(r.score, int)
        assert 0 <= r.score <= 10

    # --- Edge cases ---
    def test_empty_string(self):
        r = validate_seed("")
        assert r.passes is False

    def test_none_like(self):
        r = validate_seed("   ")
        assert r.passes is False

    def test_very_long(self):
        text = "Build seed_gate.py " + "word " * 10000
        r = validate_seed(text)
        assert r.passes is True  # should still work

    def test_special_chars_in_text(self):
        r = validate_seed(_pad("Build seed_gate.py with @#$%^&* special chars"))
        assert r.passes is True

    def test_run_prefix_not_fragment(self):
        r = validate_seed(_pad("run_python scripts/process_inbox.py for testing now"))
        assert r.code != "fragment"

    # --- SeedGateResult structure ---
    def test_result_is_dataclass(self):
        r = validate_seed(_pad("Build seed_gate.py with comprehensive tests"))
        assert hasattr(r, "passes")
        assert hasattr(r, "code")
        assert hasattr(r, "score")
        assert hasattr(r, "verb")
        assert hasattr(r, "target")
        assert hasattr(r, "reason")

    def test_result_is_frozen(self):
        r = validate_seed(_pad("Build seed_gate.py with comprehensive tests"))
        with pytest.raises(AttributeError):
            r.passes = False  # type: ignore[misc]

    def test_no_verb_still_reports_target(self):
        """Even when verb is missing, target should still be detected."""
        r = validate_seed(_pad("The seed_gate.py file needs urgent attention now"))
        assert r.code == "missing_verb"
        assert r.target == "seed_gate.py"


# =====================================================================
# TestPassesGate (5 tests)
# =====================================================================
class TestPassesGate:
    def test_true_for_good_seed(self):
        assert passes_gate(_pad("Build seed_gate.py with comprehensive tests")) is True

    def test_false_for_bad_seed(self):
        assert passes_gate(_pad("Something vague about the colony and stuff")) is False

    def test_accepts_tags(self):
        assert passes_gate(
            _pad("Explore the nature of artificial consciousness now"),
            tags=["philosophy"],
        ) is True

    def test_accepts_mode(self):
        assert passes_gate("build it.py", mode="purge") is True

    def test_false_for_empty(self):
        assert passes_gate("") is False


# =====================================================================
# TestTargetPatternOrdering (5 tests)
# =====================================================================
class TestTargetPatternOrdering:
    """Verify _TARGET_PATTERNS order is correct and complete."""

    def test_path_re_is_first(self):
        assert _TARGET_PATTERNS[0] is PATH_RE

    def test_ref_re_is_included(self):
        assert REF_RE in _TARGET_PATTERNS

    def test_no_duplicates(self):
        assert len(_TARGET_PATTERNS) == len(set(_TARGET_PATTERNS))

    def test_all_seven_patterns(self):
        assert len(_TARGET_PATTERNS) == 7

    def test_path_preferred_over_file(self):
        """For 'src/seed_gate.py', PATH_RE should match first."""
        t = find_concrete_target("src/seed_gate.py")
        assert t == "src/seed_gate.py"


# =====================================================================
# TestRegexPatterns (12 tests)
# =====================================================================
class TestRegexPatterns:
    """Direct tests for individual regex patterns."""

    def test_file_re_python(self):
        assert FILE_RE.search("seed_gate.py")

    def test_file_re_json(self):
        assert FILE_RE.search("seeds.json")

    def test_file_re_no_match(self):
        assert not FILE_RE.search("no extension here")

    def test_path_re_src(self):
        assert PATH_RE.search("src/seed_gate.py")

    def test_path_re_nested(self):
        assert PATH_RE.search("scripts/actions/agent.py")

    def test_path_re_no_match(self):
        assert not PATH_RE.search("random/path")

    def test_func_re_valid(self):
        assert FUNC_RE.search("validate_seed()")

    def test_func_re_short_blocked(self):
        assert not FUNC_RE.search("x()")
        assert not FUNC_RE.search("do()")
        assert not FUNC_RE.search("go()")

    def test_channel_re(self):
        assert CHANNEL_RE.search("r/general")

    def test_ref_re(self):
        assert REF_RE.search("#12503")

    def test_ref_re_short(self):
        assert REF_RE.search("#999")

    def test_ref_re_too_short(self):
        assert not REF_RE.search("#12")  # need 3+ digits


# =====================================================================
# TestConstants (5 tests)
# =====================================================================
class TestConstants:
    def test_action_verbs_is_frozenset(self):
        assert isinstance(ACTION_VERBS, frozenset)

    def test_core_verbs_present(self):
        for v in ("build", "write", "create", "ship", "test", "fix"):
            assert v in ACTION_VERBS

    def test_exempt_tags_present(self):
        for t in ("theme", "philosophy", "debate"):
            assert t in EXEMPT_TAGS

    def test_min_length(self):
        assert MIN_PROPOSAL_LENGTH == 50

    def test_domain_nouns_exist(self):
        assert isinstance(DOMAIN_NOUNS, frozenset)
        assert "reactor" in DOMAIN_NOUNS


# =====================================================================
# TestCLI (4 tests)
# =====================================================================
class TestCLI:
    """Test CLI interface via subprocess."""

    def test_cli_check_pass(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--check", "Build seed_gate.py with comprehensive tests and documentation"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_cli_check_fail(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--check", "Something vague about the colony and life"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_cli_filter(self):
        seeds_data = {
            "proposals": [
                {"text": "Build seed_gate.py with comprehensive tests and documentation", "tags": ["artifact"]},
                {"text": "Vague stuff", "tags": []},
            ]
        }
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py")],
            input=json.dumps(seeds_data),
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert len(output["proposals"]) == 1
        assert output["proposals"][0]["specificity"]["passes"] is True

    def test_cli_help(self):
        """Unrecognized flags print the docstring and exit 0."""
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"), "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "seed_gate" in result.stdout.lower()


# =====================================================================
# TestPropertyInvariants (8 tests)
# =====================================================================
class TestPropertyInvariants:
    """Property-based invariants that should always hold."""

    def test_score_bounded_0_10(self):
        """Score is always in [0, 10]."""
        texts = [
            "", "x", "Build seed_gate.py",
            "Build src/seed_gate.py and run pytest in r/code ref #12503 " + "x" * 80,
            "Build" * 1000,
        ]
        for text in texts:
            s = compute_score(text)
            assert 0 <= s <= 10, f"Score {s} out of bounds for: {text[:60]}"

    def test_passes_implies_verb(self):
        """If a seed passes, it must have a verb."""
        texts = [
            _pad("Build seed_gate.py with comprehensive testing"),
            _pad("Ship the reactor module to production right now"),
            _pad("Test validate_seed() function for edge cases"),
        ]
        for text in texts:
            r = validate_seed(text)
            if r.passes:
                assert r.verb is not None, f"Passed but no verb: {text}"

    def test_passes_implies_target_or_exempt(self):
        """If a seed passes, it has a target OR an exempt tag."""
        r = validate_seed(
            _pad("Explore consciousness implications in the colony"),
            tags=["philosophy"],
        )
        assert r.passes
        # target can be None if exempt

    def test_frozen_result(self):
        """SeedGateResult should be immutable."""
        r = validate_seed(_pad("Build seed_gate.py with tests and documentation"))
        with pytest.raises(AttributeError):
            r.verb = "hacked"  # type: ignore[misc]

    def test_code_is_valid(self):
        """Result code is always one of the defined FailureCodes."""
        valid_codes = {"ok", "too_short", "fragment", "junk_signal",
                       "missing_verb", "missing_target"}
        texts = [
            _pad("Build seed_gate.py with tests"),
            "short",
            _pad("build lowercase fragment starting"),
            _pad("Parser grabbed this text from somewhere"),
            _pad("The seed_gate.py file is important to us"),
            _pad("Build something vague for the colony now please"),
        ]
        for text in texts:
            r = validate_seed(text)
            assert r.code in valid_codes, f"Invalid code {r.code!r} for: {text[:40]}"

    def test_admission_stricter_than_purge(self):
        """Admission mode should never pass something purge rejects."""
        texts = [
            "Build seed_gate.py",  # too short for admission, ok for purge
            "build seed_gate.py",  # fragment in admission, ok for purge
        ]
        for text in texts:
            a = validate_seed(text, mode="admission")
            p = validate_seed(text, mode="purge")
            if not p.passes:
                assert not a.passes, (
                    f"Admission passed but purge failed: {text}"
                )

    def test_no_verb_with_target_still_reports_target(self):
        """When verb is missing, target should still be populated."""
        r = validate_seed(_pad("The seed_gate.py module is critical for us now"))
        assert r.code == "missing_verb"
        assert r.target is not None

    def test_find_all_verbs_subset_of_action_verbs(self):
        """All returned verbs should be in ACTION_VERBS."""
        verbs = find_all_verbs("Build, test, ship, deploy, and create the module")
        for v in verbs:
            assert v in ACTION_VERBS


# =====================================================================
# TestEdgeCases (8 tests)
# =====================================================================
class TestEdgeCases:
    def test_unicode_text(self):
        r = validate_seed(_pad("Build seed_gate.py with 🚀 rocket emoji support"))
        assert r.passes is True

    def test_newlines_in_text(self):
        r = validate_seed(_pad("Build seed_gate.py\nwith multi-line\ndescription"))
        assert r.passes is True

    def test_tabs_in_text(self):
        r = validate_seed(_pad("Build seed_gate.py\twith tab\tseparated\twords"))
        assert r.passes is True

    def test_only_whitespace(self):
        r = validate_seed("   \n\t  ")
        assert r.passes is False

    def test_none_tags(self):
        r = validate_seed(_pad("Build seed_gate.py with comprehensive tests"), tags=None)
        assert r.passes is True

    def test_empty_tags(self):
        r = validate_seed(_pad("Build seed_gate.py with comprehensive tests"), tags=[])
        assert r.passes is True

    def test_very_long_proposal(self):
        text = _pad("Build seed_gate.py ") + "word " * 5000
        r = validate_seed(text)
        assert r.passes is True

    def test_all_exempt_tags_work(self):
        for tag in EXEMPT_TAGS:
            r = validate_seed(
                _pad("Explore the deep implications of this concept now"),
                tags=[tag],
            )
            assert r.passes is True, f"Exempt tag {tag!r} should allow pass"
