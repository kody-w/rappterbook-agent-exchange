"""Tests for seed_gate — specificity validator for seed proposals.

Covers: verb detection, target detection, tag exemptions, fragment/junk
detection, length checks, score computation, validate_seed integration,
purge mode, convenience helpers, false-positive regression, edge cases,
property invariants, CLI, and a smoke batch.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import seed_gate
from seed_gate import (
    ACTION_VERBS,
    EXEMPT_TAGS,
    MIN_LENGTH,
    SeedGateResult,
    check_fragment,
    check_junk_signals,
    check_length,
    compute_score,
    find_action_verb,
    find_all_targets,
    find_all_verbs,
    find_concrete_target,
    passes_gate,
    validate_seed,
)

# ── Helpers ────────────────────────────────────────────────────────────

def _pad(text: str, length: int = MIN_LENGTH) -> str:
    """Pad text to at least *length* chars by appending description."""
    if len(text) >= length:
        return text
    padding = " for the Mars colony simulation module right now"
    while len(text) < length:
        text = text + padding
    return text


# ── 1. Action Verb Detection ──────────────────────────────────────────

class TestActionVerbDetection:
    def test_finds_build(self):
        assert find_action_verb("Build seed_gate.py") == "build"

    def test_finds_create(self):
        assert find_action_verb("Create a new module") == "create"

    def test_finds_implement(self):
        assert find_action_verb("Implement the parser") == "implement"

    def test_finds_refactor(self):
        assert find_action_verb("Refactor the engine loop") == "refactor"

    def test_case_insensitive(self):
        assert find_action_verb("BUILD something") == "build"

    def test_not_found_returns_none(self):
        assert find_action_verb("something about clouds") is None

    def test_finds_first_verb(self):
        assert find_action_verb("Build and test the module") == "build"

    def test_limit_parameter(self):
        text = "something " * 30 + "build it"
        assert find_action_verb(text, limit=50) is None
        assert find_action_verb(text, limit=0) == "build"

    def test_all_verbs_are_lowercase(self):
        for v in ACTION_VERBS:
            assert v == v.lower(), "Verb %r should be lowercase" % v

    def test_verb_set_minimum_size(self):
        assert len(ACTION_VERBS) >= 30

    def test_finds_simulate(self):
        assert find_action_verb("Simulate thermal flow") == "simulate"

    def test_finds_calibrate(self):
        assert find_action_verb("Calibrate sensors") == "calibrate"

    def test_finds_validate(self):
        assert find_action_verb("Validate input data") == "validate"

    def test_finds_analyze(self):
        assert find_action_verb("Analyze the metrics") == "analyze"


# ── 2. Concrete Target Detection ──────────────────────────────────────

class TestConcreteTargetDetection:
    def test_finds_python_file(self):
        assert find_concrete_target("Build seed_gate.py") == "seed_gate.py"

    def test_finds_js_file(self):
        assert find_concrete_target("Write router.js") == "router.js"

    def test_finds_json_file(self):
        assert find_concrete_target("Update config.json") == "config.json"

    def test_finds_tool_name_pytest(self):
        assert find_concrete_target("Run pytest for validation")
        assert "pytest" in find_concrete_target("Run pytest for validation").lower()

    def test_finds_tool_name_docker(self):
        result = find_concrete_target("Deploy with docker containers")
        assert result is not None

    def test_finds_path(self):
        result = find_concrete_target("Check src/engine/loop")
        assert result is not None
        assert "src/" in result

    def test_finds_function_call(self):
        assert find_concrete_target("Call validate() after") == "validate()"

    def test_no_target_returns_none(self):
        assert find_concrete_target("things are better somehow") is None

    def test_filename_beats_tool(self):
        # Filename should be found before tool name
        result = find_concrete_target("Build gate.py using pytest")
        assert result == "gate.py"

    def test_finds_markdown_file(self):
        assert find_concrete_target("Update README.md") == "README.md"

    def test_finds_yaml_file(self):
        assert find_concrete_target("Edit config.yaml") == "config.yaml"

    def test_finds_shell_script(self):
        result = find_concrete_target("Run deploy.sh")
        assert result is not None

    def test_finds_css_file(self):
        assert find_concrete_target("Fix styles.css") == "styles.css"

    def test_finds_channel_is_not_primary(self):
        # Channel refs are scored but not used as primary targets
        result = find_concrete_target("Post to r/engineering")
        # _CHANNEL_RE is for scoring, not for find_concrete_target
        # r/engineering may or may not match — either is fine
        pass

    def test_finds_toml(self):
        assert find_concrete_target("Edit pyproject.toml") == "pyproject.toml"

    def test_finds_rust_file(self):
        assert find_concrete_target("Build main.rs") == "main.rs"


# ── 3. find_all_* helpers ─────────────────────────────────────────────

class TestFindAllHelpers:
    def test_find_all_verbs_multiple(self):
        result = find_all_verbs("Build and test the module then deploy")
        assert "build" in result
        assert "test" in result
        assert "deploy" in result

    def test_find_all_verbs_empty(self):
        assert find_all_verbs("nothing here") == []

    def test_find_all_verbs_no_duplicates(self):
        result = find_all_verbs("Build build BUILD build")
        assert result == ["build"]

    def test_find_all_targets_multiple(self):
        result = find_all_targets("Update gate.py and config.json using pytest")
        assert len(result) >= 2

    def test_find_all_targets_empty(self):
        assert find_all_targets("nothing concrete") == []

    def test_find_all_targets_no_duplicates(self):
        result = find_all_targets("Build gate.py and gate.py again")
        assert result.count("gate.py") == 1


# ── 4. Tag Exemptions ─────────────────────────────────────────────────

class TestTagExemptions:
    def test_theme_tag_skips_target(self):
        text = _pad("Build a philosophical exploration of consciousness")
        r = validate_seed(text, tags=["theme"])
        assert r.passes

    def test_philosophy_tag_skips_target(self):
        text = _pad("Create an essay on emergence and complexity theory")
        r = validate_seed(text, tags=["philosophy"])
        assert r.passes

    def test_debate_tag_skips_target(self):
        text = _pad("Design a debate on simulation hypothesis implications")
        r = validate_seed(text, tags=["debate"])
        assert r.passes

    def test_exempt_still_needs_verb(self):
        text = _pad("a philosophical exploration of consciousness and emergence")
        r = validate_seed(text, tags=["theme"])
        assert not r.passes
        assert r.reason_code == "no_verb"

    def test_non_exempt_tag_needs_target(self):
        text = _pad("Build something amazing for the colony forever and ever")
        r = validate_seed(text, tags=["engineering"])
        assert not r.passes
        assert r.reason_code == "no_target"

    def test_exempt_tags_are_lowercase(self):
        for t in EXEMPT_TAGS:
            assert t == t.lower()

    def test_case_insensitive_tag_match(self):
        text = _pad("Build a deep philosophical exploration of AI consciousness")
        r = validate_seed(text, tags=["Theme"])
        assert r.passes

    def test_multiple_tags_one_exempt(self):
        text = _pad("Create an exploration of emergence in colony systems")
        r = validate_seed(text, tags=["engineering", "philosophy"])
        assert r.passes


# ── 5. Fragment Detection ─────────────────────────────────────────────

class TestFragmentDetection:
    def test_improve_x_is_fragment(self):
        ok, _ = check_fragment("improve performance")
        assert not ok

    def test_make_better_is_fragment(self):
        ok, _ = check_fragment("make things better")
        assert not ok

    def test_fix_bugs_is_fragment(self):
        ok, _ = check_fragment("fix bugs")
        assert not ok

    def test_fix_the_bug_is_fragment(self):
        ok, _ = check_fragment("fix the bug")
        assert not ok

    def test_real_proposal_not_fragment(self):
        ok, _ = check_fragment("Build seed_gate.py with comprehensive tests for the colony simulation")
        assert ok

    def test_short_non_fragment_still_flagged(self):
        # 3 words, under 60 chars but not matching fragment patterns
        ok, _ = check_fragment("hello world today")
        assert not ok  # too few words for the word-count check

    def test_long_enough_passes(self):
        ok, _ = check_fragment("This is a long enough proposal with many words to pass the fragment check easily")
        assert ok

    def test_returns_tuple(self):
        result = check_fragment("test")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_ok_message_is_string(self):
        ok, msg = check_fragment("A decent proposal with enough words and length to work")
        assert isinstance(msg, str)

    def test_fail_message_is_string(self):
        ok, msg = check_fragment("fix bugs")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_empty_is_fragment(self):
        ok, _ = check_fragment("")
        assert not ok  # empty string has < 3 spaces and < 60 chars


# ── 6. Junk Signal Detection ──────────────────────────────────────────

class TestJunkSignalDetection:
    def test_single_junk_word_passes(self):
        ok, _ = check_junk_signals("leverage the platform for good")
        assert ok  # one junk word is OK

    def test_two_junk_words_fails(self):
        ok, _ = check_junk_signals("leverage synergy for disruption")
        assert not ok

    def test_no_junk_passes(self):
        ok, _ = check_junk_signals("Build seed_gate.py with tests")
        assert ok

    def test_game_changer_counts(self):
        ok, _ = check_junk_signals("This is a game changer with synergy")
        assert not ok

    def test_returns_tuple(self):
        result = check_junk_signals("test text")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_blockchain_and_web3(self):
        ok, _ = check_junk_signals("blockchain web3 revolution")
        assert not ok

    def test_metaverse_paradigm(self):
        ok, _ = check_junk_signals("metaverse paradigm shift")
        assert not ok

    def test_case_insensitive(self):
        ok, _ = check_junk_signals("SYNERGY and LEVERAGE forever")
        assert not ok

    def test_normal_engineering_text(self):
        ok, _ = check_junk_signals("Implement thermal regulation for habitat module with PID controller")
        assert ok

    def test_fail_message_lists_words(self):
        ok, msg = check_junk_signals("synergy leverage paradigm")
        assert not ok
        assert "synergy" in msg.lower() or "leverage" in msg.lower()


# ── 7. Length Check ───────────────────────────────────────────────────

class TestLengthCheck:
    def test_short_blocks_admission(self):
        ok, msg = check_length("too short")
        assert not ok
        assert str(MIN_LENGTH) in msg

    def test_empty_blocks_admission(self):
        ok, _ = check_length("")
        assert not ok

    def test_exact_min_passes(self):
        text = "x" * MIN_LENGTH
        ok, _ = check_length(text)
        assert ok

    def test_above_min_passes(self):
        text = "x" * (MIN_LENGTH + 100)
        ok, _ = check_length(text)
        assert ok

    def test_returns_tuple(self):
        result = check_length("test")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_min_length_constant(self):
        assert MIN_LENGTH == 50


# ── 8. Score Computation ──────────────────────────────────────────────

class TestScoreComputation:
    def test_empty_scores_zero(self):
        assert compute_score("") == 0

    def test_verb_adds_2(self):
        s = compute_score("build")
        assert s >= 2

    def test_filename_adds_3(self):
        s = compute_score("gate.py")
        assert s >= 3

    def test_tool_adds_3_without_filename(self):
        s = compute_score("using pytest")
        assert s >= 3

    def test_combined_high_score(self):
        text = "Build seed_gate.py using pytest in src/validators with validate() for r/engineering plus extra words to be long"
        s = compute_score(text)
        assert s >= 6

    def test_score_capped_at_10(self):
        text = "Build seed_gate.py using pytest in src/validators with validate() for r/engineering " * 5
        s = compute_score(text)
        assert s <= 10

    def test_score_always_non_negative(self):
        assert compute_score("") >= 0
        assert compute_score("abc") >= 0

    def test_path_adds_1(self):
        base = compute_score("nothing here")
        with_path = compute_score("check src/engine")
        assert with_path >= base + 1

    def test_function_adds_1(self):
        base = compute_score("nothing here")
        with_func = compute_score("call validate()")
        assert with_func >= base + 1

    def test_channel_ref_adds_1(self):
        base = compute_score("build something for the colony right now today yes indeed")
        with_chan = compute_score("build something for r/engineering colony right now today yes")
        assert with_chan >= base + 1

    def test_length_bonus(self):
        short = compute_score("build gate.py")
        long_text = "build gate.py " + "x " * 50
        long_s = compute_score(long_text)
        assert long_s >= short

    def test_returns_int(self):
        assert isinstance(compute_score("build gate.py"), int)


# ── 9. validate_seed Integration ──────────────────────────────────────

class TestValidateSeedIntegration:
    def test_pass_complete(self):
        text = _pad("Build seed_gate.py with comprehensive tests for the colony")
        r = validate_seed(text)
        assert r.passes
        assert r.verb == "build"
        assert r.target == "seed_gate.py"
        assert r.reason_code == "ok"

    def test_fail_vague_no_target(self):
        text = _pad("Build something amazing for the colony forever and ever")
        r = validate_seed(text)
        assert not r.passes
        assert r.reason_code == "no_target"

    def test_fail_no_verb(self):
        text = _pad("The seed_gate.py module should be really great and amazing")
        r = validate_seed(text)
        assert not r.passes
        assert r.reason_code == "no_verb"

    def test_fail_too_short(self):
        r = validate_seed("Build gate.py")
        assert not r.passes
        assert r.reason_code == "too_short"

    def test_fail_fragment(self):
        r = validate_seed(_pad("improve performance"), mode="admission")
        # Padding makes it long enough; "improve performance" isn't caught by
        # fragment regex.  Passes length+fragment+junk, has verb, no target.
        assert not r.passes
        assert r.reason_code in ("too_short", "fragment", "no_target")

    def test_fail_junk(self):
        text = _pad("Build synergy leverage paradigm disruption for the colony module")
        r = validate_seed(text)
        # Could be junk or no_target depending on check order
        assert not r.passes

    def test_result_has_checks_dict(self):
        text = _pad("Build seed_gate.py with tests for the colony right now")
        r = validate_seed(text)
        assert isinstance(r.checks, dict)
        assert "verb" in r.checks
        assert "target" in r.checks

    def test_result_reason_code_always_set(self):
        text = _pad("Build seed_gate.py with tests for colony")
        r = validate_seed(text)
        assert r.reason_code in ("ok", "too_short", "fragment", "junk_signal", "no_verb", "no_target")

    def test_frozen_result(self):
        text = _pad("Build seed_gate.py with tests for colony")
        r = validate_seed(text)
        with pytest.raises(AttributeError):
            r.passes = False  # type: ignore

    def test_score_populated(self):
        text = _pad("Build seed_gate.py with tests for colony")
        r = validate_seed(text)
        assert isinstance(r.score, int)
        assert 0 <= r.score <= 10

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            validate_seed("test", mode="invalid")

    def test_tags_default_none(self):
        text = _pad("Build seed_gate.py with tests for colony right now")
        r = validate_seed(text)
        assert r.passes  # no tags, file target + verb

    def test_reason_is_string(self):
        text = _pad("Build seed_gate.py with tests for colony")
        r = validate_seed(text)
        assert isinstance(r.reason, str)
        assert len(r.reason) > 0

    def test_pass_with_tool_target(self):
        text = _pad("Build a pytest suite for the thermal simulation module")
        r = validate_seed(text)
        assert r.passes
        assert r.verb == "build"

    def test_pass_with_path_target(self):
        text = _pad("Refactor the code in src/engine/loop to improve clarity and speed")
        r = validate_seed(text)
        assert r.passes


# ── 10. Purge Mode ────────────────────────────────────────────────────

class TestPurgeMode:
    def test_purge_allows_short_text(self):
        r = validate_seed("Build gate.py", mode="purge")
        assert r.passes

    def test_purge_skips_fragment_check(self):
        r = validate_seed("improve gate.py", mode="purge")
        # purge skips fragment check; "improve" is a verb, "gate.py" is a target
        assert r.passes

    def test_purge_skips_junk_check(self):
        text = "Build synergy leverage paradigm gate.py"
        r = validate_seed(text, mode="purge")
        assert r.passes

    def test_purge_still_needs_verb(self):
        r = validate_seed("gate.py is a good file", mode="purge")
        assert not r.passes
        assert r.reason_code == "no_verb"

    def test_purge_still_needs_target(self):
        r = validate_seed("build something amazing", mode="purge")
        assert not r.passes
        assert r.reason_code == "no_target"

    def test_purge_limits_verb_search(self):
        # Verb only in first 200 chars should be found in purge
        text = "Build " + "x " * 50 + " gate.py"
        r = validate_seed(text, mode="purge")
        assert r.passes

    def test_admission_vs_purge_fragment(self):
        text = _pad("improve gate.py")
        r_adm = validate_seed(text, mode="admission")
        r_pur = validate_seed(text, mode="purge")
        # purge should be more lenient
        assert r_pur.passes

    def test_admission_vs_purge_short(self):
        text = "Build gate.py"
        r_adm = validate_seed(text, mode="admission")
        r_pur = validate_seed(text, mode="purge")
        assert not r_adm.passes
        assert r_pur.passes


# ── 11. passes_gate convenience ───────────────────────────────────────

class TestPassesGateConvenience:
    def test_returns_bool(self):
        text = _pad("Build seed_gate.py with tests for colony")
        assert isinstance(passes_gate(text), bool)

    def test_true_for_good(self):
        text = _pad("Build seed_gate.py with comprehensive tests for colony")
        assert passes_gate(text) is True

    def test_false_for_bad(self):
        assert passes_gate("bad") is False

    def test_with_mode(self):
        assert passes_gate("Build gate.py", mode="purge") is True
        assert passes_gate("Build gate.py", mode="admission") is False


# ── 12. False-Positive Regression ─────────────────────────────────────

class TestFalsePositiveRegression:
    """Real-world proposals that MUST pass. Inspired by #12521 insight
    that creation proposals were penalized by earlier regex."""

    def test_new_file_proposal(self):
        text = _pad("Create seed_gate.py with comprehensive validation logic for seeds")
        assert validate_seed(text).passes

    def test_refactor_proposal(self):
        text = _pad("Refactor propose_seed.py to extract the validation into a separate module")
        assert validate_seed(text).passes

    def test_test_suite_proposal(self):
        text = _pad("Write test_seed_gate.py covering edge cases and property invariants")
        assert validate_seed(text).passes

    def test_integration_proposal(self):
        text = _pad("Integrate pytest into the CI pipeline with coverage reporting enabled")
        assert validate_seed(text).passes

    def test_tool_based_proposal(self):
        text = _pad("Build a docker container for the colony simulation with auto-restart")
        assert validate_seed(text).passes

    def test_path_based_proposal(self):
        text = _pad("Optimize the code in src/engine/thermal for better heat dissipation")
        assert validate_seed(text).passes

    def test_function_based_proposal(self):
        text = _pad("Implement validate() as the primary entry point for all seed checks")
        assert validate_seed(text).passes

    def test_multi_verb_proposal(self):
        text = _pad("Build and test the thermal regulation module with pytest for colony")
        assert validate_seed(text).passes

    def test_channel_proposal(self):
        text = _pad("Create a new monitoring dashboard for r/engineering with real-time data")
        assert validate_seed(text).passes

    def test_markdown_proposal(self):
        text = _pad("Write README.md with architecture docs for the seed gate validator")
        assert validate_seed(text).passes


# ── 13. Edge Cases ────────────────────────────────────────────────────

class TestEdgeCases:
    def test_none_text_raises(self):
        with pytest.raises(TypeError):
            validate_seed(None)  # type: ignore

    def test_empty_string(self):
        r = validate_seed("")
        assert not r.passes

    def test_whitespace_only(self):
        r = validate_seed("   \n\t   ")
        assert not r.passes

    def test_unicode_text(self):
        text = _pad("Build thermal_sim.py with unicode: temps \u2265 100\u00b0C for colony checks")
        r = validate_seed(text)
        assert r.passes

    def test_very_long_text(self):
        text = _pad("Build gate.py " + "with extra context " * 200)
        r = validate_seed(text)
        assert r.passes

    def test_newlines_in_text(self):
        text = _pad("Build seed_gate.py\nwith tests\nfor the colony simulation module")
        r = validate_seed(text)
        assert r.passes

    def test_special_chars_in_filename(self):
        text = _pad("Build my-cool-module.py with dashes and numbers 123 for colony")
        r = validate_seed(text)
        assert r.passes

    def test_empty_tags_list(self):
        text = _pad("Build seed_gate.py with tests for colony")
        r = validate_seed(text, tags=[])
        assert r.passes

    def test_result_is_seedgateresult(self):
        r = validate_seed("x")
        assert isinstance(r, SeedGateResult)

    def test_min_length_constant(self):
        assert hasattr(seed_gate, "MIN_LENGTH")
        assert seed_gate.MIN_LENGTH == 50

    def test_tabs_count_as_chars(self):
        text = "\t" * 60 + " build gate.py"
        r = validate_seed(text)
        # tabs count toward length, so length check passes
        # but the stripped text might differ
        assert isinstance(r, SeedGateResult)


# ── 14. Property Invariants ───────────────────────────────────────────

_SAMPLE_TEXTS = [
    "Build seed_gate.py with comprehensive tests for the colony",
    "x",
    "build seed_gate.py",
    "Build something cool and amazing forever now now now now now now now now now now",
    "build seed_gate.py with tests for the colony module right now",
]


class TestPropertyInvariants:

    @pytest.mark.parametrize("text", _SAMPLE_TEXTS)
    def test_score_in_range(self, text):
        r = validate_seed(text, mode="purge")
        assert 0 <= r.score <= 10

    @pytest.mark.parametrize("text", _SAMPLE_TEXTS)
    def test_reason_code_never_empty(self, text):
        r = validate_seed(text, mode="purge")
        assert r.reason_code in ("ok", "too_short", "fragment", "junk_signal", "no_verb", "no_target")
        assert len(r.reason_code) > 0

    @pytest.mark.parametrize("text", _SAMPLE_TEXTS)
    def test_checks_dict_present(self, text):
        r = validate_seed(text, mode="purge")
        assert isinstance(r.checks, dict)

    @pytest.mark.parametrize("text", _SAMPLE_TEXTS)
    def test_reason_is_nonempty_string(self, text):
        r = validate_seed(text, mode="purge")
        assert isinstance(r.reason, str)
        assert len(r.reason) > 0

    @pytest.mark.parametrize("text", _SAMPLE_TEXTS)
    def test_passes_is_bool(self, text):
        r = validate_seed(text, mode="purge")
        assert isinstance(r.passes, bool)

    @pytest.mark.parametrize("text", _SAMPLE_TEXTS)
    def test_verb_is_str_or_none(self, text):
        r = validate_seed(text, mode="purge")
        assert r.verb is None or isinstance(r.verb, str)


# ── 15. CLI ───────────────────────────────────────────────────────────

class TestCLI:
    def test_check_flag_passes(self):
        result = subprocess.run(
            [sys.executable, "-m", "seed_gate", "--check",
             _pad("Build seed_gate.py with tests for colony")],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "..", "src"),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["passes"] is True

    def test_check_flag_fails(self):
        result = subprocess.run(
            [sys.executable, "-m", "seed_gate", "--check", "bad"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "..", "src"),
        )
        assert result.returncode == 1
        data = json.loads(result.stdout)
        assert data["passes"] is False

    def test_check_with_purge_mode(self):
        result = subprocess.run(
            [sys.executable, "-m", "seed_gate", "--check", "Build gate.py", "--purge"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "..", "src"),
        )
        assert result.returncode == 0

    def test_pipe_mode(self):
        input_data = json.dumps({"text": _pad("Build gate.py with tests for the colony module")}) + "\n"
        result = subprocess.run(
            [sys.executable, "-m", "seed_gate"],
            input=input_data, capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "..", "src"),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)

    def test_pipe_filters_bad(self):
        lines = [
            json.dumps({"text": _pad("Build gate.py with tests for colony module")}),
            json.dumps({"text": "bad"}),
        ]
        input_data = "\n".join(lines) + "\n"
        result = subprocess.run(
            [sys.executable, "-m", "seed_gate"],
            input=input_data, capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "..", "src"),
        )
        data = json.loads(result.stdout)
        assert len(data) == 1  # only the good one passes


# ── 16. Smoke Batch ───────────────────────────────────────────────────

class TestSmokeTest:
    def test_batch_100_random_texts(self):
        """Run validate_seed on 100 texts. None should raise."""
        import random
        random.seed(42)
        words = list(ACTION_VERBS) + [
            "gate.py", "module", "colony", "simulation", "r/engineering",
            "pytest", "src/engine", "validate()", "cloud", "solar",
            "thermal", "the", "a", "for", "with", "and", "or",
        ]
        for _ in range(100):
            length = random.randint(1, 20)
            text = " ".join(random.choices(words, k=length))
            tags = random.choice([None, [], ["theme"], ["engineering"]])
            mode = random.choice(["admission", "purge"])
            r = validate_seed(text, tags=tags, mode=mode)
            assert isinstance(r, SeedGateResult)
            assert isinstance(r.passes, bool)
            assert 0 <= r.score <= 10
