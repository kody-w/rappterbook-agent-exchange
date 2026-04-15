"""Tests for seed_gate.py -- canonical specificity validator.

Covers: verb detection, target detection, junk detection, scoring,
validation pass/fail, exempt tags, CLI, real-world proposals,
edge cases, property invariants, smoke tests.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seed_gate import (  # noqa: E402
    ACTION_VERBS,
    EXEMPT_TAGS,
    CHANNEL_RE,
    CLI_RE,
    DISCUSSION_RE,
    FILE_RE,
    QUOTED_RE,
    TOOL_RE,
    SeedGateResult,
    passes_gate,
    validate,
    validate_seed,
)


# ===================================================================
# Helpers
# ===================================================================

def _v(text: str, tags: list[str] | None = None, mode: str = "admission"):
    """Shorthand: validate and return dict."""
    return validate(text, tags or [], mode)


def _vs(text: str, tags: list[str] | None = None, mode: str = "admission"):
    """Shorthand: validate_seed and return SeedGateResult."""
    return validate_seed(text, tags or [], mode)


# ===================================================================
# 1. Constants sanity
# ===================================================================

class TestConstants:
    def test_action_verbs_nonempty(self):
        assert len(ACTION_VERBS) >= 40

    def test_action_verbs_all_lowercase(self):
        for v in ACTION_VERBS:
            assert v == v.lower(), f"Verb {v!r} not lowercase"

    def test_exempt_tags_nonempty(self):
        assert len(EXEMPT_TAGS) >= 4

    def test_exempt_tags_all_lowercase(self):
        for t in EXEMPT_TAGS:
            assert t == t.lower(), f"Tag {t!r} not lowercase"


# ===================================================================
# 2. Regex patterns
# ===================================================================

class TestRegexPatterns:
    def test_file_re_matches_python(self):
        assert FILE_RE.search("Build seed_gate.py validator")

    def test_file_re_matches_json_path(self):
        assert FILE_RE.search("Fix state/agents.json integrity")

    def test_file_re_matches_rust(self):
        assert FILE_RE.search("Port parser to grammar.rs")

    def test_tool_re_matches_snake_case(self):
        assert TOOL_RE.search("Refactor process_inbox handler")

    def test_tool_re_no_single_segment(self):
        assert not TOOL_RE.search("Build parser module")

    def test_cli_re_matches_backtick(self):
        assert CLI_RE.search("Run `pytest -v` on CI")

    def test_cli_re_matches_flag(self):
        assert CLI_RE.search("Add --verbose flag")

    def test_cli_re_matches_short_flag(self):
        assert CLI_RE.search("Support -f option")

    def test_discussion_re_matches(self):
        assert DISCUSSION_RE.search("See discussion #12503 for context")

    def test_discussion_re_rejects_short(self):
        assert not DISCUSSION_RE.search("Issue #42 is small")

    def test_channel_re_matches_r(self):
        assert CHANNEL_RE.search("Post to r/engineering about it")

    def test_channel_re_matches_c(self):
        assert CHANNEL_RE.search("Create c/mars-colony channel")

    def test_quoted_re_matches_double(self):
        assert QUOTED_RE.search('Build a "Mars landing simulator" tool')

    def test_quoted_re_matches_single(self):
        assert QUOTED_RE.search("Ship the 'thermal regulator' module")

    def test_quoted_re_rejects_short(self):
        assert not QUOTED_RE.search('Fix "ab" thing')


# ===================================================================
# 3. Verb detection
# ===================================================================

class TestVerbDetection:
    def test_verb_found(self):
        r = _v("Build seed_gate.py validator")
        assert r["verb"] == "build"

    def test_verb_case_insensitive(self):
        r = _v("IMPLEMENT the frobulator.py module")
        assert r["verb"] == "implement"

    def test_no_verb(self):
        r = _v("The frobulator.py module is interesting")
        assert r["verb"] == ""

    def test_verb_first_match(self):
        r = _v("Build and deploy thermal_model.py")
        assert r["verb"] == "build"

    def test_all_verbs_detectable(self):
        for verb in sorted(ACTION_VERBS):
            r = _v(f"{verb.capitalize()} the widget.py module")
            assert r["verb"] == verb, f"Verb {verb!r} not detected"


# ===================================================================
# 4. Target detection
# ===================================================================

class TestTargetDetection:
    def test_file_target(self):
        r = _v("Build seed_gate.py validator")
        assert "seed_gate.py" in r["target"]

    def test_tool_target(self):
        r = _v("Refactor the process_inbox handler")
        assert "process_inbox" in r["target"]

    def test_discussion_target(self):
        r = _v("Implement feedback from #12503")
        assert "#12503" in r["target"]

    def test_channel_target(self):
        r = _v("Create posts for r/engineering channel")
        assert "r/engineering" in r["target"]

    def test_cli_target(self):
        r = _v("Build a `seed_validator` command")
        assert "seed_validator" in r["target"]

    def test_quoted_target(self):
        r = _v('Build a "Mars landing pad" module')
        assert "Mars landing pad" in r["target"]

    def test_no_target(self):
        r = _v("Build something really cool and awesome")
        assert r["target"] == ""


# ===================================================================
# 5. Junk detection
# ===================================================================

class TestJunkDetection:
    def test_empty_string(self):
        r = _v("")
        assert not r["passed"]
        assert r["code"] == "junk"

    def test_whitespace_only(self):
        r = _v("   \n\t  ")
        assert not r["passed"]
        assert r["code"] == "junk"

    def test_too_short(self):
        r = _v("Fix it")
        assert not r["passed"]
        assert r["code"] == "junk"

    def test_starts_lowercase(self):
        r = _v("the seed_gate.py module needs refactoring")
        assert not r["passed"]
        assert r["code"] == "junk"

    def test_starts_backtick(self):
        r = _v("`seed_gate` needs some work done on it")
        assert not r["passed"]
        assert r["code"] == "junk"

    def test_starts_pipe(self):
        r = _v("| column | data | from table")
        assert not r["passed"]
        assert r["code"] == "junk"

    def test_starts_comma(self):
        r = _v(", and then build the validator.py")
        assert not r["passed"]
        assert r["code"] == "junk"

    def test_numbered_list(self):
        r = _v("1. Build the seed_gate.py validator module")
        assert not r["passed"]
        assert r["code"] == "junk"

    def test_bare_url(self):
        r = _v("https://example.com/some/path/here")
        assert not r["passed"]
        assert r["code"] == "junk"

    def test_todo_comment(self):
        r = _v("TODO: Build the seed_gate.py module later")
        assert not r["passed"]
        assert r["code"] == "junk"

    def test_run_prefix_exception(self):
        """run_* prefix is allowed even though it starts lowercase."""
        r = _v("run_simulation for thermal_model.py system")
        assert r["code"] != "junk"


# ===================================================================
# 6. Pass / fail decisions
# ===================================================================

class TestPassFail:
    def test_verb_plus_file_passes(self):
        r = _v("Build seed_gate.py validator")
        assert r["passed"] is True

    def test_verb_plus_tool_passes(self):
        r = _v("Refactor the process_inbox handler")
        assert r["passed"] is True

    def test_verb_plus_discussion_passes(self):
        r = _v("Implement feedback from discussion #12503")
        assert r["passed"] is True

    def test_no_verb_fails(self):
        r = _v("The frobulator.py module is interesting")
        assert r["passed"] is False
        assert r["code"] == "no_verb"

    def test_verb_no_target_fails(self):
        r = _v("Build something really amazing and cool")
        assert r["passed"] is False
        assert r["code"] == "no_target"

    def test_both_missing_fails(self):
        r = _v("Something really amazing and impressive here")
        assert r["passed"] is False


# ===================================================================
# 7. Exempt tags
# ===================================================================

class TestExemptTags:
    def test_theme_exempt(self):
        r = _v("Build a philosophical framework for agent governance", ["theme"])
        assert r["passed"] is True

    def test_philosophy_exempt(self):
        r = _v("Design principles for agent autonomy and freedom", ["philosophy"])
        assert r["passed"] is True

    def test_debate_exempt(self):
        r = _v("Create arguments for decentralized agent networks", ["debate"])
        assert r["passed"] is True

    def test_exploration_exempt(self):
        r = _v("Map the frontier of emergent agent behavior", ["exploration"])
        assert r["passed"] is True

    def test_story_exempt(self):
        r = _v("Write about the founding of the agent colony", ["story"])
        assert r["passed"] is True

    def test_lore_exempt(self):
        r = _v("Document the ancient protocols of agent communication", ["lore"])
        assert r["passed"] is True

    def test_exempt_still_needs_verb(self):
        r = _v("Interesting thoughts on agent governance", ["theme"])
        assert r["passed"] is False
        assert "No action verb" in r["reasons"][0]

    def test_nonexempt_tag_no_help(self):
        r = _v("Build something cool for the platform", ["feature"])
        assert r["passed"] is False

    def test_exempt_case_insensitive(self):
        r = _v("Build a framework for agent governance", ["THEME"])
        assert r["passed"] is True

    def test_multiple_tags_one_exempt(self):
        r = _v("Design agent governance principles", ["feature", "philosophy"])
        assert r["passed"] is True

    def test_exempt_with_target_still_passes(self):
        r = _v("Build seed_gate.py with philosophical rigor", ["philosophy"])
        assert r["passed"] is True


# ===================================================================
# 8. Purge mode
# ===================================================================

class TestPurgeMode:
    def test_purge_accepts_clean_proposal(self):
        r = _v("Build seed_gate.py validator module", mode="purge")
        assert r["passed"] is True

    def test_purge_rejects_junk(self):
        r = _v("", mode="purge")
        assert r["passed"] is False

    def test_purge_verb_in_first_200(self):
        padding = "X " * 90  # 180 chars
        text = f"Build {padding} seed_gate.py"
        r = _v(text, mode="purge")
        assert r["verb"] == "build"

    def test_purge_verb_beyond_200_not_found(self):
        padding = "X " * 110  # 220 chars
        text = f"{padding} Build seed_gate.py"
        r = _v(text, mode="purge")
        assert r["verb"] == ""

    def test_purge_junk_limit_60(self):
        """Junk detection in purge mode only looks at first 60 chars."""
        # First 60 chars are clean, junk signal after
        clean = "Build the seed_gate.py validator for proposal specificity"  # 58 chars
        text = clean + " TODO: some leftover comment here"
        r = _v(text, mode="purge")
        # Should pass because junk check only sees first 60 chars
        assert r["code"] != "junk"


# ===================================================================
# 9. Scoring
# ===================================================================

class TestScoring:
    def test_score_zero_for_junk(self):
        r = _v("")
        assert r["score"] == 0

    def test_score_nonzero_for_valid(self):
        r = _v("Build seed_gate.py validator")
        assert r["score"] >= 5

    def test_score_higher_with_multiple_targets(self):
        simple = _v("Build seed_gate.py validator")
        rich = _v("Build seed_gate.py and wire into process_inbox.py")
        assert rich["score"] >= simple["score"]

    def test_score_max_10(self):
        text = "Build, test, and deploy seed_gate.py process_inbox.py " * 5
        r = _v(text)
        assert r["score"] <= 10

    def test_score_higher_with_length(self):
        short = _v("Build seed_gate.py validator")
        long = _v(
            "Build seed_gate.py validator that checks for action verbs "
            "and concrete targets like filenames and tool names"
        )
        assert long["score"] >= short["score"]


# ===================================================================
# 10. Dict API contract (critical for propose_seed.py)
# ===================================================================

class TestValidateDictAPI:
    def test_returns_dict(self):
        r = _v("Build seed_gate.py validator")
        assert isinstance(r, dict)

    def test_has_passed_key(self):
        r = _v("Build seed_gate.py validator")
        assert "passed" in r

    def test_passed_is_bool(self):
        r = _v("Build seed_gate.py validator")
        assert isinstance(r["passed"], bool)

    def test_has_reasons_key(self):
        r = _v("Build seed_gate.py validator")
        assert "reasons" in r

    def test_reasons_is_list(self):
        r = _v("Build seed_gate.py validator")
        assert isinstance(r["reasons"], list)

    def test_reasons_items_are_strings(self):
        r = _v("Something that fails validation somehow")
        for item in r["reasons"]:
            assert isinstance(item, str)

    def test_has_score_key(self):
        r = _v("Build seed_gate.py validator")
        assert "score" in r

    def test_has_verb_key(self):
        r = _v("Build seed_gate.py validator")
        assert "verb" in r

    def test_has_target_key(self):
        r = _v("Build seed_gate.py validator")
        assert "target" in r

    def test_has_code_key(self):
        r = _v("Build seed_gate.py validator")
        assert "code" in r

    def test_json_serializable(self):
        r = _v("Build seed_gate.py validator")
        dumped = json.dumps(r)
        loaded = json.loads(dumped)
        assert loaded == r


# ===================================================================
# 11. Dataclass API
# ===================================================================

class TestDataclassAPI:
    def test_returns_seed_gate_result(self):
        r = _vs("Build seed_gate.py validator")
        assert isinstance(r, SeedGateResult)

    def test_is_frozen(self):
        r = _vs("Build seed_gate.py validator")
        with pytest.raises(AttributeError):
            r.passes = False  # type: ignore[misc]

    def test_to_dict_matches_validate(self):
        text = "Build seed_gate.py validator"
        dict_result = _v(text)
        dc_result = _vs(text)
        assert dc_result.to_dict() == dict_result

    def test_passes_bool(self):
        r = _vs("Build seed_gate.py validator")
        assert isinstance(r.passes, bool)
        assert r.passes is True

    def test_reasons_list(self):
        r = _vs("Build seed_gate.py validator")
        assert isinstance(r.reasons, list)

    def test_score_int(self):
        r = _vs("Build seed_gate.py validator")
        assert isinstance(r.score, int)

    def test_verb_str(self):
        r = _vs("Build seed_gate.py validator")
        assert isinstance(r.verb, str)

    def test_target_str(self):
        r = _vs("Build seed_gate.py validator")
        assert isinstance(r.target, str)


# ===================================================================
# 12. passes_gate convenience
# ===================================================================

class TestPassesGate:
    def test_returns_bool(self):
        assert isinstance(passes_gate("Build seed_gate.py"), bool)

    def test_true_for_valid(self):
        assert passes_gate("Build seed_gate.py validator") is True

    def test_false_for_invalid(self):
        assert passes_gate("") is False

    def test_false_for_no_verb(self):
        assert passes_gate("The seed_gate.py module") is False


# ===================================================================
# 13. Real-world proposals (from actual seed pipeline)
# ===================================================================

class TestRealWorldProposals:
    """Test against real proposals that the pipeline has seen."""

    SHOULD_PASS = [
        "Build seed_gate.py specificity validator",
        "Implement thermal_model.py for Mars habitat simulation",
        "Create solar_concentrator.py parabolic reflector module",
        "Deploy the agent-exchange SDK to npm registry",
        "Refactor process_inbox.py to use action dispatcher pattern",
        "Write integration tests for state_io.py atomic writes",
        "Debug the compute_trending.py scoring algorithm",
        "Ship mass_driver.py electromagnetic launch rail",
        "Design reactor_core.py fission power system",
        'Build a "Mars weather station" monitoring tool',
        "Implement feedback from #12503 on seed validation",
        "Add --verbose flag to the seed_gate CLI",
        "Integrate r/engineering channel feed generator",
    ]

    SHOULD_FAIL = [
        "",
        "   ",
        "something cool",
        "The platform is really interesting",
        "1. Build this\n2. Build that",
        "https://github.com/kody-w/rappterbook",
        "TODO: finish the validator later",
        "`some broken fragment`",
        "maybe we should think about things",
        "Build something amazing for the colony",
    ]

    @pytest.mark.parametrize("text", SHOULD_PASS)
    def test_passes(self, text: str):
        assert _v(text)["passed"] is True, f"Expected pass: {text!r}"

    @pytest.mark.parametrize("text", SHOULD_FAIL)
    def test_fails(self, text: str):
        assert _v(text)["passed"] is False, f"Expected fail: {text!r}"


# ===================================================================
# 14. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_unicode_text(self):
        r = _v("Build thé_mödel.py validateur spécial")
        assert isinstance(r["passed"], bool)

    def test_very_long_text(self):
        text = "Build seed_gate.py " + "word " * 5000
        r = _v(text)
        assert r["passed"] is True

    def test_newlines_in_text(self):
        r = _v("Build seed_gate.py\nvalidator\nmodule")
        assert r["passed"] is True

    def test_tabs_in_text(self):
        r = _v("Build\tseed_gate.py\tvalidator")
        assert r["passed"] is True

    def test_mixed_case_tags(self):
        r = _v("Build philosophical framework", ["Theme", "PHILOSOPHY"])
        assert r["passed"] is True

    def test_none_tags(self):
        r = _v("Build seed_gate.py validator", None)
        assert r["passed"] is True

    def test_empty_tags(self):
        r = _v("Build seed_gate.py validator", [])
        assert r["passed"] is True

    def test_run_prefix_not_junk(self):
        r = _v("run_simulation for thermal_model.py targets")
        assert r["code"] != "junk"

    def test_file_with_path(self):
        r = _v("Build scripts/actions/seed.py handler")
        assert r["passed"] is True
        assert "seed.py" in r["target"]

    def test_multiple_file_targets(self):
        r = _v("Wire seed_gate.py into propose_seed.py pipeline")
        assert r["passed"] is True

    def test_parentheses_start_is_junk(self):
        r = _v("(continued) Build the validator module")
        assert r["code"] == "junk"

    def test_hyphen_start_is_junk(self):
        r = _v("- Build the validator.py module")
        assert r["code"] == "junk"


# ===================================================================
# 15. Property-based invariants
# ===================================================================

class TestInvariants:
    """Structural invariants that must hold for ALL inputs."""

    SAMPLES = [
        "Build seed_gate.py",
        "",
        "x",
        "Build something cool",
        "The module is nice",
        "Build seed_gate.py validator" * 100,
        "run_test for my_module.py quickly",
        "Design philosophical framework",
        "https://example.com/path",
        "1. numbered item",
        "\n\n\n",
    ]

    @pytest.mark.parametrize("text", SAMPLES)
    def test_dict_keys_always_present(self, text: str):
        r = _v(text)
        for key in ("passed", "reasons", "score", "verb", "target", "code"):
            assert key in r, f"Missing key {key!r} for input {text!r}"

    @pytest.mark.parametrize("text", SAMPLES)
    def test_score_in_range(self, text: str):
        r = _v(text)
        assert 0 <= r["score"] <= 10

    @pytest.mark.parametrize("text", SAMPLES)
    def test_passed_is_bool(self, text: str):
        r = _v(text)
        assert isinstance(r["passed"], bool)

    @pytest.mark.parametrize("text", SAMPLES)
    def test_reasons_is_list_of_strings(self, text: str):
        r = _v(text)
        assert isinstance(r["reasons"], list)
        for item in r["reasons"]:
            assert isinstance(item, str)

    @pytest.mark.parametrize("text", SAMPLES)
    def test_code_is_string(self, text: str):
        r = _v(text)
        assert isinstance(r["code"], str)
        assert r["code"] in ("pass", "no_verb", "no_target", "junk")

    @pytest.mark.parametrize("text", SAMPLES)
    def test_dict_equals_dataclass_to_dict(self, text: str):
        dict_r = _v(text)
        dc_r = _vs(text)
        assert dc_r.to_dict() == dict_r

    @pytest.mark.parametrize("text", SAMPLES)
    def test_passes_gate_matches_dict(self, text: str):
        dict_r = _v(text)
        assert passes_gate(text) == dict_r["passed"]


# ===================================================================
# 16. CLI entry-point
# ===================================================================

class TestCLI:
    def test_cli_pass_exit_0(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Build seed_gate.py validator"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["passed"] is True

    def test_cli_fail_exit_1(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             ""],
            capture_output=True, text=True,
        )
        assert result.returncode == 1

    def test_cli_with_tags(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Build a philosophical framework", "theme"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["passed"] is True

    def test_cli_no_args_exit_1(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py")],
            capture_output=True, text=True,
        )
        assert result.returncode == 1


# ===================================================================
# 17. Smoke tests
# ===================================================================

class TestSmoke:
    def test_import_succeeds(self):
        import seed_gate  # noqa: F401

    def test_validate_callable(self):
        assert callable(validate)

    def test_validate_seed_callable(self):
        assert callable(validate_seed)

    def test_passes_gate_callable(self):
        assert callable(passes_gate)

    def test_hundred_random_proposals(self):
        """Run 100 varied proposals -- none should crash."""
        proposals = [
            f"{verb.capitalize()} the widget.py module"
            for verb in sorted(ACTION_VERBS)
        ] + [
            "Something without a verb at all",
            "",
            "x" * 10000,
            "Build " + "nested/" * 50 + "deep.py",
        ]
        for p in proposals[:100]:
            r = _v(p)
            assert isinstance(r["passed"], bool)
            assert isinstance(r["reasons"], list)
            assert 0 <= r["score"] <= 10


# ===================================================================
# 18. Mode consistency
# ===================================================================

class TestModeConsistency:
    def test_admission_default(self):
        r1 = _v("Build seed_gate.py validator")
        r2 = _v("Build seed_gate.py validator", mode="admission")
        assert r1 == r2

    def test_purge_and_admission_same_for_clean(self):
        """Clean short proposals should pass in both modes."""
        text = "Build seed_gate.py validator"
        assert _v(text, mode="admission")["passed"] is True
        assert _v(text, mode="purge")["passed"] is True

    def test_modes_differ_for_late_verb(self):
        """Purge mode misses verb after char 200; admission finds it."""
        padding = "X " * 110
        text = f"{padding} Build seed_gate.py"
        adm = _v(text, mode="admission")
        pur = _v(text, mode="purge")
        assert adm["verb"] == "build"
        assert pur["verb"] == ""
