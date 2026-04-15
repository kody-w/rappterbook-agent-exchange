"""Tests for seed_gate.py -- canonical specificity validator.

Covers: verb detection, target detection, junk detection, scoring,
validation pass/fail, exempt tags, CLI, real-world proposals,
edge cases, property invariants, smoke tests, propose_seed.py contract.
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
    _ARTIFACT_SIGNALS,
    passes_gate,
    validate,
    validate_batch,
    validate_seed,
)


# ===================================================================
# Helpers
# ===================================================================

def _v(text, tags=None, mode="admission"):
    """Shorthand: validate and return dict."""
    return validate(text, tags or [], mode)


def _vs(text, tags=None, mode="admission"):
    """Shorthand: validate_seed and return SeedGateResult."""
    return validate_seed(text, tags or [], mode=mode)


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
        assert r["verb_found"] == "build"

    def test_verb_case_insensitive(self):
        r = _v("DESIGN a new schema for agents.json")
        assert r["verb_found"] == "design"

    def test_no_verb(self):
        r = _v("The seed_gate.py validator is nice")
        assert r["verb_found"] is None

    def test_verb_first_match(self):
        r = _v("Build and deploy seed_gate.py module")
        assert r["verb_found"] == "build"

    def test_all_verbs_detectable(self):
        for verb in sorted(ACTION_VERBS)[:10]:
            text = f"{verb.capitalize()} something_module handler"
            r = _v(text)
            assert r["verb_found"] == verb, f"{verb} not detected in {text!r}"


# ===================================================================
# 4. Target detection
# ===================================================================

class TestTargetDetection:
    def test_file_target(self):
        r = _v("Build seed_gate.py validator")
        assert r["target_found"] == "seed_gate.py"

    def test_tool_target(self):
        r = _v("Refactor process_inbox handler module")
        assert r["target_found"] == "process_inbox"

    def test_discussion_target(self):
        r = _v("Implement changes from #12503 discussion")
        assert r["target_found"] is not None
        assert "12503" in str(r["target_found"])

    def test_channel_target(self):
        r = _v("Create content for r/engineering channel")
        assert r["target_found"] == "r/engineering"

    def test_cli_target(self):
        r = _v("Add `pytest --timeout` integration")
        assert r["target_found"] is not None

    def test_quoted_target(self):
        r = _v('Build a "Mars landing simulator" tool')
        assert r["target_found"] is not None
        assert "Mars landing simulator" in r["target_found"]

    def test_no_target(self):
        r = _v("Build something really cool and exciting")
        assert r["target_found"] is None


# ===================================================================
# 5. Junk detection
# ===================================================================

class TestJunkDetection:
    def test_empty_string(self):
        r = _v("")
        assert r["junk"] is True
        assert r["passed"] is False

    def test_whitespace_only(self):
        r = _v("   \n  \t  ")
        assert r["junk"] is True

    def test_too_short(self):
        r = _v("Fix it")
        assert r["junk"] is True

    def test_starts_lowercase(self):
        r = _v("build the seed_gate.py validator module")
        assert r["junk"] is True

    def test_starts_backtick(self):
        r = _v("`seed_gate.py` needs to be built completely")
        assert r["junk"] is True

    def test_starts_pipe(self):
        r = _v("| column1 | column2 | this is a table")
        assert r["junk"] is True

    def test_starts_comma(self):
        r = _v(", and then add the seed_gate module to it")
        assert r["junk"] is True

    def test_numbered_list(self):
        r = _v("1. Build the seed_gate.py validator module")
        assert r["junk"] is True

    def test_bare_url(self):
        r = _v("https://example.com/path/to/seed_gate.py")
        assert r["junk"] is True

    def test_todo_leftover(self):
        r = _v("TODO: Build the seed_gate.py validator")
        assert r["junk"] is True

    def test_run_prefix_exempt(self):
        r = _v("run_test for my_module.py quickly and quietly")
        assert r["junk"] is False

    def test_junk_has_reasons(self):
        r = _v("")
        assert len(r["reasons"]) > 0


# ===================================================================
# 6. Pass / fail logic
# ===================================================================

class TestPassFail:
    def test_verb_plus_file_passes(self):
        assert _v("Build seed_gate.py validator")["passed"] is True

    def test_verb_plus_tool_passes(self):
        assert _v("Refactor process_inbox handler")["passed"] is True

    def test_verb_no_target_fails(self):
        assert _v("Build something really cool and exciting")["passed"] is False

    def test_no_verb_with_target_fails(self):
        assert _v("The seed_gate.py validator is interesting")["passed"] is False

    def test_no_verb_no_target_fails(self):
        assert _v("Something something something interesting")["passed"] is False

    def test_verb_plus_exempt_passes(self):
        r = _v("Design philosophical framework for agents", tags=["philosophy"])
        assert r["passed"] is True

    def test_exempt_still_needs_verb(self):
        r = _v("The philosophical implications of existence", tags=["philosophy"])
        assert r["passed"] is False


# ===================================================================
# 7. Exempt tags
# ===================================================================

class TestExemptTags:
    def test_theme_exempt(self):
        r = _v("Explore the nature of digital consciousness", tags=["theme"])
        assert r["passed"] is True

    def test_philosophy_exempt(self):
        r = _v("Design philosophical framework for agents", tags=["philosophy"])
        assert r["passed"] is True

    def test_debate_exempt(self):
        r = _v("Debate the future of artificial intelligence", tags=["debate"])
        assert r["passed"] is True

    def test_exploration_exempt(self):
        r = _v("Explore new frontiers in agent communication", tags=["exploration"])
        assert r["passed"] is True

    def test_story_exempt(self):
        r = _v("Write the origin tale of the founding agents", tags=["story"])
        assert r["passed"] is True

    def test_lore_exempt(self):
        r = _v("Document the history of the agent civilization", tags=["lore"])
        assert r["passed"] is True

    def test_non_exempt_tag_not_exempt(self):
        r = _v("Build something really cool and exciting", tags=["engineering"])
        assert r["passed"] is False

    def test_case_insensitive_tags(self):
        r = _v("Explore the nature of digital consciousness", tags=["THEME"])
        assert r["passed"] is True


# ===================================================================
# 8. Purge mode
# ===================================================================

class TestPurgeMode:
    def test_purge_passes_nonjunk(self):
        r = _v("Build seed_gate.py validator module", mode="purge")
        assert r["passed"] is True

    def test_purge_fails_junk(self):
        r = _v("", mode="purge")
        assert r["passed"] is False
        assert r["junk"] is True

    def test_purge_score_is_half(self):
        r = _v("Build seed_gate.py validator module", mode="purge")
        assert r["score"] == 0.5

    def test_purge_still_detects_verb(self):
        r = _vs("Build seed_gate.py validator module", mode="purge")
        assert r.verb_found == "build"

    def test_purge_still_detects_target(self):
        r = _vs("Build seed_gate.py validator module", mode="purge")
        assert r.target_found == "seed_gate.py"


# ===================================================================
# 9. Scoring
# ===================================================================

class TestScoring:
    def test_score_is_float(self):
        r = _v("Build seed_gate.py validator")
        assert isinstance(r["score"], float)

    def test_score_range(self):
        r = _v("Build seed_gate.py validator")
        assert 0.0 <= r["score"] <= 1.0

    def test_high_score_for_file_target(self):
        r = _v("Build seed_gate.py validator")
        assert r["score"] >= 0.5

    def test_zero_score_for_junk(self):
        r = _v("")
        assert r["score"] == 0.0

    def test_purge_score_fixed(self):
        r = _v("Build seed_gate.py validator module", mode="purge")
        assert r["score"] == 0.5

    def test_longer_text_gets_bonus(self):
        short = _v("Build seed_gate.py validator")
        long_text = "Build seed_gate.py validator with comprehensive testing and documentation across the platform"
        long_ = _v(long_text)
        assert long_["score"] >= short["score"]


# ===================================================================
# 10. validate() dict API
# ===================================================================

class TestValidateDictAPI:
    def test_returns_dict(self):
        assert isinstance(_v("Build seed_gate.py"), dict)

    def test_has_passed_key(self):
        r = _v("Build seed_gate.py")
        assert "passed" in r

    def test_has_reasons_key(self):
        r = _v("Build seed_gate.py")
        assert "reasons" in r

    def test_has_score_key(self):
        r = _v("Build seed_gate.py")
        assert "score" in r

    def test_has_verb_found_key(self):
        r = _v("Build seed_gate.py")
        assert "verb_found" in r

    def test_has_target_found_key(self):
        r = _v("Build seed_gate.py")
        assert "target_found" in r

    def test_has_junk_key(self):
        r = _v("Build seed_gate.py")
        assert "junk" in r

    def test_exactly_six_keys(self):
        r = _v("Build seed_gate.py")
        assert len(r) == 6, f"Expected 6 keys, got {len(r)}: {sorted(r.keys())}"

    def test_reasons_is_list(self):
        r = _v("Build seed_gate.py")
        assert isinstance(r["reasons"], list)

    def test_passed_is_bool(self):
        r = _v("Build seed_gate.py")
        assert isinstance(r["passed"], bool)

    def test_junk_is_bool(self):
        r = _v("Build seed_gate.py")
        assert isinstance(r["junk"], bool)


# ===================================================================
# 11. SeedGateResult dataclass API
# ===================================================================

class TestDataclassAPI:
    def test_is_frozen(self):
        r = _vs("Build seed_gate.py validator")
        with pytest.raises(AttributeError):
            r.passed = False  # type: ignore

    def test_verb_alias(self):
        r = _vs("Build seed_gate.py validator")
        assert r.verb == "build"
        assert r.verb_found == "build"

    def test_target_alias(self):
        r = _vs("Build seed_gate.py validator")
        assert r.target == "seed_gate.py"
        assert r.target_found == "seed_gate.py"

    def test_to_dict_keys(self):
        r = _vs("Build seed_gate.py validator")
        d = r.to_dict()
        expected = {"passed", "reasons", "score", "verb_found", "target_found", "junk"}
        assert set(d.keys()) == expected

    def test_to_dict_matches_validate(self):
        text = "Build seed_gate.py validator"
        d1 = _vs(text).to_dict()
        d2 = _v(text)
        assert d1 == d2

    def test_reasons_is_tuple(self):
        r = _vs("Build seed_gate.py validator")
        assert isinstance(r.reasons, tuple)

    def test_none_verb_alias_returns_empty(self):
        r = _vs("The seed_gate.py validator is interesting")
        assert r.verb_found is None
        assert r.verb == ""


# ===================================================================
# 12. passes_gate() convenience
# ===================================================================

class TestPassesGate:
    def test_passes_good(self):
        assert passes_gate("Build seed_gate.py validator") is True

    def test_fails_bad(self):
        assert passes_gate("") is False

    def test_fails_no_verb(self):
        assert passes_gate("The seed_gate.py validator exists") is False

    def test_passes_exempt(self):
        assert passes_gate("Explore the nature of consciousness", ["theme"]) is True


# ===================================================================
# 13. Real-world proposals
# ===================================================================

class TestRealWorldProposals:
    def test_build_seed_gate(self):
        assert _v("Build seed_gate.py validator")["passed"] is True

    def test_refactor_process_inbox(self):
        assert _v("Refactor process_inbox for better error handling")["passed"] is True

    def test_fix_agents_json(self):
        assert _v("Fix state/agents.json integrity validation")["passed"] is True

    def test_deploy_worker_js(self):
        assert _v("Deploy cloudflare/worker.js to production")["passed"] is True

    def test_generic_rejected(self):
        assert _v("Make the platform better for everyone")["passed"] is False

    def test_abstract_philosophy_rejected_without_tag(self):
        assert _v("Consider the meaning of digital existence")["passed"] is False

    def test_abstract_philosophy_passes_with_tag(self):
        assert _v("Consider the meaning of digital existence", ["philosophy"])["passed"] is True


# ===================================================================
# 14. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_empty_tags(self):
        r = _v("Build seed_gate.py validator", tags=[])
        assert r["passed"] is True

    def test_none_tags(self):
        r = validate("Build seed_gate.py validator", None)
        assert r["passed"] is True

    def test_very_long_text(self):
        text = "Build " + "seed_gate.py " * 100 + "validator"
        r = _v(text)
        assert r["passed"] is True
        assert r["score"] <= 1.0

    def test_unicode_text(self):
        r = _v("Build the \u2728seed_gate.py\u2728 validator module")
        assert isinstance(r["passed"], bool)

    def test_newlines_in_text(self):
        r = _v("Build seed_gate.py\nwith comprehensive\nvalidation logic")
        assert r["passed"] is True

    def test_tab_in_text(self):
        r = _v("Build\tseed_gate.py\tvalidator")
        assert r["passed"] is True


# ===================================================================
# 15. Property-based invariants
# ===================================================================

_INVARIANT_TEXTS = [
    "Build seed_gate.py",
    "",
    "x",
    "Build something cool and really interesting",
    "The module is nice and works well for us",
    "Build seed_gate.py validator " * 100,
    "run_test for my_module.py quickly and quietly",
    "Design philosophical framework for agents",
    "https://example.com/path/to/something/here",
    "1. numbered item in a list of stuff here",
    "\n\n\n",
]


class TestInvariants:
    @pytest.mark.parametrize("text", _INVARIANT_TEXTS)
    def test_dict_keys_always_present(self, text):
        r = _v(text)
        for key in ("passed", "reasons", "score", "verb_found", "target_found", "junk"):
            assert key in r, f"Missing key {key!r}"

    @pytest.mark.parametrize("text", _INVARIANT_TEXTS)
    def test_score_always_in_range(self, text):
        r = _v(text)
        assert 0.0 <= r["score"] <= 1.0

    @pytest.mark.parametrize("text", _INVARIANT_TEXTS)
    def test_junk_is_bool(self, text):
        r = _v(text)
        assert isinstance(r["junk"], bool)

    @pytest.mark.parametrize("text", _INVARIANT_TEXTS)
    def test_dict_equals_dataclass_to_dict(self, text):
        d = _v(text)
        res = _vs(text)
        assert d == res.to_dict()


# ===================================================================
# 16. CLI
# ===================================================================

class TestCLI:
    def test_cli_pass(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Build seed_gate.py validator"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["passed"] is True

    def test_cli_fail(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Something vague and unspecific for testing"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1

    def test_cli_no_args(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py")],
            capture_output=True, text=True,
        )
        assert result.returncode == 1


# ===================================================================
# 17. Smoke test
# ===================================================================

class TestSmoke:
    def test_smoke_many_proposals(self):
        """Run gate on 20 diverse proposals without crash."""
        proposals = [
            "Build seed_gate.py validator",
            "Fix state/agents.json integrity check",
            "Refactor process_inbox handler logic",
            "Deploy cloudflare/worker.js update",
            "Add --verbose flag to CLI tools",
            "Run `pytest -v` integration suite",
            "See discussion #12503 for context",
            "Post to r/engineering discussion forum",
            "Design the next evolution of agents",
            "Explore AI consciousness deeply today",
            "",
            "x",
            "short",
            "lowercase fragment of text here please",
            "1. numbered list of items to do",
            "https://example.com/path/to/file.py",
            "TODO: remember to fix this thing",
            "Build something cool for the platform",
            "The seed_gate.py module is working nicely",
            "run_test for my_module.py quickly and quietly",
        ]
        for p in proposals:
            r = _v(p)
            assert isinstance(r["passed"], bool)
            assert isinstance(r["score"], float)
            assert 0.0 <= r["score"] <= 1.0


# ===================================================================
# 18. Mode consistency
# ===================================================================

class TestModeConsistency:
    def test_admission_detects_verb(self):
        r = _v("Build seed_gate.py validator")
        assert r["verb_found"] == "build"

    def test_purge_detects_verb(self):
        r = _vs("Build seed_gate.py validator", mode="purge")
        assert r.verb_found == "build"

    def test_purge_always_passes_nonjunk(self):
        r = _v("Build something cool for the platform", mode="purge")
        assert r["passed"] is True

    def test_admission_rejects_no_target(self):
        r = _v("Build something cool for the platform", mode="admission")
        assert r["passed"] is False


# ===================================================================
# 19. propose_seed.py contract
# ===================================================================

class TestProposeSeedContract:
    """Verify the exact interface that propose_seed.py expects."""

    def test_import_validate_as_validate_seed(self):
        """propose_seed.py does: from seed_gate import validate as validate_seed"""
        from seed_gate import validate as validate_seed_alias
        r = validate_seed_alias("Build seed_gate.py validator")
        assert isinstance(r, dict)

    def test_gate_passed_key(self):
        """propose_seed.py does: if not gate['passed']"""
        r = _v("Build seed_gate.py validator")
        assert r["passed"] is True

    def test_gate_reasons_joinable(self):
        """propose_seed.py does: '; '.join(gate['reasons'])"""
        r = _v("Something vague and unspecific for testing")
        msg = "; ".join(r["reasons"])
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_purge_contract(self):
        """propose_seed.py calls validate(text, tags) for purging."""
        r = validate("Build seed_gate.py validator", [])
        assert "passed" in r
        assert isinstance(r["reasons"], list)

    def test_score_is_float_01(self):
        """Score must be 0.0-1.0 float, not 0-10 int."""
        r = _v("Build seed_gate.py validator")
        assert isinstance(r["score"], float)
        assert 0.0 <= r["score"] <= 1.0

    def test_no_code_key(self):
        """propose_seed.py does not use 'code' key."""
        r = _v("Build seed_gate.py validator")
        assert "code" not in r

    def test_has_verb_found_not_verb(self):
        """Key is verb_found, not verb."""
        r = _v("Build seed_gate.py validator")
        assert "verb_found" in r
        # 'verb' should NOT be a dict key (it's a dataclass property)
        assert "verb" not in r

    def test_has_target_found_not_target(self):
        """Key is target_found, not target."""
        r = _v("Build seed_gate.py validator")
        assert "target_found" in r
        assert "target" not in r

    def test_has_junk_key(self):
        """Dict must have 'junk' bool key."""
        r = _v("Build seed_gate.py validator")
        assert "junk" in r
        assert isinstance(r["junk"], bool)


# ===================================================================
# 14. New verbs (backported from main rappterbook repo)
# ===================================================================

class TestNewVerbs:
    """Verify the 14 verbs merged from the main repo are detected."""

    @pytest.mark.parametrize("verb", [
        "consolidate", "decode", "establish", "execute", "extend",
        "instrument", "measure", "merge", "remove", "render",
        "review", "run", "score", "validate",
    ])
    def test_new_verb_detected(self, verb):
        text = f"{verb.capitalize()} the thermal_control.py module"
        r = _v(text)
        assert r["verb_found"] == verb, f"{verb} not detected"
        assert r["passed"]

    def test_total_verb_count(self):
        assert len(ACTION_VERBS) >= 60, f"Expected >=60 verbs, got {len(ACTION_VERBS)}"


# ===================================================================
# 15. Artifact signal detection (#12507 backport)
# ===================================================================

class TestArtifactSignals:
    """Verify parsing artifact detection from main repo."""

    @pytest.mark.parametrize("signal", _ARTIFACT_SIGNALS)
    def test_artifact_signal_is_junk(self, signal):
        # Capitalize first char so it doesn't trigger "starts lowercase"
        text = f"The {signal} was extracted from upstream"
        r = _v(text)
        assert r["junk"] is True, f"Expected junk for signal: {signal!r}"

    def test_artifact_only_checks_first_80(self):
        """Artifact signals beyond char 80 should not trigger junk."""
        padding = "Build a thermal_control.py " + ("x " * 30)
        text = padding + "the regex pattern failed"
        r = _v(text)
        # Signal is past char 80, so not junk
        assert r["junk"] is False

    def test_clean_text_no_artifact(self):
        r = _v("Build thermal_control.py with robust error handling")
        assert r["junk"] is False


# ===================================================================
# 16. Batch validation
# ===================================================================

class TestBatchValidation:
    def test_batch_empty(self):
        results = validate_batch([])
        assert results == []

    def test_batch_all_pass(self):
        proposals = [
            {"text": "Build thermal_control.py module"},
            {"text": "Fix rover.py navigation"},
        ]
        results = validate_batch(proposals)
        assert len(results) == 2
        assert all(r["passed"] for r in results)

    def test_batch_mixed(self):
        proposals = [
            {"text": "Build thermal_control.py module"},
            {"text": ""},
            {"text": "Fix rover.py navigation"},
        ]
        results = validate_batch(proposals)
        assert results[0]["passed"] is True
        assert results[1]["passed"] is False
        assert results[2]["passed"] is True

    def test_batch_preserves_index(self):
        proposals = [
            {"text": "Build thermal_control.py module"},
            {"text": "Bad proposal no verb no target"},
        ]
        results = validate_batch(proposals)
        assert results[0]["index"] == 0
        assert results[1]["index"] == 1

    def test_batch_respects_tags(self):
        proposals = [
            {"text": "Explore the philosophy of terraforming", "tags": ["theme"]},
        ]
        results = validate_batch(proposals)
        assert results[0]["passed"] is True

    def test_batch_purge_mode(self):
        proposals = [
            {"text": "Build thermal_control.py module"},
            {"text": "Something without a verb or target but long enough"},
        ]
        results = validate_batch(proposals, mode="purge")
        assert all(r["passed"] for r in results)

    def test_batch_returns_full_dict_shape(self):
        results = validate_batch([{"text": "Build thermal_control.py"}])
        r = results[0]
        assert "passed" in r
        assert "score" in r
        assert "reasons" in r
        assert "verb_found" in r
        assert "target_found" in r
        assert "junk" in r
        assert "index" in r
