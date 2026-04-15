"""Tests for seed_gate.py -- canonical specificity validator.

Covers: verb detection, target detection, junk detection, scoring,
validation pass/fail, exempt tags, CLI, real-world proposals,
edge cases, property invariants, smoke tests, propose_seed.py contract,
artifact signal detection, path/func targets, public helpers,
FILE_RE false positive filter, batch validation API, stress/fuzz inputs.
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
    KNOWN_MODULES,
    MODULE_CONTEXT_RE,
    PATH_RE,
    QUOTED_RE,
    QUESTION_STEMS,
    TOOL_RE,
    BatchResult,
    BatchStats,
    SeedGateResult,
    _FALSE_FILE_MATCHES,
    _HARD_ARTIFACT_SIGNALS,
    _SOFT_ARTIFACT_SIGNALS,
    _detect_target,
    _detect_verb,
    _is_junk,
    _score,
    find_question_intent,
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

    def test_false_file_matches_populated(self):
        assert len(_FALSE_FILE_MATCHES) >= 7

    def test_false_file_matches_all_lowercase(self):
        for m in _FALSE_FILE_MATCHES:
            assert m == m.lower(), f"False match {m!r} not lowercase"


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

    def test_discussion_re_ignores_short(self):
        assert not DISCUSSION_RE.search("Issue #42 is short")

    def test_channel_re_matches(self):
        assert CHANNEL_RE.search("Post in r/engineering channel")

    def test_path_re_matches(self):
        assert PATH_RE.search("Check state/agents for issues")

    def test_quoted_re_matches(self):
        assert QUOTED_RE.search('Build a "Mars landing simulator"')


# ===================================================================
# 3. Verb detection
# ===================================================================

class TestVerbDetection:
    def test_basic_verb(self):
        assert _detect_verb("Build seed_gate.py") == "build"

    def test_case_insensitive(self):
        assert _detect_verb("DEPLOY the worker now") == "deploy"

    def test_no_verb(self):
        assert _detect_verb("The module works fine") == ""

    def test_first_verb_wins(self):
        assert _detect_verb("Build and test seed_gate.py") == "build"

    def test_limit_parameter(self):
        text = "Something " * 50 + "Build seed_gate.py"
        assert _detect_verb(text, limit=20) == ""
        assert _detect_verb(text) == "build"


# ===================================================================
# 4. Target detection
# ===================================================================

class TestTargetDetection:
    def test_file_target(self):
        t, k = _detect_target("Build seed_gate.py")
        assert t == "seed_gate.py"
        assert k == "file"

    def test_path_target(self):
        t, k = _detect_target("Audit state/agents for issues")
        assert "state/agents" in t
        assert k == "path"

    def test_tool_target(self):
        t, k = _detect_target("Refactor process_inbox handler")
        assert t == "process_inbox"
        assert k == "tool"

    def test_cli_target(self):
        t, k = _detect_target("Add --verbose flag support here")
        assert "--verbose" in t
        assert k == "cli"

    def test_discussion_target(self):
        t, k = _detect_target("See discussion #12503 for context")
        assert "12503" in t
        assert k == "discussion"

    def test_channel_target(self):
        t, k = _detect_target("Post to r/engineering channel now")
        assert t == "r/engineering"
        assert k == "channel"

    def test_quoted_target(self):
        t, k = _detect_target('Build a "Mars landing simulator"')
        assert "Mars landing simulator" in t
        assert k == "quoted"

    def test_no_target(self):
        t, k = _detect_target("Something vague and unspecific")
        assert t == ""
        assert k == ""


# ===================================================================
# 5. FILE_RE false positive filter
# ===================================================================

class TestFileReFalsePositives:
    """FILE_RE should not match common abbreviations as filenames."""

    def test_eg_not_matched(self):
        t, k = _detect_target("Build docs, e.g. examples for developers")
        assert t != "e.g"

    def test_ie_not_matched(self):
        t, k = _detect_target("Fix the bug, i.e. the validation error")
        assert t != "i.e"

    def test_etc_not_matched(self):
        t, k = _detect_target("Add modules, etc. to the project tree")
        assert t != "etc"

    def test_vs_not_matched(self):
        t, k = _detect_target("Compare admission vs. purge mode scores")
        assert t != "vs"

    def test_cf_not_matched(self):
        t, k = _detect_target("Review approach, cf. the older design")
        assert t != "cf"

    def test_real_py_still_matched(self):
        t, k = _detect_target("Build seed_gate.py validator")
        assert t == "seed_gate.py"
        assert k == "file"

    def test_real_json_still_matched(self):
        t, k = _detect_target("Fix state/agents.json integrity")
        assert "agents.json" in t
        assert k == "file"

    def test_real_rs_still_matched(self):
        t, k = _detect_target("Port the parser to grammar.rs now")
        assert t == "grammar.rs"
        assert k == "file"

    def test_real_html_still_matched(self):
        t, k = _detect_target("Deploy docs/index.html to Pages")
        assert "index.html" in t
        assert k == "file"

    def test_eg_proposal_doesnt_falsely_pass(self):
        r = _v("Build modules, e.g. examples for developers")
        if r["target_found"]:
            assert r["target_found"] != "e.g"

    def test_abbreviation_with_real_file(self):
        r = _v("Fix seed_gate.py, i.e. the validation module")
        assert r["passed"] is True
        assert r["target_found"] == "seed_gate.py"

    def test_multiple_abbreviations_no_false_target(self):
        r = _v("Build tools, e.g. linters, i.e. code quality checks")
        if r["target_found"]:
            assert r["target_found"] not in ("e.g", "i.e")


# ===================================================================
# 6. Junk detection
# ===================================================================

class TestJunkDetection:
    def test_empty(self):
        assert _is_junk("") != ""

    def test_whitespace_only(self):
        assert _is_junk("   ") != ""

    def test_too_short(self):
        assert _is_junk("Fix it") != ""

    def test_lowercase_fragment(self):
        assert _is_junk("the module needs work and more updates") != ""

    def test_numbered_list(self):
        assert _is_junk("1. first item in the list here") != ""

    def test_bare_url(self):
        assert _is_junk("https://example.com/path/to/page") != ""

    def test_todo_comment(self):
        assert _is_junk("TODO: fix the validation module here") != ""

    def test_run_prefix_exception(self):
        assert _is_junk("run_test for module quickly and reliably") == ""

    def test_file_start_exception(self):
        assert _is_junk("greenhouse.py needs fixes for pressure") == ""

    def test_clean_text(self):
        assert _is_junk("Build seed_gate.py with validation logic") == ""

    def test_hard_artifact(self):
        reason = _is_junk("Parser grabbed too much text from output")
        assert reason != ""
        assert "artifact" in reason

    def test_backtick_junk(self):
        assert _is_junk("`some code fragment` and more stuff") != ""


# ===================================================================
# 7. Scoring
# ===================================================================

class TestScoring:
    def test_file_target_scores_high(self):
        s = _score("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert 0.5 <= s <= 1.0

    def test_no_target_scores_low(self):
        s = _score("Build something cool", "build", "", "")
        assert 0.0 < s <= 0.5

    def test_no_verb_still_scores(self):
        s = _score("The seed_gate.py thing", "", "seed_gate.py", "file")
        assert 0.0 < s <= 1.0

    def test_nothing_scores_zero(self):
        s = _score("Something vague", "", "", "")
        assert s == 0.0

    def test_score_capped_at_one(self):
        text = "Build seed_gate.py and process_inbox.py and state_io.py " * 5
        s = _score(text, "build", "seed_gate.py", "file")
        assert s <= 1.0

    def test_longer_text_scores_higher(self):
        short = _score("Build seed_gate.py", "build", "seed_gate.py", "file")
        long = _score(
            "Build seed_gate.py with comprehensive validation, scoring, "
            "and batch processing capabilities for the proposal pipeline",
            "build", "seed_gate.py", "file",
        )
        assert long >= short


# ===================================================================
# 8. Pass/fail decisions
# ===================================================================

class TestPassFail:
    def test_verb_plus_file_passes(self):
        assert _v("Build seed_gate.py validator")["passed"] is True

    def test_verb_plus_tool_passes(self):
        assert _v("Refactor process_inbox handler now")["passed"] is True

    def test_verb_only_fails(self):
        assert _v("Build something cool and exciting")["passed"] is False

    def test_target_only_fails(self):
        r = _v("The seed_gate.py module is interesting")
        assert r["passed"] is False

    def test_empty_fails(self):
        assert _v("")["passed"] is False

    def test_junk_fails_with_junk_flag(self):
        r = _v("")
        assert r["junk"] is True

    def test_failure_has_reasons(self):
        r = _v("Build something cool and exciting")
        assert len(r["reasons"]) > 0


# ===================================================================
# 9. Exempt tags
# ===================================================================

class TestExemptTags:
    def test_theme_exempt(self):
        r = _v("Explore the nature of digital existence", ["theme"])
        assert r["passed"] is True

    def test_philosophy_exempt(self):
        r = _v("Consider the meaning of consciousness", ["philosophy"])
        assert r["passed"] is True

    def test_debate_exempt(self):
        r = _v("Debate whether agents need embodiment", ["debate"])
        assert r["passed"] is True

    def test_story_exempt(self):
        r = _v("Write the chronicle of the founding", ["story"])
        assert r["passed"] is True

    def test_lore_exempt(self):
        r = _v("Explore the deep history of Rappterbook", ["lore"])
        assert r["passed"] is True

    def test_exempt_still_needs_verb(self):
        r = _v("The nature of digital consciousness", ["theme"])
        assert r["passed"] is False

    def test_case_insensitive_tags(self):
        r = _v("Explore consciousness deeply here", ["THEME"])
        assert r["passed"] is True

    def test_non_exempt_tag_ignored(self):
        r = _v("Something vague and unspecific here", ["random"])
        assert r["passed"] is False


# ===================================================================
# 10. Question-stem intent
# ===================================================================

class TestQuestionStem:
    def test_what_if_maps_to_explore(self):
        stem, verb = find_question_intent("What if agents could dream?")
        assert stem == "what if"
        assert verb == "explore"

    def test_how_might_maps_to_design(self):
        stem, verb = find_question_intent("How might we build better tools?")
        assert stem == "how might"
        assert verb == "design"

    def test_no_stem_returns_empty(self):
        stem, verb = find_question_intent("Build seed_gate.py")
        assert stem == ""
        assert verb == ""

    def test_question_stem_with_exempt_tag_passes(self):
        r = _v("What if agents could dream about code?", ["theme"])
        assert r["passed"] is True
        assert r["verb_found"] == "explore"


# ===================================================================
# 11. Purge mode
# ===================================================================

class TestPurgeMode:
    def test_purge_passes_nonjunk(self):
        r = _v("Something vague and unspecific here", mode="purge")
        assert r["passed"] is True

    def test_purge_fails_junk(self):
        r = _v("", mode="purge")
        assert r["passed"] is False
        assert r["junk"] is True

    def test_purge_detects_verb(self):
        r = _vs("Build seed_gate.py validator", mode="purge")
        assert r.verb_found == "build"

    def test_purge_score_fixed(self):
        r = _v("Something vague and unspecific here", mode="purge")
        assert r["score"] == 0.5


# ===================================================================
# 12. Dataclass API
# ===================================================================

class TestDataclass:
    def test_frozen(self):
        r = _vs("Build seed_gate.py validator")
        with pytest.raises(Exception):
            r.passed = False  # type: ignore

    def test_to_dict_shape(self):
        r = _vs("Build seed_gate.py validator")
        d = r.to_dict()
        assert set(d.keys()) == {"passed", "reasons", "score", "verb_found", "target_found", "junk"}

    def test_reasons_is_tuple(self):
        r = _vs("Build seed_gate.py validator")
        assert isinstance(r.reasons, tuple)

    def test_verb_alias(self):
        r = _vs("Build seed_gate.py validator")
        assert r.verb == "build"
        assert r.verb_found == "build"

    def test_none_verb_alias_returns_empty(self):
        r = _vs("The seed_gate.py validator is interesting")
        assert r.verb_found is None
        assert r.verb == ""

    def test_target_alias(self):
        r = _vs("Build seed_gate.py validator")
        assert r.target == "seed_gate.py"


# ===================================================================
# 13. passes_gate() convenience
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
# 14. Real-world proposals
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

    def test_run_verb_works(self):
        assert _v("Run seed_gate.py against the full proposal set")["passed"] is True

    def test_score_verb_works(self):
        r = _v("Score seed_gate.py proposals for specificity")
        # 'score' is not an action verb -- proposal finds target via FILE_RE
        assert isinstance(r["passed"], bool)

    def test_validate_verb_works(self):
        assert _v("Validate seed_gate.py output against the contract")["passed"] is True


# ===================================================================
# 15. Edge cases
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

    def test_mixed_targets(self):
        r = _v("Build seed_gate.py and refactor process_inbox handler")
        assert r["passed"] is True
        assert r["target_found"] == "seed_gate.py"

    def test_path_without_extension(self):
        r = _v("Audit scripts/process_inbox handler for bugs")
        assert r["passed"] is True


# ===================================================================
# 16. Property-based invariants
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
    "Wire validate() into the pipeline for proposals",
    "Audit state/agents for consistency issues here",
    "The regex captured too much of the output",
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

    @pytest.mark.parametrize("text", _INVARIANT_TEXTS)
    def test_passed_implies_nonjunk(self, text):
        r = _v(text)
        if r["passed"]:
            assert r["junk"] is False

    @pytest.mark.parametrize("text", _INVARIANT_TEXTS)
    def test_junk_implies_failed(self, text):
        r = _v(text)
        if r["junk"]:
            assert r["passed"] is False

    @pytest.mark.parametrize("text", _INVARIANT_TEXTS)
    def test_failed_has_reasons(self, text):
        r = _v(text)
        if not r["passed"]:
            assert len(r["reasons"]) > 0


# ===================================================================
# 17. CLI
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

    def test_cli_with_tags(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Explore the nature of digital consciousness",
             "theme"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["passed"] is True


# ===================================================================
# 18. Smoke test
# ===================================================================

class TestSmoke:
    def test_smoke_many_proposals(self):
        """Run gate on 25 diverse proposals without crash."""
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
            "Wire validate() into the seed pipeline",
            "Audit state/agents for consistency issues",
            "Run seed_gate.py against the proposal set",
            "The regex captured something incorrectly",
            "Score the proposals with compute_score()",
        ]
        for p in proposals:
            r = _v(p)
            assert isinstance(r["passed"], bool)
            assert isinstance(r["score"], float)
            assert 0.0 <= r["score"] <= 1.0


# ===================================================================
# 19. Mode consistency
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
# 20. propose_seed.py contract
# ===================================================================

class TestProposeSeedContract:
    def test_import_validate_as_validate_seed(self):
        from seed_gate import validate as validate_seed_alias
        r = validate_seed_alias("Build seed_gate.py validator")
        assert isinstance(r, dict)

    def test_gate_passed_key(self):
        r = _v("Build seed_gate.py validator")
        assert r["passed"] is True

    def test_gate_reasons_joinable(self):
        r = _v("Something vague and unspecific for testing")
        msg = "; ".join(r["reasons"])
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_purge_contract(self):
        r = validate("Build seed_gate.py validator", [])
        assert "passed" in r
        assert isinstance(r["reasons"], list)

    def test_score_is_float_01(self):
        r = _v("Build seed_gate.py validator")
        assert isinstance(r["score"], float)
        assert 0.0 <= r["score"] <= 1.0

    def test_no_code_key(self):
        r = _v("Build seed_gate.py validator")
        assert "code" not in r

    def test_has_verb_found_not_verb(self):
        r = _v("Build seed_gate.py validator")
        assert "verb_found" in r
        assert "verb" not in r

    def test_has_target_found_not_target(self):
        r = _v("Build seed_gate.py validator")
        assert "target_found" in r
        assert "target" not in r

    def test_has_junk_key(self):
        r = _v("Build seed_gate.py validator")
        assert "junk" in r
        assert isinstance(r["junk"], bool)


# ===================================================================
# 21. Conservation laws (property-based)
# ===================================================================

class TestConservationLaws:
    def test_idempotent(self):
        text = "Build seed_gate.py validator module"
        r1 = _v(text)
        r2 = _v(text)
        assert r1 == r2

    def test_junk_never_has_verb_or_target(self):
        for text in ["", "x", "   ", "\n\n"]:
            r = _v(text)
            if r["junk"]:
                assert r["verb_found"] is None
                assert r["target_found"] is None

    def test_passed_or_reasons(self):
        texts = [
            "Build seed_gate.py", "", "Something vague and abstract",
            "Build something cool and exciting", "Fix state/agents.json now",
        ]
        for text in texts:
            r = _v(text)
            assert r["passed"] or len(r["reasons"]) > 0

    def test_score_monotonic_with_verb(self):
        no_verb = _score("The seed_gate.py thing", "", "seed_gate.py", "file")
        with_verb = _score("Build seed_gate.py thing", "build", "seed_gate.py", "file")
        assert with_verb >= no_verb

    def test_score_monotonic_with_target(self):
        no_target = _score("Build something here now", "build", "", "")
        with_target = _score("Build seed_gate.py here", "build", "seed_gate.py", "file")
        assert with_target >= no_target


# ===================================================================
# 22. Batch validation API
# ===================================================================

class TestBatchValidation:
    def test_batch_empty(self):
        batch = validate_batch([])
        assert len(batch.results) == 0
        assert batch.stats.total == 0
        assert batch.stats.pass_rate == 0.0

    def test_batch_single_pass(self):
        batch = validate_batch([("Build seed_gate.py validator", [])])
        assert batch.stats.total == 1
        assert batch.stats.passed == 1
        assert batch.stats.failed == 0
        assert batch.stats.junk == 0

    def test_batch_single_junk(self):
        batch = validate_batch([("", [])])
        assert batch.stats.total == 1
        assert batch.stats.passed == 0
        assert batch.stats.junk == 1
        assert batch.stats.failed == 0

    def test_batch_single_failed(self):
        batch = validate_batch([("Something vague and unspecific for testing", [])])
        assert batch.stats.total == 1
        assert batch.stats.passed == 0
        assert batch.stats.failed == 1
        assert batch.stats.junk == 0

    def test_batch_mixed(self):
        proposals = [
            ("Build seed_gate.py validator", []),          # pass
            ("", []),                                       # junk
            ("Something vague and unspecific here", []),    # fail
            ("Fix state/agents.json integrity", []),       # pass
            ("x", []),                                      # junk
        ]
        batch = validate_batch(proposals)
        assert batch.stats.total == 5
        assert batch.stats.passed == 2
        assert batch.stats.failed == 1
        assert batch.stats.junk == 2
        assert abs(batch.stats.pass_rate - 0.4) < 0.01
        assert abs(batch.stats.junk_rate - 0.4) < 0.01

    def test_batch_results_match_individual(self):
        proposals = [
            ("Build seed_gate.py validator", []),
            ("Explore consciousness and meaning", ["theme"]),
            ("", []),
        ]
        batch = validate_batch(proposals)
        for i, (text, tags) in enumerate(proposals):
            individual = validate_seed(text, tags)
            assert batch.results[i] == individual

    def test_batch_junk_items(self):
        batch = validate_batch([
            ("Build seed_gate.py", []),
            ("", []),
            ("x", []),
        ])
        junk = batch.junk_items
        assert len(junk) == 2
        assert all(r.junk for r in junk)

    def test_batch_failed_items(self):
        batch = validate_batch([
            ("Build seed_gate.py", []),
            ("Something vague and unspecific here", []),
            ("", []),
        ])
        failed = batch.failed_items
        assert len(failed) == 1
        assert all(not r.passed and not r.junk for r in failed)

    def test_batch_junk_disjoint_from_failed(self):
        batch = validate_batch([
            ("Build seed_gate.py", []),
            ("", []),
            ("Something vague here", []),
        ])
        junk_set = set(id(r) for r in batch.junk_items)
        failed_set = set(id(r) for r in batch.failed_items)
        assert junk_set.isdisjoint(failed_set)

    def test_batch_to_dicts(self):
        batch = validate_batch([
            ("Build seed_gate.py", []),
            ("", []),
        ])
        dicts = batch.to_dicts()
        assert len(dicts) == 2
        assert all(isinstance(d, dict) for d in dicts)
        assert all("passed" in d for d in dicts)

    def test_batch_purge_mode(self):
        batch = validate_batch(
            [("Something vague and unspecific here", [])],
            mode="purge",
        )
        assert batch.stats.passed == 1

    def test_batch_stats_conservation(self):
        """passed + failed + junk == total."""
        proposals = [
            ("Build seed_gate.py", []),
            ("", []),
            ("Something vague here", []),
            ("Fix state/agents.json now", []),
            ("x", []),
            ("Deploy worker.js to prod", []),
        ]
        batch = validate_batch(proposals)
        s = batch.stats
        assert s.passed + s.failed + s.junk == s.total


# ===================================================================
# 23. BatchStats
# ===================================================================

class TestBatchStats:
    def test_stats_frozen(self):
        s = BatchStats(total=3, passed=1, failed=1, junk=1)
        with pytest.raises(Exception):
            s.total = 5  # type: ignore

    def test_pass_rate(self):
        s = BatchStats(total=10, passed=7, failed=2, junk=1)
        assert abs(s.pass_rate - 0.7) < 0.01

    def test_junk_rate(self):
        s = BatchStats(total=10, passed=7, failed=2, junk=1)
        assert abs(s.junk_rate - 0.1) < 0.01

    def test_zero_total(self):
        s = BatchStats(total=0, passed=0, failed=0, junk=0)
        assert s.pass_rate == 0.0
        assert s.junk_rate == 0.0

    def test_merge(self):
        s1 = BatchStats(total=5, passed=3, failed=1, junk=1)
        s2 = BatchStats(total=3, passed=1, failed=2, junk=0)
        merged = s1.merge(s2)
        assert merged.total == 8
        assert merged.passed == 4
        assert merged.failed == 3
        assert merged.junk == 1

    def test_merge_conservation(self):
        s1 = BatchStats(total=5, passed=2, failed=2, junk=1)
        s2 = BatchStats(total=7, passed=4, failed=1, junk=2)
        m = s1.merge(s2)
        assert m.passed + m.failed + m.junk == m.total


# ===================================================================
# 24. BatchResult structure
# ===================================================================

class TestBatchResult:
    def test_results_is_tuple(self):
        batch = validate_batch([("Build seed_gate.py", [])])
        assert isinstance(batch.results, tuple)

    def test_stats_is_batchstats(self):
        batch = validate_batch([("Build seed_gate.py", [])])
        assert isinstance(batch.stats, BatchStats)

    def test_frozen(self):
        batch = validate_batch([("Build seed_gate.py", [])])
        with pytest.raises(Exception):
            batch.stats = None  # type: ignore


# ===================================================================
# 25. Stress / fuzz-style edge cases
# ===================================================================

class TestStressEdgeCases:
    """Gate must never crash on adversarial input."""

    def test_10kb_text(self):
        text = "Build seed_gate.py " * 500
        r = _v(text)
        assert isinstance(r["passed"], bool)
        assert r["score"] <= 1.0

    def test_null_bytes(self):
        r = _v("Build seed_gate.py\x00validator module")
        assert isinstance(r["passed"], bool)

    def test_emoji_heavy(self):
        r = _v("\U0001f680 Build seed_gate.py \U0001f3af validator \U0001f9ea module")
        assert isinstance(r["passed"], bool)

    def test_only_unicode(self):
        r = _v("\u6784\u5efa\u9a8c\u8bc1\u5668\u6a21\u5757\u6765\u68c0\u67e5\u79cd\u5b50\u63d0\u6848\u7684\u8d28\u91cf\u548c\u6709\u6548\u6027")
        assert isinstance(r["passed"], bool)

    def test_multiline_prose(self):
        text = """Build seed_gate.py with these features:
        1. Action verb detection
        2. Concrete target matching
        3. Junk rejection
        4. Scoring system"""
        r = _v(text)
        assert r["passed"] is True

    def test_repeated_single_char(self):
        r = _v("A" * 500)
        assert isinstance(r["passed"], bool)

    def test_whitespace_varieties(self):
        r = _v("Build\tseed_gate.py\r\nvalidator\fmodule")
        assert isinstance(r["passed"], bool)

    def test_backslash_heavy(self):
        r = _v(r"Build src\\seed_gate.py validator module")
        assert isinstance(r["passed"], bool)

    def test_batch_with_fuzz_inputs(self):
        proposals = [
            ("Build seed_gate.py", []),
            ("\U0001f680" * 100, []),
            ("\x00\x01\x02" * 10, []),
            ("A" * 5000, []),
            ("\n" * 100, []),
            ("Build" + "\t" * 50 + "seed_gate.py", []),
        ]
        batch = validate_batch(proposals)
        assert batch.stats.total == 6
        assert batch.stats.passed + batch.stats.failed + batch.stats.junk == 6

    def test_only_periods(self):
        r = _v("..." * 100)
        assert isinstance(r["passed"], bool)

    def test_single_word_repeated(self):
        r = _v("Build " * 200)
        assert isinstance(r["passed"], bool)
