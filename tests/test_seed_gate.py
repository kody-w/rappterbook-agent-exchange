"""Tests for seed_gate.py -- canonical specificity validator.

Covers: verb detection, target detection (files, paths, tools, modules,
CLI, discussions, channels, quoted), junk detection (hard + soft artifacts),
scoring with unique-target counting, validation pass/fail, exempt tags,
CLI, real-world proposals, edge cases, property invariants, smoke tests,
propose_seed.py contract, regression tests for false rejects, and
consolidated features from PRs #279-#287:
  VerbMatch, find_verb_with_position(), negation detection, commit-prefix
  normalization, score_breakdown(), explain(), enriched SeedGateResult
  (target_kind, verb_source, verb_position, negated, is_imperative,
  score_parts), numbered-ref filter, expanded abbreviation filter.
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
    ACTION_VERBS,
    EXEMPT_TAGS,
    KNOWN_MODULES,
    CHANNEL_RE,
    CLI_RE,
    DISCUSSION_RE,
    FILE_RE,
    MODULE_CONTEXT_RE,
    PATH_RE,
    QUOTED_RE,
    TOOL_RE,
    SeedGateResult,
    VerbMatch,
    BatchResult,
    BatchStats,
    passes_gate,
    validate,
    validate_seed,
    validate_batch,
    canonicalize_target,
    count_unique_targets,
    is_soft_artifact,
    find_verb,
    find_verb_with_position,
    find_all_verbs,
    find_target,
    is_junk,
    compute_score,
    normalize_proposal,
    detect_negation,
    score_breakdown,
    explain,
    suggest,
    _compute_score_parts,
    _extract_commit_prefix_verb,
    _is_false_file_match,
    _INFLECTION_MAP,
    PHRASAL_VERBS,
    TAG_IMPLIED_VERBS,
    QUESTION_STEMS,
)


def _v(text, tags=None, mode="admission"):
    return validate(text, tags or [], mode)


def _vs(text, tags=None, mode="admission"):
    return validate_seed(text, tags or [], mode=mode)


# ===================================================================
# 1. Constants
# ===================================================================

class TestConstants:
    def test_action_verbs_nonempty(self):
        assert len(ACTION_VERBS) >= 40

    def test_action_verbs_all_lowercase(self):
        for v in ACTION_VERBS:
            assert v == v.lower()

    def test_exempt_tags_nonempty(self):
        assert len(EXEMPT_TAGS) >= 4

    def test_exempt_tags_all_lowercase(self):
        for t in EXEMPT_TAGS:
            assert t == t.lower()

    def test_known_modules_discovered(self):
        assert len(KNOWN_MODULES) >= 10

    def test_known_modules_no_test_or_run(self):
        for m in KNOWN_MODULES:
            assert not m.startswith("test_")
            assert not m.startswith("run_")

    def test_known_modules_only_snake_case(self):
        for m in KNOWN_MODULES:
            assert "_" in m

    def test_known_modules_contains_expected(self):
        for m in ("water_mining", "solar_array", "seed_gate"):
            assert m in KNOWN_MODULES

    def test_prevent_avoid_are_action_verbs(self):
        assert "prevent" in ACTION_VERBS
        assert "avoid" in ACTION_VERBS


# ===================================================================
# 2. Regex patterns
# ===================================================================

class TestRegexPatterns:
    def test_file_re_python(self):
        assert FILE_RE.search("Build seed_gate.py validator")

    def test_file_re_json(self):
        assert FILE_RE.search("Fix state/agents.json integrity")

    def test_file_re_rust(self):
        assert FILE_RE.search("Port parser to grammar.rs")

    def test_file_re_no_match(self):
        assert not FILE_RE.search("Build the best system")

    def test_path_re_src(self):
        assert PATH_RE.search("Refactor src/thermal/core module")

    def test_path_re_engine(self):
        assert PATH_RE.search("Review engine/prompts/frame.md")

    def test_path_re_tests(self):
        assert PATH_RE.search("Add tests/test_seed_gate.py coverage")

    def test_tool_re_snake(self):
        assert TOOL_RE.search("Wire up state_io for persistence")

    def test_tool_re_kebab(self):
        assert TOOL_RE.search("Deploy via safe-commit pipeline")

    def test_cli_re_backtick(self):
        assert CLI_RE.search("Run `python -m pytest` on CI")

    def test_cli_re_flag(self):
        assert CLI_RE.search("Add --verbose flag to CLI")

    def test_discussion_re(self):
        m = DISCUSSION_RE.search("See discussion #12345")
        assert m and m.group(1) == "12345"

    def test_channel_re(self):
        assert CHANNEL_RE.search("Post to r/mars-engineering")

    def test_quoted_re_double(self):
        assert QUOTED_RE.search('Implement "circuit breaker" pattern')

    def test_quoted_re_single(self):
        assert QUOTED_RE.search("Add 'retry logic' to handler")

    def test_module_context_backtick(self):
        assert MODULE_CONTEXT_RE.search("Refactor `water_mining` module")

    def test_module_context_import(self):
        assert MODULE_CONTEXT_RE.search("Fix import water_mining call")


# ===================================================================
# 3. VerbMatch + find_verb_with_position (consolidated from PRs #284, #287)
# ===================================================================

class TestVerbMatch:
    def test_dataclass_fields(self):
        m = VerbMatch(verb="build", token_index=0, source="text")
        assert m.verb == "build"
        assert m.token_index == 0
        assert m.source == "text"

    def test_frozen(self):
        m = VerbMatch(verb="build", token_index=0, source="text")
        with pytest.raises(Exception):
            m.verb = "fix"  # type: ignore

    def test_equality(self):
        a = VerbMatch(verb="build", token_index=0, source="text")
        b = VerbMatch(verb="build", token_index=0, source="text")
        assert a == b


class TestFindVerbWithPosition:
    def test_simple_verb(self):
        m = find_verb_with_position("Build auth.py validator")
        assert m is not None
        assert m.verb == "build"
        assert m.token_index == 0
        assert m.source == "text"

    def test_verb_not_first(self):
        m = find_verb_with_position("Please fix auth.py crash")
        assert m is not None
        assert m.verb == "fix"
        assert m.token_index == 1

    def test_inflected_verb(self):
        m = find_verb_with_position("Building auth.py validator now")
        assert m is not None
        assert m.verb == "build"
        assert m.source == "inflected"

    def test_phrasal_verb(self):
        m = find_verb_with_position("Set up the auth.py pipeline")
        assert m is not None
        assert m.verb == "set up"
        assert m.source == "phrasal"
        assert m.token_index == 0

    def test_inflected_phrasal(self):
        m = find_verb_with_position("Setting up the auth.py pipeline")
        assert m is not None
        assert m.verb == "set up"
        assert m.source == "inflected"

    def test_no_verb(self):
        assert find_verb_with_position("The great system") is None

    def test_limit_parameter(self):
        m = find_verb_with_position("xxxx build auth.py", limit=4)
        assert m is None

    def test_wraps_find_verb(self):
        """find_verb() returns the same verb as find_verb_with_position()."""
        texts = [
            "Build auth.py", "Creating solar_array.py now",
            "Set up the pipeline", "The system", "",
        ]
        for text in texts:
            vm = find_verb_with_position(text)
            fv = find_verb(text)
            assert (vm.verb if vm else None) == fv, f"Mismatch on: {text!r}"


# ===================================================================
# 4. find_verb / find_all_verbs (existing tests, expanded)
# ===================================================================

class TestFindVerb:
    def test_base_form(self):
        assert find_verb("Build the auth module") == "build"

    def test_inflected_builds(self):
        assert find_verb("Builds the auth module") == "build"

    def test_inflected_creating(self):
        assert find_verb("Creating new test module") == "create"

    def test_inflected_deployed(self):
        assert find_verb("Deployed new release today") == "deploy"

    def test_phrasal_set_up(self):
        assert find_verb("Set up the pipeline for deployment") == "set up"

    def test_phrasal_inflected_setting_up(self):
        assert find_verb("Setting up the pipeline now") == "set up"

    def test_no_verb(self):
        assert find_verb("The quick brown fox") is None

    def test_empty(self):
        assert find_verb("") is None

    def test_irregular_built(self):
        assert find_verb("Built the new module from scratch") == "build"

    def test_limit(self):
        assert find_verb("xxxx build it", limit=4) is None

    def test_prevent_is_action_verb(self):
        assert find_verb("Prevent duplicate proposals in seed_gate.py") == "prevent"

    def test_avoid_is_action_verb(self):
        assert find_verb("Avoid crashes in water_mining.py") == "avoid"


class TestFindAllVerbs:
    def test_multiple(self):
        verbs = find_all_verbs("Build and test the auth.py module")
        assert "build" in verbs and "test" in verbs

    def test_dedup(self):
        verbs = find_all_verbs("Build, build, build the thing")
        assert verbs.count("build") == 1

    def test_empty(self):
        assert find_all_verbs("") == []

    def test_inflected_and_base(self):
        verbs = find_all_verbs("Building tests then deploy the fix")
        assert "build" in verbs and "test" in verbs and "deploy" in verbs


# ===================================================================
# 5. find_target
# ===================================================================

class TestFindTarget:
    def test_python_file(self):
        t, k = find_target("Build auth.py module")
        assert t == "auth.py" and k == "file"

    def test_json_file(self):
        t, k = find_target("Fix state/agents.json schema")
        assert "agents.json" in t and k == "file"

    def test_path(self):
        t, k = find_target("Refactor code in src/thermal/core")
        assert t.startswith("src/") and k == "path"

    def test_func(self):
        t, k = find_target("Fix the validate() function")
        assert "validate()" in t and k == "func"

    def test_channel(self):
        t, k = find_target("Post to r/mars-engineering today")
        assert "r/mars-engineering" in t and k == "channel"

    def test_env_var(self):
        t, k = find_target("Configure $STATE_DIR variable")
        assert "$STATE_DIR" in t and k == "env"

    def test_const(self):
        t, k = find_target("Refactor ACTION_VERBS constant set")
        assert "ACTION_VERBS" in t and k == "const"

    def test_tool(self):
        t, k = find_target("Wire up state_io for persistence")
        assert "state_io" in t

    def test_discussion(self):
        t, k = find_target("See #12345 for context")
        assert "#12345" in t and k == "discussion"

    def test_quoted(self):
        t, k = find_target('Implement "circuit breaker" pattern')
        assert "circuit breaker" in t and k == "quoted"

    def test_none(self):
        t, k = find_target("Make everything better")
        assert t == "" and k == ""


# ===================================================================
# 6. False-file-match filtering
# ===================================================================

class TestFalseFileMatch:
    def test_eg(self):
        assert _is_false_file_match("e.g")

    def test_ie(self):
        assert _is_false_file_match("i.e")

    def test_version(self):
        assert _is_false_file_match("2.0.1")

    def test_version_v(self):
        assert _is_false_file_match("v1.2.3")

    def test_not_false(self):
        assert not _is_false_file_match("auth.py")

    # Numbered refs (PR #281)
    def test_fig_1(self):
        assert _is_false_file_match("fig.1")

    def test_ch_3(self):
        assert _is_false_file_match("ch.3")

    def test_vol_2(self):
        assert _is_false_file_match("vol.2")

    def test_eq_1(self):
        assert _is_false_file_match("eq.1")

    def test_sec_4(self):
        assert _is_false_file_match("sec.4")

    # Expanded abbreviations (PR #281)
    def test_phd(self):
        assert _is_false_file_match("Ph.D")

    def test_us(self):
        assert _is_false_file_match("U.S")

    def test_uk(self):
        assert _is_false_file_match("U.K")


# ===================================================================
# 7. Commit-prefix normalization (consolidated from PRs #281-#283)
# ===================================================================

class TestCommitPrefixNormalization:
    def test_feat_prefix(self):
        assert normalize_proposal("feat: Build auth.py") == "Build auth.py"

    def test_fix_prefix(self):
        assert normalize_proposal("fix: Resolve crash in gate") == "Resolve crash in gate"

    def test_scoped_prefix(self):
        assert normalize_proposal("fix(gate): Resolve crash") == "Resolve crash"

    def test_no_prefix(self):
        assert normalize_proposal("Build auth.py") == "Build auth.py"

    def test_case_insensitive(self):
        assert normalize_proposal("FEAT: Build auth.py") == "Build auth.py"

    def test_refactor_prefix(self):
        assert normalize_proposal("refactor: Clean up imports") == "Clean up imports"

    def test_docs_prefix(self):
        assert normalize_proposal("docs: Document the API") == "Document the API"


class TestCommitPrefixVerb:
    def test_feat(self):
        assert _extract_commit_prefix_verb("feat: something") == "build"

    def test_fix(self):
        assert _extract_commit_prefix_verb("fix: something") == "fix"

    def test_no_prefix(self):
        assert _extract_commit_prefix_verb("Build something") is None

    def test_perf(self):
        assert _extract_commit_prefix_verb("perf: optimize hotpath") == "optimize"


class TestCommitPrefixIntegration:
    def test_feat_prefix_passes(self):
        r = _v("feat: Build auth.py validator for login")
        assert r["passed"]
        assert r["verb_found"] == "build"

    def test_fix_prefix_passes(self):
        r = _v("fix: Resolve crash in seed_gate.py parser")
        assert r["passed"]
        assert r["verb_found"] == "resolve"

    def test_prefix_only_no_body_verb_uses_fallback(self):
        r = _v("feat: auth.py login module improvements for users")
        # commit prefix verb "build" is fallback
        assert r["verb_found"] == "build" or r["verb_found"] == "improve"

    def test_prefix_stripped_for_junk_check(self):
        # "feat: x" -> "x" which is too short -> junk
        r = _v("feat: x")
        assert r["junk"] is True


# ===================================================================
# 8. Negation detection (consolidated from PRs #279-#287)
# ===================================================================

class TestNegationDetection:
    def test_dont_build(self):
        assert detect_negation("Don't build auth.py")

    def test_wont_deploy(self):
        assert detect_negation("Won't deploy the service")

    def test_shouldnt_test(self):
        assert detect_negation("Shouldn't test this module")

    def test_do_not_build(self):
        assert detect_negation("Do not build auth.py")

    def test_does_not_deploy(self):
        assert detect_negation("Does not deploy correctly")

    def test_never_deploy(self):
        assert detect_negation("Never deploy to production directly")

    def test_positive_not_negated(self):
        assert not detect_negation("Build auth.py validator")

    def test_subordinate_clause_immunity(self):
        assert not detect_negation("Fix auth.py so tests don't hang")

    def test_purpose_clause_immunity(self):
        assert not detect_negation("Build retry logic that prevents duplicate writes")

    def test_so_that_immunity(self):
        assert not detect_negation("Add guardrails so that users can't crash seed_gate.py")

    def test_when_clause_immunity(self):
        assert not detect_negation("Deploy auth.py when tests don't fail")

    def test_if_clause_immunity(self):
        assert not detect_negation("Build fallback if service doesn't respond")

    def test_but_redemption(self):
        assert not detect_negation("Build auth.py but don't break tests")

    def test_smart_quotes(self):
        assert detect_negation("Don\u2019t build auth.py")

    def test_question_immunity(self):
        assert not detect_negation("Should we avoid deploying auth.py?")

    def test_simple_positive(self):
        assert not detect_negation("Build, test, and deploy auth.py")

    def test_empty(self):
        assert not detect_negation("")


class TestNegationValidation:
    def test_dont_build_fails_gate(self):
        r = _v("Don't build auth.py validator")
        assert not r["passed"]
        assert r["negated"]
        assert any("negat" in reason.lower() for reason in r["reasons"])

    def test_positive_build_passes(self):
        r = _v("Build auth.py validator module")
        assert r["passed"]
        assert not r["negated"]

    def test_subordinate_negation_passes(self):
        r = _v("Fix seed_gate.py so tests don't crash")
        assert r["passed"]
        assert not r["negated"]

    def test_prevent_as_action_passes(self):
        r = _v("Prevent duplicate proposals in seed_gate.py")
        assert r["passed"]

    def test_avoid_as_action_passes(self):
        r = _v("Avoid crashes in water_mining.py module")
        assert r["passed"]


# ===================================================================
# 9. Junk detection
# ===================================================================

class TestJunkDetection:
    def test_empty(self):
        assert is_junk("") != ""

    def test_whitespace(self):
        assert is_junk("   ") != ""

    def test_too_short(self):
        assert is_junk("fix x") != ""

    def test_backtick_start(self):
        assert is_junk("`something here is broken") != ""

    def test_url_start(self):
        assert is_junk("https://example.com/something") != ""

    def test_todo_marker(self):
        assert is_junk("TODO: fix the thing later on") != ""

    def test_lowercase_no_verb(self):
        assert is_junk("the quick brown fox jumps") != ""

    def test_lowercase_with_verb_ok(self):
        assert is_junk("build the auth module properly") == ""

    def test_hard_artifact(self):
        assert is_junk("Parser grabbed something weird from there") != ""

    def test_valid_proposal(self):
        assert is_junk("Build auth.py validator module") == ""

    def test_run_prefix_exception(self):
        assert is_junk("run_proof executed the validator successfully") == ""


# ===================================================================
# 10. Soft artifacts
# ===================================================================

class TestSoftArtifact:
    def test_regex_mention(self):
        assert is_soft_artifact("the regex captured something weird")

    def test_parser_mention(self):
        assert is_soft_artifact("the parser failed on input data")

    def test_normal_text(self):
        assert not is_soft_artifact("Build auth.py validator module")


# ===================================================================
# 11. Canonicalization and unique targets
# ===================================================================

class TestCanonicalization:
    def test_strip_extension(self):
        assert canonicalize_target("auth.py") == "auth"

    def test_strip_path(self):
        assert canonicalize_target("src/auth.py") == "auth"

    def test_strip_quotes(self):
        assert canonicalize_target('"something"') == "something"

    def test_lowercase(self):
        assert canonicalize_target("AUTH.PY") == "auth"

    def test_empty(self):
        assert canonicalize_target("") == ""


class TestUniqueTargets:
    def test_two_files(self):
        assert count_unique_targets("Build auth.py and config.py") == 2

    def test_same_file_twice(self):
        assert count_unique_targets("Fix auth.py and also auth.py") == 1

    def test_substring_dedup(self):
        assert count_unique_targets("Build seed_gate.py and test seed_gate") >= 1

    def test_no_targets(self):
        assert count_unique_targets("Make everything better") == 0


# ===================================================================
# 12. Score computation
# ===================================================================

class TestScoreComputation:
    def test_verb_only(self):
        s = compute_score("Build something great", "build", None, "")
        assert 0.0 < s < 0.5

    def test_verb_plus_file(self):
        s = compute_score("Build auth.py module", "build", "auth.py", "file")
        assert s >= 0.5

    def test_file_worth_more_than_quoted(self):
        sf = compute_score("Build auth.py", "build", "auth.py", "file")
        sq = compute_score('Build "auth"', "build", '"auth"', "quoted")
        assert sf > sq

    def test_long_text_bonus(self):
        short = "Build auth.py"
        long = "Build auth.py with comprehensive validation logic for the entire system including tests and docs"
        assert compute_score(long, "build", "auth.py", "file") >= compute_score(short, "build", "auth.py", "file")

    def test_multi_target_bonus(self):
        single = "Build auth.py module"
        multi = "Build auth.py and config.py modules"
        assert compute_score(multi, "build", "auth.py", "file") >= compute_score(single, "build", "auth.py", "file")

    def test_max_one(self):
        s = compute_score("Build auth.py and config.py and test.py with comprehensive validation " * 3,
                          "build", "auth.py", "file")
        assert s <= 1.0

    def test_imperative_bonus(self):
        imp = compute_score("Build auth.py module", "build", "auth.py", "file")
        non_imp = compute_score("Please build auth.py module", "build", "auth.py", "file")
        assert imp >= non_imp


# ===================================================================
# 13. Score decomposition (_compute_score_parts + score_breakdown)
# ===================================================================

class TestScoreParts:
    def test_returns_list_of_tuples(self):
        parts = _compute_score_parts("Build auth.py", "build", "auth.py", "file")
        assert isinstance(parts, list)
        for name, value in parts:
            assert isinstance(name, str)
            assert isinstance(value, (int, float))

    def test_verb_component(self):
        parts = dict(_compute_score_parts("Build auth.py", "build", "auth.py", "file"))
        assert parts["verb"] == 2.5

    def test_target_component_file(self):
        parts = dict(_compute_score_parts("Build auth.py", "build", "auth.py", "file"))
        assert parts["target"] == 4.0

    def test_no_verb_no_component(self):
        parts = dict(_compute_score_parts("The auth.py system", None, "auth.py", "file"))
        assert "verb" not in parts

    def test_imperative_bonus(self):
        parts = dict(_compute_score_parts("Build auth.py", "build", "auth.py", "file"))
        assert "imperative" in parts

    def test_consistency_with_compute_score(self):
        text = "Build auth.py and config.py validator modules"
        verb = find_verb(text)
        target, kind = find_target(text)
        parts = _compute_score_parts(text, verb, target, kind)
        raw = sum(v for _, v in parts)
        expected = min(raw / 10.0, 1.0)
        actual = compute_score(text, verb, target, kind)
        assert abs(expected - actual) < 0.001


class TestScoreBreakdown:
    def test_returns_dict(self):
        bd = score_breakdown("Build auth.py validator module")
        assert isinstance(bd, dict)
        assert "score" in bd
        assert "total_raw" in bd

    def test_has_verb_component(self):
        bd = score_breakdown("Build auth.py validator module")
        assert "verb" in bd

    def test_has_target_component(self):
        bd = score_breakdown("Build auth.py validator module")
        assert "target" in bd

    def test_score_matches_validate(self):
        text = "Build auth.py and config.py validator"
        bd = score_breakdown(text)
        v = _v(text)
        assert abs(bd["score"] - v["score"]) < 0.01

    def test_feat_prefix(self):
        bd = score_breakdown("feat: Build auth.py module")
        assert "verb" in bd

    def test_no_verb_no_component(self):
        bd = score_breakdown("The great system of wonders")
        assert "verb" not in bd


# ===================================================================
# 14. explain() API
# ===================================================================

class TestExplainAPI:
    def test_passing_contains_pass(self):
        result = explain("Build auth.py validator module")
        assert "PASS" in result

    def test_failing_contains_fail(self):
        result = explain("Make everything better and nicer")
        assert "FAIL" in result

    def test_junk_contains_junk(self):
        result = explain("x")
        assert "JUNK" in result

    def test_shows_verb(self):
        result = explain("Build auth.py validator module")
        assert "build" in result.lower()

    def test_shows_target(self):
        result = explain("Build auth.py validator module")
        assert "auth.py" in result

    def test_negated_shows_negation(self):
        result = explain("Don't build auth.py ever again")
        assert "negat" in result.lower()

    def test_returns_string(self):
        assert isinstance(explain("Build auth.py"), str)


# ===================================================================
# 15. suggest() API
# ===================================================================

class TestSuggestAPI:
    def test_passing_no_suggestions(self):
        assert suggest("Build auth.py validator module") == []

    def test_no_verb_suggestion(self):
        sugg = suggest("The auth.py module needs work badly")
        assert any("verb" in s.lower() for s in sugg)

    def test_no_target_suggestion(self):
        sugg = suggest("Build something amazing for everyone")
        assert any("target" in s.lower() or "filename" in s.lower() for s in sugg)

    def test_negated_suggestion(self):
        sugg = suggest("Don't build auth.py validator ever")
        assert any("positive" in s.lower() or "rephrase" in s.lower() for s in sugg)


# ===================================================================
# 16. Enriched SeedGateResult fields
# ===================================================================

class TestEnrichedSeedGateResult:
    def test_target_kind_file(self):
        r = _vs("Build auth.py validator module")
        assert r.target_kind == "file"

    def test_target_kind_path(self):
        r = _vs("Refactor code in src/thermal/core")
        assert r.target_kind == "path"

    def test_verb_source_text(self):
        r = _vs("Build auth.py validator module")
        assert r.verb_source in ("text", "inflected", "phrasal")

    def test_verb_source_tag(self):
        r = _vs("The auth.py module needs attention", tags=["code"])
        assert r.verb_source == "tag"

    def test_verb_source_question(self):
        r = _vs("What if we redesign the whole architecture?", tags=["philosophy"])
        assert r.verb_source == "question"

    def test_verb_position_zero(self):
        r = _vs("Build auth.py validator module")
        assert r.verb_position == 0

    def test_verb_position_nonzero(self):
        r = _vs("Please fix auth.py crash issue")
        assert r.verb_position is not None and r.verb_position > 0

    def test_score_parts_populated(self):
        r = _vs("Build auth.py validator module")
        assert len(r.score_parts) > 0

    def test_negated_field_false(self):
        r = _vs("Build auth.py validator module")
        assert r.negated is False

    def test_negated_field_true(self):
        r = _vs("Don't build auth.py ever again")
        assert r.negated is True

    def test_is_imperative_true(self):
        r = _vs("Build auth.py validator module")
        assert r.is_imperative is True

    def test_is_imperative_false(self):
        r = _vs("Please build auth.py module now")
        assert r.is_imperative is False

    def test_to_dict_has_all_fields(self):
        r = _vs("Build auth.py validator module")
        d = r.to_dict()
        for key in ("target_kind", "verb_source", "verb_position", "score_parts",
                     "negated", "is_imperative"):
            assert key in d, f"Missing key: {key}"

    def test_confidence_high(self):
        r = _vs("Build auth.py and config.py validator modules")
        assert r.confidence in ("high", "medium")

    def test_confidence_none_on_fail(self):
        r = _vs("Make everything better and amazing")
        assert r.confidence is None


# ===================================================================
# 17. Validation pass/fail (comprehensive)
# ===================================================================

class TestValidation:
    def test_verb_plus_file(self):
        assert _v("Build auth.py validator module")["passed"]

    def test_verb_plus_path(self):
        assert _v("Refactor code in src/thermal/core")["passed"]

    def test_verb_plus_tool(self):
        assert _v("Wire up state_io for data persistence")["passed"]

    def test_verb_plus_discussion(self):
        assert _v("Review discussion #12345 thoroughly")["passed"]

    def test_verb_plus_channel(self):
        assert _v("Post findings to r/mars-engineering")["passed"]

    def test_verb_plus_env(self):
        assert _v("Configure $STATE_DIR for local dev")["passed"]

    def test_verb_plus_const(self):
        assert _v("Refactor ACTION_VERBS constant set")["passed"]

    def test_verb_plus_func(self):
        assert _v("Fix the validate() function for edge cases")["passed"]

    def test_verb_plus_cli(self):
        assert _v("Add --verbose flag to the CLI output")["passed"]

    def test_verb_plus_quoted(self):
        assert _v('Implement "circuit breaker" pattern here')["passed"]

    def test_no_verb_fails(self):
        assert not _v("The auth.py module needs more work")["passed"]

    def test_no_target_fails(self):
        assert not _v("Build something amazing for everyone")["passed"]

    def test_empty_fails(self):
        assert not _v("")["passed"]

    def test_junk_fails(self):
        assert not _v("x")["passed"]

    def test_exempt_tag_no_target(self):
        assert _v("Explore the nature of consciousness deeply", tags=["philosophy"])["passed"]

    def test_exempt_tag_still_needs_verb(self):
        assert not _v("The nature of consciousness is complex", tags=["philosophy"])["passed"]

    def test_inflected_verb(self):
        assert _v("Building auth.py validator module now")["passed"]

    def test_phrasal_verb(self):
        assert _v("Set up auth.py pipeline for deployment")["passed"]


class TestValidationRealProposals:
    """Real-world proposals from the swarm."""

    def test_build_water_mining(self):
        assert _v("Build water_mining.py optimizer for drilling")["passed"]

    def test_fix_seed_gate(self):
        assert _v("Fix seed_gate.py false positive on abbreviations")["passed"]

    def test_test_solar_array(self):
        assert _v("Test solar_array.py degradation model thoroughly")["passed"]

    def test_refactor_state_io(self):
        assert _v("Refactor state_io to use atomic writes everywhere")["passed"]

    def test_deploy_discussion(self):
        assert _v("Deploy the fix discussed in #12503")["passed"]

    def test_wire_up_pipeline(self):
        assert _v("Wire up the CI pipeline for auth.py")["passed"]

    def test_configure_env(self):
        assert _v("Configure $GITHUB_TOKEN for API access")["passed"]

    def test_feat_prefix(self):
        assert _v("feat: Build auth.py validator module")["passed"]

    def test_fix_prefix(self):
        assert _v("fix: Resolve crash in seed_gate.py parser")["passed"]

    def test_prevent_duplicates(self):
        assert _v("Prevent duplicate proposals in propose_seed.py")["passed"]


class TestValidationFalseRejects:
    """Regression: things that SHOULD pass but were rejected in earlier versions."""

    def test_lowercase_verb_start(self):
        assert _v("build seed_gate.py validator for proposals")["passed"]

    def test_file_first(self):
        r = _v("seed_gate.py needs a comprehensive overhaul soon")
        # File detected, but no verb -> should fail
        assert not r["passed"] or r["verb_found"] is not None

    def test_version_not_file(self):
        r = _v("Upgrade to version 2.0.1 of the system")
        # "2.0.1" should not match as file
        assert r["verb_found"] == "upgrade"

    def test_eg_not_file(self):
        r = _v("Build a validator, e.g. for proposals and similar things")
        # "e.g" should not match as file target
        # "e.g." correctly filtered; no other target found
        assert r["verb_found"] == "build"
        assert r["target_found"] is None

    def test_fig_1_not_file(self):
        r = _v("Build the chart shown in fig.1 of the proposal document")
        # fig.1 should not match as file
        assert r["verb_found"] == "build"


# ===================================================================
# 18. Batch API
# ===================================================================

class TestBatchValidation:
    def test_basic_batch(self):
        proposals = [
            "Build auth.py validator module",
            "x",
            "Make everything better and nicer",
        ]
        br = validate_batch(proposals)
        assert br.stats.total == 3
        assert br.stats.passed >= 1
        assert br.stats.junk >= 1

    def test_batch_result_types(self):
        br = validate_batch(["Build auth.py validator module"])
        assert isinstance(br, BatchResult)
        assert isinstance(br.stats, BatchStats)

    def test_batch_stats_pass_rate(self):
        br = validate_batch(["Build auth.py module", "Fix config.py file"])
        assert br.stats.pass_rate > 0.0

    def test_batch_stats_merge(self):
        a = BatchStats(total=10, passed=5, failed=3, junk=2)
        b = BatchStats(total=5, passed=3, failed=1, junk=1)
        m = a.merge(b)
        assert m.total == 15 and m.passed == 8

    def test_empty_batch(self):
        br = validate_batch([])
        assert br.stats.total == 0


# ===================================================================
# 19. Property-based invariants
# ===================================================================

_CORPUS = [
    "Build auth.py validator module",
    "Fix state/agents.json integrity bug",
    "Refactor src/thermal/core for performance",
    "Test solar_array.py degradation model",
    "Deploy the fix discussed in #12503",
    "Wire up state_io for persistence layer",
    "Configure $GITHUB_TOKEN for API access",
    "Refactor ACTION_VERBS constant set definition",
    'Implement "circuit breaker" pattern here',
    "Add --verbose flag to the CLI output",
    "Post findings to r/mars-engineering channel",
    "Set up auth.py pipeline for deployment",
    "Building auth.py validator module now",
    "feat: Build auth.py validator module here",
    "Make everything better and more amazing",
    "x",
    "",
    "The quick brown fox jumps over lazy dog",
    "Don't build auth.py ever again please",
    "Fix seed_gate.py so tests don't crash",
    "Prevent duplicate proposals in propose_seed.py",
]


class TestPropertyInvariants:
    @pytest.mark.parametrize("text", _CORPUS)
    def test_score_in_range(self, text):
        r = _v(text)
        assert 0.0 <= r["score"] <= 1.0

    @pytest.mark.parametrize("text", _CORPUS)
    def test_passed_implies_verb(self, text):
        r = _v(text)
        if r["passed"]:
            assert r["verb_found"] is not None

    @pytest.mark.parametrize("text", _CORPUS)
    def test_junk_implies_not_passed(self, text):
        r = _v(text)
        if r["junk"]:
            assert not r["passed"]

    @pytest.mark.parametrize("text", _CORPUS)
    def test_dict_has_all_keys(self, text):
        r = _v(text)
        for key in ("passed", "score", "verb_found", "target_found", "junk",
                     "reasons", "advisory", "confidence", "all_verbs", "all_targets",
                     "target_kind", "verb_source", "verb_position", "score_parts",
                     "negated", "is_imperative"):
            assert key in r, f"Missing key {key} for: {text!r}"

    @pytest.mark.parametrize("text", _CORPUS)
    def test_find_verb_consistency(self, text):
        """find_verb() agrees with find_verb_with_position() for lexical verbs."""
        vm = find_verb_with_position(text)
        fv = find_verb(text)
        assert (vm.verb if vm else None) == fv

    @pytest.mark.parametrize("text", _CORPUS)
    def test_score_breakdown_consistency(self, text):
        """score_breakdown score matches validate score for passing proposals."""
        r = _v(text)
        bd = score_breakdown(text)
        # Scores should be close (may differ slightly due to tag inference)
        assert abs(bd["score"] - r["score"]) < 0.15 or not r["passed"]

    @pytest.mark.parametrize("text", _CORPUS)
    def test_negated_implies_not_passed(self, text):
        r = _v(text)
        if r["negated"] and r["verb_found"]:
            assert not r["passed"]


# ===================================================================
# 20. Inflection map
# ===================================================================

class TestInflectionMap:
    def test_builds(self):
        assert _INFLECTION_MAP.get("builds") == "build"

    def test_creating(self):
        assert _INFLECTION_MAP.get("creating") == "create"

    def test_deployed(self):
        assert _INFLECTION_MAP.get("deployed") == "deploy"

    def test_shipped(self):
        assert _INFLECTION_MAP.get("shipped") == "ship"

    def test_running(self):
        assert _INFLECTION_MAP.get("running") == "run"

    def test_built(self):
        assert _INFLECTION_MAP.get("built") == "build"

    def test_wrote(self):
        assert _INFLECTION_MAP.get("wrote") == "write"

    def test_base_not_in_map(self):
        assert "build" not in _INFLECTION_MAP


# ===================================================================
# 21. Tag-implied verbs and question stems
# ===================================================================

class TestTagImpliedVerbs:
    def test_code_tag_implies_build(self):
        r = _vs("The auth.py module needs some attention", tags=["code"])
        assert r.verb_found == "build"
        assert r.verb_source == "tag"

    def test_test_tag_implies_test(self):
        r = _vs("The solar_array.py coverage is low", tags=["test"])
        assert r.verb_found == "test"
        assert r.verb_source == "tag"

    def test_debug_tag_implies_debug(self):
        r = _vs("The water_mining.py module is broken", tags=["debug"])
        assert r.verb_found == "debug"
        assert r.verb_source == "tag"


class TestQuestionStems:
    def test_what_if(self):
        r = _vs("What if we redesign the architecture?", tags=["philosophy"])
        assert r.verb_found == "explore"
        assert r.verb_source == "question"

    def test_how_might(self):
        r = _vs("How might we improve agent communication?", tags=["debate"])
        assert r.verb_found == "improve"
        assert r.verb_source == "text"  # "improve" found directly in text

    def test_how_might_pure_question(self):
        r = _vs("How might we approach the problem differently?", tags=["debate"])
        assert r.verb_found == "design"
        assert r.verb_source == "question"

    def test_should_we(self):
        r = _vs("Should we evaluate the governance model?", tags=["philosophy"])
        assert r.verb_found == "evaluate"
        assert r.verb_source == "text"  # evaluate found directly in text

    def test_question_without_exempt_tag_fails(self):
        r = _vs("What if we redesign everything from scratch?")
        # No exempt tag -> question stem not used, "redesign" not in verbs
        assert r.verb_source != "question"


# ===================================================================
# 22. Purge mode
# ===================================================================

class TestPurgeMode:
    def test_purge_mode_lenient(self):
        r = _v("The auth.py module is somewhat okay now", mode="purge")
        assert r["passed"]

    def test_purge_mode_junk_still_fails(self):
        r = _v("x", mode="purge")
        assert r["junk"]


# ===================================================================
# 23. CLI smoke test
# ===================================================================

class TestCLI:
    def test_cli_passing(self):
        result = subprocess.run(
            [sys.executable, "-m", "seed_gate", "Build auth.py validator module"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT / "src"),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["passed"]

    def test_cli_failing(self):
        result = subprocess.run(
            [sys.executable, "-m", "seed_gate", "Make everything better"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT / "src"),
        )
        assert result.returncode == 1

    def test_cli_explain(self):
        result = subprocess.run(
            [sys.executable, "-m", "seed_gate", "--explain", "Build auth.py module"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT / "src"),
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout or "build" in result.stdout.lower()


# ===================================================================
# 24. propose_seed.py contract
# ===================================================================

class TestProposeSeedContract:
    """Verify the dict API shape that propose_seed.py depends on."""

    def test_passed_is_bool(self):
        assert isinstance(_v("Build auth.py module")["passed"], bool)

    def test_score_is_float(self):
        assert isinstance(_v("Build auth.py module")["score"], float)

    def test_reasons_is_list(self):
        assert isinstance(_v("x")["reasons"], list)

    def test_junk_is_bool(self):
        assert isinstance(_v("x")["junk"], bool)

    def test_verb_found_nullable(self):
        r = _v("Build auth.py module")
        assert r["verb_found"] is not None
        r2 = _v("The great system of wonders")
        assert r2["verb_found"] is None or r2["junk"]

    def test_target_found_nullable(self):
        r = _v("Build auth.py module")
        assert r["target_found"] is not None


# ===================================================================
# 25. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_unicode(self):
        r = _v("Build auth.py — the validator module")
        assert r["passed"]

    def test_very_long_text(self):
        text = "Build auth.py " + "word " * 500
        r = _v(text)
        assert r["score"] > 0

    def test_numbers_only(self):
        assert not _v("1234567890123456789")["passed"]

    def test_special_chars(self):
        assert not _v("!@#$%^&*()_+{}|:<>?")["passed"]

    def test_mixed_case_tag(self):
        r = _v("Explore the nature of existence deeply", tags=["Philosophy"])
        assert r["passed"]

    def test_multiple_verbs(self):
        r = _v("Build, test, and deploy auth.py module")
        assert r["passed"]
        assert len(r["all_verbs"]) >= 2


# ===================================================================
# 26. Smoke test (simulation)
# ===================================================================

class TestSmoke:
    def test_10_proposals(self):
        proposals = [
            "Build water_mining.py optimizer for drilling",
            "Fix seed_gate.py false positive on abbreviations",
            "Test solar_array.py degradation model thoroughly",
            "Refactor state_io to use atomic writes everywhere",
            "Deploy the fix discussed in #12503 now",
            "Wire up the CI pipeline for auth.py module",
            "Configure $GITHUB_TOKEN for API access setup",
            "Refactor ACTION_VERBS constant set definition",
            'Implement "circuit breaker" pattern in handler',
            "Add --verbose flag to the CLI output now",
        ]
        for p in proposals:
            r = _v(p)
            assert r["passed"], f"Expected pass: {p!r}"
            assert r["score"] > 0

    def test_10_rejections(self):
        rejects = [
            "", "x", "   ",
            "Make everything better and amazing",
            "The great system of wonders and mysteries",
            "1234567890123456789012345",
            "https://example.com/something",
            "TODO: fix the thing later",
            "`something here is broken and weird`",
            "Don't build auth.py ever again please",
        ]
        for p in rejects:
            r = _v(p)
            assert not r["passed"], f"Expected reject: {p!r}"
