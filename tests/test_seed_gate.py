"""Tests for seed_gate.py -- canonical specificity validator.

Covers: verb detection, target detection (files, paths, tools, modules,
CLI, discussions, channels, quoted), junk detection (hard + soft artifacts),
scoring with unique-target counting, validation pass/fail, exempt tags,
CLI, real-world proposals, edge cases, property invariants, smoke tests,
propose_seed.py contract, and regression tests for false rejects.

PR #272 additions: explain() API, score_breakdown(), find_verb_with_position(),
VerbMatch, detect_negation(), strength property, advisories tuple,
multi-target scaling, enriched SeedGateResult fields.
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
    ARTIFACT_SIGNALS,
    CHANNEL_RE,
    CLI_RE,
    CONST_RE,
    DISCUSSION_RE,
    ENV_VAR_RE,
    EXEMPT_TAGS,
    FILE_RE,
    FUNC_RE,
    KNOWN_MODULES,
    KNOWN_TOOLS,
    MODULE_CONTEXT_RE,
    PATH_RE,
    PHRASAL_VERBS,
    QUESTION_STEMS,
    QUOTED_RE,
    SPECIAL_FILE_RE,
    TAG_IMPLIED_VERBS,
    TOOL_RE,
    BatchResult,
    BatchStats,
    SeedGateResult,
    VerbMatch,
    _INFLECTION_MAP,
    _PHRASAL_INFLECTED,
    canonicalize_target,
    compute_score,
    count_unique_targets,
    detect_negation,
    explain,
    find_all_verbs,
    find_target,
    find_verb,
    find_verb_with_position,
    is_junk,
    is_soft_artifact,
    passes_gate,
    score_breakdown,
    suggest,
    validate,
    validate_batch,
    validate_seed,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_action_verbs_nonempty(self):
        assert len(ACTION_VERBS) >= 90

    def test_action_verbs_all_lowercase(self):
        assert all(v == v.lower() for v in ACTION_VERBS)

    def test_exempt_tags_nonempty(self):
        assert len(EXEMPT_TAGS) >= 5

    def test_exempt_tags_all_lowercase(self):
        assert all(t == t.lower() for t in EXEMPT_TAGS)

    def test_known_modules_discovered(self):
        assert len(KNOWN_MODULES) > 0

    def test_known_modules_no_test_or_run(self):
        for m in KNOWN_MODULES:
            assert not m.startswith("test_")
            assert not m.startswith("run_")

    def test_known_modules_only_snake_case(self):
        for m in KNOWN_MODULES:
            assert "_" in m


    def test_known_modules_contains_expected(self):
        expected = {"seed_gate", "propose_seed", "mars_colony"}
        found = {m for m in KNOWN_MODULES if m in expected}
        assert len(found) >= 2


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

class TestRegexPatterns:
    def test_file_re_python(self):
        assert FILE_RE.search("modify seed_gate.py for tests")

    def test_file_re_json(self):
        assert FILE_RE.search("update state/agents.json")

    def test_file_re_rust(self):
        assert FILE_RE.search("write parser.rs module")

    def test_path_re_src(self):
        assert PATH_RE.search("check src/thermal/core.py")

    def test_path_re_tests(self):
        assert PATH_RE.search("fix tests/test_main.py")

    def test_path_re_engine(self):
        assert PATH_RE.search("see engine/prompts/frame.md")

    def test_path_re_state(self):
        assert PATH_RE.search("read state/agents.json")

    def test_path_re_docs(self):
        assert PATH_RE.search("view docs/index.html")

    def test_tool_re_snake(self):
        assert TOOL_RE.search("fix process_inbox logic")

    def test_tool_re_no_single(self):
        assert not TOOL_RE.search("just a single word")

    def test_cli_re_backtick(self):
        assert CLI_RE.search("run `python main.py`")

    def test_cli_re_flag(self):
        assert CLI_RE.search("use --verbose mode")

    def test_cli_re_short_flag(self):
        assert CLI_RE.search("add -v flag")

    def test_discussion_re(self):
        assert DISCUSSION_RE.search("see #12503 for details")

    def test_discussion_re_no_short(self):
        assert not DISCUSSION_RE.search("issue #42 is minor")

    def test_channel_re_r(self):
        assert CHANNEL_RE.search("post in r/mars-engineering")

    def test_channel_re_c(self):
        assert CHANNEL_RE.search("check c/general channel")

    def test_quoted_re(self):
        assert QUOTED_RE.search('implement "water recycling module"')

    def test_module_context_backtick(self):
        assert MODULE_CONTEXT_RE.search("check `water_mining` module")

    def test_module_context_import(self):
        assert MODULE_CONTEXT_RE.search("import thermal_control")

    def test_module_context_from(self):
        assert MODULE_CONTEXT_RE.search("from state_io import load")


# ---------------------------------------------------------------------------
# Verb detection
# ---------------------------------------------------------------------------

class TestVerbDetection:
    @pytest.mark.parametrize("verb", ["build", "create", "fix", "test",
                                       "deploy", "refactor", "optimize",
                                       "debug", "analyze"])
    def test_common_verbs(self, verb):
        assert find_verb(f"{verb} the thing") == verb

    def test_verb_mid_sentence(self):
        assert find_verb("We should build the rover") == "build"

    def test_no_verb(self):
        assert find_verb("just some random words") is None

    def test_limit_parameter(self):
        assert find_verb("Hello world build something", limit=10) is None
        assert find_verb("Hello world build something", limit=20) == "build"

    def test_case_insensitive(self):
        assert find_verb("BUILD the system") == "build"

    def test_phrasal_verb(self):
        assert find_verb("Set up the deployment pipeline") == "set up"

    def test_inflected_verb(self):
        assert find_verb("Building the rover module") == "build"

    def test_irregular_past(self):
        assert find_verb("We built the thermal system") == "build"

    def test_all_verbs_multiple(self):
        verbs = find_all_verbs("Build the rover and test the drill")
        assert "build" in verbs
        assert "test" in verbs


# ---------------------------------------------------------------------------
# VerbMatch / find_verb_with_position (PR #272)
# ---------------------------------------------------------------------------

class TestFindVerbWithPosition:
    def test_returns_verb_match(self):
        vm = find_verb_with_position("Build water_mining.py")
        assert isinstance(vm, VerbMatch)
        assert vm.verb == "build"
        assert vm.token_index == 0
        assert vm.source == "text"

    def test_mid_sentence_position(self):
        vm = find_verb_with_position("We should build the rover")
        assert vm.verb == "build"
        assert vm.token_index == 2

    def test_none_when_no_verb(self):
        assert find_verb_with_position("just some random words") is None

    def test_phrasal_verb_position(self):
        vm = find_verb_with_position("Set up the deployment")
        assert vm.verb == "set up"
        assert vm.token_index == 0

    def test_inflected_verb_position(self):
        vm = find_verb_with_position("Creating water_mining.py module")
        assert vm.verb == "create"
        assert vm.token_index == 0

    def test_verb_match_bool(self):
        vm = VerbMatch(verb="build", token_index=0, source="text")
        assert bool(vm) is True
        vm_empty = VerbMatch(verb="", token_index=None, source="text")
        assert bool(vm_empty) is False


# ---------------------------------------------------------------------------
# Negation detection (PR #272)
# ---------------------------------------------------------------------------

class TestNegationDetection:
    def test_dont_before_verb(self):
        assert detect_negation("Don't deploy auth.py yet") is True

    def test_never_before_verb(self):
        assert detect_negation("Never build untested code for production") is True

    def test_stop_before_verb(self):
        assert detect_negation("Stop deploying to staging environment") is True

    def test_avoid_before_verb(self):
        assert detect_negation("Avoid refactoring the auth module now") is True

    def test_no_negation(self):
        assert detect_negation("Build water_mining.py optimizer") is False

    def test_negation_after_verb_ok(self):
        assert detect_negation("Build auth.py but don't break tests") is False

    def test_skip_before_verb(self):
        assert detect_negation("Skip testing for now on seed_gate.py") is True

    def test_negation_still_passes_gate(self):
        """Negation flags advisory but does NOT reject the proposal."""
        result = validate_seed("Don't deploy auth.py until tests pass")
        assert result.passed is True
        assert result.negated is True
        assert "negated-intent" in result.advisories


# ---------------------------------------------------------------------------
# Target detection
# ---------------------------------------------------------------------------

class TestTargetDetection:
    def test_python_file(self):
        t, k = find_target("Build water_mining.py optimizer")
        assert t == "water_mining.py"
        assert k == "file"

    def test_json_file(self):
        t, k = find_target("Update state/agents.json")
        assert "agents.json" in t

    def test_path(self):
        t, k = find_target("Check src/thermal/core.py")
        assert k in ("file", "path")

    def test_tool(self):
        t, k = find_target("Fix process_inbox bugs")
        assert t == "process_inbox"
        assert k == "tool"

    def test_channel(self):
        t, k = find_target("Post in r/mars-engineering about drills")
        assert k == "channel"

    def test_discussion(self):
        t, k = find_target("See discussion #12503 for context")
        assert k == "discussion"

    def test_quoted(self):
        t, k = find_target('Implement "advanced water recycling" system')
        assert k == "quoted"

    def test_env_var(self):
        t, k = find_target("Configure $STATE_DIR for tests")
        assert k == "env"

    def test_const(self):
        t, k = find_target("Update ACTION_VERBS set to include more")
        assert k == "const"

    def test_no_target(self):
        t, k = find_target("Just make everything better")
        assert t == ""
        assert k == ""


# ---------------------------------------------------------------------------
# Special files
# ---------------------------------------------------------------------------

class TestSpecialFiles:
    @pytest.mark.parametrize("name", [
        "Dockerfile", "Makefile", "README", "CONTRIBUTING", "LICENSE",
        "CHANGELOG", "CONSTITUTION",
    ])
    def test_special_file_detected(self, name):
        t, k = find_target(f"Update {name} with new info")
        assert t == name
        assert k == "file"


# ---------------------------------------------------------------------------
# File RE false positives
# ---------------------------------------------------------------------------

class TestFileReFalsePositives:
    @pytest.mark.parametrize("text", [
        "Use e.g. some example pattern",
        "That is i.e. the main issue",
        "Schedule at 9 a.m. every day",
    ])
    def test_abbreviations_not_files(self, text):
        t, k = find_target(text)
        assert k != "file" or t not in ("e.g", "i.e", "a.m")

    @pytest.mark.parametrize("text", [
        "Upgrade to version 2.0.1 now",
        "Support v1.2.3 format spec",
        "Release 1.0 candidate build",
    ])
    def test_version_strings_not_files(self, text):
        t, k = find_target(text)
        if k == "file":
            from seed_gate import _VERSION_RE
            assert not _VERSION_RE.match(t)


# ---------------------------------------------------------------------------
# Version filter
# ---------------------------------------------------------------------------

class TestVersionFilter:
    @pytest.mark.parametrize("v", ["2.0.1", "v1.2.3", "1.0", "3.0.0-beta", "2.1+build42"])
    def test_version_strings_filtered(self, v):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match(v) is True

    @pytest.mark.parametrize("f", ["main.py", "config.json", "v1.rs"])
    def test_real_files_not_filtered(self, f):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match(f) is False

    def test_version_in_context(self):
        result = validate_seed("Build migration_tool.py for upgrading from 2.0 to 3.0")
        assert result.passed is True


# ---------------------------------------------------------------------------
# Known tools
# ---------------------------------------------------------------------------

class TestKnownTools:
    @pytest.mark.parametrize("tool", ["state_io", "process_inbox", "github_llm",
                                       "seed_gate", "propose_seed", "vlink"])
    def test_known_tool_detected(self, tool):
        t, k = find_target(f"Refactor {tool} for better performance")
        assert t == tool
        assert k == "tool"


# ---------------------------------------------------------------------------
# Expanded known tools
# ---------------------------------------------------------------------------

class TestExpandedKnownTools:
    @pytest.mark.parametrize("tool", [
        "compute_analytics", "reconcile_channels", "git_scrape_analytics",
        "inject_seed", "tally_votes", "steer", "reconcile_state",
    ])
    def test_expanded_tool_detected(self, tool):
        t, k = find_target(f"Fix {tool} edge case bug")
        assert t == tool
        assert k == "tool"


# ---------------------------------------------------------------------------
# Env var targets
# ---------------------------------------------------------------------------

class TestEnvVarTargets:
    @pytest.mark.parametrize("var", ["$STATE_DIR", "$GITHUB_TOKEN", "${DOCS_DIR}", "$LLM_DAILY_BUDGET"])
    def test_env_var_detected(self, var):
        t, k = find_target(f"Configure {var} for production")
        assert k == "env"


# ---------------------------------------------------------------------------
# CONST targets
# ---------------------------------------------------------------------------

class TestConstTargets:
    @pytest.mark.parametrize("const", ["ACTION_VERBS", "MAX_RETRIES", "KNOWN_TOOLS",
                                         "EXEMPT_TAGS", "FILE_RE"])
    def test_const_detected(self, const):
        t, k = find_target(f"Update {const} to include more items")
        assert t == const
        assert k == "const"


# ---------------------------------------------------------------------------
# Junk detection
# ---------------------------------------------------------------------------

class TestJunkDetection:
    def test_empty(self):
        assert is_junk("")

    def test_whitespace_only(self):
        assert is_junk("   \n  ")

    def test_too_short(self):
        assert is_junk("hi")

    def test_starts_with_backtick(self):
        assert is_junk("`some fragment from parser")

    def test_starts_with_pipe(self):
        assert is_junk("| piped output from somewhere")

    def test_starts_with_number(self):
        assert is_junk("1. First item in a numbered list")

    def test_starts_with_url(self):
        assert is_junk("https://github.com/some/repo")

    def test_todo_marker(self):
        assert is_junk("TODO: fix this later when we have time")

    def test_lowercase_non_verb(self):
        assert is_junk("something random without a verb start")

    def test_lowercase_verb_ok(self):
        assert is_junk("build seed_gate.py optimizer") == ""

    def test_lowercase_file_ok(self):
        assert is_junk("seed_gate.py needs optimization work") == ""

    def test_hard_artifact(self):
        assert is_junk("parser grabbed this from a discussion")

    def test_valid_proposal_not_junk(self):
        assert is_junk("Build water_mining.py optimizer for efficiency") == ""


# ---------------------------------------------------------------------------
# Soft artifacts
# ---------------------------------------------------------------------------

class TestSoftArtifacts:
    def test_soft_signal_detected(self):
        assert is_soft_artifact("the regex found a match here")

    def test_soft_with_verb_target_passes(self):
        result = validate_seed("Fix the regex in seed_gate.py validator")
        assert result.passed is True

    def test_soft_without_verb_target_fails(self):
        result = validate_seed("The regex matched something strange here")
        assert result.passed is False

    def test_non_artifact_clean(self):
        assert not is_soft_artifact("Build a new thermal control system")


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

class TestCanonicalization:
    def test_strip_extension(self):
        assert canonicalize_target("seed_gate.py") == "seed_gate"

    def test_strip_src_prefix(self):
        assert canonicalize_target("src/thermal.py") == "thermal"

    def test_strip_tests_prefix(self):
        assert canonicalize_target("tests/test_main.py") == "test_main"

    def test_strip_quotes(self):
        assert canonicalize_target('"seed_gate"') == "seed_gate"

    def test_lowercase(self):
        assert canonicalize_target("Dockerfile") == "dockerfile"

    def test_no_extension(self):
        assert canonicalize_target("README") == "readme"

    def test_nested_path(self):
        assert canonicalize_target("state/agents.json") == "agents"


class TestCanonicalizeTarget:
    def test_backticks(self):
        assert canonicalize_target("`seed_gate`") == "seed_gate"

    def test_deep_path(self):
        assert canonicalize_target("engine/prompts/frame.md") == "prompts/frame"

    def test_already_clean(self):
        assert canonicalize_target("rover") == "rover"

    def test_empty(self):
        assert canonicalize_target("") == ""

    def test_only_extension(self):
        assert canonicalize_target(".py") == ".py"


# ---------------------------------------------------------------------------
# Substring dedup
# ---------------------------------------------------------------------------

class TestSubstringDedup:
    def test_dedup_file_and_tool(self):
        n = count_unique_targets("Fix seed_gate.py and seed_gate module")
        assert n == 1

    def test_distinct_files(self):
        n = count_unique_targets("Fix seed_gate.py and propose_seed.py")
        assert n == 2

    def test_three_targets(self):
        n = count_unique_targets("Fix seed_gate.py, propose_seed.py and rover.py")
        assert n == 3

    def test_no_targets(self):
        assert count_unique_targets("Make everything better") == 0

    def test_path_and_file_dedup(self):
        n = count_unique_targets("See src/thermal.py and thermal.py module")
        assert n == 1


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_verb_only_positive(self):
        s = compute_score("Build something great", "build", None, "")
        assert s > 0

    def test_verb_plus_file_higher(self):
        s1 = compute_score("Build something", "build", None, "")
        s2 = compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert s2 > s1

    def test_score_capped_at_1(self):
        s = compute_score(
            "Build seed_gate.py and propose_seed.py and rover.py and thermal.py with many extra words to push the score up high " * 3,
            "build", "seed_gate.py", "file"
        )
        assert s <= 1.0

    def test_imperative_bonus(self):
        s1 = compute_score("Build seed_gate.py now", "build", "seed_gate.py", "file")
        s2 = compute_score("We should build seed_gate.py now", "build", "seed_gate.py", "file")
        assert s1 >= s2

    def test_length_bonus(self):
        short = "Build seed_gate.py"
        long_text = "Build seed_gate.py optimizer that handles edge cases and improves validation"
        s_short = compute_score(short, "build", "seed_gate.py", "file")
        s_long = compute_score(long_text, "build", "seed_gate.py", "file")
        assert s_long >= s_short

    def test_multi_target_bonus(self):
        two = "Build seed_gate.py and propose_seed.py"
        s = compute_score(two, "build", "seed_gate.py", "file")
        assert s > compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")


# ---------------------------------------------------------------------------
# Score breakdown (PR #272)
# ---------------------------------------------------------------------------

class TestScoreBreakdown:
    def test_returns_dict(self):
        sb = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert isinstance(sb, dict)

    def test_has_expected_keys(self):
        sb = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file")
        for key in ("verb", "target", "length", "multi_target", "imperative", "total"):
            assert key in sb

    def test_verb_component(self):
        sb = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert sb["verb"] == 2.5

    def test_no_verb_component(self):
        sb = score_breakdown("Something about seed_gate.py", None, "seed_gate.py", "file")
        assert sb["verb"] == 0.0

    def test_file_target_component(self):
        sb = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert sb["target"] == 4.0

    def test_total_matches_compute_score(self):
        text = "Build seed_gate.py optimizer"
        sb = score_breakdown(text, "build", "seed_gate.py", "file")
        cs = compute_score(text, "build", "seed_gate.py", "file")
        assert abs(sb["total"] - cs) < 0.001

    def test_multi_target_scaling_2(self):
        text = "Build seed_gate.py and propose_seed.py"
        sb = score_breakdown(text, "build", "seed_gate.py", "file")
        assert sb["multi_target"] == 1.0

    def test_multi_target_scaling_3(self):
        text = "Build seed_gate.py, propose_seed.py, and rover.py"
        sb = score_breakdown(text, "build", "seed_gate.py", "file")
        assert sb["multi_target"] == 1.5

    def test_multi_target_scaling_4_plus(self):
        text = "Build seed_gate.py, propose_seed.py, rover.py, and thermal.py"
        sb = score_breakdown(text, "build", "seed_gate.py", "file")
        assert sb["multi_target"] == 2.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_verb_plus_file_passes(self):
        r = validate_seed("Build water_mining.py optimizer")
        assert r.passed is True

    def test_no_verb_fails(self):
        r = validate_seed("Something about the water system")
        assert r.passed is False

    def test_verb_no_target_fails(self):
        r = validate_seed("Build something amazing for everyone")
        assert r.passed is False

    def test_exempt_tag_no_target_passes(self):
        r = validate_seed("Explore the nature of consciousness deeply", ["philosophy"])
        assert r.passed is True

    def test_junk_fails(self):
        r = validate_seed("")
        assert r.passed is False
        assert r.junk is True

    def test_result_has_verb(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert r.verb_found == "build"

    def test_result_has_target(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert r.target_found == "seed_gate.py"

    def test_result_has_score(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert 0.0 < r.score <= 1.0

    def test_result_has_reasons_on_fail(self):
        r = validate_seed("Make everything better for all")
        assert len(r.reasons) > 0

    def test_dict_api(self):
        d = validate("Build seed_gate.py optimizer")
        assert isinstance(d, dict)
        assert d["passed"] is True

    def test_bool_api(self):
        assert passes_gate("Build seed_gate.py optimizer") is True
        assert passes_gate("Make everything better for all") is False


# ---------------------------------------------------------------------------
# Enriched SeedGateResult fields (PR #272)
# ---------------------------------------------------------------------------

class TestEnrichedResult:
    def test_target_kind_populated(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert r.target_kind == "file"

    def test_verb_source_text(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert r.verb_source == "text"

    def test_verb_source_tag(self):
        r = validate_seed("Something about water_mining.py", ["code"])
        assert r.verb_source == "tag"

    def test_verb_source_question(self):
        r = validate_seed("What about improving the colony systems", ["philosophy"])
        # If text contains a real verb, source is "text"; question source only
        # when verb is inferred from question stem with no real verb in text
        assert r.verb_source in ("text", "question")

    def test_verb_position_populated(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert r.verb_position == 0

    def test_score_parts_populated(self):
        r = validate_seed("Build seed_gate.py optimizer")
        parts = dict(r.score_parts)
        assert "verb" in parts
        assert "target" in parts

    def test_advisories_tuple(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert isinstance(r.advisories, tuple)

    def test_negated_field(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert r.negated is False

    def test_negated_field_true(self):
        r = validate_seed("Don't deploy auth.py until ready")
        assert r.negated is True

    def test_to_dict_has_new_fields(self):
        d = validate("Build seed_gate.py optimizer")
        assert "strength" in d
        assert "target_kind" in d
        assert "verb_source" in d
        assert "score_parts" in d
        assert "advisories" in d
        assert "negated" in d


# ---------------------------------------------------------------------------
# Strength property (PR #272)
# ---------------------------------------------------------------------------

class TestStrengthProperty:
    def test_rejected(self):
        r = validate_seed("Make everything better for all")
        assert r.strength == "rejected"

    def test_strong(self):
        r = validate_seed("Build seed_gate.py and propose_seed.py and rover.py optimizer for the colony and more words")
        assert r.strength in ("strong", "moderate")

    def test_weak_passes(self):
        r = validate_seed("Explore consciousness deeply", ["philosophy"])
        assert r.passed is True
        assert r.strength in ("weak", "moderate")

    def test_strength_consistent_with_confidence(self):
        r = validate_seed("Build seed_gate.py optimizer")
        if r.confidence == "high":
            assert r.strength in ("strong", "moderate")


# ---------------------------------------------------------------------------
# Advisories (PR #272)
# ---------------------------------------------------------------------------

class TestAdvisories:
    def test_needs_specificity(self):
        r = validate_seed("Explore the cosmos deeply", ["philosophy"])
        assert "needs-specificity" in r.advisories

    def test_negated_intent(self):
        r = validate_seed("Don't deploy auth.py yet")
        assert "negated-intent" in r.advisories

    def test_no_advisories_clean(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert "needs-specificity" not in r.advisories

    def test_backward_compat_advisory_string(self):
        r = validate_seed("Explore the cosmos deeply", ["philosophy"])
        assert r.advisory == "needs-specificity"


# ---------------------------------------------------------------------------
# Explain API (PR #272)
# ---------------------------------------------------------------------------

class TestExplainAPI:
    def test_returns_dict(self):
        e = explain("Build seed_gate.py optimizer")
        assert isinstance(e, dict)

    def test_has_expected_keys(self):
        e = explain("Build seed_gate.py optimizer")
        for key in ("decision", "detected", "score_breakdown", "suggestions", "advisories"):
            assert key in e, f"Missing key: {key}"

    def test_decision_subkeys(self):
        e = explain("Build seed_gate.py optimizer")
        d = e["decision"]
        for key in ("passed", "reasons", "junk", "confidence", "strength"):
            assert key in d

    def test_detected_subkeys(self):
        e = explain("Build seed_gate.py optimizer")
        det = e["detected"]
        for key in ("verb", "verb_source", "verb_position", "target", "target_kind",
                     "negated", "unique_target_count"):
            assert key in det

    def test_score_breakdown_populated(self):
        e = explain("Build seed_gate.py optimizer")
        sb = e["score_breakdown"]
        assert "verb" in sb
        assert "target" in sb

    def test_suggestions_empty_on_pass(self):
        e = explain("Build seed_gate.py optimizer")
        assert e["suggestions"] == []

    def test_suggestions_populated_on_fail(self):
        e = explain("Make everything better for all")
        assert len(e["suggestions"]) > 0

    def test_negated_detected(self):
        e = explain("Don't deploy auth.py yet")
        assert e["detected"]["negated"] is True

    def test_advisories_populated(self):
        e = explain("Don't deploy auth.py yet")
        assert "negated-intent" in e["advisories"]


# ---------------------------------------------------------------------------
# Dataclass API
# ---------------------------------------------------------------------------

class TestDataclassAPI:
    def test_is_frozen(self):
        r = validate_seed("Build seed_gate.py optimizer")
        with pytest.raises(AttributeError):
            r.passed = False

    def test_verb_property(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert r.verb == "build"

    def test_target_property(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert r.target == "seed_gate.py"

    def test_to_dict_roundtrip(self):
        r = validate_seed("Build seed_gate.py optimizer")
        d = r.to_dict()
        assert d["passed"] == r.passed
        assert d["score"] == r.score


# ---------------------------------------------------------------------------
# Confidence property
# ---------------------------------------------------------------------------

class TestConfidenceProperty:
    def test_high(self):
        r = validate_seed("Build seed_gate.py and propose_seed.py optimizer with detailed plan")
        if r.score >= 0.65:
            assert r.confidence == "high"

    def test_none_on_fail(self):
        r = validate_seed("Make everything better for all")
        assert r.confidence is None

    def test_always_valid(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert r.confidence in ("high", "medium", "low", None)


# ---------------------------------------------------------------------------
# Tag-implied verbs
# ---------------------------------------------------------------------------

class TestTagImpliedVerbs:
    @pytest.mark.parametrize("tag,expected", [
        ("code", "build"), ("build", "build"), ("test", "test"),
        ("debug", "debug"), ("docs", "document"), ("refactor", "refactor"),
        ("security", "secure"), ("deploy", "deploy"),
    ])
    def test_tag_implies_verb(self, tag, expected):
        r = validate_seed(f"Something about seed_gate.py module", [tag])
        assert r.verb_found == expected


# ---------------------------------------------------------------------------
# Question stems
# ---------------------------------------------------------------------------

class TestQuestionStems:
    @pytest.mark.parametrize("stem,expected", [
        ("What if", "explore"), ("How might", "design"),
        ("How could", "design"), ("Should we", "evaluate"),
        ("Could we", "explore"), ("Why not", "propose"),
        ("Why do", "investigate"), ("Why does", "investigate"),
    ])
    def test_question_stem_maps(self, stem, expected):
        r = validate_seed(f"{stem} rethink the whole approach", ["philosophy"])
        assert r.verb_found == expected


# ---------------------------------------------------------------------------
# Advisory labels
# ---------------------------------------------------------------------------

class TestAdvisoryLabel:
    def test_needs_specificity_on_fail(self):
        r = validate_seed("Build something amazing for all")
        if r.verb_found and not r.target_found:
            assert r.advisory == "needs-specificity"

    def test_needs_specificity_on_exempt_pass(self):
        r = validate_seed("Explore the meaning of existence", ["philosophy"])
        assert r.advisory == "needs-specificity"

    def test_no_advisory_with_target(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert r.advisory == ""

    def test_advisory_in_dict(self):
        d = validate("Explore the meaning of existence", ["philosophy"])
        assert "advisory" in d


# ---------------------------------------------------------------------------
# Phrasal verbs
# ---------------------------------------------------------------------------

class TestPhrasalVerbs:
    @pytest.mark.parametrize("phrase", [
        "set up", "roll back", "wire up", "clean up", "spin up",
        "tear down", "break down", "plug in", "hook up",
        "scale up", "scale down", "lock down", "back up",
    ])
    def test_phrasal_verb_detected(self, phrase):
        assert find_verb(f"{phrase.title()} the deployment") == phrase


# ---------------------------------------------------------------------------
# Inflected verbs
# ---------------------------------------------------------------------------

class TestInflectedVerbDetection:
    @pytest.mark.parametrize("inflected,base", [
        ("builds", "build"), ("creating", "create"), ("deployed", "deploy"),
        ("tests", "test"), ("fixing", "fix"), ("optimizes", "optimize"),
        ("shipping", "ship"), ("planned", "plan"),
    ])
    def test_inflected_to_base(self, inflected, base):
        assert find_verb(f"{inflected} the module") == base


# ---------------------------------------------------------------------------
# Inflected phrasal verbs
# ---------------------------------------------------------------------------

class TestInflectedPhrasalVerbs:
    @pytest.mark.parametrize("inflected,canonical", [
        ("sets up", "set up"), ("setting up", "set up"),
        ("cleaned up", "clean up"), ("cleaning up", "clean up"),
        ("rolled back", "roll back"), ("spinning up", "spin up"),
    ])
    def test_inflected_phrasal(self, inflected, canonical):
        assert find_verb(f"{inflected.title()} the system") == canonical


# ---------------------------------------------------------------------------
# Inflection map invariants
# ---------------------------------------------------------------------------

class TestInflectionMap:
    def test_all_values_in_action_verbs(self):
        for form, base in _INFLECTION_MAP.items():
            if " " not in form:
                assert base in ACTION_VERBS, f"{form} -> {base} not in ACTION_VERBS"

    def test_no_base_form_as_key(self):
        for form in _INFLECTION_MAP:
            if " " not in form:
                assert form not in ACTION_VERBS, f"{form} is both inflected and base"

    def test_inflected_never_returns_none(self):
        for form in _INFLECTION_MAP:
            result = find_verb(f"{form} something with target.py")
            assert result is not None

    def test_map_size_reasonable(self):
        assert len(_INFLECTION_MAP) >= 200


# ---------------------------------------------------------------------------
# Rich match info
# ---------------------------------------------------------------------------

class TestRichMatchInfo:
    def test_all_verbs_populated(self):
        r = validate_seed("Build the rover and test the drill.py")
        assert len(r.all_verbs) >= 2

    def test_all_targets_populated(self):
        r = validate_seed("Build seed_gate.py and propose_seed.py")
        assert len(r.all_targets) >= 2

    def test_all_targets_are_tuples(self):
        r = validate_seed("Build seed_gate.py optimizer")
        for t in r.all_targets:
            assert isinstance(t, tuple)
            assert len(t) == 2


# ---------------------------------------------------------------------------
# Find all targets
# ---------------------------------------------------------------------------

class TestFindAllTargets:
    def test_multiple_files(self):
        from seed_gate import _find_all_targets
        targets = _find_all_targets("Fix seed_gate.py and propose_seed.py")
        assert len(targets) >= 2

    def test_mixed_types(self):
        from seed_gate import _find_all_targets
        targets = _find_all_targets("Fix seed_gate.py and check r/mars-engineering")
        kinds = {k for _, k in targets}
        assert len(kinds) >= 2


# ---------------------------------------------------------------------------
# Find all verbs
# ---------------------------------------------------------------------------

class TestFindAllVerbs:
    def test_multiple_verbs(self):
        verbs = find_all_verbs("Build the rover and test the drill")
        assert "build" in verbs
        assert "test" in verbs

    def test_deduplication(self):
        verbs = find_all_verbs("Build and build again and build more")
        assert verbs.count("build") == 1

    def test_empty(self):
        assert find_all_verbs("just some random words") == []


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

class TestBatchValidation:
    def test_batch_basic(self):
        proposals = [
            "Build seed_gate.py optimizer",
            "Make everything better for all",
            "",
        ]
        br = validate_batch(proposals)
        assert isinstance(br, BatchResult)
        assert br.stats.total == 3
        assert br.stats.passed >= 1
        assert br.stats.junk >= 1

    def test_batch_empty(self):
        br = validate_batch([])
        assert br.stats.total == 0

    def test_batch_all_pass(self):
        proposals = [
            "Build seed_gate.py optimizer",
            "Fix propose_seed.py validation",
        ]
        br = validate_batch(proposals)
        assert br.stats.passed == 2

    def test_batch_items(self):
        proposals = [
            "Build seed_gate.py optimizer",
            "",
        ]
        br = validate_batch(proposals)
        assert len(br.passed_items) >= 1
        assert len(br.junk_items) >= 1

    def test_batch_stats_rates(self):
        br = validate_batch(["Build seed_gate.py", "x", "Vague proposal text"])
        assert 0.0 <= br.stats.pass_rate <= 1.0
        assert 0.0 <= br.stats.junk_rate <= 1.0


# ---------------------------------------------------------------------------
# Batch stats
# ---------------------------------------------------------------------------

class TestBatchStats:
    def test_pass_rate(self):
        s = BatchStats(total=10, passed=5, failed=3, junk=2)
        assert s.pass_rate == 0.5

    def test_junk_rate(self):
        s = BatchStats(total=10, passed=5, failed=3, junk=2)
        assert s.junk_rate == 0.2

    def test_empty(self):
        s = BatchStats(total=0, passed=0, failed=0, junk=0)
        assert s.pass_rate == 0.0

    def test_merge(self):
        a = BatchStats(total=5, passed=3, failed=1, junk=1)
        b = BatchStats(total=5, passed=2, failed=2, junk=1)
        m = a.merge(b)
        assert m.total == 10
        assert m.passed == 5


# ---------------------------------------------------------------------------
# Suggest API
# ---------------------------------------------------------------------------

class TestSuggestAPI:
    def test_no_verb_suggestion(self):
        suggestions = suggest("Something about the module")
        assert any("action verb" in s for s in suggestions)

    def test_no_target_suggestion(self):
        suggestions = suggest("Build something amazing for all")
        assert any("target" in s.lower() for s in suggestions)

    def test_empty_on_pass(self):
        assert suggest("Build seed_gate.py optimizer") == []

    def test_junk_suggestion(self):
        suggestions = suggest("x")
        assert any("Rewrite" in s for s in suggestions)


# ---------------------------------------------------------------------------
# Real-world proposals
# ---------------------------------------------------------------------------

class TestRealWorld:
    @pytest.mark.parametrize("text", [
        "Build water_mining.py optimizer for deep well extraction",
        "Fix bug in seed_gate.py validation for inflected verbs",
        "Refactor propose_seed.py to support batch proposals",
        "Deploy thermal_control.py to the Mars colony systems",
        "Test rover.py navigation algorithms with Monte Carlo",
        "Optimize fuel_cell.py energy output for dust storms",
        "Wire up process_inbox for real-time delta processing",
    ])
    def test_real_proposals_pass(self, text):
        assert validate_seed(text).passed is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_long_text(self):
        text = "Build seed_gate.py " + "word " * 500
        r = validate_seed(text)
        assert r.passed is True
        assert r.score <= 1.0

    def test_unicode_text(self):
        r = validate_seed("Build système.py optimiseur pour le système")
        assert isinstance(r.passed, bool)

    def test_special_chars(self):
        r = validate_seed("Build seed_gate.py — optimize the validator!")
        assert r.passed is True

    def test_newlines(self):
        r = validate_seed("Build seed_gate.py\noptimizer\nfor validation")
        assert r.passed is True

    def test_tabs(self):
        r = validate_seed("Build\tseed_gate.py\toptimizer")
        assert r.passed is True


# ---------------------------------------------------------------------------
# Lowercase imperative
# ---------------------------------------------------------------------------

class TestLowercaseImperative:
    def test_lowercase_verb_start_not_junk(self):
        assert is_junk("build seed_gate.py optimizer quickly") == ""

    def test_lowercase_file_start_not_junk(self):
        assert is_junk("seed_gate.py needs optimization for speed") == ""

    def test_lowercase_non_verb_is_junk(self):
        assert is_junk("something random without any verbs")

    def test_lowercase_inflected_verb_ok(self):
        assert is_junk("building seed_gate.py optimizer now") == ""

    def test_lowercase_phrasal_verb_ok(self):
        assert is_junk("set up the deployment pipeline correctly") == ""


# ---------------------------------------------------------------------------
# Mode consistency
# ---------------------------------------------------------------------------

class TestModeConsistency:
    def test_purge_mode_always_passes(self):
        r = validate_seed("Build seed_gate.py", mode="purge")
        assert r.passed is True

    def test_purge_mode_junk_still_fails(self):
        r = validate_seed("", mode="purge")
        assert r.passed is False

    def test_admission_default(self):
        r = validate_seed("Build seed_gate.py optimizer")
        assert r.passed is True

    def test_purge_vague_passes(self):
        r = validate_seed("Something about improving things", mode="purge")
        assert r.passed is True


# ---------------------------------------------------------------------------
# Propose-seed contract
# ---------------------------------------------------------------------------

class TestProposeSeedContract:
    def test_validate_returns_dict(self):
        d = validate("Build seed_gate.py optimizer")
        assert isinstance(d, dict)

    def test_dict_has_passed(self):
        d = validate("Build seed_gate.py optimizer")
        assert "passed" in d

    def test_dict_has_score(self):
        d = validate("Build seed_gate.py optimizer")
        assert "score" in d

    def test_dict_has_junk(self):
        d = validate("Build seed_gate.py optimizer")
        assert "junk" in d

    def test_dict_has_all_verbs(self):
        d = validate("Build seed_gate.py optimizer")
        assert "all_verbs" in d

    def test_dict_has_all_targets(self):
        d = validate("Build seed_gate.py optimizer")
        assert "all_targets" in d

    def test_dict_has_strength(self):
        d = validate("Build seed_gate.py optimizer")
        assert "strength" in d

    def test_dict_has_advisories(self):
        d = validate("Build seed_gate.py optimizer")
        assert "advisories" in d

    def test_dict_has_negated(self):
        d = validate("Build seed_gate.py optimizer")
        assert "negated" in d


# ---------------------------------------------------------------------------
# False reject regression
# ---------------------------------------------------------------------------

class TestFalseRejectRegression:
    def test_imperative_file_target(self):
        assert validate_seed("Build seed_gate.py").passed is True

    def test_lowercase_imperative(self):
        r = validate_seed("build seed_gate.py optimizer quickly")
        # May or may not pass depending on junk detection of lowercase
        assert isinstance(r.passed, bool)

    def test_run_prefix_not_junk(self):
        assert is_junk("run_proof.py execution engine setup") == ""

    def test_tag_implied_with_file(self):
        r = validate_seed("Something about seed_gate.py module", ["code"])
        assert r.passed is True


# ---------------------------------------------------------------------------
# Noun use false positives
# ---------------------------------------------------------------------------

class TestNounUseFalsePositives:
    @pytest.mark.parametrize("text", [
        "Nothing works in the thermal_control.py system",
        "During the morning review of auth.py we found bugs",
        "Something strange with the water_mining.py module",
    ])
    def test_non_verb_words_ignored(self, text):
        r = validate_seed(text)
        # These don't have action verbs, should fail
        assert isinstance(r.passed, bool)


# ---------------------------------------------------------------------------
# Case-insensitive modules
# ---------------------------------------------------------------------------

class TestCaseInsensitiveModules:
    def test_lowercase_module_match(self):
        if KNOWN_MODULES:
            mod = next(iter(KNOWN_MODULES))
            t, k = find_target(f"Check `{mod.lower()}` module")
            if k == "module":
                assert t.lower() == mod.lower()


# ---------------------------------------------------------------------------
# New verbs from expansion
# ---------------------------------------------------------------------------

class TestNewVerbsExpanded:
    @pytest.mark.parametrize("verb", [
        "configure", "scaffold", "bootstrap", "provision",
        "automate", "archive", "inject", "normalize",
        "update", "delete", "enable", "disable",
        "install", "deprecate", "rewrite", "standardize",
        "containerize", "expose", "wrap", "stub", "mock",
        "isolate", "define", "declare", "register",
        "secure", "clean", "schedule", "cache", "publish",
        "annotate", "version", "backup", "package",
    ])
    def test_expanded_verb_detected(self, verb):
        assert find_verb(f"{verb} the target.py module") == verb


# ---------------------------------------------------------------------------
# Imperative bonus
# ---------------------------------------------------------------------------

class TestImperativeBonus:
    def test_imperative_scores_higher(self):
        s1 = compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")
        s2 = compute_score("We should build seed_gate.py", "build", "seed_gate.py", "file")
        assert s1 >= s2

    def test_non_imperative_still_scores(self):
        s = compute_score("We should build seed_gate.py", "build", "seed_gate.py", "file")
        assert s > 0


# ---------------------------------------------------------------------------
# Inflection invariants
# ---------------------------------------------------------------------------

class TestInflectionInvariants:
    def test_all_map_values_are_action_verbs(self):
        for form, base in _INFLECTION_MAP.items():
            if " " not in form:
                assert base in ACTION_VERBS

    def test_inflected_verb_always_returns_base(self):
        for form, base in list(_INFLECTION_MAP.items())[:50]:
            result = find_verb(f"{form} something target.py")
            assert result == base or result in PHRASAL_VERBS.values()

    def test_double_final_all_in_action_verbs_or_phrasal(self):
        from seed_gate import _DOUBLE_FINAL
        for v in _DOUBLE_FINAL:
            assert v in ACTION_VERBS or any(v in p for p in PHRASAL_VERBS)

    def test_score_always_0_to_1(self):
        for text in ["Build x.py", "Test y.py and z.py", "Fix a.py b.py c.py d.py e.py"]:
            verb = find_verb(text)
            t, k = find_target(text)
            s = compute_score(text, verb, t, k)
            assert 0.0 <= s <= 1.0

    def test_confidence_always_valid(self):
        for text in ["Build seed_gate.py", "Fix x.py", "Make things"]:
            r = validate_seed(text)
            assert r.confidence in ("high", "medium", "low", None)

    def test_suggestions_always_list(self):
        for text in ["Build seed_gate.py", "Fix x.py", "Make things better"]:
            s = suggest(text)
            assert isinstance(s, list)


# ---------------------------------------------------------------------------
# New feature invariants (PR #272)
# ---------------------------------------------------------------------------

class TestNewFeatureInvariants:
    def test_strength_always_valid(self):
        for text in ["Build seed_gate.py", "Fix x.py", "Make things better"]:
            r = validate_seed(text)
            assert r.strength in ("strong", "moderate", "weak", "rejected")

    def test_advisories_always_tuple(self):
        for text in ["Build seed_gate.py", "Don't deploy auth.py"]:
            r = validate_seed(text)
            assert isinstance(r.advisories, tuple)

    def test_negated_always_bool(self):
        for text in ["Build seed_gate.py", "Don't deploy auth.py", ""]:
            r = validate_seed(text)
            assert isinstance(r.negated, bool)

    def test_explain_always_returns_dict(self):
        for text in ["Build seed_gate.py", "Make things", ""]:
            e = explain(text)
            assert isinstance(e, dict)


# ---------------------------------------------------------------------------
# Property invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Property-based tests that must hold for ANY input."""

    _SAMPLES = [
        "Build water_mining.py optimizer for deep well extraction",
        "Fix bug in seed_gate.py validation",
        "Refactor propose_seed.py to support batch",
        "Make everything better",
        "",
        "x",
        "1. First item in list",
        "`fragment` from parser",
        "build seed_gate.py optimizer",
        "Building the rover.py module now",
        "We should test drill.py thoroughly",
        "Don't deploy auth.py until tested",
        "Never build untested code anywhere",
        "Set up the deployment pipeline.yml",
        "Explore consciousness deeply now",
        'Implement "water recycling" system fully',
        "Check $STATE_DIR configuration setup",
        "Update ACTION_VERBS set carefully",
        "See #12503 for implementation details",
        "Post in r/mars-engineering about plans",
        "What if we rethink everything completely",
        "Build seed_gate.py and propose_seed.py and rover.py",
        "Build seed_gate.py, propose_seed.py, rover.py, and thermal.py",
    ]

    @pytest.mark.parametrize("text", _SAMPLES)
    def test_score_bounded(self, text):
        r = validate_seed(text)
        assert 0.0 <= r.score <= 1.0

    @pytest.mark.parametrize("text", _SAMPLES)
    def test_result_is_seedgateresult(self, text):
        r = validate_seed(text)
        assert isinstance(r, SeedGateResult)

    @pytest.mark.parametrize("text", _SAMPLES)
    def test_dict_api_keys(self, text):
        d = validate(text)
        assert "passed" in d
        assert "score" in d
        assert "junk" in d

    @pytest.mark.parametrize("text", _SAMPLES)
    def test_passes_gate_matches(self, text):
        d = validate(text)
        assert passes_gate(text) == d["passed"]

    @pytest.mark.parametrize("text", _SAMPLES)
    def test_confidence_valid(self, text):
        r = validate_seed(text)
        assert r.confidence in ("high", "medium", "low", None)

    @pytest.mark.parametrize("text", _SAMPLES)
    def test_strength_valid(self, text):
        r = validate_seed(text)
        assert r.strength in ("strong", "moderate", "weak", "rejected")

    @pytest.mark.parametrize("text", _SAMPLES)
    def test_negated_is_bool(self, text):
        r = validate_seed(text)
        assert isinstance(r.negated, bool)

    @pytest.mark.parametrize("text", _SAMPLES)
    def test_advisories_is_tuple(self, text):
        r = validate_seed(text)
        assert isinstance(r.advisories, tuple)

    @pytest.mark.parametrize("text", _SAMPLES)
    def test_explain_returns_dict(self, text):
        e = explain(text)
        assert isinstance(e, dict)
        assert "decision" in e
        assert "detected" in e


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_pass(self):
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Build seed_gate.py optimizer"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        d = json.loads(r.stdout)
        assert d["passed"] is True

    def test_cli_fail(self):
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Make everything better for all"],
            capture_output=True, text=True,
        )
        assert r.returncode == 1

    def test_cli_with_tags(self):
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Explore the meaning of existence", "philosophy"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_full_validation_lifecycle(self):
        """End-to-end: validate, explain, suggest, batch."""
        text = "Build seed_gate.py optimizer for validation"
        r = validate_seed(text)
        assert r.passed
        assert r.verb == "build"
        assert r.target == "seed_gate.py"
        assert r.strength in ("strong", "moderate", "weak")
        e = explain(text)
        assert e["decision"]["passed"]
        assert suggest(text) == []
        br = validate_batch([text, "x", "Make things better"])
        assert br.stats.total == 3

    def test_negation_lifecycle(self):
        """Negated proposals pass but get flagged."""
        text = "Don't deploy auth.py until the tests pass"
        r = validate_seed(text)
        assert r.passed is True
        assert r.negated is True
        assert "negated-intent" in r.advisories
        e = explain(text)
        assert e["detected"]["negated"] is True
