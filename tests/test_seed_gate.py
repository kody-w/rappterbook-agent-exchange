"""Tests for seed_gate.py -- canonical specificity validator.

Covers: verb detection, target detection (files, paths, tools, modules,
CLI, discussions, channels, quoted, proper nouns), junk detection
(hard + soft artifacts), negation awareness, scoring with placeholder
penalty, validation pass/fail, exempt tags, VerbMatch dataclass,
verb weights, CLI, real-world proposals, edge cases, property invariants,
smoke tests, propose_seed.py contract, batch diagnostics, and regression
tests for false rejects.
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
    PATH_RE,
    PHRASAL_VERBS,
    PROPER_NOUN_RE,
    QUESTION_STEMS,
    QUOTED_RE,
    SPECIAL_FILE_RE,
    TAG_IMPLIED_VERBS,
    TOOL_RE,
    BatchResult,
    BatchStats,
    SeedGateResult,
    VerbMatch,
    _ABBREV_REF_RE,
    _INFLECTION_MAP,
    _KIND_SCORES,
    _NEGATION_WORDS,
    _PLACEHOLDER_FILES,
    _VERB_WEIGHTS,
    _is_negated,
    canonicalize_target,
    compute_score,
    count_unique_targets,
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
    verb_weight,
)


# ===================================================================
# Verb detection
# ===================================================================

class TestFindVerb:
    def test_base_verb(self):
        assert find_verb("Build a new module") == "build"

    def test_inflected_builds(self):
        assert find_verb("Building the reactor core") == "build"

    def test_inflected_past(self):
        assert find_verb("Deployed the service") == "deploy"

    def test_irregular_past(self):
        assert find_verb("Built the thermal system") == "build"

    def test_no_verb(self):
        assert find_verb("The quick brown fox") is None

    def test_phrasal_verb(self):
        assert find_verb("Set up the pipeline") == "set up"

    def test_phrasal_inflected(self):
        assert find_verb("Setting up the auth") == "set up"

    def test_limit(self):
        assert find_verb("Build the system and deploy it", limit=10) == "build"

    def test_case_insensitive(self):
        assert find_verb("BUILD a reactor") == "build"

    def test_multiple_returns_first(self):
        assert find_verb("Build and deploy the system") == "build"

    def test_known_verb_count(self):
        assert len(ACTION_VERBS) >= 95

    def test_all_single_word(self):
        for v in ACTION_VERBS:
            assert " " not in v, f"Multi-word verb in ACTION_VERBS: {v}"


class TestFindAllVerbs:
    def test_multiple(self):
        verbs = find_all_verbs("Build and deploy the system")
        assert "build" in verbs
        assert "deploy" in verbs

    def test_dedup(self):
        verbs = find_all_verbs("Build the build system to build faster")
        assert verbs.count("build") == 1

    def test_inflected(self):
        verbs = find_all_verbs("Creating and deploying services")
        assert "create" in verbs
        assert "deploy" in verbs

    def test_empty(self):
        assert find_all_verbs("") == []

    def test_phrasal(self):
        verbs = find_all_verbs("Set up and tear down the env")
        assert "set up" in verbs
        assert "tear down" in verbs


class TestFindVerbWithPosition:
    def test_returns_verb_match(self):
        vm = find_verb_with_position("Build the reactor")
        assert isinstance(vm, VerbMatch)
        assert vm.verb == "build"
        assert vm.position == 0
        assert vm.origin == "text"
        assert vm.match_kind == "explicit"

    def test_inflected_kind(self):
        vm = find_verb_with_position("Creating a new module")
        assert vm.verb == "create"
        assert vm.match_kind == "inflected"

    def test_phrasal_kind(self):
        vm = find_verb_with_position("Set up the pipeline")
        assert vm.verb == "set up"
        assert vm.match_kind == "phrasal"

    def test_none_when_no_verb(self):
        assert find_verb_with_position("The quick brown fox") is None

    def test_position_mid_sentence(self):
        vm = find_verb_with_position("We should build the system")
        assert vm.position == 2

    def test_limit(self):
        vm = find_verb_with_position("The big deploy happened", limit=8)
        assert vm is None  # "deploy" is past limit

    def test_agrees_with_find_verb(self):
        texts = [
            "Build the reactor", "Setting up auth", "The quick fox",
            "Deploy and test it", "Creating modules",
        ]
        for t in texts:
            vm = find_verb_with_position(t)
            fv = find_verb(t)
            if vm:
                assert vm.verb == fv
            else:
                assert fv is None


class TestVerbMatchDataclass:
    def test_frozen(self):
        vm = VerbMatch(verb="build", position=0, origin="text", match_kind="explicit")
        with pytest.raises(AttributeError):
            vm.verb = "fix"

    def test_fields(self):
        vm = VerbMatch(verb="set up", position=1, origin="text", match_kind="phrasal")
        assert vm.verb == "set up"
        assert vm.position == 1
        assert vm.origin == "text"
        assert vm.match_kind == "phrasal"


# ===================================================================
# Target detection
# ===================================================================

class TestFindTarget:
    def test_py_file(self):
        t, k = find_target("Build seed_gate.py")
        assert t == "seed_gate.py"
        assert k == "file"

    def test_path(self):
        t, k = find_target("Fix src/thermal/controller.py")
        assert "thermal" in t or "controller" in t

    def test_function(self):
        t, k = find_target("Optimize compute_score() performance")
        assert t == "compute_score()"
        assert k == "func"

    def test_channel(self):
        t, k = find_target("Post to r/mars-engineering")
        assert "mars-engineering" in t
        assert k == "channel"

    def test_env_var(self):
        t, k = find_target("Set $STATE_DIR properly")
        assert "STATE_DIR" in t
        assert k == "env"

    def test_const(self):
        t, k = find_target("Update MAX_RETRIES constant")
        assert t == "MAX_RETRIES"
        assert k == "const"

    def test_tool(self):
        t, k = find_target("Fix state_io module")
        assert t == "state_io"
        assert k == "tool"

    def test_cli(self):
        t, k = find_target("Add --verbose flag")
        assert "--verbose" in t
        assert k == "cli"

    def test_discussion(self):
        t, k = find_target("Address feedback from #12503")
        assert "12503" in t
        assert k == "discussion"

    def test_quoted(self):
        t, k = find_target('Implement "thermal regulation system" for habitat')
        assert "thermal regulation" in t
        assert k == "quoted"

    def test_special_file(self):
        t, k = find_target("Update the Dockerfile")
        assert t == "Dockerfile"
        assert k == "file"

    def test_no_target(self):
        t, k = find_target("Make everything better")
        assert t == ""
        assert k == ""

    def test_false_file_eg(self):
        t, _ = find_target("This is important e.g. for testing")
        assert t != "e.g"

    def test_version_not_file(self):
        t, _ = find_target("Upgrade to version 2.0.1")
        assert t != "2.0.1"


class TestAbbreviatedRefFilter:
    """Tests for the abbreviated reference false-file filter (#12505)."""

    def test_fig_1_not_file(self):
        assert _ABBREV_REF_RE.match("fig.1")

    def test_sec_2_not_file(self):
        assert _ABBREV_REF_RE.match("sec.2")

    def test_vol_3_not_file(self):
        assert _ABBREV_REF_RE.match("vol.3")

    def test_no_5_not_file(self):
        assert _ABBREV_REF_RE.match("no.5")

    def test_eq_7_not_file(self):
        assert _ABBREV_REF_RE.match("eq.7")

    def test_ch_4_not_file(self):
        assert _ABBREV_REF_RE.match("ch.4")

    def test_case_insensitive(self):
        assert _ABBREV_REF_RE.match("Fig.1")

    def test_real_file_not_matched(self):
        assert not _ABBREV_REF_RE.match("main.py")

    def test_abbrev_ref_not_target(self):
        # "fig.1" should not show up as a file target
        t, k = find_target("See fig.1 for the architecture diagram of the colony")
        assert t != "fig.1"

    def test_sec_ref_not_target(self):
        t, k = find_target("Refer to sec.3 in the documentation about water systems")
        assert t != "sec.3"


class TestProperNounTargets:
    """Tests for proper noun target detection (#12503)."""

    def test_protocol_suffix(self):
        t, k = find_target("Implement the Dream Catcher Protocol for this system")
        assert t == "Dream Catcher Protocol"
        assert k == "proper_noun"

    def test_model_suffix(self):
        t, k = find_target("Build the Wright Fisher Model for population genetics sim")
        assert t == "Wright Fisher Model"
        assert k == "proper_noun"

    def test_three_plus_words(self):
        t, k = find_target("Design the Good Neighbor Protocol Guards for safety")
        assert k == "proper_noun"

    def test_stopword_stripped(self):
        # "The" should be stripped, leaving "Dream Catcher Protocol"
        t, k = find_target("Review The Dream Catcher Protocol carefully for errors")
        assert "Dream" in t
        assert k == "proper_noun"

    def test_no_false_positive_short(self):
        # Two capitalized words without substance suffix
        t, k = find_target("Fix The Bug in the system logic")
        assert k != "proper_noun"

    def test_only_last_resort(self):
        # File target takes priority over proper noun
        t, k = find_target("Build seed_gate.py for Dream Catcher Protocol integration")
        assert k == "file"  # file wins

    def test_framework_suffix(self):
        t, k = find_target("Evaluate the Reactive Streams Framework for data handling")
        assert "Reactive Streams Framework" in t
        assert k == "proper_noun"


class TestPlaceholderFiles:
    """Tests for placeholder file penalty (#12530)."""

    def test_test_py_is_placeholder(self):
        assert "test.py" in _PLACEHOLDER_FILES

    def test_foo_py_is_placeholder(self):
        assert "foo.py" in _PLACEHOLDER_FILES

    def test_seed_gate_py_not_placeholder(self):
        assert "seed_gate.py" not in _PLACEHOLDER_FILES

    def test_placeholder_reduces_score(self):
        real = compute_score("Build seed_gate.py optimizer", "build", "seed_gate.py", "file")
        placeholder = compute_score("Build test.py optimizer", "build", "test.py", "file")
        assert placeholder < real

    def test_placeholder_still_passes(self):
        result = validate_seed("Build test.py optimizer module")
        assert result.passed is True

    def test_score_never_negative(self):
        score = compute_score("Build foo.py", "build", "foo.py", "file")
        assert score >= 0.0


# ===================================================================
# Negation detection (#12503)
# ===================================================================

class TestNegation:
    def test_dont_build(self):
        result = validate_seed("Don't build seed_gate.py anymore")
        assert result.passed is False
        assert result.negated is True
        assert "negated" in result.reasons[0].lower()

    def test_never_deploy(self):
        result = validate_seed("Never deploy this to production server")
        assert result.passed is False
        assert result.negated is True

    def test_avoid_using(self):
        # "avoid" is a negation word but "using" -> "use" is not in ACTION_VERBS
        # so no verb is found -- still correctly rejects
        result = validate_seed("Avoid using state_io for this task")
        assert result.passed is False

    def test_stop_running(self):
        result = validate_seed("Stop running the test_seed_gate.py suite")
        assert result.passed is False
        assert result.negated is True

    def test_not_only_exception(self):
        # "not only build" should NOT be treated as negation
        result = validate_seed("Not only build seed_gate.py but also deploy it")
        assert result.passed is True
        assert result.negated is False

    def test_not_just_exception(self):
        result = validate_seed("Not just test seed_gate.py, also benchmark it")
        assert result.passed is True
        assert result.negated is False

    def test_clause_boundary_resets(self):
        # Negation before comma shouldn't affect verb after comma
        result = validate_seed("Don't worry, build seed_gate.py properly")
        # The first verb "build" is after the clause boundary
        # But "worry" is not an action verb, so "build" is the primary verb
        # and the negation doesn't cross the comma
        assert result.passed is True

    def test_no_negation_positive(self):
        result = validate_seed("Build seed_gate.py for the colony")
        assert result.negated is False
        assert result.passed is True

    def test_tag_implied_verb_not_negated(self):
        # Tag-implied verbs are never negated (origin != "text")
        result = validate_seed("Not the usual approach to mars_colony.py", tags=["code"])
        assert result.negated is False

    def test_negated_field_in_dict(self):
        d = validate("Don't build seed_gate.py testing")
        assert d["negated"] is True

    def test_negation_suggestion(self):
        tips = suggest("Don't build seed_gate.py anymore")
        assert any("negation" in t.lower() for t in tips)


class TestIsNegated:
    def test_direct_not(self):
        assert _is_negated("Do not build it", 2) is True

    def test_dont_contraction(self):
        assert _is_negated("Don't build it", 1) is True

    def test_never_before_verb(self):
        assert _is_negated("Never deploy services", 1) is True

    def test_too_far(self):
        # "not" is 5+ words back from "build" -- outside the 3-word window
        assert _is_negated("We might not possibly really ever build it", 6) is False

    def test_clause_boundary_comma(self):
        assert _is_negated("Although not great, build it", 3) is False

    def test_clause_boundary_but(self):
        assert _is_negated("It was not ideal but build it", 5) is False

    def test_no_negation(self):
        assert _is_negated("Please build it now", 1) is False

    def test_position_zero(self):
        assert _is_negated("Build it now", 0) is False


# ===================================================================
# Verb weights (#12511, #12530)
# ===================================================================

class TestVerbWeights:
    def test_build_is_high(self):
        assert verb_weight("build") == "high"

    def test_explore_is_low(self):
        assert verb_weight("explore") == "low"

    def test_fix_is_medium(self):
        assert verb_weight("fix") == "medium"

    def test_all_verbs_have_weight(self):
        for v in ACTION_VERBS:
            w = verb_weight(v)
            assert w in ("high", "medium", "low"), f"{v} has no weight"

    def test_unknown_verb_medium(self):
        assert verb_weight("zzznotaverb") == "medium"

    def test_weight_in_to_dict(self):
        d = validate("Build seed_gate.py optimizer module")
        assert d["verb_weight"] == "high"

    def test_no_verb_weight_none(self):
        d = validate("Make everything better somehow")
        assert d["verb_weight"] is None

    def test_deploy_is_high(self):
        assert verb_weight("deploy") == "high"

    def test_document_is_low(self):
        assert verb_weight("document") == "low"


# ===================================================================
# Junk detection
# ===================================================================

class TestIsJunk:
    def test_empty(self):
        assert is_junk("") != ""

    def test_too_short(self):
        assert "too short" in is_junk("x")

    def test_starts_backtick(self):
        assert is_junk("`some code fragment here that is long enough`") != ""

    def test_starts_number(self):
        assert is_junk("1. First item in a numbered list") != ""

    def test_starts_url(self):
        assert is_junk("https://example.com/some/path/to/resource") != ""

    def test_todo_signal(self):
        assert is_junk("TODO fix the broken thing in here") != ""

    def test_lowercase_not_verb(self):
        assert is_junk("some random lowercase text that is long enough") != ""

    def test_lowercase_verb_ok(self):
        assert is_junk("build the new reactor thermal system") == ""

    def test_lowercase_file_ok(self):
        assert is_junk("seed_gate.py needs to be updated for colony") == ""

    def test_hard_artifact(self):
        assert "artifact" in is_junk("parser grabbed this from the wrong place")

    def test_valid_proposal(self):
        assert is_junk("Build seed_gate.py for the colony") == ""

    def test_commit_prefix_verb(self):
        assert is_junk("fix: build seed_gate.py optimizer") == ""

    def test_run_exception(self):
        assert is_junk("run_proof generates math proofs for colony") == ""


class TestSoftArtifact:
    def test_detected(self):
        assert is_soft_artifact("the regex matched something wrong here") is True

    def test_not_detected(self):
        assert is_soft_artifact("Build a new thermal system") is False


# ===================================================================
# Scoring
# ===================================================================

class TestComputeScore:
    def test_verb_only(self):
        s = compute_score("Build", "build", "", "")
        assert 0.0 < s <= 1.0

    def test_verb_plus_file(self):
        s = compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert s > 0.5

    def test_long_bonus(self):
        short = compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")
        long_text = "Build seed_gate.py with verb detection and target matching and junk filtering and scoring"
        long_s = compute_score(long_text, "build", "seed_gate.py", "file")
        assert long_s >= short

    def test_multi_target_bonus(self):
        s1 = compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")
        s2 = compute_score("Build seed_gate.py and propose_seed.py", "build", "seed_gate.py", "file")
        assert s2 > s1

    def test_max_1(self):
        long_text = "Build seed_gate.py propose_seed.py state_io.py content_engine.py and more modules " * 5
        s = compute_score(long_text, "build", "seed_gate.py", "file")
        assert s <= 1.0

    def test_min_0(self):
        s = compute_score("", None, "", "")
        assert s >= 0.0

    def test_imperative_bonus(self):
        imp = compute_score("Build seed_gate.py module", "build", "seed_gate.py", "file")
        non = compute_score("The seed_gate.py module needs building", "build", "seed_gate.py", "file")
        assert imp > non

    def test_placeholder_penalty(self):
        real = compute_score("Build seed_gate.py module", "build", "seed_gate.py", "file")
        placeholder = compute_score("Build foo.py module test", "build", "foo.py", "file")
        assert placeholder < real


class TestCountUniqueTargets:
    def test_two_files(self):
        assert count_unique_targets("seed_gate.py and propose_seed.py") == 2

    def test_dedup(self):
        assert count_unique_targets("seed_gate.py and seed_gate.py") == 1

    def test_substring_dedup(self):
        # seed_gate is substring of seed_gate in the canonical form
        assert count_unique_targets("seed_gate and seed_gate.py") == 1

    def test_empty(self):
        assert count_unique_targets("nothing here at all") == 0


# ===================================================================
# Validation (pass/fail)
# ===================================================================

class TestValidateSeed:
    def test_passes_verb_plus_file(self):
        r = validate_seed("Build seed_gate.py for the colony")
        assert r.passed is True
        assert r.verb_found == "build"
        assert r.target_found == "seed_gate.py"

    def test_fails_no_verb(self):
        r = validate_seed("The seed_gate.py needs attention badly")
        assert r.passed is False
        assert "No action verb" in r.reasons[0]

    def test_fails_no_target(self):
        r = validate_seed("Build something great and wonderful")
        assert r.passed is False

    def test_exempt_tag_no_target(self):
        r = validate_seed("Explore the nature of consciousness deeply", tags=["philosophy"])
        assert r.passed is True

    def test_junk_hard_fail(self):
        r = validate_seed("")
        assert r.passed is False
        assert r.junk is True

    def test_soft_artifact_no_redemption(self):
        r = validate_seed("The regex matched something wrong here across")
        assert r.passed is False

    def test_soft_artifact_with_verb_target(self):
        r = validate_seed("Build the regex in seed_gate.py system")
        assert r.passed is True

    def test_tag_implied_verb(self):
        r = validate_seed("The seed_gate.py validation logic stuff", tags=["code"])
        assert r.passed is True
        assert r.verb_found == "build"

    def test_question_stem_exempt(self):
        r = validate_seed("What if we explored consciousness deeply", tags=["philosophy"])
        assert r.passed is True
        assert r.verb_found == "explore"

    def test_advisory_needs_specificity(self):
        r = validate_seed("Build something for the colony project", tags=["theme"])
        assert r.advisory == "needs-specificity"

    def test_inflected_verb_passes(self):
        r = validate_seed("Creating seed_gate.py for the colony")
        assert r.passed is True
        assert r.verb_found == "create"

    def test_phrasal_verb_passes(self):
        r = validate_seed("Set up the seed_gate.py pipeline system")
        assert r.passed is True
        assert r.verb_found == "set up"

    def test_commit_prefix_passes(self):
        r = validate_seed("fix: Build seed_gate.py validator module")
        assert r.passed is True

    def test_purge_mode_always_passes(self):
        r = validate_seed("Build something great and wonderful", mode="purge")
        assert r.passed is True


class TestValidateDict:
    def test_returns_dict(self):
        d = validate("Build seed_gate.py for the colony")
        assert isinstance(d, dict)
        assert d["passed"] is True

    def test_has_all_keys(self):
        d = validate("Build seed_gate.py for the colony")
        required = {
            "passed", "reasons", "score", "verb_found", "target_found",
            "junk", "advisory", "confidence", "strength", "negated",
            "all_verbs", "all_targets", "verb_match", "verb_weight",
        }
        assert required.issubset(d.keys())

    def test_verb_match_in_dict(self):
        d = validate("Build seed_gate.py for the colony")
        vm = d["verb_match"]
        assert vm is not None
        assert vm["verb"] == "build"
        assert vm["origin"] == "text"
        assert vm["match_kind"] == "explicit"

    def test_verb_match_none_when_no_verb(self):
        d = validate("The quick brown fox jumps over")
        assert d["verb_match"] is None


class TestPassesGate:
    def test_true(self):
        assert passes_gate("Build seed_gate.py for the colony") is True

    def test_false(self):
        assert passes_gate("Make everything better somehow") is False


# ===================================================================
# SeedGateResult properties
# ===================================================================

class TestSeedGateResultProperties:
    def test_confidence_high(self):
        r = validate_seed("Build seed_gate.py and propose_seed.py and state_io.py and content_engine.py with optimizations")
        assert r.confidence == "high"

    def test_confidence_medium(self):
        r = validate_seed("Build seed_gate.py for the colony")
        assert r.confidence in ("medium", "high")

    def test_confidence_none_when_fail(self):
        r = validate_seed("Make everything better somehow")
        assert r.confidence is None

    def test_strength_strong(self):
        r = validate_seed("Build seed_gate.py and propose_seed.py and state_io.py and content_engine.py with optimizations")
        assert r.strength == "strong"

    def test_strength_none_when_zero(self):
        r = validate_seed("")
        assert r.strength == "none"

    def test_verb_property(self):
        r = validate_seed("Build seed_gate.py for the colony")
        assert r.verb == "build"

    def test_target_property(self):
        r = validate_seed("Build seed_gate.py for the colony")
        assert r.target == "seed_gate.py"

    def test_verb_empty_when_none(self):
        r = validate_seed("The quick brown fox is lazy stuff")
        assert r.verb == ""

    def test_all_verbs_populated(self):
        r = validate_seed("Build and deploy seed_gate.py system")
        assert "build" in r.all_verbs
        assert "deploy" in r.all_verbs

    def test_all_targets_populated(self):
        r = validate_seed("Build seed_gate.py and propose_seed.py")
        targets = [t[0] for t in r.all_targets]
        assert "seed_gate.py" in targets


# ===================================================================
# Batch API
# ===================================================================

class TestValidateBatch:
    def test_separates_categories(self):
        br = validate_batch([
            "Build seed_gate.py for the colony",
            "Make everything better somehow",
            "",
        ])
        assert br.stats.passed == 1
        assert br.stats.failed == 1
        assert br.stats.junk == 1

    def test_stats_total(self):
        br = validate_batch(["Build seed_gate.py for colony", "Fix water_mining.py module"])
        assert br.stats.total == 2

    def test_pass_rate(self):
        br = validate_batch(["Build seed_gate.py colony", "Fix water_mining.py module"])
        assert br.stats.pass_rate == 1.0

    def test_junk_rate(self):
        br = validate_batch(["", "x"])
        assert br.stats.junk_rate == 1.0

    def test_merge_stats(self):
        s1 = BatchStats(10, 5, 3, 2)
        s2 = BatchStats(5, 2, 2, 1)
        m = s1.merge(s2)
        assert m.total == 15
        assert m.passed == 7

    def test_empty_batch(self):
        br = validate_batch([])
        assert br.stats.total == 0
        assert br.stats.pass_rate == 0.0


class TestBatchSummary:
    """Tests for BatchResult.summary() (#12521)."""

    def test_summary_format(self):
        br = validate_batch([
            "Build seed_gate.py for the colony",
            "Make everything better somehow",
            "",
        ])
        s = br.summary()
        assert "3 total" in s
        assert "1 passed" in s
        assert "1 failed" in s
        assert "1 junk" in s

    def test_summary_percentages(self):
        br = validate_batch(["Build seed_gate.py colony", "Fix water_mining.py module"])
        s = br.summary()
        assert "100%" in s

    def test_summary_empty(self):
        br = validate_batch([])
        s = br.summary()
        assert "0 total" in s

    def test_summary_returns_string(self):
        br = validate_batch(["Build seed_gate.py colony"])
        assert isinstance(br.summary(), str)


# ===================================================================
# Score breakdown
# ===================================================================

class TestScoreBreakdown:
    def test_has_verb_component(self):
        bd = score_breakdown("Build seed_gate.py for the colony")
        assert bd["verb"] == 2.5

    def test_has_target_component(self):
        bd = score_breakdown("Build seed_gate.py for the colony")
        assert bd["target"] == _KIND_SCORES["file"]

    def test_file_target_score(self):
        bd = score_breakdown("Build seed_gate.py for the colony")
        assert bd["target"] == 4.0

    def test_discussion_target_score(self):
        bd = score_breakdown("Fix the issue from #12503 feedback")
        assert bd["target"] == _KIND_SCORES["discussion"]

    def test_length_bonus_short(self):
        bd = score_breakdown("Build seed_gate.py module")
        assert bd["length"] == 0.0

    def test_length_bonus_medium(self):
        bd = score_breakdown("Build seed_gate.py with verb detection and target matching and filtering")
        assert bd["length"] >= 0.5

    def test_multi_target_bonus(self):
        bd = score_breakdown("Build seed_gate.py and propose_seed.py pipes")
        assert bd["multi_target"] == 1.0

    def test_imperative_bonus(self):
        bd = score_breakdown("Build seed_gate.py for the colony")
        assert bd["imperative"] == 0.5

    def test_no_imperative_bonus(self):
        bd = score_breakdown("The seed_gate.py needs building for colony")
        assert bd["imperative"] == 0.0

    def test_placeholder_component(self):
        bd = score_breakdown("Build foo.py for testing the system")
        assert bd["placeholder"] == -0.5

    def test_no_placeholder_normal(self):
        bd = score_breakdown("Build seed_gate.py for the colony")
        assert bd["placeholder"] == 0.0

    def test_total_is_sum(self):
        bd = score_breakdown("Build seed_gate.py and propose_seed.py pipes")
        expected = sum(v for k, v in bd.items() if k != "total")
        assert abs(bd["total"] - expected) < 0.001

    def test_all_values_non_negative_except_placeholder(self):
        bd = score_breakdown("Build seed_gate.py for the colony")
        for k, v in bd.items():
            if k != "placeholder" and k != "total":
                assert v >= 0.0, f"{k} is negative: {v}"


# ===================================================================
# Explain
# ===================================================================

class TestExplain:
    def test_passing_contains_pass(self):
        assert "PASS" in explain("Build seed_gate.py for colony")

    def test_failing_contains_fail(self):
        assert "FAIL" in explain("Make everything better somehow")

    def test_shows_verb(self):
        assert "verb=build" in explain("Build seed_gate.py for colony")

    def test_shows_target(self):
        assert "seed_gate.py" in explain("Build seed_gate.py for colony")

    def test_shows_score(self):
        assert "score=" in explain("Build seed_gate.py for colony")

    def test_shows_confidence_when_passing(self):
        e = explain("Build seed_gate.py and propose_seed.py and state_io.py and more_stuff.py system")
        assert "confidence=" in e

    def test_shows_suggestions_when_failing(self):
        e = explain("Make everything better somehow")
        assert "suggestions=" in e

    def test_shows_junk_flag(self):
        assert "junk=true" in explain("")

    def test_no_verb_shows_none(self):
        assert "verb=none" in explain("The quick brown fox is lazy stuff")

    def test_returns_string(self):
        assert isinstance(explain("Build seed_gate.py colony"), str)

    def test_pipe_separated(self):
        assert " | " in explain("Build seed_gate.py for colony")

    def test_shows_weight(self):
        assert "weight=high" in explain("Build seed_gate.py colony")

    def test_shows_negated(self):
        assert "negated=true" in explain("Don't build seed_gate.py anymore")


# ===================================================================
# Suggest
# ===================================================================

class TestSuggest:
    def test_empty_when_passes(self):
        assert suggest("Build seed_gate.py for the colony") == []

    def test_verb_suggestion(self):
        tips = suggest("The quick brown fox is lazy stuff")
        assert any("verb" in t.lower() for t in tips)

    def test_target_suggestion(self):
        tips = suggest("Build something great and wonderful")
        assert any("target" in t.lower() for t in tips)

    def test_junk_suggestion(self):
        tips = suggest("x")
        assert any("rewrite" in t.lower() for t in tips)

    def test_negation_suggestion(self):
        tips = suggest("Don't build seed_gate.py anymore")
        assert any("negation" in t.lower() for t in tips)


# ===================================================================
# Canonicalization
# ===================================================================

class TestCanonicalizeTarget:
    def test_strips_extension(self):
        assert canonicalize_target("seed_gate.py") == "seed_gate"

    def test_strips_path(self):
        assert canonicalize_target("src/seed_gate.py") == "seed_gate"

    def test_strips_quotes(self):
        assert canonicalize_target('"seed_gate"') == "seed_gate"

    def test_lowercases(self):
        assert canonicalize_target("SeedGate.py") == "seedgate"


# ===================================================================
# Inflection map
# ===================================================================

class TestInflectionMap:
    def test_builds(self):
        assert _INFLECTION_MAP.get("builds") == "build"

    def test_creating(self):
        assert _INFLECTION_MAP.get("creating") == "create"

    def test_deployed(self):
        assert _INFLECTION_MAP.get("deployed") == "deploy"

    def test_no_self_reference(self):
        for form, base in _INFLECTION_MAP.items():
            assert form != base, f"Self-mapping: {form}"

    def test_all_map_to_action_verbs(self):
        for form, base in _INFLECTION_MAP.items():
            if " " not in base:
                assert base in ACTION_VERBS, f"{form} maps to non-verb {base}"

    def test_built_irregular(self):
        assert _INFLECTION_MAP.get("built") == "build"


# ===================================================================
# Phrasal verbs
# ===================================================================

class TestPhrasalVerbs:
    def test_set_up(self):
        assert PHRASAL_VERBS["set up"] == "set up"

    def test_all_two_words(self):
        for phrase in PHRASAL_VERBS:
            assert len(phrase.split()) == 2, f"Not two words: {phrase}"

    def test_detection(self):
        assert find_verb("Wire up the auth system") == "wire up"


# ===================================================================
# Real-world proposals (regression tests)
# ===================================================================

class TestRealWorldProposals:
    """Proposals from actual agent frames -- should never false-reject."""

    @pytest.mark.parametrize("text", [
        "Build water_mining.py optimizer for drilling",
        "Implement thermal_control.py for habitat temperature regulation",
        "Fix the broken import in src/mars_colony.py module",
        "Deploy the new solar_array.py controller system",
        "Refactor process_inbox to use action dispatcher pattern",
        "Add --verbose flag to seed_gate CLI system",
        "Create unit tests for water_purifier.py module",
        "Wire up seed_gate.py to propose_seed.py pipeline",
        "Optimize compute_score() for large batch processing",
        "Integrate the Dream Catcher Protocol for sim",
        "Consolidate the seed_gate.py implementations from PR #12503",
        "Update Dockerfile for multi-stage build system",
        "Review r/mars-engineering channel activity metrics",
        "Monitor $STATE_DIR for corruption events regularly",
        "Clean up MAX_RETRIES constant usage across system",
        "Scale up the nuclear_reactor.py output system",
        "fix: resolve seed_gate false positive on fig.1 refs",
    ])
    def test_passes(self, text):
        assert passes_gate(text), f"False reject: {text}"

    @pytest.mark.parametrize("text", [
        "Make everything better",
        "Improve the codebase",
        "Something needs fixing",
        "The system is broken",
        "",
        "x",
        "1. First step in the plan",
        "https://example.com/path",
    ])
    def test_fails(self, text):
        assert not passes_gate(text), f"False accept: {text}"


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_whitespace_only(self):
        assert validate_seed("   \n  \t  ").junk is True

    def test_very_long_text(self):
        text = "Build " + "seed_gate.py " * 100
        r = validate_seed(text)
        assert r.passed is True
        assert r.score <= 1.0

    def test_unicode(self):
        r = validate_seed("Build the résumé parser module system")
        assert r.passed is True or r.passed is False  # no crash

    def test_special_chars(self):
        r = validate_seed("Build <seed_gate> && deploy it now!")
        # Should not crash
        assert isinstance(r.passed, bool)

    def test_none_tags(self):
        r = validate_seed("Build seed_gate.py for the colony", tags=None)
        assert r.passed is True

    def test_empty_tags(self):
        r = validate_seed("Build seed_gate.py for the colony", tags=[])
        assert r.passed is True


# ===================================================================
# Property invariants
# ===================================================================

class TestPropertyInvariants:
    SAMPLE_TEXTS = [
        "Build seed_gate.py for the Mars colony system",
        "Make everything wonderful and great today",
        "Explore consciousness in artificial agents deeply",
        "Don't build the reactor core anymore please",
        "Set up the auth pipeline for secure access",
        "",
        "fix: resolve broken import in mars_env.py",
        "Not only build seed_gate.py but also deploy it",
    ]

    def test_score_bounded(self):
        for text in self.SAMPLE_TEXTS:
            r = validate_seed(text)
            assert 0.0 <= r.score <= 1.0, f"Score out of bounds for: {text}"

    def test_junk_implies_fail(self):
        for text in self.SAMPLE_TEXTS:
            r = validate_seed(text)
            if r.junk:
                assert not r.passed, f"Junk but passed: {text}"

    def test_negated_implies_fail(self):
        for text in self.SAMPLE_TEXTS:
            r = validate_seed(text)
            if r.negated:
                assert not r.passed, f"Negated but passed: {text}"

    def test_confidence_only_when_passed(self):
        for text in self.SAMPLE_TEXTS:
            r = validate_seed(text)
            if not r.passed:
                assert r.confidence is None, f"Confidence on failure: {text}"

    def test_to_dict_roundtrip(self):
        for text in self.SAMPLE_TEXTS:
            r = validate_seed(text)
            d = r.to_dict()
            assert d["passed"] == r.passed
            assert d["score"] == r.score
            assert d["negated"] == r.negated

    def test_find_verb_agrees_with_find_verb_with_position(self):
        for text in self.SAMPLE_TEXTS:
            fv = find_verb(text)
            vm = find_verb_with_position(text)
            if vm:
                assert vm.verb == fv
            else:
                assert fv is None

    def test_score_breakdown_total_matches_compute_score(self):
        for text in self.SAMPLE_TEXTS:
            r = validate_seed(text)
            if r.passed and not r.junk:
                bd = score_breakdown(text)
                normalized = min(max(bd["total"] / 10.0, 0.0), 1.0)
                assert abs(normalized - r.score) < 0.02, (
                    f"Breakdown drift for: {text} ({normalized} vs {r.score})"
                )

    def test_explain_always_returns_string(self):
        for text in self.SAMPLE_TEXTS:
            assert isinstance(explain(text), str)

    def test_explain_contains_pass_or_fail(self):
        for text in self.SAMPLE_TEXTS:
            e = explain(text)
            assert "PASS" in e or "FAIL" in e

    def test_kind_scores_constant_matches(self):
        assert "file" in _KIND_SCORES
        assert "proper_noun" in _KIND_SCORES

    def test_redesign_in_action_verbs(self):
        assert "redesign" in ACTION_VERBS

    def test_commit_prefix_seeds_pass(self):
        prefixed = [
            "fix: Build seed_gate.py validator module",
            "feat: Add water_mining.py optimizer system",
            "refactor: Clean up propose_seed.py pipeline",
        ]
        for text in prefixed:
            assert passes_gate(text), f"Commit prefix rejected: {text}"


# ===================================================================
# Smoke tests
# ===================================================================

class TestSmoke:
    def test_import(self):
        import seed_gate
        assert hasattr(seed_gate, "validate")
        assert hasattr(seed_gate, "validate_seed")
        assert hasattr(seed_gate, "validate_batch")
        assert hasattr(seed_gate, "passes_gate")
        assert hasattr(seed_gate, "find_verb")
        assert hasattr(seed_gate, "find_target")
        assert hasattr(seed_gate, "is_junk")
        assert hasattr(seed_gate, "compute_score")
        assert hasattr(seed_gate, "explain")
        assert hasattr(seed_gate, "score_breakdown")
        assert hasattr(seed_gate, "find_verb_with_position")
        assert hasattr(seed_gate, "VerbMatch")
        assert hasattr(seed_gate, "verb_weight")
        assert hasattr(seed_gate, "suggest")

    def test_backward_compat_aliases(self):
        import seed_gate
        assert hasattr(seed_gate, "_detect_verb")
        assert hasattr(seed_gate, "_detect_target")
        assert hasattr(seed_gate, "_is_junk")

    def test_batch_10_proposals(self):
        proposals = [
            "Build water_mining.py optimizer system",
            "Fix solar_array.py controller bugs",
            "Deploy nuclear_reactor.py power system",
            "Test thermal_control.py module fully",
            "Make everything better and wonderful",
            "",
            "The quick brown fox is lazy stuff",
            "Create rover.py navigation algorithm system",
            "Refactor seed_gate.py validator module",
            "Optimize compute_score() performance speed",
        ]
        br = validate_batch(proposals)
        assert br.stats.total == 10
        assert br.stats.passed > 0
        assert br.stats.junk > 0
        assert isinstance(br.summary(), str)

    def test_known_modules_populated(self):
        # Should discover modules from src/
        assert len(KNOWN_MODULES) > 0

    def test_known_tools_populated(self):
        assert len(KNOWN_TOOLS) > 10


# ===================================================================
# propose_seed.py contract tests
# ===================================================================

class TestProposeSeedContract:
    """Verify the dict shape that propose_seed.py expects."""

    def test_passed_key(self):
        d = validate("Build seed_gate.py for the colony")
        assert "passed" in d
        assert isinstance(d["passed"], bool)

    def test_score_key(self):
        d = validate("Build seed_gate.py for the colony")
        assert "score" in d
        assert isinstance(d["score"], float)

    def test_junk_key(self):
        d = validate("Build seed_gate.py for the colony")
        assert "junk" in d
        assert isinstance(d["junk"], bool)

    def test_rejected_empty_dict_not_used(self):
        # propose_seed checks gate["passed"], not empty dict
        d = validate("Make everything better somehow")
        assert d["passed"] is False

    def test_batch_junk_items(self):
        br = validate_batch(["", "x", "Build seed_gate.py colony"])
        assert len(br.junk_items) == 2
        for text, result in br.junk_items:
            assert result["junk"] is True


# ===================================================================
# Integration tests (cross-cutting, from rubber-duck critique)
# ===================================================================

class TestConsolidationIntegration:
    """Cross-cutting tests ensuring all new features work together."""

    def test_negated_verb_with_file_target(self):
        r = validate_seed("Don't build seed_gate.py anymore please")
        assert r.negated is True
        assert r.passed is False
        assert r.verb_found == "build"
        assert r.target_found == "seed_gate.py"

    def test_not_only_build_passes(self):
        r = validate_seed("Not only build seed_gate.py but deploy it too")
        assert r.passed is True
        assert r.negated is False

    def test_file_plus_proper_noun(self):
        r = validate_seed("Build seed_gate.py for Dream Catcher Protocol integration")
        assert r.target_found == "seed_gate.py"  # file wins over proper noun
        targets = [t[0] for t in r.all_targets]
        assert any("Dream Catcher Protocol" in t for t in targets)

    def test_placeholder_affects_score_not_pass(self):
        r = validate_seed("Build foo.py for testing the system")
        assert r.passed is True
        r2 = validate_seed("Build seed_gate.py for testing system")
        assert r.score < r2.score

    def test_new_fields_in_dict(self):
        d = validate("Build seed_gate.py for the colony")
        assert "negated" in d
        assert "verb_match" in d
        assert "verb_weight" in d
        assert "strength" in d

    def test_new_fields_in_explain(self):
        e = explain("Build seed_gate.py for the colony")
        assert "weight=" in e

    def test_abbrev_ref_plus_real_file(self):
        r = validate_seed("Build seed_gate.py as shown in fig.1 of docs")
        assert r.passed is True
        assert r.target_found == "seed_gate.py"

    def test_proper_noun_only_target(self):
        r = validate_seed("Implement the Dream Catcher Protocol in this system")
        assert r.passed is True
        assert r.target_found == "Dream Catcher Protocol"

    def test_verb_weight_in_result(self):
        d = validate("Build seed_gate.py for the colony")
        assert d["verb_weight"] == "high"

    def test_negated_not_junk(self):
        r = validate_seed("Don't build seed_gate.py anymore please")
        assert r.junk is False  # negated != junk

    def test_batch_summary_includes_all(self):
        br = validate_batch([
            "Build seed_gate.py for the colony",
            "Don't build anything here please now",
            "",
        ])
        s = br.summary()
        assert "3 total" in s


class TestConsolidationInvariants:
    """Property-based invariants for the consolidated validator."""

    TEXTS = [
        "Build seed_gate.py for the colony system",
        "Don't build seed_gate.py anymore please",
        "Not only build but deploy seed_gate.py system",
        "Build foo.py for testing the whole system",
        "Implement the Dream Catcher Protocol in sim",
        "See fig.1 for the architecture of colony system",
        "fix: Build seed_gate.py validator module system",
        "Explore consciousness in agents very deeply now",
    ]

    def test_negated_implies_fail(self):
        for t in self.TEXTS:
            r = validate_seed(t)
            if r.negated:
                assert not r.passed

    def test_verb_match_consistent(self):
        for t in self.TEXTS:
            r = validate_seed(t)
            if r.verb_match is not None:
                assert r.verb_found == r.verb_match.verb

    def test_verb_weight_present_when_verb(self):
        for t in self.TEXTS:
            d = validate(t)
            if d["verb_found"]:
                assert d["verb_weight"] in ("high", "medium", "low")

    def test_strength_always_present(self):
        for t in self.TEXTS:
            d = validate(t)
            assert d["strength"] in ("strong", "moderate", "weak", "none")

    def test_score_bounded(self):
        for t in self.TEXTS:
            r = validate_seed(t)
            assert 0.0 <= r.score <= 1.0
