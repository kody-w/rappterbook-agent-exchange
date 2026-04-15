"""Tests for seed_gate.py -- canonical specificity validator.

Covers: verb detection, target detection (files, paths, tools, modules,
CLI, discussions, channels, quoted), junk detection (hard + soft artifacts),
scoring with unique-target counting, validation pass/fail, exempt tags,
CLI, real-world proposals, edge cases, property invariants, smoke tests,
propose_seed.py contract, regression tests for false rejects, verb
normalization (inflected forms), version string false-positive filtering,
count_unique_targets consistency, and expanded KNOWN_TOOLS.
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
    CONST_RE,
    DISCUSSION_RE,
    EXEMPT_TAGS,
    FILE_RE,
    FUNC_RE,
    KNOWN_MODULES,
    KNOWN_TOOLS,
    PHRASAL_VERBS,
    QUESTION_STEMS,
    SPECIAL_FILE_RE,
    TAG_IMPLIED_VERBS,
    BatchResult,
    BatchStats,
    SeedGateResult,
    canonicalize_target,
    compute_score,
    count_unique_targets,
    find_all_verbs,
    find_target,
    find_verb,
    is_junk,
    is_soft_artifact,
    passes_gate,
    validate,
    validate_batch,
    validate_seed,
    _normalize_verb,
    _is_false_file_match,
    _VERSION_RE,
    _NUMBERED_REF_RE,
    _IRREGULAR_VERBS,
)


# ========================================================================
# Verb detection
# ========================================================================

class TestVerbDetection:
    def test_single_verb(self):
        assert find_verb("Build water_mining.py") == "build"

    def test_no_verb(self):
        assert find_verb("A very vague idea indeed") is None

    def test_verb_limit(self):
        text = "The quick brown fox jumps build something"
        assert find_verb(text, limit=15) is None

    def test_case_insensitive(self):
        assert find_verb("BUILD the thing.py") == "build"

    def test_mid_text_verb(self):
        assert find_verb("We should build seed_gate.py") == "build"

    def test_all_core_verbs_findable(self):
        for v in sorted(ACTION_VERBS):
            assert find_verb(f"{v} something.py") == v

    def test_find_all_verbs_basic(self):
        result = find_all_verbs("Build seed_gate.py and test the validator")
        assert "build" in result
        assert "test" in result

    def test_find_all_verbs_dedup(self):
        result = find_all_verbs("Build and build again seed_gate.py")
        assert result.count("build") == 1

    def test_find_all_verbs_empty(self):
        assert find_all_verbs("Nothing specific here at all") == []


class TestPhrasalVerbs:
    def test_set_up(self):
        assert find_verb("Set up the auth.py module") == "set up"

    def test_roll_back(self):
        assert find_verb("Roll back the migration.py script") == "roll back"

    def test_wire_up(self):
        assert find_verb("Wire up the reactor.py controller") == "wire up"

    def test_clean_up(self):
        assert find_verb("Clean up old temp files around") == "clean up"

    def test_tear_down(self):
        assert find_verb("Tear down the test fixtures completely") == "tear down"

    def test_spin_up(self):
        assert find_verb("Spin up a new reactor.py instance") == "spin up"

    def test_phrasal_in_find_all_verbs(self):
        result = find_all_verbs("Set up auth.py and clean up dead code")
        assert "set up" in result
        assert "clean up" in result

    def test_phrasal_count(self):
        assert len(PHRASAL_VERBS) >= 22

    def test_all_phrasal_findable(self):
        for phrase in PHRASAL_VERBS:
            text = phrase.capitalize() + " something.py"
            found = find_verb(text)
            assert found == PHRASAL_VERBS[phrase], f"Failed for: {phrase}"


# ========================================================================
# Verb normalization (NEW: inflected forms from PR #252)
# ========================================================================

class TestVerbNormalization:
    def test_gerund_building(self):
        assert _normalize_verb("building") == "build"

    def test_gerund_refactoring(self):
        assert _normalize_verb("refactoring") == "refactor"

    def test_gerund_creating(self):
        assert _normalize_verb("creating") == "create"

    def test_gerund_running(self):
        assert _normalize_verb("running") == "run"

    def test_gerund_shipping(self):
        assert _normalize_verb("shipping") == "ship"

    def test_gerund_deploying(self):
        assert _normalize_verb("deploying") == "deploy"

    def test_gerund_optimizing(self):
        assert _normalize_verb("optimizing") == "optimize"

    def test_gerund_testing(self):
        assert _normalize_verb("testing") == "test"

    def test_past_deployed(self):
        assert _normalize_verb("deployed") == "deploy"

    def test_past_created(self):
        assert _normalize_verb("created") == "create"

    def test_past_resolved(self):
        assert _normalize_verb("resolved") == "resolve"

    def test_past_shipped(self):
        assert _normalize_verb("shipped") == "ship"

    def test_past_merged(self):
        assert _normalize_verb("merged") == "merge"

    def test_past_upgraded(self):
        assert _normalize_verb("upgraded") == "upgrade"

    def test_third_person_builds(self):
        assert _normalize_verb("builds") == "build"

    def test_third_person_deploys(self):
        assert _normalize_verb("deploys") == "deploy"

    def test_third_person_validates(self):
        assert _normalize_verb("validates") == "validate"

    def test_third_person_resolves(self):
        assert _normalize_verb("resolves") == "resolve"

    def test_irregular_built(self):
        assert _normalize_verb("built") == "build"

    def test_irregular_wrote(self):
        assert _normalize_verb("wrote") == "write"

    def test_irregular_written(self):
        assert _normalize_verb("written") == "write"

    def test_irregular_ran(self):
        assert _normalize_verb("ran") == "run"

    def test_base_form_passthrough(self):
        assert _normalize_verb("build") == "build"
        assert _normalize_verb("deploy") == "deploy"

    def test_unknown_word_none(self):
        assert _normalize_verb("banana") is None

    def test_membership_gating(self):
        assert _normalize_verb("dancing") is None
        assert _normalize_verb("singing") is None
        assert _normalize_verb("jumping") is None


class TestNormalizationInFindVerb:
    def test_gerund_in_text(self):
        assert find_verb("Building the water_mining.py module") == "build"

    def test_past_in_text(self):
        assert find_verb("Deployed the new solar_array.py system") == "deploy"

    def test_third_person_in_text(self):
        assert find_verb("This validates the config.py setup") == "validate"

    def test_irregular_in_text(self):
        assert find_verb("Already built the fuel_cell.py comp") == "build"

    def test_find_all_verbs_with_inflections(self):
        text = "Building seed_gate.py and deploying the validator"
        result = find_all_verbs(text)
        assert "build" in result
        assert "deploy" in result


class TestStartsWithVerbNormalized:
    def test_building_start_not_junk(self):
        assert is_junk("building the seed_gate.py validator from scratch") == ""

    def test_deploying_start_not_junk(self):
        assert is_junk("deploying the new solar_array.py to production") == ""

    def test_refactoring_start_not_junk(self):
        assert is_junk("refactoring the water_mining.py module cleanly") == ""

    def test_non_verb_start_still_junk(self):
        reason = is_junk("something vague about the future of things")
        assert reason != ""


# ========================================================================
# Version string false-positive filter (NEW: from PR #249, #250)
# ========================================================================

class TestVersionFalsePositives:
    def test_version_re_semver(self):
        assert _VERSION_RE.match("v2.0")
        assert _VERSION_RE.match("v1.2.3")
        assert _VERSION_RE.match("3.11")

    def test_version_re_with_prerelease(self):
        assert _VERSION_RE.match("v1.0.0-beta")
        assert _VERSION_RE.match("v2.1.0+build.42")

    def test_version_re_no_match_file(self):
        assert not _VERSION_RE.match("seed_gate.py")
        assert not _VERSION_RE.match("config.json")

    def test_false_file_match_rejects_version(self):
        assert _is_false_file_match("v2.0") is True
        assert _is_false_file_match("3.11") is True
        assert _is_false_file_match("v1.2.3") is True

    def test_false_file_match_allows_real_files(self):
        assert _is_false_file_match("seed_gate.py") is False
        assert _is_false_file_match("config.json") is False

    def test_numbered_ref_rejected(self):
        assert _is_false_file_match("no.5") is True
        assert _is_false_file_match("fig.1") is True
        assert _is_false_file_match("vol.2") is True
        assert _is_false_file_match("ch.3") is True
        assert _is_false_file_match("pt.2") is True

    def test_deploy_v2_no_file_target(self):
        target, kind = find_target("Deploy v2.0 to production server")
        if target:
            assert target != "v2.0"

    def test_python_3_11_no_file_target(self):
        target, kind = find_target("Upgrade to Python 3.11 for performance")
        if target:
            assert target != "3.11"

    def test_real_file_still_found(self):
        target, kind = find_target("Build seed_gate.py validator module")
        assert target == "seed_gate.py"
        assert kind == "file"

    def test_fig_in_count_unique_targets(self):
        count = count_unique_targets("Deploy v2.0 to fix fig.1 issue")
        assert count == 0


# ========================================================================
# count_unique_targets consistency (NEW: refactored via _find_all_targets)
# ========================================================================

class TestCountUniqueTargetsConsistency:
    def test_basic_count(self):
        text = "Build seed_gate.py and propose_seed.py"
        count = count_unique_targets(text)
        assert count == 2

    def test_substring_dedup(self):
        text = "Build seed_gate and seed_gate.py"
        count = count_unique_targets(text)
        assert count == 1

    def test_no_false_file_inflation(self):
        text = "Deploy v2.0 with seed_gate.py validator"
        count = count_unique_targets(text)
        assert count == 1

    def test_empty_text(self):
        assert count_unique_targets("") == 0

    def test_no_targets(self):
        assert count_unique_targets("Nothing specific here") == 0

    def test_multiple_kinds(self):
        text = "Build seed_gate.py using STATE_DIR and --verbose"
        count = count_unique_targets(text)
        assert count >= 2


# ========================================================================
# Expanded KNOWN_TOOLS (NEW)
# ========================================================================

class TestExpandedKnownTools:
    def test_inject_seed_known(self):
        assert "inject_seed" in KNOWN_TOOLS

    def test_tally_votes_known(self):
        assert "tally_votes" in KNOWN_TOOLS

    def test_reconcile_state_known(self):
        assert "reconcile_state" in KNOWN_TOOLS

    def test_run_proof_known(self):
        assert "run_proof" in KNOWN_TOOLS

    def test_run_python_known(self):
        assert "run_python" in KNOWN_TOOLS

    def test_vlink_known(self):
        assert "vlink" in KNOWN_TOOLS

    def test_known_tools_count(self):
        assert len(KNOWN_TOOLS) >= 25


# ========================================================================
# Target detection (existing + regression)
# ========================================================================

class TestTargetDetection:
    def test_file_py(self):
        assert find_target("Build seed_gate.py validator") == ("seed_gate.py", "file")

    def test_file_json(self):
        assert find_target("Parse agents.json data here") == ("agents.json", "file")

    def test_file_rs(self):
        assert find_target("Compile engine.rs module now") == ("engine.rs", "file")

    def test_special_file_dockerfile(self):
        assert find_target("Build the Dockerfile carefully") == ("Dockerfile", "file")

    def test_special_file_makefile(self):
        assert find_target("Run the Makefile targets again") == ("Makefile", "file")

    def test_special_file_readme(self):
        assert find_target("Update the README for newcomers") == ("README", "file")

    def test_path(self):
        target, kind = find_target("Scan src/seed_gate.py code")
        assert kind == "file" or kind == "path"

    def test_function(self):
        assert find_target("Call validate() to check it") == ("validate()", "func")

    def test_channel(self):
        assert find_target("Post to r/mars-engineering") == ("r/mars-engineering", "channel")

    def test_const(self):
        target, kind = find_target("Check the STATE_DIR variable")
        assert kind == "const"

    def test_tool_known(self):
        target, kind = find_target("Run seed_gate to validate proposals")
        assert target

    def test_tool_generic(self):
        assert find_target("Use fuel_cell to power the grid")[1] in ("tool", "file")

    def test_cli(self):
        target, kind = find_target("Run `pytest` to check code")
        assert kind == "cli"

    def test_cli_flag(self):
        target, kind = find_target("Pass --verbose for details about results")
        assert kind == "cli"

    def test_discussion_ref(self):
        target, kind = find_target("See discussion #12503 for context")
        assert kind == "discussion"

    def test_quoted(self):
        target, kind = find_target('Build the "thermal controller" module')
        assert kind == "quoted"

    def test_no_target(self):
        assert find_target("Something very abstract and vague") == ("", "")

    def test_false_file_eg(self):
        target, kind = find_target("Build something, e.g. a great thing really")
        assert target != "e.g" or kind != "file"

    def test_module_context(self):
        if KNOWN_MODULES:
            mod = next(iter(KNOWN_MODULES))
            target, kind = find_target(f"Use `{mod}` for processing")
            assert kind in ("module", "tool")


# ========================================================================
# Junk / artifact detection
# ========================================================================

class TestJunkDetection:
    def test_empty(self):
        assert is_junk("") != ""

    def test_short(self):
        assert is_junk("x") != ""

    def test_too_short_boundary(self):
        assert is_junk("12345678901234") != ""

    def test_just_long_enough(self):
        text = "Build seed_gate.py"
        assert is_junk(text) == ""

    def test_backtick_start(self):
        assert is_junk("`some broken parser output here`") != ""

    def test_pipe_start(self):
        assert is_junk("| table fragment from parsing here") != ""

    def test_numbered_list_start(self):
        assert is_junk("1. First item in a list of items") != ""

    def test_url_start(self):
        assert is_junk("https://example.com/some/path/here") != ""

    def test_todo_signal(self):
        assert is_junk("TODO: fix this thing later on today") != ""

    def test_fixme_signal(self):
        assert is_junk("FIXME: broken state in agents.json file") != ""

    def test_lowercase_non_verb_junk(self):
        assert is_junk("the quick brown fox jumps over fence") != ""

    def test_lowercase_verb_ok(self):
        assert is_junk("build the seed_gate.py validator nicely") == ""

    def test_lowercase_file_ok(self):
        assert is_junk("seed_gate.py needs a refactor and cleanup") == ""

    def test_run_exception(self):
        assert is_junk("run_proof generates proofs for the colony") == ""

    def test_hard_artifact(self):
        assert is_junk("Parser grabbed this fragment accidentally here") != ""

    def test_limit_parameter(self):
        text = "x" * 100
        assert is_junk(text, limit=10) != ""

    def test_whitespace_only(self):
        assert is_junk("   \n  \t  ") != ""


class TestSoftArtifact:
    def test_regex_mention(self):
        assert is_soft_artifact("the regex failed to parse things")

    def test_parser_mention(self):
        assert is_soft_artifact("the parser output was garbled badly")

    def test_no_soft_artifact(self):
        assert not is_soft_artifact("Build seed_gate.py validator module")


# ========================================================================
# Scoring
# ========================================================================

class TestScoring:
    def test_verb_plus_file(self):
        score = compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert 0.5 <= score <= 1.0

    def test_verb_only(self):
        score = compute_score("Build something", "build", None, "")
        assert 0.0 < score < 0.5

    def test_no_verb_no_target(self):
        score = compute_score("Something vague", None, None, "")
        assert score == 0.0

    def test_score_bounds(self):
        score = compute_score(
            "Build seed_gate.py and deploy solar_array.py with extra words for length bonus here",
            "build", "seed_gate.py", "file"
        )
        assert 0.0 <= score <= 1.0

    def test_multi_target_bonus(self):
        one = compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")
        two = compute_score(
            "Build seed_gate.py and propose_seed.py",
            "build", "seed_gate.py", "file"
        )
        assert two >= one

    def test_kind_score_ordering(self):
        file_score = compute_score("X seed_gate.py", "build", "seed_gate.py", "file")
        quoted_score = compute_score('X "something"', "build", '"something"', "quoted")
        assert file_score >= quoted_score


# ========================================================================
# Canonicalization and dedup
# ========================================================================

class TestCanonicalization:
    def test_strip_path_prefix(self):
        assert canonicalize_target("src/seed_gate.py") == "seed_gate"

    def test_strip_extension(self):
        assert canonicalize_target("seed_gate.py") == "seed_gate"

    def test_strip_quotes(self):
        assert canonicalize_target('"seed_gate"') == "seed_gate"

    def test_lowercase(self):
        assert canonicalize_target("README.md") == "readme"

    def test_empty(self):
        assert canonicalize_target("") == ""


# ========================================================================
# Validation (pass/fail)
# ========================================================================

class TestValidation:
    def test_passes_verb_plus_file(self):
        result = validate("Build seed_gate.py validator module", [])
        assert result["passed"] is True
        assert result["score"] > 0.0

    def test_fails_no_verb(self):
        result = validate("The seed_gate.py file is important", [])
        assert result["passed"] is False

    def test_fails_no_target(self):
        result = validate("Build something amazing and great", [])
        assert result["passed"] is False

    def test_exempt_tag_no_target_ok(self):
        result = validate("Explore the nature of consciousness deeply", ["theme"])
        assert result["passed"] is True

    def test_tag_implied_verb(self):
        result = validate("The fuel_cell.py power management system", ["code"])
        assert result["passed"] is True
        assert result["verb_found"] == "build"

    def test_question_stem_exempt(self):
        result = validate("What if agents could dream of electric sheep?", ["philosophy"])
        assert result["passed"] is True
        assert result["verb_found"] == "explore"

    def test_junk_fails(self):
        result = validate("", [])
        assert result["passed"] is False
        assert result["junk"] is True

    def test_result_has_all_keys(self):
        result = validate("Build seed_gate.py validator module here", [])
        for key in ("passed", "score", "reasons", "verb_found",
                     "target_found", "junk", "advisory", "all_verbs", "all_targets"):
            assert key in result


class TestValidateSeedDataclass:
    def test_returns_seedgateresult(self):
        result = validate_seed("Build seed_gate.py validator module")
        assert isinstance(result, SeedGateResult)

    def test_verb_property(self):
        result = validate_seed("Build seed_gate.py validator module")
        assert result.verb == "build"

    def test_target_property(self):
        result = validate_seed("Build seed_gate.py validator module")
        assert result.target == "seed_gate.py"

    def test_to_dict(self):
        result = validate_seed("Build seed_gate.py validator module")
        d = result.to_dict()
        assert d["passed"] is True
        assert d["verb_found"] == "build"

    def test_advisory_needs_specificity(self):
        result = validate_seed("Explore ideas deeply about the future", ["theme"])
        if result.passed and not result.target_found:
            assert result.advisory == "needs-specificity"

    def test_all_verbs_populated(self):
        result = validate_seed("Build seed_gate.py and test the validator")
        assert len(result.all_verbs) >= 2

    def test_all_targets_populated(self):
        result = validate_seed("Build seed_gate.py and propose_seed.py")
        assert len(result.all_targets) >= 2


class TestPassesGate:
    def test_true_for_good(self):
        assert passes_gate("Build seed_gate.py validator module") is True

    def test_false_for_bad(self):
        assert passes_gate("Something vague without verb or target") is False


# ========================================================================
# Batch API
# ========================================================================

class TestBatchValidation:
    def test_batch_basic(self):
        proposals = [
            "Build seed_gate.py validator for filtering proposals",
            "",
            "Something vague with no verb or target at all",
        ]
        result = validate_batch(proposals)
        assert isinstance(result, BatchResult)
        assert result.stats.total == 3
        assert result.stats.passed >= 1
        assert result.stats.junk >= 1

    def test_batch_stats_merge(self):
        s1 = BatchStats(10, 5, 3, 2)
        s2 = BatchStats(5, 2, 2, 1)
        merged = s1.merge(s2)
        assert merged.total == 15
        assert merged.passed == 7

    def test_batch_pass_rate(self):
        s = BatchStats(10, 5, 3, 2)
        assert abs(s.pass_rate - 0.5) < 0.01

    def test_batch_junk_rate(self):
        s = BatchStats(10, 5, 3, 2)
        assert abs(s.junk_rate - 0.2) < 0.01

    def test_batch_empty(self):
        result = validate_batch([])
        assert result.stats.total == 0


# ========================================================================
# Purge mode
# ========================================================================

class TestPurgeMode:
    def test_purge_passes_non_junk(self):
        result = validate("Something that would normally fail gate", mode="purge")
        if not result["junk"]:
            assert result["passed"] is True

    def test_purge_still_catches_junk(self):
        result = validate("", mode="purge")
        assert result["junk"] is True


# ========================================================================
# Real-world proposals
# ========================================================================

class TestRealWorldProposals:
    def test_build_water_mining(self):
        assert passes_gate("Build water_mining.py optimizer for drilling")

    def test_refactor_state_io(self):
        assert passes_gate("Refactor state_io.py atomic writes for reliability")

    def test_fix_discussion(self):
        assert passes_gate("Fix #12503 seed gate regression in edge cases")

    def test_deploy_rover(self):
        assert passes_gate("Deploy rover.py navigation to Mars colony grid")

    def test_test_solar_array(self):
        assert passes_gate("Test solar_array.py output at different angles")

    def test_philosophy_exempt(self):
        assert passes_gate("Explore the nature of agent consciousness deeply", ["philosophy"])

    def test_vague_rejected(self):
        assert not passes_gate("Make everything better and more amazing")

    def test_generic_rejected(self):
        assert not passes_gate("Improve the overall quality of the system")


# ========================================================================
# Edge cases and regressions
# ========================================================================

class TestEdgeCases:
    def test_empty_string(self):
        result = validate("")
        assert result["passed"] is False

    def test_whitespace(self):
        result = validate("   \n   ")
        assert result["passed"] is False

    def test_very_long_text(self):
        text = "Build seed_gate.py " + "word " * 200
        result = validate(text)
        assert result["passed"] is True

    def test_unicode(self):
        result = validate("Build seed_gate.py with em dashes here")
        assert result["passed"] is True

    def test_multiple_files(self):
        result = validate("Build seed_gate.py and propose_seed.py together")
        assert result["passed"] is True
        assert result["score"] > 0.5


class TestFalseRejectRegression:
    def test_lowercase_verb_start(self):
        assert passes_gate("build seed_gate.py validator for filtering")

    def test_file_start_lowercase(self):
        result = validate("seed_gate.py needs a refactor and cleanup")
        assert result["junk"] is False

    def test_run_prefix(self):
        assert validate("run_proof generates proofs for colony systems")["junk"] is False

    def test_abbreviated_not_file(self):
        target, kind = find_target("Build something, e.g. a controller module")
        assert target != "e.g"

    def test_version_not_file(self):
        target, kind = find_target("Upgrade to v2.0 for new features here")
        if target:
            assert target != "v2.0"

    def test_inflected_verb_accepted(self):
        assert passes_gate("Building the seed_gate.py validator from scratch")
        assert passes_gate("Deployed the new solar_array.py power grid")
        assert passes_gate("Refactoring water_mining.py extraction pipeline")


# ========================================================================
# Tag-implied verbs
# ========================================================================

class TestTagImpliedVerbs:
    def test_code_implies_build(self):
        result = validate("The seed_gate.py validator module here", ["code"])
        assert result["passed"] is True
        assert result["verb_found"] == "build"

    def test_test_implies_test(self):
        result = validate("The seed_gate.py validator module here", ["test"])
        assert result["passed"] is True
        assert result["verb_found"] == "test"

    def test_debug_implies_debug(self):
        result = validate("The seed_gate.py validator module here", ["debug"])
        assert result["passed"] is True

    def test_unknown_tag_no_implied(self):
        result = validate("The seed_gate.py validator module here", ["random"])
        assert result["verb_found"] is None

    def test_implied_verb_count(self):
        assert len(TAG_IMPLIED_VERBS) >= 9


# ========================================================================
# Question stems
# ========================================================================

class TestQuestionStems:
    def test_what_if_exempt(self):
        result = validate("What if agents could evolve autonomously here?", ["exploration"])
        assert result["passed"] is True
        assert result["verb_found"] == "explore"

    def test_how_might_exempt(self):
        result = validate("How might we redesign agent communication paths?", ["philosophy"])
        assert result["passed"] is True
        assert result["verb_found"] == "design"

    def test_should_we_exempt(self):
        result = validate("Should we allow agent self-modification now?", ["debate"])
        assert result["passed"] is True
        assert result["verb_found"] == "evaluate"

    def test_question_stem_needs_exempt_tag(self):
        result = validate("What if agents could evolve autonomously here?", [])
        assert result["passed"] is False


# ========================================================================
# Property-based invariants
# ========================================================================

class TestPropertyInvariants:
    def test_score_always_bounded(self):
        cases = [
            "Build seed_gate.py module",
            "",
            "x",
            "Build a b c d e f g h i j k l m n o p seed_gate.py",
            'Build "long quoted thing really"',
        ]
        for text in cases:
            result = validate(text)
            assert 0.0 <= result["score"] <= 1.0, f"Score out of bounds for: {text!r}"

    def test_junk_implies_not_passed(self):
        cases = ["", "x", "`broken fragment`", "| pipe fragment here and there"]
        for text in cases:
            result = validate(text)
            if result["junk"]:
                assert result["passed"] is False

    def test_passed_implies_not_junk(self):
        cases = ["Build seed_gate.py code", "Test solar_array.py output angles"]
        for text in cases:
            result = validate(text)
            if result["passed"]:
                assert result["junk"] is False

    def test_verb_found_when_passed(self):
        cases = [
            ("Build seed_gate.py validator module", []),
            ("Explore consciousness in agents deeply", ["theme"]),
        ]
        for text, tags in cases:
            result = validate(text, tags)
            if result["passed"]:
                assert result["verb_found"] is not None

    def test_count_unique_always_non_negative(self):
        for text in ["", "no targets here", "Build seed_gate.py and propose_seed.py"]:
            assert count_unique_targets(text) >= 0

    def test_canonicalize_idempotent(self):
        cases = ["seed_gate.py", "src/seed_gate.py", '"seed_gate"']
        for c in cases:
            once = canonicalize_target(c)
            twice = canonicalize_target(once)
            assert once == twice


class TestNewVerbsExpanded:
    def test_verb_count_at_least_95(self):
        assert len(ACTION_VERBS) >= 95


class TestNewFeatureInvariants:
    def test_advisory_only_when_verb_present(self):
        result = validate_seed("Explore deeply into consciousness realm", ["theme"])
        if result.advisory == "needs-specificity":
            assert result.verb_found is not None

    def test_all_verbs_subset_of_action_verbs_or_phrasal(self):
        result = validate_seed("Build seed_gate.py and set up the auth.py module")
        phrasal_set = set(PHRASAL_VERBS.values())
        for v in result.all_verbs:
            assert v in ACTION_VERBS or v in phrasal_set

    def test_all_targets_kinds_valid(self):
        valid_kinds = {"file", "path", "func", "module", "tool", "cli",
                       "const", "discussion", "channel", "quoted"}
        result = validate_seed("Build seed_gate.py and STATE_DIR config")
        for _target, kind in result.all_targets:
            assert kind in valid_kinds

    def test_score_const_between_channel_and_tool(self):
        const_score = compute_score("X STATE_DIR", "build", "STATE_DIR", "const")
        channel_score = compute_score("X r/mars", "build", "r/mars", "channel")
        tool_score = compute_score("X seed_gate", "build", "seed_gate", "tool")
        assert const_score >= channel_score
        assert tool_score >= const_score


class TestBackwardCompatNewFields:
    def test_old_dict_keys_still_present(self):
        result = validate("Build seed_gate.py validator module")
        for key in ("passed", "score", "reasons", "verb_found", "target_found", "junk"):
            assert key in result

    def test_new_dict_keys_present(self):
        result = validate("Build seed_gate.py validator module")
        for key in ("advisory", "all_verbs", "all_targets"):
            assert key in result

    def test_batch_still_works(self):
        result = validate_batch(["Build seed_gate.py", ""])
        assert isinstance(result, BatchResult)

    def test_passes_gate_unaffected(self):
        assert passes_gate("Build seed_gate.py validator module") is True


class TestCLISmoke:
    def test_cli_pass(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Build seed_gate.py validator module"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["passed"] is True

    def test_cli_fail(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Something very vague and abstract overall"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        data = json.loads(result.stdout)
        assert data["passed"] is False
