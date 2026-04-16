"""Tests for seed_gate.py -- canonical specificity validator.

Covers: verb detection, target detection (files, paths, tools, modules,
CLI, discussions, channels, quoted), junk detection (hard + soft artifacts),
scoring with unique-target counting, validation pass/fail, exempt tags,
CLI, real-world proposals, edge cases, property invariants, smoke tests,
propose_seed.py contract, and regression tests for false rejects.

PR #289 additions: negation awareness, compound-name filtering,
VerbMatch dataclass, numbered-ref filter, multi-target scaling,
enriched SeedGateResult, strength tiers, advisories, CLI modes.
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
    passes_gate,
    validate,
    validate_seed,
    canonicalize_target,
    count_unique_targets,
    is_soft_artifact,
)
from seed_gate import (
    _strip_commit_prefix,
    _COMMIT_PREFIX_RE,
    _KIND_SCORES,
    _expand_contractions,
    _is_negated,
    _is_in_compound,
    _scan_verbs,
    _CONTRACTION_MAP,
    _NEGATION_WORDS,
    _NUMBERED_REF_RE,
    compute_score,
    find_verb,
    find_verb_match,
    find_target,
    find_verb_with_position,
    find_all_verbs,
    score_breakdown,
    explain,
    suggest,
    is_junk,
    _starts_with_verb,
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


# ===================================================================
# 2. Regex patterns
# ===================================================================

class TestRegexPatterns:
    def test_file_re_matches_py(self):
        assert FILE_RE.search("fix seed_gate.py now")

    def test_file_re_matches_json(self):
        assert FILE_RE.search("update state/mars.json")

    def test_path_re_matches(self):
        assert PATH_RE.search("add src/water_mining.py")

    def test_tool_re_snake(self):
        assert TOOL_RE.search("integrate seed_gate into pipe")

    def test_cli_re_backtick(self):
        assert CLI_RE.search("run `python -m pytest`")

    def test_cli_re_flag(self):
        assert CLI_RE.search("add --verbose flag")

    def test_discussion_re(self):
        assert DISCUSSION_RE.search("see #12503 for details")

    def test_channel_re(self):
        assert CHANNEL_RE.search("post in r/mars-engineering")

    def test_quoted_re(self):
        assert QUOTED_RE.search('fix "thermal control" bug')

    def test_module_context_re(self):
        assert MODULE_CONTEXT_RE.search("import water_mining")


# ===================================================================
# 3. Verb detection
# ===================================================================

class TestVerbDetection:
    def test_base_verb(self):
        assert find_verb("Build the reactor") == "build"

    def test_inflected_builds(self):
        assert find_verb("Builds the system") == "build"

    def test_inflected_creating(self):
        assert find_verb("Creating a new module") == "create"

    def test_inflected_deployed(self):
        assert find_verb("Deployed the reactor") == "deploy"

    def test_phrasal_set_up(self):
        assert find_verb("Set up the reactor") == "set up"

    def test_phrasal_inflected_setting_up(self):
        assert find_verb("Setting up the reactor") == "set up"

    def test_no_verb(self):
        assert find_verb("The reactor is hot") is None

    def test_limit(self):
        assert find_verb("Build the reactor", limit=3) is None

    def test_case_insensitive(self):
        assert find_verb("BUILD the reactor") == "build"

    def test_multiple_verbs_returns_first(self):
        assert find_verb("Build and test the reactor") == "build"


class TestFindAllVerbs:
    def test_multiple(self):
        v = find_all_verbs("Build and test the water_mining.py module")
        assert "build" in v
        assert "test" in v

    def test_deduped(self):
        v = find_all_verbs("Build, build, build the reactor")
        assert v.count("build") == 1

    def test_empty(self):
        assert find_all_verbs("The reactor is hot") == []

    def test_phrasal(self):
        v = find_all_verbs("Set up and tear down the reactor")
        assert "set up" in v
        assert "tear down" in v


class TestFindAllTargets:
    def test_multiple_files(self):
        r = _vs("Build water_mining.py and solar_array.py")
        assert len(r.all_targets) >= 2

    def test_deduped(self):
        r = _vs("Build seed_gate.py and src/seed_gate.py")
        assert len(r.all_targets) >= 1


# ===================================================================
# 4. Target detection
# ===================================================================

class TestTargetDetection:
    def test_file(self):
        t, k = find_target("Build seed_gate.py validator")
        assert t == "seed_gate.py"
        assert k == "file"

    def test_path(self):
        t, k = find_target("Scan tests/test_seed_gate.py for issues")
        assert "test_seed_gate" in t

    def test_tool(self):
        t, k = find_target("Integrate state_io into pipeline")
        assert "state_io" in t

    def test_discussion(self):
        t, k = find_target("Address feedback in #12503")
        assert "#12503" in t
        assert k == "discussion"

    def test_channel(self):
        t, k = find_target("Post to r/mars-engineering channel")
        assert k == "channel"

    def test_env_var(self):
        t, k = find_target("Read $STATE_DIR environment variable")
        assert k == "env"

    def test_const(self):
        t, k = find_target("Expand ACTION_VERBS constant set")
        assert k == "const"

    def test_cli(self):
        t, k = find_target("Add --verbose flag to CLI")
        assert t is not None
        assert k in ("cli", "const")  # --verbose is a CLI constant

    def test_quoted(self):
        t, k = find_target('Implement "thermal override" feature')
        assert k == "quoted"

    def test_func(self):
        t, k = find_target("Refactor compute_score() logic")
        assert k == "func"

    def test_no_target(self):
        t, k = find_target("Make everything better")
        assert t == ""
        assert k == ""


# ===================================================================
# 5. Junk detection
# ===================================================================

class TestJunkDetection:
    def test_empty(self):
        assert is_junk("") != ""

    def test_short(self):
        assert is_junk("x") != ""

    def test_whitespace(self):
        assert is_junk("   ") != ""

    def test_backtick_start(self):
        assert is_junk("`some parser output` and more") != ""

    def test_url(self):
        assert is_junk("https://example.com/path") != ""

    def test_todo(self):
        assert is_junk("TODO: fix this later please") != ""

    def test_numbered_list(self):
        assert is_junk("1. First step of the plan") != ""

    def test_lowercase_non_verb(self):
        assert is_junk("the reactor is running nicely now") != ""

    def test_lowercase_verb_ok(self):
        assert is_junk("build the reactor core module") == ""

    def test_hard_artifact(self):
        assert is_junk("parser grabbed this fragment from the code") != ""

    def test_valid_proposal(self):
        assert is_junk("Build water_mining.py optimizer") == ""

    def test_run_exception(self):
        assert is_junk("run_proof validates the math engine") == ""


class TestSoftArtifacts:
    def test_soft_artifact_detected(self):
        assert is_soft_artifact("the regex matched a pattern")

    def test_no_soft_artifact(self):
        assert not is_soft_artifact("Build water_mining.py")

    def test_soft_blocks_without_verb_target(self):
        r = _v("the regex matched something in the codebase")
        assert not r["passed"]
        assert r["junk"]


# ===================================================================
# 6. Scoring
# ===================================================================

class TestScoring:
    def test_verb_plus_file_high(self):
        s = compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert s >= 0.5

    def test_verb_only_low(self):
        s = compute_score("Build something", "build", None, "")
        assert s < 0.5

    def test_no_verb_no_target_zero(self):
        s = compute_score("Make things better", None, None, "")
        assert s == 0.0

    def test_long_text_bonus(self):
        short_score = compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")
        long_text = "Build seed_gate.py with lots of extra context words that make it longer and more specific"
        long_score = compute_score(long_text, "build", "seed_gate.py", "file")
        assert long_score >= short_score

    def test_multi_target_bonus(self):
        single = compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")
        multi = compute_score("Build seed_gate.py and propose_seed.py together", "build", "seed_gate.py", "file")
        assert multi > single

    def test_score_bounds(self):
        s = compute_score("x", "build", "seed_gate.py", "file")
        assert 0.0 <= s <= 1.0

    def test_imperative_bonus(self):
        imperative = compute_score("Build seed_gate.py", "build", "seed_gate.py", "file")
        non_imperative = compute_score("The plan is to build seed_gate.py", "build", "seed_gate.py", "file")
        assert imperative >= non_imperative


# ===================================================================
# 7. Canonicalization
# ===================================================================

class TestCanonicalization:
    def test_strip_extension(self):
        assert canonicalize_target("seed_gate.py") == "seed_gate"

    def test_strip_path(self):
        assert canonicalize_target("src/seed_gate.py") == "seed_gate"

    def test_strip_quotes(self):
        assert canonicalize_target('"seed_gate"') == "seed_gate"

    def test_lowercase(self):
        assert canonicalize_target("Seed_Gate.py") == "seed_gate"


class TestCanonicalizeTarget:
    def test_all_prefixes(self):
        for prefix in ("src/", "tests/", "engine/", "state/", "docs/", "scripts/"):
            assert canonicalize_target(prefix + "foo.py") == "foo"


class TestSubstringDedup:
    def test_seed_gate_and_py(self):
        count = count_unique_targets("Build seed_gate and seed_gate.py")
        assert count == 1

    def test_different_targets(self):
        count = count_unique_targets("Build seed_gate.py and propose_seed.py")
        assert count == 2


# ===================================================================
# 8. Validation
# ===================================================================

class TestValidation:
    def test_pass_verb_plus_file(self):
        r = _v("Build seed_gate.py validator")
        assert r["passed"]

    def test_fail_no_verb(self):
        r = _v("The seed_gate.py module")
        assert not r["passed"]

    def test_fail_no_target(self):
        r = _v("Build something amazing and wonderful")
        assert not r["passed"]

    def test_exempt_tag_no_target(self):
        r = _v("Explore consciousness in the simulation", tags=["theme"])
        assert r["passed"]

    def test_tag_implied_verb(self):
        r = _v("The water_mining.py module needs love", tags=["code"])
        assert r["passed"]

    def test_question_stem_exempt(self):
        r = _v("What if consciousness emerges from data", tags=["philosophy"])
        assert r["passed"]

    def test_dict_shape(self):
        r = _v("Build seed_gate.py validator")
        assert "passed" in r
        assert "reasons" in r
        assert "score" in r
        assert "verb_found" in r
        assert "target_found" in r
        assert "junk" in r

    def test_score_is_float(self):
        r = _v("Build seed_gate.py validator")
        assert isinstance(r["score"], float)


class TestModeConsistency:
    def test_purge_mode_passes_everything(self):
        r = _v("Build seed_gate.py", mode="purge")
        assert r["passed"]

    def test_admission_rejects_vague(self):
        r = _v("Make everything better somehow please")
        assert not r["passed"]


class TestDataclassAPI:
    def test_result_is_frozen(self):
        r = _vs("Build seed_gate.py validator")
        with pytest.raises(AttributeError):
            r.passed = False

    def test_to_dict(self):
        r = _vs("Build seed_gate.py validator")
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["passed"] is True

    def test_verb_property(self):
        r = _vs("Build seed_gate.py validator")
        assert r.verb == "build"

    def test_target_property(self):
        r = _vs("Build seed_gate.py validator")
        assert "seed_gate" in r.target

    def test_confidence_high(self):
        r = _vs("Build seed_gate.py and propose_seed.py and water_mining.py for the colony pipeline system")
        assert r.confidence in ("high", "medium")

    def test_confidence_none_on_fail(self):
        r = _vs("Make everything better")
        assert r.confidence is None


# ===================================================================
# 9. Real-world proposals
# ===================================================================

class TestRealWorld:
    @pytest.mark.parametrize("text", [
        "Build water_mining.py optimizer for drilling efficiency",
        "Fix the solar_array.py power output calculation",
        "Test nuclear_reactor.py under high-load conditions",
        "Refactor dust_filter.py to reduce memory usage",
        "Deploy greenhouse.py crop rotation scheduler",
        "Add --verbose flag to the seed_gate CLI",
        "Integrate state_io into the feed pipeline",
        "Debug #12503 regression in fuel_cell.py",
        "Wire up propose_seed.py to seed_gate validator",
        "Create thermal_control.py PID controller",
    ])
    def test_real_proposals_pass(self, text):
        assert _v(text)["passed"], f"Should pass: {text}"

    @pytest.mark.parametrize("text", [
        "Make everything better",
        "Improve the system",
        "Think about stuff",
        "Do something cool",
        "Consider options",
    ])
    def test_vague_proposals_fail(self, text):
        assert not _v(text)["passed"], f"Should fail: {text}"


# ===================================================================
# 10. False-reject regression
# ===================================================================

class TestFalseRejectRegression:
    def test_lowercase_verb_start(self):
        assert _v("build seed_gate.py now")["passed"]

    def test_commit_prefix_fix(self):
        assert _v("fix: build seed_gate.py validator")["passed"]

    def test_commit_prefix_feat(self):
        assert _v("feat: add water_mining.py optimizer")["passed"]

    def test_inflected_verb(self):
        assert _v("Building seed_gate.py validator module")["passed"]

    def test_version_string_not_file(self):
        r = _v("Upgrade to version 2.0.1")
        assert r.get("target_found") != "2.0.1"

    def test_eg_not_file(self):
        r = _v("Build a module, e.g. seed_gate.py")
        assert r["passed"]
        assert r["target_found"] != "e.g"


# ===================================================================
# 11. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_empty_string(self):
        r = _v("")
        assert not r["passed"]
        assert r["junk"]

    def test_only_whitespace(self):
        r = _v("   ")
        assert not r["passed"]

    def test_very_long_text(self):
        r = _v("Build " + "word " * 500 + "seed_gate.py")
        assert r["passed"]

    def test_unicode(self):
        r = _v("Build seed_gate.py — the UTF-8 validator")
        assert r["passed"]

    def test_special_chars(self):
        r = _v("Build seed_gate.py (v2) [final]")
        assert r["passed"]


# ===================================================================
# 12. Invariants
# ===================================================================

class TestInvariants:
    @pytest.mark.parametrize("text", [
        "Build seed_gate.py", "Fix solar_array.py", "Test water_mining.py",
        "", "x", "Make things better", "the reactor works",
    ])
    def test_score_bounds(self, text):
        r = _v(text)
        assert 0.0 <= r["score"] <= 1.0

    def test_junk_implies_fail(self):
        r = _v("")
        assert r["junk"] and not r["passed"]

    def test_pass_implies_not_junk(self):
        r = _v("Build seed_gate.py validator")
        assert r["passed"] and not r["junk"]

    def test_passes_gate_matches_validate(self):
        for text in ["Build seed_gate.py", "Make stuff better", ""]:
            assert passes_gate(text) == _v(text)["passed"]


# ===================================================================
# 13. CLI
# ===================================================================

class TestCLI:
    def test_cli_pass(self):
        result = subprocess.run(
            [sys.executable, "-m", "seed_gate", "Build seed_gate.py validator"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT / "src"),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["passed"]

    def test_cli_fail(self):
        result = subprocess.run(
            [sys.executable, "-m", "seed_gate", "Make everything better and nicer"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT / "src"),
        )
        assert result.returncode == 1


# ===================================================================
# 14. Smoke test
# ===================================================================

class TestSmoke:
    def test_10_proposals(self):
        proposals = [
            "Build water_mining.py optimizer",
            "Fix the solar_array.py power bug",
            "Test nuclear_reactor.py safety",
            "Refactor dust_filter.py memory",
            "Deploy greenhouse.py scheduler",
            "Add --verbose to seed_gate",
            "Wire propose_seed.py to gate",
            "Debug #12503 fuel_cell.py",
            "Create thermal_control.py PID",
            "Optimize rover.py pathfinding",
        ]
        for p in proposals:
            r = _v(p)
            assert r["passed"], f"Smoke fail: {p}"
            assert 0.0 < r["score"] <= 1.0


# ===================================================================
# 15. propose_seed.py contract
# ===================================================================

class TestProposeSeedContract:
    def test_validate_returns_dict(self):
        r = validate("Build seed_gate.py", [])
        assert isinstance(r, dict)

    def test_dict_has_passed_key(self):
        assert "passed" in validate("Build seed_gate.py", [])

    def test_dict_has_score_key(self):
        assert "score" in validate("Build seed_gate.py", [])

    def test_passed_is_bool(self):
        assert isinstance(validate("Build seed_gate.py", [])["passed"], bool)


# ===================================================================
# 16. FILE_RE false positives
# ===================================================================

class TestFileReFalsePositives:
    @pytest.mark.parametrize("text", ["e.g.", "i.e.", "a.m.", "p.m.", "vs."])
    def test_abbreviation_not_file(self, text):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match(text)

    @pytest.mark.parametrize("text", ["2.0.1", "v1.2.3", "1.0"])
    def test_version_not_file(self, text):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match(text)


# ===================================================================
# 17. Special files
# ===================================================================

class TestSpecialFiles:
    @pytest.mark.parametrize("name", [
        "Dockerfile", "Makefile", "README", "AGENTS", "CLAUDE",
    ])
    def test_special_file_detected(self, name):
        t, k = find_target(f"Build {name} for the project")
        assert t == name and k == "file"


# ===================================================================
# 18. Known tools
# ===================================================================

class TestKnownTools:
    @pytest.mark.parametrize("tool", [
        "state_io", "process_inbox", "seed_gate", "github_llm",
    ])
    def test_known_tool_found(self, tool):
        t, k = find_target(f"Integrate {tool} into pipeline")
        assert tool in t


# ===================================================================
# 19. Question stems
# ===================================================================

class TestQuestionStems:
    @pytest.mark.parametrize("stem,verb", [
        ("What if consciousness emerges", "explore"),
        ("How might we improve the sim", "improve"),
        ("Should we reconsider the approach", "evaluate"),
    ])
    def test_stem_implied_verb(self, stem, verb):
        r = _vs(stem, tags=["philosophy"])
        assert r.verb_found == verb


# ===================================================================
# 20. Batch validation
# ===================================================================

class TestBatchValidation:
    def test_batch_basic(self):
        from seed_gate import validate_batch
        br = validate_batch([
            "Build seed_gate.py validator",
            "Make stuff better somehow please",
            "",
        ])
        assert br.stats.total == 3
        assert br.stats.passed == 1
        assert br.stats.junk >= 1

    def test_batch_empty(self):
        from seed_gate import validate_batch
        br = validate_batch([])
        assert br.stats.total == 0


class TestBatchStats:
    def test_pass_rate(self):
        from seed_gate import BatchStats
        s = BatchStats(total=10, passed=7, failed=2, junk=1)
        assert abs(s.pass_rate - 0.7) < 0.01

    def test_junk_rate(self):
        from seed_gate import BatchStats
        s = BatchStats(total=10, passed=7, failed=2, junk=1)
        assert abs(s.junk_rate - 0.1) < 0.01

    def test_merge(self):
        from seed_gate import BatchStats
        a = BatchStats(total=5, passed=3, failed=1, junk=1)
        b = BatchStats(total=5, passed=2, failed=2, junk=1)
        c = a.merge(b)
        assert c.total == 10 and c.passed == 5


# ===================================================================
# 21. Lowercase imperative
# ===================================================================

class TestLowercaseImperative:
    def test_lowercase_build(self):
        assert _v("build seed_gate.py now")["passed"]

    def test_lowercase_fix(self):
        assert _v("fix the solar_array.py bug")["passed"]


# ===================================================================
# 22. Phrasal verbs
# ===================================================================

class TestPhrasalVerbs:
    def test_set_up(self):
        assert find_verb("Set up the reactor") == "set up"

    def test_tear_down(self):
        assert find_verb("Tear down the old habitat") == "tear down"

    def test_wire_up(self):
        assert find_verb("Wire up seed_gate.py") == "wire up"


# ===================================================================
# 23. Tag-implied verbs
# ===================================================================

class TestTagImpliedVerbs:
    def test_code_tag(self):
        r = _vs("The water_mining.py module", tags=["code"])
        assert r.verb_found == "build"
        assert r.passed

    def test_test_tag(self):
        r = _vs("The seed_gate.py module", tags=["test"])
        assert r.verb_found == "test"


# ===================================================================
# 24. Advisory labels
# ===================================================================

class TestAdvisoryLabel:
    def test_needs_specificity(self):
        r = _vs("Build something cool and amazing", tags=["theme"])
        assert r.advisory == "needs-specificity" or r.passed

    def test_no_advisory_on_pass(self):
        r = _vs("Build seed_gate.py validator")
        assert r.advisory == "" or r.advisory == "needs-specificity"


# ===================================================================
# 25. Rich match info
# ===================================================================

class TestRichMatchInfo:
    def test_all_verbs_populated(self):
        r = _vs("Build and test water_mining.py")
        assert len(r.all_verbs) >= 2

    def test_all_targets_populated(self):
        r = _vs("Build water_mining.py and solar_array.py")
        assert len(r.all_targets) >= 2


# ===================================================================
# 26. CONST targets
# ===================================================================

class TestConstTargets:
    def test_const_detected(self):
        t, k = find_target("Expand ACTION_VERBS set")
        assert t == "ACTION_VERBS" and k == "const"


# ===================================================================
# 27. Case-insensitive modules
# ===================================================================

class TestCaseInsensitiveModules:
    def test_module_backtick(self):
        t, k = find_target("Update `water_mining` module")
        assert "water_mining" in t.lower()


# ===================================================================
# 28. Inflection map
# ===================================================================

class TestInflectionMap:
    def test_builds(self):
        from seed_gate import _INFLECTION_MAP
        assert _INFLECTION_MAP.get("builds") == "build"

    def test_creating(self):
        from seed_gate import _INFLECTION_MAP
        assert _INFLECTION_MAP.get("creating") == "create"

    def test_deployed(self):
        from seed_gate import _INFLECTION_MAP
        assert _INFLECTION_MAP.get("deployed") == "deploy"

    def test_built(self):
        from seed_gate import _INFLECTION_MAP
        assert _INFLECTION_MAP.get("built") == "build"


class TestInflectedVerbDetection:
    def test_building(self):
        assert find_verb("Building the reactor") == "build"

    def test_tests(self):
        assert find_verb("Tests the module") == "test"

    def test_optimized(self):
        assert find_verb("Optimized the drill code") == "optimize"


class TestInflectedPhrasalVerbs:
    def test_setting_up(self):
        assert find_verb("Setting up the reactor") == "set up"

    def test_scaling_up(self):
        assert find_verb("Scaling up the habitat") == "scale up"


# ===================================================================
# 29. Version filter
# ===================================================================

class TestVersionFilter:
    @pytest.mark.parametrize("v", ["2.0.1", "v1.2.3", "1.0", "3.0.0-beta"])
    def test_version_not_target(self, v):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match(v)


# ===================================================================
# 30. Confidence property
# ===================================================================

class TestConfidenceProperty:
    def test_high(self):
        r = _vs("Build seed_gate.py and propose_seed.py and water_mining.py for the colony pipeline system")
        if r.passed:
            assert r.confidence in ("high", "medium")

    def test_none_on_fail(self):
        r = _vs("Make everything better")
        assert r.confidence is None


# ===================================================================
# 31. Suggest API
# ===================================================================

class TestSuggestAPI:
    def test_no_verb_suggestion(self):
        tips = suggest("The seed_gate.py module is important")
        assert any("verb" in t.lower() for t in tips)

    def test_no_target_suggestion(self):
        tips = suggest("Build something amazing and wonderful")
        assert any("target" in t.lower() for t in tips)

    def test_empty_on_pass(self):
        assert suggest("Build seed_gate.py validator") == []


# ===================================================================
# 32. Env var targets
# ===================================================================

class TestEnvVarTargets:
    def test_dollar_env(self):
        t, k = find_target("Read $STATE_DIR variable")
        assert k == "env"

    def test_braced_env(self):
        t, k = find_target("Set ${GITHUB_TOKEN} value")
        assert k == "env"


# ===================================================================
# 33. Expanded known tools
# ===================================================================

class TestExpandedKnownTools:
    def test_vlink(self):
        from seed_gate import KNOWN_TOOLS
        assert "vlink" in KNOWN_TOOLS

    def test_steer(self):
        from seed_gate import KNOWN_TOOLS
        assert "steer" in KNOWN_TOOLS


# ===================================================================
# 34. Imperative bonus
# ===================================================================

class TestImperativeBonus:
    def test_imperative_higher_score(self):
        imp = _v("Build seed_gate.py validator")["score"]
        non = _v("The plan is to build seed_gate.py module")["score"]
        assert imp >= non


# ===================================================================
# 35. Noun-use false positives (verbs used as nouns)
# ===================================================================

class TestNounUseFalsePositives:
    def test_map_noun(self):
        # "map" as noun should not prevent finding real verb
        r = _v("Build a site map for seed_gate.py")
        assert r["passed"]


# ===================================================================
# 36. Inflection invariants
# ===================================================================

class TestInflectionInvariants:
    def test_all_base_verbs_have_inflections(self):
        from seed_gate import _INFLECTION_MAP
        for verb in ACTION_VERBS:
            has_any = any(base == verb for base in _INFLECTION_MAP.values())
            assert has_any, f"{verb} has no inflected forms"


# ===================================================================
# 37. Commit prefix
# ===================================================================

class TestCommitPrefix:
    def test_strip_fix(self):
        assert _strip_commit_prefix("fix: build seed_gate.py") == "build seed_gate.py"

    def test_strip_feat_scope(self):
        assert _strip_commit_prefix("feat(gate): add validator") == "add validator"

    def test_no_strip_uppercase(self):
        assert _strip_commit_prefix("Build: the reactor") == "Build: the reactor"


class TestCommitPrefixRegression:
    def test_fix_build_passes(self):
        assert _v("fix: build seed_gate.py validator")["passed"]

    def test_feat_add_passes(self):
        assert _v("feat: add water_mining.py optimizer")["passed"]

    def test_chore_update_passes(self):
        assert _v("chore: update propose_seed.py contract")["passed"]


# ===================================================================
# 38. find_verb_with_position
# ===================================================================

class TestFindVerbWithPosition:
    def test_first_word(self):
        v, p = find_verb_with_position("Build the reactor core")
        assert v == "build" and p == 0

    def test_second_word(self):
        v, p = find_verb_with_position("Please build seed_gate.py")
        assert v == "build" and p == 1

    def test_none(self):
        v, p = find_verb_with_position("The reactor is hot today")
        assert v is None and p is None

    def test_phrasal(self):
        v, p = find_verb_with_position("Set up the reactor core")
        assert v == "set up" and p == 0


# ===================================================================
# 39. Score breakdown
# ===================================================================

class TestScoreBreakdown:
    def test_has_verb_component(self):
        sb = score_breakdown("Build seed_gate.py validator")
        assert sb["verb"] == 2.5

    def test_has_target_component(self):
        sb = score_breakdown("Build seed_gate.py validator")
        assert sb["target"] > 0

    def test_has_total(self):
        sb = score_breakdown("Build seed_gate.py validator")
        assert sb["total"] > 0

    def test_total_is_sum(self):
        sb = score_breakdown("Build seed_gate.py validator")
        expected = sb["verb"] + sb["target"] + sb["length"] + sb["multi_target"] + sb["imperative"]
        assert abs(sb["total"] - expected) < 0.01


# ===================================================================
# 40. Explain
# ===================================================================

class TestExplain:
    def test_pass_in_output(self):
        assert "PASS" in explain("Build seed_gate.py validator")

    def test_fail_in_output(self):
        assert "FAIL" in explain("Make everything better and nicer")

    def test_contains_verb(self):
        assert "verb=build" in explain("Build seed_gate.py validator")

    def test_contains_score(self):
        assert "score=" in explain("Build seed_gate.py validator")

    def test_contains_strength(self):
        e = explain("Build seed_gate.py validator")
        assert "strength=" in e


# ===================================================================
# 41. New API invariants
# ===================================================================

class TestNewAPIInvariants:
    def test_find_verb_with_position_returns_tuple(self):
        r = find_verb_with_position("Build the reactor")
        assert isinstance(r, tuple) and len(r) == 2

    def test_position_valid_when_verb_found(self):
        v, p = find_verb_with_position("Please build seed_gate.py")
        if v:
            assert isinstance(p, int) and p >= 0

    def test_score_breakdown_total_matches_compute_score(self):
        text = "Build seed_gate.py validator"
        sb = score_breakdown(text)
        r = _v(text)
        expected = min(sb["total"] / 10.0, 1.0)
        assert abs(r["score"] - expected) < 0.01

    def test_explain_always_returns_string(self):
        assert isinstance(explain("Build seed_gate.py"), str)

    def test_explain_contains_pass_or_fail(self):
        e = explain("Build seed_gate.py validator")
        assert "PASS" in e or "FAIL" in e

    def test_kind_scores_constant_matches(self):
        assert _KIND_SCORES["file"] == 4.0

    def test_redesign_in_action_verbs(self):
        assert "redesign" in ACTION_VERBS

    def test_commit_prefix_seeds_pass(self):
        assert _v("fix: build seed_gate.py validator")["passed"]

    def test_find_verb_agrees_with_find_verb_with_position(self):
        texts = ["Build seed_gate.py", "The reactor is hot", "Setting up the reactor"]
        for text in texts:
            v1 = find_verb(text)
            v2, _ = find_verb_with_position(text)
            assert v1 == v2, f"Mismatch for {text!r}: {v1} vs {v2}"


# ===================================================================
# 42. Negation awareness (PR #289 consolidation)
# ===================================================================

class TestContractionExpansion:
    def test_dont(self):
        assert "do not" in _expand_contractions("don't build it")

    def test_cant(self):
        assert "cannot" in _expand_contractions("can't deploy this")

    def test_wont(self):
        assert "will not" in _expand_contractions("won't merge it")

    def test_doesnt(self):
        assert "does not" in _expand_contractions("doesn't work")

    def test_no_contraction(self):
        assert _expand_contractions("Build the reactor") == "Build the reactor"

    def test_case_insensitive(self):
        result = _expand_contractions("Don't build it")
        assert "not" in result.lower()

    def test_multiple_contractions(self):
        result = _expand_contractions("Don't build and can't deploy")
        assert "do not" in result
        assert "cannot" in result


class TestNegationDetection:
    def test_not_before_verb(self):
        words = ["do", "not", "build", "the", "reactor"]
        assert _is_negated(words, 2)

    def test_never_before_verb(self):
        words = ["never", "deploy", "the", "reactor"]
        assert _is_negated(words, 1)

    def test_no_negation(self):
        words = ["build", "the", "reactor", "now"]
        assert not _is_negated(words, 0)

    def test_clause_boundary_blocks(self):
        words = ["not", "this", "but", "build", "the", "reactor"]
        assert not _is_negated(words, 3)

    def test_far_negation_ignored(self):
        words = ["not", "a", "b", "c", "build", "the", "reactor"]
        assert not _is_negated(words, 4)

    def test_without_before_verb(self):
        words = ["without", "building", "the", "reactor"]
        assert _is_negated(words, 1)


class TestNegatedVerbFinding:
    def test_dont_build_no_verb(self):
        assert find_verb("Don't build the reactor core") is None

    def test_never_deploy_no_verb(self):
        assert find_verb("Never deploy untested code modules") is None

    def test_cant_merge_no_verb(self):
        assert find_verb("Can't merge this pull request now") is None

    def test_wont_ship_no_verb(self):
        assert find_verb("Won't ship the release this week") is None

    def test_negation_doesnt_block_later_verb(self):
        v = find_verb("Don't worry, just build seed_gate.py")
        assert v == "build"

    def test_build_system_that_doesnt_crash(self):
        v = find_verb("Build a system that doesn't crash easily")
        assert v == "build"

    def test_not_yet_but_will_build(self):
        v = find_verb("Not yet done, but build seed_gate.py")
        assert v == "build"

    def test_negated_validation_fails(self):
        r = _v("Don't deploy the reactor core module today")
        assert not r["passed"]

    def test_negated_reason_is_distinct(self):
        r = _v("Don't deploy the reactor core module today")
        assert any("negat" in reason.lower() for reason in r["reasons"])

    def test_negated_advisory(self):
        r = _vs("Don't deploy the reactor core module today")
        assert "negated-intent" in r.advisories


class TestNegatedSuggestions:
    def test_negated_verb_suggestion(self):
        tips = suggest("Don't deploy the reactor core module")
        assert any("negat" in t.lower() or "rephrase" in t.lower() for t in tips)


# ===================================================================
# 43. Compound-name filtering (PR #289 consolidation)
# ===================================================================

class TestCompoundFiltering:
    def test_is_in_compound_underscore(self):
        assert _is_in_compound("run_proof.py", "run", 0)

    def test_is_in_compound_hyphen(self):
        assert _is_in_compound("pre-build step", "build", 4)

    def test_is_in_compound_slash(self):
        assert _is_in_compound("scripts/deploy", "deploy", 8)

    def test_is_in_compound_dot(self):
        assert _is_in_compound("pkg.build_tool", "build", 4)

    def test_not_in_compound_space(self):
        assert not _is_in_compound("Build the reactor", "Build", 0)

    def test_compound_verb_not_found(self):
        # "run" inside "run_proof" should not be found as verb
        # but "validates" should still be found
        v = find_verb("The run_proof.py script validates math")
        assert v == "validate"

    def test_real_verb_after_compound(self):
        v = find_verb("The run_proof.py script validates math correctly")
        assert v == "validate"

    def test_compound_in_path(self):
        # "deploy" inside "scripts/deploy" should not be the verb
        v = find_verb("Check scripts/deploy for issues then fix it")
        assert v == "fix"

    def test_compound_kebab(self):
        v = find_verb("The pre-build step needs a fix now")
        assert v == "fix"


# ===================================================================
# 44. VerbMatch dataclass (PR #289 consolidation)
# ===================================================================

class TestVerbMatch:
    def test_fields(self):
        vm = VerbMatch(verb="build", token_index=0, source="base")
        assert vm.verb == "build"
        assert vm.token_index == 0
        assert vm.source == "base"
        assert vm.negated is False

    def test_frozen(self):
        vm = VerbMatch(verb="build", token_index=0)
        with pytest.raises(AttributeError):
            vm.verb = "test"

    def test_negated_field(self):
        vm = VerbMatch(verb="build", token_index=0, negated=True)
        assert vm.negated is True


class TestFindVerbMatch:
    def test_returns_verbmatch(self):
        m = find_verb_match("Build seed_gate.py validator")
        assert isinstance(m, VerbMatch)
        assert m.verb == "build"
        assert m.token_index == 0
        assert m.source == "base"

    def test_inflected_source(self):
        m = find_verb_match("Building the reactor core now")
        assert m is not None
        assert m.verb == "build"
        assert m.source == "inflected"

    def test_phrasal_source(self):
        m = find_verb_match("Set up the reactor core now")
        assert m is not None
        assert m.verb == "set up"
        assert m.source == "phrasal"

    def test_none_when_no_verb(self):
        assert find_verb_match("The reactor is hot today") is None

    def test_none_when_negated(self):
        assert find_verb_match("Don't build the reactor core") is None

    def test_skips_compound(self):
        m = find_verb_match("The run_proof.py validates math")
        assert m is not None
        assert m.verb == "validate"


# ===================================================================
# 45. Unified _scan_verbs (PR #289 consolidation)
# ===================================================================

class TestScanVerbs:
    def test_finds_base_verb(self):
        matches = _scan_verbs("Build the reactor core")
        assert len(matches) >= 1
        assert matches[0].verb == "build"
        assert matches[0].source == "base"

    def test_finds_inflected(self):
        matches = _scan_verbs("Building the reactor core")
        assert len(matches) >= 1
        assert matches[0].verb == "build"
        assert matches[0].source == "inflected"

    def test_finds_phrasal(self):
        matches = _scan_verbs("Set up the reactor core")
        assert len(matches) >= 1
        assert matches[0].verb == "set up"
        assert matches[0].source == "phrasal"

    def test_marks_negated(self):
        matches = _scan_verbs("Do not build the reactor")
        negated = [m for m in matches if m.negated]
        assert len(negated) >= 1
        assert negated[0].verb == "build"

    def test_skips_compound(self):
        matches = _scan_verbs("The run_proof.py validates math")
        verbs = [m.verb for m in matches]
        assert "run" not in verbs
        assert "validate" in verbs

    def test_limit(self):
        matches = _scan_verbs("Build the reactor now", limit=3)
        assert len(matches) == 0

    def test_empty(self):
        assert _scan_verbs("") == []

    def test_multiple_verbs(self):
        matches = _scan_verbs("Build and test the water_mining.py module")
        verbs = [m.verb for m in matches]
        assert "build" in verbs
        assert "test" in verbs


# ===================================================================
# 46. Numbered-reference filter (PR #289 consolidation)
# ===================================================================

class TestNumberedRefFilter:
    @pytest.mark.parametrize("ref", ["fig.1", "vol.2", "ch.3", "no.5", "pt.1", "sec.4"])
    def test_numbered_ref_not_file(self, ref):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match(ref), f"{ref} should be filtered"

    def test_fig_not_target(self):
        t, k = find_target("Build system as shown in fig.1 diagram")
        assert t != "fig.1"

    def test_real_file_still_works(self):
        t, k = find_target("Build config.py for the module")
        assert t == "config.py" and k == "file"


# ===================================================================
# 47. Multi-target scaling (PR #289 consolidation)
# ===================================================================

class TestMultiTargetScaling:
    def test_two_targets(self):
        sb = score_breakdown("Build seed_gate.py and propose_seed.py")
        assert sb["multi_target"] == 1.0

    def test_three_targets(self):
        sb = score_breakdown("Build seed_gate.py and propose_seed.py and water_mining.py")
        assert sb["multi_target"] == 1.5

    def test_four_targets(self):
        sb = score_breakdown("Build seed_gate.py and propose_seed.py and water_mining.py and solar_array.py")
        assert sb["multi_target"] == 2.0

    def test_single_target_no_bonus(self):
        sb = score_breakdown("Build seed_gate.py validator")
        assert sb["multi_target"] == 0.0


# ===================================================================
# 48. Enriched SeedGateResult (PR #289 consolidation)
# ===================================================================

class TestEnrichedResult:
    def test_target_kind(self):
        r = _vs("Build seed_gate.py validator")
        assert r.target_kind == "file"

    def test_verb_source_base(self):
        r = _vs("Build seed_gate.py validator")
        assert r.verb_source == "base"

    def test_verb_source_inflected(self):
        r = _vs("Building seed_gate.py validator")
        assert r.verb_source == "inflected"

    def test_verb_source_tag_implied(self):
        r = _vs("The water_mining.py module", tags=["code"])
        assert r.verb_source == "tag-implied"

    def test_verb_position(self):
        r = _vs("Build seed_gate.py validator")
        assert r.verb_position == 0

    def test_score_parts_populated(self):
        r = _vs("Build seed_gate.py validator")
        assert len(r.score_parts) > 0
        parts_dict = dict(r.score_parts)
        assert "verb" in parts_dict

    def test_is_imperative(self):
        r = _vs("Build seed_gate.py validator")
        assert r.is_imperative is True

    def test_not_imperative(self):
        r = _vs("The plan is to build seed_gate.py module")
        assert r.is_imperative is False

    def test_to_dict_has_new_fields(self):
        r = _vs("Build seed_gate.py validator")
        d = r.to_dict()
        assert "target_kind" in d
        assert "verb_source" in d
        assert "verb_position" in d
        assert "score_parts" in d
        assert "is_imperative" in d
        assert "strength" in d
        assert "advisories" in d


# ===================================================================
# 49. Strength tiers (PR #289 consolidation)
# ===================================================================

class TestStrength:
    def test_rejected_on_fail(self):
        r = _vs("Make everything better")
        assert r.strength == "rejected"

    def test_strong_on_high_score(self):
        r = _vs("Build seed_gate.py and propose_seed.py and water_mining.py for the colony pipeline system")
        if r.passed and r.score >= 0.65:
            assert r.strength == "strong"

    def test_weak_on_low_score(self):
        r = _vs("Explore consciousness", tags=["theme"])
        if r.passed and r.score < 0.35:
            assert r.strength == "weak"

    def test_strength_in_to_dict(self):
        r = _vs("Build seed_gate.py validator")
        assert "strength" in r.to_dict()


# ===================================================================
# 50. Advisories (PR #289 consolidation)
# ===================================================================

class TestAdvisories:
    def test_advisories_tuple(self):
        r = _vs("Build seed_gate.py validator")
        assert isinstance(r.advisories, tuple)

    def test_negated_has_advisory(self):
        r = _vs("Don't deploy the reactor core module today")
        assert "negated-intent" in r.advisories

    def test_needs_specificity_advisory(self):
        r = _vs("Build something amazing", tags=["theme"])
        if r.passed and not r.target_found:
            assert "needs-specificity" in r.advisories

    def test_advisories_in_to_dict(self):
        r = _vs("Build seed_gate.py validator")
        d = r.to_dict()
        assert isinstance(d["advisories"], list)


# ===================================================================
# 51. Property-based invariants (PR #289 additions)
# ===================================================================

class TestPropertyInvariants:
    @pytest.mark.parametrize("text", [
        "Build seed_gate.py", "Don't deploy reactor", "run_proof.py validates",
        "Set up the reactor", "Building seed_gate.py", "", "x",
        "fig.1 shows the data", "Never merge untested code",
    ])
    def test_find_verb_and_find_verb_match_agree(self, text):
        v = find_verb(text)
        m = find_verb_match(text)
        if m:
            assert v == m.verb
        else:
            assert v is None

    @pytest.mark.parametrize("text", [
        "Build seed_gate.py", "Don't deploy reactor", "run_proof.py validates",
    ])
    def test_find_verb_and_position_agree(self, text):
        v1 = find_verb(text)
        v2, _ = find_verb_with_position(text)
        assert v1 == v2

    @pytest.mark.parametrize("text", [
        "Build seed_gate.py", "Fix solar_array.py", "Don't build reactor",
        "", "Make things better", "the reactor works",
    ])
    def test_strength_consistent_with_passed(self, text):
        r = _vs(text)
        if r.passed:
            assert r.strength in ("strong", "moderate", "weak")
        else:
            assert r.strength == "rejected"

    def test_score_parts_sum_matches_total(self):
        r = _vs("Build seed_gate.py and propose_seed.py for the colony")
        parts = dict(r.score_parts)
        if "total" in parts:
            non_total = sum(v for k, v in parts.items() if k != "total")
            assert abs(parts["total"] - non_total) < 0.01

    def test_negated_verb_means_no_verb_found(self):
        r = _vs("Don't deploy the reactor core module today")
        # If all verbs are negated, verb_found should be None
        if not r.passed and "negat" in str(r.reasons).lower():
            assert r.verb_found is None


# ===================================================================
# 52. Backward compatibility
# ===================================================================

class TestBackwardCompat:
    def test_advisory_still_string(self):
        r = _vs("Build seed_gate.py validator")
        assert isinstance(r.advisory, str)

    def test_advisory_and_advisories_consistent(self):
        r = _vs("Build something amazing", tags=["theme"])
        if r.advisory:
            assert r.advisory in r.advisories

    def test_old_dict_keys_present(self):
        d = _v("Build seed_gate.py validator")
        for key in ("passed", "reasons", "score", "verb_found", "target_found",
                     "junk", "advisory", "confidence", "all_verbs", "all_targets"):
            assert key in d, f"Missing key: {key}"

    def test_scan_verbs_not_in_public_api(self):
        """_scan_verbs is internal (underscore prefix)."""
        assert _scan_verbs.__name__ == "_scan_verbs"

    def test_validate_returns_dict(self):
        assert isinstance(validate("Build seed_gate.py", []), dict)

    def test_validate_seed_returns_dataclass(self):
        assert isinstance(validate_seed("Build seed_gate.py"), SeedGateResult)

    def test_passes_gate_returns_bool(self):
        assert isinstance(passes_gate("Build seed_gate.py"), bool)
