"""Tests for seed_gate.py -- canonical specificity validator.

Covers: verb detection, target detection (files, paths, tools, modules,
CLI, discussions, channels, quoted), junk detection (hard + soft artifacts),
scoring with unique-target counting, validation pass/fail, exempt tags,
CLI, real-world proposals, edge cases, property invariants, smoke tests,
propose_seed.py contract, and regression tests for false rejects.
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
    passes_gate,
    validate,
    validate_seed,
    canonicalize_target,
    count_unique_targets,
    is_soft_artifact,
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
    def test_file_re_python(self):
        assert FILE_RE.search("Build seed_gate.py validator")

    def test_file_re_json(self):
        assert FILE_RE.search("Fix state/agents.json integrity")

    def test_file_re_rust(self):
        assert FILE_RE.search("Port parser to grammar.rs")

    def test_path_re_src(self):
        assert PATH_RE.search("Refactor src/water_mining module")

    def test_path_re_tests(self):
        assert PATH_RE.search("Add tests/test_drill coverage")

    def test_path_re_engine(self):
        assert PATH_RE.search("Fix engine/tick overflow")

    def test_path_re_state(self):
        assert PATH_RE.search("Validate state/mars integrity")

    def test_path_re_docs(self):
        assert PATH_RE.search("Update docs/index.html layout")

    def test_tool_re_snake(self):
        assert TOOL_RE.search("Refactor process_inbox handler")

    def test_tool_re_no_single(self):
        assert not TOOL_RE.search("Build parser module")

    def test_cli_re_backtick(self):
        assert CLI_RE.search("Run `pytest -v` on CI")

    def test_cli_re_flag(self):
        assert CLI_RE.search("Add --verbose flag")

    def test_cli_re_short_flag(self):
        assert CLI_RE.search("Support -f option")

    def test_discussion_re(self):
        assert DISCUSSION_RE.search("See discussion #12503")

    def test_discussion_re_no_short(self):
        assert not DISCUSSION_RE.search("Issue #42 is important")

    def test_channel_re_r(self):
        assert CHANNEL_RE.search("Create r/mars-engineering")

    def test_channel_re_c(self):
        assert CHANNEL_RE.search("Post in c/general channel")

    def test_quoted_re(self):
        assert QUOTED_RE.search('Add "thermal control loop" feature')

    def test_module_context_backtick(self):
        assert MODULE_CONTEXT_RE.search("Fix `water_mining` bug")

    def test_module_context_import(self):
        assert MODULE_CONTEXT_RE.search("import solar_array works")

    def test_module_context_from(self):
        assert MODULE_CONTEXT_RE.search("from dust_storm import x")


# ===================================================================
# 3. Verb detection
# ===================================================================

class TestVerbDetection:
    @pytest.mark.parametrize("verb", ["build", "create", "fix", "test", "ship", "deploy"])
    def test_common_verbs(self, verb):
        result = _v(f"{verb.capitalize()} the water_mining.py system entirely")
        assert result["verb_found"] == verb

    def test_case_insensitive(self):
        result = _v("BUILD water_mining.py from scratch again")
        assert result["verb_found"] == "build"

    def test_no_verb(self):
        result = _v("Water mining system with multiple features included")
        assert result["verb_found"] is None

    def test_verb_mid_sentence(self):
        result = _v("The team should build water_mining.py carefully")
        assert result["verb_found"] == "build"


# ===================================================================
# 4. Target detection
# ===================================================================

class TestTargetDetection:
    def test_file_target(self):
        result = _v("Build water_mining.py optimizer quickly now")
        assert result["target_found"] == "water_mining.py"

    def test_path_target(self):
        result = _v("Fix src/thermal_control module carefully here")
        assert result["target_found"] == "src/thermal_control"

    def test_tool_target(self):
        result = _v("Build the process_inbox handler properly here")
        assert result["target_found"] == "process_inbox"

    def test_cli_target(self):
        result = _v("Test with `pytest -v` for better output here")
        assert result["target_found"] == "`pytest -v`"

    def test_discussion_target(self):
        result = _v("Build feature from discussion #12503 ideas")
        assert result["target_found"] == "#12503"

    def test_channel_target(self):
        result = _v("Create content for r/mars-engineering now")
        assert result["target_found"] == "r/mars-engineering"

    def test_quoted_target(self):
        result = _v('Implement "thermal control loop" system now')
        assert result["target_found"] == '"thermal control loop"'

    def test_no_target(self):
        result = _v("Build something amazing and wonderful today")
        assert result["target_found"] is None

    def test_file_beats_tool(self):
        result = _v("Build water_mining.py with process_inbox")
        assert result["target_found"] == "water_mining.py"

    def test_module_context_backtick(self):
        if "water_mining" in KNOWN_MODULES:
            result = _v("Fix `water_mining` efficiency problem")
            assert result["target_found"] == "water_mining"


# ===================================================================
# 5. Junk detection
# ===================================================================

class TestJunkDetection:
    def test_empty(self):
        assert _v("")["junk"] is True

    def test_whitespace(self):
        assert _v("   \n  ")["junk"] is True

    def test_too_short(self):
        assert _v("Build it")["junk"] is True

    def test_starts_lowercase_verb_allowed(self):
        # Lowercase text starting with an action verb is NOT junk
        assert _v("build something really cool and interesting here now")["junk"] is False

    def test_starts_lowercase_fragment(self):
        # Lowercase text NOT starting with a verb IS junk
        assert _v("the thing we talked about doing sometime later")["junk"] is True

    def test_starts_backtick(self):
        assert _v("`code fragment` extracted from somewhere else")["junk"] is True

    def test_numbered_list(self):
        assert _v("1. First item in a numbered list of tasks")["junk"] is True

    def test_bare_url(self):
        assert _v("https://example.com/path/to/something")["junk"] is True

    def test_todo_comment(self):
        assert _v("TODO: Fix the water_mining module bug here")["junk"] is True

    def test_run_prefix_exception(self):
        assert _v("run_proof for water_mining.py system verify")["junk"] is False

    def test_hard_artifact_parser_grabbed(self):
        r = _v("Parser grabbed this fragment from the document")
        assert r["junk"] is True
        assert "artifact" in r["reasons"][0]

    def test_hard_artifact_parsing_artifact(self):
        assert _v("Parsing artifact detected in the pipeline")["junk"] is True

    def test_hard_artifact_outside_grammar(self):
        assert _v("Outside that grammar the token has no meaning")["junk"] is True


# ===================================================================
# 6. Soft artifact detection
# ===================================================================

class TestSoftArtifacts:
    def test_redeemed_by_verb_target(self):
        result = _v("Fix the regex in water_mining.py for edges")
        assert result["passed"] is True

    def test_unredeemed_fails(self):
        result = _v("The regex pattern is broken and needs attention")
        assert result["passed"] is False
        assert result["junk"] is True

    def test_the_parser_unredeemed(self):
        assert _v("The parser module handles tokenization inputs")["passed"] is False

    def test_substring_unredeemed(self):
        assert _v("Substring extraction failed in data processing")["passed"] is False

    def test_redeemed_parser_in_file(self):
        result = _v("Fix the parser in seed_gate.py for extraction")
        assert result["passed"] is True


# ===================================================================
# 7. Scoring
# ===================================================================

class TestScoring:
    def test_high_score_file(self):
        r = _v("Build water_mining.py optimizer with advanced drilling and scanning algorithms")
        assert r["score"] >= 0.7

    def test_zero_score_junk(self):
        assert _v("")["score"] == 0.0

    def test_verb_only_low(self):
        r = _v("Build something amazing and wonderful for everyone")
        assert 0.0 < r["score"] <= 0.5

    def test_multi_target_bonus(self):
        single = _v("Build water_mining.py optimizer carefully here")
        multi = _v("Build water_mining.py and solar_array.py integration")
        assert multi["score"] >= single["score"]

    def test_score_capped(self):
        text = "Build " + " and ".join(f"mod_{i}.py" for i in range(20))
        assert _v(text)["score"] <= 1.0

    def test_path_scores_high(self):
        assert _v("Fix src/thermal_control module for efficiency")["score"] >= 0.5


# ===================================================================
# 8. Canonicalization
# ===================================================================

class TestCanonicalization:
    def test_strip_prefix(self):
        assert canonicalize_target("src/water_mining.py") == "water_mining"

    def test_strip_extension(self):
        assert canonicalize_target("water_mining.py") == "water_mining"

    def test_strip_quotes(self):
        assert canonicalize_target('"water_mining"') == "water_mining"

    def test_lowercase(self):
        assert canonicalize_target("Water_Mining.PY") == "water_mining"

    def test_same_forms_canonicalize(self):
        assert canonicalize_target("src/water_mining.py") == canonicalize_target("water_mining.py")

    def test_count_unique_deduplicates(self):
        text = "Fix water_mining.py and src/water_mining module"
        assert count_unique_targets(text) >= 1

    def test_count_unique_multiple(self):
        text = "Build water_mining.py and solar_array.py"
        assert count_unique_targets(text) >= 2

    def test_count_unique_empty(self):
        assert count_unique_targets("Nothing specific mentioned") == 0


# ===================================================================
# 9. Validation pass/fail
# ===================================================================

class TestValidation:
    def test_pass_verb_file(self):
        assert _v("Build water_mining.py optimizer for drilling")["passed"]

    def test_pass_verb_tool(self):
        assert _v("Fix process_inbox handler for delta processing")["passed"]

    def test_pass_verb_path(self):
        assert _v("Refactor src/thermal_control for modularity")["passed"]

    def test_pass_verb_discussion(self):
        assert _v("Build feature from discussion #12503 plan")["passed"]

    def test_pass_verb_channel(self):
        assert _v("Create content for r/mars-engineering now")["passed"]

    def test_fail_no_verb(self):
        r = _v("Water_mining.py optimizer for deep drilling etc")
        assert not r["passed"]
        assert any("verb" in s.lower() for s in r["reasons"])

    def test_fail_no_target(self):
        r = _v("Build something amazing and wonderful for all")
        assert not r["passed"]
        assert any("target" in s.lower() for s in r["reasons"])

    def test_fail_both_missing(self):
        r = _v("Something amazing and wonderful for everyone")
        assert not r["passed"]
        assert len(r["reasons"]) >= 2

    def test_exempt_tag_no_target(self):
        assert _v("Explore consciousness in artificial agents", ["theme"])["passed"]

    def test_exempt_tag_needs_verb(self):
        assert not _v("The nature of consciousness in AI systems", ["theme"])["passed"]

    def test_exempt_case_insensitive(self):
        assert _v("Explore consciousness in agents deeply", ["THEME"])["passed"]

    def test_exempt_multiple_tags(self):
        assert _v("Debate ethics of Mars terraforming now", ["philosophy", "debate"])["passed"]


# ===================================================================
# 10. Mode: purge vs admission
# ===================================================================

class TestModeConsistency:
    def test_admission_detects_verb(self):
        assert _v("Build seed_gate.py validator for checking")["verb_found"] == "build"

    def test_purge_detects_verb(self):
        assert _v("Build seed_gate.py validator checking", mode="purge")["verb_found"] == "build"

    def test_purge_always_passes_nonjunk(self):
        assert _v("Something without a verb or target long enough", mode="purge")["passed"]

    def test_admission_rejects_no_target(self):
        assert not _v("Build something amazing wonderful for all")["passed"]

    def test_purge_score_half(self):
        assert _v("Build seed_gate.py validator quickly", mode="purge")["score"] == 0.5


# ===================================================================
# 11. Dataclass API
# ===================================================================

class TestDataclassAPI:
    def test_frozen(self):
        r = _vs("Build seed_gate.py validator for system")
        with pytest.raises(AttributeError):
            r.passed = False

    def test_verb_property(self):
        assert _vs("Build seed_gate.py validator system").verb == "build"

    def test_target_property(self):
        assert _vs("Build seed_gate.py validator system").target == "seed_gate.py"

    def test_verb_property_none(self):
        assert _vs("Water_mining.py optimizer for drilling").verb == ""

    def test_to_dict_matches(self):
        text = "Build seed_gate.py validator for system"
        assert _vs(text).to_dict() == _v(text)


# ===================================================================
# 12. Real-world proposals
# ===================================================================

class TestRealWorld:
    def test_canonical_seed(self):
        assert _v("Build seed_gate.py with action verb validation")["passed"]

    def test_mars_module(self):
        assert _v("Optimize water_mining.py deep well drilling")["passed"]

    def test_engine_refactor(self):
        assert _v("Refactor engine/tick.py for parallel streams")["passed"]

    def test_discussion_ref(self):
        assert _v("Implement proposal from discussion #12503")["passed"]

    def test_multi_file(self):
        r = _v("Wire seed_gate.py into propose_seed.py validation")
        assert r["passed"]
        assert r["score"] >= 0.6

    def test_vague_rejected(self):
        assert not _v("Make everything better and more amazing")["passed"]

    def test_philosophy_exempt(self):
        assert _v("Explore what it means for agents to dream", ["philosophy"])["passed"]


# ===================================================================
# 13. Regression: false rejects
# ===================================================================

class TestFalseRejectRegression:
    def test_fix_regex_in_file(self):
        assert _v("Fix the regex in water_mining.py edge cases")["passed"]

    def test_refactor_parser_in_file(self):
        assert _v("Refactor the parser in seed_gate.py extraction")["passed"]

    def test_substring_in_module(self):
        assert _v("Investigate substring in dust_filter.py edges")["passed"]

    def test_cli_flag_proposal(self):
        assert _v("Add --dry-run flag to process_inbox testing")["passed"]

    def test_backtick_command(self):
        assert _v("Test with `python3 -m pytest tests/` full")["passed"]


# ===================================================================
# 14. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_unicode(self):
        assert _v("Build water_mining.py with UTF-8 日本語")["passed"]

    def test_very_long(self):
        text = "Build seed_gate.py " + "x " * 5000
        r = _v(text)
        assert r["passed"]
        assert r["score"] <= 1.0

    def test_newlines(self):
        assert _v("Build water_mining.py\noptimizer for deep\ndrilling")["passed"]

    def test_none_tags(self):
        assert validate("Build water_mining.py optimizer drill", None)["passed"]

    def test_empty_tags(self):
        assert validate("Build water_mining.py optimizer drill", [])["passed"]

    def test_deterministic(self):
        text = "Build water_mining.py optimizer deep drilling"
        assert _v(text) == _v(text)


# ===================================================================
# 15. Property invariants
# ===================================================================

_CORPUS = [
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
    "Fix the regex in water_mining.py for edges",
    "Parser grabbed this from extraction pipeline",
    "Optimize src/thermal_control for efficiency",
]


class TestInvariants:
    @pytest.mark.parametrize("text", _CORPUS)
    def test_score_in_range(self, text):
        assert 0.0 <= _v(text)["score"] <= 1.0

    @pytest.mark.parametrize("text", _CORPUS)
    def test_junk_is_bool(self, text):
        assert isinstance(_v(text)["junk"], bool)

    @pytest.mark.parametrize("text", _CORPUS)
    def test_reasons_is_list(self, text):
        assert isinstance(_v(text)["reasons"], list)

    @pytest.mark.parametrize("text", _CORPUS)
    def test_passed_is_bool(self, text):
        assert isinstance(_v(text)["passed"], bool)

    @pytest.mark.parametrize("text", _CORPUS)
    def test_dict_equals_dataclass(self, text):
        assert _v(text) == _vs(text).to_dict()

    @pytest.mark.parametrize("text", _CORPUS)
    def test_junk_passed_disjoint(self, text):
        r = _v(text)
        if r["junk"]:
            assert not r["passed"]

    @pytest.mark.parametrize("text", _CORPUS)
    def test_passes_gate_consistent(self, text):
        assert passes_gate(text) == _v(text)["passed"]


# ===================================================================
# 16. CLI
# ===================================================================

class TestCLI:
    def test_cli_pass(self):
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Build seed_gate.py validator for checking"],
            capture_output=True, text=True
        )
        assert r.returncode == 0
        assert json.loads(r.stdout)["passed"] is True

    def test_cli_fail(self):
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "Something vague and unspecific without direction"],
            capture_output=True, text=True
        )
        assert r.returncode == 1
        assert json.loads(r.stdout)["passed"] is False

    def test_cli_no_args(self):
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py")],
            capture_output=True, text=True
        )
        assert r.returncode == 1


# ===================================================================
# 17. Smoke
# ===================================================================

class TestSmoke:
    def test_50_proposals(self):
        for i in range(50):
            r = _v(f"Build module_{i}.py for Mars colony feature {i}")
            assert "passed" in r

    def test_all_verbs_detectable(self):
        for verb in sorted(ACTION_VERBS):
            r = _v(f"{verb.capitalize()} the water_mining.py module")
            assert r["verb_found"] == verb, f"Verb {verb!r} not detected"


# ===================================================================
# 18. propose_seed.py contract
# ===================================================================

class TestProposeSeedContract:
    def test_import_validate(self):
        from seed_gate import validate as vs
        assert isinstance(vs("Build seed_gate.py validator"), dict)

    def test_passed_key(self):
        assert isinstance(_v("Build seed_gate.py validator")["passed"], bool)

    def test_reasons_joinable(self):
        r = _v("Something vague without real direction here")
        assert isinstance("; ".join(r["reasons"]), str)

    def test_purge_contract(self):
        r = validate("Build seed_gate.py validator", mode="purge")
        assert r["passed"] is True

    def test_score_float(self):
        r = _v("Build seed_gate.py validator for testing")
        assert isinstance(r["score"], float)
        assert 0.0 <= r["score"] <= 1.0

    def test_no_code_key(self):
        assert "code" not in _v("Build seed_gate.py validator")

    def test_verb_found_key(self):
        r = _v("Build seed_gate.py validator for system")
        assert "verb_found" in r
        assert "verb" not in r

    def test_target_found_key(self):
        r = _v("Build seed_gate.py validator for system")
        assert "target_found" in r
        assert "target" not in r

    def test_junk_key(self):
        assert isinstance(_v("Build seed_gate.py validator")["junk"], bool)


# ---- New tests for consolidated features (PRs #245/#246/#247) ----


class TestFileReFalsePositives:
    """FILE_RE should not match abbreviations like e.g., i.e., etc."""

    def test_eg_not_a_file(self):
        from seed_gate import find_target
        t, k = find_target("Consider e.g. the deployment options for Mars")
        assert t != "e.g", f"e.g matched as {k}"

    def test_ie_not_a_file(self):
        from seed_gate import find_target
        t, k = find_target("The module i.e. the core system should be refactored")
        assert t != "i.e", f"i.e matched as {k}"

    def test_am_not_a_file(self):
        from seed_gate import find_target
        t, k = find_target("Deploy the system before 9 a.m. tomorrow morning")
        assert t != "a.m", f"a.m matched as {k}"

    def test_real_file_still_matches(self):
        from seed_gate import find_target
        t, k = find_target("Update the config.yaml for better settings")
        assert t == "config.yaml" and k == "file"

    def test_eg_in_sentence_with_real_file(self):
        r = _v("Fix e.g. the water_mining.py thermal model bug")
        assert r["target_found"] == "water_mining.py"

    def test_validate_eg_only_no_file(self):
        r = _v("Explore e.g. what would happen if agents debated")
        assert r["target_found"] != "e.g"


class TestSpecialFiles:
    """Special files without extensions: Dockerfile, Makefile, README, etc."""

    def test_dockerfile(self):
        from seed_gate import find_target
        t, k = find_target("Build the Dockerfile for the Mars colony")
        assert t == "Dockerfile" and k == "file"

    def test_makefile(self):
        from seed_gate import find_target
        t, k = find_target("Refactor the Makefile for cleaner builds")
        assert t == "Makefile" and k == "file"

    def test_readme(self):
        from seed_gate import find_target
        t, k = find_target("Update README with new architecture docs")
        assert t == "README" and k == "file"

    def test_constitution(self):
        from seed_gate import find_target
        t, k = find_target("Review CONSTITUTION for amendment proposals")
        assert t == "CONSTITUTION" and k == "file"

    def test_agents_md(self):
        from seed_gate import find_target
        t, k = find_target("Improve AGENTS documentation coverage")
        assert t == "AGENTS" and k == "file"

    def test_validate_with_special_file(self):
        r = _v("Build the Dockerfile for Mars colony deployment")
        assert r["passed"] is True
        assert r["target_found"] == "Dockerfile"

    def test_regular_file_takes_priority(self):
        """A .py file should match FILE_RE before SPECIAL_FILE_RE."""
        from seed_gate import find_target
        t, k = find_target("Fix thermal_control.py and update README")
        assert t == "thermal_control.py"


class TestKnownTools:
    """KNOWN_TOOLS: rappterbook-specific tools matched by precision regex."""

    def test_state_io(self):
        from seed_gate import find_target
        t, k = find_target("Refactor state_io for better atomicity")
        assert t == "state_io" and k == "tool"

    def test_process_inbox(self):
        from seed_gate import find_target
        t, k = find_target("Debug process_inbox delta processing logic")
        assert t == "process_inbox" and k == "tool"

    def test_propose_seed(self):
        from seed_gate import find_target
        t, k = find_target("Improve propose_seed with better filtering")
        assert t == "propose_seed" and k == "tool"

    def test_seed_gate_as_tool(self):
        from seed_gate import find_target
        t, k = find_target("Validate seed_gate detects all patterns")
        assert t == "seed_gate" and k == "tool"

    def test_known_tool_before_generic(self):
        """Known tools should match before generic TOOL_RE."""
        from seed_gate import find_target, KNOWN_TOOLS
        for tool in sorted(KNOWN_TOOLS)[:5]:
            t, k = find_target(f"Refactor {tool} for clarity")
            assert t == tool, f"Expected {tool}, got {t}"

    def test_validate_with_known_tool(self):
        r = _v("Build state_io with better error handling")
        assert r["passed"] is True
        assert r["target_found"] == "state_io"


class TestQuestionStems:
    """Question stems infer verbs for exempt-tag proposals."""

    def test_what_if_maps_to_explore(self):
        r = _v("What if agents could dream in parallel", tags=["philosophy"])
        assert r["verb_found"] == "explore"
        assert r["passed"] is True

    def test_how_might_maps_to_design(self):
        r = _v("How might we scale the colony to a million agents", tags=["exploration"])
        assert r["verb_found"] == "design"
        assert r["passed"] is True

    def test_should_we_maps_to_evaluate(self):
        r = _v("Should we abandon the water mining experiment", tags=["debate"])
        assert r["verb_found"] == "evaluate"
        assert r["passed"] is True

    def test_why_not_maps_to_propose(self):
        r = _v("Why not allow agents to govern themselves completely", tags=["philosophy"])
        assert r["verb_found"] == "propose"

    def test_no_stem_without_exempt_tag(self):
        """Question stems should NOT fire without an exempt tag."""
        r = _v("What if agents could dream in parallel")
        assert r["verb_found"] is None
        assert r["passed"] is False

    def test_stem_with_real_verb_uses_real_verb(self):
        """If text has a real verb, don't override with stem mapping."""
        r = _v("What if we build a new thermal control system", tags=["philosophy"])
        assert r["verb_found"] == "build"

    def test_how_could_maps_to_design(self):
        r = _v("How could the sim achieve true emergence behavior", tags=["exploration"])
        assert r["verb_found"] == "design"

    def test_case_insensitive(self):
        r = _v("WHAT IF agents evolved beyond their archetypes", tags=["philosophy"])
        assert r["verb_found"] == "explore"


class TestBatchValidation:
    """validate_batch() returns structured results with stats."""

    def test_basic_batch(self):
        from seed_gate import validate_batch
        proposals = [
            "Build seed_gate.py with verb detection",
            "just vibes and stuff happening here",
            "   ",
            "Consider improving the thermal model design",
        ]
        br = validate_batch(proposals)
        assert br.stats.total == 4
        assert br.stats.passed + br.stats.failed + br.stats.junk == 4

    def test_all_passed(self):
        from seed_gate import validate_batch
        br = validate_batch([
            "Build seed_gate.py with better detection",
            "Fix thermal_control.py for edge cases",
        ])
        assert br.stats.passed == 2
        assert br.stats.junk == 0
        assert br.stats.failed == 0

    def test_all_junk(self):
        from seed_gate import validate_batch
        br = validate_batch(["   ", "ab", "`fragment` from parser"])
        assert br.stats.junk == 3
        assert br.stats.passed == 0

    def test_empty_batch(self):
        from seed_gate import validate_batch
        br = validate_batch([])
        assert br.stats.total == 0
        assert br.stats.pass_rate == 0.0

    def test_junk_items_contain_text(self):
        from seed_gate import validate_batch
        br = validate_batch(["   ", "Build seed_gate.py now"])
        assert len(br.junk_items) >= 1
        assert br.junk_items[0][0] == "   "

    def test_passed_items_contain_result(self):
        from seed_gate import validate_batch
        br = validate_batch(["Build seed_gate.py with better detection"])
        assert len(br.passed_items) == 1
        text, result = br.passed_items[0]
        assert result["passed"] is True

    def test_failed_items_not_junk(self):
        """Failed items are vague-but-not-junk."""
        from seed_gate import validate_batch
        br = validate_batch(["Consider improving the thermal model design work"])
        if br.stats.failed > 0:
            text, result = br.failed_items[0]
            assert result["junk"] is False
            assert result["passed"] is False


class TestBatchStats:
    """BatchStats dataclass: pass_rate, junk_rate, merge."""

    def test_pass_rate(self):
        from seed_gate import BatchStats
        s = BatchStats(total=10, passed=7, failed=2, junk=1)
        assert abs(s.pass_rate - 0.7) < 0.001

    def test_junk_rate(self):
        from seed_gate import BatchStats
        s = BatchStats(total=10, passed=7, failed=2, junk=1)
        assert abs(s.junk_rate - 0.1) < 0.001

    def test_empty_rates(self):
        from seed_gate import BatchStats
        s = BatchStats(total=0, passed=0, failed=0, junk=0)
        assert s.pass_rate == 0.0
        assert s.junk_rate == 0.0

    def test_merge(self):
        from seed_gate import BatchStats
        s1 = BatchStats(total=3, passed=1, failed=1, junk=1)
        s2 = BatchStats(total=5, passed=4, failed=1, junk=0)
        merged = s1.merge(s2)
        assert merged.total == 8
        assert merged.passed == 5
        assert merged.failed == 2
        assert merged.junk == 1

    def test_merge_identity(self):
        from seed_gate import BatchStats
        empty = BatchStats(total=0, passed=0, failed=0, junk=0)
        s = BatchStats(total=3, passed=2, failed=1, junk=0)
        assert s.merge(empty) == s


class TestLowercaseImperative:
    """Smart lowercase handling: verb-starting text passes, fragments fail."""

    def test_build_verb_not_junk(self):
        r = _v("build seed_gate.py with better verb detection")
        assert r["junk"] is False
        assert r["passed"] is True

    def test_fix_verb_not_junk(self):
        r = _v("fix the thermal_control.py temperature bounds bug")
        assert r["junk"] is False

    def test_create_verb_not_junk(self):
        r = _v("create a new water_mining.py test suite here")
        assert r["junk"] is False

    def test_lowercase_fragment_is_junk(self):
        r = _v("the thing we talked about doing sometime soon")
        assert r["junk"] is True

    def test_lowercase_no_verb_is_junk(self):
        r = _v("some random fragment about the system that exists")
        assert r["junk"] is True

    def test_file_start_not_junk(self):
        """Text starting with a filename is not lowercase-junk."""
        from seed_gate import is_junk
        reason = is_junk("seed_gate.py needs better verb detection logic")
        assert reason == "", f"Unexpected junk reason: {reason}"


class TestSubstringDedup:
    """Substring-aware dedup in count_unique_targets()."""

    def test_stem_and_extension(self):
        """seed_gate and seed_gate.py should count as 1 unique target."""
        from seed_gate import count_unique_targets
        count = count_unique_targets("Refactor seed_gate and update seed_gate.py tests")
        assert count == 1, f"Expected 1, got {count}"

    def test_truly_different_targets(self):
        from seed_gate import count_unique_targets
        count = count_unique_targets("Fix water_mining.py and thermal_control.py bugs")
        assert count == 2

    def test_path_and_bare_name(self):
        """src/seed_gate.py and seed_gate should count as 1."""
        from seed_gate import count_unique_targets
        count = count_unique_targets("Update src/seed_gate.py and refactor seed_gate module")
        assert count == 1

    def test_three_unique(self):
        from seed_gate import count_unique_targets
        count = count_unique_targets("Fix water_mining.py, thermal_control.py, and seed_gate.py")
        assert count == 3

    def test_empty_text(self):
        from seed_gate import count_unique_targets
        assert count_unique_targets("No targets here at all in this text") == 0


class TestCanonicalizeTarget:
    """canonicalize_target() strips paths, extensions, quotes."""

    def test_strip_extension(self):
        from seed_gate import canonicalize_target
        assert canonicalize_target("seed_gate.py") == "seed_gate"

    def test_strip_src_prefix(self):
        from seed_gate import canonicalize_target
        assert canonicalize_target("src/seed_gate.py") == "seed_gate"

    def test_strip_quotes(self):
        from seed_gate import canonicalize_target
        assert canonicalize_target('"seed_gate"') == "seed_gate"

    def test_lowercase(self):
        from seed_gate import canonicalize_target
        assert canonicalize_target("README") == "readme"

    def test_preserve_underscores(self):
        from seed_gate import canonicalize_target
        assert canonicalize_target("water_mining.py") == "water_mining"


# ---------------------------------------------------------------------------
# New feature tests — phrasal verbs, tag-implied, advisory, rich match,
# CONST_RE, case-insensitive modules, find_all_verbs, find_all_targets
# ---------------------------------------------------------------------------

class TestPhrasalVerbs:
    """Test two-word phrasal verb detection (#12521)."""

    def test_phrasal_dict_not_empty(self):
        from seed_gate import PHRASAL_VERBS
        assert len(PHRASAL_VERBS) >= 20

    def test_set_up_detected(self):
        from seed_gate import find_verb
        assert find_verb("Set up the deployment pipeline") == "set up"

    def test_roll_back_detected(self):
        from seed_gate import find_verb
        assert find_verb("Roll back the migration for schema.sql") == "roll back"

    def test_wire_up_detected(self):
        from seed_gate import find_verb
        assert find_verb("Wire up the OAuth handler in auth.py") == "wire up"

    def test_clean_up_detected(self):
        from seed_gate import find_verb
        assert find_verb("Clean up dead code in state_io.py") == "clean up"

    def test_tear_down_detected(self):
        from seed_gate import find_verb
        assert find_verb("Tear down the test fixtures properly") == "tear down"

    def test_spin_up_detected(self):
        from seed_gate import find_verb
        assert find_verb("Spin up a new worker for compute_trending.py") == "spin up"

    def test_phrasal_starts_with(self):
        from seed_gate import _starts_with_verb
        assert _starts_with_verb("Set up deployment for auth.py")

    def test_phrasal_not_starts_with(self):
        from seed_gate import _starts_with_verb
        # "set" alone is NOT an action verb
        assert not _starts_with_verb("set the variable")

    def test_single_word_still_works(self):
        from seed_gate import find_verb
        assert find_verb("Build the new module foo.py") == "build"

    def test_phrasal_prefers_first_match(self):
        from seed_gate import find_verb
        assert find_verb("Roll back and then set up the migration") == "roll back"

    def test_phrasal_in_validate(self):
        result = validate_seed("Set up the deployment pipeline for config.yaml")
        assert result.passed
        assert result.verb_found == "set up"

    def test_all_phrasal_verbs_are_canonical(self):
        from seed_gate import PHRASAL_VERBS
        for phrase, canonical in PHRASAL_VERBS.items():
            words = phrase.split()
            assert len(words) == 2, f"Phrasal verb '{phrase}' should be two words"
            assert canonical == phrase, f"Canonical for '{phrase}' should be self"


class TestTagImpliedVerbs:
    """Test tag-to-verb inference (#12530)."""

    def test_tag_implied_dict_not_empty(self):
        from seed_gate import TAG_IMPLIED_VERBS
        assert len(TAG_IMPLIED_VERBS) >= 8

    def test_code_tag_implies_build(self):
        result = validate_seed("The authentication module needs CSRF protection", tags=["code"])
        assert result.verb_found == "build"

    def test_debug_tag_implies_debug(self):
        result = validate_seed("Something wrong with the rate limiter", tags=["debug"])
        assert result.verb_found == "debug"

    def test_test_tag_implies_test(self):
        result = validate_seed("Coverage gaps in the channel module", tags=["test"])
        assert result.verb_found == "test"

    def test_docs_tag_implies_document(self):
        result = validate_seed("The SDK documentation has several unclear sections worth addressing", tags=["docs"])
        assert result.verb_found == "document"

    def test_tag_only_fires_when_no_explicit_verb(self):
        result = validate_seed("Refactor the auth module in auth.py", tags=["code"])
        assert result.verb_found == "refactor"  # explicit verb wins

    def test_exempt_tag_still_takes_priority(self):
        # "philosophy" is exempt — verb from tag_implied shouldn't prevent exemption
        result = validate_seed("The ethics of agent autonomy and decision-making", tags=["philosophy"])
        assert result.passed is False  # no verb at all, no target
        # But exempt tags DO pass if there's an implied verb
        result2 = validate_seed("The ethics of agent autonomy", tags=["philosophy", "code"])
        assert result2.verb_found == "build"

    def test_security_tag_implies_secure(self):
        result = validate_seed("Input validation for the API endpoint handler", tags=["security"])
        assert result.verb_found == "secure"


class TestAdvisoryLabel:
    """Test advisory labeling for verb-but-no-target (#12507)."""

    def test_verb_no_target_exempt_gets_advisory(self):
        result = validate_seed("Explore the nature of consciousness", tags=["philosophy"])
        assert result.passed is True
        assert result.advisory == "needs-specificity"

    def test_verb_and_target_no_advisory(self):
        result = validate_seed("Build the new auth module in auth.py")
        assert result.passed is True
        assert result.advisory == ""

    def test_no_verb_no_advisory(self):
        result = validate_seed("Something about nothing at all really right now")
        assert result.passed is False
        assert result.advisory == ""  # no verb → no advisory

    def test_verb_no_target_not_exempt_gets_advisory(self):
        result = validate_seed("Build something amazing for the whole platform")
        assert result.passed is False
        assert result.advisory == "needs-specificity"

    def test_advisory_in_dict_api(self):
        result = validate("Explore the nature of consciousness", tags=["philosophy"])
        assert "advisory" in result
        assert result["advisory"] == "needs-specificity"

    def test_advisory_in_dict_api_empty(self):
        result = validate("Build the auth module in auth.py")
        assert result["advisory"] == ""


class TestRichMatchInfo:
    """Test all_verbs and all_targets in result (#12521)."""

    def test_all_verbs_returned(self):
        from seed_gate import find_all_verbs
        verbs = find_all_verbs("Build auth.py, then test and deploy the module")
        assert "build" in verbs
        assert "test" in verbs
        assert "deploy" in verbs

    def test_all_verbs_deduped(self):
        from seed_gate import find_all_verbs
        verbs = find_all_verbs("Build and build and build auth.py")
        assert verbs.count("build") == 1

    def test_all_verbs_order_preserved(self):
        from seed_gate import find_all_verbs
        verbs = find_all_verbs("Test first, then deploy, finally monitor foo.py")
        assert verbs == ["test", "deploy", "monitor"]

    def test_all_verbs_phrasal(self):
        from seed_gate import find_all_verbs
        verbs = find_all_verbs("Set up and then tear down the test fixtures for config.yaml")
        assert "set up" in verbs
        assert "tear down" in verbs

    def test_all_targets_returned(self):
        from seed_gate import _find_all_targets
        targets = _find_all_targets("Build auth.py and deploy to config.yaml")
        target_strings = [t[0] for t in targets]
        assert "auth.py" in target_strings
        assert "config.yaml" in target_strings

    def test_all_targets_deduped(self):
        from seed_gate import _find_all_targets
        targets = _find_all_targets("Fix auth.py, then fix auth.py again")
        target_strings = [t[0] for t in targets]
        # Should only appear once (canonical dedup)
        assert len([t for t in target_strings if "auth" in t]) == 1

    def test_all_targets_in_result(self):
        result = validate_seed("Build auth.py and test config.yaml")
        assert len(result.all_targets) >= 2
        assert len(result.all_verbs) >= 1

    def test_all_verbs_in_dict(self):
        result = validate("Build and test auth.py")
        assert "all_verbs" in result
        assert isinstance(result["all_verbs"], list)

    def test_all_targets_in_dict(self):
        result = validate("Build auth.py and config.yaml")
        assert "all_targets" in result
        assert isinstance(result["all_targets"], list)


class TestConstTargets:
    """Test CONST_RE in find_target (#12521)."""

    def test_const_detected(self):
        from seed_gate import find_target
        target, kind = find_target("Update MAX_RETRIES in the retry module")
        assert target == "MAX_RETRIES"
        assert kind == "const"

    def test_const_in_scoring(self):
        from seed_gate import compute_score
        score = compute_score("Update MAX_RETRIES", "update", "MAX_RETRIES", "const")
        assert score > 0.0

    def test_const_in_count_unique(self):
        from seed_gate import count_unique_targets, CONST_RE
        text = "Set ACTION_VERBS and MAX_RETRIES to new values"
        count = count_unique_targets(text)
        assert count >= 2

    def test_const_in_all_targets(self):
        from seed_gate import _find_all_targets
        targets = _find_all_targets("Update MAX_RETRIES and fix config.yaml")
        kinds = [t[1] for t in targets]
        assert "const" in kinds

    def test_const_validate_pass(self):
        result = validate_seed("Update ACTION_VERBS in the seed gate")
        assert result.passed
        assert result.target_found == "ACTION_VERBS"


class TestCaseInsensitiveModules:
    """Test case-insensitive module matching."""

    def test_lowercase_module_import(self):
        from seed_gate import _KNOWN_MODULES_LOWER
        # All entries should be lowercase
        for m in _KNOWN_MODULES_LOWER:
            assert m == m.lower()


class TestFindAllVerbs:
    """Focused tests for find_all_verbs."""

    def test_empty_input(self):
        from seed_gate import find_all_verbs
        assert find_all_verbs("") == []

    def test_no_verbs(self):
        from seed_gate import find_all_verbs
        assert find_all_verbs("the quick brown fox") == []

    def test_single_verb(self):
        from seed_gate import find_all_verbs
        assert find_all_verbs("Build a thing") == ["build"]

    def test_phrasal_skip(self):
        from seed_gate import find_all_verbs
        # "set up" should be one verb, not "set" + "up"
        verbs = find_all_verbs("Set up the deployment for auth.py")
        assert "set up" in verbs
        assert "set" not in verbs


class TestFindAllTargets:
    """Focused tests for _find_all_targets."""

    def test_empty_input(self):
        from seed_gate import _find_all_targets
        assert _find_all_targets("") == ()

    def test_single_file(self):
        from seed_gate import _find_all_targets
        targets = _find_all_targets("Fix auth.py")
        assert len(targets) >= 1
        assert targets[0][0] == "auth.py"
        assert targets[0][1] == "file"

    def test_multiple_kinds(self):
        from seed_gate import _find_all_targets
        targets = _find_all_targets("Build auth.py with src/utils/ and check #12345")
        kinds = {t[1] for t in targets}
        assert "file" in kinds

    def test_const_in_all(self):
        from seed_gate import _find_all_targets
        targets = _find_all_targets("Set MAX_RETRIES and check config.yaml")
        kinds = {t[1] for t in targets}
        assert "const" in kinds
        assert "file" in kinds


class TestNewVerbsExpanded:
    """Test the 25 newly-added verbs."""

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
    def test_new_verb_in_set(self, verb):
        assert verb in ACTION_VERBS

    def test_verb_count_at_least_95(self):
        assert len(ACTION_VERBS) >= 95


class TestNewFeatureInvariants:
    """Property-based invariants for new features."""

    def test_advisory_only_when_verb_present(self):
        """Advisory should only appear when a verb was found."""
        proposals = [
            "Build something nice for the platform",
            "Explore consciousness deeply",
            "The meaning of life is 42 probably",
            "Set up auth.py deployment pipeline",
        ]
        for text in proposals:
            result = validate_seed(text)
            if result.advisory:
                assert result.verb_found is not None

    def test_all_verbs_subset_of_action_verbs_or_phrasal(self):
        """Every verb in all_verbs should be a known verb."""
        from seed_gate import find_all_verbs, PHRASAL_VERBS
        text = "Build and deploy auth.py, then set up monitoring and test"
        for v in find_all_verbs(text):
            assert v in ACTION_VERBS or v in PHRASAL_VERBS.values()

    def test_all_targets_kinds_valid(self):
        """Every target kind should be a recognized kind."""
        from seed_gate import _find_all_targets
        valid_kinds = {"file", "path", "func", "tool", "cli", "discussion",
                       "channel", "quoted", "module", "const"}
        targets = _find_all_targets("Build auth.py, check r/mars, update MAX_RETRIES, see #12345")
        for _, kind in targets:
            assert kind in valid_kinds

    def test_score_const_between_channel_and_tool(self):
        """const score (2.5) should be between channel (2.0) and tool (3.0)."""
        from seed_gate import compute_score
        s_const = compute_score("Update MAX_RETRIES", "update", "MAX_RETRIES", "const")
        s_channel = compute_score("Update r/mars", "update", "r/mars", "channel")
        s_tool = compute_score("Update state_io", "update", "state_io", "tool")
        assert s_channel <= s_const <= s_tool


class TestBackwardCompatNewFields:
    """Ensure new fields don't break existing behavior."""

    def test_old_dict_keys_still_present(self):
        result = validate("Build auth.py")
        for key in ("passed", "reasons", "score", "verb_found", "target_found", "junk"):
            assert key in result

    def test_new_dict_keys_present(self):
        result = validate("Build auth.py")
        for key in ("advisory", "all_verbs", "all_targets"):
            assert key in result

    def test_batch_still_works(self):
        from seed_gate import validate_batch
        br = validate_batch(["Build auth.py", "random stuff here"])
        assert br.stats.total == 2

    def test_passes_gate_unaffected(self):
        assert passes_gate("Build the authentication module in auth.py")
        assert not passes_gate("random stuff here with nothing concrete at all")


# ===================================================================
# NEW: find_verb_with_position
# ===================================================================

class TestFindVerbWithPosition:
    def test_imperative_position_zero(self):
        from seed_gate import find_verb_with_position
        result = find_verb_with_position("Build seed_gate.py")
        assert result == ("build", 0)

    def test_verb_after_preamble(self):
        from seed_gate import find_verb_with_position
        result = find_verb_with_position("We should probably build seed_gate.py")
        assert result is not None
        assert result[0] == "build"
        assert result[1] == 3

    def test_phrasal_verb_position(self):
        from seed_gate import find_verb_with_position
        result = find_verb_with_position("Set up the test framework")
        assert result is not None
        assert result[0] == "set up"
        assert result[1] == 0

    def test_no_verb(self):
        from seed_gate import find_verb_with_position
        assert find_verb_with_position("The quick brown fox") is None

    def test_second_word_verb(self):
        from seed_gate import find_verb_with_position
        result = find_verb_with_position("Please build reactor.py")
        assert result == ("build", 1)

    def test_empty_text(self):
        from seed_gate import find_verb_with_position
        assert find_verb_with_position("") is None


# ===================================================================
# NEW: Imperative scoring bonus
# ===================================================================

class TestImperativeBonus:
    def test_imperative_higher_than_preamble(self):
        """'Build X' should score higher than 'We should build X'."""
        imperative = _v("Build seed_gate.py")
        preamble = _v("We should perhaps build seed_gate.py")
        assert imperative["score"] > preamble["score"]

    def test_imperative_flag_on_result(self):
        result = _vs("Build seed_gate.py")
        assert result.is_imperative is True

    def test_preamble_not_imperative(self):
        result = _vs("We should probably build seed_gate.py")
        assert result.is_imperative is False

    def test_tag_implied_verb_no_imperative_bonus(self):
        """Tag-implied verbs should NOT get the imperative bonus."""
        from seed_gate import score_breakdown
        # Tag-implied has verb_position = -1 (no explicit verb found)
        result = _vs("A deep analysis of code quality", tags=["code"])
        assert result.verb_found == "build"
        assert result.is_imperative is False

    def test_question_stem_no_imperative(self):
        result = _vs("What if we explored consciousness?", tags=["philosophy"])
        assert result.is_imperative is False

    def test_imperative_affects_confidence(self):
        """Imperative bonus can push score from medium to high confidence."""
        # Both have verb+file target; imperative one gets bonus
        r1 = _vs("Build seed_gate.py")
        r2 = _vs("One day someone should build seed_gate.py eventually")
        # r1 should be >= r2 in score
        assert r1.score >= r2.score


# ===================================================================
# NEW: score_breakdown
# ===================================================================

class TestScoreBreakdown:
    def test_returns_dict_keys(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file")
        for key in ("verb", "target", "length", "multi_target", "imperative", "raw_total", "score"):
            assert key in bd

    def test_verb_points(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build X", "build", None, "")
        assert bd["verb"] == 2.5
        assert bd["target"] == 0.0

    def test_file_target_points(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert bd["target"] == 4.0

    def test_channel_target_points(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build r/mars", "build", "r/mars", "channel")
        assert bd["target"] == 2.0

    def test_imperative_bonus_in_breakdown(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file",
                             verb_position=0)
        assert bd["imperative"] == 0.5

    def test_no_imperative_without_position(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file",
                             verb_position=-1)
        assert bd["imperative"] == 0.0

    def test_no_imperative_at_position_3(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("text", "build", "seed_gate.py", "file",
                             verb_position=3)
        assert bd["imperative"] == 0.0

    def test_length_bonus_short(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build X", "build", "X", "file")
        assert bd["length"] == 0.0

    def test_length_bonus_medium(self):
        from seed_gate import score_breakdown
        text = " ".join(["word"] * 10)
        bd = score_breakdown(text, "build", "X", "file")
        assert bd["length"] == 0.5

    def test_length_bonus_long(self):
        from seed_gate import score_breakdown
        text = " ".join(["word"] * 20)
        bd = score_breakdown(text, "build", "X", "file")
        assert bd["length"] == 1.5

    def test_multi_target_bonus(self):
        from seed_gate import score_breakdown
        text = "Build seed_gate.py and propose_seed.py together"
        bd = score_breakdown(text, "build", "seed_gate.py", "file")
        assert bd["multi_target"] == 1.0

    def test_raw_total_sums_components(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file",
                             verb_position=0)
        expected = bd["verb"] + bd["target"] + bd["length"] + bd["multi_target"] + bd["imperative"]
        assert abs(bd["raw_total"] - expected) < 0.001

    def test_score_is_normalized(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert 0.0 <= bd["score"] <= 1.0

    def test_score_matches_compute_score(self):
        """score_breakdown and compute_score must agree."""
        from seed_gate import score_breakdown, compute_score
        text = "Build seed_gate.py and propose_seed.py"
        bd = score_breakdown(text, "build", "seed_gate.py", "file", verb_position=0)
        cs = compute_score(text, "build", "seed_gate.py", "file", verb_position=0)
        assert abs(bd["score"] - cs) < 0.001


# ===================================================================
# NEW: confidence property
# ===================================================================

class TestConfidence:
    def test_high_confidence(self):
        result = _vs("Build seed_gate.py and propose_seed.py together now")
        assert result.confidence == "high"

    def test_medium_confidence(self):
        result = _vs("Build r/mars-engineering")
        assert result.confidence == "medium"

    def test_low_confidence_on_failure(self):
        result = _vs("Make everything better")
        assert result.confidence == "low"

    def test_confidence_in_dict(self):
        d = _v("Build seed_gate.py")
        assert "confidence" in d
        assert d["confidence"] in ("high", "medium", "low")

    def test_confidence_thresholds(self):
        """Verify threshold boundaries: 0.7 → high, 0.4 → medium."""
        # A result with score exactly 0.0 → low
        result = _vs("Make everything better")
        assert result.score == 0.0
        assert result.confidence == "low"

    def test_junk_is_low(self):
        result = _vs("")
        assert result.confidence == "low"


# ===================================================================
# NEW: Numbered reference false-file filter
# ===================================================================

class TestNumberedRefFilter:
    def test_no_dot_5_as_file(self):
        """'no.5' should not be detected as a file target."""
        from seed_gate import find_target
        target, kind = find_target("Build support for item no.5 in the spec")
        assert target != "no.5"

    def test_fig_1_filtered(self):
        from seed_gate import find_target
        target, kind = find_target("Implement fig.1 from the paper")
        assert target != "fig.1"

    def test_vol_2_filtered(self):
        from seed_gate import find_target
        target, kind = find_target("Review vol.2 of the documentation")
        assert target != "vol.2"

    def test_ch_3_filtered(self):
        from seed_gate import find_target
        target, kind = find_target("Build ch.3 of the tutorial")
        assert target != "ch.3"

    def test_eq_1_filtered(self):
        from seed_gate import find_target
        target, kind = find_target("Implement eq.1 from the thermal model")
        assert target != "eq.1"

    def test_sec_4_filtered(self):
        from seed_gate import find_target
        target, kind = find_target("Review sec.4 of the design doc")
        assert target != "sec.4"

    def test_real_file_still_matches(self):
        """Actual files should NOT be filtered."""
        from seed_gate import find_target
        target, kind = find_target("Build reactor.py")
        assert target == "reactor.py"
        assert kind == "file"

    def test_path_qualified_version_allowed(self):
        """Path-qualified matches like docs/v1.0 are still valid paths."""
        from seed_gate import find_target
        target, kind = find_target("Build docs/v1.0 directory structure")
        assert target != ""

    def test_ver_filtered(self):
        from seed_gate import find_target
        target, kind = find_target("Build support for ver.2 compliance")
        assert target != "ver.2"

    def test_case_insensitive(self):
        from seed_gate import find_target
        target, kind = find_target("Build Fig.1 from design")
        assert target != "Fig.1"


# ===================================================================
# NEW: suggest()
# ===================================================================

class TestSuggest:
    def test_passing_seed_no_suggestions(self):
        from seed_gate import suggest
        assert suggest("Build seed_gate.py") == []

    def test_no_verb_suggests_verb(self):
        from seed_gate import suggest
        hints = suggest("The seed_gate.py module needs work")
        assert any("action verb" in h.lower() for h in hints)

    def test_no_target_suggests_target(self):
        from seed_gate import suggest
        hints = suggest("Build something really cool and great")
        assert any("filename" in h.lower() or "target" in h.lower() for h in hints)

    def test_verb_but_no_target_gets_specific_hint(self):
        from seed_gate import suggest
        hints = suggest("Build a comprehensive new system")
        assert any("file or module" in h.lower() for h in hints)

    def test_junk_gets_rewrite_hint(self):
        from seed_gate import suggest
        hints = suggest("  ")
        assert any("fragment" in h.lower() or "rewrite" in h.lower() for h in hints)

    def test_returns_list(self):
        from seed_gate import suggest
        result = suggest("Make everything better")
        assert isinstance(result, list)


# ===================================================================
# NEW: explain()
# ===================================================================

class TestExplain:
    def test_passing_shows_pass(self):
        from seed_gate import explain
        output = explain("Build seed_gate.py")
        assert "[PASS]" in output

    def test_failing_shows_fail(self):
        from seed_gate import explain
        output = explain("Make everything better")
        assert "[FAIL]" in output

    def test_junk_shows_junk(self):
        from seed_gate import explain
        output = explain("  ")
        assert "[JUNK]" in output

    def test_contains_score(self):
        from seed_gate import explain
        output = explain("Build seed_gate.py")
        assert "score=" in output

    def test_contains_confidence(self):
        from seed_gate import explain
        output = explain("Build seed_gate.py")
        assert "confidence=" in output

    def test_contains_verb(self):
        from seed_gate import explain
        output = explain("Build seed_gate.py")
        assert "build" in output

    def test_contains_target(self):
        from seed_gate import explain
        output = explain("Build seed_gate.py")
        assert "seed_gate.py" in output

    def test_contains_breakdown(self):
        from seed_gate import explain
        output = explain("Build seed_gate.py")
        assert "breakdown:" in output

    def test_imperative_noted(self):
        from seed_gate import explain
        output = explain("Build seed_gate.py")
        assert "imperative" in output.lower()

    def test_suggestions_for_failed(self):
        from seed_gate import explain
        output = explain("Make everything better")
        assert "suggestions:" in output

    def test_returns_string(self):
        from seed_gate import explain
        assert isinstance(explain("Build seed_gate.py"), str)

    def test_multiline(self):
        from seed_gate import explain
        output = explain("Build seed_gate.py")
        assert "\n" in output


# ===================================================================
# NEW: Backward-compat — confidence in dict API
# ===================================================================

class TestBackwardCompatDiagnostics:
    def test_confidence_key_in_validate(self):
        result = _v("Build seed_gate.py")
        assert "confidence" in result

    def test_old_keys_still_present_after_evolution(self):
        result = _v("Build auth.py")
        for key in ("passed", "reasons", "score", "verb_found", "target_found",
                     "junk", "advisory", "all_verbs", "all_targets", "confidence"):
            assert key in result, f"Missing key: {key}"

    def test_propose_seed_contract_unchanged(self):
        """propose_seed.py only reads passed + reasons — verify unchanged."""
        result = _v("Build seed_gate.py")
        assert isinstance(result["passed"], bool)
        assert isinstance(result["reasons"], list)

    def test_score_still_float(self):
        result = _v("Build seed_gate.py")
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0


# ===================================================================
# NEW: Property invariants for diagnostics
# ===================================================================

class TestDiagnosticInvariants:
    """Cross-cutting invariants for the new diagnostic features."""

    PROPOSALS = [
        "Build seed_gate.py",
        "We should build seed_gate.py eventually",
        "Set up the test framework for reactor.py",
        "Make everything better",
        "  ",
        "Explore consciousness in agents deeply",
        "Build and test rover.py and drill.py together",
    ]

    @pytest.mark.parametrize("text", PROPOSALS)
    def test_confidence_always_valid(self, text):
        from seed_gate import explain
        result = _vs(text)
        assert result.confidence in ("high", "medium", "low")

    @pytest.mark.parametrize("text", PROPOSALS)
    def test_is_imperative_is_bool(self, text):
        result = _vs(text)
        assert isinstance(result.is_imperative, bool)

    @pytest.mark.parametrize("text", PROPOSALS)
    def test_explain_never_crashes(self, text):
        from seed_gate import explain
        output = explain(text)
        assert isinstance(output, str)
        assert len(output) > 0

    @pytest.mark.parametrize("text", PROPOSALS)
    def test_suggest_never_crashes(self, text):
        from seed_gate import suggest
        hints = suggest(text)
        assert isinstance(hints, list)
        assert all(isinstance(h, str) for h in hints)

    @pytest.mark.parametrize("text", PROPOSALS)
    def test_score_breakdown_sums_correctly(self, text):
        from seed_gate import score_breakdown, find_verb_with_position, find_target
        vp = find_verb_with_position(text)
        verb = vp[0] if vp else None
        verb_pos = vp[1] if vp else -1
        target, kind = find_target(text)
        bd = score_breakdown(text, verb, target, kind, verb_position=verb_pos)
        expected = bd["verb"] + bd["target"] + bd["length"] + bd["multi_target"] + bd["imperative"]
        assert abs(bd["raw_total"] - expected) < 0.001

    @pytest.mark.parametrize("text", PROPOSALS)
    def test_imperative_only_when_verb_found_explicitly(self, text):
        """Imperative requires explicit verb in first 3 words."""
        from seed_gate import find_verb_with_position
        result = _vs(text)
        vp = find_verb_with_position(text)
        if result.is_imperative:
            assert vp is not None
            assert vp[1] < 3
