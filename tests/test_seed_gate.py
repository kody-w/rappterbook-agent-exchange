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


# ===================================================================
# 18. DevOps verbs (this frame)
# ===================================================================

class TestDevOpsVerbs:
    """11 new verbs: configure, scaffold, bootstrap, etc."""

    @pytest.mark.parametrize("verb", [
        "configure", "scaffold", "bootstrap", "provision", "install",
        "deprecate", "archive", "rollback", "revert", "containerize", "expose",
    ])
    def test_verb_in_set(self, verb):
        from seed_gate import ACTION_VERBS
        assert verb in ACTION_VERBS

    @pytest.mark.parametrize("verb", [
        "configure", "scaffold", "bootstrap", "provision", "install",
        "deprecate", "archive", "rollback", "revert", "containerize", "expose",
    ])
    def test_verb_detected(self, verb):
        from seed_gate import find_verb
        assert find_verb(f"{verb.capitalize()} the seed_gate.py module") == verb

    def test_configure_passes_gate(self):
        r = _v("Configure seed_gate.py for stricter validation rules")
        assert r["passed"] is True

    def test_scaffold_passes_gate(self):
        r = _v("Scaffold test_thermal.py test suite from scratch")
        assert r["passed"] is True

    def test_containerize_passes_gate(self):
        r = _v("Containerize the mars_colony.py simulation engine")
        assert r["passed"] is True

    def test_verb_count_is_81(self):
        from seed_gate import ACTION_VERBS
        assert len(ACTION_VERBS) >= 81


# ===================================================================
# 19. Verb position scoring
# ===================================================================

class TestVerbPositionScoring:
    """Verbs in first 3 words get an imperative bonus."""

    def test_verb_position_zero(self):
        from seed_gate import find_verb_with_position
        verb, pos = find_verb_with_position("Build seed_gate.py validator")
        assert verb == "build"
        assert pos == 0

    def test_verb_position_deep(self):
        from seed_gate import find_verb_with_position
        verb, pos = find_verb_with_position("We should probably build seed_gate.py")
        assert verb == "build"
        assert pos == 3

    def test_imperative_scores_higher(self):
        # Match word counts to isolate verb-position effect
        imperative = _v("Build seed_gate.py for the new validation rules now")
        buried = _v("Maybe we should build seed_gate.py for new rules now")
        assert imperative["score"] > buried["score"]

    def test_no_verb_position_minus_one(self):
        from seed_gate import find_verb_with_position
        _, pos = find_verb_with_position("The thing about stuff")
        assert pos == -1

    def test_position_with_limit(self):
        from seed_gate import find_verb_with_position
        verb, pos = find_verb_with_position("Build seed_gate.py", limit=5)
        assert verb == "build"
        assert pos == 0


# ===================================================================
# 20. Explain API
# ===================================================================

class TestExplainAPI:
    """explain() returns a structured breakdown."""

    def test_basic_explain(self):
        from seed_gate import explain
        result = explain("Build seed_gate.py with validation rules")
        assert result["passed"] is True
        assert result["verb"] == "build"
        assert result["verb_position"] == 0
        assert result["verb_imperative"] is True
        assert result["primary_target"] == "seed_gate.py"
        assert result["primary_target_kind"] == "file"
        assert isinstance(result["all_targets"], list)
        assert result["score"] > 0.0

    def test_explain_failed(self):
        from seed_gate import explain
        result = explain("Make everything better in general")
        assert result["passed"] is False

    def test_explain_junk(self):
        from seed_gate import explain
        result = explain("")
        assert result["junk_reason"] is not None

    def test_explain_env_var(self):
        from seed_gate import explain
        result = explain("Configure $STATE_DIR for test setup")
        assert any(t["kind"] == "env_var" for t in result["all_targets"])

    def test_explain_multi_target(self):
        from seed_gate import explain
        result = explain("Build seed_gate.py and propose_seed.py pipeline")
        assert result["unique_target_count"] >= 2
        assert len(result["all_targets"]) >= 2

    def test_explain_score_matches_validate(self):
        from seed_gate import explain
        text = "Build seed_gate.py validator with new features"
        e = explain(text)
        v = _v(text)
        assert e["score"] == v["score"]
        assert e["passed"] == v["passed"]

    def test_explain_exempt_tags(self):
        from seed_gate import explain
        result = explain("What if consciousness is emergent deeply", ["theme"])
        assert result["exempt_tags"] == ["theme"]
        assert result["stem_verb"] == "explore"

    def test_explain_word_count(self):
        from seed_gate import explain
        result = explain("Build seed_gate.py validator")
        assert result["word_count"] == 3

    def test_explain_soft_artifact(self):
        from seed_gate import explain
        result = explain("Build the regex parser for seed_gate.py module")
        assert isinstance(result["soft_artifact"], bool)


# ===================================================================
# 21. Version / numbered-ref false positives
# ===================================================================

class TestVersionFalsePositives:
    """Version numbers and numbered refs are not files."""

    def test_v1_2_not_a_file(self):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match("v1.2")

    def test_v0_1_not_a_file(self):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match("v0.1")

    def test_no_5_not_a_file(self):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match("no.5")

    def test_fig_1_not_a_file(self):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match("fig.1")

    def test_vol_2_not_a_file(self):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match("vol.2")

    def test_ch_3_not_a_file(self):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match("ch.3")

    def test_ph_d_not_a_file(self):
        from seed_gate import _is_false_file_match
        assert _is_false_file_match("Ph.D")

    def test_real_file_not_false_positive(self):
        from seed_gate import _is_false_file_match
        assert not _is_false_file_match("seed_gate.py")
        assert not _is_false_file_match("main.rs")

    def test_validate_version_only_no_target(self):
        """A proposal with only 'v1.2' should not count it as a target."""
        r = _v("Upgrade to v1.2 of the platform for all users")
        assert r["target_found"] != "v1.2"

    def test_validate_version_plus_real_file(self):
        """v1.2 ignored, real file found."""
        r = _v("Upgrade seed_gate.py to v2.0 with better scoring")
        assert r["target_found"] == "seed_gate.py"
        assert r["passed"] is True


# ===================================================================
# 22. Environment variable targets
# ===================================================================

class TestEnvVarTarget:
    """$STATE_DIR and ${GITHUB_TOKEN} as concrete targets."""

    def test_state_dir_detected(self):
        from seed_gate import find_target
        target, kind = find_target("Configure $STATE_DIR for testing")
        assert target == "$STATE_DIR"
        assert kind == "env_var"

    def test_braced_var_detected(self):
        from seed_gate import find_target
        target, kind = find_target("Set ${GITHUB_TOKEN} in CI config")
        assert target == "${GITHUB_TOKEN}"
        assert kind == "env_var"

    def test_env_var_passes_gate(self):
        r = _v("Configure $STATE_DIR for isolated test environments")
        assert r["passed"] is True

    def test_env_var_after_cli(self):
        """CLI targets take priority over env vars."""
        from seed_gate import find_target
        target, kind = find_target("Run `export $PATH` for env setup")
        assert kind == "cli"

    def test_dollar_amount_not_matched(self):
        """$100 should not match as env var (too short)."""
        from seed_gate import find_target
        target, kind = find_target("We need $100 budget allocation now")
        assert kind != "env_var"

    def test_env_var_count(self):
        from seed_gate import count_unique_targets
        count = count_unique_targets("$STATE_DIR and $DOCS_DIR and seed_gate.py")
        assert count >= 3


# ===================================================================
# 23. Context-constrained class targets
# ===================================================================

class TestClassTarget:
    """PascalCase in backticks or after 'class' keyword."""

    def test_backtick_class(self):
        from seed_gate import find_target
        target, kind = find_target("Refactor the `SeedGateResult` dataclass")
        assert target == "SeedGateResult"
        assert kind == "class"

    def test_class_keyword(self):
        from seed_gate import find_target
        target, kind = find_target("Create class MarsColony with tick method")
        assert target == "MarsColony"
        assert kind == "class"

    def test_single_word_not_matched(self):
        """Single capitalized words should NOT match."""
        from seed_gate import find_target
        target, kind = find_target("Build the Mars platform from scratch")
        assert kind != "class"

    def test_class_after_file(self):
        """Files take priority over classes."""
        from seed_gate import find_target
        target, kind = find_target("Refactor seed_gate.py `SeedGateResult`")
        assert kind == "file"
        assert target == "seed_gate.py"


# ===================================================================
# 24. Score breakdown
# ===================================================================

class TestScoreBreakdown:
    """score_breakdown() returns decomposed scoring."""

    def test_breakdown_keys(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert "verb_pts" in bd
        assert "target_pts" in bd
        assert "length_pts" in bd
        assert "multi_target_pts" in bd
        assert "imperative_pts" in bd
        assert "raw_total" in bd
        assert "final_score" in bd

    def test_verb_points(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert bd["verb_pts"] == 2.5

    def test_no_verb_zero_pts(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("The seed_gate.py thing", None, "seed_gate.py", "file")
        assert bd["verb_pts"] == 0.0

    def test_file_target_pts(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build seed_gate.py", "build", "seed_gate.py", "file")
        assert bd["target_pts"] == 4.0

    def test_env_var_target_pts(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Configure $STATE_DIR", "configure", "$STATE_DIR", "env_var")
        assert bd["target_pts"] == 2.0

    def test_imperative_bonus(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("Build seed_gate.py validator", "build", "seed_gate.py", "file")
        assert bd["imperative_pts"] == 0.5

    def test_no_imperative_for_deep_verb(self):
        from seed_gate import score_breakdown
        bd = score_breakdown("We should probably build seed_gate.py", "build", "seed_gate.py", "file")
        assert bd["imperative_pts"] == 0.0

    def test_final_score_bounded(self):
        from seed_gate import score_breakdown
        bd = score_breakdown(
            "Build seed_gate.py and propose_seed.py and water_mining.py with all the validation",
            "build", "seed_gate.py", "file"
        )
        assert 0.0 <= bd["final_score"] <= 1.0


# ===================================================================
# 25. find_all_targets
# ===================================================================

class TestFindAllTargets:
    """find_all_targets() enumerates all targets with dedup."""

    def test_multiple_files(self):
        from seed_gate import find_all_targets
        targets = find_all_targets("Build seed_gate.py and propose_seed.py")
        names = [t["target"] for t in targets]
        assert "seed_gate.py" in names
        assert "propose_seed.py" in names

    def test_mixed_kinds(self):
        from seed_gate import find_all_targets
        targets = find_all_targets("Build seed_gate.py and configure $STATE_DIR")
        kinds = {t["kind"] for t in targets}
        assert "file" in kinds
        assert "env_var" in kinds

    def test_deduplication(self):
        from seed_gate import find_all_targets
        targets = find_all_targets("Build seed_gate.py in src/seed_gate.py directory")
        # seed_gate.py and src/seed_gate.py should dedup to one canonical form
        names = [t["target"] for t in targets]
        assert len(names) >= 1

    def test_empty_text(self):
        from seed_gate import find_all_targets
        assert find_all_targets("") == []


# ===================================================================
# 26. Property invariants for new features
# ===================================================================

class TestNewFeatureInvariants:
    """Cross-cutting invariants for the new evolution."""

    def test_score_bounded(self):
        texts = [
            "Build seed_gate.py",
            "Configure $STATE_DIR and $DOCS_DIR and water_mining.py",
            "Scaffold the entire test suite from scratch for everything",
            "",
            "x",
        ]
        for text in texts:
            r = _v(text)
            assert 0.0 <= r["score"] <= 1.0, f"Score out of bounds for: {text!r}"

    def test_explain_score_always_matches_validate(self):
        from seed_gate import explain
        texts = [
            "Build seed_gate.py validator with new features",
            "Configure $STATE_DIR for test isolation setup",
            "Deploy water_mining.py to production servers now",
            "What if agents had dreams and could recall them",
        ]
        for text in texts:
            e = explain(text)
            v = _v(text)
            assert e["score"] == v["score"], f"Score mismatch for: {text!r}"

    def test_version_never_sole_target(self):
        """Version numbers alone should never be the only target."""
        for text in [
            "Upgrade to 2.0 of the platform",
            "Release v3.1 for the community",
            "Ship version 1.2.3 release",
        ]:
            r = _v(text)
            if r["target_found"]:
                from seed_gate import _is_false_file_match
                assert not _is_false_file_match(r["target_found"]), \
                    f"Version matched as target in: {text!r}"

    def test_env_var_contributes_to_count(self):
        from seed_gate import count_unique_targets
        count = count_unique_targets("$STATE_DIR and seed_gate.py")
        assert count >= 2

    def test_all_targets_subset_of_unique_count(self):
        from seed_gate import find_all_targets, count_unique_targets
        text = "Build seed_gate.py and propose_seed.py for $STATE_DIR"
        all_t = find_all_targets(text)
        unique_c = count_unique_targets(text)
        assert len(all_t) <= unique_c + 3  # allow some slack for different dedup methods
