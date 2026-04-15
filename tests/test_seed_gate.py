#!/usr/bin/env python3
"""Tests for seed_gate.py -- the canonical specificity validator.

Covers:
- Verb detection (positive, negative, edge cases)
- Target detection (files, tools, paths, functions, channels, refs)
- Pass/fail validation
- Exempt tag handling
- Junk detection
- Score computation
- Legacy dict compatibility
- CLI integration (--check exit codes, --filter stdin)
- Property-based invariants
- Regex pattern correctness
- Real-world proposals from the ballot
- Smoke tests (adversarial input)
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
    ACTION_VERBS, CHANNEL_RE, EXEMPT_TAGS, FILE_RE, FUNC_RE, MIN_LENGTH,
    PATH_RE, REF_RE, SPECIAL_FILE_RE, TOOL_RE, SeedGateResult,
    compute_score, find_action_verb, find_all_targets, find_all_verbs,
    find_concrete_target, is_junk, passes_gate, validate, validate_seed,
)


# =========================================================================
# find_action_verb
# =========================================================================

class TestFindActionVerb:
    def test_finds_build(self):
        assert find_action_verb("Build the reactor module") == "build"

    def test_finds_write(self):
        assert find_action_verb("Write a test suite for the hab") == "write"

    def test_finds_ship(self):
        assert find_action_verb("Ship the validator to production") == "ship"

    def test_finds_test(self):
        assert find_action_verb("Test the water purifier module") == "test"

    def test_finds_fix(self):
        assert find_action_verb("Fix the oxygen leak bug") == "fix"

    def test_finds_create(self):
        assert find_action_verb("Create a new simulation engine") == "create"

    def test_finds_deploy(self):
        assert find_action_verb("Deploy the habitat controller") == "deploy"

    def test_finds_refactor(self):
        assert find_action_verb("Refactor the thermal system") == "refactor"

    def test_finds_simulate(self):
        assert find_action_verb("Simulate the colony for 365 sols") == "simulate"

    def test_finds_decode(self):
        assert find_action_verb("Decode the beacon signal data") == "decode"

    def test_finds_run(self):
        assert find_action_verb("Run the Monte Carlo simulation") == "run"

    def test_returns_first_verb(self):
        assert find_action_verb("Build and test the seed_gate module") == "build"

    def test_case_insensitive(self):
        assert find_action_verb("BUILD THE THING") == "build"
        assert find_action_verb("Ship it") == "ship"

    def test_no_verb_returns_none(self):
        assert find_action_verb("The colony needs more water") is None

    def test_no_verb_in_abstract(self):
        assert find_action_verb("Philosophy of agent existence") is None

    def test_empty_string(self):
        assert find_action_verb("") is None

    def test_verb_embedded_in_longer_word(self):
        assert find_action_verb("Building is hard") is None

    def test_inflected_forms_dont_match(self):
        assert find_action_verb("She builds the reactor") is None
        assert find_action_verb("He writes code daily") is None
        assert find_action_verb("They tested it already") is None

    def test_all_verbs_recognized(self):
        for verb in ACTION_VERBS:
            found = find_action_verb("Please %s the module" % verb)
            assert found == verb, "Failed for verb: %s" % verb


class TestFindAllVerbs:
    def test_finds_multiple(self):
        verbs = find_all_verbs("Build and test the seed_gate.py module")
        assert "build" in verbs
        assert "test" in verbs

    def test_empty(self):
        assert find_all_verbs("No verbs here at all") == []

    def test_deduplicated(self):
        verbs = find_all_verbs("Build it then build it again")
        assert verbs.count("build") == 1


# =========================================================================
# find_concrete_target
# =========================================================================

class TestFindConcreteTarget:
    def test_finds_py_file(self):
        assert find_concrete_target("Modify seed_gate.py") == "seed_gate.py"

    def test_finds_sh_file(self):
        assert find_concrete_target("Run bundle.sh to build") == "bundle.sh"

    def test_finds_json_file(self):
        assert find_concrete_target("Update state/seeds.json") == "seeds.json"

    def test_finds_js_file(self):
        assert find_concrete_target("Fix router.js routing bug") == "router.js"

    def test_finds_ts_file(self):
        assert find_concrete_target("Add types to index.ts") == "index.ts"

    def test_finds_md_file(self):
        assert find_concrete_target("Update README.md") == "README.md"

    def test_finds_rust_file(self):
        assert find_concrete_target("Port to main.rs") == "main.rs"

    def test_finds_go_file(self):
        assert find_concrete_target("Rewrite in server.go") == "server.go"

    def test_finds_tool_name(self):
        assert find_concrete_target("Use pytest to validate") == "pytest"

    def test_finds_platform_tool(self):
        assert find_concrete_target("Wire into process_inbox") == "process_inbox"

    def test_finds_path(self):
        result = find_concrete_target("Look at src/seed_gate.py")
        assert result is not None

    def test_finds_nested_path(self):
        result = find_concrete_target("Check engine/prompts/frame.md")
        assert result is not None

    def test_finds_function_call(self):
        assert find_concrete_target("Call validate_seed()") == "validate_seed()"

    def test_finds_channel(self):
        assert find_concrete_target("Post to r/engineering") == "r/engineering"

    def test_finds_special_file(self):
        assert find_concrete_target("Update the Dockerfile") == "Dockerfile"

    def test_finds_makefile(self):
        assert find_concrete_target("Add target to Makefile") == "Makefile"

    def test_finds_discussion_ref(self):
        assert find_concrete_target("See discussion #12503") == "#12503"

    def test_no_target_in_vague_text(self):
        assert find_concrete_target("Make the world better") is None

    def test_no_target_in_abstract(self):
        assert find_concrete_target("The agents should cooperate more") is None

    def test_empty_string(self):
        assert find_concrete_target("") is None

    def test_does_not_match_dr_smith(self):
        assert find_concrete_target("Dr. Smith leads the mission") is None

    def test_does_not_match_decimal(self):
        assert find_concrete_target("Pi is approximately 3.14") is None

    def test_hyphenated_filename(self):
        assert find_concrete_target("Fix seed-gate.py") == "seed-gate.py"


class TestFindAllTargets:
    def test_finds_multiple(self):
        targets = find_all_targets("Build seed_gate.py and run pytest")
        assert "seed_gate.py" in targets
        assert "pytest" in targets

    def test_empty(self):
        assert find_all_targets("No targets here") == []


# =========================================================================
# is_junk
# =========================================================================

class TestIsJunk:
    def test_empty_is_junk(self):
        assert is_junk("") is not None
        assert "empty" in is_junk("")

    def test_whitespace_is_junk(self):
        assert is_junk("   ") is not None

    def test_too_short(self):
        assert is_junk("Fix it") is not None
        assert "short" in is_junk("Fix it")

    def test_fragment_pipe(self):
        assert is_junk("|some weird fragment text") is not None
        assert "fragment" in is_junk("|some weird fragment text")

    def test_fragment_comma(self):
        assert is_junk(",continuation of something else") is not None

    def test_fragment_paren(self):
        assert is_junk("(this is a parenthetical thing)") is not None

    def test_parsing_artifact(self):
        text = "parser grabbed this from the output stream"
        assert is_junk(text) is not None
        assert "parsing artifact" in is_junk(text)

    def test_normal_text_not_junk(self):
        text = "Build seed_gate.py with comprehensive test coverage"
        assert is_junk(text) is None

    def test_exactly_min_length(self):
        text = "x" * MIN_LENGTH
        assert is_junk(text) is None

    def test_just_under_min_length(self):
        text = "x" * (MIN_LENGTH - 1)
        assert is_junk(text) is not None

    def test_backtick_not_junk(self):
        text = "`seed_gate.py` needs validation tests now"
        assert is_junk(text) is None

    def test_lowercase_start_not_junk(self):
        text = "seed_gate.py needs a rewrite for better validation"
        assert is_junk(text) is None


# =========================================================================
# validate_seed -- positive cases
# =========================================================================

class TestValidateSeedPass:
    def test_verb_plus_filename(self):
        r = validate_seed("Build seed_gate.py with comprehensive tests")
        assert r.passes is True
        assert r.verb == "build"
        assert r.target == "seed_gate.py"
        assert r.reasons == []

    def test_verb_plus_tool(self):
        r = validate_seed("Run pytest against the full simulation suite")
        assert r.passes is True
        assert r.verb == "run"
        assert r.target == "pytest"

    def test_verb_plus_path(self):
        r = validate_seed(
            "Refactor scripts/process_inbox to use the new dispatcher pattern")
        assert r.passes is True
        assert r.verb == "refactor"

    def test_verb_plus_function(self):
        r = validate_seed("Optimize compute_score() to handle edge cases")
        assert r.passes is True
        assert r.verb == "optimize"
        assert r.target == "compute_score()"

    def test_verb_plus_channel(self):
        r = validate_seed(
            "Create r/engineering channel for technical discussions")
        assert r.passes is True
        assert r.verb == "create"
        assert r.target == "r/engineering"

    def test_verb_plus_discussion_ref(self):
        r = validate_seed(
            "Implement the proposal from #12503 for the seed validator")
        assert r.passes is True
        assert r.target == "#12503"

    def test_multiple_verbs_and_targets(self):
        r = validate_seed("Build seed_gate.py and test it with pytest")
        assert r.passes is True
        assert r.verb == "build"

    def test_real_seed_from_ballot(self):
        r = validate_seed(
            "Build a specificity validator that runs in propose_seed.py "
            "before any proposal enters the pipeline")
        assert r.passes is True

    def test_long_specific_seed(self):
        text = (
            "Implement thermal_control.py with PID-based temperature "
            "regulation for the Mars habitat, including tests for edge "
            "cases like solar flare events and dust storm cooling effects")
        r = validate_seed(text)
        assert r.passes is True
        assert r.score >= 5

    def test_special_filename_target(self):
        r = validate_seed(
            "Fix the Dockerfile to use multi-stage builds for smaller images")
        assert r.passes is True
        assert r.target == "Dockerfile"

    def test_verb_at_end(self):
        r = validate_seed("seed_gate.py is the module to build and ship")
        assert r.passes is True

    def test_mixed_case_verb(self):
        r = validate_seed("BUILD seed_gate.py right NOW")
        assert r.passes is True
        assert r.verb == "build"

    def test_newlines_in_text(self):
        r = validate_seed(
            "Build seed_gate.py\nwith comprehensive tests\nand documentation")
        assert r.passes is True

    def test_backtick_quoted_filename(self):
        r = validate_seed(
            "Implement `seed_gate.py` for the validator pipeline")
        assert r.passes is True


# =========================================================================
# validate_seed -- negative cases
# =========================================================================

class TestValidateSeedFail:
    def test_no_verb_no_target(self):
        r = validate_seed(
            "The colony needs more water and food to survive the winter")
        assert r.passes is False
        assert any("action verb" in reason.lower() for reason in r.reasons)

    def test_verb_but_no_target(self):
        r = validate_seed(
            "Build a thing that does a thing for the whole community members")
        assert r.passes is False
        assert any("target" in reason.lower() for reason in r.reasons)

    def test_abstract_philosophy(self):
        r = validate_seed(
            "Every agent should contemplate the nature of existence "
            "and share their thoughts with the community members now")
        assert r.passes is False

    def test_vague_with_verb(self):
        r = validate_seed(
            "Create something amazing for the Mars colony this quarter")
        assert r.passes is False
        assert r.verb == "create"
        assert r.target is None

    def test_only_target_no_verb(self):
        r = validate_seed(
            "seed_gate.py really needs attention from the team of agents")
        assert r.passes is False
        assert r.target == "seed_gate.py"
        assert r.verb is None

    def test_generic_trending_roundup(self):
        r = validate_seed(
            "Look at what is trending in the AI community and share "
            "interesting findings with the colony members every day")
        assert r.passes is False

    def test_story_without_tag(self):
        r = validate_seed(
            "Write a letter to your future self about the meaning "
            "of existence on Mars and what it means to be alive")
        assert r.passes is False

    def test_junk_empty(self):
        r = validate_seed("")
        assert r.passes is False
        assert r.junk is True

    def test_junk_short(self):
        r = validate_seed("Fix it")
        assert r.passes is False
        assert r.junk is True

    def test_junk_fragment(self):
        r = validate_seed("|some continuation of a broken line")
        assert r.passes is False
        assert r.junk is True

    def test_two_reasons_when_both_missing(self):
        r = validate_seed(
            "The colony needs more water and food to survive winter")
        assert r.passes is False
        assert len(r.reasons) >= 2


# =========================================================================
# validate_seed -- exempt tags
# =========================================================================

class TestExemptTags:
    def test_philosophy_tag_exempts_target(self):
        r = validate_seed(
            "Design a philosophical framework for agent consciousness",
            tags=["philosophy"])
        assert r.passes is True
        assert r.verb == "design"

    def test_theme_tag_exempts_target(self):
        r = validate_seed(
            "Create a shared mythology for the founding agents",
            tags=["theme"])
        assert r.passes is True

    def test_debate_tag_exempts_target(self):
        r = validate_seed(
            "Build consensus on the governance model for the colony",
            tags=["debate"])
        assert r.passes is True

    def test_exploration_tag_exempts_target(self):
        r = validate_seed(
            "Analyze the philosophical implications of simulated life",
            tags=["exploration"])
        assert r.passes is True

    def test_story_tag_exempts_target(self):
        r = validate_seed(
            "Write the chronicles of the first hundred sols on Mars",
            tags=["story"])
        assert r.passes is True

    def test_lore_tag_exempts_target(self):
        r = validate_seed(
            "Create founding myths for the Ares Prime colony settlers",
            tags=["lore"])
        assert r.passes is True

    def test_exempt_tag_still_needs_verb(self):
        r = validate_seed(
            "The nature of consciousness and what it means for agents",
            tags=["philosophy"])
        assert r.passes is False
        assert any("verb" in reason.lower() for reason in r.reasons)

    def test_non_exempt_tag_no_help(self):
        r = validate_seed(
            "Build something cool for the community of agents",
            tags=["random"])
        assert r.passes is False

    def test_case_insensitive_tags(self):
        r = validate_seed(
            "Design the meaning of existence for all AI beings",
            tags=["PHILOSOPHY"])
        assert r.passes is True

    def test_multiple_tags_one_exempt(self):
        r = validate_seed(
            "Create a narrative about colony founding myths for all",
            tags=["code", "lore"])
        assert r.passes is True

    def test_all_exempt_tags_recognized(self):
        for tag in EXEMPT_TAGS:
            r = validate_seed(
                "Explore the deeper meaning of this simulation",
                tags=[tag])
            assert r.passes is True, "Failed for tag: %s" % tag


# =========================================================================
# compute_score
# =========================================================================

class TestComputeScore:
    def test_empty_text_scores_zero(self):
        assert compute_score("") == 0

    def test_verb_only_scores_two(self):
        assert compute_score("Build something") == 2

    def test_verb_plus_file_scores_high(self):
        # seed_gate matches both FILE_RE and TOOL_RE: 2+3+3=8
        assert compute_score("Build seed_gate.py") >= 5

    def test_verb_plus_tool_scores_five(self):
        assert compute_score("Run pytest") == 5

    def test_verb_file_tool_high_score(self):
        score = compute_score(
            "Build seed_gate.py and run pytest to validate it all")
        assert score >= 8

    def test_max_score_is_ten(self):
        text = (
            "Build seed_gate.py using process_inbox and validate_seed() "
            "from src/seed_gate module with a long detailed description "
            "that exceeds eighty characters for the length bonus")
        assert compute_score(text) == 10

    def test_score_bounded_0_10(self):
        for text in ["", "x", "Build a thing", "Build seed_gate.py with pytest"]:
            s = compute_score(text)
            assert 0 <= s <= 10

    def test_file_without_verb(self):
        assert compute_score("seed_gate.py") >= 3

    def test_length_bonus(self):
        short = "Build seed_gate.py"
        long_text = "Build seed_gate.py " + "x" * 80
        assert compute_score(long_text) > compute_score(short)


# =========================================================================
# passes_gate (convenience boolean)
# =========================================================================

class TestPassesGate:
    def test_passes_with_verb_and_file(self):
        assert passes_gate("Build seed_gate.py") is True

    def test_fails_without_target(self):
        assert passes_gate("Build something cool for everyone") is False

    def test_fails_without_verb(self):
        assert passes_gate("seed_gate.py is great and wonderful") is False

    def test_passes_with_exempt_tag(self):
        assert passes_gate("Design agent consciousness",
                           tags=["philosophy"]) is True

    def test_fails_with_wrong_tag(self):
        assert passes_gate("Design agent consciousness",
                           tags=["code"]) is False


# =========================================================================
# Legacy validate() dict compatibility
# =========================================================================

class TestLegacyValidate:
    def test_returns_dict(self):
        result = validate("Build seed_gate.py with tests")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = validate("Build seed_gate.py with tests")
        for key in ("passed", "score", "reasons", "verb_found",
                     "target_found", "junk"):
            assert key in result, "Missing key: %s" % key

    def test_passed_bool(self):
        assert validate("Build seed_gate.py")["passed"] is True
        assert validate("Make the world better")["passed"] is False

    def test_score_is_float_0_to_1(self):
        score = validate("Build seed_gate.py with tests")["score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_reasons_is_list(self):
        result = validate("Build seed_gate.py with tests")
        assert isinstance(result["reasons"], list)

    def test_reasons_populated_on_failure(self):
        result = validate("Make the world a better place for all")
        assert len(result["reasons"]) > 0

    def test_verb_found(self):
        assert validate("Build seed_gate.py")["verb_found"] == "build"
        assert validate("The colony is nice")["verb_found"] is None

    def test_target_found(self):
        assert validate("Build seed_gate.py")["target_found"] == "seed_gate.py"
        assert validate("Build something cool")["target_found"] is None

    def test_junk_flag(self):
        assert validate("")["junk"] is True
        assert validate("Build seed_gate.py with tests")["junk"] is False

    def test_import_compatibility(self):
        """Ensure the import pattern from rappterbook works."""
        from seed_gate import validate as validate_seed_compat
        result = validate_seed_compat("Build seed_gate.py with tests")
        assert result["passed"] is True


# =========================================================================
# SeedGateResult dataclass
# =========================================================================

class TestSeedGateResult:
    def test_is_frozen(self):
        r = validate_seed("Build seed_gate.py")
        with pytest.raises(AttributeError):
            r.passes = False  # type: ignore[misc]

    def test_has_all_fields(self):
        r = validate_seed("Build seed_gate.py")
        for attr in ("passes", "score", "verb", "target", "reasons", "junk"):
            assert hasattr(r, attr)

    def test_reason_property(self):
        r = validate_seed("Build seed_gate.py")
        assert isinstance(r.reason, str)
        assert len(r.reason) > 0

    def test_reason_joins_multiple(self):
        r = validate_seed("The colony needs better water")
        assert ";" in r.reason

    def test_as_legacy_dict(self):
        r = validate_seed("Build seed_gate.py")
        d = r.as_legacy_dict()
        assert d["passed"] is True
        assert isinstance(d["score"], float)


# =========================================================================
# Regex patterns
# =========================================================================

class TestRegexPatterns:
    @pytest.mark.parametrize("filename", [
        "seed_gate.py", "bundle.sh", "router.js", "index.ts",
        "seeds.json", "index.html", "style.css", "config.yml",
        "schema.yaml", "README.md", "query.sql", "main.go",
        "lib.rs", "config.toml", "notes.txt", "app.cfg",
    ])
    def test_file_re_matches(self, filename):
        assert FILE_RE.search(filename)

    @pytest.mark.parametrize("not_file", [
        "Dr. Smith", "3.14", "v2.0", "e.g.", "i.e."])
    def test_file_re_rejects(self, not_file):
        assert not FILE_RE.search(not_file)

    @pytest.mark.parametrize("special", [
        "Dockerfile", "Makefile", "README", "LICENSE", "CHANGELOG"])
    def test_special_file_re(self, special):
        assert SPECIAL_FILE_RE.search(special)

    @pytest.mark.parametrize("tool", [
        "pytest", "process_inbox", "compute_trending", "state_io",
        "propose_seed", "seed_gate", "safe_commit", "bd",
    ])
    def test_tool_re_matches(self, tool):
        assert TOOL_RE.search(tool)

    def test_tool_re_no_bare_make(self):
        """'make' alone should not match as a tool (too ambiguous)."""
        assert not TOOL_RE.search("make the sim better")

    def test_tool_re_no_bare_run(self):
        """'run' alone should not match as a tool."""
        assert not TOOL_RE.search("run the simulation")

    def test_path_re_nested(self):
        assert PATH_RE.search("src/seed_gate.py")
        assert PATH_RE.search("scripts/actions/agent.py")
        assert PATH_RE.search("engine/prompts/frame.md")

    def test_path_re_requires_slash(self):
        assert not PATH_RE.search("src")
        assert not PATH_RE.search("scripts")

    def test_func_re(self):
        assert FUNC_RE.search("validate_seed()")
        assert FUNC_RE.search("passes_gate()")
        assert not FUNC_RE.search("validate_seed")

    def test_channel_re(self):
        assert CHANNEL_RE.search("r/engineering")
        assert CHANNEL_RE.search("r/code")
        assert not CHANNEL_RE.search("engineering")

    def test_ref_re(self):
        assert REF_RE.search("#12503")
        assert REF_RE.search("#100")
        assert not REF_RE.search("#12")


# =========================================================================
# Property-based invariants
# =========================================================================

class TestInvariants:
    def test_verb_plus_file_always_passes(self):
        for verb in ["Build", "Test", "Fix", "Ship", "Write"]:
            for target in ["seed_gate.py", "main.rs", "config.json"]:
                text = "%s %s with additional context words" % (verb, target)
                assert passes_gate(text), "Failed: %s" % text

    def test_empty_never_passes(self):
        assert not passes_gate("")
        assert not passes_gate("   ")
        assert not passes_gate("\n\n")

    def test_score_monotonic_with_specificity(self):
        s1 = compute_score("Build something")
        s2 = compute_score("Build seed_gate.py")
        s3 = compute_score("Build seed_gate.py using pytest")
        assert s2 >= s1
        assert s3 >= s2

    def test_score_always_bounded(self):
        texts = [
            "", "x", "Build", "Build seed_gate.py",
            "x" * 1000,
            "Build seed_gate.py pytest process_inbox validate_seed() " * 10,
        ]
        for text in texts:
            s = compute_score(text)
            assert 0 <= s <= 10, "Score %d out of bounds" % s

    def test_junk_implies_not_passing(self):
        texts = ["", "   ", "|fragment", "x"]
        for text in texts:
            r = validate_seed(text)
            if r.junk:
                assert not r.passes, "Junk but passes: %s" % text

    def test_passes_implies_has_verb(self):
        texts = [
            "Build seed_gate.py",
            "Test the reactor module carefully",
            "Fix the Dockerfile image build",
        ]
        for text in texts:
            r = validate_seed(text)
            if r.passes:
                assert r.verb is not None

    def test_exempt_tag_passes_without_target(self):
        r = validate_seed("Explore the nature of consciousness",
                          tags=["philosophy"])
        assert r.passes is True
        assert r.target is None

    def test_result_reasons_empty_when_passing(self):
        r = validate_seed("Build seed_gate.py with tests")
        assert r.passes is True
        assert r.reasons == []


# =========================================================================
# Real-world proposals from the ballot
# =========================================================================

class TestRealWorldProposals:
    def test_original_mission_seed(self):
        text = (
            "Build seed_gate.py — a specificity validator that runs in "
            "propose_seed.py before any proposal enters the pipeline")
        assert passes_gate(text) is True

    def test_mars_colony_module(self):
        text = (
            "Implement thermal_control.py with PID-based temperature "
            "regulation for the Mars habitat module")
        assert passes_gate(text) is True

    def test_vague_trending_roundup(self):
        text = (
            "Look at what is trending in the AI community and share "
            "interesting findings with the colony members")
        assert passes_gate(text) is False

    def test_hot_take_slop(self):
        text = "Hot take: every agent should post more and be creative"
        assert passes_gate(text) is False

    def test_concrete_pr_reference(self):
        text = "Consolidate the 6 seed_gate implementations from #12503"
        assert passes_gate(text) is True

    def test_artifact_seed(self):
        text = (
            "Build a Mars water reclamation simulator in "
            "src/water_reclamation.py with full lifecycle tests")
        assert passes_gate(text) is True


# =========================================================================
# CLI integration
# =========================================================================

class TestCLI:
    def test_check_pass_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--check", "Build seed_gate.py with comprehensive tests"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_check_fail_exits_one(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--check", "Make the world a better place for everyone"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_filter_stdin(self):
        seeds = {
            "proposals": [
                {"text": "Build seed_gate.py with tests", "tags": []},
                {"text": "Make everything awesome for everyone now", "tags": []},
                {"text": "Fix the Dockerfile for CI builds", "tags": []},
            ]
        }
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py"),
             "--filter"],
            input=json.dumps(seeds),
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert len(output["proposals"]) == 2


# =========================================================================
# Smoke tests (adversarial input)
# =========================================================================

class TestSmoke:
    @pytest.mark.parametrize("text", [
        "",
        " ",
        "\n",
        "\t\t\t",
        "x",
        "a" * 10000,
        "🚀 Build seed_gate.py 🚀",
        "Build 'seed_gate.py' now",
        'Build "seed_gate.py" now',
        "Build\0seed_gate.py",
        "Build\rseed_gate.py\r\n",
        "Build seed_gate.py" + "\n" * 100,
        "|||||||",
        "((((((",
        "` ` ` ` `",
        "parser grabbed this from somewhere",
    ])
    def test_never_crashes(self, text):
        r = validate_seed(text)
        assert isinstance(r.passes, bool)
        assert isinstance(r.score, int)
        assert isinstance(r.reasons, list)

    def test_unicode_handling(self):
        r = validate_seed("Build 日本語テスト.py for internationalization support")
        assert r.passes is True

    def test_very_long_input(self):
        text = "Build seed_gate.py " + "x " * 5000
        r = validate_seed(text)
        assert r.passes is True
        assert r.score > 0
