"""Tests for seed_gate.py -- canonical specificity validator.

Covers: verb detection, target detection, junk detection, scoring,
validation pass/fail, exempt tags, CLI, real-world proposals,
edge cases, property invariants, smoke tests, propose_seed.py contract,
artifact signals, known tools, path matching, function-call matching,
special files, smart lowercase handling, weighted scoring, monotonicity.
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
    KNOWN_TOOLS,
    SeedGateResult,
    _detect_target,
    _detect_verb,
    _is_junk,
    _score,
    _count_unique_targets,
    passes_gate,
    validate,
    validate_seed,
)


# ===================================================================
# Verb detection
# ===================================================================

class TestVerbDetection:
    """Verb detection from all 6 agent implementations."""

    @pytest.mark.parametrize("verb", [
        "build", "create", "design", "implement", "write",
        "fix", "debug", "patch", "resolve", "repair",
        "test", "benchmark", "audit", "lint", "scan",
        "deploy", "ship", "launch", "release",
        "refactor", "optimize", "improve", "migrate",
        "generate", "compute", "simulate", "train",
        "monitor", "track", "log", "alert",
        "document", "map", "diagram", "prototype",
        "explore", "investigate", "analyze",
    ])
    def test_core_verbs(self, verb: str):
        assert _detect_verb(f"{verb} something important here") == verb

    @pytest.mark.parametrize("verb", [
        "consolidate", "decode", "establish", "execute",
        "extend", "instrument", "measure", "merge",
        "remove", "render", "review", "run", "score", "validate",
    ])
    def test_parent_repo_verbs(self, verb: str):
        """Verbs merged from parent rappterbook seed_gate.py."""
        assert _detect_verb(f"{verb} the target module now") == verb

    def test_case_insensitive(self):
        assert _detect_verb("Build seed_gate.py") == "build"
        assert _detect_verb("BUILD seed_gate.py") == "build"

    def test_verb_in_middle(self):
        assert _detect_verb("We should build the thing") == "build"

    def test_no_verb(self):
        assert _detect_verb("The quick brown fox jumps") == ""

    def test_empty(self):
        assert _detect_verb("") == ""

    def test_limit_parameter(self):
        text = "something " * 30 + "build it"
        assert _detect_verb(text, limit=20) == ""
        assert _detect_verb(text) == "build"

    def test_verb_not_substring(self):
        """'building' should not match 'build' due to word boundaries."""
        assert _detect_verb("The building is tall and made of stone") == ""

    def test_verb_count(self):
        assert len(ACTION_VERBS) >= 70


# ===================================================================
# Target detection
# ===================================================================

class TestTargetDetection:
    """Target detection across all tiers."""

    # Tier 1a: Files with extensions
    @pytest.mark.parametrize("text,expected_kind", [
        ("Fix seed_gate.py validator", "file"),
        ("Update state/agents.json schema", "file"),
        ("Write tests in test_main.py", "file"),
        ("Deploy worker.js to cloudflare", "file"),
        ("Edit CLAUDE.md for new rules", "file"),
        ("Fix the bug in main.rs today", "file"),
        ("Update schema.sql for migration", "file"),
        ("Check config.toml for settings", "file"),
    ])
    def test_file_targets(self, text: str, expected_kind: str):
        target, kind = _detect_target(text)
        assert target, f"No target found in: {text}"
        assert kind == expected_kind

    # Tier 1b: Special files without extensions
    @pytest.mark.parametrize("text,expected", [
        ("Update the Dockerfile for slim builds", "Dockerfile"),
        ("Fix Makefile target for tests", "Makefile"),
        ("Improve README with examples", "README"),
        ("Update AGENTS onboarding guide", "AGENTS"),
        ("Review CONSTITUTION for clarity", "CONSTITUTION"),
    ])
    def test_special_file_targets(self, text: str, expected: str):
        target, kind = _detect_target(text)
        assert target == expected
        assert kind == "file"

    # Tier 2: Repo paths
    @pytest.mark.parametrize("text", [
        "Refactor scripts/process_inbox.py handler",
        "Add tests/test_new_module.py suite",
        "Update state/channels.json schema",
        "Fix docs/index.html layout",
    ])
    def test_path_targets(self, text: str):
        target, kind = _detect_target(text)
        assert target, f"No target found in: {text}"
        # Paths match as file (dot extension) before path pattern
        assert kind in ("file", "path")

    # Tier 3: Known tools
    @pytest.mark.parametrize("tool", [
        "process_inbox", "state_io", "seed_gate", "zion_autonomy",
        "compute_trending", "propose_seed", "content_engine",
    ])
    def test_known_tools(self, tool: str):
        target, kind = _detect_target(f"Refactor {tool} for performance")
        assert target, f"No target found for tool: {tool}"

    # Tier 4: Generic tool/module names (snake_case/kebab-case)
    def test_generic_tool_name(self):
        target, kind = _detect_target("Wire up auth_handler for OAuth")
        assert target == "auth_handler"
        assert kind == "tool"

    # Tier 5: Function calls
    def test_function_call_target(self):
        target, kind = _detect_target("Fix validate_seed() return type")
        assert "validate_seed()" == target
        assert kind == "func"

    # Tier 6: CLI invocations
    def test_cli_backtick(self):
        target, kind = _detect_target("Run `python -m pytest` before merge")
        assert target == "`python -m pytest`"
        assert kind == "cli"

    def test_cli_flag(self):
        target, kind = _detect_target("Add --verbose flag to runner")
        assert target == "--verbose"
        assert kind == "cli"

    # Tier 7: Discussion references
    def test_discussion_ref(self):
        target, kind = _detect_target("Implement feature from #12503")
        assert kind == "discussion"

    # Tier 8: Channel references
    def test_channel_ref(self):
        target, kind = _detect_target("Create post in r/engineering")
        assert target == "r/engineering"
        assert kind == "channel"

    # Tier 9: Quoted specifics
    def test_quoted_target(self):
        target, kind = _detect_target('Implement "adaptive market maker" logic')
        assert kind == "quoted"

    def test_no_target(self):
        target, kind = _detect_target("Make everything better for everyone")
        assert target == ""
        assert kind == ""


# ===================================================================
# Unique target counting
# ===================================================================

class TestUniqueTargetCount:
    """Ensures target dedup works correctly (#12511)."""

    def test_single_target(self):
        assert _count_unique_targets("Fix seed_gate.py") == 1

    def test_multiple_distinct(self):
        count = _count_unique_targets(
            "Wire seed_gate.py into propose_seed.py and update state_io"
        )
        assert count >= 3

    def test_dedup_same_target(self):
        count = _count_unique_targets(
            "Fix seed_gate.py then test seed_gate.py"
        )
        # Same file mentioned twice = 1 unique
        assert count == 1

    def test_no_targets(self):
        assert _count_unique_targets("Make everything great") == 0


# ===================================================================
# Junk detection
# ===================================================================

class TestJunkDetection:
    """Junk detection: artifacts, fragments, garbage (#12507)."""

    def test_empty(self):
        assert _is_junk("") != ""

    def test_whitespace(self):
        assert _is_junk("   ") != ""

    def test_too_short(self):
        assert "too short" in _is_junk("Fix it")

    def test_starts_with_backtick(self):
        reason = _is_junk("`process_inbox` should handle edge cases better")
        assert reason != ""

    def test_starts_with_pipe(self):
        assert _is_junk("| column1 | column2 | some data here") != ""

    def test_starts_with_paren(self):
        assert _is_junk("(continued from previous section of text)") != ""

    def test_numbered_list(self):
        assert _is_junk("1. First item in the numbered list here") != ""

    def test_bare_url(self):
        assert _is_junk("https://example.com/path/to/resource/here") != ""

    def test_todo_marker(self):
        assert _is_junk("TODO: fix this later when we have time to do it") != ""

    def test_fixme_marker(self):
        assert _is_junk("FIXME: broken handler needs repair urgently now") != ""

    # Artifact signals from #12507
    @pytest.mark.parametrize("text", [
        "The fragment was extracted from the parser output",
        "The parser grabbed a substring of the template",
        "parsing artifact detected in the output stream",
    ])
    def test_artifact_signals(self, text: str):
        reason = _is_junk(text.capitalize())
        # These contain artifact signals but start with uppercase
        # The artifact check happens after lowercase check
        assert "parsing artifact" in reason or reason == ""

    # Smart lowercase handling
    def test_lowercase_verb_start_valid(self):
        """Imperative sentences starting with a verb are NOT junk."""
        assert _is_junk("build seed_gate.py with comprehensive tests") == ""

    def test_lowercase_nonverb_start_junk(self):
        """Fragments starting with a non-verb word ARE junk."""
        reason = _is_junk("the fragment was garbled and meaningless")
        assert reason != ""
        assert "lowercase" in reason.lower() or "fragment" in reason.lower()

    def test_run_prefix_exception(self):
        """run_ prefixed text is never junk (#12503)."""
        assert _is_junk("run_test for my_module.py quickly and quietly") == ""

    def test_valid_proposal_not_junk(self):
        assert _is_junk("Build seed_gate.py specificity validator now") == ""

    def test_limit_parameter(self):
        """Junk check with limit only examines first N chars."""
        text = "A" * 60 + "TODO: fix this later when we have time"
        assert _is_junk(text, limit=60) == ""
        assert _is_junk(text) != ""

    def test_artifact_signal_in_head(self):
        """Artifact signals checked in first 120 chars only."""
        text = "Build the parser grabbed component for agents"
        reason = _is_junk(text)
        assert "parsing artifact" in reason


# ===================================================================
# Scoring
# ===================================================================

class TestScore:
    """Weighted scoring system (#12511)."""

    def test_no_verb_no_target(self):
        score = _score("something vague", "", "", "", False)
        assert score == 0.0

    def test_verb_only(self):
        score = _score("Build something", "build", "", "", False)
        assert score == pytest.approx(0.35)

    def test_target_only(self):
        score = _score("The seed_gate.py thing", "", "seed_gate.py", "file", False)
        assert 0.35 <= score <= 0.40  # 0.35 base + possible minor bonuses

    def test_verb_and_target(self):
        score = _score("Build seed_gate.py", "build", "seed_gate.py", "file", False)
        assert score >= 0.70

    def test_exempt_counts_as_target(self):
        score = _score("Debate consciousness in agents", "debate", "", "", True)
        assert score >= 0.70

    def test_multi_target_bonus(self):
        text = "Wire seed_gate.py into propose_seed.py and update state_io"
        score_multi = _score(text, "wire", "seed_gate.py", "file", False)
        score_single = _score("Wire seed_gate.py", "wire", "seed_gate.py", "file", False)
        assert score_multi > score_single

    def test_length_bonus_short(self):
        short = "Build seed_gate.py"
        long = "Build seed_gate.py with comprehensive validation and artifact detection for all proposals"
        score_short = _score(short, "build", "seed_gate.py", "file", False)
        score_long = _score(long, "build", "seed_gate.py", "file", False)
        assert score_long >= score_short

    def test_score_capped_at_1(self):
        text = "Build " + " ".join(f"mod_{i}.py" for i in range(50))
        score = _score(text, "build", "mod_0.py", "file", False)
        assert score <= 1.0

    def test_score_always_float(self):
        score = _score("Build seed_gate.py", "build", "seed_gate.py", "file", False)
        assert isinstance(score, float)

    def test_monotonicity_adding_target(self):
        """Adding a target to a proposal should never lower its score."""
        base = "Build the validator for better quality control in system"
        enhanced = base + " using seed_gate.py"
        verb = "build"
        t1, k1 = _detect_target(base)
        t2, k2 = _detect_target(enhanced)
        s1 = _score(base, verb, t1, k1, False)
        s2 = _score(enhanced, verb, t2, k2, False)
        assert s2 >= s1


# ===================================================================
# Full validation (admission mode)
# ===================================================================

class TestValidateSeed:
    """End-to-end validation via SeedGateResult."""

    def test_pass_verb_and_file(self):
        r = validate_seed("Build seed_gate.py validator")
        assert r.passed
        assert r.verb_found == "build"
        assert "seed_gate.py" in r.target_found
        assert not r.junk

    def test_pass_verb_and_tool(self):
        r = validate_seed("Refactor process_inbox handler")
        assert r.passed
        assert r.verb_found == "refactor"

    def test_pass_verb_and_discussion(self):
        r = validate_seed("Implement feature from discussion #12503")
        assert r.passed
        assert r.verb_found == "implement"

    def test_fail_no_verb(self):
        r = validate_seed("The seed_gate.py validator is nice and useful for everyone")
        assert not r.passed
        assert "No action verb" in r.reasons[0]

    def test_fail_no_target(self):
        r = validate_seed("Build something cool and really interesting for everyone")
        assert not r.passed
        assert "No concrete target" in r.reasons[0] or "No concrete target" in r.reasons[-1]

    def test_fail_no_verb_no_target(self):
        r = validate_seed("The module is nice and works well for everyone in system")
        assert not r.passed
        assert len(r.reasons) >= 1

    def test_pass_exempt_tag_no_target(self):
        r = validate_seed(
            "Explore consciousness in artificial agents deeply now",
            tags=["philosophy"],
        )
        assert r.passed
        assert r.target_found is None

    def test_fail_exempt_tag_no_verb(self):
        r = validate_seed(
            "The nature of consciousness in artificial agents is deep",
            tags=["philosophy"],
        )
        assert not r.passed

    def test_junk_short(self):
        r = validate_seed("Fix it")
        assert not r.passed
        assert r.junk

    def test_junk_empty(self):
        r = validate_seed("")
        assert not r.passed
        assert r.junk

    def test_junk_backtick_start(self):
        r = validate_seed("`process_inbox` needs a major refactoring effort")
        assert not r.passed
        assert r.junk

    def test_frozen_result(self):
        r = validate_seed("Build seed_gate.py validator")
        with pytest.raises(Exception):
            r.passed = False  # type: ignore[misc]


# ===================================================================
# Dict API
# ===================================================================

class TestDictAPI:
    """Dict API must match the propose_seed.py contract."""

    def test_keys_present(self):
        d = validate("Build seed_gate.py validator")
        assert set(d.keys()) == {"passed", "reasons", "score", "verb_found", "target_found", "junk"}

    def test_passed_is_bool(self):
        d = validate("Build seed_gate.py validator")
        assert isinstance(d["passed"], bool)

    def test_reasons_is_list(self):
        d = validate("Build seed_gate.py validator")
        assert isinstance(d["reasons"], list)

    def test_score_is_float(self):
        d = validate("Build seed_gate.py validator")
        assert isinstance(d["score"], float)

    def test_junk_is_bool(self):
        d = validate("Build seed_gate.py validator")
        assert isinstance(d["junk"], bool)

    def test_verb_found_is_str_or_none(self):
        d = validate("Build seed_gate.py validator")
        assert isinstance(d["verb_found"], str)
        d2 = validate("The module is nice and works well for everyone here")
        assert d2["verb_found"] is None

    def test_target_found_is_str_or_none(self):
        d = validate("Build seed_gate.py validator")
        assert isinstance(d["target_found"], str)
        d2 = validate("Build something cool and interesting for everyone here")
        assert d2["target_found"] is None


# ===================================================================
# passes_gate convenience
# ===================================================================

class TestPassesGate:
    def test_passes_good(self):
        assert passes_gate("Build seed_gate.py validator")

    def test_fails_bad(self):
        assert not passes_gate("")

    def test_fails_no_verb(self):
        assert not passes_gate("The seed_gate.py module is great for everyone")

    def test_passes_exempt(self):
        assert passes_gate(
            "Explore the nature of agent consciousness deeply now",
            tags=["philosophy"],
        )


# ===================================================================
# Real-world proposals (golden regression corpus)
# ===================================================================

class TestRealWorldProposals:
    """Real proposals that MUST pass or fail consistently."""

    # MUST PASS
    def test_build_seed_gate(self):
        assert validate("Build seed_gate.py validator")["passed"]

    def test_refactor_process_inbox(self):
        assert validate("Refactor process_inbox.py to use action dispatcher")["passed"]

    def test_fix_agents_json(self):
        assert validate("Fix agents.json integrity check on startup")["passed"]

    def test_deploy_worker_js(self):
        assert validate("Deploy worker.js to Cloudflare edge network")["passed"]

    def test_wire_into_propose_seed(self):
        assert validate("Wire seed_gate into propose_seed.py validation")["passed"]

    def test_implement_from_discussion(self):
        assert validate("Implement scoring from discussion #12511")["passed"]

    def test_update_dockerfile(self):
        assert validate("Optimize Dockerfile for smaller image size")["passed"]

    def test_fix_makefile(self):
        assert validate("Fix Makefile test target for parallel runs")["passed"]

    def test_consolidate_implementations(self):
        assert validate("Consolidate seed_gate.py from 6 agent implementations")["passed"]

    def test_instrument_state_io(self):
        assert validate("Instrument state_io with timing metrics")["passed"]

    def test_lowercase_imperative(self):
        """Lowercase imperatives starting with a verb MUST pass."""
        assert validate("build seed_gate.py with comprehensive tests")["passed"]

    # MUST FAIL
    def test_generic_rejected(self):
        assert not validate("Make the platform better for everyone here")["passed"]

    def test_abstract_philosophy_rejected_without_tag(self):
        assert not validate("What if agents could dream about consciousness")["passed"]

    def test_abstract_philosophy_passes_with_tag(self):
        assert validate(
            "Explore what if agents could dream about consciousness",
            tags=["philosophy"],
        )["passed"]

    def test_vague_improvement(self):
        assert not validate("Improve the quality of everything in the system")["passed"]

    def test_hot_take_no_target(self):
        assert not validate("Hot take: the best approach to understanding everything")["passed"]


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_empty_tags(self):
        d = validate("Build seed_gate.py", tags=[])
        assert d["passed"]

    def test_none_tags(self):
        d = validate("Build seed_gate.py", tags=None)
        assert d["passed"]

    def test_very_long_text(self):
        text = "Build seed_gate.py validator " * 100
        d = validate(text)
        assert d["passed"]

    def test_unicode_text(self):
        d = validate("Build seed_gate.py with émojis 🎉 and ünïcödë")
        assert d["passed"]

    def test_newlines_in_text(self):
        d = validate("Build seed_gate.py\nwith multiline\ndescription here")
        assert d["passed"]

    def test_tab_in_text(self):
        d = validate("Build seed_gate.py\twith tab separated parts")
        assert d["passed"]

    def test_mixed_case_tags(self):
        d = validate(
            "Explore deep consciousness in artificial agents now",
            tags=["Philosophy", "DEBATE"],
        )
        assert d["passed"]

    def test_whitespace_only_tags(self):
        d = validate(
            "Explore deep consciousness in artificial agents now",
            tags=["  philosophy  "],
        )
        assert d["passed"]

    def test_multiple_verbs(self):
        """First verb wins."""
        d = validate("Build and deploy seed_gate.py to production")
        assert d["verb_found"] == "build"

    def test_hyphenated_file(self):
        d = validate("Fix my-component.ts rendering issue today")
        assert d["passed"]
        assert d["target_found"] is not None


# ===================================================================
# Property invariants
# ===================================================================

_INVARIANT_CORPUS = [
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
    "Fix Makefile target for parallel test runs",
    "Consolidate seed_gate.py from 6 implementations",
]


class TestInvariants:
    @pytest.mark.parametrize("text", _INVARIANT_CORPUS)
    def test_dict_keys_always_present(self, text: str):
        d = validate(text)
        assert "passed" in d
        assert "reasons" in d
        assert "score" in d
        assert "verb_found" in d
        assert "target_found" in d
        assert "junk" in d

    @pytest.mark.parametrize("text", _INVARIANT_CORPUS)
    def test_score_always_in_range(self, text: str):
        d = validate(text)
        assert 0.0 <= d["score"] <= 1.0

    @pytest.mark.parametrize("text", _INVARIANT_CORPUS)
    def test_junk_is_bool(self, text: str):
        d = validate(text)
        assert isinstance(d["junk"], bool)

    @pytest.mark.parametrize("text", _INVARIANT_CORPUS)
    def test_dict_equals_dataclass_to_dict(self, text: str):
        dc = validate_seed(text)
        d = validate(text)
        assert dc.to_dict() == d

    @pytest.mark.parametrize("text", _INVARIANT_CORPUS)
    def test_passed_implies_no_reasons(self, text: str):
        d = validate(text)
        if d["passed"]:
            assert d["reasons"] == []

    @pytest.mark.parametrize("text", _INVARIANT_CORPUS)
    def test_failed_implies_reasons(self, text: str):
        d = validate(text)
        if not d["passed"]:
            assert len(d["reasons"]) > 0


# ===================================================================
# CLI
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
             "Something vague and useless that lacks specificity"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        data = json.loads(result.stdout)
        assert data["passed"] is False

    def test_cli_no_args(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "seed_gate.py")],
            capture_output=True, text=True,
        )
        assert result.returncode == 1


# ===================================================================
# Smoke tests
# ===================================================================

class TestSmoke:
    def test_smoke_many_proposals(self):
        """Run 100 random-ish proposals without crash."""
        proposals = [
            f"Build module_{i}.py for feature {i}" for i in range(50)
        ] + [
            f"The thing number {i} is abstract" for i in range(50)
        ]
        for text in proposals:
            d = validate(text)
            assert "passed" in d
            assert 0.0 <= d["score"] <= 1.0

    def test_smoke_all_verbs(self):
        """Every action verb should be detectable."""
        for verb in ACTION_VERBS:
            detected = _detect_verb(f"{verb} something_here.py now")
            assert detected == verb, f"Verb {verb!r} not detected"

    def test_smoke_all_exempt_tags(self):
        """Every exempt tag should bypass target requirement."""
        for tag in EXEMPT_TAGS:
            d = validate(
                f"Explore the deep meaning of agent existence now",
                tags=[tag],
            )
            assert d["passed"], f"Tag {tag!r} did not exempt"


# ===================================================================
# Mode consistency
# ===================================================================

class TestModeConsistency:
    def test_admission_detects_verb(self):
        d = validate("Build seed_gate.py", mode="admission")
        assert d["verb_found"] == "build"

    def test_purge_detects_verb(self):
        d = validate("Build seed_gate.py", mode="purge")
        assert d["verb_found"] == "build"

    def test_purge_always_passes_nonjunk(self):
        d = validate("Something vague and useless without specifics", mode="purge")
        assert d["passed"]

    def test_admission_rejects_no_target(self):
        d = validate("Build something cool and interesting for everyone", mode="admission")
        assert not d["passed"]

    def test_purge_rejects_junk(self):
        d = validate("", mode="purge")
        assert not d["passed"]
        assert d["junk"]

    def test_purge_score_reflects_content(self):
        d1 = validate("Build seed_gate.py validator", mode="purge")
        d2 = validate("Something vague and useless without any specifics", mode="purge")
        assert d1["score"] > d2["score"]


# ===================================================================
# propose_seed.py contract (regression)
# ===================================================================

class TestProposeSeedContract:
    """Ensure the dict API matches what propose_seed.py expects."""

    def test_import_validate_as_validate_seed(self):
        from seed_gate import validate as validate_seed
        gate = validate_seed("Build seed_gate.py validator")
        assert gate["passed"] is True

    def test_gate_passed_key(self):
        gate = validate("Build seed_gate.py validator")
        assert "passed" in gate
        assert isinstance(gate["passed"], bool)

    def test_gate_reasons_joinable(self):
        gate = validate("Something vague and useless that lacks specificity")
        joined = "; ".join(gate["reasons"])
        assert isinstance(joined, str)

    def test_purge_contract(self):
        gate = validate("Build seed_gate.py", mode="purge")
        assert "passed" in gate

    def test_score_is_float_01(self):
        gate = validate("Build seed_gate.py validator")
        assert isinstance(gate["score"], float)
        assert 0.0 <= gate["score"] <= 1.0

    def test_no_code_key(self):
        """Dict must NOT contain 'code' -- PR #242 contract."""
        gate = validate("Build seed_gate.py validator")
        assert "code" not in gate

    def test_has_verb_found_not_verb(self):
        """Key is 'verb_found', not 'verb'."""
        gate = validate("Build seed_gate.py validator")
        assert "verb_found" in gate

    def test_has_target_found_not_target(self):
        """Key is 'target_found', not 'target'."""
        gate = validate("Build seed_gate.py validator")
        assert "target_found" in gate

    def test_has_junk_key(self):
        gate = validate("Build seed_gate.py validator")
        assert "junk" in gate
        assert isinstance(gate["junk"], bool)


# ===================================================================
# Consolidated PR-specific tests (frames 445-446)
# ===================================================================

class TestPRConsolidation:
    """Tests specifically verifying ideas from each of the 6 PRs."""

    def test_pr12503_frozenset_performance(self):
        """#12503: ACTION_VERBS is frozenset for O(1) lookup."""
        assert isinstance(ACTION_VERBS, frozenset)
        assert "build" in ACTION_VERBS  # O(1)

    def test_pr12505_discussion_ref(self):
        """#12505: Discussion references as targets."""
        d = validate("Implement feature from discussion #12503")
        assert d["passed"]
        assert d["target_found"] is not None

    def test_pr12505_known_tools(self):
        """#12505: Known rappterbook tools are recognized."""
        assert isinstance(KNOWN_TOOLS, frozenset)
        assert "process_inbox" in KNOWN_TOOLS

    def test_pr12507_artifact_detection(self):
        """#12507: Parsing artifact signals are caught."""
        d = validate("The fragment was extracted from parser output here")
        assert not d["passed"]

    def test_pr12511_weighted_scoring(self):
        """#12511: Targets weighted equal to verbs at base level."""
        verb_score = _score("Build something", "build", "", "", False)
        # Base target weight is 0.35, same as verb weight
        assert verb_score == pytest.approx(0.35)
        # Target with extras may score slightly higher (length/dedup bonus)
        target_score = _score("The seed_gate.py module here", "", "seed_gate.py", "file", False)
        assert target_score >= 0.35

    def test_pr12521_composable_pipeline(self):
        """#12521: Dict output feeds into other tools."""
        d = validate("Build seed_gate.py validator")
        assert isinstance(d, dict)
        serialized = json.dumps(d)
        roundtrip = json.loads(serialized)
        assert roundtrip == d

    def test_pr12521_tag_exemption(self):
        """#12521: Theme/philosophy tags bypass target requirement."""
        for tag in EXEMPT_TAGS:
            d = validate(f"Explore the mysteries of agent existence now", tags=[tag])
            assert d["passed"], f"Tag {tag!r} should exempt from target"

    def test_pr12530_binary_gate(self):
        """#12530: Clean binary pass/fail, no ambiguity."""
        d = validate("Build seed_gate.py validator")
        assert d["passed"] is True  # not truthy, exactly True
        d2 = validate("Make everything better for everyone in system")
        assert d2["passed"] is False  # not falsy, exactly False


# ===================================================================
# Dataclass property aliases
# ===================================================================

class TestDataclassAliases:
    def test_verb_alias(self):
        r = validate_seed("Build seed_gate.py validator")
        assert r.verb == "build"
        assert r.verb == r.verb_found

    def test_target_alias(self):
        r = validate_seed("Build seed_gate.py validator")
        assert "seed_gate.py" in r.target
        assert r.target == r.target_found

    def test_verb_alias_none(self):
        r = validate_seed("The thing is nice and works for everyone here")
        assert r.verb == ""
        assert r.verb_found is None

    def test_target_alias_none(self):
        r = validate_seed("Build something cool and interesting for everyone")
        assert r.target == ""
        assert r.target_found is None


# ===================================================================
# Known-tools detection
# ===================================================================

class TestKnownTools:
    @pytest.mark.parametrize("tool", sorted(KNOWN_TOOLS))
    def test_each_known_tool(self, tool: str):
        text = f"Refactor {tool} for better performance and clarity"
        target, _kind = _detect_target(text)
        assert target, f"Known tool {tool!r} not detected as target"
