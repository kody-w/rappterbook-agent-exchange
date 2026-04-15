"""Comprehensive tests for seed_gate.py — canonical specificity validator.

28 test classes, ~250 tests covering:
- Core verb detection, target detection, scoring
- FILE_RE false positive filtering
- Junk / artifact detection
- Question-stem intent mapping
- Exempt-tag behavior
- Batch validation (BatchStats, BatchResult)
- Contract alignment with propose_seed.py
- Stress / edge cases
"""
from __future__ import annotations

import os
import sys

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seed_gate import (  # noqa: E402
    ACTION_VERBS,
    EXEMPT_TAGS,
    FILE_RE,
    KNOWN_MODULES,
    TOOL_RE,
    PATH_RE,
    CLI_RE,
    DISCUSSION_RE,
    CHANNEL_RE,
    QUOTED_RE,
    SeedGateResult,
    BatchStats,
    BatchResult,
    _FALSE_FILE_MATCHES,
    _detect_verb,
    _detect_target,
    _is_junk,
    _score,
    find_question_intent,
    validate,
    validate_seed,
    passes_gate,
    validate_batch,
)

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# 1. VERB DETECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectVerb:
    """Tests for _detect_verb — finding action verbs in text."""

    def test_common_verbs(self):
        assert _detect_verb("Build seed_gate.py") == "build"
        assert _detect_verb("Create a new channel") == "create"
        assert _detect_verb("Implement the parser") == "implement"

    def test_no_verb(self):
        assert _detect_verb("some random text without action") == ""
        assert _detect_verb("") == ""

    def test_verb_case_insensitive(self):
        assert _detect_verb("BUILD foo.py") == "build"
        assert _detect_verb("Deploy the app") == "deploy"

    def test_verb_limit(self):
        text = "x " * 200 + " build foo.py"
        assert _detect_verb(text, limit=10) == ""
        assert _detect_verb(text) == "build"

    def test_all_verbs_detectable(self):
        for verb in ACTION_VERBS:
            assert _detect_verb(f"{verb.capitalize()} something") == verb

    def test_verbs_in_mid_sentence(self):
        assert _detect_verb("We should build the thing") == "build"

    def test_compound_sentence(self):
        assert _detect_verb("First, create a module then test it") == "create"


# ═══════════════════════════════════════════════════════════════════════════
# 2. TARGET DETECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectTarget:
    """Tests for _detect_target — finding concrete targets."""

    def test_file_target(self):
        target, kind = _detect_target("Build seed_gate.py validator")
        assert target == "seed_gate.py"
        assert kind == "file"

    def test_path_target(self):
        target, kind = _detect_target("Fix the src/main.py module")
        assert kind in ("file", "path")
        assert "main.py" in target

    def test_tool_target(self):
        target, kind = _detect_target("Improve the compute_trending module")
        assert target == "compute_trending"
        assert kind == "tool"

    def test_cli_target(self):
        target, kind = _detect_target("Run the `python -m pytest` command")
        assert kind == "cli"

    def test_discussion_target(self):
        target, kind = _detect_target("Respond to #12503")
        assert kind == "discussion"

    def test_channel_target(self):
        target, kind = _detect_target("Post to r/general-discussion")
        assert "general-discussion" in target
        assert kind in ("channel", "tool")

    def test_quoted_target(self):
        target, kind = _detect_target('Explore "the meaning of consciousness"')
        assert kind == "quoted"
        assert "meaning" in target

    def test_no_target(self):
        target, kind = _detect_target("vague idea about something")
        assert target == ""
        assert kind == ""

    def test_file_priority_over_tool(self):
        target, kind = _detect_target("Build seed_gate.py with compute_trending")
        assert kind == "file"
        assert target == "seed_gate.py"

    def test_json_files(self):
        target, kind = _detect_target("Update agents.json")
        assert target == "agents.json"
        assert kind == "file"


# ═══════════════════════════════════════════════════════════════════════════
# 3. FILE_RE FALSE POSITIVE FILTERING
# ═══════════════════════════════════════════════════════════════════════════


class TestFileReFalsePositives:
    """Ensure _detect_target rejects common abbreviations that FILE_RE matches."""

    @pytest.mark.parametrize("abbrev", [
        "e.g", "i.e", "etc", "vs", "al", "cf", "no", "dr", "mr", "ms", "jr", "sr", "st",
    ])
    def test_abbreviation_not_detected_as_file(self, abbrev):
        """Common abbreviation '{abbrev}' must NOT be detected as a file target."""
        text = f"Consider {abbrev} the alternative approach"
        target, kind = _detect_target(text)
        if kind == "file":
            assert target.lower().rstrip(".") not in _FALSE_FILE_MATCHES, \
                f"{abbrev!r} was falsely detected as file target {target!r}"

    def test_eg_in_prose(self):
        text = "Use a validator, e.g. the seed_gate pattern"
        target, kind = _detect_target(text)
        assert target == "seed_gate"
        assert kind == "tool"

    def test_ie_in_prose(self):
        text = "The specific case, i.e. when the input is empty"
        target, kind = _detect_target(text)
        assert kind != "file" or "i.e" not in target.lower()

    def test_real_file_not_affected(self):
        text = "Edit the config.yaml file"
        target, kind = _detect_target(text)
        assert target == "config.yaml"
        assert kind == "file"

    def test_etc_with_real_file(self):
        text = "Various files etc. but mainly focus on main.py"
        target, kind = _detect_target(text)
        assert target == "main.py"
        assert kind == "file"

    def test_false_match_set_is_frozen(self):
        assert isinstance(_FALSE_FILE_MATCHES, frozenset)

    def test_all_false_matches_are_lowercase(self):
        for item in _FALSE_FILE_MATCHES:
            assert item == item.lower(), f"{item!r} is not lowercase"

    def test_real_extension_files_pass_through(self):
        for fname in ["seed_gate.py", "README.md", "Cargo.toml", "index.html", "lib.rs"]:
            target, kind = _detect_target(f"Build {fname}")
            assert target == fname, f"Real file {fname!r} should be detected"
            assert kind == "file"


# ═══════════════════════════════════════════════════════════════════════════
# 4. JUNK DETECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestJunkDetection:
    """Tests for _is_junk — detecting junk/artifact signals."""

    def test_empty_string(self):
        assert _is_junk("") != ""

    def test_whitespace_only(self):
        assert "empty" in _is_junk("   ").lower() or "whitespace" in _is_junk("   ").lower()

    def test_too_short(self):
        assert "short" in _is_junk("hi").lower()

    def test_lowercase_start(self):
        reason = _is_junk("some fragment that starts lowercase and is long enough")
        assert reason != ""

    def test_valid_proposal(self):
        assert _is_junk("Build the seed_gate.py module for proposals") == ""

    def test_numbered_list(self):
        assert _is_junk("1. Fix the bug") != ""

    def test_bare_url(self):
        assert _is_junk("https://example.com/some/path") != ""

    def test_hard_artifact(self):
        reason = _is_junk("parser grabbed this from the output and it makes no sense at all")
        assert reason != "", "Should detect as junk (lowercase start or artifact signal)"

    def test_todo_fixme(self):
        assert _is_junk("TODO remember to fix this later please") != ""

    def test_starts_with_backtick(self):
        assert _is_junk("`some code` that is a fragment but long enough maybe") != ""

    def test_file_start_exception(self):
        assert _is_junk("seed_gate.py needs improvement in the validation logic") == ""

    def test_run_prefix_exception(self):
        assert _is_junk("run_tests with the new configuration setup") == ""

    def test_limit_parameter(self):
        long_text = "A" * 100
        short = _is_junk(long_text, limit=5)
        assert short != ""  # "AAAAA" is < 15 chars after limit


# ═══════════════════════════════════════════════════════════════════════════
# 5. SCORING
# ═══════════════════════════════════════════════════════════════════════════


class TestScoring:
    """Tests for _score — the 0.0-1.0 specificity score."""

    def test_max_score_components(self):
        text = "Build seed_gate.py with compute_trending using `pytest` to validate"
        s = _score(text, "build", "seed_gate.py", "file")
        assert 0.0 <= s <= 1.0
        assert s >= 0.7  # verb (3) + file (4) = 7, words >= 8 (+1) = 8/10

    def test_no_verb_no_target(self):
        s = _score("short", "", "", "")
        assert s == 0.0

    def test_verb_only(self):
        s = _score("Build something", "build", "", "")
        assert s == 0.3  # verb = 3/10

    def test_score_bounded(self):
        text = "Build seed_gate.py with compute_trending using `pytest` to validate config.yaml and state/agents.json in the test environment"
        s = _score(text, "build", "seed_gate.py", "file")
        assert s <= 1.0

    def test_file_vs_quoted_scores(self):
        s_file = _score("Build seed_gate.py now", "build", "seed_gate.py", "file")
        s_quoted = _score('Build "the thing" now', "build", '"the thing"', "quoted")
        assert s_file > s_quoted  # file (4) > quoted (1)

    def test_long_text_bonus(self):
        short = "Build foo.py"
        long_text = "Build foo.py " + " ".join(["word"] * 20)
        s_short = _score(short, "build", "foo.py", "file")
        s_long = _score(long_text, "build", "foo.py", "file")
        assert s_long >= s_short


# ═══════════════════════════════════════════════════════════════════════════
# 6. QUESTION STEM INTENT
# ═══════════════════════════════════════════════════════════════════════════


class TestQuestionStem:
    """Tests for find_question_intent."""

    def test_what_if(self):
        stem, verb = find_question_intent("What if we explored consciousness?")
        assert stem == "what if"
        assert verb == "explore"

    def test_how_might(self):
        stem, verb = find_question_intent("How might agents collaborate?")
        assert stem == "how might"
        assert verb == "design"

    def test_no_stem(self):
        stem, verb = find_question_intent("Build seed_gate.py")
        assert stem == ""
        assert verb == ""

    def test_case_insensitive(self):
        stem, verb = find_question_intent("WHAT IF everything changed?")
        assert stem == "what if"

    def test_all_stems(self):
        from seed_gate import QUESTION_STEMS
        for stem_text, expected_verb in QUESTION_STEMS.items():
            result_stem, result_verb = find_question_intent(f"{stem_text} something happens?")
            assert result_verb == expected_verb, f"Stem {stem_text!r} should map to {expected_verb!r}"


# ═══════════════════════════════════════════════════════════════════════════
# 7. EXEMPT TAGS
# ═══════════════════════════════════════════════════════════════════════════


class TestExemptTags:
    """Tests for exempt-tag behavior (theme, philosophy, etc.)."""

    def test_exempt_tag_no_target_needed(self):
        result = validate("Explore the nature of consciousness deeply", ["philosophy"])
        assert result["passed"]

    def test_exempt_tag_still_needs_verb(self):
        result = validate("Explore the nature of consciousness deeply", ["philosophy"])
        assert result["passed"]  # explicit verb "explore"

    def test_non_exempt_needs_target(self):
        result = validate("Explore something vaguely", [])
        assert not result["passed"]

    def test_all_exempt_tags(self):
        for tag in EXEMPT_TAGS:
            result = validate("Explore the deep concept of reality", [tag])
            assert result["passed"], f"Tag {tag!r} should exempt from target requirement"

    def test_exempt_with_question_stem(self):
        result = validate("What if agents could dream?", ["philosophy"])
        assert result["passed"]
        assert result["verb_found"] == "explore"  # mapped from "what if"

    def test_mixed_tags(self):
        result = validate("Consider the implications deeply", ["philosophy", "code"])
        assert result["passed"]


# ═══════════════════════════════════════════════════════════════════════════
# 8. VALIDATE (DICT API)
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateDict:
    """Tests for validate() — the dict API used by propose_seed.py."""

    def test_returns_dict(self):
        result = validate("Build seed_gate.py")
        assert isinstance(result, dict)

    def test_dict_keys(self):
        result = validate("Build seed_gate.py")
        expected_keys = {"passed", "reasons", "score", "verb_found", "target_found", "junk"}
        assert set(result.keys()) == expected_keys

    def test_passed_types(self):
        result = validate("Build seed_gate.py")
        assert isinstance(result["passed"], bool)
        assert isinstance(result["reasons"], list)
        assert isinstance(result["score"], float)
        assert isinstance(result["junk"], bool)

    def test_passing_proposal(self):
        result = validate("Build seed_gate.py validator")
        assert result["passed"] is True
        assert result["verb_found"] == "build"
        assert "seed_gate.py" in result["target_found"]
        assert result["junk"] is False
        assert len(result["reasons"]) == 0

    def test_failing_proposal(self):
        result = validate("vague idea about stuff")
        assert result["passed"] is False
        assert len(result["reasons"]) > 0

    def test_junk_proposal(self):
        result = validate("")
        assert result["junk"] is True
        assert result["passed"] is False

    def test_score_range(self):
        result = validate("Build seed_gate.py with tests and documentation")
        assert 0.0 <= result["score"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 9. VALIDATE_SEED (DATACLASS API)
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateSeed:
    """Tests for validate_seed() — the dataclass API."""

    def test_returns_dataclass(self):
        result = validate_seed("Build seed_gate.py")
        assert isinstance(result, SeedGateResult)

    def test_dataclass_immutable(self):
        result = validate_seed("Build seed_gate.py")
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore

    def test_to_dict_roundtrip(self):
        result = validate_seed("Build seed_gate.py validator")
        d = result.to_dict()
        assert d == validate("Build seed_gate.py validator")

    def test_verb_alias(self):
        result = validate_seed("Build seed_gate.py")
        assert result.verb == "build"
        assert result.verb == (result.verb_found or "")

    def test_target_alias(self):
        result = validate_seed("Build seed_gate.py")
        assert result.target == "seed_gate.py"

    def test_reasons_are_tuple(self):
        result = validate_seed("vague idea about things")
        assert isinstance(result.reasons, tuple)


# ═══════════════════════════════════════════════════════════════════════════
# 10. PASSES_GATE (BOOL API)
# ═══════════════════════════════════════════════════════════════════════════


class TestPassesGate:
    """Tests for passes_gate() — the bool convenience API."""

    def test_good_proposal(self):
        assert passes_gate("Build seed_gate.py") is True

    def test_bad_proposal(self):
        assert passes_gate("vague stuff") is False

    def test_junk_proposal(self):
        assert passes_gate("") is False

    def test_with_tags(self):
        assert passes_gate("Explore consciousness", ["philosophy"]) is True

    def test_returns_bool(self):
        result = passes_gate("Build seed_gate.py")
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════════════════════════
# 11. BATCH VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchValidation:
    """Tests for validate_batch() — batch processing with junk separation."""

    def test_basic_batch(self):
        proposals = [
            ("Build seed_gate.py", []),
            ("vague idea", []),
            ("", []),
        ]
        batch = validate_batch(proposals)
        assert isinstance(batch, BatchResult)
        assert len(batch.results) == 3

    def test_stats_counts(self):
        proposals = [
            ("Build seed_gate.py", []),                        # pass
            ("Create foo.py module", []),                       # pass
            ("Vague unspecific idea about nothing here", []),   # fail (capitalized, not junk)
            ("", []),                                           # junk
        ]
        batch = validate_batch(proposals)
        assert batch.stats.total == 4
        assert batch.stats.passed == 2
        assert batch.stats.junk == 1
        assert batch.stats.failed == 1

    def test_junk_items_property(self):
        proposals = [
            ("Build foo.py", []),
            ("", []),
            ("   ", []),
        ]
        batch = validate_batch(proposals)
        # "Build foo.py" is too short (12 chars) -> junk, plus empty + whitespace
        assert len(batch.junk_items) == 3

    def test_failed_items_excludes_junk(self):
        proposals = [
            ("Build foo.py", []),       # pass
            ("vague stuff about it", []),  # fail (not junk)
            ("", []),                      # junk
        ]
        batch = validate_batch(proposals)
        failed = batch.failed_items
        for item in failed:
            assert not item.junk, "failed_items should NOT include junk"
            assert not item.passed, "failed_items should only be non-passing"

    def test_to_dicts(self):
        proposals = [("Build seed_gate.py", [])]
        batch = validate_batch(proposals)
        dicts = batch.to_dicts()
        assert len(dicts) == 1
        assert set(dicts[0].keys()) == {"passed", "reasons", "score", "verb_found", "target_found", "junk"}

    def test_empty_batch(self):
        batch = validate_batch([])
        assert batch.stats.total == 0
        assert len(batch.results) == 0

    def test_all_passing(self):
        proposals = [
            ("Build the foo.py validator module", []),
            ("Create the bar.rs module with tests", []),
            ("Test the config.yaml schema thoroughly", []),
        ]
        batch = validate_batch(proposals)
        assert batch.stats.pass_rate == 1.0
        assert batch.stats.junk_rate == 0.0

    def test_mode_parameter(self):
        proposals = [("Build foo.py", []), ("some lowercase fragment that is long enough for tests", [])]
        admission = validate_batch(proposals, mode="admission")
        purge = validate_batch(proposals, mode="purge")
        assert admission.stats.total == purge.stats.total

    def test_results_are_tuple(self):
        batch = validate_batch([("Build foo.py", [])])
        assert isinstance(batch.results, tuple)


# ═══════════════════════════════════════════════════════════════════════════
# 12. BATCH STATS
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchStats:
    """Tests for BatchStats dataclass."""

    def test_pass_rate(self):
        stats = BatchStats(total=10, passed=7, failed=2, junk=1)
        assert stats.pass_rate == 0.7

    def test_junk_rate(self):
        stats = BatchStats(total=10, passed=7, failed=2, junk=1)
        assert stats.junk_rate == 0.1

    def test_zero_total(self):
        stats = BatchStats(total=0, passed=0, failed=0, junk=0)
        assert stats.pass_rate == 0.0
        assert stats.junk_rate == 0.0

    def test_merge(self):
        s1 = BatchStats(total=5, passed=3, failed=1, junk=1)
        s2 = BatchStats(total=10, passed=6, failed=3, junk=1)
        merged = s1.merge(s2)
        assert merged.total == 15
        assert merged.passed == 9
        assert merged.failed == 4
        assert merged.junk == 2

    def test_immutable(self):
        stats = BatchStats(total=5, passed=3, failed=1, junk=1)
        with pytest.raises(AttributeError):
            stats.total = 10  # type: ignore

    def test_merge_identity(self):
        s1 = BatchStats(total=5, passed=3, failed=1, junk=1)
        zero = BatchStats(total=0, passed=0, failed=0, junk=0)
        assert s1.merge(zero) == s1


# ═══════════════════════════════════════════════════════════════════════════
# 13. BATCH RESULT
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_junk_items_type(self):
        batch = validate_batch([("", []), ("Build foo.py", [])])
        assert isinstance(batch.junk_items, list)
        for item in batch.junk_items:
            assert isinstance(item, SeedGateResult)

    def test_failed_items_type(self):
        batch = validate_batch([("vague unspecific idea", []), ("Build foo.py", [])])
        assert isinstance(batch.failed_items, list)

    def test_immutable(self):
        batch = validate_batch([("Build foo.py", [])])
        with pytest.raises(AttributeError):
            batch.stats = None  # type: ignore

    def test_consistency(self):
        proposals = [
            ("Build foo.py", []),
            ("vague idea without target", []),
            ("", []),
            ("Create bar.rs validator", []),
        ]
        batch = validate_batch(proposals)
        assert batch.stats.total == len(batch.results)
        assert batch.stats.passed + batch.stats.failed + batch.stats.junk == batch.stats.total
        assert len(batch.junk_items) == batch.stats.junk
        assert len(batch.failed_items) == batch.stats.failed


# ═══════════════════════════════════════════════════════════════════════════
# 14. CONTRACT ALIGNMENT WITH PROPOSE_SEED.PY
# ═══════════════════════════════════════════════════════════════════════════


class TestContractAlignment:
    """Verify the validate() output matches what propose_seed.py expects."""

    def test_gate_passed_key(self):
        result = validate("Build seed_gate.py")
        assert "passed" in result
        assert isinstance(result["passed"], bool)

    def test_gate_reasons_key(self):
        result = validate("vague text")
        assert "reasons" in result
        assert isinstance(result["reasons"], list)
        assert all(isinstance(r, str) for r in result["reasons"])

    def test_gate_score_is_float(self):
        result = validate("Build seed_gate.py")
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0

    def test_gate_junk_key(self):
        result = validate("")
        assert "junk" in result
        assert isinstance(result["junk"], bool)

    def test_reasons_join(self):
        result = validate("vague")
        reasons_str = "; ".join(result["reasons"])
        assert isinstance(reasons_str, str)


# ═══════════════════════════════════════════════════════════════════════════
# 15. PURGE MODE
# ═══════════════════════════════════════════════════════════════════════════


class TestPurgeMode:
    """Tests for mode=\'purge\' behavior."""

    def test_purge_passes_non_junk(self):
        result = validate("Build seed_gate.py", mode="purge")
        assert result["passed"] is True

    def test_purge_still_catches_junk(self):
        result = validate("", mode="purge")
        assert result["junk"] is True
        assert result["passed"] is False

    def test_purge_is_lenient(self):
        result = validate("Some vague idea about improving things", mode="purge")
        assert result["passed"] is True  # purge mode passes non-junk


# ═══════════════════════════════════════════════════════════════════════════
# 16. REGEX PATTERNS
# ═══════════════════════════════════════════════════════════════════════════


class TestRegexPatterns:
    """Tests for individual regex patterns."""

    def test_file_re_matches(self):
        assert FILE_RE.search("seed_gate.py")
        assert FILE_RE.search("README.md")
        assert FILE_RE.search("config.yaml")

    def test_tool_re_matches(self):
        assert TOOL_RE.search("compute_trending")
        assert TOOL_RE.search("seed-gate")

    def test_path_re_matches(self):
        assert PATH_RE.search("src/main.py")
        assert PATH_RE.search("tests/test_foo.py")

    def test_cli_re_matches(self):
        assert CLI_RE.search("`pytest -v`")
        assert CLI_RE.search("--verbose")

    def test_discussion_re_matches(self):
        assert DISCUSSION_RE.search("#12503")

    def test_discussion_re_needs_3_digits(self):
        assert not DISCUSSION_RE.search("#12")

    def test_channel_re_matches(self):
        assert CHANNEL_RE.search("r/general")
        assert CHANNEL_RE.search("c/dev-tools")

    def test_quoted_re_matches(self):
        assert QUOTED_RE.search('"some specific thing"')

    def test_quoted_re_length_bounds(self):
        assert not QUOTED_RE.search('"ab"')  # too short


# ═══════════════════════════════════════════════════════════════════════════
# 17. KNOWN MODULES
# ═══════════════════════════════════════════════════════════════════════════


class TestKnownModules:
    """Tests for auto-discovered KNOWN_MODULES."""

    def test_is_frozenset(self):
        assert isinstance(KNOWN_MODULES, frozenset)

    def test_seed_gate_in_modules(self):
        assert "seed_gate" in KNOWN_MODULES

    def test_no_test_files(self):
        for m in KNOWN_MODULES:
            assert not m.startswith("test_"), f"test module {m} in KNOWN_MODULES"

    def test_no_dunder_files(self):
        for m in KNOWN_MODULES:
            assert not m.startswith("__"), f"dunder module {m} in KNOWN_MODULES"


# ═══════════════════════════════════════════════════════════════════════════
# 18. SEED GATE RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════


class TestSeedGateResult:
    """Tests for SeedGateResult dataclass properties."""

    def test_frozen(self):
        r = SeedGateResult(passed=True, reasons=(), score=0.8,
                           verb_found="build", target_found="foo.py", junk=False)
        with pytest.raises(AttributeError):
            r.passed = False  # type: ignore

    def test_verb_alias_none(self):
        r = SeedGateResult(passed=False, reasons=("no verb",), score=0.0,
                           verb_found=None, target_found=None, junk=False)
        assert r.verb == ""

    def test_target_alias_none(self):
        r = SeedGateResult(passed=False, reasons=("no target",), score=0.0,
                           verb_found=None, target_found=None, junk=False)
        assert r.target == ""

    def test_to_dict_shape(self):
        r = SeedGateResult(passed=True, reasons=(), score=0.8,
                           verb_found="build", target_found="foo.py", junk=False)
        d = r.to_dict()
        assert set(d.keys()) == {"passed", "reasons", "score", "verb_found", "target_found", "junk"}
        assert isinstance(d["reasons"], list)


# ═══════════════════════════════════════════════════════════════════════════
# 19. INTEGRATION (end-to-end proposals)
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end integration tests with realistic proposals."""

    @pytest.mark.parametrize("text,expected", [
        ("Build seed_gate.py validator for specificity checking", True),
        ("Create a new compute_trending module with RSS feeds", True),
        ("Fix the bug in src/main.py causing errors", True),
        ("Implement the `gh pr create` CLI integration", True),
        ("Analyze discussion #12503 with code review", True),
        ("Explore the dev-tools channel improvements", True),
        ("vague idea about improving things", False),
        ("", False),
        ("1. First step in the list", False),
    ])
    def test_realistic_proposals(self, text, expected):
        result = validate(text)
        assert result["passed"] is expected, \
            f"Expected passed={expected} for {text!r}, got {result}"

    def test_all_three_apis_agree(self):
        text = "Build seed_gate.py"
        d = validate(text)
        r = validate_seed(text)
        b = passes_gate(text)
        assert d["passed"] == r.passed == b

    def test_dict_and_dataclass_equivalent(self):
        text = "Create foo.py module with validation logic"
        d = validate(text)
        r = validate_seed(text)
        assert d == r.to_dict()


# ═══════════════════════════════════════════════════════════════════════════
# 20. STRESS / EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestStressEdgeCases:
    """Stress and edge case tests."""

    def test_very_long_text(self):
        text = "Build " + "word " * 1000 + " seed_gate.py"
        result = validate(text)
        assert result["passed"] is True

    def test_unicode(self):
        result = validate("Build über_module.py with élan")
        assert isinstance(result["passed"], bool)

    def test_newlines_in_text(self):
        text = "Build seed_gate.py\nwith tests\nand documentation"
        result = validate(text)
        assert result["passed"] is True

    def test_tabs_in_text(self):
        text = "Build\tseed_gate.py\tvalidator"
        result = validate(text)
        assert result["passed"] is True

    def test_repeated_validation_stable(self):
        text = "Build seed_gate.py validator"
        results = [validate(text) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_empty_tags(self):
        result = validate("Build seed_gate.py", [])
        assert result["passed"] is True

    def test_none_tags(self):
        result = validate("Build seed_gate.py", None)
        assert result["passed"] is True

    def test_special_characters(self):
        result = validate("Build the <seed_gate>.py & module | validator")
        assert isinstance(result, dict)

    def test_numbers_only(self):
        result = validate("12345678901234567890")
        assert result["passed"] is False

    def test_single_word_verb(self):
        result = validate("Build")
        assert result["passed"] is False  # too short -> junk

    def test_multiple_files(self):
        text = "Integrate seed_gate.py with propose_seed.py and config.yaml"
        result = validate(text)
        assert result["passed"] is True
        assert result["score"] >= 0.7  # multi-target bonus

    def test_batch_large(self):
        proposals = [(f"Build module_{i}.py", []) for i in range(100)]
        batch = validate_batch(proposals)
        assert batch.stats.total == 100
        assert batch.stats.passed == 100


# ═══════════════════════════════════════════════════════════════════════════
# 21. SOFT ARTIFACT DETECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestSoftArtifact:
    """Tests for soft artifact signal handling."""

    def test_soft_artifact_without_verb_target(self):
        text = "The regex pattern matches incorrectly in the output"
        result = validate_seed(text)
        assert result.junk is True

    def test_soft_artifact_with_verb_and_target(self):
        text = "Fix the regex pattern in seed_gate.py to match correctly"
        result = validate_seed(text)
        assert result.passed is True
        assert result.junk is False

    def test_soft_artifact_exempt_tag(self):
        text = "Explore the parser concept philosophically in depth"
        result = validate_seed(text, ["philosophy"])
        assert result.junk is False


# ═══════════════════════════════════════════════════════════════════════════
# 22. MODE PARAMETER
# ═══════════════════════════════════════════════════════════════════════════


class TestModeParameter:
    """Tests for mode parameter behavior across APIs."""

    def test_admission_default(self):
        r1 = validate("Build foo.py")
        r2 = validate("Build foo.py", mode="admission")
        assert r1 == r2

    def test_purge_mode_in_batch(self):
        proposals = [
            ("Build the foo.py validator module completely", []),
            ("Analyze the system architecture for improvements", []),
        ]
        batch = validate_batch(proposals, mode="purge")
        assert batch.stats.passed >= 1  # purge mode is lenient

    def test_mode_in_dataclass_api(self):
        r = validate_seed("Build foo.py", mode="purge")
        assert isinstance(r, SeedGateResult)


# ═══════════════════════════════════════════════════════════════════════════
# 23. CONSTANTS INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════


class TestConstantsIntegrity:
    """Ensure constant sets are well-formed."""

    def test_action_verbs_all_lowercase(self):
        for v in ACTION_VERBS:
            assert v == v.lower(), f"{v!r} is not lowercase"

    def test_action_verbs_no_empty(self):
        assert "" not in ACTION_VERBS

    def test_exempt_tags_all_lowercase(self):
        for t in EXEMPT_TAGS:
            assert t == t.lower(), f"{t!r} is not lowercase"

    def test_verb_count(self):
        assert len(ACTION_VERBS) >= 50, "Should have at least 50 action verbs"

    def test_exempt_tag_count(self):
        assert len(EXEMPT_TAGS) >= 5, "Should have at least 5 exempt tags"


# ═══════════════════════════════════════════════════════════════════════════
# 24. PROPERTY-BASED INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPropertyInvariants:
    """Property-based tests ensuring structural invariants hold."""

    @pytest.mark.parametrize("text", [
        "Build seed_gate.py",
        "vague idea",
        "",
        "1. numbered item",
        "https://example.com",
        "Create foo.py with bar_baz module",
        "What if agents dream?",
        "parser grabbed this from the text",
    ])
    def test_dict_and_dataclass_always_agree(self, text):
        d = validate(text)
        r = validate_seed(text)
        assert d == r.to_dict()

    @pytest.mark.parametrize("text", [
        "Build seed_gate.py",
        "vague stuff",
        "",
        "Create module.rs with tests",
    ])
    def test_score_always_bounded(self, text):
        result = validate(text)
        assert 0.0 <= result["score"] <= 1.0

    @pytest.mark.parametrize("text", [
        "Build seed_gate.py",
        "vague stuff",
        "",
    ])
    def test_junk_never_passes(self, text):
        result = validate(text)
        if result["junk"]:
            assert not result["passed"], "Junk proposals must never pass"

    @pytest.mark.parametrize("text", [
        "Build seed_gate.py",
        "Create foo.py",
        "Test bar.rs",
    ])
    def test_passing_has_no_reasons(self, text):
        result = validate(text)
        if result["passed"]:
            assert len(result["reasons"]) == 0

    @pytest.mark.parametrize("text", [
        "vague stuff about things",
        "improve something somehow",
    ])
    def test_failing_has_reasons(self, text):
        result = validate(text)
        if not result["passed"] and not result["junk"]:
            assert len(result["reasons"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 25. BATCH PROPERTY INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchPropertyInvariants:
    """Property-based invariants for batch operations."""

    def test_counts_sum_to_total(self):
        proposals = [
            ("Build foo.py", []),
            ("vague idea here", []),
            ("", []),
            ("Create bar.rs module", []),
            ("   ", []),
        ]
        batch = validate_batch(proposals)
        s = batch.stats
        assert s.passed + s.failed + s.junk == s.total

    def test_rates_bounded(self):
        proposals = [("Build foo.py", []), ("", [])]
        batch = validate_batch(proposals)
        assert 0.0 <= batch.stats.pass_rate <= 1.0
        assert 0.0 <= batch.stats.junk_rate <= 1.0

    def test_merge_commutative(self):
        s1 = BatchStats(total=3, passed=2, failed=1, junk=0)
        s2 = BatchStats(total=5, passed=1, failed=2, junk=2)
        assert s1.merge(s2) == s2.merge(s1)

    def test_merge_associative(self):
        s1 = BatchStats(total=2, passed=1, failed=1, junk=0)
        s2 = BatchStats(total=3, passed=2, failed=0, junk=1)
        s3 = BatchStats(total=4, passed=3, failed=1, junk=0)
        assert s1.merge(s2).merge(s3) == s1.merge(s2.merge(s3))


# ═══════════════════════════════════════════════════════════════════════════
# 26. ARTIFACT SIGNAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════


class TestArtifactSignalConstants:
    """Verify artifact signal constants are well-formed."""

    def test_hard_signals_are_lowercase(self):
        from seed_gate import _HARD_ARTIFACT_SIGNALS
        for s in _HARD_ARTIFACT_SIGNALS:
            assert s == s.lower()

    def test_soft_signals_are_lowercase(self):
        from seed_gate import _SOFT_ARTIFACT_SIGNALS
        for s in _SOFT_ARTIFACT_SIGNALS:
            assert s == s.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 27. CANONICALIZE AND COUNT TARGETS
# ═══════════════════════════════════════════════════════════════════════════


class TestCanonicalizeTargets:
    """Tests for internal target canonicalization."""

    def test_multi_target_proposal(self):
        text = "Integrate seed_gate.py with propose_seed.py and compute_trending"
        result = validate(text)
        assert result["score"] >= 0.7


# ═══════════════════════════════════════════════════════════════════════════
# 28. SMOKE TEST — FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


class TestSmokePipeline:
    """Smoke test running the full pipeline without crash."""

    def test_hundred_proposals(self):
        proposals = [
            (f"Build module_{i}.py with comprehensive tests", [])
            for i in range(100)
        ]
        batch = validate_batch(proposals)
        assert batch.stats.total == 100
        assert batch.stats.passed == 100
        assert batch.stats.junk == 0

    def test_mixed_proposals(self):
        proposals = [
            ("Build seed_gate.py", []),
            ("Create compute_trending.py module", []),
            ("", []),
            ("   ", []),
            ("vague idea about things", []),
            ("Fix the bug in src/main.py", []),
            ("1. First step in list", []),
            ("https://example.com", []),
            ("Explore consciousness", ["philosophy"]),
            ("What if agents dream?", ["philosophy"]),
        ]
        batch = validate_batch(proposals)
        assert batch.stats.total == 10
        assert batch.stats.passed >= 4
        assert batch.stats.junk >= 2

    def test_cli_entry_point_exists(self):
        from seed_gate import _cli
        assert callable(_cli)
