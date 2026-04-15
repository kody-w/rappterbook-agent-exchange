"""New tests for seed_gate.py evolution — version filters, colon prefixes,
centralized file validation, expanded verbs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seed_gate import (
    ACTION_VERBS,
    _COMMIT_PREFIX_RE,
    _VERSION_RE,
    _is_false_file_match,
    compute_score,
    count_unique_targets,
    find_target,
    find_verb,
    is_junk,
    validate,
    validate_batch,
    validate_seed,
)


# ===================================================================
# Version string false-positive filter
# ===================================================================

class TestVersionFilter:
    """Version strings (v2.0, v1.2.3, 3.11) must not match as files."""

    def test_v2_0(self):
        assert _is_false_file_match("v2.0")

    def test_v1_2_3(self):
        assert _is_false_file_match("v1.2.3")

    def test_3_11(self):
        assert _is_false_file_match("3.11")

    def test_v0_1_0_alpha(self):
        assert _is_false_file_match("v0.1.0")

    def test_10_0(self):
        assert _is_false_file_match("10.0")

    def test_real_file_not_filtered(self):
        assert not _is_false_file_match("parser.py")

    def test_config_yaml_not_filtered(self):
        assert not _is_false_file_match("config.yaml")

    def test_dotfile_not_filtered(self):
        assert not _is_false_file_match(".gitignore")

    def test_version_re_matches_semver(self):
        assert _VERSION_RE.match("v1.2.3")

    def test_version_re_matches_bare_version(self):
        assert _VERSION_RE.match("3.11")

    def test_version_re_no_match_word(self):
        assert not _VERSION_RE.match("parser.py")


class TestVersionInValidation:
    """Validation should not use version strings as targets."""

    def test_deploy_v2_0_no_other_target(self):
        """v2.0 alone should not satisfy target requirement."""
        r = validate("Deploy v2.0 to production")
        assert not r["passed"]
        assert r["target_found"] is None

    def test_deploy_v2_0_with_real_target(self):
        """Real file after version string should be found."""
        r = validate("Deploy v2.0 with config.yaml")
        assert r["passed"]
        assert r["target_found"] == "config.yaml"

    def test_build_parser_with_python_version(self):
        """parser.py should be found, not 3.11."""
        r = validate("Build parser.py for Python 3.11")
        assert r["passed"]
        assert r["target_found"] == "parser.py"

    def test_upgrade_semver_with_real_file(self):
        r = validate("Upgrade to v1.2.3 with config.yaml support")
        assert r["target_found"] == "config.yaml"

    def test_version_does_not_inflate_unique_count(self):
        """Version strings should not count as unique targets."""
        count = count_unique_targets("Deploy v2.0 and v3.0 to production")
        assert count == 0

    def test_version_with_real_files_correct_count(self):
        """Only real files should count."""
        count = count_unique_targets("Build parser.py v2.0 and config.yaml v1.0")
        assert count == 2


# ===================================================================
# Colon-prefix (conventional commit style) handling
# ===================================================================

class TestColonPrefixJunk:
    """Commit-style prefixes should not cause lowercase junk rejection."""

    def test_build_prefix_not_junk(self):
        assert is_junk("build: fix CI pipeline for mars_colony tests") == ""

    def test_fix_prefix_not_junk(self):
        assert is_junk("fix: resolve water_mining.py overflow bug") == ""

    def test_feat_prefix_not_junk(self):
        assert is_junk("feat: add solar_array.py controller") == ""

    def test_test_prefix_not_junk(self):
        assert is_junk("test: benchmark fuel_cell.py throughput") == ""

    def test_docs_prefix_not_junk(self):
        assert is_junk("docs: update README.md guide here") == ""

    def test_refactor_scope_prefix(self):
        """refactor(parser): ... should pass junk check."""
        assert is_junk("refactor(parser): clean up seed_gate.py internals") == ""

    def test_fix_bang_prefix(self):
        """fix!: ... (breaking change) should pass junk check."""
        assert is_junk("fix!: breaking change to validate() API contract") == ""

    def test_ci_prefix(self):
        assert is_junk("ci: update workflow for automated testing") == ""

    def test_perf_prefix(self):
        assert is_junk("perf: optimize compute_trending.py hot loop") == ""

    def test_revert_prefix(self):
        assert is_junk("revert: undo PR #248 changes to seed_gate.py") == ""

    def test_empty_body_is_junk(self):
        """Prefix with no body should be caught."""
        result = is_junk("build:")
        assert result  # not empty string

    def test_prefix_only_whitespace_body(self):
        result = is_junk("fix:   ")
        assert result

    def test_case_insensitive_prefix(self):
        assert is_junk("BUILD: fix CI pipeline for mars_colony") == ""

    def test_non_prefix_lowercase_still_junk(self):
        """Words that aren't commit prefixes should still fail."""
        result = is_junk("random: nothing specific here at all")
        assert result != ""


class TestColonPrefixValidation:
    """Full validation with commit-style prefixes."""

    def test_build_fix_with_target(self):
        r = validate("build: fix CI pipeline for mars_colony tests")
        assert r["passed"]
        assert r["verb_found"] == "build"  # prefix word is first verb

    def test_fix_with_file_target(self):
        r = validate("fix: resolve water_mining.py overflow bug")
        assert r["passed"]
        assert r["target_found"] == "water_mining.py"

    def test_feat_with_file(self):
        r = validate("feat: add solar_array.py controller here")
        assert r["passed"]

    def test_docs_update_readme(self):
        r = validate("docs: update README.md installation guide")
        assert r["passed"]
        assert r["verb_found"] == "update"

    def test_refactor_scope_with_target(self):
        r = validate("refactor(parser): clean up seed_gate.py internals")
        assert r["passed"]
        assert r["target_found"] == "seed_gate.py"

    def test_chore_without_target_fails(self):
        r = validate("chore: nothing specific here at all folks today")
        assert not r["passed"]

    def test_style_without_target_fails(self):
        r = validate("style: just some random lowercase stuff here")
        assert not r["passed"]

    def test_prefix_verb_counts(self):
        """The prefix word should be findable as a verb."""
        r = validate("fix!: breaking change to validate() API contract")
        assert r["passed"]
        assert r["verb_found"] == "fix"


class TestCommitPrefixRegex:
    """Unit tests for the _COMMIT_PREFIX_RE pattern."""

    def test_simple_build(self):
        assert _COMMIT_PREFIX_RE.match("build: something")

    def test_fix_with_scope(self):
        assert _COMMIT_PREFIX_RE.match("fix(parser): something")

    def test_feat_bang(self):
        assert _COMMIT_PREFIX_RE.match("feat!: breaking")

    def test_full_conventional(self):
        assert _COMMIT_PREFIX_RE.match("refactor(core)!: rewrite")

    def test_no_colon_no_match(self):
        assert not _COMMIT_PREFIX_RE.match("build something")

    def test_unknown_prefix_no_match(self):
        assert not _COMMIT_PREFIX_RE.match("random: stuff")

    def test_case_insensitive(self):
        assert _COMMIT_PREFIX_RE.match("BUILD: uppercase")


# ===================================================================
# Expanded verbs
# ===================================================================

class TestExpandedVerbs:
    """New verbs added this frame should be recognized."""

    @pytest.mark.parametrize("verb", [
        "configure", "scaffold", "bootstrap", "provision",
        "automate", "archive", "inject", "normalize",
        "update", "delete", "enable", "disable",
    ])
    def test_verb_in_set(self, verb):
        assert verb in ACTION_VERBS

    @pytest.mark.parametrize("verb", [
        "configure", "scaffold", "bootstrap", "provision",
        "automate", "archive", "inject", "normalize",
        "update", "delete", "enable", "disable",
    ])
    def test_verb_detected(self, verb):
        assert find_verb(f"{verb} the thing") == verb

    def test_configure_passes(self):
        r = validate("Configure solar_array.py for max output")
        assert r["passed"]
        assert r["verb_found"] == "configure"

    def test_scaffold_passes(self):
        r = validate("Scaffold new water_mining.py test harness")
        assert r["passed"]

    def test_bootstrap_passes(self):
        r = validate("Bootstrap the state_io.py migration tool")
        assert r["passed"]

    def test_update_passes(self):
        r = validate("Update README.md with new installation steps")
        assert r["passed"]
        assert r["verb_found"] == "update"

    def test_delete_passes(self):
        r = validate("Delete deprecated fuel_cell.py module")
        assert r["passed"]

    def test_enable_passes(self):
        r = validate("Enable feature flags for water_mining experiments")
        assert r["passed"]


# ===================================================================
# count_unique_targets with false-file filtering
# ===================================================================

class TestUniqueTargetFiltering:
    """count_unique_targets must filter false file matches."""

    def test_versions_not_counted(self):
        assert count_unique_targets("Deploy v2.0 and v3.0") == 0

    def test_abbreviations_not_counted(self):
        assert count_unique_targets("Compare e.g. the approaches i.e. this") == 0

    def test_real_files_counted(self):
        assert count_unique_targets("Build parser.py and config.yaml") == 2

    def test_mixed_real_and_version(self):
        """Only real files should count, not version strings."""
        assert count_unique_targets("Build parser.py v2.0 and config.yaml") == 2

    def test_score_not_inflated_by_versions(self):
        """Score should not get bonus from version-string 'targets'."""
        s1 = compute_score("Deploy to production", "deploy", None, "")
        s2 = compute_score("Deploy v2.0 v3.0 v4.0 to production", "deploy", None, "")
        assert s1 == s2


# ===================================================================
# Smoke / integration
# ===================================================================

class TestEvolutionSmoke:
    """Smoke tests for the complete evolution frame."""

    def test_total_verb_count(self):
        assert len(ACTION_VERBS) >= 86

    def test_batch_with_prefixed_proposals(self):
        proposals = [
            "build: fix CI for mars_colony tests",
            "chore: vague stuff without targets",
            "",
        ]
        br = validate_batch(proposals)
        assert br.stats.passed >= 1
        assert br.stats.junk >= 1

    def test_dataclass_api_still_works(self):
        r = validate_seed("Build water_mining.py optimizer")
        assert r.passed
        assert r.score > 0
        assert r.junk is False

    def test_backward_compat_aliases(self):
        """Backward-compat aliases should still exist."""
        from seed_gate import _detect_verb, _detect_target, _is_junk
        assert _detect_verb("build something") == "build"
        assert _is_junk("") != ""


# ===================================================================
# Property invariants
# ===================================================================

class TestPropertyInvariants:
    """Property-based invariants that must hold for all inputs."""

    @pytest.mark.parametrize("text", [
        "Build water_mining.py optimizer",
        "fix: resolve water_mining.py bug",
        "Deploy v2.0 to production",
        "just some random stuff here",
        "",
        "x",
        "build: fix seed_gate.py internals",
        "What if agents could dream",
    ])
    def test_score_in_bounds(self, text):
        r = validate(text)
        assert 0.0 <= r["score"] <= 1.0

    @pytest.mark.parametrize("text", [
        "Build water_mining.py optimizer",
        "fix: resolve water_mining.py bug",
        "",
    ])
    def test_dict_has_required_keys(self, text):
        r = validate(text)
        assert "passed" in r
        assert "score" in r
        assert "reasons" in r
        assert "verb_found" in r
        assert "target_found" in r
        assert "junk" in r

    def test_passed_implies_verb(self):
        """If passed is True, verb_found must be non-None."""
        r = validate("Build water_mining.py optimizer")
        assert r["passed"]
        assert r["verb_found"] is not None

    def test_junk_implies_not_passed(self):
        """If junk is True, passed must be False."""
        r = validate("")
        assert r["junk"]
        assert not r["passed"]

    def test_unique_targets_non_negative(self):
        assert count_unique_targets("") >= 0
        assert count_unique_targets("no targets here") >= 0
