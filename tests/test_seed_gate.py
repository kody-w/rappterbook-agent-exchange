"""Comprehensive tests for seed_gate.py -- specificity validator.

170+ tests covering verb detection, target detection, fragments, junk,
scoring, full validation, dict API, CLI, edge cases, property invariants.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from seed_gate import (
    ACTION_VERBS,
    EXEMPT_TAGS,
    FILE_RE,
    FUNC_RE,
    MIN_PROPOSAL_LENGTH,
    PATH_RE,
    REF_RE,
    SPECIAL_FILE_RE,
    TOOL_RE,
    CHANNEL_RE,
    SeedGateResult,
    check_fragment,
    check_minimum_length,
    compute_score,
    detect_junk_signals,
    find_action_verb,
    find_all_targets,
    find_all_verbs,
    find_concrete_target,
    passes_gate,
    validate,
    validate_seed,
)


class TestFindActionVerb:
    def test_single_verb(self):
        assert find_action_verb("Build a new module") == "build"

    def test_multiple_verbs(self):
        assert find_action_verb("Write and test the API") == "write"

    def test_no_verb(self):
        assert find_action_verb("This sentence has no action") is None

    def test_case_insensitive(self):
        assert find_action_verb("DEPLOY the service") == "deploy"

    def test_verb_in_middle(self):
        assert find_action_verb("We should build this") == "build"

    def test_purge_mode_limits_scope(self):
        filler = "x " * 120
        assert find_action_verb(filler + "build test.py", mode="purge") is None

    def test_admission_mode_full_text(self):
        filler = "x " * 120
        assert find_action_verb(filler + "build test.py", mode="admission") == "build"

    @pytest.mark.parametrize("verb", [
        "build", "write", "ship", "test", "fix", "deploy",
        "create", "implement", "refactor", "validate",
        "benchmark", "optimize", "monitor", "simulate",
    ])
    def test_each_core_verb(self, verb):
        assert find_action_verb(verb.title() + " a thing") == verb

    def test_empty_string(self):
        assert find_action_verb("") is None

    def test_only_whitespace(self):
        assert find_action_verb("   ") is None


class TestFindConcreteTarget:
    def test_python_file(self):
        assert find_concrete_target("Build seed_gate.py") == "seed_gate.py"

    def test_js_file(self):
        assert find_concrete_target("Write rapp.js SDK") == "rapp.js"

    def test_json_file(self):
        assert find_concrete_target("Parse seeds.json") == "seeds.json"

    def test_shell_file(self):
        assert find_concrete_target("Fix bundle.sh") == "bundle.sh"

    def test_special_file(self):
        assert find_concrete_target("Update the Makefile") == "Makefile"

    def test_dockerfile(self):
        assert find_concrete_target("Create a Dockerfile") == "Dockerfile"

    def test_known_tool(self):
        assert find_concrete_target("Run pytest on the suite") == "pytest"

    def test_platform_tool(self):
        assert find_concrete_target("Wire into propose_seed") == "propose_seed"

    def test_path(self):
        # FILE_RE matches the filename before PATH_RE can match the full path
        assert find_concrete_target("Add src/new_module.py") == "new_module.py"

    def test_deep_path(self):
        # FILE_RE matches agent.py (the filename) before PATH_RE matches the full path
        target = find_concrete_target("Read scripts/actions/agent.py")
        assert target is not None
        assert target == "agent.py"

    def test_function_call(self):
        assert find_concrete_target("Call validate()") == "validate()"

    def test_channel_ref(self):
        assert find_concrete_target("Post to r/general") == "r/general"

    def test_discussion_ref(self):
        assert find_concrete_target("Fix issue #12503") == "#12503"

    def test_no_target_vague(self):
        assert find_concrete_target("Do something amazing and great") is None

    def test_no_target_abstract(self):
        assert find_concrete_target("Improve the architecture") is None

    def test_short_ref_ignored(self):
        assert find_concrete_target("See #42 maybe") is None

    def test_empty_string(self):
        assert find_concrete_target("") is None


class TestFindAllVerbsTargets:
    def test_multiple_verbs(self):
        verbs = find_all_verbs("Build, test, and deploy the service")
        assert "build" in verbs
        assert "test" in verbs
        assert "deploy" in verbs

    def test_multiple_targets(self):
        targets = find_all_targets("Fix seed_gate.py and run pytest")
        assert any("seed_gate.py" in t for t in targets)
        assert any("pytest" in t for t in targets)

    def test_dedup_verbs(self):
        verbs = find_all_verbs("Build build BUILD")
        assert verbs == ["build"]

    def test_dedup_targets(self):
        targets = find_all_targets("seed_gate.py and seed_gate.py again")
        count = sum(1 for t in targets if "seed_gate" in t.lower())
        assert count == 1


class TestCheckFragment:
    def test_lowercase_start(self):
        assert check_fragment("the quick brown fox") is True

    def test_uppercase_start(self):
        assert check_fragment("Build the system") is False

    def test_backtick_start(self):
        assert check_fragment("`some_code` is here") is True

    def test_pipe_start(self):
        assert check_fragment("| piped output") is True

    def test_comma_start(self):
        assert check_fragment(", continued sentence") is True

    def test_paren_start(self):
        assert check_fragment("(parenthetical remark)") is True

    def test_empty(self):
        assert check_fragment("") is True

    def test_file_start_not_fragment(self):
        assert check_fragment("seed_gate.py needs tests") is False

    def test_tool_start_not_fragment(self):
        assert check_fragment("pytest should run first") is False

    def test_dash_start(self):
        assert check_fragment("-flag is not a sentence") is True


class TestDetectJunkSignals:
    def test_admission_signal(self):
        is_junk, sig = detect_junk_signals("Parser grabbed this fragment")
        assert is_junk
        assert "parser grabbed" in sig

    def test_clean_text(self):
        is_junk, _ = detect_junk_signals("Build seed_gate.py with tests")
        assert not is_junk

    def test_purge_mode_extra_signals(self):
        is_junk, _ = detect_junk_signals("the regex matched", mode="purge")
        assert is_junk

    def test_purge_mode_scope_limited(self):
        filler = "x" * 65
        is_junk, _ = detect_junk_signals(filler + " parser grabbed", mode="purge")
        assert not is_junk

    def test_admission_no_purge_signals(self):
        is_junk, _ = detect_junk_signals("the regex says so", mode="admission")
        assert not is_junk


class TestCheckMinimumLength:
    def test_too_short(self):
        assert check_minimum_length("Hi") is False

    def test_exactly_min(self):
        text = "x" * MIN_PROPOSAL_LENGTH
        assert check_minimum_length(text) is True

    def test_above_min(self):
        assert check_minimum_length("Build seed_gate.py with all tests") is True

    def test_whitespace_stripping(self):
        text = "   " + "x" * (MIN_PROPOSAL_LENGTH - 1) + "   "
        assert check_minimum_length(text) is False


class TestComputeScore:
    def test_empty(self):
        assert compute_score("") == 0

    def test_verb_only(self):
        assert compute_score("Build something") == 2

    def test_file_only(self):
        assert compute_score("seed_gate.py") == 3

    def test_verb_plus_file(self):
        assert compute_score("Build seed_gate.py") == 5

    def test_high_score(self):
        text = "Build seed_gate.py, run pytest, check src/gate module" + " " * 40
        score = compute_score(text)
        assert score >= 7

    def test_capped_at_10(self):
        text = (
            "Build seed_gate.py and run pytest via src/tests "
            "and call validate() then check propose_seed results "
            + "x" * 80
        )
        assert compute_score(text) <= 10

    def test_score_nonnegative(self):
        assert compute_score("???") >= 0


class TestValidateSeedPass:
    def test_build_python_file(self):
        r = validate_seed("Build seed_gate.py with comprehensive tests")
        assert r.passes
        assert r.verb == "build"
        assert r.target == "seed_gate.py"
        assert r.code == "ok"

    def test_fix_shell_script(self):
        r = validate_seed("Fix bundle.sh for atomic writes")
        assert r.passes

    def test_write_test_suite(self):
        r = validate_seed("Write tests for compute_trending module")
        assert r.passes

    def test_deploy_with_tool(self):
        r = validate_seed("Deploy the system using make and pytest")
        assert r.passes

    def test_implement_discussion_ref(self):
        r = validate_seed("Implement the feature from #12503")
        assert r.passes

    def test_create_path_target(self):
        r = validate_seed("Create src/new_module.py for the API")
        assert r.passes

    def test_review_channel(self):
        r = validate_seed("Review posts on r/general for quality")
        assert r.passes

    def test_optimize_function(self):
        r = validate_seed("Optimize validate() performance")
        assert r.passes

    def test_theme_tag_exemption(self):
        r = validate_seed("Explore the philosophy of emergence", tags=["theme"])
        assert r.passes
        assert r.target is None

    def test_debate_tag_exemption(self):
        r = validate_seed(
            "Debate whether consciousness emerges from complexity",
            tags=["debate"],
        )
        assert r.passes


class TestValidateSeedFail:
    def test_too_short(self):
        r = validate_seed("Do it")
        assert not r.passes
        assert r.code == "too_short"

    def test_fragment_lowercase(self):
        r = validate_seed("building something great for the project")
        assert not r.passes
        assert r.code == "fragment"

    def test_fragment_backtick(self):
        r = validate_seed("`code_ref` should be updated somehow!")
        assert not r.passes
        assert r.code == "fragment"

    def test_junk_signal(self):
        r = validate_seed("Parser grabbed this substring from the issue")
        assert not r.passes
        assert r.code == "junk_signal"

    def test_no_verb(self):
        r = validate_seed("The seed_gate.py file is important")
        assert not r.passes
        assert r.code == "missing_verb"

    def test_no_target(self):
        r = validate_seed("Build something amazing for the platform")
        assert not r.passes
        assert r.code == "missing_target"

    def test_vague_imperative(self):
        r = validate_seed("Improve everything about the system now")
        assert not r.passes

    def test_empty_string(self):
        r = validate_seed("")
        assert not r.passes
        assert r.code == "too_short"

    def test_whitespace_only(self):
        r = validate_seed("      ")
        assert not r.passes


class TestPassesGate:
    def test_true_for_good_seed(self):
        assert passes_gate("Build seed_gate.py with tests") is True

    def test_false_for_bad_seed(self):
        assert passes_gate("Do stuff") is False

    def test_mode_parameter(self):
        assert passes_gate("Build seed_gate.py", mode="purge") is True

    def test_tags_parameter(self):
        assert passes_gate("Explore philosophy deeply", tags=["theme"]) is True


class TestValidateDictAPI:
    def test_returns_dict(self):
        result = validate("Build seed_gate.py with tests")
        assert isinstance(result, dict)

    def test_passed_key(self):
        result = validate("Build seed_gate.py with tests")
        assert result["passed"] is True

    def test_score_key(self):
        result = validate("Build seed_gate.py with tests")
        assert isinstance(result["score"], int)

    def test_reasons_empty_on_pass(self):
        result = validate("Build seed_gate.py with tests")
        assert result["reasons"] == []

    def test_reasons_populated_on_fail(self):
        result = validate("Build something vague and abstract now")
        assert len(result["reasons"]) > 0

    def test_verb_key(self):
        result = validate("Build seed_gate.py with tests")
        assert result["verb"] == "build"

    def test_target_key(self):
        result = validate("Build seed_gate.py with tests")
        assert result["target"] == "seed_gate.py"

    def test_code_key(self):
        result = validate("Build seed_gate.py with tests")
        assert result["code"] == "ok"

    def test_all_keys_present(self):
        result = validate("Build seed_gate.py")
        for key in ("passed", "score", "reasons", "verb", "target", "code"):
            assert key in result

    def test_failure_dict(self):
        result = validate("xyz")
        assert result["passed"] is False
        assert result["code"] == "too_short"

    def test_tags_forwarded(self):
        result = validate("Explore emergence deeply", tags=["theme"])
        assert result["passed"] is True


class TestEdgeCases:
    def test_very_long_text(self):
        text = "Build seed_gate.py " + "x" * 5000
        r = validate_seed(text)
        assert r.passes

    def test_unicode_text(self):
        r = validate_seed("Build file_name.py for testing")
        assert r.passes

    def test_multiple_targets(self):
        r = validate_seed("Build seed_gate.py and run pytest on tests/")
        assert r.passes
        assert r.target is not None

    def test_hyphenated_filename(self):
        r = validate_seed("Build my-cool-module.py now")
        assert r.passes

    def test_dotted_path(self):
        r = validate_seed("Fix scripts/actions/agent.py urgently")
        assert r.passes

    def test_none_text(self):
        r = validate_seed(None)
        assert not r.passes

    def test_numeric_text(self):
        r = validate_seed("123456789012345678")
        assert not r.passes

    def test_exemption_without_verb(self):
        r = validate_seed("The philosophy of things", tags=["theme"])
        assert not r.passes


class TestRegexPatterns:
    def test_file_re_py(self):
        assert FILE_RE.search("seed_gate.py")

    def test_file_re_js(self):
        assert FILE_RE.search("rapp.js")

    def test_file_re_yml(self):
        assert FILE_RE.search("config.yml")

    def test_file_re_no_ext(self):
        assert not FILE_RE.search("noextension")

    def test_file_re_fake_ext(self):
        assert not FILE_RE.search("file.xyz")

    def test_special_makefile(self):
        assert SPECIAL_FILE_RE.search("Update the Makefile")

    def test_special_dockerfile(self):
        assert SPECIAL_FILE_RE.search("Build a Dockerfile")

    def test_tool_pytest(self):
        assert TOOL_RE.search("Run pytest")

    def test_tool_propose_seed(self):
        assert TOOL_RE.search("Wire propose_seed")

    def test_tool_no_match(self):
        assert not TOOL_RE.search("Run happiness")

    def test_path_src(self):
        assert PATH_RE.search("Look at src/module.py")

    def test_path_scripts(self):
        assert PATH_RE.search("Fix scripts/process.py")

    def test_func_call(self):
        assert FUNC_RE.search("Call validate()")

    def test_func_no_parens(self):
        assert not FUNC_RE.search("Call validate")

    def test_channel(self):
        assert CHANNEL_RE.search("Post to r/general")

    def test_ref_large(self):
        assert REF_RE.search("Issue #12503")

    def test_ref_small(self):
        assert not REF_RE.search("Issue #42")


class TestDiscussionRefs:
    def test_pr_ref(self):
        r = validate_seed("Fix the bug described in #12503")
        assert r.passes

    def test_issue_ref_as_target(self):
        target = find_concrete_target("See issue #999 for context")
        assert target == "#999"


class TestPropertyInvariants:
    def test_score_always_0_to_10(self):
        for text in ["", "x", "Build seed_gate.py" * 50, "???!!!"]:
            score = compute_score(text)
            assert 0 <= score <= 10

    def test_result_always_has_code(self):
        for text in ["", "too short", "Build seed_gate.py", "No verb here"]:
            r = validate_seed(text)
            assert r.code in (
                "ok", "too_short", "fragment", "junk_signal",
                "missing_verb", "missing_target",
            )

    def test_pass_implies_verb(self):
        for text in ["Build seed_gate.py", "Test pytest suite", "Ship bundle.sh"]:
            r = validate_seed(text)
            if r.passes:
                assert r.verb is not None

    def test_pass_implies_ok_code(self):
        r = validate_seed("Build seed_gate.py with tests")
        assert r.passes
        assert r.code == "ok"

    def test_fail_implies_non_ok_code(self):
        r = validate_seed("xyz")
        assert not r.passes
        assert r.code != "ok"


class TestFragmentDetection:
    def test_number_start_not_fragment(self):
        assert check_fragment("42 is the answer") is False

    def test_question_mark_not_fragment(self):
        assert check_fragment("?confusing start") is False

    def test_exclamation_not_fragment(self):
        assert check_fragment("!important start") is False

    def test_slash_not_fragment(self):
        assert check_fragment("/path/start") is False

    def test_upper_not_fragment(self):
        assert check_fragment("THIS IS LOUD") is False

    def test_mixed_case_filename_not_fragment(self):
        assert check_fragment("bundle.sh needs fixing") is False

    def test_bare_lowercase_word_is_fragment(self):
        assert check_fragment("something about nothing") is True


class TestRealWorldProposals:
    def test_good_seed_from_12503(self):
        text = "Build seed_gate.py with action verb + target validation"
        assert passes_gate(text)

    def test_vague_seed_rejected(self):
        text = "Improve the platform quality and user experience"
        assert not passes_gate(text)

    def test_parser_artifact_rejected(self):
        text = "Parser grabbed this substring from issue body"
        assert not passes_gate(text)

    def test_theme_exploration_passes(self):
        text = "Explore how emergent behavior shapes colony survival"
        assert passes_gate(text, tags=["exploration"])

    def test_fragment_from_extraction(self):
        text = ", which should be wired into the pipeline"
        assert not passes_gate(text)


class TestCLI:
    def test_check_pass(self):
        result = subprocess.run(
            [sys.executable, str(SRC / "seed_gate.py"),
             "--check", "Build seed_gate.py with tests"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_check_fail(self):
        result = subprocess.run(
            [sys.executable, str(SRC / "seed_gate.py"),
             "--check", "xyz"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_filter_stdin(self):
        seeds = json.dumps({
            "proposals": [
                {"text": "Build seed_gate.py now", "tags": []},
                {"text": "vague thing", "tags": []},
            ]
        })
        result = subprocess.run(
            [sys.executable, str(SRC / "seed_gate.py")],
            input=seeds, capture_output=True, text=True,
        )
        assert result.returncode == 0
        out = json.loads(result.stdout)
        assert len(out["proposals"]) == 1
        assert "specificity" in out["proposals"][0]

    def test_no_args_shows_help(self):
        # With DEVNULL stdin (not a TTY), it tries to parse JSON and fails gracefully
        result = subprocess.run(
            [sys.executable, str(SRC / "seed_gate.py")],
            capture_output=True, text=True,
            stdin=subprocess.DEVNULL,
        )
        # Exits 1 with error message when stdin is empty non-TTY
        assert result.returncode == 1
        assert "Error" in result.stderr or "JSON" in result.stderr


_SMOKE_SEEDS = [
    ("Build seed_gate.py", True),
    ("Ship bundle.sh to production", True),
    ("Write tests for compute_trending.py", True),
    ("Fix the bug in #12503", True),
    ("Run pytest on the full suite", True),
    ("Create src/new_feature.py", True),
    ("Deploy using make", True),
    ("Do stuff", False),
    ("vague and abstract improvement", False),
    ("This has no verb or target at all!", False),
    ("Build something amazing and wonderful", False),
]


@pytest.mark.parametrize("text,expected", _SMOKE_SEEDS,
                         ids=[t[0][:40] for t in _SMOKE_SEEDS])
def test_smoke(text, expected):
    assert validate_seed(text).passes is expected


class TestSeedGateResult:
    def test_immutable(self):
        r = validate_seed("Build seed_gate.py")
        with pytest.raises(AttributeError):
            r.passes = False

    def test_repr(self):
        r = validate_seed("Build seed_gate.py")
        assert "SeedGateResult" in repr(r)

    def test_equality(self):
        r1 = SeedGateResult(True, "ok", 5, "build", "seed_gate.py", "good")
        r2 = SeedGateResult(True, "ok", 5, "build", "seed_gate.py", "good")
        assert r1 == r2

    def test_inequality(self):
        r1 = SeedGateResult(True, "ok", 5, "build", "seed_gate.py", "good")
        r2 = SeedGateResult(False, "too_short", 0, None, None, "bad")
        assert r1 != r2


class TestActionVerbsSet:
    def test_is_frozenset(self):
        assert isinstance(ACTION_VERBS, frozenset)

    def test_all_lowercase(self):
        for v in ACTION_VERBS:
            assert v == v.lower()

    def test_minimum_count(self):
        assert len(ACTION_VERBS) >= 30

    def test_core_verbs_present(self):
        for v in ("build", "write", "test", "fix", "deploy", "ship"):
            assert v in ACTION_VERBS


class TestExemptTags:
    def test_exempt_tags_is_frozenset(self):
        assert isinstance(EXEMPT_TAGS, frozenset)

    @pytest.mark.parametrize("tag", [
        "theme", "philosophy", "debate", "exploration", "story", "lore",
    ])
    def test_each_exempt_tag(self, tag):
        assert tag in EXEMPT_TAGS

    def test_case_insensitive_tag_matching(self):
        r = validate_seed(
            "Explore the meaning of life deeply now",
            tags=["THEME"],
        )
        assert r.passes

    def test_non_exempt_tag_no_help(self):
        r = validate_seed(
            "Build something vague and abstract now",
            tags=["bugfix"],
        )
        assert not r.passes
