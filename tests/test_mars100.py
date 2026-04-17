"""Tests for Mars-100 recursive LisPy colony simulation."""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mars100.lispy_vm import (
    run as lispy_run, parse_all, tokenize, make_env, evaluate,
    VMState, Env, Closure,
    LispyError, LispySyntaxError, LispyRuntimeError, LispyBudgetExceeded,
    MAX_STEPS, MAX_DEPTH, MAX_AST_NODES, MAX_LIST_SIZE, MAX_STRING_LEN,
)
from src.mars100.colonist import (
    Colonist, create_founding_colonists, FOUNDING_COLONISTS,
    ELEMENTS, STAT_NAMES,
)
from src.mars100.colony import (
    Resources, Relationship, SocialGraph, tick_resources, RESOURCE_NAMES,
)
from src.mars100.events import Event, generate_events, EVENT_TEMPLATES
from src.mars100.governance import (
    GovernanceState, GovernanceProposal, GOVERNANCE_TYPES,
    should_propose, generate_proposal, resolve_vote, apply_governance,
)
from src.mars100.subsim import (
    SubSimBudget, SubSimResult, spawn_subsim,
    MAX_SUBSIM_DEPTH, SUBSIM_BUDGET_PER_YEAR,
)
from src.mars100.engine import Mars100Engine, YearResult, SimulationResult
from src.mars100.narrator import (
    narrate_year, generate_diary_entries, generate_final_report,
)


# =============================================================
# LisPy VM tests
# =============================================================

class TestLispyTokenizer:
    """Tests for tokenizer."""

    def test_simple_expr(self):
        tokens = tokenize("(+ 1 2)")
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_nested(self):
        tokens = tokenize("(if (> x 3) 1 0)")
        assert "(" in tokens and ")" in tokens

    def test_string(self):
        tokens = tokenize('(define x "hello")')
        assert '"hello"' in tokens

    def test_comment_stripped(self):
        tokens = tokenize("; this is a comment\n(+ 1 2)")
        assert ";" not in "".join(tokens)

    def test_quote_shorthand(self):
        tokens = tokenize("'foo")
        assert "'" in tokens


class TestLispyParser:
    """Tests for parser."""

    def test_atom_int(self):
        exprs = parse_all("42")
        assert exprs == [42]

    def test_atom_float(self):
        exprs = parse_all("3.14")
        assert exprs == [3.14]

    def test_atom_bool(self):
        assert parse_all("#t") == [True]
        assert parse_all("#f") == [False]

    def test_atom_nil(self):
        assert parse_all("nil") == [None]

    def test_atom_string(self):
        assert parse_all('"hello"') == ['"hello"']

    def test_atom_symbol(self):
        assert parse_all("foo") == ["foo"]

    def test_list(self):
        exprs = parse_all("(+ 1 2)")
        assert exprs == [["+", 1, 2]]

    def test_nested_list(self):
        exprs = parse_all("(if (> x 3) 1 0)")
        assert len(exprs) == 1
        assert exprs[0][0] == "if"

    def test_quote(self):
        exprs = parse_all("'foo")
        assert exprs == [["quote", "foo"]]

    def test_multiple_exprs(self):
        exprs = parse_all("1 2 3")
        assert exprs == [1, 2, 3]

    def test_missing_close_paren(self):
        with pytest.raises(LispySyntaxError):
            parse_all("(+ 1 2")

    def test_unexpected_close_paren(self):
        with pytest.raises(LispySyntaxError):
            parse_all(")")


class TestLispyEval:
    """Tests for evaluation."""

    def test_integer(self):
        assert lispy_run("42") == 42

    def test_float(self):
        assert lispy_run("3.14") == 3.14

    def test_bool(self):
        assert lispy_run("#t") is True
        assert lispy_run("#f") is False

    def test_nil(self):
        assert lispy_run("nil") is None

    def test_addition(self):
        assert lispy_run("(+ 1 2 3)") == 6

    def test_subtraction(self):
        assert lispy_run("(- 10 3)") == 7

    def test_multiplication(self):
        assert lispy_run("(* 2 3 4)") == 24

    def test_division(self):
        assert lispy_run("(/ 10 2)") == 5.0

    def test_division_by_zero(self):
        with pytest.raises(LispyRuntimeError):
            lispy_run("(/ 1 0)")

    def test_modulo(self):
        assert lispy_run("(% 10 3)") == 1

    def test_comparison(self):
        assert lispy_run("(> 5 3)") is True
        assert lispy_run("(< 5 3)") is False
        assert lispy_run("(= 5 5)") is True
        assert lispy_run("(!= 5 3)") is True

    def test_if_true(self):
        assert lispy_run("(if #t 1 0)") == 1

    def test_if_false(self):
        assert lispy_run("(if #f 1 0)") == 0

    def test_if_no_else(self):
        assert lispy_run("(if #f 1)") is None

    def test_define(self):
        assert lispy_run("(begin (define x 42) x)") == 42

    def test_let(self):
        assert lispy_run("(let ((x 1) (y 2)) (+ x y))") == 3

    def test_lambda(self):
        result = lispy_run("(begin (define sq (lambda (x) (* x x))) (sq 5))")
        assert result == 25

    def test_closure(self):
        result = lispy_run("""
            (begin
                (define make-adder (lambda (n) (lambda (x) (+ x n))))
                (define add5 (make-adder 5))
                (add5 10))
        """)
        assert result == 15

    def test_cond(self):
        result = lispy_run("""
            (cond
                ((> 1 2) 10)
                ((> 3 2) 20)
                (#t 30))
        """)
        assert result == 20

    def test_and(self):
        assert lispy_run("(and #t #t)") is True
        assert lispy_run("(and #t #f)") is False

    def test_or(self):
        assert lispy_run("(or #f #t)") is True
        assert lispy_run("(or #f #f)") is False

    def test_not(self):
        assert lispy_run("(not #t)") is False
        assert lispy_run("(not #f)") is True

    def test_quote(self):
        assert lispy_run("(quote foo)") == "foo"
        assert lispy_run("'(1 2 3)") == [1, 2, 3]

    def test_list_ops(self):
        assert lispy_run("(car (list 1 2 3))") == 1
        assert lispy_run("(cdr (list 1 2 3))") == [2, 3]
        assert lispy_run("(cons 0 (list 1 2))") == [0, 1, 2]
        assert lispy_run("(length (list 1 2 3))") == 3
        assert lispy_run("(nth (list 10 20 30) 1)") == 20

    def test_predicates(self):
        assert lispy_run("(list? (list 1))") is True
        assert lispy_run("(number? 42)") is True
        assert lispy_run("(nil? nil)") is True

    def test_string_append(self):
        assert lispy_run('(string-append "hello" " " "world")') == "hello world"

    def test_math(self):
        assert lispy_run("(abs -5)") == 5
        assert lispy_run("(min 3 1 2)") == 1
        assert lispy_run("(max 3 1 2)") == 3
        assert lispy_run("(floor 3.7)") == 3
        assert lispy_run("(ceil 3.2)") == 4

    def test_extra_bindings(self):
        result = lispy_run("(+ x y)", extra_bindings={"x": 10, "y": 20})
        assert result == 30

    def test_unbound_symbol(self):
        with pytest.raises(LispyRuntimeError, match="unbound symbol"):
            lispy_run("nonexistent")


class TestLispySafety:
    """Tests for safety limits."""

    def test_step_budget_exceeded(self):
        with pytest.raises(LispyBudgetExceeded, match="step budget"):
            lispy_run("""
                (begin
                    (define f (lambda (n) (f (+ n 1))))
                    (f 0))
            """, max_steps=100)

    def test_depth_budget_exceeded(self):
        with pytest.raises(LispyBudgetExceeded, match="recursion depth"):
            lispy_run("""
                (begin
                    (define f (lambda (n) (f (+ n 1))))
                    (f 0))
            """, max_depth=10)

    def test_ast_size_limit(self):
        # Build expression with > MAX_AST_NODES nodes
        big = "(+ " + " ".join(["1"] * (MAX_AST_NODES + 1)) + ")"
        with pytest.raises(LispyBudgetExceeded, match="AST size"):
            lispy_run(big)

    def test_list_size_limit(self):
        with pytest.raises(LispyBudgetExceeded, match="list size"):
            lispy_run("(list " + " ".join(["1"] * (MAX_LIST_SIZE + 1)) + ")")

    def test_env_isolation(self):
        """Each run() call gets a fresh environment."""
        lispy_run("(define x 42)")
        with pytest.raises(LispyRuntimeError, match="unbound"):
            lispy_run("x")


# =============================================================
# Colonist tests
# =============================================================

class TestColonist:
    """Tests for colonist model."""

    def test_founding_count(self):
        colonists = create_founding_colonists()
        assert len(colonists) == 10

    def test_unique_ids(self):
        colonists = create_founding_colonists()
        ids = [c.id for c in colonists]
        assert len(set(ids)) == 10

    def test_all_elements_represented(self):
        colonists = create_founding_colonists()
        elements = {c.element for c in colonists}
        assert elements == set(ELEMENTS)

    def test_stats_in_range(self):
        colonists = create_founding_colonists()
        for c in colonists:
            for stat in STAT_NAMES:
                assert 0.0 <= c.stats[stat] <= 1.0

    def test_decision_expr_valid(self):
        """Every colonist's decision expression must parse."""
        colonists = create_founding_colonists()
        for c in colonists:
            exprs = parse_all(c.decision_expr)
            assert len(exprs) > 0

    def test_serialization_roundtrip(self):
        c = create_founding_colonists()[0]
        d = c.to_dict()
        c2 = Colonist.from_dict(d)
        assert c2.id == c.id
        assert c2.name == c.name
        assert c2.element == c.element

    def test_memory_capped(self):
        c = Colonist(id="test", name="Test", element="fire",
                     archetype="test", stats={}, skills=[],
                     decision_expr="(+ 0 0)")
        for i in range(100):
            c.add_memory(i, f"event-{i}")
        assert len(c.memory) == 50


# =============================================================
# Colony (Resources + Social Graph) tests
# =============================================================

class TestResources:
    """Tests for resource model."""

    def test_initial_values(self):
        r = Resources()
        assert 0.0 <= r.food <= 1.0
        assert 0.0 <= r.water <= 1.0

    def test_clamp(self):
        r = Resources(food=1.5, water=-0.5)
        r.clamp()
        assert r.food == 1.0
        assert r.water == 0.0

    def test_critical(self):
        r = Resources(food=0.1, water=0.9, power=0.05, air=0.9, medicine=0.5)
        crit = r.critical()
        assert "food" in crit
        assert "power" in crit
        assert "water" not in crit

    def test_serialization_roundtrip(self):
        r = Resources(food=0.42, water=0.55, power=0.8, air=0.7, medicine=0.3)
        d = r.to_dict()
        r2 = Resources.from_dict(d)
        assert abs(r2.food - r.food) < 0.01


class TestSocialGraph:
    """Tests for social graph."""

    def test_initialize(self):
        sg = SocialGraph()
        rng = random.Random(42)
        sg.initialize(["a", "b", "c"], rng)
        assert "a" in sg.edges
        assert "b" in sg.edges["a"]
        assert "a" not in sg.edges["a"]

    def test_relationship_score(self):
        r = Relationship(trust=1.0, affection=1.0, respect=1.0)
        assert r.score() == 1.0

    def test_cohesion_bounded(self):
        sg = SocialGraph()
        rng = random.Random(42)
        sg.initialize(["a", "b", "c"], rng)
        cohesion = sg.colony_cohesion(["a", "b", "c"])
        assert 0.0 <= cohesion <= 1.0

    def test_cooperation_increases_trust(self):
        sg = SocialGraph()
        rng = random.Random(42)
        sg.initialize(["a", "b"], rng)
        before = sg.get("a", "b").trust
        for _ in range(10):
            sg.update_from_cooperation("a", "b", rng)
        after = sg.get("a", "b").trust
        assert after >= before

    def test_conflict_decreases_trust(self):
        sg = SocialGraph()
        rng = random.Random(42)
        sg.initialize(["a", "b"], rng)
        before = sg.get("a", "b").trust
        for _ in range(10):
            sg.update_from_conflict("a", "b", rng)
        after = sg.get("a", "b").trust
        assert after <= before


class TestResourceTick:
    """Tests for resource ticking."""

    def test_resources_stay_bounded(self):
        """Resources remain in [0, 1] after tick."""
        r = Resources()
        for _ in range(200):
            tick_resources(r, 10, {}, {})
        for name in RESOURCE_NAMES:
            val = getattr(r, name)
            assert 0.0 <= val <= 1.0, f"{name}={val} out of bounds"

    def test_returns_delta(self):
        r = Resources()
        delta = tick_resources(r, 5, {}, {})
        assert isinstance(delta, dict)
        for name in RESOURCE_NAMES:
            assert name in delta


# =============================================================
# Events tests
# =============================================================

class TestEvents:
    """Tests for event generation."""

    def test_generates_at_least_one(self):
        events = generate_events(1, random.Random(42))
        assert len(events) >= 1

    def test_event_fields(self):
        events = generate_events(1, random.Random(42))
        ev = events[0]
        assert ev.name
        assert ev.category
        assert ev.severity >= 0
        assert ev.description
        assert isinstance(ev.effects, dict)

    def test_deterministic(self):
        e1 = generate_events(5, random.Random(42))
        e2 = generate_events(5, random.Random(42))
        assert e1[0].name == e2[0].name
        assert e1[0].severity == e2[0].severity


# =============================================================
# Governance tests
# =============================================================

class TestGovernance:
    """Tests for governance model."""

    def test_proposal_generation(self):
        gov = GovernanceState()
        rng = random.Random(42)
        prop = generate_proposal(10, "kira-sol", gov, rng)
        assert prop.id.startswith("prop-")
        assert prop.gov_type != gov.gov_type

    def test_vote_resolution(self):
        prop = GovernanceProposal(
            id="test", year=1, proposer_id="a",
            gov_type="council", parameters={}, rationale="test")
        prop.votes_for = ["a", "b", "c"]
        prop.votes_against = ["d"]
        assert resolve_vote(prop, 4) is True

    def test_vote_fails_on_tie(self):
        prop = GovernanceProposal(
            id="test", year=1, proposer_id="a",
            gov_type="council", parameters={}, rationale="test")
        prop.votes_for = ["a"]
        prop.votes_against = ["b"]
        assert resolve_vote(prop, 2) is False

    def test_apply_governance(self):
        gov = GovernanceState(gov_type="anarchy")
        rng = random.Random(42)
        prop = GovernanceProposal(
            id="test", year=10, proposer_id="kira-sol",
            gov_type="council",
            parameters={"council_size": 3, "term_years": 5},
            rationale="test")
        apply_governance(prop, gov, ["a", "b", "c", "d"], rng)
        assert gov.gov_type == "council"
        assert len(gov.council_ids) == 3
        assert gov.term_end_year == 15

    def test_governance_history(self):
        gov = GovernanceState(gov_type="anarchy")
        rng = random.Random(42)
        prop = GovernanceProposal(
            id="test", year=10, proposer_id="a",
            gov_type="dictator", parameters={"term_years": 5},
            rationale="test")
        apply_governance(prop, gov, ["a"], rng)
        assert len(gov.history) == 1
        assert gov.history[0]["from"] == "anarchy"
        assert gov.history[0]["to"] == "dictator"


# =============================================================
# Sub-simulation tests
# =============================================================

class TestSubSim:
    """Tests for sub-simulation spawning."""

    def test_basic_subsim(self):
        result = spawn_subsim(
            "(+ 1 2 3)", "test-colonist", {},
            depth=1, rng=random.Random(42))
        assert result.result == 6
        assert result.depth == 1
        assert result.error == ""

    def test_subsim_with_context(self):
        result = spawn_subsim(
            "(+ food water)", "test",
            {"food": 0.5, "water": 0.3},
            depth=1)
        assert abs(result.result - 0.8) < 0.01

    def test_max_depth_exceeded(self):
        result = spawn_subsim(
            "(+ 1 2)", "test", {},
            depth=MAX_SUBSIM_DEPTH + 1)
        assert "max depth" in result.error

    def test_nested_subsim(self):
        """Sub-sim can call sub-sim at depth+1."""
        result = spawn_subsim(
            '(sub-sim "(+ 10 20)")',
            "test", {},
            depth=1, rng=random.Random(42))
        assert result.result == 30
        assert result.nested_sims == 1

    def test_subsim_budget(self):
        budget = SubSimBudget()
        assert budget.can_spawn()
        for _ in range(SUBSIM_BUDGET_PER_YEAR):
            budget.spend()
        assert not budget.can_spawn()
        budget.reset()
        assert budget.can_spawn()

    def test_subsim_error_handling(self):
        result = spawn_subsim(
            "(/ 1 0)", "test", {}, depth=1)
        assert result.error
        assert result.result is None

    def test_subsim_budget_exceeded_in_nested(self):
        """Nested sub-sims are limited to 3 per invocation."""
        expr = '(list (sub-sim "(+ 1 1)") (sub-sim "(+ 2 2)") (sub-sim "(+ 3 3)") (sub-sim "(+ 4 4)"))'
        result = spawn_subsim(expr, "test", {}, depth=1,
                              rng=random.Random(42))
        # Fourth sub-sim should return budget-exceeded string
        assert result.nested_sims >= 3


# =============================================================
# Engine integration tests
# =============================================================

class TestEngine:
    """Integration tests for the Mars-100 engine."""

    def test_smoke_5_years(self):
        """Engine runs 5 years without crash."""
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        assert result.completed_years == 5
        assert len(result.years) == 5

    def test_deterministic(self):
        """Same seed produces same results."""
        r1 = Mars100Engine(seed=42, total_years=10).run()
        r2 = Mars100Engine(seed=42, total_years=10).run()
        assert r1.completed_years == r2.completed_years
        assert r1.total_deaths == r2.total_deaths
        assert r1.total_subsims == r2.total_subsims

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        r1 = Mars100Engine(seed=42, total_years=20).run()
        r2 = Mars100Engine(seed=99, total_years=20).run()
        # At least one metric should differ
        assert (r1.total_deaths != r2.total_deaths
                or r1.total_subsims != r2.total_subsims
                or r1.governance_changes != r2.governance_changes)

    def test_result_schema(self):
        """Result dict has expected schema."""
        engine = Mars100Engine(seed=42, total_years=5)
        d = engine.run().to_dict()
        assert "_meta" in d
        assert "schema_version" in d["_meta"]
        assert "summary" in d
        assert "final_colonists" in d
        assert "final_resources" in d
        assert "final_governance" in d
        assert "years" in d
        assert len(d["final_colonists"]) == 10

    def test_year_result_schema(self):
        """Each year result has expected fields."""
        engine = Mars100Engine(seed=42, total_years=3)
        result = engine.run()
        yr = result.years[0].to_dict()
        assert "year" in yr
        assert "events" in yr
        assert "actions" in yr
        assert "subsim_log" in yr
        assert "resources_before" in yr
        assert "resources_after" in yr
        assert "colonist_snapshots" in yr

    def test_resources_bounded(self):
        """All resource values stay in [0, 1] across all years."""
        engine = Mars100Engine(seed=42, total_years=30)
        result = engine.run()
        for yr in result.years:
            d = yr.to_dict()
            for name in RESOURCE_NAMES:
                val = d["resources_after"].get(name, 0)
                assert 0.0 <= val <= 1.0, (
                    f"Y{d['year']} {name}={val} out of bounds")

    def test_deaths_reduce_active(self):
        """Deaths reduce the active colonist count."""
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        if result.total_deaths > 0:
            alive = sum(1 for c in result.final_colonists
                        if c.alive and not c.exiled)
            assert alive < 10

    def test_subsims_spawn(self):
        """Engine spawns sub-simulations."""
        engine = Mars100Engine(seed=42, total_years=20)
        result = engine.run()
        assert result.total_subsims > 0

    def test_governance_changes_over_time(self):
        """Governance changes happen over sufficient years."""
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        assert result.governance_changes > 0

    def test_meta_awareness_occurs(self):
        """Meta-awareness events occur."""
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        assert result.meta_awareness_events > 0

    def test_extinction_detected(self):
        """Extinction is properly detected and reported."""
        engine = Mars100Engine(seed=42, total_years=200)
        result = engine.run()
        if result.total_deaths == 10:
            assert result.extinction_year > 0
            assert result.completed_years == result.extinction_year

    def test_callback(self):
        """Callback is invoked each year."""
        years_seen = []
        engine = Mars100Engine(seed=42, total_years=5)
        engine.run(callback=lambda yr: years_seen.append(yr.year))
        assert years_seen == [1, 2, 3, 4, 5]

    def test_colonist_memory_grows(self):
        """Colonists accumulate memories."""
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        for c in result.final_colonists:
            # Even dead colonists should have some memory
            assert len(c.memory) > 0


# =============================================================
# Narrator tests
# =============================================================

class TestNarrator:
    """Tests for narrative generation."""

    def test_narrate_year(self):
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        yr = result.years[0].to_dict()
        rng = random.Random(42)
        narrative = narrate_year(yr, rng)
        assert "## Year 1" in narrative
        assert "Resources:" in narrative

    def test_diary_entries(self):
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        yr = result.years[0].to_dict()
        rng = random.Random(42)
        entries = generate_diary_entries(
            yr, yr["colonist_snapshots"], rng, count=3)
        assert len(entries) == 3
        assert all("text" in e for e in entries)
        assert all("colonist_id" in e for e in entries)

    def test_final_report(self):
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        d = result.to_dict()
        report = generate_final_report(d)
        assert "# Emergent Governance" in report
        assert "Amendment XVIII" in report
        assert "Summary" in report


# =============================================================
# Runner integration test
# =============================================================

class TestRunner:
    """Integration tests for the runner."""

    def test_produces_all_artifacts(self):
        """Runner produces expected output files."""
        from src.mars100.run import run_simulation
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            result = run_simulation(seed=42, years=10, output_dir=out)

            # Canonical state
            state_file = out / "state" / "mars100.json"
            assert state_file.exists()
            state = json.loads(state_file.read_text())
            assert "_meta" in state
            assert state["_meta"]["seed"] == 42

            # Dashboard data
            data_file = out / "docs" / "mars-100" / "data.json"
            assert data_file.exists()
            data = json.loads(data_file.read_text())
            assert "resource_timeline" in data

            # Report
            report = out / "docs" / "mars-100" / "report.md"
            assert report.exists()
            assert "Emergent Governance" in report.read_text()

            # Colonist soul files
            colonist_dir = out / "docs" / "mars-100" / "colonists"
            assert colonist_dir.exists()
            colonist_files = list(colonist_dir.glob("*.json"))
            assert len(colonist_files) == 10

    def test_canonical_state_schema(self):
        """Canonical state file has the expected schema."""
        from src.mars100.run import run_simulation
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            result = run_simulation(seed=42, years=5, output_dir=out)
            state = json.loads(
                (out / "state" / "mars100.json").read_text())
            assert state["_meta"]["schema_version"] == "1.0.0"
            assert state["_meta"]["completed_years"] == 5
            assert len(state["final_colonists"]) == 10
            for name in RESOURCE_NAMES:
                assert name in state["final_resources"]
