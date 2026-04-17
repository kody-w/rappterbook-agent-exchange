"""Tests for the Dreaming Engine — colonists that rewrite themselves."""
from __future__ import annotations

import random
import pytest

from src.mars100.dreaming import (
    DreamResult,
    dream,
    should_dream,
    mutate_expr,
    evaluate_fitness,
    _serialize,
    _collect_numbers,
    _collect_operators,
    _swap_operator,
    _eval_expr_numeric,
    _sample_bindings,
    ADOPTION_THRESHOLD,
    CANDIDATES_PER_DREAM,
)
from src.mars100.lispy_vm import parse_all, run as lispy_run, LispyError
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.subsim import SubSimBudget
from src.mars100.engine import Mars100Engine


# ---- Helpers ----

SAMPLE_EXPRS = [
    "(+ resolve 0.5)",
    "(if (> resolve 0.7) (+ resolve improvisation) (* empathy 2))",
    "(* empathy (+ faith 0.5))",
    "(let ((risk (* paranoia 0.5))) (- improvisation risk))",
    "(if (> faith 0.5) (* faith empathy) improvisation)",
]

SAMPLE_BINDINGS = {
    "resolve": 0.6, "improvisation": 0.5, "empathy": 0.7,
    "hoarding": 0.3, "faith": 0.4, "paranoia": 0.2,
    "terraforming": 0.5, "hydroponics": 0.3, "mediation": 0.4,
    "coding": 0.6, "prayer": 0.2, "sabotage": 0.1,
    "food": 0.6, "water": 0.7, "power": 0.8, "air": 0.9, "medicine": 0.5,
    "element": "fire", "alive": True, "memory-count": 5,
}

def _make_colonist(expr: str = "(+ resolve 0.5)") -> Colonist:
    return Colonist(
        id="test-1", name="Test", element="fire", archetype="engineer",
        stats=ColonistStats(resolve=0.6, improvisation=0.5, empathy=0.7,
                            hoarding=0.3, faith=0.4, paranoia=0.2),
        skills=ColonistSkills(terraforming=0.5, hydroponics=0.3, mediation=0.4,
                              coding=0.6, prayer=0.2, sabotage=0.1),
        decision_expr=expr,
    )


# ---- Serialization ----

class TestSerialize:
    def test_roundtrip_simple(self):
        expr = "(+ 1 2)"
        parsed = parse_all(expr)[0]
        result = _serialize(parsed)
        assert lispy_run(result) == lispy_run(expr)

    def test_roundtrip_if(self):
        expr = "(if (> 3 2) 1 0)"
        parsed = parse_all(expr)[0]
        result = _serialize(parsed)
        assert lispy_run(result) == 1

    def test_serialize_bool(self):
        assert _serialize(True) == "#t"
        assert _serialize(False) == "#f"
        assert _serialize(None) == "nil"

    def test_serialize_float(self):
        assert _serialize(0.1234) == "0.1234"

    def test_serialize_nested(self):
        expr = "(let ((x 5)) (+ x 1))"
        parsed = parse_all(expr)[0]
        result = _serialize(parsed)
        assert lispy_run(result) == 6


# ---- AST collection ----

class TestASTCollectors:
    def test_collect_numbers_finds_literals(self):
        ast = parse_all("(+ 1 2.5)")[0]
        nums = _collect_numbers(ast)
        assert len(nums) == 2
        values = [parent[idx] for parent, idx in nums]
        assert 1 in values
        assert 2.5 in values

    def test_collect_numbers_nested(self):
        ast = parse_all("(if (> x 0.3) (+ 1 0.5) 0)")[0]
        nums = _collect_numbers(ast)
        assert len(nums) == 4  # 0.3, 1, 0.5, 0

    def test_collect_operators_finds_ops(self):
        ast = parse_all("(+ 1 (* 2 3))")[0]
        ops = _collect_operators(ast)
        assert len(ops) == 2
        symbols = [parent[idx] for parent, idx in ops]
        assert "+" in symbols
        assert "*" in symbols

    def test_swap_operator_within_group(self):
        rng = random.Random(42)
        assert _swap_operator("+", rng) == "-"
        assert _swap_operator("*", rng) in {"/", "*"}
        assert _swap_operator(">", rng) == "<"


# ---- Mutation ----

class TestMutateExpr:
    def test_returns_valid_lispy(self):
        rng = random.Random(42)
        for expr in SAMPLE_EXPRS:
            for _ in range(10):
                result = mutate_expr(expr, rng)
                if result is not None:
                    parsed = parse_all(result)
                    assert len(parsed) > 0, f"Failed to parse mutated: {result}"

    def test_produces_different_output(self):
        rng = random.Random(42)
        expr = "(+ resolve 0.5)"
        different = False
        for _ in range(20):
            result = mutate_expr(expr, rng)
            if result is not None and result != expr:
                different = True
                break
        assert different, "mutation never produced a different expression"

    def test_point_mutation_changes_numbers(self):
        rng = random.Random(42)
        changed = False
        for _ in range(30):
            result = mutate_expr("(+ 1.0 0.5)", rng)
            if result is not None and result != "(+ 1.0 0.5)":
                # Verify it's still valid
                lispy_run(result, extra_bindings=SAMPLE_BINDINGS)
                changed = True
                break
        assert changed

    def test_returns_none_for_invalid(self):
        rng = random.Random(42)
        assert mutate_expr("(((", rng) is None
        assert mutate_expr("", rng) is None

    def test_respects_node_limit(self):
        rng = random.Random(42)
        # Build a large expression
        big = "(+ " * 30 + "1" + ")" * 30
        result = mutate_expr(big, rng)
        if result is not None:
            from src.mars100.lispy_vm import _count_nodes
            assert _count_nodes(parse_all(result)[0]) <= 60

    def test_deterministic_with_same_seed(self):
        expr = "(if (> resolve 0.5) 1 0)"
        r1 = mutate_expr(expr, random.Random(99))
        r2 = mutate_expr(expr, random.Random(99))
        assert r1 == r2


# ---- Fitness ----

class TestEvaluateFitness:
    def test_returns_0_to_1(self):
        rng = random.Random(42)
        for expr in SAMPLE_EXPRS:
            f = evaluate_fitness(expr, SAMPLE_BINDINGS, rng)
            assert 0.0 <= f <= 1.0, f"fitness {f} out of range for {expr}"

    def test_handles_errors_gracefully(self):
        rng = random.Random(42)
        f = evaluate_fitness("(/ 1 0)", SAMPLE_BINDINGS, rng)
        assert f == 0.0

    def test_constant_expr_has_low_diversity(self):
        rng = random.Random(42)
        f_const = evaluate_fitness("1", SAMPLE_BINDINGS, rng)
        f_varied = evaluate_fitness("(+ resolve improvisation)", SAMPLE_BINDINGS, rng)
        # Varied should score at least as high (diversity component)
        assert f_varied >= f_const * 0.5

    def test_numeric_only(self):
        rng = random.Random(42)
        f = evaluate_fitness("(list 1 2 3)", SAMPLE_BINDINGS, rng)
        assert f == 0.0  # returns a list, not numeric


# ---- eval_expr_numeric ----

class TestEvalExprNumeric:
    def test_returns_float(self):
        result = _eval_expr_numeric("(+ 1 2)", {})
        assert result == 3.0

    def test_returns_none_for_list(self):
        assert _eval_expr_numeric("(list 1 2)", {}) is None

    def test_returns_none_for_error(self):
        assert _eval_expr_numeric("(/ 1 0)", {}) is None

    def test_returns_none_for_bool(self):
        assert _eval_expr_numeric("(> 1 0)", {}) is None


# ---- should_dream ----

class TestShouldDream:
    def test_high_coding_increases_chance(self):
        rng = random.Random(42)
        high = sum(should_dream({"coding": 1.0, "faith": 0.0, "improvisation": 0.0}, random.Random(i))
                   for i in range(100))
        low = sum(should_dream({"coding": 0.0, "faith": 0.0, "improvisation": 0.0}, random.Random(i))
                  for i in range(100))
        assert high > low

    def test_high_faith_increases_chance(self):
        rng = random.Random(42)
        high = sum(should_dream({"coding": 0.0, "faith": 1.0, "improvisation": 0.0}, random.Random(i))
                   for i in range(100))
        low = sum(should_dream({"coding": 0.0, "faith": 0.0, "improvisation": 0.0}, random.Random(i))
                  for i in range(100))
        assert high > low


# ---- Dream loop ----

class TestDream:
    def test_returns_result(self):
        rng = random.Random(42)
        dr = dream("test-1", 10, "(+ resolve 0.5)", SAMPLE_BINDINGS, None, rng)
        assert isinstance(dr, DreamResult)
        assert dr.colonist_id == "test-1"
        assert dr.year == 10
        assert dr.old_expr == "(+ resolve 0.5)"

    def test_can_modify_expr(self):
        # Run many dreams to find at least one adoption
        adopted = False
        for seed in range(100):
            rng = random.Random(seed)
            dr = dream("test-1", 10, "(+ resolve 0.5)", SAMPLE_BINDINGS, None, rng)
            if dr.adopted:
                adopted = True
                assert dr.new_expr is not None
                assert dr.new_expr != "(+ resolve 0.5)"
                assert dr.fitness_delta > 0
                break
        assert adopted, "no dream adopted in 100 tries"

    def test_threshold_prevents_trivial_changes(self):
        rng = random.Random(42)
        dr = dream("test-1", 10, "(+ resolve 0.5)", SAMPLE_BINDINGS, None, rng)
        if dr.adopted:
            assert dr.fitness_delta >= ADOPTION_THRESHOLD

    def test_respects_subsim_budget(self):
        budget = SubSimBudget(year=10)
        # Exhaust budget
        for _ in range(10):
            budget.record("test-1")
        rng = random.Random(42)
        dr = dream("test-1", 10, "(+ resolve 0.5)", SAMPLE_BINDINGS, budget, rng)
        assert not dr.subsim_used

    def test_to_dict_serializable(self):
        rng = random.Random(42)
        dr = dream("test-1", 10, "(+ resolve 0.5)", SAMPLE_BINDINGS, None, rng)
        d = dr.to_dict()
        assert "colonist_id" in d
        assert "year" in d
        assert isinstance(d["fitness_delta"], float)

    def test_adopted_expr_returns_numeric(self):
        """Any adopted expression must return a finite numeric value."""
        for seed in range(200):
            rng = random.Random(seed)
            dr = dream("test-1", 10, "(+ resolve 0.5)", SAMPLE_BINDINGS, None, rng)
            if dr.adopted and dr.new_expr is not None:
                val = _eval_expr_numeric(dr.new_expr, SAMPLE_BINDINGS)
                assert val is not None, f"adopted expr returned non-numeric: {dr.new_expr}"


# ---- Colonist integration ----

class TestColonistExprHistory:
    def test_adopt_expr_records_history(self):
        c = _make_colonist()
        old = c.decision_expr
        c.adopt_expr(5, "(* empathy 2)")
        assert c.decision_expr == "(* empathy 2)"
        assert c.expr_generation == 1
        assert len(c.expr_history) == 1
        assert c.expr_history[0]["old_expr"] == old
        assert c.expr_history[0]["new_expr"] == "(* empathy 2)"

    def test_expr_history_capped_at_20(self):
        c = _make_colonist()
        for i in range(25):
            c.adopt_expr(i, f"(+ resolve {i})")
        assert len(c.expr_history) <= 20
        assert c.expr_generation == 25

    def test_to_dict_from_dict_roundtrip(self):
        c = _make_colonist()
        c.adopt_expr(5, "(* empathy 2)")
        c.adopt_expr(10, "(+ faith 0.1)")
        d = c.to_dict()
        c2 = Colonist.from_dict(d)
        assert c2.expr_generation == 2
        assert len(c2.expr_history) == 2
        assert c2.decision_expr == "(+ faith 0.1)"

    def test_child_starts_fresh(self):
        colonists = create_founding_ten(42)
        from src.mars100.colonist import create_child
        child = create_child(colonists[0], colonists[1], "child-1", 10, random.Random(42))
        assert child.expr_generation == 0
        assert child.expr_history == []


# ---- Engine integration ----

class TestDreamingInEngine:
    def test_engine_tick_includes_dream_log(self):
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.tick()
        assert hasattr(result, "dream_log")
        d = result.to_dict()
        assert "dream_log" in d

    def test_engine_run_tracks_dreams(self):
        engine = Mars100Engine(seed=42, total_years=10)
        sim = engine.run()
        d = sim.to_dict()
        assert "total_dreams" in d["summary"]
        assert "total_expr_adoptions" in d["summary"]
        assert d["summary"]["total_dreams"] >= 0

    def test_expr_generation_monotonic(self):
        """expr_generation should never decrease for any colonist."""
        engine = Mars100Engine(seed=42, total_years=20)
        prev_gen: dict[str, int] = {}
        for _ in range(20):
            if not engine._active_colonists():
                break
            result = engine.tick()
            for snap in result.colonist_snapshots:
                cid = snap["id"]
                gen = snap.get("expr_generation", 0)
                if cid in prev_gen:
                    assert gen >= prev_gen[cid], f"{cid} generation decreased"
                prev_gen[cid] = gen

    def test_all_exprs_remain_valid_lispy_after_100_years(self):
        """After a full sim, every colonist's decision_expr must be parseable."""
        engine = Mars100Engine(seed=42, total_years=100)
        sim = engine.run()
        for c in sim.final_colonists:
            expr = c["decision_expr"]
            try:
                parsed = parse_all(expr)
                assert len(parsed) > 0
            except LispyError:
                pytest.fail(f"colonist {c['id']} has invalid expr: {expr}")

    def test_deterministic_with_same_seed(self):
        """Same seed produces same dream log."""
        e1 = Mars100Engine(seed=99, total_years=5)
        r1 = e1.run()
        e2 = Mars100Engine(seed=99, total_years=5)
        r2 = e2.run()
        for y1, y2 in zip(r1.years, r2.years):
            assert y1.dream_log == y2.dream_log


# ---- sample_bindings ----

class TestSampleBindings:
    def test_perturbs_numeric_values(self):
        rng = random.Random(42)
        base = {"x": 0.5, "name": "test"}
        result = _sample_bindings(base, rng)
        assert result["name"] == "test"  # non-numeric unchanged
        assert isinstance(result["x"], float)

    def test_stays_in_0_1(self):
        rng = random.Random(42)
        base = {"x": 0.0, "y": 1.0}
        for _ in range(50):
            result = _sample_bindings(base, rng)
            assert 0.0 <= result["x"] <= 1.0
            assert 0.0 <= result["y"] <= 1.0
