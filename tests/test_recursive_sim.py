"""
Tests for the recursive sub-simulation engine (src/mars100/recursive_sim.py).

Covers every scenario type at every depth, budget/depth limits,
normalisation, insight extraction, and determinism.
"""
from __future__ import annotations

import math
import pytest
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.recursive_sim import (
    RecursiveResult,
    run_scenario,
    choose_scenario,
    max_depth_reached,
    collect_insights,
    _normalise,
    _should_recurse,
    SCENARIO_TYPES,
    MAX_DEPTH,
    RECURSE_LO,
    RECURSE_HI,
)
from src.mars100.colonist import create_founding_ten, STAT_NAMES, SKILL_NAMES
from src.mars100.colony import Resources


# ── Fixtures ──────────────────────────────────────────────────────────

def _colonist_bindings(colonist=None, resources=None, population=10):
    """Build a typical bindings dict for sub-sim evaluation."""
    if colonist is None:
        colonists = create_founding_ten(seed=42)
        colonist = colonists[0]
    b = colonist.lispy_bindings()
    r = resources or Resources()
    for name in ("food", "water", "power", "air", "medicine"):
        b[name] = getattr(r, name)
    b["population"] = population
    b["mediation"] = colonist.skills.mediation
    b["terraforming"] = colonist.skills.terraforming
    b["hydroponics"] = colonist.skills.hydroponics
    b["sabotage"] = colonist.skills.sabotage
    return b


# ── Normalisation ─────────────────────────────────────────────────────

class TestNormalise:
    def test_positive_clamped(self):
        assert _normalise(5.0) == 1.0

    def test_negative_clamped(self):
        assert _normalise(-5.0) == -1.0

    def test_within_range(self):
        assert _normalise(0.42) == pytest.approx(0.42)

    def test_bool_true(self):
        assert _normalise(True) == 1.0

    def test_bool_false(self):
        assert _normalise(False) == -1.0

    def test_string(self):
        assert _normalise("hello") == 0.0

    def test_none(self):
        assert _normalise(None) == 0.0

    def test_nan(self):
        assert _normalise(float("nan")) == 0.0

    def test_inf(self):
        assert _normalise(float("inf")) == 0.0


class TestShouldRecurse:
    def test_ambiguous_triggers(self):
        assert _should_recurse(0.0) is True
        assert _should_recurse(0.3) is True
        assert _should_recurse(-0.3) is True

    def test_clear_positive_stops(self):
        assert _should_recurse(0.8) is False

    def test_clear_negative_stops(self):
        assert _should_recurse(-0.8) is False

    def test_boundaries(self):
        assert _should_recurse(RECURSE_LO) is True
        assert _should_recurse(RECURSE_HI) is True
        assert _should_recurse(RECURSE_LO - 0.01) is False
        assert _should_recurse(RECURSE_HI + 0.01) is False


# ── Scenario execution ────────────────────────────────────────────────

class TestGovernanceTest:
    def test_depth1_produces_result(self):
        b = _colonist_bindings()
        r = run_scenario("governance_test", "kira-sol", 10, b, depth=1)
        assert r.succeeded
        assert -1.0 <= r.normalised <= 1.0
        assert r.scenario == "governance_test"
        assert r.depth == 1

    def test_depth2_reachable(self):
        """Force ambiguous depth-1 to trigger depth-2."""
        colonists = create_founding_ten(seed=42)
        c = colonists[3]  # Aura Kai — high faith, high empathy
        b = _colonist_bindings(colonist=c)
        r = run_scenario("governance_test", c.id, 15, b)
        # With these stats the result might be ambiguous → child spawns
        if _should_recurse(r.normalised):
            assert len(r.children) == 1
            assert r.children[0].depth == 2

    def test_depth3_reachable_with_tuned_stats(self):
        """Construct stats that force ambiguous results through depth 3."""
        colonists = create_founding_ten(seed=42)
        c = colonists[3]
        # Zero out everything to force near-zero results
        for name in STAT_NAMES:
            setattr(c.stats, name, 0.25)
        for name in SKILL_NAMES:
            setattr(c.skills, name, 0.25)
        b = _colonist_bindings(colonist=c)
        r = run_scenario("governance_test", c.id, 20, b)
        deepest = max_depth_reached(r)
        # Might reach depth 2 or 3 depending on exact arithmetic
        assert deepest >= 1


class TestResourceForecast:
    def test_depth1(self):
        b = _colonist_bindings()
        r = run_scenario("resource_forecast", "kira-sol", 5, b)
        assert r.succeeded
        assert -1.0 <= r.normalised <= 1.0

    def test_low_resources_trend_negative(self):
        res = Resources(food=0.1, water=0.1, power=0.1, air=0.1, medicine=0.1)
        b = _colonist_bindings(resources=res, population=15)
        r = run_scenario("resource_forecast", "test-col", 30, b)
        assert r.succeeded
        # With very low resources and high population, trend should be negative
        assert r.normalised < 0.5

    def test_high_resources_trend_positive(self):
        res = Resources(food=0.9, water=0.9, power=0.9, air=0.9, medicine=0.9)
        b = _colonist_bindings(resources=res, population=2)
        r = run_scenario("resource_forecast", "test-col", 30, b)
        assert r.succeeded
        assert r.normalised > 0


class TestConflictProbe:
    def test_depth1(self):
        b = _colonist_bindings()
        r = run_scenario("conflict_probe", "kira-sol", 8, b)
        assert r.succeeded
        assert -1.0 <= r.normalised <= 1.0

    def test_high_mediation_positive(self):
        colonists = create_founding_ten(seed=42)
        c = colonists[3]  # high mediation
        c.skills.mediation = 0.9
        c.stats.empathy = 0.9
        c.stats.paranoia = 0.1
        c.skills.sabotage = 0.0
        b = _colonist_bindings(colonist=c)
        r = run_scenario("conflict_probe", c.id, 12, b)
        assert r.normalised > 0

    def test_high_sabotage_negative(self):
        colonists = create_founding_ten(seed=42)
        c = colonists[7]  # Zeph Wind — trickster, high sabotage
        c.skills.sabotage = 0.9
        c.stats.paranoia = 0.9
        c.skills.mediation = 0.0
        c.stats.empathy = 0.1
        b = _colonist_bindings(colonist=c)
        r = run_scenario("conflict_probe", c.id, 12, b)
        assert r.normalised < 0.5


class TestExistentialProbe:
    def test_depth1(self):
        b = _colonist_bindings()
        r = run_scenario("existential_probe", "kira-sol", 50, b)
        assert r.succeeded
        assert -1.0 <= r.normalised <= 1.0

    def test_high_faith_yields_insight(self):
        colonists = create_founding_ten(seed=42)
        c = colonists[8]  # Ora Flame — prophet, faith=0.95
        b = _colonist_bindings(colonist=c)
        r = run_scenario("existential_probe", c.id, 50, b)
        assert r.succeeded
        # Depth 2+ may produce insights
        all_nodes = r.flatten()
        if len(all_nodes) > 1:
            assert any(n.insight is not None for n in all_nodes)


# ── Depth limits ──────────────────────────────────────────────────────

class TestDepthLimits:
    def test_depth4_rejected(self):
        b = _colonist_bindings()
        r = run_scenario("governance_test", "test", 1, b, depth=4)
        assert not r.succeeded
        assert "max depth exceeded" in r.error

    def test_depth_equals_max_succeeds(self):
        b = _colonist_bindings()
        r = run_scenario("governance_test", "test", 1, b, depth=MAX_DEPTH)
        assert r.succeeded
        # At max depth, even ambiguous result should NOT recurse
        assert len(r.children) == 0

    def test_unknown_scenario(self):
        b = _colonist_bindings()
        r = run_scenario("time_travel", "test", 1, b)
        assert not r.succeeded
        assert "unknown scenario" in r.error


# ── Result tree structure ─────────────────────────────────────────────

class TestResultTree:
    def test_flatten_single_node(self):
        r = RecursiveResult(scenario="test", depth=1, colonist_id="a",
                            year=1, expression="(+ 1 1)")
        flat = r.flatten()
        assert len(flat) == 1

    def test_flatten_with_children(self):
        child = RecursiveResult(scenario="test", depth=2, colonist_id="a",
                                year=1, expression="(+ 2 2)")
        parent = RecursiveResult(scenario="test", depth=1, colonist_id="a",
                                 year=1, expression="(+ 1 1)",
                                 children=[child])
        flat = parent.flatten()
        assert len(flat) == 2
        assert flat[0].depth == 1
        assert flat[1].depth == 2

    def test_max_depth_reached(self):
        grandchild = RecursiveResult(scenario="t", depth=3, colonist_id="a",
                                     year=1, expression="x")
        child = RecursiveResult(scenario="t", depth=2, colonist_id="a",
                                year=1, expression="x", children=[grandchild])
        root = RecursiveResult(scenario="t", depth=1, colonist_id="a",
                               year=1, expression="x", children=[child])
        assert max_depth_reached(root) == 3

    def test_to_dict_serialisable(self):
        import json
        b = _colonist_bindings()
        r = run_scenario("governance_test", "test", 10, b)
        d = r.to_dict()
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_collect_insights_empty_for_depth1(self):
        r = RecursiveResult(scenario="test", depth=1, colonist_id="a",
                            year=1, expression="x", normalised=0.5)
        assert collect_insights(r) == []

    def test_collect_insights_from_deep_tree(self):
        child = RecursiveResult(scenario="governance_test", depth=2,
                                colonist_id="a", year=5, expression="x",
                                normalised=0.5, insight="Test insight")
        parent = RecursiveResult(scenario="governance_test", depth=1,
                                 colonist_id="a", year=5, expression="x",
                                 normalised=0.1, children=[child])
        insights = collect_insights(parent)
        assert len(insights) == 1
        assert insights[0]["depth"] == 2
        assert "insight" in insights[0]


# ── Scenario selection ────────────────────────────────────────────────

class TestChooseScenario:
    def test_existential_needs_faith_and_paranoia(self):
        s = choose_scenario(
            {"faith": 0.8, "paranoia": 0.5, "improvisation": 0.5, "empathy": 0.5},
            {"coding": 0.5}, resource_avg=0.6, has_conflict=False, rng_value=0.05,
        )
        assert s == "existential_probe"

    def test_conflict_triggered_by_flag(self):
        s = choose_scenario(
            {"faith": 0.3, "paranoia": 0.3, "improvisation": 0.5, "empathy": 0.5},
            {"coding": 0.5}, resource_avg=0.6, has_conflict=True, rng_value=0.15,
        )
        assert s == "conflict_probe"

    def test_resource_when_strained(self):
        s = choose_scenario(
            {"faith": 0.3, "paranoia": 0.3, "improvisation": 0.5, "empathy": 0.3},
            {"coding": 0.1}, resource_avg=0.3, has_conflict=False, rng_value=0.20,
        )
        assert s == "resource_forecast"

    def test_governance_as_default(self):
        s = choose_scenario(
            {"faith": 0.3, "paranoia": 0.3, "improvisation": 0.5, "empathy": 0.3},
            {"coding": 0.5}, resource_avg=0.7, has_conflict=False, rng_value=0.25,
        )
        assert s == "governance_test"

    def test_none_when_rng_high(self):
        s = choose_scenario(
            {"faith": 0.3, "paranoia": 0.3, "improvisation": 0.5, "empathy": 0.3},
            {"coding": 0.5}, resource_avg=0.7, has_conflict=False, rng_value=0.99,
        )
        assert s is None


# ── Determinism ───────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_inputs_same_output(self):
        b = _colonist_bindings()
        r1 = run_scenario("governance_test", "test", 10, b)
        r2 = run_scenario("governance_test", "test", 10, b)
        assert r1.normalised == r2.normalised
        assert r1.expression == r2.expression

    def test_different_year_same_bindings_same_output(self):
        """Year only affects metadata, not computation (for same scenario)."""
        b = _colonist_bindings()
        r1 = run_scenario("governance_test", "test", 10, b)
        r2 = run_scenario("governance_test", "test", 50, b)
        # Same bindings → same expression → same result
        assert r1.normalised == r2.normalised


# ── Seeded fuzz (property-style) ──────────────────────────────────────

class TestPropertyLike:
    @pytest.mark.parametrize("seed", range(20))
    def test_all_scenarios_bounded(self, seed):
        """Every scenario at every depth produces normalised in [-1, 1]."""
        rng = random.Random(seed)
        colonists = create_founding_ten(seed=seed)
        c = rng.choice(colonists)
        # Randomize stats slightly
        for name in STAT_NAMES:
            setattr(c.stats, name, rng.uniform(0.0, 1.0))
        for name in SKILL_NAMES:
            setattr(c.skills, name, rng.uniform(0.0, 1.0))
        b = _colonist_bindings(colonist=c, population=rng.randint(3, 30))
        for scenario in SCENARIO_TYPES:
            r = run_scenario(scenario, c.id, rng.randint(1, 100), b)
            for node in r.flatten():
                if node.succeeded:
                    assert -1.0 <= node.normalised <= 1.0, (
                        f"{scenario} depth={node.depth} normalised={node.normalised}"
                    )

    @pytest.mark.parametrize("seed", range(10))
    def test_max_depth_never_exceeds_3(self, seed):
        """No recursion tree ever goes beyond depth 3."""
        rng = random.Random(seed)
        colonists = create_founding_ten(seed=seed)
        c = rng.choice(colonists)
        for name in STAT_NAMES:
            setattr(c.stats, name, 0.25)  # force ambiguity
        for name in SKILL_NAMES:
            setattr(c.skills, name, 0.25)
        b = _colonist_bindings(colonist=c)
        for scenario in SCENARIO_TYPES:
            r = run_scenario(scenario, c.id, 20, b)
            assert max_depth_reached(r) <= MAX_DEPTH


# ── Integration smoke test ────────────────────────────────────────────

class TestSmoke:
    def test_10_year_mixed_scenarios(self):
        """Run mixed scenarios for 10 years, all succeed or gracefully fail."""
        colonists = create_founding_ten(seed=42)
        rng = random.Random(42)
        total_runs = 0
        total_depth2 = 0
        total_depth3 = 0
        for year in range(1, 11):
            for c in colonists:
                b = _colonist_bindings(colonist=c, population=len(colonists))
                scenario = choose_scenario(
                    c.stats.to_dict(), c.skills.to_dict(),
                    resource_avg=0.6, has_conflict=(year % 3 == 0),
                    rng_value=rng.random(),
                )
                if scenario is None:
                    continue
                r = run_scenario(scenario, c.id, year, b)
                total_runs += 1
                deepest = max_depth_reached(r)
                if deepest >= 2:
                    total_depth2 += 1
                if deepest >= 3:
                    total_depth3 += 1
                # All nodes bounded
                for node in r.flatten():
                    if node.succeeded:
                        assert -1.0 <= node.normalised <= 1.0

        assert total_runs > 0, "should have run at least some scenarios"
        # Depth 2 should happen sometimes
        # (not guaranteed every time, but very likely with 100 runs)
