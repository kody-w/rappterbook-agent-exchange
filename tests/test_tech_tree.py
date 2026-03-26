"""
tests/test_tech_tree.py — Dedicated tests for tech_tree.py.

Second file shipped. Community proved it can ship mars_env tests.
Now it ships tech_tree tests. Momentum > governance.

Covers:
  - TECH_CATALOG data integrity (8 techs, 5 branches, unique names)
  - STRATEGY_RESEARCH_WEIGHT correctness
  - TechUnlock dataclass
  - ResearchEngine initialization and state
  - Research point generation (strategy weights, population scaling)
  - Tech selection per strategy (conservative=cheapest, aggressive=most expensive)
  - Unlock lifecycle (points accumulate, tech deducted, added to unlocked)
  - No double-unlock (each tech unlockable once)
  - Low population blocks research
  - All 8 techs eventually unlock with enough points
  - get_modifier aggregation
  - has_tech lookup
  - snapshot serialization
  - Determinism (same seed = same unlock order)
  - Strategy divergence (different strategies pick different first techs)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tech_tree import (
    STRATEGY_RESEARCH_WEIGHT,
    TECH_CATALOG,
    ResearchEngine,
    TechUnlock,
)


# ──────────────────────────────────────────────
# TECH_CATALOG data integrity
# ──────────────────────────────────────────────

class TestTechCatalog:
    """The static tech catalog must be well-formed."""

    def test_eight_techs(self) -> None:
        assert len(TECH_CATALOG) == 8

    def test_unique_names(self) -> None:
        names = [t["name"] for t in TECH_CATALOG]
        assert len(names) == len(set(names))

    def test_unique_effects(self) -> None:
        effects = [t["effect"] for t in TECH_CATALOG]
        assert len(effects) == len(set(effects))

    def test_five_branches(self) -> None:
        branches = {t["branch"] for t in TECH_CATALOG}
        assert branches == {"power", "food", "defense", "water", "construction"}

    def test_all_costs_positive(self) -> None:
        for tech in TECH_CATALOG:
            assert tech["cost"] > 0, f"{tech['name']} has non-positive cost"

    def test_all_values_positive(self) -> None:
        for tech in TECH_CATALOG:
            assert tech["value"] > 0, f"{tech['name']} has non-positive value"

    def test_all_have_description(self) -> None:
        for tech in TECH_CATALOG:
            assert len(tech["description"]) > 5, f"{tech['name']} missing description"

    def test_required_keys(self) -> None:
        required = {"name", "branch", "cost", "effect", "value", "description"}
        for tech in TECH_CATALOG:
            assert required.issubset(tech.keys()), f"{tech.get('name', '?')} missing keys"

    def test_costs_span_reasonable_range(self) -> None:
        """Costs should range from affordable to expensive."""
        costs = sorted(t["cost"] for t in TECH_CATALOG)
        assert costs[0] >= 100   # cheapest isn't trivial
        assert costs[-1] <= 5000  # most expensive isn't absurd
        assert costs[-1] > costs[0] * 2  # meaningful spread


# ──────────────────────────────────────────────
# STRATEGY_RESEARCH_WEIGHT
# ──────────────────────────────────────────────

class TestStrategyWeights:
    """Strategy research weights must be consistent."""

    def test_three_strategies(self) -> None:
        assert set(STRATEGY_RESEARCH_WEIGHT.keys()) == {
            "conservative", "balanced", "aggressive",
        }

    def test_balanced_is_one(self) -> None:
        assert STRATEGY_RESEARCH_WEIGHT["balanced"] == 1.0

    def test_conservative_lower(self) -> None:
        assert STRATEGY_RESEARCH_WEIGHT["conservative"] < STRATEGY_RESEARCH_WEIGHT["balanced"]

    def test_aggressive_higher(self) -> None:
        assert STRATEGY_RESEARCH_WEIGHT["aggressive"] > STRATEGY_RESEARCH_WEIGHT["balanced"]

    def test_all_positive(self) -> None:
        for strategy, weight in STRATEGY_RESEARCH_WEIGHT.items():
            assert weight > 0, f"{strategy} weight is non-positive"


# ──────────────────────────────────────────────
# TechUnlock dataclass
# ──────────────────────────────────────────────

class TestTechUnlock:
    """TechUnlock records are simple data containers."""

    def test_fields(self) -> None:
        t = TechUnlock(name="Test", branch="power", sol=42, effect="boost", value=1.5)
        assert t.name == "Test"
        assert t.branch == "power"
        assert t.sol == 42
        assert t.effect == "boost"
        assert t.value == 1.5

    def test_equality(self) -> None:
        a = TechUnlock("A", "power", 1, "x", 0.5)
        b = TechUnlock("A", "power", 1, "x", 0.5)
        assert a == b

    def test_inequality(self) -> None:
        a = TechUnlock("A", "power", 1, "x", 0.5)
        b = TechUnlock("B", "power", 1, "x", 0.5)
        assert a != b


# ──────────────────────────────────────────────
# ResearchEngine — initialization
# ──────────────────────────────────────────────

class TestResearchEngineInit:
    """Fresh engine state."""

    def test_starts_empty(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        assert eng.research_points == 0.0
        assert eng.unlocked == []
        assert eng.unlocked_names == set()

    def test_all_techs_available(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        assert len(eng.available_techs()) == 8

    def test_unknown_strategy_uses_default_weight(self) -> None:
        eng = ResearchEngine(strategy="unknown", rng=random.Random(1))
        pts = eng.generate_points(100, 1.0)
        expected = 100 * 1.0 * 1.0 * 0.1  # default weight = 1.0
        assert abs(pts - expected) < 0.001


# ──────────────────────────────────────────────
# Research point generation
# ──────────────────────────────────────────────

class TestPointGeneration:
    """Points scale with population, morale, and strategy."""

    def test_formula_balanced(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        pts = eng.generate_points(100, 0.8)
        expected = 100 * 0.8 * 1.0 * 0.1
        assert abs(pts - expected) < 0.001

    def test_formula_conservative(self) -> None:
        eng = ResearchEngine(strategy="conservative", rng=random.Random(1))
        pts = eng.generate_points(100, 1.0)
        expected = 100 * 1.0 * 0.8 * 0.1
        assert abs(pts - expected) < 0.001

    def test_formula_aggressive(self) -> None:
        eng = ResearchEngine(strategy="aggressive", rng=random.Random(1))
        pts = eng.generate_points(100, 1.0)
        expected = 100 * 1.0 * 1.3 * 0.1
        assert abs(pts - expected) < 0.001

    def test_zero_population_zero_points(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        assert eng.generate_points(0, 1.0) == 0.0

    def test_zero_morale_zero_points(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        assert eng.generate_points(100, 0.0) == 0.0

    def test_points_scale_linearly_with_pop(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        pts_50 = eng.generate_points(50, 1.0)
        pts_100 = eng.generate_points(100, 1.0)
        assert abs(pts_100 - 2 * pts_50) < 0.001

    def test_aggressive_generates_more_than_conservative(self) -> None:
        agg = ResearchEngine(strategy="aggressive", rng=random.Random(1))
        con = ResearchEngine(strategy="conservative", rng=random.Random(1))
        assert agg.generate_points(100, 1.0) > con.generate_points(100, 1.0)


# ──────────────────────────────────────────────
# Tech selection strategy
# ──────────────────────────────────────────────

class TestTechSelection:
    """Each strategy picks techs differently."""

    def test_conservative_picks_cheapest(self) -> None:
        eng = ResearchEngine(strategy="conservative", rng=random.Random(1))
        eng.research_points = 2000  # enough for several techs
        tech = eng._select_tech(eng.available_techs())
        assert tech is not None
        cheapest_cost = min(t["cost"] for t in TECH_CATALOG)
        assert tech["cost"] == cheapest_cost

    def test_aggressive_picks_most_expensive_affordable(self) -> None:
        eng = ResearchEngine(strategy="aggressive", rng=random.Random(1))
        eng.research_points = 2000
        tech = eng._select_tech(eng.available_techs())
        assert tech is not None
        affordable = [t for t in TECH_CATALOG if t["cost"] <= 2000]
        most_expensive = max(t["cost"] for t in affordable)
        assert tech["cost"] == most_expensive

    def test_no_affordable_returns_none(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.research_points = 1  # can't afford anything
        assert eng._select_tech(eng.available_techs()) is None

    def test_empty_catalog_returns_none(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.research_points = 10000
        assert eng._select_tech([]) is None

    def test_balanced_selects_from_affordable(self) -> None:
        """Balanced strategy picks from the affordable pool (weighted random)."""
        eng = ResearchEngine(strategy="balanced", rng=random.Random(42))
        eng.research_points = 700
        tech = eng._select_tech(eng.available_techs())
        assert tech is not None
        assert tech["cost"] <= 700


# ──────────────────────────────────────────────
# Unlock lifecycle
# ──────────────────────────────────────────────

class TestUnlockLifecycle:
    """Points accumulate, techs unlock, state updates correctly."""

    def test_tick_accumulates_points(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.tick(100, 1.0, sol=1)
        assert eng.research_points > 0 or len(eng.unlocked) > 0

    def test_low_population_blocks_research(self) -> None:
        """Population < 5 produces no research."""
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        result = eng.tick(4, 1.0, sol=1)
        assert result is None
        assert eng.research_points == 0.0

    def test_unlock_deducts_cost(self) -> None:
        """When a tech is unlocked, its cost is deducted from points."""
        eng = ResearchEngine(strategy="conservative", rng=random.Random(1))
        eng.research_points = 600  # just above cheapest (500)
        unlock = eng.tick(100, 1.0, sol=10)
        if unlock is not None:
            # Points should have been deducted
            assert eng.research_points < 600

    def test_unlock_records_sol(self) -> None:
        eng = ResearchEngine(strategy="conservative", rng=random.Random(1))
        eng.research_points = 1000
        unlock = eng.tick(100, 1.0, sol=42)
        if unlock is not None:
            assert unlock.sol == 42

    def test_no_double_unlock(self) -> None:
        """Each tech can only be unlocked once."""
        eng = ResearchEngine(strategy="conservative", rng=random.Random(1))
        # Give enough points to unlock everything
        eng.research_points = 50000
        unlocked_names = []
        for sol in range(1, 100):
            unlock = eng.tick(200, 1.0, sol=sol)
            if unlock is not None:
                assert unlock.name not in unlocked_names, f"Double unlock: {unlock.name}"
                unlocked_names.append(unlock.name)
        # At most 8 unique techs
        assert len(unlocked_names) <= 8

    def test_all_eight_unlock_eventually(self) -> None:
        """With enough points, all 8 techs unlock."""
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.research_points = 100000  # massive surplus
        for sol in range(1, 100):
            eng.tick(500, 1.0, sol=sol)
        assert len(eng.unlocked) == 8
        assert eng.available_techs() == []

    def test_points_never_negative(self) -> None:
        """Research points should never go negative during normal operation."""
        eng = ResearchEngine(strategy="aggressive", rng=random.Random(42))
        for sol in range(1, 500):
            eng.tick(100, 0.8, sol=sol)
            assert eng.research_points >= 0, f"Negative points at sol {sol}"


# ──────────────────────────────────────────────
# Modifiers and queries
# ──────────────────────────────────────────────

class TestModifiersAndQueries:
    """get_modifier, has_tech, unlocked_names, available_techs."""

    def test_get_modifier_empty(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        assert eng.get_modifier("solar_boost") == 0.0

    def test_get_modifier_after_unlock(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.unlocked.append(
            TechUnlock("Test", "power", 1, "solar_boost", 0.25)
        )
        assert eng.get_modifier("solar_boost") == 0.25

    def test_get_modifier_sums_multiple(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.unlocked.append(TechUnlock("A", "p", 1, "boost", 0.5))
        eng.unlocked.append(TechUnlock("B", "p", 2, "boost", 0.3))
        assert abs(eng.get_modifier("boost") - 0.8) < 0.001

    def test_has_tech_false(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        assert eng.has_tech("Advanced Solar Cells") is False

    def test_has_tech_true(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.unlocked.append(
            TechUnlock("Advanced Solar Cells", "power", 1, "solar_boost", 0.25)
        )
        assert eng.has_tech("Advanced Solar Cells") is True

    def test_available_techs_shrinks(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        assert len(eng.available_techs()) == 8
        eng.unlocked.append(
            TechUnlock("Advanced Solar Cells", "power", 1, "solar_boost", 0.25)
        )
        assert len(eng.available_techs()) == 7

    def test_unlocked_names_set(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.unlocked.append(TechUnlock("A", "p", 1, "x", 0.1))
        eng.unlocked.append(TechUnlock("B", "q", 2, "y", 0.2))
        assert eng.unlocked_names == {"A", "B"}


# ──────────────────────────────────────────────
# Snapshot serialization
# ──────────────────────────────────────────────

class TestSnapshot:
    """Snapshot must be JSON-serializable and well-formed."""

    def test_json_serializable(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.research_points = 123.456
        eng.unlocked.append(TechUnlock("Test", "power", 10, "boost", 0.5))
        snap = eng.snapshot()
        # Must not raise
        serialized = json.dumps(snap)
        parsed = json.loads(serialized)
        assert parsed == snap

    def test_snapshot_keys(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        snap = eng.snapshot()
        assert set(snap.keys()) == {"research_points", "unlocked_count", "unlocked"}

    def test_snapshot_unlocked_count(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.unlocked.append(TechUnlock("A", "p", 1, "x", 0.1))
        eng.unlocked.append(TechUnlock("B", "q", 2, "y", 0.2))
        snap = eng.snapshot()
        assert snap["unlocked_count"] == 2
        assert len(snap["unlocked"]) == 2

    def test_snapshot_unlock_fields(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.unlocked.append(TechUnlock("Test", "food", 42, "boost", 0.3))
        snap = eng.snapshot()
        item = snap["unlocked"][0]
        assert item == {"name": "Test", "branch": "food", "sol": 42}

    def test_points_rounded(self) -> None:
        eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
        eng.research_points = 3.14159
        snap = eng.snapshot()
        assert snap["research_points"] == 3.1


# ──────────────────────────────────────────────
# Determinism and strategy divergence
# ──────────────────────────────────────────────

class TestDeterminismAndDivergence:
    """Same seed = same results. Different strategies = different paths."""

    def test_deterministic_same_seed(self) -> None:
        """Identical engines produce identical unlock sequences."""
        def run_engine(seed: int) -> list[str]:
            eng = ResearchEngine(strategy="balanced", rng=random.Random(seed))
            for sol in range(1, 300):
                eng.tick(100, 0.8, sol=sol)
            return [t.name for t in eng.unlocked]

        assert run_engine(42) == run_engine(42)

    def test_different_seeds_may_diverge(self) -> None:
        """Different seeds can produce different unlock orders.

        With enough points and seeds, balanced strategy's weighted random
        should eventually pick different orders.
        """
        def run_engine(seed: int) -> list[str]:
            eng = ResearchEngine(strategy="balanced", rng=random.Random(seed))
            eng.research_points = 50000  # enough to unlock all 8
            for sol in range(1, 50):
                eng.tick(500, 1.0, sol=sol)
            return [t.name for t in eng.unlocked]

        # With enough seeds and points, balanced weighted-random produces
        # at least two different orderings
        results = [run_engine(s) for s in range(20)]
        assert len(set(tuple(r) for r in results)) > 1

    def test_strategies_pick_different_first_tech(self) -> None:
        """Conservative and aggressive pick different first techs."""
        def first_unlock(strategy: str) -> str:
            eng = ResearchEngine(strategy=strategy, rng=random.Random(42))
            eng.research_points = 2000  # enough for many techs
            unlock = eng.tick(100, 1.0, sol=1)
            return unlock.name if unlock else ""

        con = first_unlock("conservative")
        agg = first_unlock("aggressive")
        assert con != agg
        # Conservative should pick cheapest
        cheapest = min(TECH_CATALOG, key=lambda t: t["cost"])
        assert con == cheapest["name"]

    def test_all_strategies_eventually_unlock_all(self) -> None:
        """Every strategy can unlock all 8 techs given enough time."""
        for strategy in ["conservative", "balanced", "aggressive"]:
            eng = ResearchEngine(strategy=strategy, rng=random.Random(42))
            eng.research_points = 100000
            for sol in range(1, 100):
                eng.tick(500, 1.0, sol=sol)
            assert len(eng.unlocked) == 8, (
                f"{strategy} only unlocked {len(eng.unlocked)}/8"
            )
