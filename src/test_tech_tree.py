"""Co-located tests for the tech_tree module.

Run: python -m pytest src/test_tech_tree.py -v
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tech_tree import (
    ResearchState, TECH_CATALOG, TECH_EFFECTS,
    _available_techs, _DEFAULT_EFFECTS,
)


def test_research_generates_points() -> None:
    """Population generates positive research points."""
    rs = ResearchState(strategy="balanced", seed=42)
    rs.tick(100, 0.8, 1)
    assert rs._total_points > 0


def test_tech_unlocks_after_enough_points() -> None:
    """A tech unlocks once enough points accumulate."""
    rs = ResearchState(strategy="conservative", seed=42)
    for sol in range(1, 500):
        name = rs.tick(200, 0.9, sol)
        if name is not None:
            assert len(rs.unlocked) >= 1
            return
    assert False, "No tech unlocked in 500 sols with 200 pop"


def test_prerequisites_enforced() -> None:
    """Tier-2 techs only become available after tier-1 prereqs."""
    tier2_with_reqs = [t for t in TECH_CATALOG if t["requires"]]
    assert len(tier2_with_reqs) > 0
    for t in tier2_with_reqs:
        available = _available_techs(set())
        ids = {a["id"] for a in available}
        assert t["id"] not in ids, f"{t['id']} available without prereqs"
        available_with = _available_techs(set(t["requires"]))
        ids_with = {a["id"] for a in available_with}
        assert t["id"] in ids_with, f"{t['id']} not available even with prereqs"


def test_strategy_conservative_cheapest_first() -> None:
    """Conservative strategy picks the cheapest tech first."""
    rs = ResearchState(strategy="conservative", seed=42)
    cheapest = min(
        (t for t in TECH_CATALOG if not t["requires"]),
        key=lambda t: t["cost"],
    )
    assert rs._current == cheapest["id"]


def test_strategy_aggressive_most_expensive_first() -> None:
    """Aggressive strategy picks the most expensive tech first."""
    rs = ResearchState(strategy="aggressive", seed=42)
    most_expensive = max(
        (t for t in TECH_CATALOG if not t["requires"]),
        key=lambda t: t["cost"],
    )
    assert rs._current == most_expensive["id"]


def test_effects_cumulative() -> None:
    """Effects from multiple techs accumulate correctly."""
    rs = ResearchState(strategy="balanced", seed=42)
    base = rs.merged_effects()
    assert base["food_production_mult"] == 1.0

    rs.unlocked.add("greenhouse_biotech_1")
    fx = rs.merged_effects()
    assert fx["food_production_mult"] > 1.0

    rs.unlocked.add("crop_diversity_1")
    fx2 = rs.merged_effects()
    assert fx2["food_production_mult"] > fx["food_production_mult"]


def test_death_rate_mult_is_multiplicative() -> None:
    """death_rate_mult compounds multiplicatively, not additively."""
    rs = ResearchState(strategy="balanced", seed=42)
    rs.unlocked.add("rad_shielding_1")
    rs.unlocked.add("medical_ai_1")
    fx = rs.merged_effects()
    expected = 0.95 * 0.90
    assert abs(fx["death_rate_mult"] - expected) < 0.001


def test_snapshot_serializable() -> None:
    """Snapshot can be JSON-serialized."""
    rs = ResearchState(strategy="balanced", seed=42)
    for sol in range(1, 200):
        rs.tick(100, 0.8, sol)
    snap = rs.snapshot()
    json.dumps(snap)  # should not raise
    assert "unlocked" in snap
    assert "effects" in snap


def test_zero_pop_no_points() -> None:
    """No research with population < 5."""
    rs = ResearchState(strategy="balanced", seed=42)
    rs.tick(0, 0.8, 1)
    assert rs._total_points == 0.0
    rs.tick(4, 0.8, 2)
    assert rs._total_points == 0.0


def test_deterministic() -> None:
    """Same seed produces same result."""
    def run(seed: int) -> dict:
        rs = ResearchState(strategy="balanced", seed=seed)
        for sol in range(1, 300):
            rs.tick(100, 0.8, sol)
        return rs.snapshot()

    s1 = run(42)
    s2 = run(42)
    assert s1 == s2


def test_all_techs_eventually_unlock() -> None:
    """Given enough time, all 8 techs unlock."""
    rs = ResearchState(strategy="balanced", seed=42)
    for sol in range(1, 5000):
        rs.tick(200, 0.9, sol)
    assert len(rs.unlocked) == len(TECH_CATALOG)


def test_mature_colony_researches_faster() -> None:
    """Colonies after sol 100 generate more points per sol."""
    rs1 = ResearchState(strategy="balanced", seed=42)
    rs1.tick(100, 0.8, 50)
    early_points = rs1._total_points

    rs2 = ResearchState(strategy="balanced", seed=42)
    rs2.tick(100, 0.8, 200)
    late_points = rs2._total_points

    assert late_points > early_points
