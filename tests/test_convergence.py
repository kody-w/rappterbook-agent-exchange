"""Tests for Mars-100 value convergence tracking."""
from __future__ import annotations

import random
import pytest
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, STAT_NAMES, SKILL_NAMES
from src.mars100.convergence import (
    ConvergenceSnapshot,
    ConvergenceTracker,
    classify_trend,
    compute_convergence,
    _std_dev,
)


def _make_colonist(cid: str, stats: dict[str, float] | None = None,
                   generation: int = 0) -> Colonist:
    """Helper to create a colonist with specified stats."""
    s = {name: 0.5 for name in STAT_NAMES}
    if stats:
        s.update(stats)
    return Colonist(
        id=cid, name=f"Col-{cid}", element="fire", archetype="pioneer",
        stats=ColonistStats.from_dict(s),
        skills=ColonistSkills.from_dict({name: 0.3 for name in SKILL_NAMES}),
        decision_expr="(+ resolve empathy)",
        generation=generation,
    )


class TestStdDev:
    def test_empty(self):
        assert _std_dev([]) == 0.0

    def test_single(self):
        assert _std_dev([5.0]) == 0.0

    def test_identical(self):
        assert _std_dev([3.0, 3.0, 3.0]) == 0.0

    def test_known_value(self):
        result = _std_dev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(result - 2.0) < 0.01

    def test_nonnegative(self):
        rng = random.Random(42)
        for _ in range(50):
            vals = [rng.random() for _ in range(10)]
            assert _std_dev(vals) >= 0.0


class TestComputeConvergence:
    def test_identical_colonists(self):
        colonists = [_make_colonist(f"c{i}") for i in range(5)]
        snap = compute_convergence(colonists, {"c0", "c1"}, year=10)
        assert snap.year == 10
        for name in STAT_NAMES:
            assert snap.population_dispersion[name] == 0.0
        assert snap.aggregate_dispersion == 0.0

    def test_diverse_colonists(self):
        colonists = [
            _make_colonist("a", {"resolve": 0.1, "empathy": 0.9}),
            _make_colonist("b", {"resolve": 0.9, "empathy": 0.1}),
        ]
        snap = compute_convergence(colonists, {"a", "b"}, year=5)
        assert snap.population_dispersion["resolve"] > 0.3
        assert snap.population_dispersion["empathy"] > 0.3
        assert snap.aggregate_dispersion > 0.0

    def test_dead_excluded(self):
        colonists = [
            _make_colonist("a", {"resolve": 0.1}),
            _make_colonist("b", {"resolve": 0.9}),
            _make_colonist("c", {"resolve": 0.5}),
        ]
        colonists[1].die(3, "test")
        snap = compute_convergence(colonists, {"a", "b", "c"}, year=5)
        # Only a (0.1) and c (0.5) are active
        assert snap.population_dispersion["resolve"] < 0.4

    def test_founder_vs_population(self):
        founders = [_make_colonist(f"f{i}", {"resolve": 0.5 + i * 0.1}) for i in range(3)]
        children = [_make_colonist(f"c{i}", {"resolve": 0.2}, generation=1) for i in range(3)]
        all_colonists = founders + children
        snap = compute_convergence(all_colonists, {"f0", "f1", "f2"}, year=20)
        # Founder and population dispersion can differ
        assert isinstance(snap.founder_dispersion["resolve"], float)
        assert isinstance(snap.population_dispersion["resolve"], float)


class TestClassifyTrend:
    def test_stable_with_few_snapshots(self):
        snaps = [ConvergenceSnapshot(i, {}, {}, 0.1, 0.1) for i in range(3)]
        assert classify_trend(snaps) == "stable"

    def test_converging(self):
        snaps = [ConvergenceSnapshot(i, {}, {}, 0.1 - i * 0.005, 0.1)
                 for i in range(20)]
        assert classify_trend(snaps) == "converging"

    def test_diverging(self):
        snaps = [ConvergenceSnapshot(i, {}, {}, 0.1 + i * 0.005, 0.1)
                 for i in range(20)]
        assert classify_trend(snaps) == "diverging"

    def test_stable_flat(self):
        snaps = [ConvergenceSnapshot(i, {}, {}, 0.1, 0.1) for i in range(20)]
        assert classify_trend(snaps) == "stable"


class TestConvergenceTracker:
    def test_record_and_trend(self):
        tracker = ConvergenceTracker(founder_ids={"c0", "c1", "c2"})
        colonists = [_make_colonist(f"c{i}") for i in range(5)]
        for year in range(1, 21):
            snap = tracker.record(colonists, year)
            assert snap.year == year
        assert tracker.trend() in ("converging", "diverging", "stable")

    def test_summary(self):
        tracker = ConvergenceTracker(founder_ids={"a"})
        colonists = [_make_colonist("a"), _make_colonist("b")]
        for year in range(1, 11):
            tracker.record(colonists, year)
        summary = tracker.summary()
        assert "trend" in summary
        assert "initial" in summary
        assert "final" in summary
        assert summary["snapshots"] == 10

    def test_empty_summary(self):
        tracker = ConvergenceTracker()
        summary = tracker.summary()
        assert summary["trend"] == "no_data"

    def test_to_curve(self):
        tracker = ConvergenceTracker(founder_ids=set())
        colonists = [_make_colonist(f"c{i}") for i in range(3)]
        for year in range(1, 6):
            tracker.record(colonists, year)
        curve = tracker.to_curve()
        assert len(curve) == 5
        assert all("year" in c and "aggregate" in c for c in curve)


class TestConvergenceInSimulation:
    """Integration test: convergence should be tracked end-to-end."""

    def test_engine_tracks_convergence(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=15)
        result = engine.run()
        assert result.convergence_summary is not None
        assert "trend" in result.convergence_summary
        assert result.convergence_curve is not None
        assert len(result.convergence_curve) == len(result.years)
        for yr in result.years:
            assert yr.convergence is not None
            assert "aggregate" in yr.convergence
