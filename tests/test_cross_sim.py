"""
Tests for the cross-simulation bridge (cross_sim.py).

Run: python -m pytest tests/test_cross_sim.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.cross_sim import (
    CrossSimulation,
    colony_accuracy,
    colony_prediction_map,
    compute_information_gain,
    compute_surprise,
)


# ===================================================================
# Surprise
# ===================================================================

class TestSurprise:
    def test_confident_correct_low_surprise(self):
        p = {"confidence": 0.95, "outcome": True}
        assert compute_surprise(p) == 1.0 - 0.95  # ~0.05

    def test_confident_wrong_high_surprise(self):
        p = {"confidence": 0.95, "outcome": False}
        assert compute_surprise(p) == 0.95

    def test_bounded(self):
        for conf in [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]:
            for outcome in [True, False]:
                s = compute_surprise({"confidence": conf, "outcome": outcome})
                assert 0.0 <= s <= 1.0

    def test_none_outcome(self):
        assert compute_surprise({"confidence": 0.5, "outcome": None}) == 0.0


# ===================================================================
# Information gain
# ===================================================================

class TestInfoGain:
    def test_positive_for_resolved(self):
        preds = [
            {"confidence": 0.8, "outcome": True},
            {"confidence": 0.3, "outcome": False},
            {"confidence": 0.6, "outcome": True},
        ]
        ig = compute_information_gain(preds)
        assert ig > 0.0

    def test_zero_for_empty(self):
        assert compute_information_gain([]) == 0.0

    def test_ignores_unresolved(self):
        preds = [
            {"confidence": 0.5, "outcome": None},
            {"confidence": 0.7, "outcome": True},
        ]
        ig = compute_information_gain(preds)
        assert ig > 0.0  # only the resolved one counts


# ===================================================================
# Colony prediction map
# ===================================================================

class TestColonyMap:
    def test_groups_by_colony(self):
        preds = [
            {"params": {"colony": "Ares Prime"}, "outcome": True},
            {"params": {"colony": "Ares Prime"}, "outcome": False},
            {"params": {}, "outcome": True},
        ]
        m = colony_prediction_map(preds)
        assert len(m["Ares Prime"]) == 2
        assert len(m["global"]) == 1

    def test_unknown_colony_goes_global(self):
        preds = [{"params": {"colony": "Nonexistent"}, "outcome": True}]
        m = colony_prediction_map(preds)
        assert len(m["global"]) == 1

    def test_all_colony_keys_present(self):
        m = colony_prediction_map([])
        assert "global" in m
        assert "Ares Prime" in m
        assert "Olympus Station" in m
        assert "Red Frontier" in m


# ===================================================================
# Colony accuracy
# ===================================================================

class TestColonyAccuracy:
    def test_returns_stats(self):
        preds = [
            {"params": {"colony": "Ares Prime"}, "confidence": 0.8,
             "outcome": True, "brier": 0.04},
            {"params": {"colony": "Ares Prime"}, "confidence": 0.6,
             "outcome": False, "brier": 0.36},
        ]
        stats = colony_accuracy(preds)
        assert stats["Ares Prime"]["n_predictions"] == 2
        assert 0.0 <= stats["Ares Prime"]["mean_brier"] <= 1.0
        assert 0.0 <= stats["Ares Prime"]["true_rate"] <= 1.0

    def test_empty_colony(self):
        stats = colony_accuracy([])
        for colony in stats.values():
            assert colony["n_predictions"] == 0


# ===================================================================
# Pipeline smoke tests
# ===================================================================

class TestPipelineSmoke:
    def test_short_run(self):
        xsim = CrossSimulation(n_predictions=10, sols=50, env_seed=42)
        report = xsim.run(quiet=True)
        assert "_meta" in report
        assert "summary" in report
        assert "colony_results" in report
        assert "market_outcomes" in report
        assert "leaderboard" in report
        assert "calibration" in report
        assert report["summary"]["total_predictions"] == 10

    def test_full_run(self):
        xsim = CrossSimulation(n_predictions=20, sols=365, env_seed=42)
        report = xsim.run(quiet=True)
        assert report["summary"]["resolved"] > 0

    def test_deterministic(self):
        r1 = CrossSimulation(n_predictions=10, sols=50, env_seed=42).run(quiet=True)
        r2 = CrossSimulation(n_predictions=10, sols=50, env_seed=42).run(quiet=True)
        assert r1["summary"]["mean_brier"] == r2["summary"]["mean_brier"]
        assert r1["summary"]["resolved"] == r2["summary"]["resolved"]

    def test_different_seeds_differ(self):
        r1 = CrossSimulation(n_predictions=15, sols=50, env_seed=42).run(quiet=True)
        r2 = CrossSimulation(n_predictions=15, sols=50, env_seed=99).run(quiet=True)
        # Different environments should generally give different results
        outcomes1 = [p["outcome"] for p in r1["market_outcomes"]]
        outcomes2 = [p["outcome"] for p in r2["market_outcomes"]]
        # At least something should differ (not guaranteed but very likely)
        assert outcomes1 != outcomes2 or r1["summary"]["mean_brier"] != r2["summary"]["mean_brier"]


# ===================================================================
# Physical bounds
# ===================================================================

class TestPhysicalBounds:
    def test_brier_bounded(self):
        xsim = CrossSimulation(n_predictions=20, sols=50, env_seed=42)
        report = xsim.run(quiet=True)
        for p in report["market_outcomes"]:
            if p["brier"] is not None:
                assert 0.0 <= p["brier"] <= 1.0

    def test_surprise_bounded(self):
        xsim = CrossSimulation(n_predictions=20, sols=50, env_seed=42)
        report = xsim.run(quiet=True)
        for p in report["market_outcomes"]:
            assert 0.0 <= p["surprise"] <= 1.0

    def test_outcomes_binary(self):
        xsim = CrossSimulation(n_predictions=20, sols=50, env_seed=42)
        report = xsim.run(quiet=True)
        for p in report["market_outcomes"]:
            if p["outcome"] is not None:
                assert isinstance(p["outcome"], bool)
