"""
Tests for the prediction market engine (market_maker.py).

Run: python -m pytest tests/test_market_maker.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.market_maker import (
    AGENT_ARCHETYPES,
    COLONY_NAMES,
    TEMPLATES,
    Prediction,
    brier_score,
    build_calibration_curve,
    build_leaderboard,
    generate_predictions,
    log_score,
    payout_from_brier,
    resolve_predictions,
    run_market,
    run_terrarium,
    score_predictions,
)


# ===================================================================
# Brier score
# ===================================================================

class TestBrierScore:
    def test_perfect_yes(self):
        assert brier_score(1.0, True) == 0.0

    def test_perfect_no(self):
        assert brier_score(0.0, False) == 0.0

    def test_worst_yes(self):
        assert brier_score(0.0, True) == 1.0

    def test_worst_no(self):
        assert brier_score(1.0, False) == 1.0

    def test_bounded_0_1(self):
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            for o in [True, False]:
                s = brier_score(p, o)
                assert 0.0 <= s <= 1.0


# ===================================================================
# Log score
# ===================================================================

class TestLogScore:
    def test_confident_correct(self):
        s = log_score(0.95, True)
        assert s < 0  # log(0.95) is negative

    def test_confident_wrong(self):
        s = log_score(0.95, False)
        assert s < log_score(0.5, False)  # worse than coin-flip

    def test_finite_extremes(self):
        assert math.isfinite(log_score(0.0, True))
        assert math.isfinite(log_score(1.0, False))


# ===================================================================
# Payout
# ===================================================================

class TestPayout:
    def test_perfect_pays_double(self):
        assert payout_from_brier(0.05, 10.0) == 20.0

    def test_worst_pays_nothing(self):
        assert payout_from_brier(0.90, 10.0) == 0.0

    def test_nonnegative(self):
        for b in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            assert payout_from_brier(b, 5.0) >= 0.0


# ===================================================================
# Prediction generation
# ===================================================================

class TestGeneratePredictions:
    def test_correct_count(self):
        preds = generate_predictions(n=50, seed=0)
        assert len(preds) == 50

    def test_deterministic(self):
        p1 = generate_predictions(n=20, seed=7)
        p2 = generate_predictions(n=20, seed=7)
        for a, b in zip(p1, p2):
            assert a.id == b.id
            assert a.confidence == b.confidence

    def test_unique_ids(self):
        preds = generate_predictions(n=100, seed=0)
        ids = [p.id for p in preds]
        assert len(ids) == len(set(ids))

    def test_all_archetypes_present(self):
        preds = generate_predictions(n=200, seed=0)
        archetypes = {p.archetype for p in preds}
        for key in AGENT_ARCHETYPES:
            assert key in archetypes

    def test_confidence_bounded(self):
        preds = generate_predictions(n=100, seed=0)
        for p in preds:
            assert 0.01 <= p.confidence <= 0.99

    def test_stake_positive(self):
        preds = generate_predictions(n=100, seed=0)
        for p in preds:
            assert p.stake > 0


# ===================================================================
# Terrarium
# ===================================================================

class TestTerrarium:
    def test_runs(self):
        result = run_terrarium(sols=50, seed=42)
        assert "colonies" in result
        assert len(result["colonies"]) == len(COLONY_NAMES)

    def test_deterministic(self):
        r1 = run_terrarium(sols=50, seed=42)
        r2 = run_terrarium(sols=50, seed=42)
        names1 = [c["name"] for c in r1["colonies"]]
        names2 = [c["name"] for c in r2["colonies"]]
        assert names1 == names2


# ===================================================================
# Resolution
# ===================================================================

class TestResolution:
    def test_resolves_all(self):
        result = run_terrarium(sols=50, seed=42)
        preds = generate_predictions(n=30, seed=0)
        resolve_predictions(preds, [result])
        resolved = [p for p in preds if p.outcome is not None]
        assert len(resolved) > 0

    def test_outcomes_binary(self):
        result = run_terrarium(sols=50, seed=42)
        preds = generate_predictions(n=30, seed=0)
        resolve_predictions(preds, [result])
        for p in preds:
            if p.outcome is not None:
                assert isinstance(p.outcome, bool)


# ===================================================================
# Full pipeline
# ===================================================================

class TestPipelineSmoke:
    def test_smoke(self):
        report = run_market(n_predictions=20, sols=50, seeds=[42], market_seed=0)
        assert "total_predictions" in report
        assert "leaderboard" in report
        assert "calibration" in report

    def test_deterministic(self):
        r1 = run_market(n_predictions=20, sols=50, seeds=[42], market_seed=0)
        r2 = run_market(n_predictions=20, sols=50, seeds=[42], market_seed=0)
        assert r1["mean_brier"] == r2["mean_brier"]

    def test_leaderboard_sorted(self):
        report = run_market(n_predictions=40, sols=50, seeds=[42], market_seed=0)
        lb = report["leaderboard"]
        briers = [r["mean_brier"] for r in lb]
        assert briers == sorted(briers)

    def test_calibration_valid(self):
        report = run_market(n_predictions=40, sols=50, seeds=[42], market_seed=0)
        cal = report["calibration"]
        assert len(cal) == 5
        for bucket in cal:
            assert 0.0 <= bucket["actual_rate"] <= 1.0


# ===================================================================
# Physical invariants
# ===================================================================

class TestPhysicalBounds:
    def test_brier_bounded(self):
        for p in [0.0, 0.01, 0.5, 0.99, 1.0]:
            for o in [True, False]:
                assert 0.0 <= brier_score(p, o) <= 1.0

    def test_karma_direction(self):
        """Good predictions should pay more than bad ones."""
        good_payout = payout_from_brier(0.05, 10.0)
        bad_payout = payout_from_brier(0.90, 10.0)
        assert good_payout > bad_payout
