"""Tests for cross_sim.py — the Mars Barn × prediction market bridge."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
from src.cross_sim import (
    CrossSimReport,
    consensus_accuracy,
    ensemble_agreement,
    run_cross_sim,
)
from src.market_maker import Prediction, brier_score


# ---------------------------------------------------------------------------
# Unit tests — consensus_accuracy
# ---------------------------------------------------------------------------

class TestConsensusAccuracy:
    """Tests for the consensus_accuracy helper."""

    def test_empty_predictions(self) -> None:
        assert consensus_accuracy([]) == 0.0

    def test_all_correct_high_confidence(self) -> None:
        preds = [
            Prediction(id="p1", agent="a", archetype="oracle", category="survival",
                       description="q", params={}, confidence=0.9, stake=10, outcome=1, brier=0.01),
            Prediction(id="p2", agent="b", archetype="oracle", category="survival",
                       description="q", params={}, confidence=0.8, stake=10, outcome=1, brier=0.04),
        ]
        assert consensus_accuracy(preds) == 1.0

    def test_all_wrong_high_confidence(self) -> None:
        preds = [
            Prediction(id="p1", agent="a", archetype="oracle", category="survival",
                       description="q", params={}, confidence=0.9, stake=10, outcome=0, brier=0.81),
        ]
        assert consensus_accuracy(preds) == 0.0

    def test_unresolved_ignored(self) -> None:
        preds = [
            Prediction(id="p1", agent="a", archetype="oracle", category="survival",
                       description="q", params={}, confidence=0.9, stake=10, outcome=None, brier=None),
        ]
        assert consensus_accuracy(preds) == 0.0

    def test_mixed(self) -> None:
        preds = [
            Prediction(id="p1", agent="a", archetype="oracle", category="survival",
                       description="q", params={}, confidence=0.9, stake=10, outcome=1, brier=0.01),
            Prediction(id="p2", agent="b", archetype="oracle", category="survival",
                       description="q", params={}, confidence=0.8, stake=10, outcome=0, brier=0.64),
        ]
        assert consensus_accuracy(preds) == 0.5


# ---------------------------------------------------------------------------
# Unit tests — ensemble_agreement
# ---------------------------------------------------------------------------

class TestEnsembleAgreement:
    """Tests for ensemble coefficient of variation."""

    def test_single_seed_returns_zero(self) -> None:
        cv = ensemble_agreement([42], sols=10)
        assert cv == 0.0

    def test_multiple_seeds_nonnegative(self) -> None:
        cv = ensemble_agreement([42, 43, 44], sols=30)
        assert cv >= 0.0

    def test_cv_bounded(self) -> None:
        cv = ensemble_agreement([42, 43, 44], sols=100)
        assert cv < 1.0  # population shouldn't vary more than 100%


# ---------------------------------------------------------------------------
# Integration tests — full cross-sim pipeline
# ---------------------------------------------------------------------------

class TestCrossSimPipeline:
    """Full pipeline smoke tests."""

    def test_runs_without_crash(self) -> None:
        report = run_cross_sim(n_predictions=20, sols=50, seeds=[42, 43])
        assert isinstance(report, CrossSimReport)

    def test_all_predictions_resolved(self) -> None:
        report = run_cross_sim(n_predictions=20, sols=50, seeds=[42, 43])
        assert report.n_resolved == report.n_predictions

    def test_brier_bounded(self) -> None:
        report = run_cross_sim(n_predictions=20, sols=50, seeds=[42, 43])
        assert 0.0 <= report.mean_brier <= 1.0

    def test_consensus_bounded(self) -> None:
        report = run_cross_sim(n_predictions=20, sols=50, seeds=[42, 43])
        assert 0.0 <= report.consensus_acc <= 1.0

    def test_calibration_has_buckets(self) -> None:
        report = run_cross_sim(n_predictions=50, sols=50, seeds=[42])
        assert len(report.calibration) > 0

    def test_leaderboard_nonempty(self) -> None:
        report = run_cross_sim(n_predictions=50, sols=50, seeds=[42])
        assert len(report.leaderboard) > 0

    def test_category_briers_present(self) -> None:
        report = run_cross_sim(n_predictions=50, sols=50, seeds=[42, 43])
        assert len(report.category_briers) > 0
        for cat, brier in report.category_briers.items():
            assert 0.0 <= brier <= 1.0, f"{cat} brier {brier} out of bounds"

    def test_terrarium_summary_has_colonies(self) -> None:
        report = run_cross_sim(n_predictions=20, sols=50, seeds=[42])
        assert len(report.terrarium_summary["colonies"]) == 3

    def test_to_dict_serializable(self) -> None:
        report = run_cross_sim(n_predictions=10, sols=30, seeds=[42])
        d = report.to_dict()
        import json
        json.dumps(d)  # must not raise

    def test_deterministic_same_seed(self) -> None:
        r1 = run_cross_sim(n_predictions=20, sols=50, seeds=[42], rng_seed=99)
        r2 = run_cross_sim(n_predictions=20, sols=50, seeds=[42], rng_seed=99)
        assert r1.mean_brier == r2.mean_brier
        assert r1.consensus_acc == r2.consensus_acc


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------

class TestConservationLaws:
    """Cross-sim invariants that must hold."""

    def test_brier_scores_nonnegative(self) -> None:
        report = run_cross_sim(n_predictions=30, sols=50, seeds=[42])
        for cat, b in report.category_briers.items():
            assert b >= 0.0, f"{cat} has negative Brier"

    def test_leaderboard_briers_bounded(self) -> None:
        report = run_cross_sim(n_predictions=30, sols=50, seeds=[42])
        for row in report.leaderboard:
            assert 0.0 <= row["mean_brier"] <= 1.0

    def test_ensemble_cv_nonnegative(self) -> None:
        report = run_cross_sim(n_predictions=10, sols=50, seeds=[42, 43])
        assert report.ensemble_cv >= 0.0

    def test_colony_populations_nonnegative(self) -> None:
        report = run_cross_sim(n_predictions=10, sols=50, seeds=[42])
        for c in report.terrarium_summary["colonies"]:
            assert c["end_pop"] >= 0
