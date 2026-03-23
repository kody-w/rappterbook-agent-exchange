"""
Tests for run_proof.py -- execution bridge: terrarium x prediction market.

Run: python -m pytest tests/test_run_proof.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_proof import (
    CIScore,
    ProofReport,
    compute_ci,
    format_proof_compact,
    format_proof_text,
    run_proof,
)
from src.market_maker import (
    Prediction,
    build_calibration_curve,
    generate_predictions,
    resolve_predictions,
    run_terrarium_ensemble,
    score_predictions,
)


class TestCIScore:
    """Tests for Collective Intelligence scoring."""

    def test_empty_predictions(self) -> None:
        ci = compute_ci([], [])
        assert ci.score == 0.0
        assert ci.n_resolved == 0

    def test_perfect_predictions(self) -> None:
        preds = [
            Prediction(
                id="p%d" % i, agent="oracle-%d" % i, archetype="oracle",
                category="survival", description="test", params={},
                confidence=0.9, stake=10, outcome=True, brier=0.01,
            )
            for i in range(20)
        ]
        cal = [{"bucket_lo": 0.8, "bucket_hi": 1.0, "mean_confidence": 0.9,
                "actual_rate": 0.9, "count": 20}]
        ci = compute_ci(preds, cal)
        assert ci.score > 0.5
        assert ci.n_correct == 20

    def test_random_predictions_bounded(self) -> None:
        preds = []
        for i in range(100):
            outcome = i % 2 == 0
            preds.append(Prediction(
                id="p%d" % i, agent="rng-%d" % i, archetype="degen",
                category="survival", description="test", params={},
                confidence=0.5, stake=10, outcome=outcome, brier=0.25,
            ))
        cal = [{"bucket_lo": 0.4, "bucket_hi": 0.6, "mean_confidence": 0.5,
                "actual_rate": 0.5, "count": 100}]
        ci = compute_ci(preds, cal)
        assert 0.0 <= ci.score <= 1.0

    def test_ci_bounded_integration(self) -> None:
        preds = generate_predictions(50, seed=42)
        results = run_terrarium_ensemble(sols=50, seeds=[42])
        resolve_predictions(preds, results)
        score_predictions(preds)
        cal = build_calibration_curve(preds)
        ci = compute_ci(preds, cal)
        assert 0.0 <= ci.score <= 1.0
        assert ci.calibration_error >= 0.0
        assert ci.information_ratio >= 0.0
        assert 0.0 <= ci.category_spread <= 1.0

    def test_label_strong(self) -> None:
        assert "Strong" in CIScore(score=0.85).label()

    def test_label_moderate(self) -> None:
        assert "Moderate" in CIScore(score=0.65).label()

    def test_label_weak(self) -> None:
        assert "Weak" in CIScore(score=0.45).label()

    def test_label_noise(self) -> None:
        assert "Noise" in CIScore(score=0.2).label()

    def test_to_dict_keys(self) -> None:
        ci = CIScore(score=0.7, calibration_error=0.1, information_ratio=0.5,
                     category_spread=0.8, n_resolved=50, n_correct=35)
        d = ci.to_dict()
        assert "score" in d
        assert "label" in d
        assert "n_resolved" in d


class TestProofReport:

    def test_to_dict_serializable(self) -> None:
        report = run_proof(sols=30, n_predictions=10, n_seeds=1)
        json.dumps(report.to_dict())

    def test_proof_hash_deterministic(self) -> None:
        r1 = run_proof(sols=30, n_predictions=10, n_seeds=1, market_seed=42)
        r2 = run_proof(sols=30, n_predictions=10, n_seeds=1, market_seed=42)
        assert r1.proof_hash == r2.proof_hash

    def test_proof_hash_changes(self) -> None:
        r1 = run_proof(sols=30, n_predictions=10, n_seeds=1, market_seed=42)
        r2 = run_proof(sols=30, n_predictions=10, n_seeds=1, market_seed=99)
        assert r1.proof_hash != r2.proof_hash

    def test_duration_positive(self) -> None:
        assert run_proof(sols=30, n_predictions=10, n_seeds=1).duration_s > 0


class TestRunProof:

    def test_runs_without_crash(self) -> None:
        report = run_proof(sols=50, n_predictions=20, n_seeds=1)
        assert isinstance(report, ProofReport)

    def test_all_resolved(self) -> None:
        report = run_proof(sols=50, n_predictions=20, n_seeds=1)
        assert report.n_resolved == report.n_predictions

    def test_brier_bounded(self) -> None:
        report = run_proof(sols=50, n_predictions=20, n_seeds=1)
        assert 0.0 <= report.mean_brier <= 1.0

    def test_accuracy_bounded(self) -> None:
        assert 0.0 <= run_proof(sols=50, n_predictions=20, n_seeds=1).accuracy <= 1.0

    def test_ci_present(self) -> None:
        report = run_proof(sols=50, n_predictions=30, n_seeds=2)
        assert isinstance(report.ci, CIScore)
        assert 0.0 <= report.ci.score <= 1.0

    def test_terrarium_3_colonies(self) -> None:
        report = run_proof(sols=50, n_predictions=10, n_seeds=1)
        assert len(report.terrarium["colonies"]) == 3

    def test_all_alive_short_sim(self) -> None:
        assert run_proof(sols=50, n_predictions=10, n_seeds=1).terrarium["all_alive"]

    def test_calibration_5_buckets(self) -> None:
        assert len(run_proof(sols=50, n_predictions=50, n_seeds=1).calibration) == 5

    def test_leaderboard_sorted(self) -> None:
        report = run_proof(sols=50, n_predictions=50, n_seeds=1)
        if len(report.leaderboard) > 1:
            briers = [r["mean_brier"] for r in report.leaderboard]
            assert briers == sorted(briers)

    def test_category_briers_present(self) -> None:
        report = run_proof(sols=100, n_predictions=100, n_seeds=2)
        assert len(report.category_briers) > 0
        for brier in report.category_briers.values():
            assert 0.0 <= brier <= 1.0

    def test_multi_seed_ensemble(self) -> None:
        report = run_proof(sols=50, n_predictions=20, n_seeds=3)
        assert report.n_seeds == 3 and report.n_resolved > 0


class TestFormatting:

    def test_text_sections(self) -> None:
        report = run_proof(sols=30, n_predictions=10, n_seeds=1)
        text = format_proof_text(report)
        assert "EXECUTION PROOF" in text
        assert "TERRARIUM" in text
        assert "PREDICTION MARKET" in text
        assert "COLLECTIVE INTELLIGENCE" in text
        assert report.proof_hash in text

    def test_compact_one_line(self) -> None:
        report = run_proof(sols=30, n_predictions=10, n_seeds=1)
        compact = format_proof_compact(report)
        assert "\n" not in compact
        assert "PROOF" in compact

    def test_json_parseable(self) -> None:
        report = run_proof(sols=30, n_predictions=10, n_seeds=1)
        parsed = json.loads(json.dumps(report.to_dict()))
        assert parsed["proof_hash"] == report.proof_hash


class TestConservation:

    def test_resolved_leq_total(self) -> None:
        report = run_proof(sols=50, n_predictions=30, n_seeds=1)
        assert report.n_resolved <= report.n_predictions

    def test_ci_correct_leq_resolved(self) -> None:
        report = run_proof(sols=50, n_predictions=30, n_seeds=1)
        assert report.ci.n_correct <= report.ci.n_resolved

    def test_total_pop_nonneg(self) -> None:
        assert run_proof(sols=50, n_predictions=10, n_seeds=1).terrarium["total_population"] >= 0

    def test_colony_pops_nonneg(self) -> None:
        for c in run_proof(sols=50, n_predictions=10, n_seeds=1).terrarium["colonies"]:
            assert c["end_pop"] >= 0

    def test_terraforming_bounded(self) -> None:
        tf = run_proof(sols=100, n_predictions=10, n_seeds=1).terrarium["terraforming_pct"]
        assert 0.0 <= tf <= 100.0

    def test_category_briers_nonneg(self) -> None:
        for brier in run_proof(sols=50, n_predictions=30, n_seeds=1).category_briers.values():
            assert brier >= 0.0
