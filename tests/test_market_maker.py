"""
Tests for market_maker.py — prediction market backed by Mars Barn terrarium.

Covers all 5 pipeline stages + integration. Property-based invariants:
- Brier scores in [0, 1]
- Log scores <= 0
- Calibration bucket counts sum to total predictions
- Leaderboard sorted by mean Brier ascending

Run: python -m pytest tests/test_market_maker.py -v
"""
from __future__ import annotations

import math
import json
import random
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.market_maker import (
    brier_score, log_score, payout_from_brier,
    generate_predictions, resolve_predictions, score_predictions,
    compute_payouts, build_calibration_curve, build_leaderboard,
    build_output, run_pipeline, _agent_confidence,
    CATEGORIES, PREDICTOR_AGENTS, Prediction, AgentRecord,
)
from src.tick_engine import Simulation


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
class TestBrierScore:
    def test_perfect_true(self):
        assert brier_score(1.0, True) == pytest.approx(0.0)

    def test_perfect_false(self):
        assert brier_score(0.0, False) == pytest.approx(0.0)

    def test_worst_true(self):
        assert brier_score(0.0, True) == pytest.approx(1.0)

    def test_worst_false(self):
        assert brier_score(1.0, False) == pytest.approx(1.0)

    def test_bounded(self):
        for conf in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for outcome in [True, False]:
                bs = brier_score(conf, outcome)
                assert 0.0 <= bs <= 1.0

    def test_symmetric_at_half(self):
        assert brier_score(0.5, True) == pytest.approx(brier_score(0.5, False))

    def test_better_prediction_lower(self):
        assert brier_score(0.8, True) < brier_score(0.5, True)


class TestLogScore:
    def test_perfect_true(self):
        assert log_score(1.0, True) == pytest.approx(0.0, abs=1e-9)

    def test_perfect_false(self):
        assert log_score(0.0, False) == pytest.approx(0.0, abs=1e-6)

    def test_always_nonpositive(self):
        for conf in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            for outcome in [True, False]:
                assert log_score(conf, outcome) <= 0.0

    def test_confident_correct_beats_uncertain(self):
        assert log_score(0.9, True) > log_score(0.5, True)


class TestPayout:
    def test_excellent(self):
        assert payout_from_brier(10, 0.05) == 20

    def test_good(self):
        assert payout_from_brier(10, 0.20) == 10

    def test_mediocre(self):
        assert payout_from_brier(10, 0.35) == 0

    def test_bad(self):
        assert payout_from_brier(10, 0.75) == -10


# ---------------------------------------------------------------------------
# Stage 1: Prediction generation
# ---------------------------------------------------------------------------
class TestGeneratePredictions:
    def test_generates_many(self):
        rng = random.Random(42)
        preds = generate_predictions(["Ares Prime", "Olympus Station", "Red Frontier"], rng)
        assert len(preds) > 50

    def test_all_fields_populated(self):
        rng = random.Random(42)
        preds = generate_predictions(["Ares Prime"], rng)
        for p in preds:
            assert p.id
            assert p.agent_id
            assert p.category in CATEGORIES
            assert 0.0 < p.confidence < 1.0
            assert p.stake > 0

    def test_all_agents_present(self):
        rng = random.Random(42)
        preds = generate_predictions(["Ares Prime", "Olympus Station", "Red Frontier"], rng)
        agent_ids = {p.agent_id for p in preds}
        for agent in PREDICTOR_AGENTS:
            assert agent["id"] in agent_ids

    def test_deterministic(self):
        p1 = generate_predictions(["A"], random.Random(99))
        p2 = generate_predictions(["A"], random.Random(99))
        assert len(p1) == len(p2)
        for a, b in zip(p1, p2):
            assert a.confidence == b.confidence

    def test_confidence_bounded(self):
        rng = random.Random(42)
        preds = generate_predictions(["Ares Prime", "Olympus Station", "Red Frontier"], rng)
        for p in preds:
            assert 0.05 <= p.confidence <= 0.95

    def test_all_categories_present(self):
        rng = random.Random(42)
        preds = generate_predictions(["Ares Prime", "Olympus Station", "Red Frontier"], rng)
        cats = {p.category for p in preds}
        for cat in CATEGORIES:
            assert cat in cats


# ---------------------------------------------------------------------------
# Stage 2 + 3: Resolve and Score
# ---------------------------------------------------------------------------
class TestResolveAndScore:
    def _run_and_resolve(self, seed=42, sols=100):
        rng = random.Random(seed)
        sim = Simulation(sols=sols, env_seed=seed)
        results = sim.run()
        preds = generate_predictions(["Ares Prime", "Olympus Station", "Red Frontier"], rng)
        preds = resolve_predictions(preds, results)
        return preds

    def test_all_resolved(self):
        preds = self._run_and_resolve()
        assert all(p.resolved for p in preds)

    def test_outcomes_are_booleans(self):
        preds = self._run_and_resolve()
        assert all(isinstance(p.outcome, bool) for p in preds)

    def test_scoring(self):
        preds = self._run_and_resolve()
        preds = score_predictions(preds)
        for p in preds:
            assert p.brier_score is not None
            assert 0.0 <= p.brier_score <= 1.0
            assert p.log_score is not None
            assert p.log_score <= 0.0

    def test_payouts(self):
        preds = self._run_and_resolve()
        preds = score_predictions(preds)
        preds = compute_payouts(preds)
        for p in preds:
            assert isinstance(p.payout, int)

    def test_survival_predictions_mostly_true(self):
        """All colonies survive by default — survival predictions should mostly resolve YES."""
        preds = self._run_and_resolve(sols=365)
        survival = [p for p in preds if p.category == "survival" and p.threshold <= 1]
        yes_rate = sum(1 for p in survival if p.outcome) / max(len(survival), 1)
        assert yes_rate > 0.5


# ---------------------------------------------------------------------------
# Stage 4: Calibration
# ---------------------------------------------------------------------------
class TestCalibration:
    def test_five_buckets(self):
        preds = self._make_scored_predictions()
        curve = build_calibration_curve(preds)
        assert len(curve) == 5

    def test_counts_sum(self):
        preds = self._make_scored_predictions()
        curve = build_calibration_curve(preds)
        total_in_curve = sum(b["count"] for b in curve)
        assert total_in_curve == len([p for p in preds if p.resolved])

    def test_actual_rate_bounded(self):
        preds = self._make_scored_predictions()
        curve = build_calibration_curve(preds)
        for b in curve:
            assert 0.0 <= b["actual_rate"] <= 1.0

    def _make_scored_predictions(self):
        rng = random.Random(42)
        sim = Simulation(sols=100, env_seed=42)
        results = sim.run()
        preds = generate_predictions(["Ares Prime", "Olympus Station", "Red Frontier"], rng)
        preds = resolve_predictions(preds, results)
        preds = score_predictions(preds)
        return preds


# ---------------------------------------------------------------------------
# Stage 5: Leaderboard + Output
# ---------------------------------------------------------------------------
class TestLeaderboard:
    def test_sorted_by_brier(self):
        board = self._make_board()
        for i in range(1, len(board)):
            assert board[i]["mean_brier"] >= board[i-1]["mean_brier"]

    def test_all_agents_present(self):
        board = self._make_board()
        assert len(board) == len(PREDICTOR_AGENTS)

    def test_ranks_sequential(self):
        board = self._make_board()
        for i, entry in enumerate(board):
            assert entry["rank"] == i + 1

    def _make_board(self):
        rng = random.Random(42)
        sim = Simulation(sols=100, env_seed=42)
        results = sim.run()
        preds = generate_predictions(["Ares Prime", "Olympus Station", "Red Frontier"], rng)
        preds = resolve_predictions(preds, results)
        preds = score_predictions(preds)
        preds = compute_payouts(preds)
        return build_leaderboard(preds)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
class TestPipeline:
    def test_runs_without_crash(self):
        output = run_pipeline(sols=100, env_seed=42, quiet=True)
        assert "_meta" in output
        assert output["_meta"]["engine"] == "market-maker"

    def test_has_predictions(self):
        output = run_pipeline(sols=100, env_seed=42, quiet=True)
        assert len(output["predictions"]) > 0

    def test_deterministic(self):
        o1 = run_pipeline(sols=100, env_seed=99, quiet=True)
        o2 = run_pipeline(sols=100, env_seed=99, quiet=True)
        assert o1["_meta"]["total_predictions"] == o2["_meta"]["total_predictions"]
        assert o1["_meta"]["mean_brier_score"] == o2["_meta"]["mean_brier_score"]

    def test_different_seeds(self):
        o1 = run_pipeline(sols=365, env_seed=1, quiet=True)
        o2 = run_pipeline(sols=365, env_seed=99, quiet=True)
        # Different seeds over 365 sols should produce different colony outcomes
        pop1 = [c["end_pop"] for c in o1["sim_summary"]["colonies"]]
        pop2 = [c["end_pop"] for c in o2["sim_summary"]["colonies"]]
        assert pop1 != pop2

    def test_brier_bounded(self):
        output = run_pipeline(sols=100, env_seed=42, quiet=True)
        for p in output["predictions"]:
            if p["brier_score"] is not None:
                assert 0.0 <= p["brier_score"] <= 1.0

    def test_serializable(self):
        output = run_pipeline(sols=100, env_seed=42, quiet=True)
        s = json.dumps(output)
        assert json.loads(s) == output

    def test_output_structure(self):
        output = run_pipeline(sols=100, env_seed=42, quiet=True)
        for key in ["market_stats", "predictions", "leaderboard",
                     "calibration_curve", "sim_summary"]:
            assert key in output
