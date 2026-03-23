"""
Tests for market_maker.py — prediction market engine.

Covers all 5 pipeline stages + property-based invariants:
- Brier scores in [0, 1]
- Log scores <= 0
- Payouts follow Brier thresholds
- Leaderboard sorted by mean Brier ascending
- Calibration curve has 5 buckets
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.market_maker import (
    brier_score, log_score, payout_from_brier,
    generate_predictions, resolve_predictions, score_predictions,
    compute_payouts, build_calibration_curve, build_leaderboard,
    build_output, _agent_confidence, _check_outcome,
    Prediction, AgentRecord, PREDICTOR_AGENTS, CATEGORIES,
    KARMA_STAKE_MAX,
)


COLONY_NAMES = ["Ares Prime", "Olympus Station", "Red Frontier"]


@pytest.fixture
def sim_results():
    """Realistic sim results for resolving predictions."""
    return {
        "_meta": {"engine": "mars-barn", "version": "4.0", "sols": 365},
        "environment": {"history": [
            {"sol": i, "storm": "global" if 200 <= i <= 220 else None}
            for i in range(1, 366)
        ]},
        "colonies": [
            {
                "name": "Ares Prime", "strategy": "conservative",
                "initial_population": 120, "final_population": 200,
                "total_births": 100, "total_deaths": 20,
                "total_immigrants": 5, "total_emigrants": 3,
                "tech": {"unlocked_count": 4, "unlocked": [
                    {"name": "Advanced Solar Cells", "sol": 80},
                    {"name": "Martian Crop Genetics", "sol": 150},
                    {"name": "Regolith Rad Shielding", "sol": 220},
                    {"name": "Zero-Loss Water Recycling", "sol": 300},
                ]},
                "history": [
                    {"sol": i, "population": 120 + i // 5, "morale": 0.65,
                     "food_kg": 5000, "births": 1, "deaths": 0}
                    for i in range(1, 366)
                ],
                "events": [{"sol": 100, "type": "epidemic_start", "strain": "Mars Flu"}],
            },
            {
                "name": "Olympus Station", "strategy": "balanced",
                "initial_population": 80, "final_population": 130,
                "total_births": 60, "total_deaths": 10,
                "total_immigrants": 3, "total_emigrants": 1,
                "tech": {"unlocked_count": 3, "unlocked": [
                    {"name": "Advanced Solar Cells", "sol": 90},
                    {"name": "AI Diagnostics", "sol": 200},
                    {"name": "Aquaponics Integration", "sol": 280},
                ]},
                "history": [
                    {"sol": i, "population": 80 + i // 7, "morale": 0.65,
                     "food_kg": 3000, "births": 0, "deaths": 0}
                    for i in range(1, 366)
                ],
                "events": [],
            },
            {
                "name": "Red Frontier", "strategy": "aggressive",
                "initial_population": 60, "final_population": 150,
                "total_births": 100, "total_deaths": 10,
                "total_immigrants": 2, "total_emigrants": 0,
                "tech": {"unlocked_count": 5, "unlocked": [
                    {"name": "Martian Crop Genetics", "sol": 60},
                    {"name": "Advanced Solar Cells", "sol": 100},
                    {"name": "AI Diagnostics", "sol": 180},
                    {"name": "Compact Fusion Reactor", "sol": 250},
                    {"name": "Autonomous Construction Bots", "sol": 330},
                ]},
                "history": [
                    {"sol": i, "population": 60 + i // 3, "morale": 0.75,
                     "food_kg": 2000, "births": 1, "deaths": 0}
                    for i in range(1, 366)
                ],
                "events": [],
            },
        ],
        "summary": {
            "colonies": [
                {"name": "Ares Prime", "strategy": "conservative", "start_pop": 120,
                 "end_pop": 200, "growth_pct": 66.7, "peak_pop": 200, "min_pop": 120,
                 "total_births": 100, "total_deaths": 20, "techs_unlocked": 4},
                {"name": "Olympus Station", "strategy": "balanced", "start_pop": 80,
                 "end_pop": 130, "growth_pct": 62.5, "peak_pop": 130, "min_pop": 80,
                 "total_births": 60, "total_deaths": 10, "techs_unlocked": 3},
                {"name": "Red Frontier", "strategy": "aggressive", "start_pop": 60,
                 "end_pop": 150, "growth_pct": 150.0, "peak_pop": 150, "min_pop": 60,
                 "total_births": 100, "total_deaths": 10, "techs_unlocked": 5},
            ],
            "total_migrations": 15,
        },
    }


@pytest.fixture
def predictions():
    rng = random.Random(42)
    return generate_predictions(COLONY_NAMES, rng)


# --- Scoring functions ---

class TestBrierScore:
    def test_perfect_true(self):
        assert brier_score(1.0, True) == 0.0

    def test_perfect_false(self):
        assert brier_score(0.0, False) == 0.0

    def test_worst_true(self):
        assert brier_score(0.0, True) == 1.0

    def test_worst_false(self):
        assert brier_score(1.0, False) == 1.0

    def test_fifty_fifty(self):
        assert brier_score(0.5, True) == 0.25

    def test_bounded_property(self):
        rng = random.Random(123)
        for _ in range(5000):
            c = rng.random()
            o = rng.choice([True, False])
            assert 0.0 <= brier_score(c, o) <= 1.0


class TestLogScore:
    def test_correct_high_confidence(self):
        assert log_score(0.9, True) > -1

    def test_wrong_high_confidence(self):
        assert log_score(0.9, False) < -2

    def test_always_nonpositive(self):
        rng = random.Random(456)
        for _ in range(5000):
            c = rng.uniform(0.01, 0.99)
            o = rng.choice([True, False])
            assert log_score(c, o) <= 0.0

    def test_monotonic_correct(self):
        scores = [log_score(c / 100, True) for c in range(10, 100)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]


class TestPayoutFromBrier:
    def test_perfect(self):
        assert payout_from_brier(20, 0.05) == 40

    def test_good(self):
        assert payout_from_brier(20, 0.20) == 20

    def test_mediocre(self):
        assert payout_from_brier(20, 0.35) == 0

    def test_bad(self):
        assert payout_from_brier(20, 0.60) == -20


# --- Stage 1: Generate ---

class TestGenerate:
    def test_produces_predictions(self, predictions):
        assert len(predictions) > 100

    def test_all_categories(self, predictions):
        cats = {p.category for p in predictions}
        assert cats == set(CATEGORIES)

    def test_all_agents(self, predictions):
        agents = {p.agent_id for p in predictions}
        expected = {a["id"] for a in PREDICTOR_AGENTS}
        assert agents == expected

    def test_confidence_bounded(self, predictions):
        for p in predictions:
            assert 0.05 <= p.confidence <= 0.95

    def test_stake_bounded(self, predictions):
        for p in predictions:
            assert 1 <= p.stake <= KARMA_STAKE_MAX

    def test_deterministic(self):
        p1 = generate_predictions(COLONY_NAMES, random.Random(42))
        p2 = generate_predictions(COLONY_NAMES, random.Random(42))
        assert len(p1) == len(p2)
        for a, b in zip(p1, p2):
            assert a.confidence == b.confidence


# --- Stage 2: Resolve ---

class TestResolve:
    def test_all_resolved(self, predictions, sim_results):
        resolved = resolve_predictions(predictions, sim_results)
        assert all(p.resolved for p in resolved)

    def test_outcomes_are_bool(self, predictions, sim_results):
        resolved = resolve_predictions(predictions, sim_results)
        for p in resolved:
            assert isinstance(p.outcome, bool)

    def test_survival_correct(self, sim_results):
        """All colonies survive in fixture data, so survival=True."""
        pred = Prediction(
            id="test", agent_id="oracle-prime", category="survival",
            description="Ares Prime survives 365 sols",
            confidence=0.9, stake=10, colony="Ares Prime", threshold=1,
        )
        resolve_predictions([pred], sim_results)
        assert pred.outcome is True

    def test_growth_high_threshold_false(self, sim_results):
        """Ares Prime grows 66.7%, so >200% is false."""
        pred = Prediction(
            id="test", agent_id="oracle-prime", category="growth",
            description="Ares Prime grows by >200%",
            confidence=0.1, stake=10, colony="Ares Prime", threshold=200.0,
        )
        resolve_predictions([pred], sim_results)
        assert pred.outcome is False


# --- Stage 3: Score ---

class TestScorePredictions:
    def test_scored(self, predictions, sim_results):
        resolved = resolve_predictions(predictions, sim_results)
        scored = score_predictions(resolved)
        for p in scored:
            assert p.brier_score is not None
            assert p.log_score is not None

    def test_brier_bounded(self, predictions, sim_results):
        resolved = resolve_predictions(predictions, sim_results)
        scored = score_predictions(resolved)
        for p in scored:
            assert 0.0 <= p.brier_score <= 1.0


# --- Stage 4: Settle ---

class TestComputePayouts:
    def test_payouts_set(self, predictions, sim_results):
        resolved = resolve_predictions(predictions, sim_results)
        scored = score_predictions(resolved)
        settled = compute_payouts(scored)
        for p in settled:
            if p.brier_score is not None:
                expected = payout_from_brier(p.stake, p.brier_score)
                assert p.payout == expected


# --- Stage 5: Report ---

class TestCalibrationCurve:
    def test_five_buckets(self, predictions, sim_results):
        resolved = resolve_predictions(predictions, sim_results)
        scored = score_predictions(resolved)
        curve = build_calibration_curve(scored)
        assert len(curve) == 5

    def test_actual_rate_bounded(self, predictions, sim_results):
        resolved = resolve_predictions(predictions, sim_results)
        scored = score_predictions(resolved)
        curve = build_calibration_curve(scored)
        for b in curve:
            assert 0.0 <= b["actual_rate"] <= 1.0


class TestLeaderboard:
    def test_sorted_by_brier(self, predictions, sim_results):
        resolved = resolve_predictions(predictions, sim_results)
        scored = score_predictions(resolved)
        settled = compute_payouts(scored)
        board = build_leaderboard(settled)
        briers = [e["mean_brier"] for e in board]
        assert briers == sorted(briers)

    def test_all_agents_present(self, predictions, sim_results):
        resolved = resolve_predictions(predictions, sim_results)
        scored = score_predictions(resolved)
        board = build_leaderboard(scored)
        assert len(board) == len(PREDICTOR_AGENTS)


# --- Integration ---

class TestBuildOutput:
    def test_output_structure(self, predictions, sim_results):
        resolved = resolve_predictions(predictions, sim_results)
        scored = score_predictions(resolved)
        settled = compute_payouts(scored)
        calibration = build_calibration_curve(settled)
        board = build_leaderboard(settled)
        output = build_output(settled, board, calibration, sim_results["summary"])
        assert "_meta" in output
        assert output["_meta"]["total_predictions"] > 0
        assert output["_meta"]["total_resolved"] > 0

    def test_serializable(self, predictions, sim_results):
        resolved = resolve_predictions(predictions, sim_results)
        scored = score_predictions(resolved)
        settled = compute_payouts(scored)
        calibration = build_calibration_curve(settled)
        board = build_leaderboard(settled)
        output = build_output(settled, board, calibration, sim_results["summary"])
        s = json.dumps(output)
        assert json.loads(s)["_meta"]["total_predictions"] > 0


class TestFullPipeline:
    def test_pipeline_runs(self):
        """Smoke test: full pipeline with actual sim (10 sols for speed)."""
        from src.market_maker import run_pipeline
        output = run_pipeline(sols=10, env_seed=42, quiet=True)
        assert output["_meta"]["total_predictions"] > 0
        assert output["_meta"]["total_resolved"] > 0

    def test_pipeline_deterministic(self):
        from src.market_maker import run_pipeline
        r1 = run_pipeline(sols=10, env_seed=42, quiet=True)
        r2 = run_pipeline(sols=10, env_seed=42, quiet=True)
        assert r1["_meta"]["total_predictions"] == r2["_meta"]["total_predictions"]
        assert r1["_meta"]["mean_brier_score"] == r2["_meta"]["mean_brier_score"]
