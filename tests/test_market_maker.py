"""
Tests for market_maker.py — prediction market engine.

Run: python -m pytest tests/test_market_maker.py -v
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.market_maker import (
    brier_score,
    log_score,
    generate_predictions,
    resolve_predictions,
    score_predictions,
    compute_payouts,
    build_calibration_curve,
    build_agent_scores,
    run_market,
    Prediction,
    AgentScore,
    MarketState,
    _consecutive_above,
    _growth_pct,
    PREDICTION_TEMPLATES,
    PREDICTOR_ARCHETYPES,
)
from src.tick_engine import Simulation


# ─── Brier score tests ───

class TestBrierScore:
    def test_perfect_positive(self) -> None:
        assert brier_score(1.0, 1.0) == 0.0

    def test_perfect_negative(self) -> None:
        assert brier_score(0.0, 0.0) == 0.0

    def test_worst_positive(self) -> None:
        assert brier_score(0.0, 1.0) == 1.0

    def test_worst_negative(self) -> None:
        assert brier_score(1.0, 0.0) == 1.0

    def test_half_confidence(self) -> None:
        assert brier_score(0.5, 1.0) == 0.25

    def test_range_bounded(self) -> None:
        for _ in range(100):
            f = random.random()
            o = random.choice([0.0, 1.0])
            score = brier_score(f, o)
            assert 0.0 <= score <= 1.0

    def test_symmetry(self) -> None:
        assert abs(brier_score(0.7, 1.0) - brier_score(0.3, 0.0)) < 1e-10


class TestLogScore:
    def test_perfect(self) -> None:
        assert log_score(1.0, 1.0) == 0.0

    def test_half(self) -> None:
        assert abs(log_score(0.5, 1.0) - math.log(0.5)) < 1e-10

    def test_confident_wrong(self) -> None:
        score = log_score(0.99, 0.0)
        assert score < -4  # very negative

    def test_always_finite(self) -> None:
        # Even extreme cases shouldn't produce -inf
        score = log_score(0.0, 1.0)
        assert math.isfinite(score)
        assert score < -20  # very bad but finite


# ─── Helper tests ───

class TestConsecutiveAbove:
    def test_basic_streak(self) -> None:
        assert _consecutive_above([0.8, 0.8, 0.8], 0.5, 3) is True

    def test_too_short(self) -> None:
        assert _consecutive_above([0.8, 0.8, 0.3, 0.8], 0.5, 3) is False

    def test_empty(self) -> None:
        assert _consecutive_above([], 0.5, 1) is False

    def test_exact_threshold(self) -> None:
        # Must be ABOVE, not equal to
        assert _consecutive_above([0.5, 0.5, 0.5], 0.5, 3) is False


class TestGrowthPct:
    def test_growth(self) -> None:
        assert _growth_pct({"initial_population": 100, "final_population": 150}) == 50.0

    def test_decline(self) -> None:
        assert _growth_pct({"initial_population": 100, "final_population": 50}) == -50.0

    def test_zero_initial(self) -> None:
        result = _growth_pct({"initial_population": 0, "final_population": 10})
        assert result == 1000.0  # 10/1 * 100


# ─── Generation tests ───

class TestGeneratePredictions:
    def test_correct_count(self) -> None:
        preds = generate_predictions(50, ["A", "B", "C"])
        assert len(preds) == 50

    def test_deterministic(self) -> None:
        p1 = generate_predictions(20, ["A", "B", "C"], seed=99)
        p2 = generate_predictions(20, ["A", "B", "C"], seed=99)
        assert [p.id for p in p1] == [p.id for p in p2]
        assert [p.confidence for p in p1] == [p.confidence for p in p2]

    def test_different_seeds_differ(self) -> None:
        p1 = generate_predictions(20, ["A", "B", "C"], seed=1)
        p2 = generate_predictions(20, ["A", "B", "C"], seed=2)
        # At least some predictions should differ
        assert [p.confidence for p in p1] != [p.confidence for p in p2]

    def test_confidence_bounds(self) -> None:
        preds = generate_predictions(100, ["A", "B", "C"])
        for p in preds:
            assert 0.5 < p.confidence <= 1.0, f"Confidence {p.confidence} out of bounds"

    def test_stake_positive(self) -> None:
        preds = generate_predictions(100, ["A", "B", "C"])
        for p in preds:
            assert p.stake > 0

    def test_all_templates_used(self) -> None:
        # With 100 predictions and 12 templates, all should be used
        preds = generate_predictions(100, ["A", "B", "C"])
        templates_used = set(p.template_idx for p in preds)
        assert len(templates_used) == len(PREDICTION_TEMPLATES)

    def test_multiple_authors(self) -> None:
        preds = generate_predictions(100, ["A", "B", "C"])
        authors = set(p.author for p in preds)
        assert len(authors) >= 5  # Should have variety


# ─── Resolution tests ───

class TestResolve:
    def test_resolution_against_sim(self) -> None:
        sim = Simulation(sols=100, env_seed=42)
        results = sim.run()
        colony_names = [c["name"] for c in results["colonies"]]
        preds = generate_predictions(50, colony_names, seed=42)
        resolved = resolve_predictions(preds, results)
        # All should be resolved (no None outcomes)
        outcomes = [p.outcome for p in resolved]
        assert all(o is not None for o in outcomes)

    def test_mix_of_true_and_false(self) -> None:
        sim = Simulation(sols=200, env_seed=42)
        results = sim.run()
        colony_names = [c["name"] for c in results["colonies"]]
        preds = generate_predictions(100, colony_names, seed=42)
        resolved = resolve_predictions(preds, results)
        true_count = sum(1 for p in resolved if p.outcome == 1.0)
        false_count = sum(1 for p in resolved if p.outcome == 0.0)
        # Should have both true and false predictions
        assert true_count > 0, "No true predictions"
        assert false_count > 0, "No false predictions"


# ─── Scoring tests ───

class TestScoring:
    def test_resolved_get_scores(self) -> None:
        pred = Prediction(
            id="test", text="test", author="a", confidence=0.8,
            stake=10, template_idx=0, params={}, outcome=1.0
        )
        scored = score_predictions([pred])
        assert scored[0].brier is not None
        assert scored[0].log_sc is not None

    def test_unresolved_no_scores(self) -> None:
        pred = Prediction(
            id="test", text="test", author="a", confidence=0.8,
            stake=10, template_idx=0, params={}, outcome=None
        )
        scored = score_predictions([pred])
        assert scored[0].brier is None

    def test_high_confidence_correct_good_score(self) -> None:
        pred = Prediction(
            id="test", text="test", author="a", confidence=0.9,
            stake=10, template_idx=0, params={}, outcome=1.0
        )
        scored = score_predictions([pred])
        assert scored[0].brier < 0.1

    def test_high_confidence_wrong_bad_score(self) -> None:
        pred = Prediction(
            id="test", text="test", author="a", confidence=0.9,
            stake=10, template_idx=0, params={}, outcome=0.0
        )
        scored = score_predictions([pred])
        assert scored[0].brier > 0.7


# ─── Payout tests ───

class TestPayouts:
    def test_perfect_prediction_doubles(self) -> None:
        pred = Prediction(
            id="test", text="test", author="a", confidence=0.95,
            stake=10, template_idx=0, params={}, outcome=1.0,
            brier=0.0025  # nearly perfect
        )
        result = compute_payouts([pred])
        assert result[0].payout == 20.0  # 2x

    def test_bad_prediction_loses_all(self) -> None:
        pred = Prediction(
            id="test", text="test", author="a", confidence=0.9,
            stake=10, template_idx=0, params={}, outcome=0.0,
            brier=0.81  # terrible
        )
        result = compute_payouts([pred])
        assert result[0].payout == 0.0

    def test_unresolved_no_payout(self) -> None:
        pred = Prediction(
            id="test", text="test", author="a", confidence=0.7,
            stake=10, template_idx=0, params={}, outcome=None
        )
        result = compute_payouts([pred])
        assert result[0].payout == 0.0

    def test_payout_never_negative(self) -> None:
        for brier_val in [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0]:
            pred = Prediction(
                id="test", text="test", author="a", confidence=0.7,
                stake=10, template_idx=0, params={}, outcome=1.0,
                brier=brier_val
            )
            result = compute_payouts([pred])
            assert result[0].payout >= 0.0


# ─── Calibration tests ───

class TestCalibration:
    def test_five_buckets(self) -> None:
        preds = [
            Prediction(id=f"p{i}", text="t", author="a",
                       confidence=0.1 * i + 0.05, stake=10,
                       template_idx=0, params={}, outcome=1.0 if i % 2 == 0 else 0.0)
            for i in range(10)
        ]
        curve = build_calibration_curve(preds)
        assert len(curve) == 5

    def test_empty_buckets(self) -> None:
        curve = build_calibration_curve([])
        assert len(curve) == 5
        for bucket in curve:
            assert bucket["count"] == 0


# ─── Agent scoring tests ───

class TestAgentScores:
    def test_aggregation(self) -> None:
        preds = [
            Prediction(id="p1", text="t", author="agent-a", confidence=0.8,
                       stake=10, template_idx=0, params={}, outcome=1.0,
                       brier=0.04, log_sc=-0.2, payout=20.0),
            Prediction(id="p2", text="t", author="agent-a", confidence=0.7,
                       stake=10, template_idx=0, params={}, outcome=0.0,
                       brier=0.49, log_sc=-0.7, payout=5.0),
        ]
        scores = build_agent_scores(preds)
        assert "agent-a" in scores
        a = scores["agent-a"]
        assert a.predictions == 2
        assert a.resolved == 2
        assert a.correct == 1  # first was correct

    def test_roi_calculation(self) -> None:
        agent = AgentScore(name="test", total_staked=100, total_payout=150.0)
        assert agent.roi == 50.0


# ─── Full pipeline test ───

class TestFullPipeline:
    def test_run_market_smoke(self) -> None:
        """Smoke test: run full pipeline for 100 sols."""
        sim = Simulation(sols=100, env_seed=42)
        results = sim.run()
        colony_names = [c["name"] for c in results["colonies"]]
        market = run_market(results, colony_names, n_predictions=50, seed=42)

        assert len(market.predictions) == 50
        assert len(market.agent_scores) > 0
        assert len(market.calibration_curve) == 5

        # Serialization
        output = market.to_dict()
        assert "_meta" in output
        assert output["_meta"]["total_predictions"] == 50
        assert "leaderboard" in output
        assert "calibration_curve" in output

    def test_run_market_365_sols(self) -> None:
        """Full year simulation."""
        sim = Simulation(sols=365, env_seed=42)
        results = sim.run()
        colony_names = [c["name"] for c in results["colonies"]]
        market = run_market(results, colony_names, n_predictions=100, seed=42)

        resolved = [p for p in market.predictions if p.outcome is not None]
        assert len(resolved) > 80  # most should resolve

        # Physical bounds on scores
        for p in resolved:
            assert 0.0 <= p.brier <= 1.0
            assert p.payout >= 0.0

    def test_deterministic_across_runs(self) -> None:
        """Same seed → same results."""
        sim1 = Simulation(sols=100, env_seed=42)
        r1 = sim1.run()
        names1 = [c["name"] for c in r1["colonies"]]
        m1 = run_market(r1, names1, n_predictions=30, seed=42)

        sim2 = Simulation(sols=100, env_seed=42)
        r2 = sim2.run()
        names2 = [c["name"] for c in r2["colonies"]]
        m2 = run_market(r2, names2, n_predictions=30, seed=42)

        for p1, p2 in zip(m1.predictions, m2.predictions):
            assert p1.id == p2.id
            assert p1.outcome == p2.outcome
            assert p1.brier == p2.brier

    def test_conservation_of_karma(self) -> None:
        """Total payouts should not exceed what's economically possible."""
        sim = Simulation(sols=200, env_seed=42)
        results = sim.run()
        colony_names = [c["name"] for c in results["colonies"]]
        market = run_market(results, colony_names, n_predictions=100, seed=42)

        # Maximum possible payout is 2x total staked
        total_staked = sum(p.stake for p in market.predictions)
        total_payout = sum(p.payout for p in market.predictions)
        assert total_payout <= total_staked * 2.0
