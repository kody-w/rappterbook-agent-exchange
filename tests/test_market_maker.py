"""
Tests for the prediction market engine (market_maker.py).

Run: python -m pytest tests/test_market_maker.py -v
"""
from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.market_maker import (
    ARCHETYPES,
    PREDICTION_TEMPLATES,
    Agent,
    Forecast,
    Prediction,
    brier_score,
    build_calibration_curve,
    build_leaderboard,
    clamp,
    deterministic_seed,
    generate_agents,
    generate_counter_positions,
    generate_predictions,
    log_score,
    resolve_predictions,
    run_market,
    score_forecasts,
    settle_payouts,
    sharpness,
    submit_forecasts,
)
from src.tick_engine import Simulation


# Cache a single sim result for resolver tests
_CACHED_RESULT = None


def _get_result():
    global _CACHED_RESULT
    if _CACHED_RESULT is None:
        sim = Simulation(sols=100, env_seed=42)
        _CACHED_RESULT = sim.run()
    return _CACHED_RESULT


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

    def test_midpoint(self):
        assert brier_score(0.5, True) == 0.25

    def test_bounded_0_1(self):
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            for o in [True, False]:
                s = brier_score(p, o)
                assert 0.0 <= s <= 1.0

    def test_symmetric(self):
        s1 = brier_score(0.3, True)
        s2 = brier_score(0.7, False)
        assert abs(s1 - s2) < 1e-9


# ===================================================================
# Log score
# ===================================================================

class TestLogScore:
    def test_confident_correct(self):
        s = log_score(0.95, True)
        assert s > 0
        assert s < 0.1

    def test_confident_wrong(self):
        s = log_score(0.95, False)
        assert s > 2.0

    def test_always_nonnegative(self):
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            for o in [True, False]:
                assert log_score(p, o) >= 0.0

    def test_handles_extreme(self):
        s1 = log_score(0.0, True)
        assert math.isfinite(s1) and s1 > 0
        s2 = log_score(1.0, False)
        assert math.isfinite(s2) and s2 > 0


# ===================================================================
# Helpers
# ===================================================================

class TestHelpers:
    def test_clamp_within(self):
        assert clamp(0.5) == 0.5

    def test_clamp_low(self):
        assert clamp(0.0) == 0.01

    def test_clamp_high(self):
        assert clamp(1.0) == 0.99

    def test_sharpness_extreme(self):
        assert sharpness(0.0) == 0.5
        assert sharpness(1.0) == 0.5

    def test_sharpness_middle(self):
        assert sharpness(0.5) == 0.0

    def test_deterministic_seed_stable(self):
        a = deterministic_seed(42, "test")
        b = deterministic_seed(42, "test")
        assert a == b

    def test_deterministic_seed_varies(self):
        a = deterministic_seed(42, "test")
        b = deterministic_seed(43, "test")
        assert a != b


# ===================================================================
# Agent generation
# ===================================================================

class TestGenerateAgents:
    def test_correct_count(self):
        rng = random.Random(0)
        agents = generate_agents(12, rng)
        assert len(agents) == 12

    def test_round_robin_archetypes(self):
        rng = random.Random(0)
        agents = generate_agents(len(ARCHETYPES) * 2, rng)
        for i, a in enumerate(agents):
            assert a.archetype == ARCHETYPES[i % len(ARCHETYPES)]

    def test_starting_karma(self):
        rng = random.Random(0)
        for a in generate_agents(5, rng):
            assert a.karma == 1000.0


# ===================================================================
# Prediction generation
# ===================================================================

class TestGeneratePredictions:
    def test_produces_predictions(self):
        rng = random.Random(1)
        preds = generate_predictions(n=30, n_colonies=3, rng=rng)
        assert len(preds) > 0
        assert len(preds) <= 30

    def test_unique_ids(self):
        rng = random.Random(2)
        preds = generate_predictions(n=50, n_colonies=3, rng=rng)
        ids = [p.prediction_id for p in preds]
        assert len(ids) == len(set(ids))

    def test_base_rate_bounded(self):
        rng = random.Random(3)
        preds = generate_predictions(n=100, n_colonies=3, rng=rng)
        for p in preds:
            assert 0.25 <= p.base_rate <= 0.75

    def test_check_fn_present(self):
        rng = random.Random(4)
        preds = generate_predictions(n=20, n_colonies=3, rng=rng)
        for p in preds:
            assert callable(p.check_fn)

    def test_deterministic(self):
        p1 = generate_predictions(n=20, n_colonies=3, rng=random.Random(99))
        p2 = generate_predictions(n=20, n_colonies=3, rng=random.Random(99))
        for a, b in zip(p1, p2):
            assert a.prediction_id == b.prediction_id
            assert a.base_rate == b.base_rate


# ===================================================================
# Forecasting
# ===================================================================

class TestSubmitForecasts:
    def test_forecasts_produced(self):
        rng = random.Random(10)
        agents = generate_agents(6, rng)
        preds = generate_predictions(5, 3, rng)
        forecasts = submit_forecasts(agents, preds, 0, rng)
        assert len(forecasts) == 6 * len(preds)

    def test_karma_decreases(self):
        rng = random.Random(11)
        agents = generate_agents(3, rng)
        preds = generate_predictions(5, 3, rng)
        initial_karma = [a.karma for a in agents]
        submit_forecasts(agents, preds, 0, rng)
        for a, ik in zip(agents, initial_karma):
            assert a.karma < ik

    def test_forecasts_bounded(self):
        rng = random.Random(12)
        agents = generate_agents(6, rng)
        preds = generate_predictions(10, 3, rng)
        forecasts = submit_forecasts(agents, preds, 0, rng)
        for f in forecasts:
            assert 0.01 <= f.probability <= 0.99
            assert f.stake > 0


# ===================================================================
# Counter positions
# ===================================================================

class TestCounterPositions:
    def test_produces_counters(self):
        rng = random.Random(20)
        agents = generate_agents(10, rng)
        preds = generate_predictions(10, 3, rng)
        fcs = submit_forecasts(agents, preds, 0, rng)
        counters = generate_counter_positions(fcs, agents, rng)
        assert len(counters) > 0
        assert all(c.is_counter for c in counters)


# ===================================================================
# Resolution
# ===================================================================

class TestResolvePredictions:
    def test_all_resolved(self):
        result = _get_result()
        rng = random.Random(30)
        preds = generate_predictions(20, 3, rng)
        resolve_predictions(preds, result)
        for p in preds:
            assert p.resolved is True
            assert isinstance(p.outcome, bool)

    def test_outcomes_deterministic(self):
        result = _get_result()
        p1 = generate_predictions(20, 3, random.Random(31))
        p2 = generate_predictions(20, 3, random.Random(31))
        resolve_predictions(p1, result)
        resolve_predictions(p2, result)
        for a, b in zip(p1, p2):
            assert a.outcome == b.outcome


# ===================================================================
# Scoring
# ===================================================================

class TestScoreForecasts:
    def test_brier_attached(self):
        result = _get_result()
        rng = random.Random(40)
        agents = generate_agents(6, rng)
        preds = generate_predictions(10, 3, rng)
        fcs = submit_forecasts(agents, preds, 0, rng)
        resolve_predictions(preds, result)
        score_forecasts(fcs, preds)
        scored = [f for f in fcs if hasattr(f, "_brier")]
        assert len(scored) > 0
        for f in scored:
            assert 0.0 <= f._brier <= 1.0


# ===================================================================
# Settle payouts
# ===================================================================

class TestSettlePayouts:
    def test_karma_changes(self):
        result = _get_result()
        rng = random.Random(50)
        agents = generate_agents(6, rng)
        preds = generate_predictions(10, 3, rng)
        fcs = submit_forecasts(agents, preds, 0, rng)
        resolve_predictions(preds, result)
        score_forecasts(fcs, preds)
        karma_before = [a.karma for a in agents]
        settle_payouts(fcs, agents, preds)
        karma_after = [a.karma for a in agents]
        assert karma_before != karma_after

    def test_payout_nonnegative(self):
        result = _get_result()
        rng = random.Random(51)
        agents = generate_agents(6, rng)
        preds = generate_predictions(10, 3, rng)
        fcs = submit_forecasts(agents, preds, 0, rng)
        resolve_predictions(preds, result)
        score_forecasts(fcs, preds)
        settle_payouts(fcs, agents, preds)
        for a in agents:
            assert a.total_payout >= 0.0


# ===================================================================
# Leaderboard
# ===================================================================

class TestLeaderboard:
    def test_sorted_by_brier(self):
        result = _get_result()
        rng = random.Random(60)
        agents = generate_agents(12, rng)
        preds = generate_predictions(20, 3, rng)
        fcs = submit_forecasts(agents, preds, 0, rng)
        resolve_predictions(preds, result)
        score_forecasts(fcs, preds)
        settle_payouts(fcs, agents, preds)
        lb = build_leaderboard(agents)
        briers = [r["mean_brier"] for r in lb]
        assert briers == sorted(briers)

    def test_all_agents_present(self):
        rng = random.Random(61)
        agents = generate_agents(8, rng)
        preds = generate_predictions(10, 3, rng)
        result = _get_result()
        fcs = submit_forecasts(agents, preds, 0, rng)
        resolve_predictions(preds, result)
        score_forecasts(fcs, preds)
        settle_payouts(fcs, agents, preds)
        lb = build_leaderboard(agents)
        assert len(lb) == 8


# ===================================================================
# Calibration curve
# ===================================================================

class TestCalibrationCurve:
    def test_ten_bins(self):
        result = _get_result()
        rng = random.Random(70)
        agents = generate_agents(6, rng)
        preds = generate_predictions(20, 3, rng)
        fcs = submit_forecasts(agents, preds, 0, rng)
        resolve_predictions(preds, result)
        score_forecasts(fcs, preds)
        curve = build_calibration_curve(fcs, preds)
        assert len(curve) == 10

    def test_observed_freq_bounded(self):
        result = _get_result()
        rng = random.Random(71)
        agents = generate_agents(6, rng)
        preds = generate_predictions(30, 3, rng)
        fcs = submit_forecasts(agents, preds, 0, rng)
        resolve_predictions(preds, result)
        score_forecasts(fcs, preds)
        curve = build_calibration_curve(fcs, preds)
        for b in curve:
            assert 0.0 <= b["observed_freq"] <= 1.0


# ===================================================================
# Pipeline smoke test
# ===================================================================

class TestPipelineSmoke:
    def test_small_run(self):
        result = _get_result()
        report = run_market(result, n_agents=10, n_predictions=15, seed=0, quiet=True)
        assert "leaderboard" in report
        assert "calibration" in report
        assert "predictions" in report
        assert report["_meta"]["n_agents"] == 10

    def test_deterministic(self):
        result = _get_result()
        r1 = run_market(result, n_agents=10, n_predictions=15, seed=7, quiet=True)
        r2 = run_market(result, n_agents=10, n_predictions=15, seed=7, quiet=True)
        assert r1["summary"] == r2["summary"]


# ===================================================================
# Physical bounds
# ===================================================================

class TestPhysicalBounds:
    def test_brier_always_01(self):
        for p in [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]:
            for o in [True, False]:
                assert 0.0 <= brier_score(p, o) <= 1.0

    def test_template_check_fns_callable(self):
        for tpl in PREDICTION_TEMPLATES:
            assert callable(tpl["check"])
            assert callable(tpl["params_fn"])

    def test_predictions_all_resolve(self):
        result = _get_result()
        rng = random.Random(81)
        preds = generate_predictions(50, 3, rng)
        resolve_predictions(preds, result)
        for p in preds:
            assert p.resolved is True
            assert isinstance(p.outcome, bool)


# ===================================================================
# Terrarium integration
# ===================================================================

class TestTerrariumIntegration:
    def test_sim_produces_colonies(self):
        result = _get_result()
        assert "colonies" in result
        assert len(result["colonies"]) == 3

    def test_colony_names(self):
        result = _get_result()
        names = {c["name"] for c in result["colonies"]}
        assert "Ares Prime" in names
        assert "Olympus Station" in names
        assert "Red Frontier" in names

    def test_sim_has_environment(self):
        result = _get_result()
        assert "environment" in result

    def test_sim_has_summary(self):
        result = _get_result()
        assert "summary" in result
