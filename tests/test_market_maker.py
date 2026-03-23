"""
Tests for market_maker.py — prediction market engine v6.0.

Covers: Brier scoring, sharpness, counter-positions, calibration,
payouts, pipeline integration, and physical invariants.

Run: python -m pytest tests/test_market_maker.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.market_maker import (
    brier_score,
    log_score,
    Prediction,
    AgentScore,
    MarketState,
    generate_predictions,
    generate_counter_positions,
    resolve_predictions,
    score_predictions,
    compute_payouts,
    build_calibration_curve,
    build_agent_scores,
    run_market,
    _consecutive_above,
    _growth_pct,
)
from src.tick_engine import Simulation


# ─── Brier Score Tests ───

class TestBrierScore:
    def test_perfect_true(self):
        assert brier_score(1.0, 1.0) == 0.0

    def test_perfect_false(self):
        assert brier_score(0.0, 0.0) == 0.0

    def test_worst_true(self):
        assert brier_score(0.0, 1.0) == 1.0

    def test_worst_false(self):
        assert brier_score(1.0, 0.0) == 1.0

    def test_halfway(self):
        assert brier_score(0.5, 1.0) == 0.25

    def test_bounded(self):
        import random
        rng = random.Random(42)
        for _ in range(500):
            b = brier_score(rng.random(), rng.choice([0.0, 1.0]))
            assert 0.0 <= b <= 1.0

    def test_symmetric(self):
        assert abs(brier_score(0.7, 1.0) - brier_score(0.3, 0.0)) < 1e-10

    def test_high_confidence_correct(self):
        assert brier_score(0.9, 1.0) < 0.1

    def test_high_confidence_wrong(self):
        assert brier_score(0.9, 0.0) > 0.8


class TestLogScore:
    def test_perfect(self):
        assert log_score(0.99, 1.0) > -0.02

    def test_terrible(self):
        assert log_score(0.99, 0.0) < -4.0


# ─── Sharpness Tests ───

class TestSharpness:
    def _pred(self, conf: float) -> Prediction:
        return Prediction(id="x", text="t", author="a", confidence=conf,
                         stake=10, template_idx=0, params={})

    def test_coinflip_zero(self):
        assert self._pred(0.5).sharpness == 0.0

    def test_high_conf(self):
        assert abs(self._pred(0.95).sharpness - 0.45) < 0.001

    def test_low_conf(self):
        assert abs(self._pred(0.1).sharpness - 0.4) < 0.001

    def test_bounded(self):
        import random
        rng = random.Random(42)
        for _ in range(100):
            s = self._pred(rng.random()).sharpness
            assert 0.0 <= s <= 0.5


# ─── Counter-Position Tests ───

class TestCounterPositions:
    def _run_sim(self, sols: int = 100) -> tuple:
        sim = Simulation(sols=sols, env_seed=42)
        results = sim.run()
        names = [c["name"] for c in results["colonies"]]
        return results, names

    def test_counters_generated(self):
        _, names = self._run_sim()
        preds = generate_predictions(50, names, seed=42)
        preds = generate_counter_positions(preds, seed=42)
        countered = [p for p in preds if p.has_counters]
        assert 0 < len(countered) < len(preds)

    def test_no_self_counter(self):
        _, names = self._run_sim()
        preds = generate_predictions(100, names, seed=42)
        preds = generate_counter_positions(preds, seed=42)
        for p in preds:
            for c in p.counter_positions:
                assert c["agent"] != p.author

    def test_counter_stake_positive(self):
        _, names = self._run_sim(50)
        preds = generate_predictions(30, names, seed=42)
        preds = generate_counter_positions(preds, seed=42)
        for p in preds:
            for c in p.counter_positions:
                assert c["stake"] > 0

    def test_counter_confidence_valid(self):
        _, names = self._run_sim(50)
        preds = generate_predictions(50, names, seed=42)
        preds = generate_counter_positions(preds, seed=42)
        for p in preds:
            for c in p.counter_positions:
                assert 0.0 < c["counter_confidence"] <= 1.0

    def test_total_counter_stake(self):
        _, names = self._run_sim(50)
        preds = generate_predictions(30, names, seed=42)
        preds = generate_counter_positions(preds, seed=42)
        for p in preds:
            expected = sum(c["stake"] for c in p.counter_positions)
            assert p.total_counter_stake == expected


# ─── Payout Tests ───

class TestPayouts:
    def _make_pred(self, conf: float, outcome: float, stake: int = 100) -> Prediction:
        pred = Prediction(id="x", text="t", author="a", confidence=conf,
                         stake=stake, template_idx=0, params={})
        pred.outcome = outcome
        pred.brier = brier_score(conf, outcome)
        return pred

    def test_perfect_gets_bonus(self):
        pred = self._make_pred(0.95, 1.0)
        compute_payouts([pred])
        assert pred.payout >= pred.stake * 2.0

    def test_coinflip_no_bonus(self):
        pred = self._make_pred(0.5, 1.0)
        compute_payouts([pred])
        assert pred.payout <= pred.stake * 1.5

    def test_awful_loses_all(self):
        pred = self._make_pred(0.95, 0.0)
        compute_payouts([pred])
        assert pred.payout == 0.0

    def test_payout_nonnegative(self):
        import random
        rng = random.Random(42)
        for _ in range(200):
            pred = self._make_pred(
                rng.uniform(0.01, 0.99),
                rng.choice([0.0, 1.0]),
                rng.randint(1, 100))
            compute_payouts([pred])
            assert pred.payout >= 0.0

    def test_sharp_beats_timid(self):
        sharp = self._make_pred(0.95, 1.0, 100)
        timid = self._make_pred(0.60, 1.0, 100)
        compute_payouts([sharp, timid])
        assert sharp.payout > timid.payout

    def test_counter_settlement(self):
        pred = self._make_pred(0.8, 1.0, 100)
        pred.counter_positions = [
            {"agent": "x", "counter_confidence": 0.7, "stake": 50, "payout": 0.0},
        ]
        compute_payouts([pred])
        assert pred.payout > 100  # base + counter winnings
        assert pred.counter_positions[0]["payout"] == 0.0


# ─── Calibration Tests ───

class TestCalibration:
    def test_five_buckets(self):
        assert len(build_calibration_curve([])) == 5

    def test_with_data(self):
        preds = []
        for i in range(20):
            p = Prediction(id=f"p{i}", text="t", author="a", confidence=0.85,
                          stake=10, template_idx=0, params={})
            p.outcome = 1.0 if i < 17 else 0.0
            p.brier = brier_score(0.85, p.outcome)
            preds.append(p)
        curve = build_calibration_curve(preds)
        assert curve[4]["count"] == 20
        assert abs(curve[4]["actual_rate"] - 0.85) < 0.01


# ─── Agent Score Tests ───

class TestAgentScores:
    def test_aggregation(self):
        preds = []
        for i in range(5):
            p = Prediction(id=f"p{i}", text="t", author="agent-1",
                          confidence=0.8, stake=10, template_idx=0, params={})
            p.outcome = 1.0 if i < 4 else 0.0
            p.brier = brier_score(0.8, p.outcome)
            p.payout = 15.0 if i < 4 else 0.0
            preds.append(p)
        scores = build_agent_scores(preds)
        assert scores["agent-1"].predictions == 5
        assert scores["agent-1"].correct == 4
        assert scores["agent-1"].accuracy == 0.8


# ─── Helper Tests ───

class TestHelpers:
    def test_consecutive_above_true(self):
        assert _consecutive_above([0.8, 0.9, 0.85, 0.7], 0.75, 3)

    def test_consecutive_above_false(self):
        assert not _consecutive_above([0.8, 0.9, 0.7, 0.85], 0.75, 3)

    def test_growth_pct(self):
        assert _growth_pct({"initial_population": 100, "final_population": 150}) == 50.0


# ─── Pipeline Integration ───

class TestPipeline:
    def _run(self, sols: int = 100, n: int = 50) -> MarketState:
        sim = Simulation(sols=sols, env_seed=42)
        results = sim.run()
        names = [c["name"] for c in results["colonies"]]
        return run_market(results, names, n_predictions=n, seed=42)

    def test_runs_without_error(self):
        market = self._run()
        assert len(market.predictions) == 50
        assert any(p.outcome is not None for p in market.predictions)

    def test_deterministic(self):
        m1 = self._run(sols=50, n=30)
        m2 = self._run(sols=50, n=30)
        d1, d2 = m1.to_dict(), m2.to_dict()
        assert d1["_meta"]["total_predictions"] == d2["_meta"]["total_predictions"]
        assert d1["_meta"]["resolved_count"] == d2["_meta"]["resolved_count"]

    def test_valid_confidence(self):
        market = self._run()
        for p in market.predictions:
            assert 0.0 < p.confidence <= 1.0

    def test_resolved_have_brier(self):
        market = self._run()
        for p in market.predictions:
            if p.outcome is not None:
                assert p.brier is not None
                assert 0.0 <= p.brier <= 1.0

    def test_serialization(self):
        market = self._run(sols=50, n=20)
        out = market.to_dict()
        json_str = json.dumps(out)
        parsed = json.loads(json_str)
        assert parsed["_meta"]["version"] == "6.0.0"

    def test_counters_in_output(self):
        market = self._run(n=100)
        out = market.to_dict()
        assert "countered_count" in out["_meta"]
        all_preds = out["resolved_predictions"] + out["open_predictions"]
        assert any(p.get("total_counter_stake", 0) > 0 for p in all_preds)

    def test_sharpness_in_output(self):
        market = self._run(sols=50, n=20)
        out = market.to_dict()
        for p in out["resolved_predictions"] + out["open_predictions"]:
            assert "sharpness" in p
            assert 0.0 <= p["sharpness"] <= 0.5


# ─── Property Invariants ───

class TestInvariants:
    def test_brier_bounded(self):
        for c in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            for o in [0.0, 1.0]:
                assert 0.0 <= brier_score(c, o) <= 1.0

    def test_total_payout_finite(self):
        sim = Simulation(sols=100, env_seed=42)
        r = sim.run()
        names = [c["name"] for c in r["colonies"]]
        market = run_market(r, names, n_predictions=100, seed=42)
        total = sum(p.payout for p in market.predictions)
        assert 0 <= total < float('inf')

    def test_agent_scores_consistent(self):
        sim = Simulation(sols=100, env_seed=42)
        r = sim.run()
        names = [c["name"] for c in r["colonies"]]
        market = run_market(r, names, n_predictions=80, seed=42)
        for s in market.agent_scores.values():
            assert s.correct <= s.resolved <= s.predictions
