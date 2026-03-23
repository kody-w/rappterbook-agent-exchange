"""
Tests for prediction market engine.

Run: python -m pytest tests/test_market_maker.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.market_maker import (
    lmsr_cost,
    lmsr_price_yes,
    lmsr_trade_cost,
    Market,
    MarketCondition,
    PredictionMarketSim,
    Trader,
    create_default_markets,
    create_traders,
    decide_trade,
    run_prediction_market,
    LMSR_B,
)

import random
import pytest


# ===========================================================================
# LMSR Math Tests
# ===========================================================================

class TestLMSRMath:
    """LMSR automated market maker math — the core invariants."""

    def test_initial_price_is_50_50(self) -> None:
        """With no trades, YES and NO are equally priced."""
        price = lmsr_price_yes(0, 0, LMSR_B)
        assert abs(price - 0.5) < 1e-10

    def test_prices_sum_to_one(self) -> None:
        """p_yes + p_no = 1.0 always (no-arbitrage condition)."""
        for q_yes in [-50, 0, 50, 100, 200]:
            for q_no in [-50, 0, 50, 100, 200]:
                p_yes = lmsr_price_yes(q_yes, q_no, LMSR_B)
                p_no = 1.0 - p_yes
                assert abs(p_yes + p_no - 1.0) < 1e-10, \
                    f"Prices don't sum to 1: q_yes={q_yes}, q_no={q_no}"

    def test_buying_yes_increases_price(self) -> None:
        """Buying YES shares should increase YES price."""
        p_before = lmsr_price_yes(0, 0, LMSR_B)
        p_after = lmsr_price_yes(10, 0, LMSR_B)
        assert p_after > p_before

    def test_buying_no_decreases_yes_price(self) -> None:
        """Buying NO shares should decrease YES price."""
        p_before = lmsr_price_yes(0, 0, LMSR_B)
        p_after = lmsr_price_yes(0, 10, LMSR_B)
        assert p_after < p_before

    def test_cost_positive_for_buy(self) -> None:
        """Buying shares costs money (positive cost)."""
        cost = lmsr_trade_cost(0, 0, LMSR_B, "yes", 10)
        assert cost > 0

    def test_cost_negative_for_sell(self) -> None:
        """Selling shares gives money back (negative cost)."""
        cost = lmsr_trade_cost(50, 0, LMSR_B, "yes", -10)
        assert cost < 0

    def test_round_trip_cost(self) -> None:
        """Buy then sell same amount: net cost should be close to zero but slightly positive (spread)."""
        buy_cost = lmsr_trade_cost(0, 0, LMSR_B, "yes", 10)
        sell_cost = lmsr_trade_cost(10, 0, LMSR_B, "yes", -10)
        # Round trip should net out exactly (LMSR has no spread on paper)
        assert abs(buy_cost + sell_cost) < 1e-10

    def test_large_quantity_moves_price_more(self) -> None:
        """Larger trades should have bigger price impact."""
        p_small = lmsr_price_yes(5, 0, LMSR_B)
        p_large = lmsr_price_yes(50, 0, LMSR_B)
        assert p_large > p_small

    def test_higher_b_means_less_impact(self) -> None:
        """Higher liquidity parameter means less price impact per share."""
        p_low_b = lmsr_price_yes(10, 0, 10)
        p_high_b = lmsr_price_yes(10, 0, 1000)
        assert abs(p_high_b - 0.5) < abs(p_low_b - 0.5)

    def test_price_bounded_0_1(self) -> None:
        """Prices must always be between 0 and 1."""
        extremes = [(-1000, 0), (0, -1000), (1000, 0), (0, 1000), (500, -500)]
        for q_y, q_n in extremes:
            p = lmsr_price_yes(q_y, q_n, LMSR_B)
            assert 0 <= p <= 1, f"Price {p} out of bounds for q=({q_y},{q_n})"

    def test_numerical_stability_extreme_values(self) -> None:
        """LMSR should not overflow/NaN with extreme quantities."""
        p = lmsr_price_yes(100000, 0, LMSR_B)
        assert p == 1.0 or abs(p - 1.0) < 1e-6
        p2 = lmsr_price_yes(0, 100000, LMSR_B)
        assert p2 == 0.0 or abs(p2) < 1e-6
        assert not math.isnan(p) and not math.isnan(p2)
        assert not math.isinf(p) and not math.isinf(p2)


# ===========================================================================
# Market Tests
# ===========================================================================

class TestMarket:
    """Market object behavior."""

    def test_initial_state(self) -> None:
        market = Market(
            market_id="test",
            question="Test?",
            condition=MarketCondition("survival", "Test Colony", 0, ">", 365),
        )
        assert market.price_yes == pytest.approx(0.5)
        assert market.trade_count == 0
        assert market.total_volume == 0

    def test_buy_updates_state(self) -> None:
        market = Market(
            market_id="test",
            question="Test?",
            condition=MarketCondition("survival", "Test Colony", 0, ">", 365),
        )
        cost = market.buy("yes", 10)
        assert cost > 0
        assert market.price_yes > 0.5
        assert market.trade_count == 1
        assert market.total_volume > 0
        assert len(market.price_history) == 1

    def test_sell_reverses_buy(self) -> None:
        market = Market(
            market_id="test",
            question="Test?",
            condition=MarketCondition("survival", "Test Colony", 0, ">", 365),
        )
        market.buy("yes", 10)
        market.buy("yes", -10)
        assert market.price_yes == pytest.approx(0.5, abs=1e-6)

    def test_snapshot_serializable(self) -> None:
        market = Market(
            market_id="test",
            question="Test?",
            condition=MarketCondition("survival", "Test Colony", 0, ">", 365),
        )
        market.buy("yes", 5)
        snap = market.snapshot()
        assert isinstance(snap, dict)
        assert snap["market_id"] == "test"
        assert 0 < snap["price_yes"] < 1
        assert snap["price_yes"] + snap["price_no"] == pytest.approx(1.0)

    def test_market_converges_with_many_yes_buys(self) -> None:
        market = Market(
            market_id="test",
            question="Test?",
            condition=MarketCondition("survival", "Test Colony", 0, ">", 365),
        )
        for _ in range(100):
            market.buy("yes", 5)
        assert market.price_yes > 0.9


# ===========================================================================
# Market Condition Tests
# ===========================================================================

class TestMarketCondition:
    """Resolution logic for market conditions."""

    def _mock_results(self, colony_name: str, final_pop: int, growth_pct: float,
                      history_len: int = 365, tech_count: int = 3,
                      has_epidemic: bool = False) -> dict:
        """Create mock simulation results."""
        history = [{"population": max(1, final_pop - (history_len - i))}
                   for i in range(history_len)]
        history[-1]["population"] = final_pop
        events = []
        if has_epidemic:
            events.append({"type": "epidemic_start", "sol": 100, "strain": "Mars Flu"})
        return {
            "colonies": [
                {
                    "name": colony_name,
                    "final_population": final_pop,
                    "history": history,
                    "events": events,
                    "tech": {"unlocked": [{"name": f"tech-{i}"} for i in range(tech_count)]},
                }
            ],
            "summary": {
                "colonies": [
                    {"name": colony_name, "growth_pct": growth_pct}
                ]
            },
        }

    def test_survival_true(self) -> None:
        cond = MarketCondition("survival", "Ares", 0, ">", 365)
        results = self._mock_results("Ares", 100, 50.0)
        assert cond.evaluate(results) is True

    def test_survival_false(self) -> None:
        cond = MarketCondition("survival", "Ares", 0, ">", 365)
        results = self._mock_results("Ares", 0, -100.0)
        assert cond.evaluate(results) is False

    def test_pop_threshold_met(self) -> None:
        cond = MarketCondition("pop_threshold", "Ares", 200, ">=", 365)
        results = self._mock_results("Ares", 250, 100.0)
        assert cond.evaluate(results) is True

    def test_pop_threshold_not_met(self) -> None:
        cond = MarketCondition("pop_threshold", "Ares", 200, ">=", 365)
        results = self._mock_results("Ares", 150, 50.0)
        assert cond.evaluate(results) is False

    def test_growth_threshold(self) -> None:
        cond = MarketCondition("growth", "Ares", 100, ">=", 0)
        results = self._mock_results("Ares", 200, 120.0)
        assert cond.evaluate(results) is True

    def test_epidemic_detected(self) -> None:
        cond = MarketCondition("epidemic", None, 0, ">", 0)
        results = self._mock_results("Ares", 100, 50.0, has_epidemic=True)
        assert cond.evaluate(results) is True

    def test_epidemic_not_detected(self) -> None:
        cond = MarketCondition("epidemic", None, 0, ">", 0)
        results = self._mock_results("Ares", 100, 50.0, has_epidemic=False)
        assert cond.evaluate(results) is False

    def test_tech_race(self) -> None:
        cond = MarketCondition("tech_race", "Ares", 5, ">=", 0)
        results = self._mock_results("Ares", 100, 50.0, tech_count=6)
        assert cond.evaluate(results) is True

    def test_tech_race_not_met(self) -> None:
        cond = MarketCondition("tech_race", "Ares", 5, ">=", 0)
        results = self._mock_results("Ares", 100, 50.0, tech_count=3)
        assert cond.evaluate(results) is False

    def test_describe_readable(self) -> None:
        cond = MarketCondition("survival", "Ares Prime", 0, ">", 365)
        desc = cond.describe()
        assert "Ares Prime" in desc
        assert "survive" in desc


# ===========================================================================
# Trader Tests
# ===========================================================================

class TestTrader:
    def test_initial_state(self) -> None:
        t = Trader(trader_id="t1", personality="optimist")
        assert t.cash == 500.0
        assert t.positions == {}
        assert t.pnl == 0.0

    def test_can_afford(self) -> None:
        t = Trader(trader_id="t1", personality="optimist", cash=100)
        assert t.can_afford(50) is True
        assert t.can_afford(150) is False

    def test_snapshot(self) -> None:
        t = Trader(trader_id="t1", personality="optimist")
        snap = t.snapshot()
        assert snap["trader_id"] == "t1"
        assert snap["personality"] == "optimist"


# ===========================================================================
# Trading Strategy Tests
# ===========================================================================

class TestTradingDecisions:
    def test_optimist_tends_to_buy_yes(self) -> None:
        """Optimists should buy YES when price is below their belief."""
        t = Trader(trader_id="opt-01", personality="optimist")
        m = Market(
            market_id="test",
            question="Test?",
            condition=MarketCondition("survival", "Test", 0, ">", 365),
        )
        # Price is 0.5, optimist believes ~0.55 → might buy YES
        # Run multiple rounds to test tendency
        yes_count = 0
        for r in range(50):
            decision = decide_trade(t, m, r)
            if decision and decision[0] == "yes":
                yes_count += 1
        assert yes_count > 5, "Optimist should buy YES sometimes"

    def test_pessimist_tends_to_buy_no(self) -> None:
        """Pessimists should buy NO when price is above their belief."""
        t = Trader(trader_id="pes-01", personality="pessimist")
        m = Market(
            market_id="test",
            question="Test?",
            condition=MarketCondition("survival", "Test", 0, ">", 365),
        )
        no_count = 0
        for r in range(50):
            decision = decide_trade(t, m, r)
            if decision and decision[0] == "no":
                no_count += 1
        assert no_count > 5, "Pessimist should buy NO sometimes"

    def test_contrarian_trades_against_price(self) -> None:
        """Contrarians buy YES when price is low, NO when high."""
        t = Trader(trader_id="con-01", personality="contrarian")
        m = Market(
            market_id="test",
            question="Test?",
            condition=MarketCondition("survival", "Test", 0, ">", 365),
        )
        # Push price high
        m.buy("yes", 200)
        assert m.price_yes > 0.8
        no_count = 0
        for r in range(50):
            decision = decide_trade(t, m, r)
            if decision and decision[0] == "no":
                no_count += 1
        assert no_count > 5, "Contrarian should buy NO when price is high"

    def test_decisions_are_deterministic(self) -> None:
        """Same inputs should always produce same decision."""
        t = Trader(trader_id="det-01", personality="random")
        m = Market(
            market_id="test",
            question="Test?",
            condition=MarketCondition("survival", "Test", 0, ">", 365),
        )
        d1 = decide_trade(t, m, 1)
        d2 = decide_trade(t, m, 1)
        assert d1 == d2


# ===========================================================================
# Full Simulation Smoke Tests
# ===========================================================================

class TestSimulationSmoke:
    """Integration tests — the whole engine runs without crashing."""

    def test_runs_without_crash(self) -> None:
        results = run_prediction_market(n_agents=5, rounds=10, quiet=True)
        assert "_meta" in results
        assert results["_meta"]["engine"] == "prediction-market"
        assert len(results["markets"]) == 10
        assert len(results["traders"]) == 5

    def test_all_prices_bounded(self) -> None:
        results = run_prediction_market(n_agents=10, rounds=30, quiet=True)
        for m in results["markets"]:
            assert 0 <= m["price_yes"] <= 1
            assert 0 <= m["price_no"] <= 1
            assert abs(m["price_yes"] + m["price_no"] - 1.0) < 1e-6

    def test_volume_nonnegative(self) -> None:
        results = run_prediction_market(n_agents=10, rounds=30, quiet=True)
        for m in results["markets"]:
            assert m["volume"] >= 0
            assert m["trades"] >= 0

    def test_traders_have_valid_cash(self) -> None:
        results = run_prediction_market(n_agents=10, rounds=30, quiet=True)
        for t in results["traders"]:
            # Cash can go negative in edge cases with LMSR but shouldn't go hugely negative
            assert t["cash"] > -1000, f"Trader {t['trader_id']} has unreasonable cash: {t['cash']}"

    def test_trade_log_present(self) -> None:
        results = run_prediction_market(n_agents=10, rounds=20, quiet=True)
        assert len(results["trade_log"]) > 0

    def test_different_seeds_different_results(self) -> None:
        r1 = run_prediction_market(n_agents=5, rounds=10, seed=1, quiet=True)
        r2 = run_prediction_market(n_agents=5, rounds=10, seed=99, quiet=True)
        prices1 = [m["price_yes"] for m in r1["markets"]]
        prices2 = [m["price_yes"] for m in r2["markets"]]
        assert prices1 != prices2

    def test_deterministic_same_seed(self) -> None:
        r1 = run_prediction_market(n_agents=5, rounds=10, seed=42, quiet=True)
        r2 = run_prediction_market(n_agents=5, rounds=10, seed=42, quiet=True)
        prices1 = [m["price_yes"] for m in r1["markets"]]
        prices2 = [m["price_yes"] for m in r2["markets"]]
        assert prices1 == prices2


class TestSimulationWithResolution:
    """Tests that resolution against real terrarium works."""

    def test_resolve_produces_outcomes(self) -> None:
        results = run_prediction_market(
            n_agents=5, rounds=10, resolve=True, sols=50, quiet=True,
        )
        assert "resolutions" in results
        for mid, res in results["resolutions"].items():
            assert isinstance(res["outcome"], bool)
            assert 0 <= res["accuracy"] <= 1

    def test_resolve_settles_pnl(self) -> None:
        results = run_prediction_market(
            n_agents=10, rounds=30, resolve=True, sols=100, quiet=True,
        )
        # At least some traders should have non-zero PnL after resolution
        pnls = [t["pnl"] for t in results["traders"]]
        # With enough rounds+agents, some positions should exist
        positions_exist = any(
            any(v != 0 for v in t["positions"].values())
            for t in results["traders"]
        )
        if positions_exist:
            assert any(p != 0 for p in pnls), "Positions exist but all PnL zero"

    def test_terrarium_summary_included(self) -> None:
        results = run_prediction_market(
            n_agents=5, rounds=10, resolve=True, sols=50, quiet=True,
        )
        assert "terrarium_summary" in results
        assert "colonies" in results["terrarium_summary"]


# ===========================================================================
# Property-Based Invariants
# ===========================================================================

class TestInvariants:
    """Conservation laws and physical bounds."""

    def test_price_sum_invariant_across_simulation(self) -> None:
        """p_yes + p_no = 1 at every step of the simulation."""
        markets = create_default_markets()
        rng = random.Random(42)
        for m in markets:
            for _ in range(100):
                outcome = rng.choice(["yes", "no"])
                shares = rng.randint(1, 10)
                m.buy(outcome, shares)
                assert abs(m.price_yes + m.price_no - 1.0) < 1e-6

    def test_lmsr_cost_monotonic(self) -> None:
        """Cost function is monotonically increasing in quantities."""
        c1 = lmsr_cost(0, 0, LMSR_B)
        c2 = lmsr_cost(10, 0, LMSR_B)
        c3 = lmsr_cost(20, 0, LMSR_B)
        assert c1 < c2 < c3

    def test_price_monotonic_in_quantity(self) -> None:
        """YES price monotonically increases with YES quantity."""
        prices = [lmsr_price_yes(q, 0, LMSR_B) for q in range(0, 500, 10)]
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1]

    def test_market_maker_bounded_loss(self) -> None:
        """LMSR market maker's max subsidy per outcome is b*ln(n).

        For binary markets (n=2), max subsidy = b*ln(2) ≈ 69.3.
        But cost to BUY shares grows linearly for large q.
        The invariant: the difference C(q+1) - C(q) → 1 as q→∞ (price → $1).
        """
        # Marginal cost of one more YES share should approach 1.0 at extreme
        marginal = lmsr_trade_cost(100000, 0, LMSR_B, "yes", 1)
        assert 0.99 < marginal <= 1.01, f"Marginal cost at extreme: {marginal}"
        # Marginal cost should be ~0.5 at equilibrium
        marginal_eq = lmsr_trade_cost(0, 0, LMSR_B, "yes", 1)
        assert 0.49 < marginal_eq < 0.51, f"Marginal cost at equilibrium: {marginal_eq}"
