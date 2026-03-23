"""
Tests for market_maker.py -- LMSR prediction market engine.
Run: python -m pytest tests/test_market_maker.py -v
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.market_maker import (
    lmsr_cost, lmsr_prices, lmsr_trade_cost,
    Trade, Market, Trader,
    create_colony_markets, simulate_trading,
    resolve_markets, score_traders, run_prediction_market,
    TRADER_PROFILES, DEFAULT_LIQUIDITY,
)


class TestLMSRCost:
    def test_zero_quantities(self):
        cost = lmsr_cost([0.0, 0.0])
        assert cost > 0

    def test_increases_with_quantity(self):
        c1 = lmsr_cost([10.0, 0.0])
        c2 = lmsr_cost([20.0, 0.0])
        assert c2 > c1

    def test_symmetric(self):
        c1 = lmsr_cost([10.0, 5.0])
        c2 = lmsr_cost([5.0, 10.0])
        assert c1 == pytest.approx(c2)


class TestLMSRPrices:
    def test_sum_to_one(self):
        prices = lmsr_prices([0.0, 0.0])
        assert sum(prices) == pytest.approx(1.0)

    def test_equal_quantities_equal_prices(self):
        prices = lmsr_prices([0.0, 0.0])
        assert prices[0] == pytest.approx(0.5)

    def test_higher_quantity_higher_price(self):
        prices = lmsr_prices([100.0, 0.0])
        assert prices[0] > prices[1]

    def test_three_outcomes_sum(self):
        prices = lmsr_prices([10.0, 20.0, 30.0])
        assert sum(prices) == pytest.approx(1.0)

    def test_prices_in_01(self):
        rng = random.Random(42)
        for _ in range(20):
            qs = [rng.uniform(-50, 50) for _ in range(3)]
            for p in lmsr_prices(qs):
                assert 0.0 < p < 1.0


class TestLMSRTradeCost:
    def test_positive_cost_for_buy(self):
        cost = lmsr_trade_cost([0.0, 0.0], 0, 10.0)
        assert cost > 0

    def test_higher_shares_higher_cost(self):
        c1 = lmsr_trade_cost([0.0, 0.0], 0, 5.0)
        c2 = lmsr_trade_cost([0.0, 0.0], 0, 15.0)
        assert c2 > c1

    def test_bounded_by_shares(self):
        cost = lmsr_trade_cost([0.0, 0.0], 0, 10.0)
        assert cost <= 10.0


class TestMarket:
    def test_initial_balanced(self):
        m = Market("t", "Q?", ["Y", "N"], [0.0, 0.0], DEFAULT_LIQUIDITY)
        assert m.prices()[0] == pytest.approx(0.5)

    def test_buy_moves_price(self):
        m = Market("t", "Q?", ["Y", "N"], [0.0, 0.0], DEFAULT_LIQUIDITY)
        m.buy(0, 10.0, "t1")
        assert m.prices()[0] > 0.5

    def test_buy_records_trade(self):
        m = Market("t", "Q?", ["Y", "N"], [0.0, 0.0], DEFAULT_LIQUIDITY)
        trade = m.buy(0, 5.0, "t1")
        assert len(m.trades) == 1
        assert trade.trader_id == "t1"

    def test_pnl_winning(self):
        m = Market("t", "Q?", ["Y", "N"], [0.0, 0.0], DEFAULT_LIQUIDITY)
        m.buy(0, 10.0, "winner")
        m.resolve(0)
        assert m.pnl("winner") > 0

    def test_pnl_losing(self):
        m = Market("t", "Q?", ["Y", "N"], [0.0, 0.0], DEFAULT_LIQUIDITY)
        m.buy(1, 10.0, "loser")
        m.resolve(0)
        assert m.pnl("loser") < 0

    def test_pnl_unresolved_zero(self):
        m = Market("t", "Q?", ["Y", "N"], [0.0, 0.0], DEFAULT_LIQUIDITY)
        m.buy(0, 10.0, "t1")
        assert m.pnl("t1") == 0.0

    def test_snapshot_serializable(self):
        m = Market("t", "Q?", ["Y", "N"], [0.0, 0.0], DEFAULT_LIQUIDITY)
        m.buy(0, 5.0, "t1")
        assert json.dumps(m.snapshot())


class TestCreateMarkets:
    def test_creates_16(self):
        ms = create_colony_markets(
            ["Ares Prime", "Olympus Station", "Red Frontier"],
            ["conservative", "balanced", "aggressive"],
        )
        assert len(ms) == 16

    def test_unique_ids(self):
        ms = create_colony_markets(["A", "B"], ["balanced", "balanced"])
        ids = [m.market_id for m in ms]
        assert len(ids) == len(set(ids))


class TestTrader:
    def test_can_afford(self):
        t = Trader("t1", "noise", 100.0)
        assert t.can_afford(50.0)
        assert not t.can_afford(101.0)

    def test_record_spend(self):
        t = Trader("t1", "noise", 100.0)
        t.record_spend(30.0)
        assert t.spent == 30.0


class TestSimulateTrading:
    def test_trades_occur(self):
        ms = create_colony_markets(
            ["Ares Prime", "Olympus Station", "Red Frontier"],
            ["conservative", "balanced", "aggressive"],
        )
        simulate_trading(ms, rounds=3)
        assert sum(len(m.trades) for m in ms) > 0

    def test_deterministic(self):
        m1 = create_colony_markets(["A", "B"], ["b", "b"])
        m2 = create_colony_markets(["A", "B"], ["b", "b"])
        simulate_trading(m1, rounds=3)
        simulate_trading(m2, rounds=3)
        for a, b in zip(m1, m2):
            assert a.prices() == b.prices()


class TestResolution:
    @pytest.fixture
    def sim_results(self):
        return {
            "colonies": [
                {"name": "Ares Prime", "strategy": "conservative",
                 "initial_population": 120, "final_population": 180,
                 "tech": {"unlocked_count": 4, "unlocked": []},
                 "events": [], "history": []},
                {"name": "Olympus Station", "strategy": "balanced",
                 "initial_population": 80, "final_population": 140,
                 "tech": {"unlocked_count": 4, "unlocked": []},
                 "events": [{"type": "epidemic_start"}], "history": []},
                {"name": "Red Frontier", "strategy": "aggressive",
                 "initial_population": 60, "final_population": 143,
                 "tech": {"unlocked_count": 4, "unlocked": []},
                 "events": [], "history": []},
            ],
            "summary": {"colonies": [
                {"name": "Ares Prime", "end_pop": 180, "growth_pct": 50.0},
                {"name": "Olympus Station", "end_pop": 140, "growth_pct": 75.0},
                {"name": "Red Frontier", "end_pop": 143, "growth_pct": 138.3},
            ]},
        }

    def test_survival(self, sim_results):
        ms = create_colony_markets(
            ["Ares Prime", "Olympus Station", "Red Frontier"],
            ["conservative", "balanced", "aggressive"],
        )
        res = resolve_markets(ms, sim_results)
        assert res["survival-ares-prime"] == "Yes"

    def test_pop_threshold(self, sim_results):
        ms = create_colony_markets(
            ["Ares Prime", "Olympus Station", "Red Frontier"],
            ["conservative", "balanced", "aggressive"],
        )
        res = resolve_markets(ms, sim_results)
        assert res["pop-ares-prime-150"] == "Yes"
        assert res["pop-ares-prime-200"] == "No"

    def test_fastest_grower(self, sim_results):
        ms = create_colony_markets(
            ["Ares Prime", "Olympus Station", "Red Frontier"],
            ["conservative", "balanced", "aggressive"],
        )
        res = resolve_markets(ms, sim_results)
        assert res["fastest-grower"] == "Red Frontier"

    def test_all_resolved(self, sim_results):
        ms = create_colony_markets(
            ["Ares Prime", "Olympus Station", "Red Frontier"],
            ["conservative", "balanced", "aggressive"],
        )
        res = resolve_markets(ms, sim_results)
        assert len(res) == len(ms)


class TestScoring:
    def test_scores_sorted(self):
        ms = create_colony_markets(["A", "B"], ["b", "b"])
        traders = simulate_trading(ms, rounds=3)
        for m in ms:
            m.resolve(0)
        scores = score_traders(ms, traders)
        pnls = [s["total_pnl"] for s in scores]
        assert pnls == sorted(pnls, reverse=True)


class TestFullPipeline:
    def test_run_pipeline(self):
        sim_results = {
            "colonies": [
                {"name": "Ares Prime", "strategy": "conservative",
                 "initial_population": 120, "final_population": 180,
                 "tech": {"unlocked_count": 4, "unlocked": []},
                 "events": [], "history": []},
                {"name": "Olympus Station", "strategy": "balanced",
                 "initial_population": 80, "final_population": 140,
                 "tech": {"unlocked_count": 4, "unlocked": []},
                 "events": [], "history": []},
                {"name": "Red Frontier", "strategy": "aggressive",
                 "initial_population": 60, "final_population": 143,
                 "tech": {"unlocked_count": 4, "unlocked": []},
                 "events": [], "history": []},
            ],
            "summary": {"colonies": [
                {"name": "Ares Prime", "end_pop": 180, "growth_pct": 50.0},
                {"name": "Olympus Station", "end_pop": 140, "growth_pct": 75.0},
                {"name": "Red Frontier", "end_pop": 143, "growth_pct": 138.3},
            ]},
        }
        result = run_prediction_market(sim_results, verbose=False)
        assert "_meta" in result
        assert len(result["markets"]) == 16
        assert len(result["traders"]) == len(TRADER_PROFILES)
        assert len(result["resolutions"]) == 16

    def test_serializable(self):
        sr = {
            "colonies": [{"name": "A", "strategy": "balanced",
                 "initial_population": 100, "final_population": 150,
                 "tech": {"unlocked_count": 2, "unlocked": []},
                 "events": [], "history": []}],
            "summary": {"colonies": [{"name": "A", "end_pop": 150, "growth_pct": 50.0}]},
        }
        result = run_prediction_market(sr, verbose=False)
        assert len(json.dumps(result)) > 100


class TestInvariants:
    def test_prices_sum_to_one(self):
        rng = random.Random(42)
        for _ in range(50):
            n = rng.randint(2, 5)
            qs = [rng.uniform(-100, 100) for _ in range(n)]
            assert sum(lmsr_prices(qs)) == pytest.approx(1.0, abs=1e-10)

    def test_prices_positive(self):
        rng = random.Random(42)
        for _ in range(50):
            qs = [rng.uniform(-100, 100) for _ in range(3)]
            for p in lmsr_prices(qs):
                assert p > 0

    def test_buying_increases_price(self):
        m = Market("t", "?", ["A", "B", "C"], [0.0, 0.0, 0.0], DEFAULT_LIQUIDITY)
        before = m.prices()[0]
        m.buy(0, 20.0, "t1")
        assert m.prices()[0] > before
