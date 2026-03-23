"""
Tests for market_maker.py -- LMSR prediction market engine.

Covers: LMSR math, market creation, trading, resolution, scoring,
full pipeline, conservation laws.

Run: python -m pytest tests/test_market_maker.py -v
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
    lmsr_cost,
    lmsr_prices,
    lmsr_trade_cost,
    Market,
    Trade,
    Trader,
    TRADER_PROFILES,
    DEFAULT_LIQUIDITY,
    create_colony_markets,
    simulate_trading,
    resolve_markets,
    score_traders,
    run_prediction_market,
)


COLONY_NAMES = ["Ares Prime", "Olympus Station", "Red Frontier"]
STRATEGIES = ["conservative", "balanced", "aggressive"]


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
                "events": [
                    {"sol": 100, "type": "epidemic_start", "strain": "Mars Flu"},
                ],
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
                {"name": "Ares Prime", "strategy": "conservative",
                 "start_pop": 120, "end_pop": 200, "growth_pct": 66.7,
                 "peak_pop": 200, "min_pop": 120,
                 "total_births": 100, "total_deaths": 20,
                 "techs_unlocked": 4},
                {"name": "Olympus Station", "strategy": "balanced",
                 "start_pop": 80, "end_pop": 130, "growth_pct": 62.5,
                 "peak_pop": 130, "min_pop": 80,
                 "total_births": 60, "total_deaths": 10,
                 "techs_unlocked": 3},
                {"name": "Red Frontier", "strategy": "aggressive",
                 "start_pop": 60, "end_pop": 150, "growth_pct": 150.0,
                 "peak_pop": 150, "min_pop": 60,
                 "total_births": 100, "total_deaths": 10,
                 "techs_unlocked": 5},
            ],
            "total_migrations": 15,
        },
    }


# ---------------------------------------------------------------------------
# LMSR Math
# ---------------------------------------------------------------------------

class TestLMSRPrices:
    def test_binary_sum_to_one(self):
        """Prices sum to 1.0 for binary markets."""
        prices = lmsr_prices([0.0, 0.0])
        assert abs(sum(prices) - 1.0) < 1e-10

    def test_ternary_sum_to_one(self):
        """Prices sum to 1.0 for 3-outcome markets."""
        prices = lmsr_prices([0.0, 0.0, 0.0])
        assert abs(sum(prices) - 1.0) < 1e-10

    def test_equal_quantities_equal_prices(self):
        """Equal q -> equal prices."""
        prices = lmsr_prices([0.0, 0.0])
        assert abs(prices[0] - 0.5) < 1e-10

    def test_higher_quantity_higher_price(self):
        """More shares -> higher price."""
        prices = lmsr_prices([100.0, 0.0])
        assert prices[0] > prices[1]

    def test_bounded_zero_one(self):
        """All prices in (0, 1)."""
        for _ in range(100):
            q = [random.uniform(-200, 200) for _ in range(2)]
            for p in lmsr_prices(q):
                assert 0 < p < 1

    def test_random_sum_property(self):
        """Sum to 1 for random quantities."""
        for _ in range(100):
            q = [random.uniform(-50, 50) for _ in range(2)]
            assert abs(sum(lmsr_prices(q)) - 1.0) < 1e-10


class TestLMSRCost:
    def test_increases_with_quantity(self):
        """More shares -> higher cost."""
        c1 = lmsr_cost([0.0, 0.0])
        c2 = lmsr_cost([50.0, 0.0])
        assert c2 > c1

    def test_symmetric(self):
        """C([a,b]) == C([b,a]) (order shouldn't matter for total cost)."""
        c1 = lmsr_cost([10.0, 20.0])
        c2 = lmsr_cost([20.0, 10.0])
        assert abs(c1 - c2) < 1e-10

    def test_positive(self):
        """Cost always positive for non-negative quantities."""
        for _ in range(50):
            q = [random.uniform(0, 100) for _ in range(2)]
            assert lmsr_cost(q) > 0


class TestLMSRTradeCost:
    def test_buy_costs_positive(self):
        """Buying shares costs money."""
        cost = lmsr_trade_cost([0.0, 0.0], 0, 10.0)
        assert cost > 0

    def test_small_cheaper_than_large(self):
        """Buying fewer shares costs less."""
        c_small = lmsr_trade_cost([0.0, 0.0], 0, 5.0)
        c_large = lmsr_trade_cost([0.0, 0.0], 0, 50.0)
        assert c_small < c_large

    def test_price_impact(self):
        """After buying, price of that outcome increases."""
        q = [0.0, 0.0]
        before = lmsr_prices(q)[0]
        q[0] += 20
        after = lmsr_prices(q)[0]
        assert after > before


# ---------------------------------------------------------------------------
# Market creation
# ---------------------------------------------------------------------------

class TestCreateMarkets:
    def test_creates_markets(self):
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        assert len(markets) > 0

    def test_unique_ids(self):
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        ids = [m.market_id for m in markets]
        assert len(ids) == len(set(ids))

    def test_initial_fair_prices(self):
        """Binary markets start at 50/50."""
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        for m in markets:
            if len(m.outcomes) == 2:
                prices = m.prices()
                assert abs(prices[0] - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------

class TestTrading:
    def test_trades_execute(self):
        m = Market(
            market_id="test", question="Test?",
            outcomes=["Yes", "No"], quantities=[0.0, 0.0],
            liquidity=100.0,
        )
        trade = m.buy("trader-1", 0, 10.0)
        assert trade.shares == 10.0
        assert trade.cost > 0

    def test_price_moves(self):
        m = Market(
            market_id="test", question="Test?",
            outcomes=["Yes", "No"], quantities=[0.0, 0.0],
            liquidity=100.0,
        )
        before = m.prices()[0]
        m.buy("trader-1", 0, 20.0)
        after = m.prices()[0]
        assert after > before

    def test_simulate_returns_traders(self):
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        traders = simulate_trading(markets, None, rounds=2)
        assert len(traders) > 0

    def test_trades_recorded(self):
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        simulate_trading(markets, None, rounds=2)
        total = sum(len(m.trades) for m in markets)
        assert total > 0


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestResolution:
    def test_all_resolved(self, sim_results):
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        resolutions = resolve_markets(markets, sim_results)
        assert len(resolutions) > 0

    def test_survival_yes(self, sim_results):
        """All colonies survive in fixture -> Yes."""
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        resolutions = resolve_markets(markets, sim_results)
        for mid, result in resolutions.items():
            if "survival" in mid:
                assert result == "Yes"

    def test_pnl_after_resolution(self, sim_results):
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        traders = simulate_trading(markets, None, rounds=2)
        resolve_markets(markets, sim_results)
        scores = score_traders(markets, traders)
        assert any(s["total_pnl"] != 0.0 for s in scores)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_scores_have_fields(self, sim_results):
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        traders = simulate_trading(markets, None, rounds=2)
        resolve_markets(markets, sim_results)
        scores = score_traders(markets, traders)
        for s in scores:
            assert "trader_id" in s
            assert "total_pnl" in s
            assert "roi_pct" in s

    def test_sorted_by_pnl(self, sim_results):
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        traders = simulate_trading(markets, None, rounds=2)
        resolve_markets(markets, sim_results)
        scores = score_traders(markets, traders)
        pnls = [s["total_pnl"] for s in scores]
        assert pnls == sorted(pnls, reverse=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_runs(self, sim_results):
        result = run_prediction_market(sim_results, verbose=False)
        assert result["_meta"]["engine"] == "prediction-market"
        assert result["_meta"]["num_markets"] > 0

    def test_deterministic(self, sim_results):
        r1 = run_prediction_market(sim_results, verbose=False)
        r2 = run_prediction_market(sim_results, verbose=False)
        assert r1["_meta"]["num_markets"] == r2["_meta"]["num_markets"]

    def test_serializable(self, sim_results):
        result = run_prediction_market(sim_results, verbose=False)
        s = json.dumps(result)
        parsed = json.loads(s)
        assert parsed["_meta"]["engine"] == "prediction-market"


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------

class TestConservation:
    def test_prices_sum_after_trading(self, sim_results):
        """After trading, prices still sum to 1."""
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        simulate_trading(markets, None, rounds=5)
        for m in markets:
            assert abs(sum(m.prices()) - 1.0) < 1e-8

    def test_prices_bounded_after_trading(self, sim_results):
        """After trading, each price in (0, 1)."""
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        simulate_trading(markets, None, rounds=5)
        for m in markets:
            for p in m.prices():
                assert 0 < p < 1

    def test_pnl_zero_when_unresolved(self):
        """Before resolution, all PnL is 0."""
        m = Market(
            market_id="test", question="Test?",
            outcomes=["Yes", "No"], quantities=[0.0, 0.0],
            liquidity=100.0,
        )
        m.buy("trader-1", 0, 10.0)
        assert m.pnl("trader-1") == 0.0

    def test_snapshot_serializable(self, sim_results):
        markets = create_colony_markets(COLONY_NAMES, STRATEGIES)
        simulate_trading(markets, None, rounds=2)
        for m in markets:
            s = json.dumps(m.snapshot())
            assert json.loads(s)["market_id"] == m.market_id


# ---------------------------------------------------------------------------
# Live smoke test
# ---------------------------------------------------------------------------

class TestLiveSmoke:
    def test_with_real_sim(self):
        """Run actual sim + market pipeline (10 sols)."""
        from src.tick_engine import Simulation
        sim = Simulation(sols=10, env_seed=42)
        results = sim.run()
        output = run_prediction_market(results, verbose=False, trading_rounds=2)
        assert output["_meta"]["num_markets"] > 0
        assert len(output["traders"]) > 0
