"""
Tests for market_maker.py — LMSR prediction market engine.

Covers: LMSR math, Market class, market factory, trading,
resolution, scoring, full pipeline, property-based invariants.
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
    lmsr_cost, lmsr_prices, lmsr_trade_cost,
    Market, Trade, Trader,
    create_colony_markets, simulate_trading,
    resolve_markets, score_traders, run_prediction_market,
    TRADER_PROFILES, DEFAULT_LIQUIDITY,
)

COLONY_NAMES = ["Ares Prime", "Olympus Station", "Red Frontier"]
STRATEGIES = ["conservative", "balanced", "aggressive"]


@pytest.fixture
def sim_results():
    """Realistic sim results for resolution."""
    return {
        "colonies": [
            {
                "name": "Ares Prime", "strategy": "conservative",
                "initial_population": 120, "final_population": 200,
                "tech": {"unlocked_count": 4, "unlocked": []},
                "events": [{"sol": 100, "type": "epidemic_start"}],
                "history": [],
            },
            {
                "name": "Olympus Station", "strategy": "balanced",
                "initial_population": 80, "final_population": 130,
                "tech": {"unlocked_count": 3, "unlocked": []},
                "events": [],
                "history": [],
            },
            {
                "name": "Red Frontier", "strategy": "aggressive",
                "initial_population": 60, "final_population": 150,
                "tech": {"unlocked_count": 5, "unlocked": []},
                "events": [],
                "history": [],
            },
        ],
        "summary": {
            "colonies": [
                {"name": "Ares Prime", "start_pop": 120, "end_pop": 200,
                 "growth_pct": 66.7, "peak_pop": 200, "total_births": 100,
                 "total_deaths": 20, "techs_unlocked": 4},
                {"name": "Olympus Station", "start_pop": 80, "end_pop": 130,
                 "growth_pct": 62.5, "peak_pop": 130, "total_births": 60,
                 "total_deaths": 10, "techs_unlocked": 3},
                {"name": "Red Frontier", "start_pop": 60, "end_pop": 150,
                 "growth_pct": 150.0, "peak_pop": 150, "total_births": 100,
                 "total_deaths": 10, "techs_unlocked": 5},
            ],
            "total_migrations": 15,
        },
    }


@pytest.fixture
def markets():
    return create_colony_markets(COLONY_NAMES, STRATEGIES)


# --- LMSR math ---

class TestLMSR:
    def test_initial_prices_uniform(self):
        p = lmsr_prices([0.0, 0.0])
        assert abs(p[0] - 0.5) < 0.001
        assert abs(p[1] - 0.5) < 0.001

    def test_prices_sum_to_one(self):
        rng = random.Random(42)
        for _ in range(100):
            n = rng.randint(2, 5)
            q = [rng.uniform(-50, 50) for _ in range(n)]
            total = sum(lmsr_prices(q))
            assert abs(total - 1.0) < 1e-6

    def test_trade_cost_positive(self):
        cost = lmsr_trade_cost([0.0, 0.0], 0, 10.0)
        assert cost > 0

    def test_cost_monotonic(self):
        c1 = lmsr_trade_cost([0.0, 0.0], 0, 5.0)
        c2 = lmsr_trade_cost([0.0, 0.0], 0, 10.0)
        assert c2 > c1

    def test_three_outcome_prices(self):
        p = lmsr_prices([0.0, 0.0, 0.0])
        for price in p:
            assert abs(price - 1 / 3) < 0.001


# --- Market class ---

class TestMarket:
    def test_create(self):
        m = Market("t1", "Q?", ["Y", "N"], [0.0, 0.0], 100.0)
        assert m.market_id == "t1"

    def test_buy_moves_price(self):
        m = Market("t2", "Q?", ["Y", "N"], [0.0, 0.0], 100.0)
        before = m.prices()[0]
        m.buy(0, 10.0, "trader-1")
        after = m.prices()[0]
        assert after > before

    def test_buy_records_trade(self):
        m = Market("t3", "Q?", ["Y", "N"], [0.0, 0.0], 100.0)
        trade = m.buy(0, 5.0, "t1")
        assert len(m.trades) == 1
        assert trade.shares == 5.0

    def test_pnl_winner(self):
        m = Market("t4", "Q?", ["Y", "N"], [0.0, 0.0], 100.0)
        m.buy(0, 10.0, "winner")
        m.resolve(0)
        assert m.pnl("winner") > 0

    def test_pnl_loser(self):
        m = Market("t5", "Q?", ["Y", "N"], [0.0, 0.0], 100.0)
        m.buy(1, 10.0, "loser")
        m.resolve(0)
        assert m.pnl("loser") < 0

    def test_pnl_unresolved_zero(self):
        m = Market("t6", "Q?", ["Y", "N"], [0.0, 0.0], 100.0)
        m.buy(0, 10.0, "t1")
        assert m.pnl("t1") == 0.0

    def test_snapshot(self):
        m = Market("t7", "Q?", ["Y", "N"], [0.0, 0.0], 100.0)
        m.buy(0, 5.0, "t1")
        snap = m.snapshot()
        assert snap["num_trades"] == 1
        assert not snap["resolved"]


# --- Market factory ---

class TestCreateMarkets:
    def test_count(self, markets):
        assert len(markets) > 10

    def test_survival_markets(self, markets):
        s = [m for m in markets if m.market_id.startswith("survival-")]
        assert len(s) == 3

    def test_pop_thresholds(self, markets):
        p = [m for m in markets if m.market_id.startswith("pop-")]
        assert len(p) == 9

    def test_multi_outcome(self, markets):
        multi = [m for m in markets if len(m.outcomes) > 2]
        assert len(multi) >= 3


# --- Trading ---

class TestTrading:
    def test_traders_created(self, markets):
        traders = simulate_trading(markets, rounds=1)
        assert len(traders) == len(TRADER_PROFILES)

    def test_trades_happen(self, markets):
        simulate_trading(markets, rounds=3)
        total = sum(len(m.trades) for m in markets)
        assert total > 0

    def test_prices_move(self, markets):
        initial = [m.prices()[0] for m in markets]
        simulate_trading(markets, rounds=3)
        final = [m.prices()[0] for m in markets]
        changed = sum(
            1 for i, f in zip(initial, final) if abs(i - f) > 0.001)
        assert changed > 0

    def test_bankroll_constraint(self, markets):
        traders = simulate_trading(markets, rounds=5)
        for t in traders:
            assert t.spent <= t.bankroll + 0.01


# --- Resolution ---

class TestResolve:
    def test_all_resolved(self, markets, sim_results):
        simulate_trading(markets, rounds=2)
        resolve_markets(markets, sim_results)
        for m in markets:
            assert m.resolution is not None

    def test_survival_yes(self, markets, sim_results):
        resolve_markets(markets, sim_results)
        survival = [m for m in markets if m.market_id.startswith("survival-")]
        for m in survival:
            assert m.resolution == 0  # Yes

    def test_fastest_grower(self, markets, sim_results):
        resolve_markets(markets, sim_results)
        fg = next(m for m in markets if m.market_id == "fastest-grower")
        assert fg.resolution == 2  # Red Frontier

    def test_largest_pop(self, markets, sim_results):
        resolve_markets(markets, sim_results)
        lp = next(m for m in markets if m.market_id == "largest-pop")
        assert lp.resolution == 0  # Ares Prime

    def test_epidemic(self, markets, sim_results):
        resolve_markets(markets, sim_results)
        epi = next(m for m in markets if m.market_id == "any-epidemic")
        assert epi.resolution == 0  # Yes


# --- Scoring ---

class TestScoring:
    def test_produces_results(self, markets, sim_results):
        traders = simulate_trading(markets, rounds=3)
        resolve_markets(markets, sim_results)
        scores = score_traders(markets, traders)
        assert len(scores) == len(traders)

    def test_sorted_by_pnl(self, markets, sim_results):
        traders = simulate_trading(markets, rounds=3)
        resolve_markets(markets, sim_results)
        scores = score_traders(markets, traders)
        pnls = [s["total_pnl"] for s in scores]
        assert pnls == sorted(pnls, reverse=True)


# --- Full pipeline ---

class TestPipeline:
    def test_runs(self, sim_results):
        r = run_prediction_market(sim_results, verbose=False, trading_rounds=2)
        assert r["_meta"]["num_markets"] > 0
        assert r["_meta"]["num_traders"] > 0

    def test_serializable(self, sim_results):
        r = run_prediction_market(sim_results, verbose=False)
        s = json.dumps(r)
        assert json.loads(s)["_meta"]["num_markets"] > 0

    def test_with_mc_stats(self, sim_results):
        mc = {
            "colony_names": COLONY_NAMES,
            "colony_strategies": STRATEGIES,
            "survival_rates": [1.0, 1.0, 1.0],
            "final_pop_stats": [
                {"mean": 200.0}, {"mean": 130.0}, {"mean": 150.0}],
            "growth_pct_stats": [
                {"mean": 66.7}, {"mean": 62.5}, {"mean": 150.0}],
        }
        r = run_prediction_market(
            sim_results, mc_stats=mc, verbose=False)
        assert r["_meta"]["num_markets"] > 0


# --- Invariants ---

class TestInvariants:
    def test_prices_always_sum_one(self):
        rng = random.Random(42)
        for _ in range(1000):
            n = rng.randint(2, 5)
            q = [rng.uniform(-100, 100) for _ in range(n)]
            total = sum(lmsr_prices(q))
            assert abs(total - 1.0) < 1e-6

    def test_buy_increases_price(self):
        rng = random.Random(42)
        for _ in range(100):
            q = [rng.uniform(-20, 20), rng.uniform(-20, 20)]
            p_before = lmsr_prices(q)[0]
            q[0] += rng.uniform(1, 20)
            p_after = lmsr_prices(q)[0]
            assert p_after >= p_before - 1e-10

    def test_resolution_deterministic(self, sim_results):
        m1 = create_colony_markets(COLONY_NAMES, STRATEGIES)
        m2 = create_colony_markets(COLONY_NAMES, STRATEGIES)
        r1 = resolve_markets(m1, sim_results)
        r2 = resolve_markets(m2, sim_results)
        assert r1 == r2

    def test_smoke_10_sol_sim(self):
        """Full pipeline with real sim (10 sols)."""
        from src.tick_engine import Simulation
        sim = Simulation(sols=10, env_seed=42)
        sim_results = sim.run()
        r = run_prediction_market(sim_results, verbose=False, trading_rounds=2)
        assert r["_meta"]["num_markets"] > 0
        for m in r["markets"]:
            assert m["resolved"]
