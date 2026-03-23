"""
Tests for market_maker.py — LMSR prediction market engine.

Covers:
  - LMSR math (cost, prices, trade cost)
  - Market creation and trading
  - Resolution and scoring
  - Full pipeline smoke test
  - Property-based invariants (prices sum to 1, costs positive, etc.)
"""
from __future__ import annotations

import math
import pytest
from src.market_maker import (
    lmsr_cost,
    lmsr_prices,
    lmsr_trade_cost,
    Market,
    Trader,
    TraderPosition,
    MarketEngine,
    run_prediction_market,
)


# ---------------------------------------------------------------------------
# LMSR math tests
# ---------------------------------------------------------------------------

class TestLMSRMath:
    """Tests for pure LMSR functions."""

    def test_cost_uniform_quantities(self) -> None:
        """Cost of [0,0] should be b * ln(2)."""
        c = lmsr_cost([0.0, 0.0], b=100.0)
        assert abs(c - 100.0 * math.log(2)) < 1e-9

    def test_cost_three_outcomes(self) -> None:
        """Cost of [0,0,0] should be b * ln(3)."""
        c = lmsr_cost([0.0, 0.0, 0.0], b=100.0)
        assert abs(c - 100.0 * math.log(3)) < 1e-9

    def test_cost_increases_with_quantity(self) -> None:
        """Higher quantities → higher cost."""
        c1 = lmsr_cost([0.0, 0.0], b=100.0)
        c2 = lmsr_cost([50.0, 0.0], b=100.0)
        assert c2 > c1

    def test_cost_invalid_b(self) -> None:
        """b must be positive."""
        with pytest.raises(ValueError):
            lmsr_cost([0.0, 0.0], b=0.0)
        with pytest.raises(ValueError):
            lmsr_cost([0.0, 0.0], b=-1.0)

    def test_prices_uniform(self) -> None:
        """Equal quantities → equal prices."""
        p = lmsr_prices([0.0, 0.0], b=100.0)
        assert len(p) == 2
        assert abs(p[0] - 0.5) < 1e-9
        assert abs(p[1] - 0.5) < 1e-9

    def test_prices_sum_to_one(self) -> None:
        """Prices must always sum to 1.0."""
        for q in [[0, 0], [10, 0], [50, 30], [0, 0, 0], [100, 200, 50]]:
            q_float = [float(x) for x in q]
            p = lmsr_prices(q_float, b=100.0)
            assert abs(sum(p) - 1.0) < 1e-9, f"Prices don't sum to 1: {p}"

    def test_prices_higher_quantity_higher_price(self) -> None:
        """Outcome with more shares should be more expensive."""
        p = lmsr_prices([100.0, 0.0], b=100.0)
        assert p[0] > p[1]

    def test_prices_three_outcomes_uniform(self) -> None:
        """Three equal outcomes → 1/3 each."""
        p = lmsr_prices([0.0, 0.0, 0.0], b=100.0)
        for pi in p:
            assert abs(pi - 1 / 3) < 1e-9

    def test_trade_cost_positive_for_buy(self) -> None:
        """Buying shares should cost positive amount."""
        cost = lmsr_trade_cost([0.0, 0.0], 0, 10.0, b=100.0)
        assert cost > 0

    def test_trade_cost_symmetric(self) -> None:
        """Buying same shares on either side from uniform state costs the same."""
        c0 = lmsr_trade_cost([0.0, 0.0], 0, 10.0, b=100.0)
        c1 = lmsr_trade_cost([0.0, 0.0], 1, 10.0, b=100.0)
        assert abs(c0 - c1) < 1e-9

    def test_trade_cost_increases_with_shares(self) -> None:
        """More shares → higher cost (convexity)."""
        c5 = lmsr_trade_cost([0.0, 0.0], 0, 5.0, b=100.0)
        c10 = lmsr_trade_cost([0.0, 0.0], 0, 10.0, b=100.0)
        assert c10 > c5

    def test_numerical_stability_large_quantities(self) -> None:
        """Should not overflow with large quantities."""
        p = lmsr_prices([1000.0, 0.0], b=100.0)
        assert abs(sum(p) - 1.0) < 1e-6
        assert p[0] > 0.99  # should be very close to 1


# ---------------------------------------------------------------------------
# Market tests
# ---------------------------------------------------------------------------

class TestMarket:
    """Tests for Market dataclass."""

    def test_create_market(self) -> None:
        """Create a simple binary market."""
        mkt = Market(
            market_id="test",
            question="Will it rain?",
            outcomes=["Yes", "No"],
            quantities=[0.0, 0.0],
            b=100.0,
        )
        assert mkt.prices() == pytest.approx([0.5, 0.5])
        assert not mkt.resolved

    def test_buy_moves_prices(self) -> None:
        """Buying shifts prices."""
        mkt = Market(
            market_id="test",
            question="Test",
            outcomes=["A", "B"],
            quantities=[0.0, 0.0],
            b=100.0,
        )
        mkt.buy(0, 50.0)
        prices = mkt.prices()
        assert prices[0] > 0.5
        assert prices[1] < 0.5

    def test_buy_returns_positive_cost(self) -> None:
        """Cost of buying is positive."""
        mkt = Market(
            market_id="test",
            question="Test",
            outcomes=["A", "B"],
            quantities=[0.0, 0.0],
            b=100.0,
        )
        cost = mkt.buy(0, 10.0)
        assert cost > 0

    def test_resolve_market(self) -> None:
        """Resolving sets winner."""
        mkt = Market(
            market_id="test",
            question="Test",
            outcomes=["A", "B"],
            quantities=[0.0, 0.0],
            b=100.0,
        )
        mkt.resolve(0)
        assert mkt.resolved
        assert mkt.winning_outcome == 0

    def test_snapshot_format(self) -> None:
        """Snapshot returns expected keys."""
        mkt = Market(
            market_id="test",
            question="Test",
            outcomes=["A", "B"],
            quantities=[0.0, 0.0],
            b=100.0,
        )
        snap = mkt.snapshot()
        assert "market_id" in snap
        assert "prices" in snap
        assert "outcomes" in snap
        assert len(snap["prices"]) == 2


# ---------------------------------------------------------------------------
# Trader tests
# ---------------------------------------------------------------------------

class TestTrader:
    """Tests for Trader scoring."""

    def test_pnl_winning_position(self) -> None:
        """Trader with winning shares gets positive P&L."""
        mkt = Market(
            market_id="m1",
            question="Test",
            outcomes=["A", "B"],
            quantities=[0.0, 0.0],
            b=100.0,
        )
        mkt.resolve(0)

        trader = Trader(trader_id="t1", strategy="oracle")
        trader.positions["m1"] = TraderPosition(
            shares={0: 100.0}, total_cost=50.0
        )

        pnl = trader.pnl({"m1": mkt})
        assert pnl == 50.0  # 100 - 50

    def test_pnl_losing_position(self) -> None:
        """Trader with losing shares gets negative P&L."""
        mkt = Market(
            market_id="m1",
            question="Test",
            outcomes=["A", "B"],
            quantities=[0.0, 0.0],
            b=100.0,
        )
        mkt.resolve(1)  # B wins

        trader = Trader(trader_id="t1", strategy="bull")
        trader.positions["m1"] = TraderPosition(
            shares={0: 100.0}, total_cost=50.0
        )

        pnl = trader.pnl({"m1": mkt})
        assert pnl == -50.0  # 0 - 50

    def test_roi_calculation(self) -> None:
        """ROI = (P&L / total_cost) * 100."""
        mkt = Market(
            market_id="m1",
            question="Test",
            outcomes=["A", "B"],
            quantities=[0.0, 0.0],
            b=100.0,
        )
        mkt.resolve(0)

        trader = Trader(trader_id="t1", strategy="oracle")
        trader.positions["m1"] = TraderPosition(
            shares={0: 150.0}, total_cost=100.0
        )

        roi = trader.roi({"m1": mkt})
        assert roi == 50.0  # (50/100)*100


# ---------------------------------------------------------------------------
# MarketEngine tests
# ---------------------------------------------------------------------------

class TestMarketEngine:
    """Tests for the engine's create/trade/resolve pipeline."""

    def test_create_colony_markets(self) -> None:
        """Creates expected number of markets."""
        engine = MarketEngine()
        markets = engine.create_colony_markets(
            ["Ares Prime", "Olympus Station", "Red Frontier"]
        )
        # 3 survival + 9 threshold + 2 comparative + 1 epidemic + 1 tech = 16
        assert len(markets) == 16
        assert len(engine.markets) == 16

    def test_default_traders(self) -> None:
        """Default traders are created when trading starts."""
        engine = MarketEngine()
        engine.create_colony_markets(["Colony A"])
        engine.simulate_trading(rounds=1)
        assert len(engine.traders) == 6

    def test_trading_produces_trades(self) -> None:
        """Trading simulation returns trade logs."""
        engine = MarketEngine()
        engine.create_colony_markets(["Colony A", "Colony B"])
        trades = engine.simulate_trading(rounds=5)
        assert len(trades) > 0

    def test_trading_moves_prices(self) -> None:
        """Trading should move at least some prices away from uniform."""
        engine = MarketEngine()
        engine.create_colony_markets(["Colony A", "Colony B"])
        engine.simulate_trading(rounds=20)

        moved = False
        for mkt in engine.markets.values():
            prices = mkt.prices()
            if any(abs(p - 1.0 / len(prices)) > 0.01 for p in prices):
                moved = True
                break
        assert moved, "No market prices moved from initial"

    def test_resolve_markets(self) -> None:
        """Resolution sets winners on all markets."""
        engine = MarketEngine()
        engine.create_colony_markets(["Colony A", "Colony B"])

        sim_results = {
            "summary": {
                "colonies": [
                    {
                        "name": "Colony A",
                        "start_pop": 100,
                        "end_pop": 180,
                        "growth_pct": 80.0,
                        "total_births": 90,
                        "total_deaths": 10,
                        "death_causes": {"epidemic": 0},
                        "net_migration": 0,
                        "techs_unlocked": 4,
                    },
                    {
                        "name": "Colony B",
                        "start_pop": 80,
                        "end_pop": 120,
                        "growth_pct": 50.0,
                        "total_births": 50,
                        "total_deaths": 10,
                        "death_causes": {"epidemic": 2},
                        "net_migration": 0,
                        "techs_unlocked": 3,
                    },
                ],
                "total_migrations": 5,
            }
        }

        resolutions = engine.resolve_markets(sim_results)
        assert len(resolutions) > 0
        # All markets should be resolved
        for mkt in engine.markets.values():
            assert mkt.resolved, f"Market {mkt.market_id} not resolved"

    def test_score_traders_after_resolution(self) -> None:
        """Scoring produces valid output."""
        engine = MarketEngine()
        engine.create_colony_markets(["Colony A"])

        sim_results = {
            "summary": {
                "colonies": [
                    {
                        "name": "Colony A",
                        "start_pop": 100,
                        "end_pop": 200,
                        "growth_pct": 100.0,
                        "total_births": 110,
                        "total_deaths": 10,
                        "death_causes": {},
                        "net_migration": 0,
                        "techs_unlocked": 5,
                    },
                ],
                "total_migrations": 0,
            }
        }

        engine.simulate_trading(rounds=10, sim_results=sim_results)
        engine.resolve_markets(sim_results)
        scores = engine.score_traders()

        assert len(scores) == 6  # 6 default traders
        for s in scores:
            assert "trader_id" in s
            assert "pnl" in s
            assert "roi" in s


# ---------------------------------------------------------------------------
# Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end smoke tests."""

    def test_run_prediction_market_completes(self) -> None:
        """Full pipeline runs without crash."""
        results = run_prediction_market(
            sols=30, seeds=1, liquidity=50.0, trading_rounds=10
        )
        assert "_meta" in results
        assert results["_meta"]["model"] == "LMSR"
        assert results["stats"]["total_markets"] > 0
        assert results["stats"]["resolved_markets"] > 0
        assert results["stats"]["total_trades"] > 0

    def test_pipeline_output_format(self) -> None:
        """Output contains expected keys."""
        results = run_prediction_market(sols=10, trading_rounds=5)
        assert "sim_summary" in results
        assert "markets" in results
        assert "trader_scores" in results
        assert "stats" in results

    def test_pipeline_all_markets_resolved(self) -> None:
        """All markets should be resolved after pipeline."""
        results = run_prediction_market(sols=30, trading_rounds=10)
        assert (
            results["stats"]["resolved_markets"]
            == results["stats"]["total_markets"]
        )


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Property-based tests for LMSR invariants."""

    @pytest.mark.parametrize("b", [10.0, 50.0, 100.0, 500.0])
    def test_prices_sum_to_one_various_b(self, b: float) -> None:
        """Prices sum to 1 for various liquidity values."""
        import random
        rng = random.Random(42)
        for _ in range(20):
            n = rng.randint(2, 5)
            q = [rng.uniform(-100, 100) for _ in range(n)]
            p = lmsr_prices(q, b)
            assert abs(sum(p) - 1.0) < 1e-8

    def test_prices_all_positive(self) -> None:
        """All prices must be strictly positive."""
        import random
        rng = random.Random(123)
        for _ in range(50):
            n = rng.randint(2, 6)
            q = [rng.uniform(-200, 200) for _ in range(n)]
            p = lmsr_prices(q, b=100.0)
            for pi in p:
                assert pi > 0, f"Price <= 0: {pi}"

    def test_cost_monotone_in_quantity(self) -> None:
        """Adding shares to any outcome increases total cost."""
        import random
        rng = random.Random(999)
        for _ in range(30):
            q = [rng.uniform(0, 100) for _ in range(3)]
            c_before = lmsr_cost(q, b=100.0)
            idx = rng.randint(0, 2)
            q[idx] += rng.uniform(1, 50)
            c_after = lmsr_cost(q, b=100.0)
            assert c_after > c_before

    def test_trade_cost_non_negative_for_positive_shares(self) -> None:
        """Buying positive shares always costs >= 0."""
        import random
        rng = random.Random(456)
        for _ in range(50):
            n = rng.randint(2, 4)
            q = [rng.uniform(0, 100) for _ in range(n)]
            idx = rng.randint(0, n - 1)
            shares = rng.uniform(0.1, 50)
            cost = lmsr_trade_cost(q, idx, shares, b=100.0)
            assert cost >= 0, f"Negative cost: {cost}"

    def test_volume_increases_with_trades(self) -> None:
        """Market volume should monotonically increase."""
        mkt = Market(
            market_id="test",
            question="Test",
            outcomes=["A", "B"],
            quantities=[0.0, 0.0],
            b=100.0,
        )
        prev_vol = 0.0
        for _ in range(10):
            mkt.buy(0, 5.0)
            assert mkt.total_volume >= prev_vol
            prev_vol = mkt.total_volume
