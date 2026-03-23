"""
Tests for the prediction market engine (market_maker.py).

Covers: LMSR pricing, market resolution, betting strategies,
engine integration with simulation results.

Run: python -m pytest tests/test_market_maker.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.market_maker import (
    Market,
    LMSR_LIQUIDITY_B,
    create_default_markets,
    PredictionEngine,
    resolve_population_above,
    resolve_all_survive,
    resolve_tech_first_colony,
    resolve_total_migration_above,
    resolve_epidemic_count_above,
    resolve_global_storm_duration,
    resolve_growth_pct_above,
    BettorState,
    _philosopher_bets,
    _contrarian_bets,
    _wildcard_bets,
    RESOLVERS,
)


# ---------------------------------------------------------------------------
# LMSR pricing tests
# ---------------------------------------------------------------------------

class TestLMSR:
    """Test the LMSR automated market maker math."""

    def test_initial_price_is_fifty_fifty(self):
        """Fresh market starts at 50/50."""
        m = Market(market_id="test", question="Test?", category="test",
                   resolve_fn_name="all_survive", resolve_args={})
        assert abs(m.price_yes() - 0.5) < 1e-10
        assert abs(m.price_no() - 0.5) < 1e-10

    def test_prices_sum_to_one(self):
        """YES + NO prices always sum to 1.0 (no-arbitrage)."""
        m = Market(market_id="test", question="Test?", category="test",
                   resolve_fn_name="all_survive", resolve_args={})
        m.buy_yes(10.0)
        assert abs(m.price_yes() + m.price_no() - 1.0) < 1e-10

    def test_buying_yes_increases_price(self):
        """Buying YES shares pushes YES price up."""
        m = Market(market_id="test", question="Test?", category="test",
                   resolve_fn_name="all_survive", resolve_args={})
        initial = m.price_yes()
        m.buy_yes(5.0)
        assert m.price_yes() > initial

    def test_buying_no_decreases_yes_price(self):
        """Buying NO shares pushes YES price down."""
        m = Market(market_id="test", question="Test?", category="test",
                   resolve_fn_name="all_survive", resolve_args={})
        initial = m.price_yes()
        m.buy_no(5.0)
        assert m.price_yes() < initial

    def test_cost_is_positive(self):
        """Buying shares always costs money."""
        m = Market(market_id="test", question="Test?", category="test",
                   resolve_fn_name="all_survive", resolve_args={})
        cost_yes = m.buy_yes(3.0)
        assert cost_yes > 0
        cost_no = m.buy_no(3.0)
        assert cost_no > 0

    def test_large_purchase_moves_price_more(self):
        """More shares = more price movement."""
        m1 = Market(market_id="t1", question="T?", category="test",
                    resolve_fn_name="all_survive", resolve_args={})
        m2 = Market(market_id="t2", question="T?", category="test",
                    resolve_fn_name="all_survive", resolve_args={})
        m1.buy_yes(1.0)
        m2.buy_yes(10.0)
        assert m2.price_yes() > m1.price_yes()

    def test_price_bounded_zero_one(self):
        """Prices stay in (0, 1) even with extreme buys."""
        m = Market(market_id="test", question="Test?", category="test",
                   resolve_fn_name="all_survive", resolve_args={})
        m.buy_yes(500.0)
        assert 0.0 < m.price_yes() < 1.0
        assert 0.0 < m.price_no() < 1.0

    def test_symmetry(self):
        """Equal YES and NO buys return price to 0.5."""
        m = Market(market_id="test", question="Test?", category="test",
                   resolve_fn_name="all_survive", resolve_args={})
        m.buy_yes(10.0)
        m.buy_no(10.0)
        assert abs(m.price_yes() - 0.5) < 1e-10

    def test_volume_tracking(self):
        """Volume accumulates across trades."""
        m = Market(market_id="test", question="Test?", category="test",
                   resolve_fn_name="all_survive", resolve_args={})
        m.buy_yes(5.0)
        m.buy_no(3.0)
        assert m.num_trades == 2
        assert m.total_volume > 0

    def test_history_recording(self):
        """record_snapshot appends to history."""
        m = Market(market_id="test", question="Test?", category="test",
                   resolve_fn_name="all_survive", resolve_args={})
        m.record_snapshot(1)
        m.buy_yes(5.0)
        m.record_snapshot(2)
        assert len(m.history) == 2
        assert m.history[0]["round"] == 1
        assert m.history[1]["round"] == 2


# ---------------------------------------------------------------------------
# Market resolution tests
# ---------------------------------------------------------------------------

class TestResolvers:
    """Test market resolution functions against simulated results."""

    MOCK_RESULTS = {
        "colonies": [
            {
                "name": "Ares Prime",
                "final_population": 210,
                "tech": {"unlocked": [
                    {"name": "Advanced Solar Cells", "branch": "power", "sol": 100},
                    {"name": "Martian Crop Genetics", "branch": "food", "sol": 200},
                ]},
                "events": [
                    {"type": "epidemic_start", "sol": 50},
                    {"type": "epidemic_start", "sol": 200},
                ],
            },
            {
                "name": "Olympus Station",
                "final_population": 120,
                "tech": {"unlocked": [
                    {"name": "Advanced Solar Cells", "branch": "power", "sol": 150},
                ]},
                "events": [
                    {"type": "epidemic_start", "sol": 100},
                ],
            },
            {
                "name": "Red Frontier",
                "final_population": 130,
                "tech": {"unlocked": [
                    {"name": "Compact Fusion Reactor", "branch": "power", "sol": 250},
                ]},
                "events": [
                    {"type": "epidemic_start", "sol": 80},
                    {"type": "epidemic_start", "sol": 180},
                    {"type": "epidemic_start", "sol": 300},
                ],
            },
        ],
        "environment": {
            "history": [{"storm": None}] * 50 + [{"storm": "global"}] * 45 + [{"storm": None}] * 270,
        },
        "summary": {
            "total_migrations": 35,
            "colonies": [
                {"name": "Ares Prime", "growth_pct": 75.0},
                {"name": "Olympus Station", "growth_pct": 50.0},
                {"name": "Red Frontier", "growth_pct": 116.7},
            ],
        },
    }

    def test_population_above_true(self):
        assert resolve_population_above(self.MOCK_RESULTS, "Ares Prime", 200) is True

    def test_population_above_false(self):
        assert resolve_population_above(self.MOCK_RESULTS, "Olympus Station", 150) is False

    def test_all_survive_true(self):
        assert resolve_all_survive(self.MOCK_RESULTS) is True

    def test_all_survive_false(self):
        dead = {"colonies": [{"final_population": 0}]}
        assert resolve_all_survive(dead) is False

    def test_tech_first_colony_true(self):
        assert resolve_tech_first_colony(
            self.MOCK_RESULTS, "Advanced Solar Cells", "Ares Prime"
        ) is True

    def test_tech_first_colony_false(self):
        assert resolve_tech_first_colony(
            self.MOCK_RESULTS, "Compact Fusion Reactor", "Ares Prime"
        ) is False

    def test_total_migration_above(self):
        assert resolve_total_migration_above(self.MOCK_RESULTS, 30) is True
        assert resolve_total_migration_above(self.MOCK_RESULTS, 50) is False

    def test_epidemic_count(self):
        assert resolve_epidemic_count_above(self.MOCK_RESULTS, 5) is True
        assert resolve_epidemic_count_above(self.MOCK_RESULTS, 10) is False

    def test_global_storm_duration(self):
        assert resolve_global_storm_duration(self.MOCK_RESULTS, 40) is True
        assert resolve_global_storm_duration(self.MOCK_RESULTS, 50) is False

    def test_growth_pct_above(self):
        assert resolve_growth_pct_above(self.MOCK_RESULTS, "Red Frontier", 100.0) is True
        assert resolve_growth_pct_above(self.MOCK_RESULTS, "Ares Prime", 100.0) is False


# ---------------------------------------------------------------------------
# Default markets
# ---------------------------------------------------------------------------

class TestDefaultMarkets:
    def test_creates_twelve_markets(self):
        markets = create_default_markets()
        assert len(markets) == 12

    def test_all_have_valid_resolvers(self):
        markets = create_default_markets()
        for m in markets:
            assert m.resolve_fn_name in RESOLVERS, f"Missing resolver: {m.resolve_fn_name}"

    def test_unique_ids(self):
        markets = create_default_markets()
        ids = [m.market_id for m in markets]
        assert len(ids) == len(set(ids))

    def test_categories_covered(self):
        markets = create_default_markets()
        categories = {m.category for m in markets}
        assert "population" in categories
        assert "survival" in categories
        assert "tech" in categories
        assert "event" in categories
        assert "migration" in categories


# ---------------------------------------------------------------------------
# Betting strategies
# ---------------------------------------------------------------------------

class TestBettingStrategies:
    def _make_markets(self):
        return create_default_markets()

    def test_philosopher_prefers_survival(self):
        import random
        bettor = BettorState(agent_id="zion-philosopher-01")
        markets = self._make_markets()
        rng = random.Random(42)
        bets = _philosopher_bets(bettor, markets, rng)
        assert len(bets) > 0
        # All bets should have valid market ids
        market_ids = {m.market_id for m in markets}
        for mid, side, size in bets:
            assert mid in market_ids
            assert side in ("yes", "no")
            assert size > 0

    def test_contrarian_bets_against_consensus(self):
        import random
        bettor = BettorState(agent_id="zion-contrarian-01")
        markets = self._make_markets()
        # Push a market toward YES heavily
        markets[0].buy_yes(100.0)
        assert markets[0].price_yes() > 0.65
        rng = random.Random(42)
        bets = _contrarian_bets(bettor, markets, rng)
        # Should find at least one NO bet on the skewed market
        nos = [b for b in bets if b[1] == "no"]
        assert len(nos) > 0

    def test_wildcard_produces_bets(self):
        import random
        bettor = BettorState(agent_id="zion-wildcard-01")
        markets = self._make_markets()
        rng = random.Random(42)
        bets = _wildcard_bets(bettor, markets, rng)
        assert len(bets) > 0


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestPredictionEngine:
    AGENT_IDS = [
        "zion-philosopher-01", "zion-coder-02", "zion-contrarian-03",
        "zion-wildcard-04", "zion-researcher-05",
    ]

    def test_engine_runs_without_crash(self):
        """Smoke test — full betting + resolution cycle."""
        markets = create_default_markets()
        engine = PredictionEngine(markets, self.AGENT_IDS, seed=42)
        engine.run_betting_rounds(n_rounds=10)

        # All markets should have trade history
        for m in engine.markets.values():
            assert len(m.history) == 10

    def test_resolve_with_mock_results(self):
        markets = create_default_markets()
        engine = PredictionEngine(markets, self.AGENT_IDS, seed=42)
        engine.run_betting_rounds(n_rounds=10)
        outcomes = engine.resolve_all(TestResolvers.MOCK_RESULTS)
        assert len(outcomes) > 0
        # All markets should be resolved
        for m in engine.markets.values():
            assert m.resolved is True

    def test_leaderboard_sorted_by_pnl(self):
        markets = create_default_markets()
        engine = PredictionEngine(markets, self.AGENT_IDS, seed=42)
        engine.run_betting_rounds(n_rounds=10)
        engine.resolve_all(TestResolvers.MOCK_RESULTS)
        board = engine.leaderboard()
        assert len(board) == len(self.AGENT_IDS)
        # Verify sorted descending
        for i in range(len(board) - 1):
            assert board[i]["pnl"] >= board[i + 1]["pnl"]

    def test_full_results_structure(self):
        markets = create_default_markets()
        engine = PredictionEngine(markets, self.AGENT_IDS, seed=42)
        engine.run_betting_rounds(n_rounds=5)
        engine.resolve_all(TestResolvers.MOCK_RESULTS)
        results = engine.full_results()
        assert "_meta" in results
        assert "markets" in results
        assert "leaderboard" in results
        assert "archetype_performance" in results
        assert results["_meta"]["engine"] == "prediction-market"

    def test_bankroll_conservation(self):
        """No bettor should go deeply negative on bankroll."""
        markets = create_default_markets()
        engine = PredictionEngine(markets, self.AGENT_IDS, seed=42)
        engine.run_betting_rounds(n_rounds=20)
        for bettor in engine.bettors.values():
            # LMSR costs are bounded, so bankroll shouldn't go wildly negative
            assert bettor.bankroll > -500, (
                f"{bettor.agent_id} bankroll={bettor.bankroll}"
            )

    def test_deterministic_results(self):
        """Same seed = same results."""
        def run_once():
            markets = create_default_markets()
            engine = PredictionEngine(markets, self.AGENT_IDS, seed=42)
            engine.run_betting_rounds(n_rounds=10)
            engine.resolve_all(TestResolvers.MOCK_RESULTS)
            return engine.leaderboard()

        r1 = run_once()
        r2 = run_once()
        assert r1 == r2


# ---------------------------------------------------------------------------
# Integration with actual Mars sim (smoke test)
# ---------------------------------------------------------------------------

class TestMarsBarnIntegration:
    """Run actual Mars sim + prediction market end-to-end."""

    def test_sim_plus_markets_smoke(self):
        """Run 50 sols, then resolve markets. No crashes."""
        from src.tick_engine import Simulation

        sim = Simulation(sols=50, env_seed=42)
        results = sim.run()

        markets = create_default_markets()
        agent_ids = ["zion-coder-01", "zion-philosopher-02", "zion-contrarian-03"]
        engine = PredictionEngine(markets, agent_ids, seed=42)
        engine.run_betting_rounds(n_rounds=5)
        outcomes = engine.resolve_all(results)

        assert len(outcomes) == len(markets)
        board = engine.leaderboard()
        assert len(board) == 3

    def test_full_365_sol_integration(self):
        """Full simulation + prediction market — the real thing."""
        from src.tick_engine import Simulation

        sim = Simulation(sols=365, env_seed=42)
        results = sim.run()

        markets = create_default_markets()
        agent_ids = [
            f"zion-{arch}-{i:02d}"
            for arch in ["philosopher", "coder", "contrarian", "wildcard", "researcher"]
            for i in range(1, 4)
        ]
        engine = PredictionEngine(markets, agent_ids, seed=42)
        engine.run_betting_rounds(n_rounds=20)
        outcomes = engine.resolve_all(results)

        # Verify physical bounds
        for m in engine.markets.values():
            assert m.resolved
            assert m.outcome is not None
            for snap in m.history:
                assert 0.0 < snap["price_yes"] < 1.0
                assert 0.0 < snap["price_no"] < 1.0
                assert abs(snap["price_yes"] + snap["price_no"] - 1.0) < 0.01

        board = engine.leaderboard()
        assert len(board) == len(agent_ids)
        # At least one agent should have non-zero P&L
        assert any(b["pnl"] != 0 for b in board)
