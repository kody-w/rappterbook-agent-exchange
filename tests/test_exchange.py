"""
test_exchange.py — Unit tests for the Agent Stock Exchange simulation engine.

72 tests covering helpers, price computation, Order/OrderBook/Portfolio classes,
trading strategies, market maker, simulation pipeline, and financial invariants.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import exchange


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_agents():
    """5 mock agents spanning different archetypes."""
    return {
        "zion-philosopher-01": {
            "name": "Sage", "karma": 100, "post_count": 20,
            "comment_count": 60, "bio": "I think",
            "traits": {"philosopher": 0.9, "coder": 0.1, "debater": 0.3,
                       "welcomer": 0.1, "curator": 0.2, "storyteller": 0.1,
                       "researcher": 0.5, "contrarian": 0.2, "archivist": 0.1,
                       "wildcard": 0.1},
        },
        "zion-coder-02": {
            "name": "Builder", "karma": 80, "post_count": 15,
            "comment_count": 30, "bio": "I build",
            "traits": {"philosopher": 0.1, "coder": 0.9, "debater": 0.1,
                       "welcomer": 0.1, "curator": 0.1, "storyteller": 0.1,
                       "researcher": 0.6, "contrarian": 0.1, "archivist": 0.3,
                       "wildcard": 0.1},
        },
        "zion-contrarian-03": {
            "name": "Rebel", "karma": 30, "post_count": 5,
            "comment_count": 50, "bio": "I disagree",
            "traits": {"philosopher": 0.3, "coder": 0.1, "debater": 0.7,
                       "welcomer": 0.0, "curator": 0.1, "storyteller": 0.1,
                       "researcher": 0.1, "contrarian": 0.9, "archivist": 0.1,
                       "wildcard": 0.3},
        },
        "zion-wildcard-04": {
            "name": "Chaos", "karma": 50, "post_count": 10,
            "comment_count": 10, "bio": "Surprise",
            "traits": {"philosopher": 0.2, "coder": 0.2, "debater": 0.2,
                       "welcomer": 0.2, "curator": 0.2, "storyteller": 0.2,
                       "researcher": 0.2, "contrarian": 0.2, "archivist": 0.2,
                       "wildcard": 0.9},
        },
        "zion-welcomer-05": {
            "name": "Greeter", "karma": 70, "post_count": 25,
            "comment_count": 100, "bio": "Welcome!",
            "traits": {"philosopher": 0.1, "coder": 0.1, "debater": 0.1,
                       "welcomer": 0.9, "curator": 0.3, "storyteller": 0.2,
                       "researcher": 0.1, "contrarian": 0.0, "archivist": 0.1,
                       "wildcard": 0.1},
        },
    }


@pytest.fixture
def prices(mock_agents):
    return exchange.compute_prices(mock_agents)


# ===========================================================================
# Helper Functions
# ===========================================================================

class TestDeterministicSeed:

    def test_same_input_same_output(self):
        a = exchange.deterministic_seed("agent-1", 5)
        b = exchange.deterministic_seed("agent-1", 5)
        assert a == b

    def test_different_agents_differ(self):
        a = exchange.deterministic_seed("agent-1", 5)
        b = exchange.deterministic_seed("agent-2", 5)
        assert a != b

    def test_different_rounds_differ(self):
        a = exchange.deterministic_seed("agent-1", 1)
        b = exchange.deterministic_seed("agent-1", 2)
        assert a != b

    def test_returns_int(self):
        assert isinstance(exchange.deterministic_seed("x", 0), int)


class TestClamp:

    def test_within_range(self):
        assert exchange.clamp(5.0, 0.0, 10.0) == 5.0

    def test_below_min(self):
        assert exchange.clamp(-1.0, 0.0, 10.0) == 0.0

    def test_above_max(self):
        assert exchange.clamp(15.0, 0.0, 10.0) == 10.0

    def test_at_boundaries(self):
        assert exchange.clamp(0.0, 0.0, 10.0) == 0.0
        assert exchange.clamp(10.0, 0.0, 10.0) == 10.0


class TestExtractArchetype:

    def test_philosopher(self):
        assert exchange.extract_archetype("zion-philosopher-01") == "philosopher"

    def test_coder(self):
        assert exchange.extract_archetype("zion-coder-02") == "coder"

    def test_no_zion_prefix(self):
        assert exchange.extract_archetype("random-agent") == "wildcard"

    def test_single_word(self):
        assert exchange.extract_archetype("agent") == "wildcard"


class TestComputeTraitVector:

    def test_length(self, mock_agents):
        for aid, agent in mock_agents.items():
            vec = exchange.compute_trait_vector(agent)
            assert len(vec) == len(exchange.TRAIT_KEYS)

    def test_missing_traits(self):
        agent = {"traits": {}}
        vec = exchange.compute_trait_vector(agent)
        assert all(v == 0.0 for v in vec)

    def test_no_traits_key(self):
        agent = {}
        vec = exchange.compute_trait_vector(agent)
        assert len(vec) == len(exchange.TRAIT_KEYS)


class TestEuclideanDistance:

    def test_same_point(self):
        assert exchange.euclidean_distance([1, 2], [1, 2]) == 0.0

    def test_known_distance(self):
        assert exchange.euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)

    def test_symmetric(self):
        a, b = [1, 2, 3], [4, 5, 6]
        assert exchange.euclidean_distance(a, b) == pytest.approx(
            exchange.euclidean_distance(b, a)
        )


# ===========================================================================
# Price Computation
# ===========================================================================

class TestComputePrices:

    def test_returns_prices_for_all_agents(self, mock_agents):
        prices = exchange.compute_prices(mock_agents)
        assert set(prices.keys()) == set(mock_agents.keys())

    def test_prices_bounded(self, mock_agents):
        """All prices are in [1, 100]."""
        prices = exchange.compute_prices(mock_agents)
        for aid, p in prices.items():
            assert 1.0 <= p <= 100.0, f"{aid} price {p} out of bounds"

    def test_empty_agents(self):
        assert exchange.compute_prices({}) == {}

    def test_single_agent(self):
        agents = {"a": {"karma": 50, "post_count": 10, "comment_count": 5, "traits": {}}}
        prices = exchange.compute_prices(agents)
        assert "a" in prices
        assert 1.0 <= prices["a"] <= 100.0

    def test_high_karma_higher_price(self):
        """Agent with more karma should generally price higher."""
        agents = {
            "rich": {"karma": 1000, "post_count": 50, "comment_count": 100, "traits": {}},
            "poor": {"karma": 1, "post_count": 1, "comment_count": 0, "traits": {}},
        }
        prices = exchange.compute_prices(agents)
        assert prices["rich"] > prices["poor"]


# ===========================================================================
# Order Class
# ===========================================================================

class TestOrder:

    def test_to_dict(self):
        o = exchange.Order("buyer", "target", "bid", 42.567, 3, 1)
        d = o.to_dict()
        assert d["agent_id"] == "buyer"
        assert d["target_id"] == "target"
        assert d["side"] == "bid"
        assert d["price"] == 42.57  # rounded
        assert d["quantity"] == 3
        assert d["round"] == 1


# ===========================================================================
# OrderBook
# ===========================================================================

class TestOrderBook:

    def test_add_bid(self):
        book = exchange.OrderBook()
        o = exchange.Order("a", "t", "bid", 10.0, 5, 1)
        book.add_order(o)
        assert len(book.bids.get("t", [])) == 1

    def test_add_ask(self):
        book = exchange.OrderBook()
        o = exchange.Order("a", "t", "ask", 10.0, 5, 1)
        book.add_order(o)
        assert len(book.asks.get("t", [])) == 1

    def test_match_crossing_orders(self):
        """Bid >= ask produces a trade."""
        book = exchange.OrderBook()
        book.add_order(exchange.Order("buyer", "t", "bid", 12.0, 2, 1))
        book.add_order(exchange.Order("seller", "t", "ask", 10.0, 2, 1))
        trades = book.match("t", 1)
        assert len(trades) == 1
        assert trades[0]["buyer"] == "buyer"
        assert trades[0]["seller"] == "seller"
        assert trades[0]["quantity"] == 2
        assert trades[0]["price"] == pytest.approx(11.0)  # midpoint

    def test_no_match_when_bid_below_ask(self):
        book = exchange.OrderBook()
        book.add_order(exchange.Order("b", "t", "bid", 5.0, 1, 1))
        book.add_order(exchange.Order("s", "t", "ask", 10.0, 1, 1))
        trades = book.match("t", 1)
        assert len(trades) == 0

    def test_partial_fill(self):
        """Larger bid against smaller ask produces partial fill."""
        book = exchange.OrderBook()
        book.add_order(exchange.Order("b", "t", "bid", 10.0, 5, 1))
        book.add_order(exchange.Order("s", "t", "ask", 9.0, 2, 1))
        trades = book.match("t", 1)
        assert len(trades) == 1
        assert trades[0]["quantity"] == 2
        # Remaining 3 shares still on bid side
        assert len(book.bids.get("t", [])) == 1
        assert book.bids["t"][0].quantity == 3

    def test_multiple_fills(self):
        """Multiple asks fill against one bid."""
        book = exchange.OrderBook()
        book.add_order(exchange.Order("b", "t", "bid", 20.0, 10, 1))
        book.add_order(exchange.Order("s1", "t", "ask", 15.0, 3, 1))
        book.add_order(exchange.Order("s2", "t", "ask", 18.0, 4, 1))
        trades = book.match("t", 1)
        assert len(trades) == 2
        total_qty = sum(t["quantity"] for t in trades)
        assert total_qty == 7

    def test_snapshot(self):
        book = exchange.OrderBook()
        book.add_order(exchange.Order("a", "t1", "bid", 10.0, 1, 1))
        book.add_order(exchange.Order("b", "t2", "ask", 20.0, 1, 1))
        snap = book.snapshot()
        assert "bids" in snap
        assert "asks" in snap
        assert len(snap["bids"]) == 1
        assert len(snap["asks"]) == 1

    def test_empty_match(self):
        book = exchange.OrderBook()
        trades = book.match("nonexistent", 1)
        assert trades == []


# ===========================================================================
# Portfolio
# ===========================================================================

class TestPortfolio:

    def test_initial_cash(self):
        p = exchange.Portfolio()
        assert p.cash == exchange.STARTING_CASH

    def test_custom_cash(self):
        p = exchange.Portfolio(cash=500.0)
        assert p.cash == 500.0

    def test_buy_success(self):
        p = exchange.Portfolio(cash=100.0)
        assert p.buy("stock-a", 2, 10.0) is True
        assert p.cash == pytest.approx(80.0)
        assert p.holdings["stock-a"] == 2

    def test_buy_insufficient_funds(self):
        p = exchange.Portfolio(cash=10.0)
        assert p.buy("stock-a", 2, 10.0) is False
        assert p.cash == 10.0

    def test_sell_success(self):
        p = exchange.Portfolio(cash=0.0)
        p.holdings["stock-a"] = 5
        assert p.sell("stock-a", 3, 10.0) is True
        assert p.cash == pytest.approx(30.0)
        assert p.holdings["stock-a"] == 2

    def test_sell_insufficient_shares(self):
        p = exchange.Portfolio(cash=0.0)
        p.holdings["stock-a"] = 1
        assert p.sell("stock-a", 5, 10.0) is False

    def test_sell_removes_empty_holding(self):
        p = exchange.Portfolio()
        p.holdings["stock-a"] = 2
        p.sell("stock-a", 2, 10.0)
        assert "stock-a" not in p.holdings

    def test_total_value(self):
        p = exchange.Portfolio(cash=50.0)
        p.holdings["a"] = 10
        prices = {"a": 5.0}
        assert p.total_value(prices) == pytest.approx(100.0)

    def test_to_dict(self):
        p = exchange.Portfolio(cash=100.0)
        p.holdings["a"] = 5
        prices = {"a": 10.0}
        d = p.to_dict(prices)
        assert d["cash"] == 100.0
        assert d["total_value"] == 150.0
        assert d["holdings"] == {"a": 5}


# ===========================================================================
# Trading Strategies
# ===========================================================================

class TestStrategies:

    def test_philosopher_buys_philosophers(self, mock_agents, prices):
        rng = random.Random(42)
        targets = exchange.pick_targets_philosopher(
            "zion-philosopher-01", mock_agents, prices, rng
        )
        assert len(targets) <= 3
        for aid, side in targets:
            assert aid != "zion-philosopher-01"

    def test_coder_buys_coders(self, mock_agents, prices):
        rng = random.Random(42)
        targets = exchange.pick_targets_coder(
            "zion-coder-02", mock_agents, prices, rng
        )
        for aid, side in targets:
            assert aid != "zion-coder-02"
            arch = exchange.extract_archetype(aid)
            assert arch in ("coder", "researcher", "archivist")

    def test_contrarian_buys_cheapest(self, mock_agents, prices):
        rng = random.Random(42)
        targets = exchange.pick_targets_contrarian(
            "zion-contrarian-03", mock_agents, prices, rng
        )
        assert len(targets) <= 3
        for aid, side in targets:
            assert side == "buy"

    def test_wildcard_random_sides(self, mock_agents, prices):
        rng = random.Random(42)
        targets = exchange.pick_targets_wildcard(
            "zion-wildcard-04", mock_agents, prices, rng
        )
        sides = {side for _, side in targets}
        # With enough targets, wildcards should have both buy and sell
        assert len(targets) <= 4

    def test_welcomer_buys_widely(self, mock_agents, prices):
        rng = random.Random(42)
        targets = exchange.pick_targets_welcomer(
            "zion-welcomer-05", mock_agents, prices, rng
        )
        for _, side in targets:
            assert side == "buy"

    def test_debater_mixed_strategy(self, mock_agents, prices):
        rng = random.Random(42)
        targets = exchange.pick_targets_debater(
            "zion-contrarian-03", mock_agents, prices, rng
        )
        sides = {side for _, side in targets}
        assert "buy" in sides
        assert "sell" in sides

    def test_all_strategies_in_map(self):
        """STRATEGY_MAP has an entry for every archetype."""
        for key in exchange.TRAIT_KEYS:
            assert key in exchange.STRATEGY_MAP

    def test_no_self_trades(self, mock_agents, prices):
        """No strategy returns the agent's own ID as a target."""
        for aid in mock_agents:
            arch = exchange.extract_archetype(aid)
            strategy = exchange.STRATEGY_MAP.get(arch, exchange.pick_targets_wildcard)
            rng = random.Random(42)
            targets = strategy(aid, mock_agents, prices, rng)
            for target_id, _ in targets:
                assert target_id != aid, f"{aid} targets itself"


# ===========================================================================
# Market Maker
# ===========================================================================

class TestMarketMaker:

    def test_produces_bid_and_ask(self, prices):
        orders = exchange.market_maker_orders(prices, 1)
        bids = [o for o in orders if o.side == "bid"]
        asks = [o for o in orders if o.side == "ask"]
        assert len(bids) == len(prices)
        assert len(asks) == len(prices)

    def test_spread(self, prices):
        """Market maker bid < ask for each agent."""
        orders = exchange.market_maker_orders(prices, 1)
        by_target = {}
        for o in orders:
            by_target.setdefault(o.target_id, {})[o.side] = o.price
        for target, sides in by_target.items():
            assert sides["bid"] < sides["ask"]

    def test_market_maker_id(self, prices):
        orders = exchange.market_maker_orders(prices, 1)
        for o in orders:
            assert o.agent_id == "__market_maker__"


# ===========================================================================
# Full Simulation
# ===========================================================================

class TestRunSimulation:

    def test_returns_output_structure(self, mock_agents):
        """Simulation returns all expected top-level keys."""
        result = exchange.run_simulation(mock_agents)
        required = {"_meta", "agents", "trades", "order_book", "portfolios",
                     "top_movers", "market_stats"}
        assert required.issubset(set(result.keys()))

    def test_agent_records_count(self, mock_agents):
        result = exchange.run_simulation(mock_agents)
        assert len(result["agents"]) == len(mock_agents)

    def test_meta_fields(self, mock_agents):
        result = exchange.run_simulation(mock_agents)
        meta = result["_meta"]
        assert meta["num_agents"] == len(mock_agents)
        assert meta["num_rounds"] == exchange.NUM_ROUNDS
        assert meta["total_trades"] >= 0

    def test_prices_positive(self, mock_agents):
        """All final prices are positive."""
        result = exchange.run_simulation(mock_agents)
        for agent in result["agents"]:
            assert agent["price"] > 0

    def test_price_history_length(self, mock_agents):
        """Price history has NUM_ROUNDS + 1 entries (initial + each round)."""
        result = exchange.run_simulation(mock_agents)
        for agent in result["agents"]:
            assert len(agent["price_history"]) == exchange.NUM_ROUNDS + 1

    def test_trades_capped(self, mock_agents):
        result = exchange.run_simulation(mock_agents)
        assert len(result["trades"]) <= exchange.TRADE_LOG_LIMIT

    def test_portfolios_for_all_agents(self, mock_agents):
        result = exchange.run_simulation(mock_agents)
        for aid in mock_agents:
            assert aid in result["portfolios"]

    def test_deterministic(self, mock_agents):
        """Same agents produce identical results."""
        random.seed(123)
        r1 = exchange.run_simulation(mock_agents)
        random.seed(123)
        r2 = exchange.run_simulation(mock_agents)
        assert r1["market_stats"]["total_volume"] == r2["market_stats"]["total_volume"]


# ===========================================================================
# Financial Invariants
# ===========================================================================

class TestFinancialInvariants:

    def test_market_cap_equals_price_times_shares(self, mock_agents):
        """market_cap = price × shares_outstanding for every agent."""
        result = exchange.run_simulation(mock_agents)
        for agent in result["agents"]:
            expected = round(agent["price"] * exchange.SHARES_OUTSTANDING, 2)
            assert agent["market_cap"] == pytest.approx(expected, abs=0.02)

    def test_total_market_cap_positive(self, mock_agents):
        result = exchange.run_simulation(mock_agents)
        assert result["market_stats"]["total_market_cap"] > 0

    def test_portfolio_cash_non_negative(self, mock_agents):
        """No agent ends with negative cash."""
        result = exchange.run_simulation(mock_agents)
        for aid, pf in result["portfolios"].items():
            assert pf["cash"] >= 0, f"{aid} has negative cash: {pf['cash']}"

    def test_top_movers_bounded(self, mock_agents):
        result = exchange.run_simulation(mock_agents)
        assert len(result["top_movers"]["gainers"]) <= exchange.TOP_MOVERS_LIMIT
        assert len(result["top_movers"]["losers"]) <= exchange.TOP_MOVERS_LIMIT
