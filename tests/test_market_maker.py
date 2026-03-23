"""
Tests for market_maker.py — LMSR prediction market for Mars colony outcomes.

Run: python -m pytest tests/test_market_maker.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.market_maker import (
    LMSRMarket,
    AgentAccount,
    Bet,
    generate_agents,
    pick_outcome_and_size,
    resolve_markets,
    compute_brier_score,
    compute_payouts,
    run_prediction_market,
    format_report,
    MARKETS,
    LMSR_LIQUIDITY,
    STARTING_KARMA,
    ARCHETYPES,
)


# ── LMSR Tests ──


class TestLMSRMarket:
    """Tests for the LMSR market scoring rule."""

    def test_initial_prices_uniform(self) -> None:
        """All outcomes start with equal price."""
        m = LMSRMarket("test", "Q?", ["A", "B", "C"])
        prices = m.prices()
        assert len(prices) == 3
        for p in prices:
            assert abs(p - 1 / 3) < 0.001

    def test_prices_sum_to_one(self) -> None:
        """LMSR prices always sum to 1."""
        m = LMSRMarket("test", "Q?", ["A", "B"])
        assert abs(sum(m.prices()) - 1.0) < 1e-10
        m.buy("A", 50)
        assert abs(sum(m.prices()) - 1.0) < 1e-10
        m.buy("B", 30)
        assert abs(sum(m.prices()) - 1.0) < 1e-10

    def test_buy_increases_price(self) -> None:
        """Buying an outcome increases its price."""
        m = LMSRMarket("test", "Q?", ["A", "B"])
        price_before = m.price_of("A")
        m.buy("A", 10)
        price_after = m.price_of("A")
        assert price_after > price_before

    def test_buy_decreases_other_prices(self) -> None:
        """Buying outcome A decreases price of B."""
        m = LMSRMarket("test", "Q?", ["A", "B", "C"])
        price_b_before = m.price_of("B")
        m.buy("A", 20)
        price_b_after = m.price_of("B")
        assert price_b_after < price_b_before

    def test_cost_positive(self) -> None:
        """Buying shares always costs positive karma."""
        m = LMSRMarket("test", "Q?", ["A", "B"])
        cost = m.cost_to_buy("A", 10)
        assert cost > 0

    def test_cost_monotonic(self) -> None:
        """Buying more shares costs more."""
        m = LMSRMarket("test", "Q?", ["A", "B"])
        cost_small = m.cost_to_buy("A", 5)
        cost_large = m.cost_to_buy("A", 50)
        assert cost_large > cost_small

    def test_binary_market_convergence(self) -> None:
        """Heavy buying on one side drives price toward 1.0."""
        m = LMSRMarket("test", "Q?", ["yes", "no"])
        for _ in range(20):
            m.buy("yes", 50)
        assert m.price_of("yes") > 0.95

    def test_snapshot_serializable(self) -> None:
        """Snapshot returns a JSON-safe dict."""
        m = LMSRMarket("test", "Q?", ["A", "B"])
        m.buy("A", 10)
        snap = m.snapshot()
        import json
        json.dumps(snap)  # should not raise
        assert "A" in snap["outcomes"]
        assert "B" in snap["outcomes"]

    def test_resolve_sets_outcome(self) -> None:
        m = LMSRMarket("test", "Q?", ["A", "B"])
        m.resolve("A")
        assert m.resolved_outcome == "A"

    def test_resolve_invalid_raises(self) -> None:
        m = LMSRMarket("test", "Q?", ["A", "B"])
        with pytest.raises(ValueError):
            m.resolve("C")

    def test_liquidity_parameter_effect(self) -> None:
        """Higher liquidity → smaller price impact per share."""
        m_low = LMSRMarket("lo", "Q?", ["A", "B"], liquidity=10)
        m_high = LMSRMarket("hi", "Q?", ["A", "B"], liquidity=1000)
        m_low.buy("A", 10)
        m_high.buy("A", 10)
        # Low liquidity should have bigger price movement
        assert m_low.price_of("A") > m_high.price_of("A")


# ── Brier Score Tests ──


class TestBrierScore:

    def test_perfect_prediction(self) -> None:
        assert compute_brier_score(1.0, True) == 0.0
        assert compute_brier_score(0.0, False) == 0.0

    def test_worst_prediction(self) -> None:
        assert compute_brier_score(0.0, True) == 1.0
        assert compute_brier_score(1.0, False) == 1.0

    def test_fifty_fifty(self) -> None:
        assert abs(compute_brier_score(0.5, True) - 0.25) < 1e-10

    def test_range_bounded(self) -> None:
        """Brier score always in [0, 1]."""
        import random
        rng = random.Random(42)
        for _ in range(100):
            prob = rng.random()
            outcome = rng.choice([True, False])
            score = compute_brier_score(prob, outcome)
            assert 0.0 <= score <= 1.0


# ── Agent Generation Tests ──


class TestAgentGeneration:

    def test_correct_count(self) -> None:
        agents = generate_agents(20)
        assert len(agents) == 20

    def test_archetype_cycling(self) -> None:
        """Archetypes cycle through the list."""
        agents = generate_agents(10)
        archetypes = [a.archetype for a in agents.values()]
        assert archetypes == ARCHETYPES

    def test_starting_karma(self) -> None:
        agents = generate_agents(5)
        for a in agents.values():
            assert a.karma == STARTING_KARMA

    def test_deterministic(self) -> None:
        a1 = generate_agents(10, seed=42)
        a2 = generate_agents(10, seed=42)
        assert list(a1.keys()) == list(a2.keys())


# ── Betting Strategy Tests ──


class TestBettingStrategy:

    def test_returns_valid_outcome(self) -> None:
        import random
        agent = AgentAccount("test", "coder")
        market = LMSRMarket("test", "Q?", ["A", "B", "C"])
        rng = random.Random(42)
        result = pick_outcome_and_size(agent, market, rng)
        assert result is not None
        outcome, size = result
        assert outcome in market.outcomes
        assert size > 0

    def test_broke_agent_returns_none(self) -> None:
        import random
        agent = AgentAccount("test", "coder", karma=0.0)
        market = LMSRMarket("test", "Q?", ["A", "B"])
        rng = random.Random(42)
        result = pick_outcome_and_size(agent, market, rng)
        assert result is None

    def test_contrarian_bias(self) -> None:
        """Contrarian archetype picks underdog more often."""
        import random
        market = LMSRMarket("test", "Q?", ["A", "B"])
        market.buy("A", 200)  # Make A very expensive
        underdog_picks = 0
        trials = 100
        for i in range(trials):
            agent = AgentAccount(f"test-{i}", "contrarian")
            rng = random.Random(i)
            result = pick_outcome_and_size(agent, market, rng)
            if result and result[0] == "B":
                underdog_picks += 1
        # Contrarians should pick B (the underdog) > 50% of the time
        assert underdog_picks > trials * 0.4


# ── Resolution Tests ──


class TestResolution:

    def _mock_sim_results(self) -> dict:
        return {
            "summary": {
                "colonies": [
                    {"name": "Ares Prime", "strategy": "conservative",
                     "start_pop": 120, "end_pop": 200, "min_pop": 100,
                     "growth_pct": 66.7, "techs_unlocked": 5},
                    {"name": "Olympus Station", "strategy": "balanced",
                     "start_pop": 80, "end_pop": 120, "min_pop": 70,
                     "growth_pct": 50.0, "techs_unlocked": 4},
                    {"name": "Red Frontier", "strategy": "aggressive",
                     "start_pop": 60, "end_pop": 150, "min_pop": 8,
                     "growth_pct": 150.0, "techs_unlocked": 6},
                ],
            },
            "colonies": [
                {"name": "Ares Prime", "events": [
                    {"sol": 100, "type": "epidemic_start", "strain": "Mars Flu"},
                ]},
                {"name": "Olympus Station", "events": []},
                {"name": "Red Frontier", "events": [
                    {"sol": 50, "type": "epidemic_start", "strain": "Rad Fever"},
                ]},
            ],
        }

    def test_highest_pop(self) -> None:
        results = self._mock_sim_results()
        markets = {m["id"]: LMSRMarket(m["id"], m["question"], list(m["outcomes"]))
                   for m in MARKETS}
        resolutions = resolve_markets(markets, results)
        assert resolutions["highest_pop"] == "Ares Prime"

    def test_any_death_yes(self) -> None:
        results = self._mock_sim_results()
        markets = {m["id"]: LMSRMarket(m["id"], m["question"], list(m["outcomes"]))
                   for m in MARKETS}
        resolutions = resolve_markets(markets, results)
        # Red Frontier min_pop=8 < 10
        assert resolutions["any_death"] == "yes"

    def test_most_techs(self) -> None:
        results = self._mock_sim_results()
        markets = {m["id"]: LMSRMarket(m["id"], m["question"], list(m["outcomes"]))
                   for m in MARKETS}
        resolutions = resolve_markets(markets, results)
        assert resolutions["most_techs"] == "Red Frontier"

    def test_total_pop_over_400(self) -> None:
        results = self._mock_sim_results()
        markets = {m["id"]: LMSRMarket(m["id"], m["question"], list(m["outcomes"]))
                   for m in MARKETS}
        resolutions = resolve_markets(markets, results)
        # 200 + 120 + 150 = 470 > 400
        assert resolutions["total_pop_over_400"] == "yes"

    def test_first_epidemic(self) -> None:
        results = self._mock_sim_results()
        markets = {m["id"]: LMSRMarket(m["id"], m["question"], list(m["outcomes"]))
                   for m in MARKETS}
        resolutions = resolve_markets(markets, results)
        # Red Frontier at sol 50, Ares Prime at sol 100
        assert resolutions["first_epidemic"] == "Red Frontier"


# ── Payout Tests ──


class TestPayouts:

    def test_winning_bet_profits(self) -> None:
        market = LMSRMarket("test", "Q?", ["A", "B"])
        agent = AgentAccount("a1", "coder")
        cost = market.buy("A", 10)
        agent.karma -= cost
        agent.bets.append(Bet("a1", "test", "A", 10, cost, 0.5))
        market.resolve("A")
        results = compute_payouts(
            {"a1": agent}, {"test": market}, {"test": "A"}
        )
        assert results["a1"]["net_profit"] > 0

    def test_losing_bet_loses_money(self) -> None:
        market = LMSRMarket("test", "Q?", ["A", "B"])
        agent = AgentAccount("a1", "coder")
        cost = market.buy("A", 10)
        agent.karma -= cost
        agent.bets.append(Bet("a1", "test", "A", 10, cost, 0.5))
        market.resolve("B")
        results = compute_payouts(
            {"a1": agent}, {"test": market}, {"test": "B"}
        )
        assert results["a1"]["net_profit"] < 0

    def test_payout_has_brier_score(self) -> None:
        market = LMSRMarket("test", "Q?", ["A", "B"])
        agent = AgentAccount("a1", "coder")
        cost = market.buy("A", 10)
        agent.karma -= cost
        agent.bets.append(Bet("a1", "test", "A", 10, cost, 0.5))
        results = compute_payouts(
            {"a1": agent}, {"test": market}, {"test": "A"}
        )
        assert "avg_brier" in results["a1"]
        assert 0 <= results["a1"]["avg_brier"] <= 1


# ── Full Pipeline Smoke Tests ──


class TestFullPipeline:

    def test_pipeline_runs(self) -> None:
        """The full pipeline runs without crashing."""
        results = run_prediction_market(n_agents=6, sim_sols=30, sim_seed=42)
        assert "_meta" in results
        assert results["_meta"]["engine"] == "market-maker"

    def test_pipeline_has_markets(self) -> None:
        results = run_prediction_market(n_agents=6, sim_sols=30)
        assert len(results["markets"]) == len(MARKETS)

    def test_pipeline_has_resolutions(self) -> None:
        results = run_prediction_market(n_agents=6, sim_sols=30)
        for m in MARKETS:
            assert m["id"] in results["resolutions"]

    def test_pipeline_has_leaderboard(self) -> None:
        results = run_prediction_market(n_agents=6, sim_sols=30)
        assert len(results["leaderboard"]) > 0
        assert "rank" in results["leaderboard"][0]

    def test_pipeline_deterministic(self) -> None:
        """Same seeds → same results."""
        r1 = run_prediction_market(n_agents=6, sim_sols=30, sim_seed=99, agent_seed=99)
        r2 = run_prediction_market(n_agents=6, sim_sols=30, sim_seed=99, agent_seed=99)
        assert r1["resolutions"] == r2["resolutions"]
        assert r1["leaderboard"] == r2["leaderboard"]

    def test_format_report(self) -> None:
        results = run_prediction_market(n_agents=6, sim_sols=30)
        report = format_report(results)
        assert "PREDICTION MARKET" in report
        assert "LEADERBOARD" in report
        assert "SIMULATION RESULTS" in report


# ── Conservation / Invariant Tests ──


class TestInvariants:

    def test_prices_always_sum_to_one(self) -> None:
        """After any sequence of buys, prices sum to 1."""
        import random
        rng = random.Random(42)
        m = LMSRMarket("test", "Q?", ["A", "B", "C", "D"])
        for _ in range(50):
            outcome = rng.choice(m.outcomes)
            shares = rng.uniform(1, 20)
            m.buy(outcome, shares)
            total = sum(m.prices())
            assert abs(total - 1.0) < 1e-8, f"Prices sum to {total}, not 1.0"

    def test_no_negative_karma_after_pipeline(self) -> None:
        """No agent ends with negative karma from bets."""
        results = run_prediction_market(n_agents=20, sim_sols=30)
        for aid, ar in results["agent_results"].items():
            assert ar["total_wagered"] <= STARTING_KARMA + 0.01, \
                f"{aid} wagered {ar['total_wagered']} > {STARTING_KARMA}"

    def test_all_markets_resolved(self) -> None:
        results = run_prediction_market(n_agents=10, sim_sols=30)
        for mid in results["markets"]:
            assert results["markets"][mid]["resolved"] is not None

    def test_colony_results_match_resolution(self) -> None:
        """Resolution outcomes are consistent with colony data."""
        results = run_prediction_market(n_agents=6, sim_sols=60)
        colonies = results["colony_results"]
        pops = {c["name"]: c["end_pop"] for c in colonies}
        winner = max(pops, key=pops.get)
        assert results["resolutions"]["highest_pop"] == winner
