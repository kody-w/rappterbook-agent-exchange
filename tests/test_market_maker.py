"""
Tests for market_maker.py — Prediction Market Engine.

Coverage: scoring functions, agent generation, market lifecycle,
settlement zero-sum, calibration, determinism, physical bounds.

Run: python -m pytest tests/test_market_maker.py -v
"""
from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.market_maker import (
    Agent,
    Forecast,
    Market,
    PredictionMarket,
    brier_score,
    log_score,
    clamp,
    deterministic_seed,
    generate_agents,
    generate_markets,
    submit_forecasts,
    resolve_markets,
    score_forecasts,
    settle_payouts,
    build_leaderboard,
    archetype_summary,
    ARCHETYPES,
    STARTING_KARMA,
    MIN_STAKE,
)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

class TestBrierScore:
    """Brier score: (predicted - outcome)^2."""

    def test_perfect_yes(self):
        assert brier_score(1.0, True) == 0.0

    def test_perfect_no(self):
        assert brier_score(0.0, False) == 0.0

    def test_worst_yes(self):
        assert brier_score(0.0, True) == 1.0

    def test_worst_no(self):
        assert brier_score(1.0, False) == 1.0

    def test_midpoint(self):
        assert brier_score(0.5, True) == pytest.approx(0.25)
        assert brier_score(0.5, False) == pytest.approx(0.25)

    def test_bounded_0_to_1(self):
        """Brier score is always in [0, 1]."""
        rng = random.Random(42)
        for _ in range(1000):
            p = rng.random()
            outcome = rng.choice([True, False])
            bs = brier_score(p, outcome)
            assert 0.0 <= bs <= 1.0

    def test_symmetric_around_half(self):
        """Brier(0.7, True) == Brier(0.3, False)."""
        assert brier_score(0.7, True) == pytest.approx(brier_score(0.3, False))


class TestLogScore:
    """Logarithmic scoring rule."""

    def test_confident_correct(self):
        assert log_score(0.99, True) == pytest.approx(math.log(0.99))

    def test_confident_wrong(self):
        assert log_score(0.99, False) == pytest.approx(math.log(0.01))

    def test_midpoint(self):
        assert log_score(0.5, True) == pytest.approx(math.log(0.5))

    def test_always_nonpositive(self):
        rng = random.Random(42)
        for _ in range(1000):
            p = clamp(rng.random(), 0.01, 0.99)
            outcome = rng.choice([True, False])
            ls = log_score(p, outcome)
            assert ls <= 0.0 + 1e-10

    def test_handles_extreme_probabilities(self):
        """Should not crash on 0.0 or 1.0."""
        ls = log_score(0.0, True)
        assert math.isfinite(ls)
        ls = log_score(1.0, False)
        assert math.isfinite(ls)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestClamp:

    def test_within_bounds(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below(self):
        assert clamp(-1.0, 0.0, 1.0) == 0.0

    def test_above(self):
        assert clamp(2.0, 0.0, 1.0) == 1.0


class TestDeterministicSeed:

    def test_same_input_same_output(self):
        a = deterministic_seed("test", 1)
        b = deterministic_seed("test", 1)
        assert a == b

    def test_different_inputs(self):
        a = deterministic_seed("test", 1)
        b = deterministic_seed("test", 2)
        assert a != b


# ---------------------------------------------------------------------------
# Agent generation
# ---------------------------------------------------------------------------

class TestGenerateAgents:

    def test_correct_count(self):
        agents = generate_agents(20, random.Random(42))
        assert len(agents) == 20

    def test_unique_ids(self):
        agents = generate_agents(50, random.Random(42))
        ids = [a.agent_id for a in agents]
        assert len(set(ids)) == 50

    def test_all_start_with_karma(self):
        agents = generate_agents(10, random.Random(42))
        for a in agents:
            assert a.karma == STARTING_KARMA

    def test_archetypes_distributed(self):
        agents = generate_agents(30, random.Random(42))
        archetypes = {a.archetype for a in agents}
        assert len(archetypes) >= 3  # at least 3 different archetypes


# ---------------------------------------------------------------------------
# Market generation
# ---------------------------------------------------------------------------

class TestGenerateMarkets:

    def test_correct_count(self):
        markets = generate_markets(10, 1, random.Random(42))
        assert len(markets) == 10

    def test_unique_ids(self):
        markets = generate_markets(15, 1, random.Random(42))
        ids = [m.market_id for m in markets]
        assert len(set(ids)) == 15

    def test_base_probabilities_bounded(self):
        markets = generate_markets(100, 1, random.Random(42))
        for m in markets:
            assert 0.05 <= m.base_probability <= 0.95

    def test_resolution_after_creation(self):
        markets = generate_markets(20, 5, random.Random(42))
        for m in markets:
            assert m.resolution_round > m.created_round


# ---------------------------------------------------------------------------
# Market price
# ---------------------------------------------------------------------------

class TestMarketPrice:

    def test_empty_market_is_half(self):
        m = Market("test", "Q?", "cat", 0.5, 1, 10)
        assert m.current_price == 0.5

    def test_single_forecast_is_that_forecast(self):
        m = Market("test", "Q?", "cat", 0.5, 1, 10)
        m.forecasts.append(Forecast("a1", "test", 0.8, 100, 1))
        assert m.current_price == pytest.approx(0.8)

    def test_weighted_average(self):
        m = Market("test", "Q?", "cat", 0.5, 1, 10)
        m.forecasts.append(Forecast("a1", "test", 0.9, 100, 1))
        m.forecasts.append(Forecast("a2", "test", 0.1, 100, 1))
        assert m.current_price == pytest.approx(0.5)

    def test_stake_weighted(self):
        m = Market("test", "Q?", "cat", 0.5, 1, 10)
        m.forecasts.append(Forecast("a1", "test", 0.9, 300, 1))
        m.forecasts.append(Forecast("a2", "test", 0.1, 100, 1))
        expected = (0.9 * 300 + 0.1 * 100) / 400
        assert m.current_price == pytest.approx(expected)

    def test_price_bounded(self):
        m = Market("test", "Q?", "cat", 0.5, 1, 10)
        m.forecasts.append(Forecast("a1", "test", 0.001, 100, 1))
        assert 0.01 <= m.current_price <= 0.99


# ---------------------------------------------------------------------------
# Forecast submission
# ---------------------------------------------------------------------------

class TestSubmitForecasts:

    def test_produces_forecasts(self):
        agents = generate_agents(10, random.Random(42))
        markets = generate_markets(5, 1, random.Random(42))
        forecasts = submit_forecasts(agents, markets, 1, random.Random(42))
        assert len(forecasts) > 0

    def test_forecasts_in_valid_range(self):
        agents = generate_agents(20, random.Random(42))
        markets = generate_markets(5, 1, random.Random(42))
        forecasts = submit_forecasts(agents, markets, 1, random.Random(42))
        for f in forecasts:
            assert 0.01 <= f.probability <= 0.99
            assert f.stake >= MIN_STAKE

    def test_agents_lose_karma_on_stake(self):
        agents = generate_agents(5, random.Random(42))
        markets = generate_markets(3, 1, random.Random(42))
        initial_karma = {a.agent_id: a.karma for a in agents}
        submit_forecasts(agents, markets, 1, random.Random(42))
        for a in agents:
            assert a.karma <= initial_karma[a.agent_id]

    def test_broke_agents_skip(self):
        agents = generate_agents(3, random.Random(42))
        for a in agents:
            a.karma = 1.0  # below MIN_STAKE
        markets = generate_markets(3, 1, random.Random(42))
        forecasts = submit_forecasts(agents, markets, 1, random.Random(42))
        assert len(forecasts) == 0


# ---------------------------------------------------------------------------
# Resolution and scoring
# ---------------------------------------------------------------------------

class TestResolveMarkets:

    def test_resolves_at_correct_round(self):
        markets = generate_markets(5, 1, random.Random(42))
        for m in markets:
            m.resolution_round = 10
        resolved = resolve_markets(markets, 10, random.Random(42))
        assert len(resolved) == 5
        for m in resolved:
            assert m.resolved
            assert m.outcome is not None

    def test_does_not_resolve_early(self):
        markets = generate_markets(5, 1, random.Random(42))
        for m in markets:
            m.resolution_round = 20
        resolved = resolve_markets(markets, 10, random.Random(42))
        assert len(resolved) == 0

    def test_double_resolution_idempotent(self):
        markets = generate_markets(1, 1, random.Random(42))
        markets[0].resolution_round = 5
        resolve_markets(markets, 5, random.Random(42))
        outcome1 = markets[0].outcome
        resolve_markets(markets, 5, random.Random(42))
        assert markets[0].outcome == outcome1


class TestScoreForecasts:

    def test_scores_computed(self):
        market = Market("m1", "Q?", "cat", 0.7, 1, 5)
        market.forecasts.append(Forecast("a1", "m1", 0.8, 100, 1))
        market.outcome = True
        market.resolved = True
        agent = Agent(agent_id="a1", archetype="calibrated")
        scores = score_forecasts([market], {"a1": agent})
        assert "m1" in scores
        assert len(scores["m1"]) == 1
        assert len(agent.brier_scores) == 1
        assert 0 <= agent.brier_scores[0] <= 1


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

class TestSettlements:

    def test_zero_sum(self):
        """Total karma out == total karma in (zero-sum market)."""
        market = Market("m1", "Q?", "cat", 0.7, 1, 5)
        agents = {f"a{i}": Agent(agent_id=f"a{i}", archetype="calibrated") for i in range(5)}
        rng = random.Random(42)
        for aid in agents:
            f = Forecast(aid, "m1", rng.random(), 100, 1)
            market.forecasts.append(f)
        market.outcome = True
        market.resolved = True
        settle_payouts([market], agents)
        total_payout = sum(a.total_payout for a in agents.values())
        total_staked = sum(f.stake for f in market.forecasts)
        assert total_payout == pytest.approx(total_staked, rel=0.01)

    def test_better_forecaster_gets_more(self):
        """Agent who predicted closer to truth gets larger share."""
        market = Market("m1", "Q?", "cat", 0.9, 1, 5)
        good = Agent(agent_id="good", archetype="calibrated")
        bad = Agent(agent_id="bad", archetype="random")
        market.forecasts = [
            Forecast("good", "m1", 0.9, 100, 1),
            Forecast("bad", "m1", 0.1, 100, 1),
        ]
        market.outcome = True
        market.resolved = True
        agents = {"good": good, "bad": bad}
        settle_payouts([market], agents)
        assert good.total_payout > bad.total_payout


# ---------------------------------------------------------------------------
# Leaderboard and archetype summary
# ---------------------------------------------------------------------------

class TestLeaderboard:

    def test_ranked_by_brier(self):
        agents = [
            Agent(agent_id="a1", archetype="calibrated", brier_scores=[0.1]),
            Agent(agent_id="a2", archetype="overconfident", brier_scores=[0.5]),
            Agent(agent_id="a3", archetype="anchored", brier_scores=[0.3]),
        ]
        lb = build_leaderboard(agents)
        assert lb[0]["agent_id"] == "a1"
        assert lb[1]["agent_id"] == "a3"
        assert lb[2]["agent_id"] == "a2"

    def test_excludes_unscored(self):
        agents = [
            Agent(agent_id="a1", archetype="calibrated", brier_scores=[0.2]),
            Agent(agent_id="a2", archetype="random"),  # no scores
        ]
        lb = build_leaderboard(agents)
        assert len(lb) == 1


class TestArchetypeSummary:

    def test_groups_by_archetype(self):
        agents = [
            Agent(agent_id="a1", archetype="calibrated", brier_scores=[0.1, 0.2]),
            Agent(agent_id="a2", archetype="calibrated", brier_scores=[0.15, 0.25]),
            Agent(agent_id="a3", archetype="random", brier_scores=[0.5]),
        ]
        summary = archetype_summary(agents)
        assert len(summary) == 2
        arch_names = [s["archetype"] for s in summary]
        assert "calibrated" in arch_names
        assert "random" in arch_names


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_records_bins(self):
        agent = Agent(agent_id="test", archetype="calibrated")
        agent.record_calibration(0.75, True)
        agent.record_calibration(0.72, False)
        agent.record_calibration(0.78, True)
        curve = agent.calibration_curve()
        assert len(curve) >= 1
        # All predictions were in the 70% bin
        for midpoint, actual, count in curve:
            assert count >= 1

    def test_perfect_calibration(self):
        """If you predict 80% and 80% of outcomes are True, you're calibrated."""
        agent = Agent(agent_id="test", archetype="calibrated")
        for i in range(100):
            agent.record_calibration(0.85, i < 80)
        curve = agent.calibration_curve()
        for midpoint, actual, count in curve:
            if count >= 10:
                assert abs(midpoint - actual) < 0.2


# ---------------------------------------------------------------------------
# Full simulation smoke tests
# ---------------------------------------------------------------------------

class TestSimulationSmoke:

    def test_runs_10_rounds(self):
        sim = PredictionMarket(n_agents=10, n_rounds=10, n_markets=5, seed=42)
        results = sim.run()
        assert results["_meta"]["engine"] == "market-maker"
        assert results["_meta"]["rounds"] == 10

    def test_runs_50_rounds(self):
        sim = PredictionMarket(n_agents=30, n_rounds=50, n_markets=15, seed=42)
        results = sim.run()
        assert results["summary"]["resolved_markets"] > 0
        assert results["summary"]["total_forecasts"] > 0
        assert len(results["leaderboard"]) > 0

    def test_deterministic(self):
        """Same seed → same results."""
        r1 = PredictionMarket(seed=99).run()
        r2 = PredictionMarket(seed=99).run()
        assert r1["summary"] == r2["summary"]
        assert len(r1["leaderboard"]) == len(r2["leaderboard"])
        for a, b in zip(r1["leaderboard"], r2["leaderboard"]):
            assert a["agent_id"] == b["agent_id"]
            assert a["mean_brier"] == b["mean_brier"]

    def test_different_seeds_differ(self):
        r1 = PredictionMarket(seed=1).run()
        r2 = PredictionMarket(seed=2).run()
        # Different seeds should produce different archetype rankings
        assert r1["summary"]["mean_brier_all"] != r2["summary"]["mean_brier_all"]


class TestPhysicalBounds:
    """Property-based invariants that must always hold."""

    def test_karma_nonnegative(self):
        sim = PredictionMarket(n_agents=30, n_rounds=50, seed=42)
        sim.run()
        for a in sim.agents:
            assert a.karma >= 0, f"{a.agent_id} has negative karma: {a.karma}"

    def test_brier_scores_bounded(self):
        sim = PredictionMarket(n_agents=20, n_rounds=30, seed=42)
        sim.run()
        for a in sim.agents:
            for bs in a.brier_scores:
                assert 0.0 <= bs <= 1.0

    def test_probabilities_bounded(self):
        sim = PredictionMarket(n_agents=20, n_rounds=20, seed=42)
        sim.run()
        for m in sim.markets:
            for f in m.forecasts:
                assert 0.01 <= f.probability <= 0.99

    def test_market_outcomes_binary(self):
        sim = PredictionMarket(n_agents=10, n_rounds=30, seed=42)
        sim.run()
        for m in sim.markets:
            if m.resolved:
                assert m.outcome in (True, False)

    def test_total_karma_conserved(self):
        """Karma is zero-sum across settlements. Total karma ≈ initial."""
        sim = PredictionMarket(n_agents=20, n_rounds=50, seed=42)
        initial_total = sum(a.karma for a in sim.agents)
        sim.run()
        final_total = sum(a.karma for a in sim.agents)
        # Unresolved markets hold staked karma, so final <= initial
        # But resolved karma is recycled, so should be close
        assert final_total <= initial_total * 1.01  # no karma creation
        assert final_total > 0  # not all karma lost

    def test_calibrated_beats_random(self):
        """Over many rounds, calibrated archetype should outperform random."""
        sim = PredictionMarket(n_agents=40, n_rounds=100, n_markets=20, seed=42)
        sim.run()
        summary = archetype_summary(sim.agents)
        by_arch = {s["archetype"]: s for s in summary}
        if "calibrated" in by_arch and "random" in by_arch:
            assert by_arch["calibrated"]["mean_brier"] < by_arch["random"]["mean_brier"]


class TestRoundLog:

    def test_round_log_length(self):
        sim = PredictionMarket(n_rounds=25, seed=42)
        results = sim.run()
        assert len(results["round_log"]) == 25

    def test_round_numbers_sequential(self):
        sim = PredictionMarket(n_rounds=15, seed=42)
        results = sim.run()
        for i, entry in enumerate(results["round_log"]):
            assert entry["round"] == i + 1

    def test_active_markets_nonnegative(self):
        sim = PredictionMarket(n_rounds=30, seed=42)
        results = sim.run()
        for entry in results["round_log"]:
            assert entry["active_markets"] >= 0
