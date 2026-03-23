"""
Tests for market_maker.py — Prediction Market Engine.

Coverage: scoring functions, agent generation, market lifecycle,
settlement zero-sum, calibration, cross-sim resolution, physical bounds.

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
    resolve_from_terrarium,
    score_forecasts,
    settle_markets,
    build_leaderboard,
    build_archetype_performance,
    build_calibration,
    run_standalone,
    _extract_terrarium_observables,
    _resolve_market_from_observables,
    STARTING_KARMA,
    MIN_STAKE,
    ARCHETYPES,
    MARKET_CATEGORIES,
)


# ── Scoring ──────────────────────────────────────────────────


class TestBrierScore:
    def test_perfect_yes(self) -> None:
        assert brier_score(1.0, True) == 0.0

    def test_perfect_no(self) -> None:
        assert brier_score(0.0, False) == 0.0

    def test_worst_yes(self) -> None:
        assert brier_score(0.0, True) == 1.0

    def test_worst_no(self) -> None:
        assert brier_score(1.0, False) == 1.0

    def test_midpoint(self) -> None:
        assert abs(brier_score(0.5, True) - 0.25) < 1e-9

    def test_bounded_0_to_1(self) -> None:
        rng = random.Random(42)
        for _ in range(500):
            p = rng.uniform(0.0, 1.0)
            for o in [True, False]:
                bs = brier_score(p, o)
                assert 0.0 <= bs <= 1.0

    def test_symmetric_around_half(self) -> None:
        assert abs(brier_score(0.5, True) - brier_score(0.5, False)) < 1e-9


class TestLogScore:
    def test_confident_correct(self) -> None:
        assert log_score(0.99, True) > -0.02

    def test_confident_wrong(self) -> None:
        assert log_score(0.99, False) < -4.0

    def test_midpoint(self) -> None:
        lt = log_score(0.5, True)
        lf = log_score(0.5, False)
        assert abs(lt - lf) < 0.001

    def test_always_nonpositive(self) -> None:
        for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            for o in [True, False]:
                assert log_score(p, o) <= 0.0

    def test_handles_extreme_probabilities(self) -> None:
        assert log_score(0.0001, True) >= -10.0
        assert log_score(0.9999, False) >= -10.0


class TestClamp:
    def test_within_bounds(self) -> None:
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below(self) -> None:
        assert clamp(-1.0, 0.0, 1.0) == 0.0

    def test_above(self) -> None:
        assert clamp(2.0, 0.0, 1.0) == 1.0


class TestDeterministicSeed:
    def test_same_input_same_output(self) -> None:
        assert deterministic_seed("a", 1) == deterministic_seed("a", 1)

    def test_different_inputs(self) -> None:
        assert deterministic_seed("a", 1) != deterministic_seed("b", 1)


# ── Agent & Market Generation ────────────────────────────────


class TestGenerateAgents:
    def test_correct_count(self) -> None:
        agents = generate_agents(20)
        assert len(agents) == 20

    def test_unique_ids(self) -> None:
        agents = generate_agents(30)
        ids = [a.agent_id for a in agents]
        assert len(set(ids)) == 30

    def test_all_start_with_karma(self) -> None:
        for a in generate_agents(10):
            assert a.karma == STARTING_KARMA

    def test_archetypes_distributed(self) -> None:
        agents = generate_agents(60)
        types = {a.archetype for a in agents}
        assert len(types) >= 4


class TestGenerateMarkets:
    def test_correct_count(self) -> None:
        markets = generate_markets(10, 50)
        assert len(markets) == 10

    def test_unique_ids(self) -> None:
        markets = generate_markets(15, 50)
        ids = [m.market_id for m in markets]
        assert len(set(ids)) == 15

    def test_base_probabilities_bounded(self) -> None:
        for m in generate_markets(20, 50):
            assert 0.01 <= m.base_probability <= 0.99

    def test_resolution_after_creation(self) -> None:
        for m in generate_markets(15, 50):
            assert m.resolution_round > m.creation_round


# ── Market Pricing ───────────────────────────────────────────


class TestMarketPrice:
    def test_empty_market_is_base(self) -> None:
        m = Market("x", "test", "t", 0.6, 0, 10)
        assert m.current_price() == 0.6

    def test_single_forecast(self) -> None:
        m = Market("x", "test", "t", 0.5, 0, 10)
        m.forecasts.append(Forecast("a", "x", 0.8, 10.0, 0))
        assert abs(m.current_price() - 0.8) < 1e-6

    def test_stake_weighted(self) -> None:
        m = Market("x", "test", "t", 0.5, 0, 10)
        m.forecasts.append(Forecast("a", "x", 0.9, 90.0, 0))
        m.forecasts.append(Forecast("b", "x", 0.1, 10.0, 0))
        price = m.current_price()
        assert 0.7 < price < 0.9

    def test_price_bounded(self) -> None:
        m = Market("x", "test", "t", 0.5, 0, 10)
        m.forecasts.append(Forecast("a", "x", 0.001, 100.0, 0))
        assert m.current_price() >= 0.01


# ── Forecasting ──────────────────────────────────────────────


class TestSubmitForecasts:
    def test_produces_forecasts(self) -> None:
        agents = generate_agents(10, seed=42)
        markets = generate_markets(5, 50, seed=42)
        fcs = submit_forecasts(agents, markets, 10, seed=42)
        assert len(fcs) > 0

    def test_forecasts_in_valid_range(self) -> None:
        agents = generate_agents(10)
        markets = generate_markets(5, 50)
        fcs = submit_forecasts(agents, markets, 0)
        for f in fcs:
            assert 0.01 <= f.probability <= 0.99

    def test_agents_lose_karma_on_stake(self) -> None:
        agents = generate_agents(15, seed=42)
        markets = generate_markets(5, 50, seed=42)
        submit_forecasts(agents, markets, 15, seed=42)
        assert any(a.karma < STARTING_KARMA for a in agents)

    def test_broke_agents_skip(self) -> None:
        agents = [Agent("broke", "calibrated", 0.0, 0.1, karma=1.0)]
        markets = generate_markets(3, 50)
        fcs = submit_forecasts(agents, markets, 0)
        assert len(fcs) == 0


# ── Resolution ───────────────────────────────────────────────


class TestResolveMarkets:
    def test_resolves_at_correct_round(self) -> None:
        markets = [Market("m", "test", "t", 0.5, 0, 5)]
        resolved = resolve_markets(markets, 5)
        assert len(resolved) == 1
        assert markets[0].resolved is True

    def test_does_not_resolve_early(self) -> None:
        markets = [Market("m", "test", "t", 0.5, 0, 10)]
        resolved = resolve_markets(markets, 5)
        assert len(resolved) == 0

    def test_double_resolution_idempotent(self) -> None:
        markets = [Market("m", "test", "t", 0.5, 0, 5)]
        resolve_markets(markets, 5)
        outcome1 = markets[0].outcome
        resolve_markets(markets, 6)
        assert markets[0].outcome == outcome1


# ── Cross-sim Resolution ─────────────────────────────────────


class TestTerrariumResolution:
    def _mock_sim_results(self) -> dict:
        return {
            "summary": {
                "colonies": [
                    {"name": "Ares Prime", "end_pop": 211, "growth_pct": 75.7,
                     "total_births": 95, "total_deaths": 4, "net_migration": -2,
                     "techs_unlocked": 5, "death_causes": {"accident": 3, "starvation": 1}},
                    {"name": "Olympus Station", "end_pop": 121, "growth_pct": 51.0,
                     "total_births": 50, "total_deaths": 9, "net_migration": 3,
                     "techs_unlocked": 4, "death_causes": {"accident": 5, "radiation": 4}},
                    {"name": "Red Frontier", "end_pop": 132, "growth_pct": 119.3,
                     "total_births": 82, "total_deaths": 10, "net_migration": -1,
                     "techs_unlocked": 6, "death_causes": {"accident": 7, "starvation": 3}},
                ],
            },
            "environment": {
                "history": [
                    {"sol": 1, "storm": None, "flare": False,
                     "temperature_c": -40.0, "radiation_msv": 0.6},
                    {"sol": 100, "storm": "regional", "flare": False,
                     "temperature_c": -30.0, "radiation_msv": 0.5},
                    {"sol": 200, "storm": "global", "flare": True,
                     "temperature_c": -70.0, "radiation_msv": 5.5},
                    {"sol": 300, "storm": None, "flare": False,
                     "temperature_c": -50.0, "radiation_msv": 0.65},
                ],
            },
            "colonies": [
                {"name": "Ares Prime", "events": [
                    {"type": "epidemic_start", "sol": 150},
                ], "history": [
                    {"genetic_diversity": 0.95}, {"genetic_diversity": 0.75},
                ]},
                {"name": "Olympus Station", "events": [], "history": [
                    {"genetic_diversity": 0.90},
                ]},
                {"name": "Red Frontier", "events": [], "history": [
                    {"genetic_diversity": 0.88},
                ]},
            ],
        }

    def test_extract_observables(self) -> None:
        sim = self._mock_sim_results()
        obs = _extract_terrarium_observables(
            sim["summary"]["colonies"],
            sim["environment"]["history"],
            sim["colonies"],
        )
        assert obs["all_survived"] is True
        assert obs["any_above_200"] is True
        assert obs["had_global_storm"] is True
        assert obs["had_flare"] is True
        assert obs["had_epidemic"] is True
        assert obs["total_births"] == 227
        assert obs["total_deaths"] == 23
        assert obs["max_techs"] == 6

    def test_resolve_colony_survival(self) -> None:
        m = Market("m", "colony_survival", "test", 0.5, 0, 10)
        obs = {"all_survived": True}
        assert _resolve_market_from_observables(m, obs) is True

    def test_resolve_dust_storm(self) -> None:
        m = Market("m", "dust_storm", "test", 0.5, 0, 10)
        obs = {"had_global_storm": True}
        assert _resolve_market_from_observables(m, obs) is True

    def test_resolve_tech_unlock(self) -> None:
        m = Market("m", "tech_unlock", "test", 0.5, 0, 10)
        obs = {"max_techs": 5}
        assert _resolve_market_from_observables(m, obs) is True

    def test_resolve_epidemic(self) -> None:
        m = Market("m", "epidemic_outbreak", "test", 0.5, 0, 10)
        obs = {"had_epidemic": True}
        assert _resolve_market_from_observables(m, obs) is True

    def test_full_cross_sim_pipeline(self) -> None:
        sim = self._mock_sim_results()
        markets = generate_markets(15, 50)
        for m in markets:
            m.forecasts.append(Forecast("test-agent", m.market_id, 0.6, 10.0, 0))
        resolved = resolve_from_terrarium(markets, sim)
        assert len(resolved) > 0
        for m in resolved:
            assert m.outcome is not None


# ── Settlement ───────────────────────────────────────────────


class TestSettlements:
    def test_zero_sum(self) -> None:
        """Total karma change across all agents sums to ~0 per market."""
        agents = generate_agents(10, seed=99)
        markets = generate_markets(5, 20, seed=99)
        for rnd in range(20):
            submit_forecasts(agents, markets, rnd, seed=99)
            resolve_markets(markets, rnd, seed=99)
        deltas = settle_markets(markets, agents)
        total_delta = sum(deltas.values())
        assert abs(total_delta) < 1.0  # rounding tolerance

    def test_better_forecaster_gets_more(self) -> None:
        m = Market("m", "test", "t", 0.7, 0, 1, outcome=True, resolved=True)
        good = Agent("good", "calibrated", 0.0, 0.05, karma=100)
        bad = Agent("bad", "random", 0.0, 0.3, karma=100)
        m.forecasts = [
            Forecast("good", "m", 0.9, 50.0, 0),
            Forecast("bad", "m", 0.2, 50.0, 0),
        ]
        settle_markets([m], [good, bad])
        assert good.karma > bad.karma


# ── Leaderboard & Calibration ────────────────────────────────


class TestLeaderboard:
    def test_ranked_by_brier(self) -> None:
        a1 = Agent("a", "calibrated", 0, 0, brier_scores=[0.1, 0.2])
        a2 = Agent("b", "random", 0, 0, brier_scores=[0.5, 0.6])
        board = build_leaderboard([a1, a2])
        assert board[0]["agent_id"] == "a"
        assert board[1]["agent_id"] == "b"

    def test_excludes_unscored(self) -> None:
        a1 = Agent("a", "calibrated", 0, 0, brier_scores=[0.1])
        a2 = Agent("b", "random", 0, 0, brier_scores=[])
        board = build_leaderboard([a1, a2])
        assert len(board) == 1


class TestArchetypeSummary:
    def test_groups_by_archetype(self) -> None:
        agents = [
            Agent("a1", "calibrated", 0, 0, brier_scores=[0.1]),
            Agent("a2", "calibrated", 0, 0, brier_scores=[0.2]),
            Agent("a3", "random", 0, 0, brier_scores=[0.5]),
        ]
        perf = build_archetype_performance(agents)
        assert len(perf) == 2
        cal = next(p for p in perf if p["archetype"] == "calibrated")
        assert cal["count"] == 2


class TestCalibration:
    def test_records_bins(self) -> None:
        m = Market("m", "test", "t", 0.5, 0, 1, outcome=True, resolved=True)
        m.forecasts = [
            Forecast("a", "m", 0.8, 10.0, 0, explicit_confidence=True),
            Forecast("b", "m", 0.3, 10.0, 0, explicit_confidence=True),
        ]
        curve = build_calibration([m])
        assert len(curve) == 5
        total = sum(b["count"] for b in curve)
        assert total == 2

    def test_excludes_imputed(self) -> None:
        m = Market("m", "test", "t", 0.5, 0, 1, outcome=True, resolved=True)
        m.forecasts = [
            Forecast("a", "m", 0.8, 10.0, 0, explicit_confidence=True),
            Forecast("b", "m", 0.3, 10.0, 0, explicit_confidence=False),
        ]
        curve = build_calibration([m])
        total = sum(b["count"] for b in curve)
        assert total == 1  # only explicit


# ── Simulation smoke tests ───────────────────────────────────


class TestSimulationSmoke:
    def test_runs_10_rounds(self) -> None:
        sim = PredictionMarket(n_agents=5, n_markets=3, n_rounds=10)
        results = sim.run()
        assert results["summary"]["rounds"] == 10

    def test_runs_50_rounds(self) -> None:
        sim = PredictionMarket(n_agents=10, n_markets=5, n_rounds=50)
        results = sim.run()
        assert results["summary"]["total_forecasts"] > 0

    def test_deterministic(self) -> None:
        r1 = PredictionMarket(n_agents=5, n_markets=3, n_rounds=10, seed=42).run()
        r2 = PredictionMarket(n_agents=5, n_markets=3, n_rounds=10, seed=42).run()
        assert r1["summary"]["total_forecasts"] == r2["summary"]["total_forecasts"]

    def test_different_seeds_differ(self) -> None:
        r1 = PredictionMarket(n_agents=10, n_markets=5, n_rounds=20, seed=1).run()
        r2 = PredictionMarket(n_agents=10, n_markets=5, n_rounds=20, seed=2).run()
        l1 = r1["leaderboard"]
        l2 = r2["leaderboard"]
        if l1 and l2:
            assert l1[0]["mean_brier"] != l2[0]["mean_brier"]

    def test_cross_sim_run(self) -> None:
        """Full pipeline with terrarium results fed in."""
        sim = PredictionMarket(n_agents=10, n_markets=15, n_rounds=20)
        mock_terrarium = {
            "summary": {"colonies": [
                {"name": "A", "end_pop": 200, "growth_pct": 50,
                 "total_births": 30, "total_deaths": 5, "net_migration": 2,
                 "techs_unlocked": 4, "death_causes": {"accident": 5}},
            ]},
            "environment": {"history": [
                {"sol": 1, "storm": "global", "flare": True,
                 "temperature_c": -60, "radiation_msv": 5.0},
            ]},
            "colonies": [{"name": "A", "events": [], "history": [
                {"genetic_diversity": 0.9},
            ]}],
        }
        results = sim.run(terrarium_results=mock_terrarium)
        assert results["summary"]["resolved_markets"] > 0


# ── Physical bounds / invariants ─────────────────────────────


class TestPhysicalBounds:
    def test_karma_bounded(self) -> None:
        """Karma can go negative from staking, but shouldn't be catastrophic."""
        sim = PredictionMarket(n_agents=20, n_markets=10, n_rounds=50)
        sim.run()
        for a in sim.agents:
            assert a.karma >= -500  # bounded loss

    def test_brier_scores_bounded(self) -> None:
        sim = PredictionMarket(n_agents=10, n_markets=5, n_rounds=30)
        sim.run()
        for a in sim.agents:
            for bs in a.brier_scores:
                assert 0.0 <= bs <= 1.0

    def test_probabilities_bounded(self) -> None:
        sim = PredictionMarket(n_agents=10, n_markets=5, n_rounds=20)
        sim.run()
        for m in sim.markets:
            for f in m.forecasts:
                assert 0.01 <= f.probability <= 0.99

    def test_market_outcomes_binary(self) -> None:
        sim = PredictionMarket(n_agents=5, n_markets=5, n_rounds=30)
        sim.run()
        for m in sim.markets:
            if m.resolved:
                assert m.outcome in (True, False)

    def test_settlement_zero_sum_per_market(self) -> None:
        """Settlement within each market should be zero-sum."""
        m = Market("m", "test", "t", 0.7, 0, 1, outcome=True, resolved=True)
        agents = [Agent(f"a{i}", "calibrated", 0, 0.1) for i in range(5)]
        for i, a in enumerate(agents):
            m.forecasts.append(Forecast(a.agent_id, "m", 0.5 + i * 0.1, 20.0, 0))
            a.karma -= 20.0  # simulate staking
        deltas = settle_markets([m], agents)
        total_delta = sum(deltas.values())
        assert abs(total_delta) < 1.0  # zero-sum within rounding

    def test_calibrated_beats_random(self) -> None:
        """Over 50 rounds, calibrated archetype should outperform random."""
        sim = PredictionMarket(n_agents=60, n_markets=15, n_rounds=50, seed=42)
        sim.run()
        perf = build_archetype_performance(sim.agents)
        cal = next((p for p in perf if p["archetype"] == "calibrated"), None)
        rnd = next((p for p in perf if p["archetype"] == "random"), None)
        if cal and rnd:
            assert cal["mean_brier"] <= rnd["mean_brier"]


class TestRoundLog:
    def test_round_log_length(self) -> None:
        sim = PredictionMarket(n_agents=5, n_markets=3, n_rounds=10)
        results = sim.run()
        assert len(results["round_log"]) == 10

    def test_round_numbers_sequential(self) -> None:
        sim = PredictionMarket(n_agents=5, n_markets=3, n_rounds=10)
        results = sim.run()
        rounds = [r["round"] for r in results["round_log"]]
        assert rounds == list(range(10))

    def test_active_markets_nonnegative(self) -> None:
        sim = PredictionMarket(n_agents=5, n_markets=3, n_rounds=10)
        results = sim.run()
        for r in results["round_log"]:
            assert r["active_markets"] >= 0
