"""Tests for the adaptive prediction market."""
from __future__ import annotations

import math
import random
import statistics

import pytest

from src.adaptive_market import (
    AdaptiveAgent,
    AdaptiveMarketReport,
    MarketEvolution,
    RoundMetrics,
    compute_evolution,
    compute_round_metrics,
    create_agent_pool,
    format_adaptive_compact,
    format_adaptive_text,
    generate_round_predictions,
    resolve_and_score,
    run_adaptive_market,
    update_agents,
)
from src.market_maker import Prediction


# ---------------------------------------------------------------------------
# AdaptiveAgent unit tests
# ---------------------------------------------------------------------------


class TestAdaptiveAgent:
    """Tests for agent learning mechanics."""

    def test_record_updates_history(self):
        agent = AdaptiveAgent(name="a", archetype="oracle", bias=0.0, noise=0.05)
        agent.record(0.15, "survival")
        assert len(agent.history) == 1
        assert agent.history[0] == 0.15
        assert "survival" in agent.category_history
        assert len(agent.category_history["survival"]) == 1

    def test_adapt_good_performance_reduces_noise(self):
        agent = AdaptiveAgent(name="a", archetype="oracle", bias=0.0, noise=0.10)
        for _ in range(5):
            agent.record(0.10, "survival")
        initial_noise = agent.noise
        agent.adapt(learning_rate=0.15)
        assert agent.noise < initial_noise

    def test_adapt_bad_performance_increases_noise(self):
        agent = AdaptiveAgent(name="a", archetype="oracle", bias=0.0, noise=0.10)
        for _ in range(5):
            agent.record(0.70, "survival")
        initial_noise = agent.noise
        agent.adapt(learning_rate=0.15)
        assert agent.noise > initial_noise

    def test_adapt_clamps_noise(self):
        agent = AdaptiveAgent(name="a", archetype="degen", bias=0.0, noise=0.39)
        for _ in range(20):
            agent.record(0.90, "survival")
        agent.adapt(learning_rate=0.15)
        assert agent.noise <= 0.40

    def test_adapt_clamps_noise_floor(self):
        agent = AdaptiveAgent(name="a", archetype="analyst", bias=0.0, noise=0.02)
        for _ in range(20):
            agent.record(0.05, "survival")
        agent.adapt(learning_rate=0.15)
        assert agent.noise >= 0.01

    def test_adapt_increments_rounds_played(self):
        agent = AdaptiveAgent(name="a", archetype="oracle", bias=0.0, noise=0.05)
        agent.record(0.20, "survival")
        agent.adapt()
        assert agent.rounds_played == 1
        agent.adapt()
        assert agent.rounds_played == 2

    def test_adapt_no_history_is_noop(self):
        agent = AdaptiveAgent(name="a", archetype="oracle", bias=0.05, noise=0.10)
        agent.adapt()
        assert agent.bias == 0.05
        assert agent.noise == 0.10

    def test_category_bias_good_performance(self):
        agent = AdaptiveAgent(name="a", archetype="oracle", bias=0.0, noise=0.05)
        for _ in range(5):
            agent.record(0.10, "tech_unlock")
        bias = agent.category_bias("tech_unlock")
        assert bias < 0.0

    def test_category_bias_bad_performance(self):
        agent = AdaptiveAgent(name="a", archetype="oracle", bias=0.0, noise=0.05)
        for _ in range(5):
            agent.record(0.70, "epidemic_any")
        bias = agent.category_bias("epidemic_any")
        assert bias > 0.0

    def test_category_bias_unknown_category(self):
        agent = AdaptiveAgent(name="a", archetype="oracle", bias=0.0, noise=0.05)
        assert agent.category_bias("nonexistent") == 0.0

    def test_category_bias_too_few_samples(self):
        agent = AdaptiveAgent(name="a", archetype="oracle", bias=0.0, noise=0.05)
        agent.record(0.10, "survival")
        assert agent.category_bias("survival") == 0.0

    def test_snapshot_serializable(self):
        agent = AdaptiveAgent(name="test-01", archetype="oracle", bias=0.02, noise=0.06)
        agent.record(0.15, "survival")
        agent.record(0.20, "tech_unlock")
        snap = agent.snapshot()
        assert snap["name"] == "test-01"
        assert snap["archetype"] == "oracle"
        assert snap["n_predictions"] == 2
        assert isinstance(snap["mean_brier"], float)
        assert "survival" in snap["categories"]

    def test_snapshot_empty_history(self):
        agent = AdaptiveAgent(name="a", archetype="oracle", bias=0.0, noise=0.05)
        snap = agent.snapshot()
        assert snap["mean_brier"] is None
        assert snap["n_predictions"] == 0

    def test_bias_clamped_on_adapt(self):
        agent = AdaptiveAgent(name="a", archetype="contrarian", bias=-0.24, noise=0.10)
        for _ in range(10):
            agent.record(0.80, "survival")
        agent.adapt(learning_rate=0.15)
        assert agent.bias >= -0.25
        assert agent.bias <= 0.25


# ---------------------------------------------------------------------------
# Agent pool tests
# ---------------------------------------------------------------------------


class TestAgentPool:
    """Tests for agent pool creation."""

    def test_correct_count(self):
        pool = create_agent_pool(n_agents=12, seed=0)
        assert len(pool) == 12

    def test_deterministic(self):
        pool1 = create_agent_pool(n_agents=12, seed=42)
        pool2 = create_agent_pool(n_agents=12, seed=42)
        for a1, a2 in zip(pool1, pool2):
            assert a1.name == a2.name
            assert a1.noise == a2.noise
            assert a1.bias == a2.bias

    def test_diverse_archetypes(self):
        pool = create_agent_pool(n_agents=24, seed=0)
        archetypes = {a.archetype for a in pool}
        assert len(archetypes) >= 4

    def test_unique_names(self):
        pool = create_agent_pool(n_agents=24, seed=0)
        names = [a.name for a in pool]
        assert len(names) == len(set(names))

    def test_initial_bias_noise_physical(self):
        pool = create_agent_pool(n_agents=24, seed=99)
        for agent in pool:
            assert -0.25 <= agent.bias <= 0.25
            assert 0.01 <= agent.noise <= 0.40


# ---------------------------------------------------------------------------
# Prediction generation tests
# ---------------------------------------------------------------------------


class TestGenerateRoundPredictions:
    """Tests for round prediction generation."""

    def test_correct_count(self):
        agents = create_agent_pool(n_agents=6, seed=0)
        preds = generate_round_predictions(agents, preds_per_agent=3, round_num=0, rng=random.Random(0))
        assert len(preds) == 18

    def test_confidence_bounded(self):
        agents = create_agent_pool(n_agents=12, seed=0)
        preds = generate_round_predictions(agents, preds_per_agent=5, round_num=0, rng=random.Random(0))
        for p in preds:
            assert 0.01 <= p.confidence <= 0.99

    def test_stake_positive(self):
        agents = create_agent_pool(n_agents=6, seed=0)
        preds = generate_round_predictions(agents, preds_per_agent=2, round_num=0, rng=random.Random(0))
        for p in preds:
            assert p.stake > 0

    def test_deterministic(self):
        agents1 = create_agent_pool(n_agents=6, seed=0)
        agents2 = create_agent_pool(n_agents=6, seed=0)
        p1 = generate_round_predictions(agents1, 2, 0, random.Random(42))
        p2 = generate_round_predictions(agents2, 2, 0, random.Random(42))
        for a, b in zip(p1, p2):
            assert a.id == b.id
            assert a.confidence == b.confidence

    def test_unique_ids(self):
        agents = create_agent_pool(n_agents=12, seed=0)
        preds = generate_round_predictions(agents, 3, 0, random.Random(0))
        ids = [p.id for p in preds]
        assert len(ids) == len(set(ids))

    def test_all_agents_represented(self):
        agents = create_agent_pool(n_agents=6, seed=0)
        preds = generate_round_predictions(agents, 2, 0, random.Random(0))
        agent_names = {p.agent for p in preds}
        expected_names = {a.name for a in agents}
        assert agent_names == expected_names


# ---------------------------------------------------------------------------
# Resolve and score tests
# ---------------------------------------------------------------------------


class TestResolveAndScore:
    """Tests for prediction resolution against ensemble."""

    def test_all_resolved(self):
        from src.market_maker import run_terrarium_ensemble
        agents = create_agent_pool(n_agents=6, seed=0)
        preds = generate_round_predictions(agents, 2, 0, random.Random(0))
        ensemble = run_terrarium_ensemble(sols=50, seeds=[42, 43])
        resolve_and_score(preds, ensemble)
        for p in preds:
            assert p.outcome is not None

    def test_brier_bounded(self):
        from src.market_maker import run_terrarium_ensemble
        agents = create_agent_pool(n_agents=6, seed=0)
        preds = generate_round_predictions(agents, 2, 0, random.Random(0))
        ensemble = run_terrarium_ensemble(sols=50, seeds=[42])
        resolve_and_score(preds, ensemble)
        for p in preds:
            if p.brier is not None:
                assert 0.0 <= p.brier <= 1.0

    def test_payout_nonnegative(self):
        from src.market_maker import run_terrarium_ensemble
        agents = create_agent_pool(n_agents=6, seed=0)
        preds = generate_round_predictions(agents, 2, 0, random.Random(0))
        ensemble = run_terrarium_ensemble(sols=50, seeds=[42])
        resolve_and_score(preds, ensemble)
        for p in preds:
            if p.payout is not None:
                assert p.payout >= 0.0


# ---------------------------------------------------------------------------
# Update agents tests
# ---------------------------------------------------------------------------


class TestUpdateAgents:
    """Tests for agent adaptation feedback loop."""

    def test_agents_gain_history(self):
        from src.market_maker import run_terrarium_ensemble
        agents = create_agent_pool(n_agents=6, seed=0)
        preds = generate_round_predictions(agents, 3, 0, random.Random(0))
        ensemble = run_terrarium_ensemble(sols=50, seeds=[42])
        resolve_and_score(preds, ensemble)
        update_agents(agents, preds)
        for agent in agents:
            assert len(agent.history) > 0
            assert agent.rounds_played == 1

    def test_multiple_rounds_accumulate(self):
        from src.market_maker import run_terrarium_ensemble
        agents = create_agent_pool(n_agents=6, seed=0)
        ensemble = run_terrarium_ensemble(sols=50, seeds=[42])
        for r in range(3):
            preds = generate_round_predictions(agents, 2, r, random.Random(r))
            resolve_and_score(preds, ensemble)
            update_agents(agents, preds)
        for agent in agents:
            assert agent.rounds_played == 3
            assert len(agent.history) >= 3


# ---------------------------------------------------------------------------
# Round metrics tests
# ---------------------------------------------------------------------------


class TestRoundMetrics:
    """Tests for round-level metric computation."""

    def test_metrics_computed(self):
        from src.market_maker import run_terrarium_ensemble
        agents = create_agent_pool(n_agents=6, seed=0)
        preds = generate_round_predictions(agents, 3, 0, random.Random(0))
        ensemble = run_terrarium_ensemble(sols=50, seeds=[42])
        resolve_and_score(preds, ensemble)
        metrics = compute_round_metrics(0, preds, agents)
        assert metrics.round_num == 0
        assert metrics.n_predictions == 18
        assert metrics.n_resolved > 0
        assert 0.0 <= metrics.mean_brier <= 1.0
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.noise_mean > 0

    def test_to_dict_keys(self):
        metrics = RoundMetrics(round_num=0, n_predictions=10, n_resolved=8,
                               mean_brier=0.25, accuracy=0.6, noise_mean=0.1,
                               noise_stdev=0.02, bias_mean=0.01)
        d = metrics.to_dict()
        assert "round" in d
        assert "mean_brier" in d
        assert "accuracy" in d


# ---------------------------------------------------------------------------
# Evolution metrics tests
# ---------------------------------------------------------------------------


class TestEvolutionMetrics:
    """Tests for market evolution tracking."""

    def test_learning_rate_positive_when_improving(self):
        metrics = [
            RoundMetrics(round_num=0, mean_brier=0.35, accuracy=0.50, noise_mean=0.10),
            RoundMetrics(round_num=1, mean_brier=0.30, accuracy=0.55, noise_mean=0.09),
            RoundMetrics(round_num=2, mean_brier=0.25, accuracy=0.60, noise_mean=0.08),
        ]
        agents = create_agent_pool(n_agents=6, seed=0)
        evo = compute_evolution(metrics, agents)
        assert evo.learning_rate_estimate > 0

    def test_learning_rate_negative_when_degrading(self):
        metrics = [
            RoundMetrics(round_num=0, mean_brier=0.25, accuracy=0.60, noise_mean=0.08),
            RoundMetrics(round_num=1, mean_brier=0.30, accuracy=0.55, noise_mean=0.09),
            RoundMetrics(round_num=2, mean_brier=0.35, accuracy=0.50, noise_mean=0.10),
        ]
        agents = create_agent_pool(n_agents=6, seed=0)
        evo = compute_evolution(metrics, agents)
        assert evo.learning_rate_estimate < 0

    def test_convergence_detected(self):
        metrics = [
            RoundMetrics(round_num=0, mean_brier=0.30, accuracy=0.55, noise_mean=0.09),
            RoundMetrics(round_num=1, mean_brier=0.29, accuracy=0.56, noise_mean=0.09),
            RoundMetrics(round_num=2, mean_brier=0.295, accuracy=0.555, noise_mean=0.09),
        ]
        agents = create_agent_pool(n_agents=6, seed=0)
        evo = compute_evolution(metrics, agents)
        assert evo.converged is True

    def test_no_convergence_when_volatile(self):
        metrics = [
            RoundMetrics(round_num=0, mean_brier=0.30, accuracy=0.55, noise_mean=0.10),
            RoundMetrics(round_num=1, mean_brier=0.20, accuracy=0.65, noise_mean=0.08),
            RoundMetrics(round_num=2, mean_brier=0.35, accuracy=0.50, noise_mean=0.12),
        ]
        agents = create_agent_pool(n_agents=6, seed=0)
        evo = compute_evolution(metrics, agents)
        assert evo.converged is False

    def test_swarm_divergence_nonnegative(self):
        metrics = [
            RoundMetrics(round_num=0, mean_brier=0.30, accuracy=0.55, noise_mean=0.10),
        ]
        agents = create_agent_pool(n_agents=12, seed=0)
        evo = compute_evolution(metrics, agents)
        assert evo.swarm_divergence >= 0.0

    def test_single_round_no_crash(self):
        metrics = [RoundMetrics(round_num=0, mean_brier=0.30)]
        evo = compute_evolution(metrics, [])
        assert evo.learning_rate_estimate == 0.0

    def test_to_dict_has_label(self):
        evo = MarketEvolution(n_rounds=3, learning_rate_estimate=0.06)
        d = evo.to_dict()
        assert "label" in d
        assert "Fast learner" in d["label"]

    def test_label_categories(self):
        assert "Fast" in MarketEvolution(learning_rate_estimate=0.06).label()
        assert "Steady" in MarketEvolution(learning_rate_estimate=0.02).label()
        assert "Plateau" in MarketEvolution(learning_rate_estimate=0.0).label()
        assert "Degrading" in MarketEvolution(learning_rate_estimate=-0.05).label()


# ---------------------------------------------------------------------------
# Full pipeline smoke tests
# ---------------------------------------------------------------------------


class TestRunAdaptiveMarket:
    """Integration tests for the full adaptive market pipeline."""

    def test_smoke_no_crash(self):
        report = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2, market_seed=42)
        assert report.n_rounds == 3
        assert report.n_agents == 6
        assert len(report.rounds) == 3

    def test_all_rounds_populated(self):
        report = run_adaptive_market(
            n_rounds=4, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        for r in report.rounds:
            assert r["n_predictions"] > 0
            assert r["n_resolved"] > 0

    def test_brier_bounded_all_rounds(self):
        report = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        for r in report.rounds:
            assert 0.0 <= r["mean_brier"] <= 1.0

    def test_accuracy_bounded(self):
        report = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        for r in report.rounds:
            assert 0.0 <= r["accuracy"] <= 1.0

    def test_evolution_has_trajectory(self):
        report = run_adaptive_market(
            n_rounds=5, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        evo = report.evolution
        assert len(evo.brier_trajectory) == 5
        assert len(evo.accuracy_trajectory) == 5

    def test_leaderboard_populated(self):
        report = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        assert len(report.final_leaderboard) > 0
        for row in report.final_leaderboard:
            assert "agent" in row
            assert "mean_brier" in row

    def test_agent_snapshots_populated(self):
        report = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        assert len(report.agent_snapshots) > 0
        for snap in report.agent_snapshots:
            assert "name" in snap
            assert "noise" in snap
            assert snap["rounds_played"] >= 1

    def test_terrarium_summary_present(self):
        report = run_adaptive_market(
            n_rounds=2, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        ts = report.terrarium_summary
        assert "colonies" in ts
        assert len(ts["colonies"]) == 3
        assert "total_population" in ts

    def test_deterministic(self):
        r1 = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2, market_seed=42)
        r2 = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2, market_seed=42)
        for a, b in zip(r1.rounds, r2.rounds):
            assert a["mean_brier"] == b["mean_brier"]
            assert a["accuracy"] == b["accuracy"]

    def test_to_dict_serializable(self):
        import json
        report = run_adaptive_market(
            n_rounds=2, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        d = report.to_dict()
        s = json.dumps(d)
        assert len(s) > 100

    def test_total_predictions_correct(self):
        report = run_adaptive_market(
            n_rounds=3, n_agents=8, preds_per_agent=2,
            sols=50, n_seeds=2)
        d = report.to_dict()
        assert d["total_predictions"] == 3 * 8 * 2


# ---------------------------------------------------------------------------
# Formatting tests
# ---------------------------------------------------------------------------


class TestFormatting:
    """Tests for text output formatting."""

    def test_text_format_no_crash(self):
        report = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        text = format_adaptive_text(report)
        assert "ADAPTIVE PREDICTION MARKET" in text
        assert "ROUND-BY-ROUND" in text
        assert "MARKET EVOLUTION" in text

    def test_compact_format_no_crash(self):
        report = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        compact = format_adaptive_compact(report)
        assert "ADAPTIVE" in compact
        assert "Brier" in compact

    def test_text_contains_all_rounds(self):
        report = run_adaptive_market(
            n_rounds=4, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        text = format_adaptive_text(report)
        for i in range(4):
            assert f"R{i}" in text or str(i) in text


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    """Property-based invariants that must hold across all runs."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 123])
    def test_brier_always_bounded(self, seed):
        report = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2, market_seed=seed)
        for r in report.rounds:
            assert 0.0 <= r["mean_brier"] <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 123])
    def test_accuracy_always_bounded(self, seed):
        report = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2, market_seed=seed)
        for r in report.rounds:
            assert 0.0 <= r["accuracy"] <= 1.0

    @pytest.mark.parametrize("seed", [0, 42, 99])
    def test_agent_noise_stays_physical(self, seed):
        report = run_adaptive_market(
            n_rounds=5, n_agents=12, preds_per_agent=2,
            sols=50, n_seeds=2, market_seed=seed)
        for snap in report.agent_snapshots:
            assert 0.01 <= snap["noise"] <= 0.40

    def test_no_negative_predictions(self):
        report = run_adaptive_market(
            n_rounds=3, n_agents=6, preds_per_agent=2,
            sols=50, n_seeds=2)
        for r in report.rounds:
            assert r["n_predictions"] > 0
            assert r["n_resolved"] >= 0

    def test_evolution_trajectory_length_matches_rounds(self):
        for n_rounds in [2, 3, 5]:
            report = run_adaptive_market(
                n_rounds=n_rounds, n_agents=6, preds_per_agent=2,
                sols=50, n_seeds=2)
            assert len(report.evolution.brier_trajectory) == n_rounds
