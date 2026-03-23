"""
Tests for adaptive_market.py — multi-round learning prediction market.

Run: python -m pytest tests/test_adaptive_market.py -v
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.adaptive_market import (
    AdaptiveAgent,
    AdaptiveMarketReport,
    BANKRUPTCY_KARMA,
    LEARNING_RATE,
    MAX_NOISE,
    MIN_NOISE,
    RoundResult,
    create_agent_pool,
    generate_adaptive_predictions,
    replace_bankrupt,
    run_adaptive_market,
)
from src.market_maker import Prediction


# ---------------------------------------------------------------------------
# AdaptiveAgent unit tests
# ---------------------------------------------------------------------------

class TestAdaptiveAgent:
    """Tests for the adaptive agent learning mechanics."""

    def test_initial_mean_brier(self) -> None:
        agent = AdaptiveAgent(name="t", archetype="oracle", bias=0.0, noise=0.05)
        assert agent.mean_brier == 0.5  # no predictions yet

    def test_good_round_tightens_noise(self) -> None:
        agent = AdaptiveAgent(name="t", archetype="oracle", bias=0.0, noise=0.10)
        original_noise = agent.noise
        agent.update(round_brier=0.10, round_preds=5)
        assert agent.noise < original_noise

    def test_bad_round_widens_noise(self) -> None:
        agent = AdaptiveAgent(name="t", archetype="degen", bias=0.0, noise=0.10)
        original_noise = agent.noise
        agent.update(round_brier=0.60, round_preds=5)
        assert agent.noise > original_noise

    def test_good_round_shrinks_bias(self) -> None:
        agent = AdaptiveAgent(name="t", archetype="optimist", bias=0.10, noise=0.05)
        agent.update(round_brier=0.10, round_preds=5)
        assert abs(agent.bias) < 0.10

    def test_noise_never_below_minimum(self) -> None:
        agent = AdaptiveAgent(name="t", archetype="oracle", bias=0.0, noise=MIN_NOISE)
        for _ in range(100):
            agent.update(round_brier=0.05, round_preds=5)
        assert agent.noise >= MIN_NOISE

    def test_noise_never_above_maximum(self) -> None:
        agent = AdaptiveAgent(name="t", archetype="degen", bias=0.0, noise=MAX_NOISE)
        for _ in range(100):
            agent.update(round_brier=0.90, round_preds=5)
        assert agent.noise <= MAX_NOISE

    def test_streak_increments_on_good_rounds(self) -> None:
        agent = AdaptiveAgent(name="t", archetype="oracle", bias=0.0, noise=0.05)
        agent.update(round_brier=0.10, round_preds=5)
        agent.update(round_brier=0.15, round_preds=5)
        agent.update(round_brier=0.20, round_preds=5)
        assert agent.streak == 3

    def test_streak_resets_on_bad_round(self) -> None:
        agent = AdaptiveAgent(name="t", archetype="oracle", bias=0.0, noise=0.05)
        agent.update(round_brier=0.10, round_preds=5)
        agent.update(round_brier=0.10, round_preds=5)
        agent.update(round_brier=0.60, round_preds=5)
        assert agent.streak == 0

    def test_to_dict_serializable(self) -> None:
        agent = AdaptiveAgent(name="t", archetype="oracle", bias=0.0, noise=0.05)
        agent.update(round_brier=0.15, round_preds=5)
        d = agent.to_dict()
        import json
        json.dumps(d)  # must not raise
        assert "name" in d
        assert "mean_brier" in d

    def test_best_worst_brier_tracked(self) -> None:
        agent = AdaptiveAgent(name="t", archetype="oracle", bias=0.0, noise=0.05)
        agent.update(round_brier=0.30, round_preds=5)
        agent.update(round_brier=0.10, round_preds=5)
        agent.update(round_brier=0.50, round_preds=5)
        assert agent.best_brier == 0.10
        assert agent.worst_brier == 0.50


# ---------------------------------------------------------------------------
# Agent pool tests
# ---------------------------------------------------------------------------

class TestAgentPool:
    """Tests for agent pool creation and management."""

    def test_creates_correct_count(self) -> None:
        rng = random.Random(42)
        agents = create_agent_pool(18, rng)
        assert len(agents) == 18

    def test_all_archetypes_represented(self) -> None:
        rng = random.Random(42)
        agents = create_agent_pool(18, rng)
        archetypes = set(a.archetype for a in agents)
        assert len(archetypes) == 6  # all 6 archetypes

    def test_unique_names(self) -> None:
        rng = random.Random(42)
        agents = create_agent_pool(18, rng)
        names = [a.name for a in agents]
        assert len(names) == len(set(names))

    def test_initial_karma_positive(self) -> None:
        rng = random.Random(42)
        agents = create_agent_pool(18, rng)
        for a in agents:
            assert a.karma > 0

    def test_replace_bankrupt_replaces(self) -> None:
        rng = random.Random(42)
        agents = create_agent_pool(6, rng)
        agents[0].karma = BANKRUPTCY_KARMA - 10
        agents[2].karma = BANKRUPTCY_KARMA - 1
        replaced = replace_bankrupt(agents, rng)
        assert replaced == 2
        assert agents[0].karma == 100.0  # fresh agent
        assert agents[2].karma == 100.0

    def test_replace_preserves_healthy(self) -> None:
        rng = random.Random(42)
        agents = create_agent_pool(6, rng)
        original_names = [a.name for a in agents]
        replaced = replace_bankrupt(agents, rng)
        assert replaced == 0
        assert [a.name for a in agents] == original_names


# ---------------------------------------------------------------------------
# Prediction generation tests
# ---------------------------------------------------------------------------

class TestAdaptivePredictions:
    """Tests for adaptive prediction generation."""

    def test_generates_correct_count(self) -> None:
        rng = random.Random(42)
        agents = create_agent_pool(6, rng)
        preds = generate_adaptive_predictions(agents, 3, 0, rng)
        assert len(preds) == 18  # 6 agents * 3 each

    def test_confidence_bounded(self) -> None:
        rng = random.Random(42)
        agents = create_agent_pool(18, rng)
        preds = generate_adaptive_predictions(agents, 5, 0, rng)
        for p in preds:
            assert 0.01 <= p.confidence <= 0.99

    def test_unique_prediction_ids(self) -> None:
        rng = random.Random(42)
        agents = create_agent_pool(12, rng)
        preds = generate_adaptive_predictions(agents, 5, 0, rng)
        ids = [p.id for p in preds]
        assert len(ids) == len(set(ids))

    def test_agent_names_match(self) -> None:
        rng = random.Random(42)
        agents = create_agent_pool(6, rng)
        preds = generate_adaptive_predictions(agents, 2, 0, rng)
        agent_names = set(a.name for a in agents)
        pred_agents = set(p.agent for p in preds)
        assert pred_agents == agent_names

    def test_adapted_agents_produce_different_predictions(self) -> None:
        """After adaptation, predictions should differ from initial round."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        agents = create_agent_pool(6, rng1)
        preds_before = generate_adaptive_predictions(agents, 3, 0, rng2)
        # Simulate adaptation
        for a in agents:
            a.update(round_brier=0.15, round_preds=3)
        rng3 = random.Random(42)
        preds_after = generate_adaptive_predictions(agents, 3, 1, rng3)
        # Confidence values should differ (agent noise has changed)
        confs_before = [p.confidence for p in preds_before]
        confs_after = [p.confidence for p in preds_after]
        assert confs_before != confs_after


# ---------------------------------------------------------------------------
# Full pipeline smoke tests
# ---------------------------------------------------------------------------

class TestAdaptiveMarketPipeline:
    """Integration tests for the full adaptive market."""

    def test_runs_without_crash(self) -> None:
        report = run_adaptive_market(
            rounds=2, n_agents=6, preds_per_agent=3, sols=50, seeds=[42],
        )
        assert isinstance(report, AdaptiveMarketReport)

    def test_correct_round_count(self) -> None:
        report = run_adaptive_market(
            rounds=3, n_agents=6, preds_per_agent=2, sols=30, seeds=[42],
        )
        assert len(report.rounds) == 3

    def test_all_rounds_have_predictions(self) -> None:
        report = run_adaptive_market(
            rounds=3, n_agents=6, preds_per_agent=3, sols=30, seeds=[42],
        )
        for r in report.rounds:
            assert r.n_predictions > 0
            assert r.n_resolved > 0

    def test_brier_bounded_all_rounds(self) -> None:
        report = run_adaptive_market(
            rounds=3, n_agents=6, preds_per_agent=3, sols=50, seeds=[42],
        )
        for r in report.rounds:
            assert 0.0 <= r.mean_brier <= 1.0

    def test_accuracy_bounded_all_rounds(self) -> None:
        report = run_adaptive_market(
            rounds=3, n_agents=6, preds_per_agent=3, sols=50, seeds=[42],
        )
        for r in report.rounds:
            assert 0.0 <= r.accuracy <= 1.0

    def test_evolution_curve_length(self) -> None:
        report = run_adaptive_market(
            rounds=4, n_agents=6, preds_per_agent=2, sols=30, seeds=[42],
        )
        assert len(report.evolution_curve) == 4

    def test_to_dict_serializable(self) -> None:
        report = run_adaptive_market(
            rounds=2, n_agents=6, preds_per_agent=2, sols=30, seeds=[42],
        )
        d = report.to_dict()
        import json
        json.dumps(d)  # must not raise

    def test_learning_signal_present(self) -> None:
        report = run_adaptive_market(
            rounds=3, n_agents=6, preds_per_agent=3, sols=50, seeds=[42],
        )
        d = report.to_dict()
        ls = d["learning_signal"]
        assert "learned" in ls
        assert "verdict" in ls
        assert "improvement" in ls

    def test_deterministic_same_seed(self) -> None:
        r1 = run_adaptive_market(
            rounds=2, n_agents=6, preds_per_agent=2, sols=30,
            seeds=[42], rng_seed=99,
        )
        r2 = run_adaptive_market(
            rounds=2, n_agents=6, preds_per_agent=2, sols=30,
            seeds=[42], rng_seed=99,
        )
        assert r1.rounds[0].mean_brier == r2.rounds[0].mean_brier
        assert r1.rounds[1].mean_brier == r2.rounds[1].mean_brier

    def test_5_round_market_produces_evolution(self) -> None:
        """5 rounds should show agent parameter evolution."""
        report = run_adaptive_market(
            rounds=5, n_agents=12, preds_per_agent=4, sols=100, seeds=[42, 43],
        )
        # Noise should change from initial to final
        first_noise = report.evolution_curve[0]["mean_noise"]
        last_noise = report.evolution_curve[-1]["mean_noise"]
        assert first_noise != last_noise  # some adaptation happened


# ---------------------------------------------------------------------------
# Conservation laws / invariants
# ---------------------------------------------------------------------------

class TestAdaptiveConservation:
    """Invariants that must hold across all rounds."""

    def test_agent_noise_bounded(self) -> None:
        report = run_adaptive_market(
            rounds=3, n_agents=12, preds_per_agent=3, sols=50, seeds=[42],
        )
        for agent in report.final_agents:
            assert MIN_NOISE <= agent["noise"] <= MAX_NOISE

    def test_predictions_per_round_consistent(self) -> None:
        report = run_adaptive_market(
            rounds=3, n_agents=6, preds_per_agent=4, sols=30, seeds=[42],
        )
        for r in report.rounds:
            assert r.n_predictions == 6 * 4

    def test_resolved_leq_total(self) -> None:
        report = run_adaptive_market(
            rounds=3, n_agents=6, preds_per_agent=3, sols=50, seeds=[42],
        )
        for r in report.rounds:
            assert r.n_resolved <= r.n_predictions

    def test_evolution_curve_brier_bounded(self) -> None:
        report = run_adaptive_market(
            rounds=5, n_agents=6, preds_per_agent=3, sols=50, seeds=[42],
        )
        for ec in report.evolution_curve:
            assert 0.0 <= ec["mean_brier"] <= 1.0
            assert ec["mean_noise"] >= MIN_NOISE
            assert ec["mean_abs_bias"] >= 0.0

    def test_final_agents_sorted_by_brier(self) -> None:
        report = run_adaptive_market(
            rounds=3, n_agents=12, preds_per_agent=3, sols=50, seeds=[42],
        )
        briers = [a["mean_brier"] for a in report.final_agents]
        assert briers == sorted(briers)

    def test_total_replaced_nonnegative(self) -> None:
        report = run_adaptive_market(
            rounds=3, n_agents=6, preds_per_agent=3, sols=50, seeds=[42],
        )
        assert report.total_replaced >= 0
