"""
Tests for cross_sim.py — Cross-Simulation Bridge.

Coverage: signal extraction, confidence adjustment, full pipeline,
determinism, physical bounds, conservation laws.

Run: python -m pytest tests/test_cross_sim.py -v
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cross_sim import (
    extract_signals,
    adjust_confidences,
    run_cross_sim,
)
from src.tick_engine import Simulation
from src.market_maker import generate_predictions, generate_counter_positions


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

class TestExtractSignals:
    """Verify signal extraction from terrarium results."""

    @pytest.fixture
    def sim_results(self) -> dict:
        sim = Simulation(sols=100, env_seed=42)
        return sim.run()

    def test_has_population_signals(self, sim_results):
        signals = extract_signals(sim_results)
        for name in ["Ares Prime", "Olympus Station", "Red Frontier"]:
            assert f"{name}_final_pop" in signals
            assert f"{name}_pop_trend" in signals
            assert f"{name}_morale" in signals

    def test_population_positive(self, sim_results):
        signals = extract_signals(sim_results)
        for name in ["Ares Prime", "Olympus Station", "Red Frontier"]:
            assert signals[f"{name}_final_pop"] > 0

    def test_morale_bounded(self, sim_results):
        signals = extract_signals(sim_results)
        for name in ["Ares Prime", "Olympus Station", "Red Frontier"]:
            assert 0.0 <= signals[f"{name}_morale"] <= 1.0

    def test_has_environment_signals(self, sim_results):
        signals = extract_signals(sim_results)
        assert "had_global_storm" in signals
        assert "flare_count" in signals
        assert isinstance(signals["had_global_storm"], bool)

    def test_has_tech_signals(self, sim_results):
        signals = extract_signals(sim_results)
        for name in ["Ares Prime", "Olympus Station", "Red Frontier"]:
            assert f"{name}_techs" in signals
            assert isinstance(signals[f"{name}_techs"], int)

    def test_has_epidemic_signals(self, sim_results):
        signals = extract_signals(sim_results)
        for name in ["Ares Prime", "Olympus Station", "Red Frontier"]:
            assert f"{name}_had_epidemic" in signals
            assert isinstance(signals[f"{name}_had_epidemic"], bool)


# ---------------------------------------------------------------------------
# Confidence adjustment
# ---------------------------------------------------------------------------

class TestAdjustConfidences:
    """Verify that signals modify prediction confidences."""

    def _make_predictions(self, seed=42):
        colony_names = ["Ares Prime", "Olympus Station", "Red Frontier"]
        preds = generate_predictions(20, colony_names, seed)
        return preds

    def test_confidences_change(self):
        preds = self._make_predictions()
        original = [p.confidence for p in preds]
        signals = {
            "Ares Prime_pop_trend": 0.3,
            "Ares Prime_final_pop": 150,
            "Ares Prime_morale": 0.7,
            "Olympus Station_pop_trend": 0.1,
            "Olympus Station_final_pop": 90,
            "Olympus Station_morale": 0.6,
            "Red Frontier_pop_trend": 0.5,
            "Red Frontier_final_pop": 80,
            "Red Frontier_morale": 0.55,
            "had_global_storm": False,
            "flare_count": 1,
            "max_dust": 0.2,
            "total_migrations": 5,
        }
        for n in ["Ares Prime", "Olympus Station", "Red Frontier"]:
            signals[f"{n}_had_epidemic"] = False
            signals[f"{n}_epidemics"] = 0
            signals[f"{n}_techs"] = 2
            signals[f"{n}_food_kg"] = 500
            signals[f"{n}_k"] = 200
            signals[f"{n}_peak_pop"] = 160
        adjusted = adjust_confidences(preds, signals, 0.15)
        new = [p.confidence for p in adjusted]
        # At least some should change
        assert original != new

    def test_confidences_stay_bounded(self):
        preds = self._make_predictions()
        signals = {
            "Ares Prime_pop_trend": 1.0,
            "Ares Prime_final_pop": 500,
            "Ares Prime_morale": 0.95,
            "had_global_storm": True,
            "flare_count": 10,
            "max_dust": 0.9,
            "total_migrations": 100,
        }
        for n in ["Ares Prime", "Olympus Station", "Red Frontier"]:
            signals[f"{n}_had_epidemic"] = True
            signals[f"{n}_epidemics"] = 3
            signals[f"{n}_techs"] = 5
            signals[f"{n}_food_kg"] = 2000
            signals[f"{n}_k"] = 500
            signals[f"{n}_peak_pop"] = 500
            signals[f"{n}_pop_trend"] = 1.0
            signals[f"{n}_final_pop"] = 500
            signals[f"{n}_morale"] = 0.95
        adjusted = adjust_confidences(preds, signals, 0.5)
        for p in adjusted:
            assert 0.01 <= p.confidence <= 0.99, (
                f"Confidence {p.confidence} out of bounds"
            )


# ---------------------------------------------------------------------------
# Full pipeline smoke tests
# ---------------------------------------------------------------------------

class TestCrossSimSmoke:
    """Integration tests for the full cross-sim bridge."""

    def test_runs_without_crash(self):
        results = run_cross_sim(
            total_sols=100, n_rounds=3, n_predictions=20,
            seed=42, quiet=True,
        )
        assert "_meta" in results
        assert results["_meta"]["engine"] == "cross-sim-bridge"

    def test_has_consensus_accuracy(self):
        results = run_cross_sim(
            total_sols=100, n_rounds=2, n_predictions=20,
            seed=42, quiet=True,
        )
        ca = results["consensus_accuracy"]
        assert "total" in ca
        assert "correct" in ca
        assert 0.0 <= ca["accuracy"] <= 1.0

    def test_has_round_log(self):
        n_rounds = 3
        results = run_cross_sim(
            total_sols=100, n_rounds=n_rounds, n_predictions=20,
            seed=42, quiet=True,
        )
        assert len(results["round_log"]) == n_rounds

    def test_has_market_data(self):
        results = run_cross_sim(
            total_sols=100, n_rounds=2, n_predictions=20,
            seed=42, quiet=True,
        )
        assert "market" in results
        assert "leaderboard" in results["market"]
        assert "market_stats" in results["market"]

    def test_has_terrarium_summary(self):
        results = run_cross_sim(
            total_sols=100, n_rounds=2, n_predictions=20,
            seed=42, quiet=True,
        )
        ts = results["terrarium_summary"]
        assert "colonies" in ts
        assert len(ts["colonies"]) == 3

    def test_has_signal_history(self):
        results = run_cross_sim(
            total_sols=100, n_rounds=3, n_predictions=20,
            seed=42, quiet=True,
        )
        assert len(results["signal_history"]) == 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_seed_same_results(self):
        r1 = run_cross_sim(
            total_sols=100, n_rounds=2, n_predictions=20,
            seed=42, quiet=True,
        )
        r2 = run_cross_sim(
            total_sols=100, n_rounds=2, n_predictions=20,
            seed=42, quiet=True,
        )
        assert r1["consensus_accuracy"] == r2["consensus_accuracy"]
        assert r1["round_log"] == r2["round_log"]

    def test_different_seeds_may_differ(self):
        r1 = run_cross_sim(
            total_sols=365, n_rounds=2, n_predictions=20,
            seed=42, quiet=True,
        )
        r2 = run_cross_sim(
            total_sols=365, n_rounds=2, n_predictions=20,
            seed=99, quiet=True,
        )
        # Different seeds → different terrarium → different outcomes
        # Use consensus accuracy which depends on both market and terrarium
        assert r1["consensus_accuracy"] != r2["consensus_accuracy"]


# ---------------------------------------------------------------------------
# Physical invariants
# ---------------------------------------------------------------------------

class TestInvariants:

    def test_brier_scores_bounded(self):
        results = run_cross_sim(
            total_sols=100, n_rounds=2, n_predictions=30,
            seed=42, quiet=True,
        )
        stats = results["market"]["market_stats"]
        if stats.get("avg_brier") is not None:
            assert 0.0 <= stats["avg_brier"] <= 1.0

    def test_confidence_bounded(self):
        results = run_cross_sim(
            total_sols=100, n_rounds=3, n_predictions=30,
            seed=42, quiet=True,
        )
        stats = results["market"]["market_stats"]
        avg_conf = stats.get("avg_confidence", 0.5)
        assert 0.0 < avg_conf < 1.0

    @pytest.mark.parametrize("seed", [1, 13, 42, 77, 99])
    def test_completes_any_seed(self, seed):
        results = run_cross_sim(
            total_sols=50, n_rounds=2, n_predictions=10,
            seed=seed, quiet=True,
        )
        assert results["consensus_accuracy"]["total"] > 0

    @pytest.mark.parametrize("rounds", [1, 3, 7])
    def test_scales_with_rounds(self, rounds):
        results = run_cross_sim(
            total_sols=100, n_rounds=rounds, n_predictions=10,
            seed=42, quiet=True,
        )
        assert len(results["round_log"]) == rounds
