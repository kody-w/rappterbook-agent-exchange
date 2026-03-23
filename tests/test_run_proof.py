"""
Tests for run_proof.py — execution bridge and Collective Intelligence scoring.

Run: python -m pytest tests/test_run_proof.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_proof import collective_intelligence, run_proof, format_proof


def _run_proof_tmp(tmp_path, **kwargs):
    """Helper: run proof with memory in a temp dir."""
    defaults = dict(sols=30, n_predictions=20, seeds=[42], adaptive_rounds=2, n_agents=6)
    defaults.update(kwargs)
    defaults["memory_path"] = tmp_path / "memory.json"
    return run_proof(**defaults)


# ---------------------------------------------------------------------------
# Collective Intelligence scoring
# ---------------------------------------------------------------------------

class TestCollectiveIntelligence:
    """Tests for the CI scoring function."""

    def test_perfect_inputs_high_ci(self) -> None:
        ci = collective_intelligence(
            market_brier=0.05,
            consensus_acc=0.95,
            calibration=[{"mean_confidence": 0.8, "actual_rate": 0.8, "count": 10}],
            learning_improvement=0.10,
        )
        assert ci["ci_score"] > 0.7

    def test_random_inputs_moderate_ci(self) -> None:
        ci = collective_intelligence(
            market_brier=0.25,
            consensus_acc=0.50,
            calibration=[
                {"mean_confidence": 0.3, "actual_rate": 0.5, "count": 10},
                {"mean_confidence": 0.7, "actual_rate": 0.6, "count": 10},
            ],
            learning_improvement=0.02,
        )
        assert 0.3 < ci["ci_score"] < 0.8

    def test_terrible_inputs_low_ci(self) -> None:
        ci = collective_intelligence(
            market_brier=0.80,
            consensus_acc=0.20,
            calibration=[{"mean_confidence": 0.9, "actual_rate": 0.1, "count": 10}],
            learning_improvement=0.0,
        )
        assert ci["ci_score"] < 0.5

    def test_ci_bounded_zero_one(self) -> None:
        for brier in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for acc in [0.0, 0.5, 1.0]:
                ci = collective_intelligence(
                    market_brier=brier,
                    consensus_acc=acc,
                    calibration=[],
                    learning_improvement=0.05,
                )
                assert 0.0 <= ci["ci_score"] <= 1.0

    def test_ci_has_components(self) -> None:
        ci = collective_intelligence(0.3, 0.6, [], 0.05)
        assert "components" in ci
        assert "brier_signal" in ci["components"]
        assert "consensus_signal" in ci["components"]
        assert "calibration_signal" in ci["components"]
        assert "learning_signal" in ci["components"]

    def test_ci_rating_values(self) -> None:
        """Rating must be one of the defined values."""
        valid_ratings = {"strong", "moderate", "marginal", "weak"}
        for brier in [0.0, 0.3, 0.5, 0.8]:
            ci = collective_intelligence(brier, 0.5, [], 0.05)
            assert ci["rating"] in valid_ratings

    def test_empty_calibration_handled(self) -> None:
        ci = collective_intelligence(0.3, 0.5, [], 0.05)
        assert 0.0 <= ci["ci_score"] <= 1.0

    def test_zero_count_calibration_handled(self) -> None:
        ci = collective_intelligence(
            0.3, 0.5,
            [{"mean_confidence": 0.5, "actual_rate": 0.0, "count": 0}],
            0.05,
        )
        assert 0.0 <= ci["ci_score"] <= 1.0


# ---------------------------------------------------------------------------
# Full pipeline smoke tests
# ---------------------------------------------------------------------------

class TestRunProof:
    """Integration tests for the full execution proof pipeline."""

    def test_runs_without_crash(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path)
        assert isinstance(results, dict)

    def test_has_all_sections(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path)
        assert "_meta" in results
        assert "terrarium" in results
        assert "market" in results
        assert "adaptive" in results
        assert "ci" in results
        assert "memory" in results

    def test_terrarium_has_colonies(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=1)
        assert len(results["terrarium"]["colonies"]) == 3

    def test_market_has_predictions(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, adaptive_rounds=1)
        assert results["market"]["n_predictions"] == 20
        assert results["market"]["resolved"] > 0

    def test_adaptive_has_evolution(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=3)
        assert len(results["adaptive"]["evolution_curve"]) == 3

    def test_ci_present_and_bounded(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path)
        assert 0.0 <= results["ci"]["ci_score"] <= 1.0

    def test_memory_present(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path)
        assert results["memory"]["current_generation"] == 1
        assert results["memory"]["agents_tracked"] > 0

    def test_memory_persists_across_runs(self, tmp_path) -> None:
        r1 = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=1)
        r2 = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=1)
        assert r1["memory"]["current_generation"] == 1
        assert r2["memory"]["current_generation"] == 2
        assert r2["memory"]["previous_generation"] == 1

    def test_meta_has_proof_hash(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=1)
        assert len(results["_meta"]["proof_hash"]) == 16

    def test_meta_has_timing(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=1)
        assert results["_meta"]["total_time_ms"] > 0
        assert "terrarium" in results["_meta"]["breakdown_ms"]
        assert "memory" in results["_meta"]["breakdown_ms"]

    def test_json_serializable(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=1)
        json.dumps(results, default=str)  # must not raise

    def test_deterministic(self, tmp_path) -> None:
        mp1 = tmp_path / "m1.json"
        mp2 = tmp_path / "m2.json"
        r1 = run_proof(sols=30, n_predictions=10, seeds=[42], adaptive_rounds=2, n_agents=6, rng_seed=99, memory_path=mp1)
        r2 = run_proof(sols=30, n_predictions=10, seeds=[42], adaptive_rounds=2, n_agents=6, rng_seed=99, memory_path=mp2)
        assert r1["market"]["mean_brier"] == r2["market"]["mean_brier"]
        assert r1["ci"]["ci_score"] == r2["ci"]["ci_score"]


# ---------------------------------------------------------------------------
# Format proof tests
# ---------------------------------------------------------------------------

class TestFormatProof:
    """Tests for the human-readable proof formatter."""

    def test_format_produces_string(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=1)
        text = format_proof(results)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_format_contains_key_sections(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=1)
        text = format_proof(results)
        assert "TERRARIUM" in text
        assert "PREDICTION MARKET" in text
        assert "ADAPTIVE MARKET" in text
        assert "COLLECTIVE INTELLIGENCE" in text
        assert "MARKET MEMORY" in text

    def test_format_contains_colony_names(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=1)
        text = format_proof(results)
        assert "Ares Prime" in text
        assert "Olympus Station" in text
        assert "Red Frontier" in text


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------

class TestProofConservation:
    """Invariants that must hold in the proof pipeline."""

    def test_colony_populations_nonnegative(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, sols=50)
        for c in results["terrarium"]["colonies"]:
            assert c["end_pop"] >= 0

    def test_market_brier_bounded(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, sols=50)
        assert 0.0 <= results["market"]["mean_brier"] <= 1.0

    def test_accuracy_bounded(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, sols=50)
        assert 0.0 <= results["market"]["accuracy"] <= 1.0

    def test_resolved_leq_total(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, sols=50, n_predictions=30, adaptive_rounds=1)
        assert results["market"]["resolved"] <= results["market"]["n_predictions"]

    def test_timing_positive(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=1)
        for key, val in results["_meta"]["breakdown_ms"].items():
            assert val >= 0, f"{key} timing is negative"

    def test_memory_generation_positive(self, tmp_path) -> None:
        results = _run_proof_tmp(tmp_path, n_predictions=10, adaptive_rounds=1)
        assert results["memory"]["current_generation"] >= 1
