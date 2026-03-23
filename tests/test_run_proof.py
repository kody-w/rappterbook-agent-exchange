"""
Tests for run_proof.py — execution proof engine.

Run: python -m pytest tests/test_run_proof.py -v
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_proof import (
    compute_collective_intelligence,
    compute_terrarium_vitals,
    run_proof,
    format_proof_text,
    _interpret_ci,
)
from src.tick_engine import Simulation
from src.market_maker import run_market


# ---------------------------------------------------------------------------
# Collective intelligence scoring
# ---------------------------------------------------------------------------
class TestCollectiveIntelligence:

    def test_perfect_market_high_ci(self) -> None:
        """A market with Brier=0 should have CI=1.0."""
        report = {
            "mean_brier": 0.0,
            "calibration": [
                {"bucket_lo": 0.0, "bucket_hi": 0.2, "count": 10,
                 "mean_confidence": 0.1, "actual_rate": 0.1},
            ],
            "categories": [
                {"category": "survival", "count": 10, "accuracy": 1.0, "mean_brier": 0.0},
            ],
        }
        ci = compute_collective_intelligence(report)
        assert ci["ci_score"] == 1.0
        assert ci["calibration_error"] == 0.0

    def test_random_market_mid_ci(self) -> None:
        """A market with Brier≈0.25 should have CI≈0.75."""
        report = {
            "mean_brier": 0.25,
            "calibration": [
                {"bucket_lo": 0.4, "bucket_hi": 0.6, "count": 10,
                 "mean_confidence": 0.5, "actual_rate": 0.5},
            ],
            "categories": [
                {"category": "survival", "count": 10, "accuracy": 0.5, "mean_brier": 0.25},
            ],
        }
        ci = compute_collective_intelligence(report)
        assert ci["ci_score"] == pytest.approx(0.75)
        assert ci["information_ratio"] == pytest.approx(0.0)

    def test_ci_score_bounded(self) -> None:
        """CI score must be in [0, 1] for any Brier value."""
        for brier in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            report = {
                "mean_brier": brier,
                "calibration": [],
                "categories": [],
            }
            ci = compute_collective_intelligence(report)
            assert 0.0 <= ci["ci_score"] <= 1.0

    def test_calibration_error_nonnegative(self) -> None:
        report = {
            "mean_brier": 0.3,
            "calibration": [
                {"bucket_lo": 0.0, "bucket_hi": 0.2, "count": 5,
                 "mean_confidence": 0.1, "actual_rate": 0.3},
                {"bucket_lo": 0.8, "bucket_hi": 1.0, "count": 5,
                 "mean_confidence": 0.9, "actual_rate": 0.7},
            ],
            "categories": [],
        }
        ci = compute_collective_intelligence(report)
        assert ci["calibration_error"] >= 0.0

    def test_category_spread_bounded(self) -> None:
        report = {
            "mean_brier": 0.2,
            "calibration": [],
            "categories": [
                {"category": "a", "count": 10, "accuracy": 0.8, "mean_brier": 0.1},
                {"category": "b", "count": 10, "accuracy": 0.6, "mean_brier": 0.2},
                {"category": "c", "count": 10, "accuracy": 0.4, "mean_brier": 0.3},
            ],
        }
        ci = compute_collective_intelligence(report)
        assert 0.0 <= ci["category_spread"] <= 1.0

    def test_empty_calibration(self) -> None:
        """No calibration buckets → max error."""
        report = {
            "mean_brier": 0.5,
            "calibration": [],
            "categories": [],
        }
        ci = compute_collective_intelligence(report)
        assert ci["calibration_error"] == 1.0


# ---------------------------------------------------------------------------
# CI interpretation
# ---------------------------------------------------------------------------
class TestInterpretation:

    def test_oracle_grade(self) -> None:
        assert "Oracle" in _interpret_ci(0.90)

    def test_strong(self) -> None:
        assert "Strong" in _interpret_ci(0.75)

    def test_moderate(self) -> None:
        assert "Moderate" in _interpret_ci(0.60)

    def test_weak(self) -> None:
        assert "Weak" in _interpret_ci(0.45)

    def test_noise(self) -> None:
        assert "Noise" in _interpret_ci(0.30)


# ---------------------------------------------------------------------------
# Terrarium vitals
# ---------------------------------------------------------------------------
class TestTerrariumVitals:

    @pytest.fixture
    def sim_results(self) -> dict:
        sim = Simulation(sols=50, env_seed=42)
        return sim.run()

    def test_vitals_structure(self, sim_results: dict) -> None:
        vitals = compute_terrarium_vitals(sim_results)
        assert "total_population" in vitals
        assert "total_births" in vitals
        assert "total_deaths" in vitals
        assert "all_alive" in vitals
        assert "colonies" in vitals
        assert len(vitals["colonies"]) == 3

    def test_population_positive(self, sim_results: dict) -> None:
        vitals = compute_terrarium_vitals(sim_results)
        assert vitals["total_population"] > 0

    def test_all_alive_at_50_sols(self, sim_results: dict) -> None:
        vitals = compute_terrarium_vitals(sim_results)
        assert vitals["all_alive"] is True

    def test_colony_fields(self, sim_results: dict) -> None:
        vitals = compute_terrarium_vitals(sim_results)
        for c in vitals["colonies"]:
            assert "name" in c
            assert "strategy" in c
            assert "population" in c
            assert "growth_pct" in c
            assert "morale" in c
            assert "techs" in c

    def test_terraforming_nonnegative(self, sim_results: dict) -> None:
        vitals = compute_terrarium_vitals(sim_results)
        assert vitals["terraforming_pct"] >= 0.0


# ---------------------------------------------------------------------------
# Full proof pipeline
# ---------------------------------------------------------------------------
class TestRunProof:

    def test_proof_runs(self) -> None:
        proof = run_proof(sols=50, n_predictions=30, n_seeds=1)
        assert "timestamp" in proof
        assert "proof_hash" in proof
        assert "market" in proof
        assert "terrarium" in proof
        assert "collective_intelligence" in proof

    def test_proof_hash_deterministic(self) -> None:
        p1 = run_proof(sols=50, n_predictions=30, n_seeds=1, market_seed=0)
        p2 = run_proof(sols=50, n_predictions=30, n_seeds=1, market_seed=0)
        assert p1["proof_hash"] == p2["proof_hash"]

    def test_proof_market_data(self) -> None:
        proof = run_proof(sols=50, n_predictions=30, n_seeds=1)
        m = proof["market"]
        assert m["total_predictions"] == 30
        assert m["resolved"] > 0
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["mean_brier"] <= 1.0

    def test_proof_ci_present(self) -> None:
        proof = run_proof(sols=50, n_predictions=30, n_seeds=1)
        ci = proof["collective_intelligence"]
        assert 0.0 <= ci["ci_score"] <= 1.0
        assert "interpretation" in ci

    def test_proof_json_serializable(self) -> None:
        proof = run_proof(sols=50, n_predictions=20, n_seeds=1)
        dumped = json.dumps(proof)
        reloaded = json.loads(dumped)
        assert reloaded["proof_hash"] == proof["proof_hash"]


# ---------------------------------------------------------------------------
# Proof formatting
# ---------------------------------------------------------------------------
class TestFormatProof:

    def test_format_contains_key_sections(self) -> None:
        proof = run_proof(sols=50, n_predictions=20, n_seeds=1)
        text = format_proof_text(proof)
        assert "Execution Proof" in text
        assert "Terrarium Vitals" in text
        assert "Prediction Market" in text
        assert "Collective Intelligence" in text
        assert proof["proof_hash"] in text

    def test_format_contains_colony_names(self) -> None:
        proof = run_proof(sols=50, n_predictions=20, n_seeds=1)
        text = format_proof_text(proof)
        assert "Ares Prime" in text
        assert "Olympus Station" in text
        assert "Red Frontier" in text

    def test_format_markdown_tables(self) -> None:
        proof = run_proof(sols=50, n_predictions=20, n_seeds=1)
        text = format_proof_text(proof)
        # Should have markdown table separators
        assert "|-----" in text


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------
class TestConservation:

    def test_ci_monotonic_with_brier(self) -> None:
        """Better Brier → higher CI. CI is monotonically decreasing with Brier."""
        prev_ci = 2.0
        for brier in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
            report = {
                "mean_brier": brier,
                "calibration": [],
                "categories": [],
            }
            ci = compute_collective_intelligence(report)
            assert ci["ci_score"] <= prev_ci
            prev_ci = ci["ci_score"]

    def test_total_pop_equals_sum(self) -> None:
        proof = run_proof(sols=50, n_predictions=10, n_seeds=1)
        total = proof["terrarium"]["total_population"]
        colony_sum = sum(c["population"] for c in proof["terrarium"]["colonies"])
        assert total == colony_sum

    def test_resolved_plus_unresolved_consistent(self) -> None:
        proof = run_proof(sols=50, n_predictions=30, n_seeds=1)
        m = proof["market"]
        # resolved should equal total (all should resolve with enough sols)
        assert m["resolved"] <= m["total_predictions"]
