"""
Tests for market_maker.py — Brier-scored prediction market for Mars terrarium.

Run: python -m pytest tests/test_market_maker.py -v
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.market_maker import (
    AGENT_ARCHETYPES,
    COLONY_NAMES,
    RESOLVERS,
    TEMPLATES,
    TECH_NAMES,
    Prediction,
    brier_score,
    log_score,
    payout_from_brier,
    generate_predictions,
    resolve_predictions,
    score_predictions,
    build_calibration_curve,
    build_leaderboard,
    assemble_report,
    run_terrarium,
    run_terrarium_ensemble,
    run_market,
    resolve_survival,
    resolve_population_peak,
    resolve_population_final,
    resolve_tech_unlock,
    resolve_epidemic_any,
    resolve_growth_rate,
    resolve_global_storm,
    resolve_morale_floor,
    resolve_total_deaths,
    resolve_total_migrations,
    resolve_highest_final_pop,
    resolve_terraforming_pct,
)
from src.tick_engine import Simulation


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------
class TestBrierScore:

    def test_perfect_true(self) -> None:
        assert brier_score(1.0, 1.0) == pytest.approx(0.0)

    def test_perfect_false(self) -> None:
        assert brier_score(0.0, 0.0) == pytest.approx(0.0)

    def test_worst_true(self) -> None:
        assert brier_score(0.0, 1.0) == pytest.approx(1.0)

    def test_worst_false(self) -> None:
        assert brier_score(1.0, 0.0) == pytest.approx(1.0)

    def test_fifty_fifty(self) -> None:
        assert brier_score(0.5, 1.0) == pytest.approx(0.25)
        assert brier_score(0.5, 0.0) == pytest.approx(0.25)

    def test_bounded_zero_one(self) -> None:
        rng = random.Random(42)
        for _ in range(100):
            pred = rng.random()
            outcome = float(rng.choice([0, 1]))
            bs = brier_score(pred, outcome)
            assert 0.0 <= bs <= 1.0

    def test_proper_scoring(self) -> None:
        """Honest forecast minimizes expected Brier score."""
        true_p = 0.7
        honest = true_p * brier_score(0.7, 1.0) + (1 - true_p) * brier_score(0.7, 0.0)
        biased = true_p * brier_score(0.9, 1.0) + (1 - true_p) * brier_score(0.9, 0.0)
        assert honest < biased


# ---------------------------------------------------------------------------
# Log score
# ---------------------------------------------------------------------------
class TestLogScore:

    def test_perfect_high(self) -> None:
        assert log_score(0.99, 1.0) > log_score(0.5, 1.0)

    def test_wrong_is_very_negative(self) -> None:
        assert log_score(0.01, 1.0) < -4.0

    def test_always_negative_or_zero(self) -> None:
        rng = random.Random(42)
        for _ in range(100):
            pred = rng.random()
            outcome = float(rng.choice([0, 1]))
            ls = log_score(pred, outcome)
            assert ls <= 0.001  # numerical tolerance


# ---------------------------------------------------------------------------
# Payout
# ---------------------------------------------------------------------------
class TestPayout:

    def test_perfect_prediction_doubles(self) -> None:
        assert payout_from_brier(0.05, 10.0) == pytest.approx(20.0)

    def test_good_prediction_1_5x(self) -> None:
        assert payout_from_brier(0.15, 10.0) == pytest.approx(15.0)

    def test_mediocre_prediction_1x(self) -> None:
        assert payout_from_brier(0.4, 10.0) == pytest.approx(10.0)

    def test_bad_prediction_half(self) -> None:
        assert payout_from_brier(0.6, 10.0) == pytest.approx(5.0)

    def test_terrible_prediction_zero(self) -> None:
        assert payout_from_brier(0.8, 10.0) == pytest.approx(0.0)

    def test_payout_nonnegative(self) -> None:
        for brier in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            assert payout_from_brier(brier, 10.0) >= 0.0


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------
class TestPredictionGeneration:

    def test_generates_correct_count(self) -> None:
        preds = generate_predictions(50, seed=42)
        assert len(preds) == 50

    def test_unique_ids(self) -> None:
        preds = generate_predictions(100, seed=42)
        ids = [p.id for p in preds]
        assert len(ids) == len(set(ids))

    def test_deterministic(self) -> None:
        p1 = generate_predictions(20, seed=99)
        p2 = generate_predictions(20, seed=99)
        for a, b in zip(p1, p2):
            assert a.id == b.id
            assert a.confidence == b.confidence

    def test_confidence_bounded(self) -> None:
        preds = generate_predictions(200, seed=42)
        for p in preds:
            assert 0.01 <= p.confidence <= 0.99

    def test_all_archetypes_present(self) -> None:
        preds = generate_predictions(500, seed=42)
        archetypes = set(p.archetype for p in preds)
        for arch in AGENT_ARCHETYPES:
            assert arch in archetypes

    def test_all_categories_present(self) -> None:
        preds = generate_predictions(500, seed=42)
        categories = set(p.category for p in preds)
        assert len(categories) >= 8


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------
class TestResolvers:

    @pytest.fixture
    def sim_result(self) -> dict:
        sim = Simulation(sols=100, env_seed=42)
        return sim.run()

    def test_survival_true(self, sim_result: dict) -> None:
        for name in COLONY_NAMES:
            result = resolve_survival({"colony": name}, sim_result)
            assert result is True

    def test_population_final(self, sim_result: dict) -> None:
        result = resolve_population_final(
            {"colony": "Ares Prime", "threshold": 50}, sim_result
        )
        assert result is True  # 120 start, should be > 50

    def test_tech_unlock(self, sim_result: dict) -> None:
        result = resolve_tech_unlock(
            {"tech": "Advanced Solar Cells"}, sim_result
        )
        assert result is True  # usually unlocked by sol 100

    def test_epidemic_any(self, sim_result: dict) -> None:
        result = resolve_epidemic_any({"colony": ""}, sim_result)
        assert isinstance(result, bool)

    def test_global_storm(self, sim_result: dict) -> None:
        result = resolve_global_storm({}, sim_result)
        assert isinstance(result, bool)

    def test_total_deaths(self, sim_result: dict) -> None:
        result = resolve_total_deaths({"threshold": 0}, sim_result)
        assert isinstance(result, bool)

    def test_total_migrations(self, sim_result: dict) -> None:
        result = resolve_total_migrations({"threshold": 0}, sim_result)
        assert isinstance(result, bool)

    def test_highest_final_pop(self, sim_result: dict) -> None:
        result = resolve_highest_final_pop(
            {"colony": "Ares Prime"}, sim_result
        )
        assert isinstance(result, bool)

    def test_morale_floor(self, sim_result: dict) -> None:
        result = resolve_morale_floor(
            {"colony": "Ares Prime", "threshold": 0.3}, sim_result
        )
        assert isinstance(result, bool)

    def test_resolver_registry_complete(self) -> None:
        for template in TEMPLATES:
            assert template["category"] in RESOLVERS


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
class TestFullPipeline:

    def test_resolve_predictions(self) -> None:
        preds = generate_predictions(30, seed=42)
        results = run_terrarium_ensemble(sols=100, seeds=[42, 43])
        resolved = resolve_predictions(preds, results)
        resolved_count = sum(1 for p in resolved if p.outcome is not None)
        assert resolved_count > 0

    def test_score_predictions(self) -> None:
        preds = generate_predictions(30, seed=42)
        results = run_terrarium_ensemble(sols=100, seeds=[42])
        resolve_predictions(preds, results)
        scored = score_predictions(preds)
        scored_count = sum(1 for p in scored if p.brier is not None)
        assert scored_count > 0
        for p in scored:
            if p.brier is not None:
                assert 0.0 <= p.brier <= 1.0

    def test_calibration_curve_5_buckets(self) -> None:
        preds = generate_predictions(50, seed=42)
        results = run_terrarium_ensemble(sols=100, seeds=[42])
        resolve_predictions(preds, results)
        score_predictions(preds)
        curve = build_calibration_curve(preds)
        assert len(curve) == 5
        for bucket in curve:
            assert "bucket_lo" in bucket
            assert "actual_rate" in bucket
            assert "count" in bucket

    def test_leaderboard_sorted(self) -> None:
        preds = generate_predictions(100, seed=42)
        results = run_terrarium_ensemble(sols=100, seeds=[42])
        resolve_predictions(preds, results)
        score_predictions(preds)
        lb = build_leaderboard(preds)
        if len(lb) > 1:
            briers = [r["mean_brier"] for r in lb]
            assert briers == sorted(briers)

    def test_run_market_complete(self) -> None:
        report = run_market(n_predictions=30, sols=100, seeds=[42])
        assert "total_predictions" in report
        assert "resolved" in report
        assert "accuracy" in report
        assert "mean_brier" in report
        assert "calibration" in report
        assert "leaderboard" in report
        assert "categories" in report

    def test_run_market_deterministic(self) -> None:
        r1 = run_market(n_predictions=30, sols=100, seeds=[42], market_seed=0)
        r2 = run_market(n_predictions=30, sols=100, seeds=[42], market_seed=0)
        assert r1["resolved"] == r2["resolved"]
        assert r1["mean_brier"] == r2["mean_brier"]


# ---------------------------------------------------------------------------
# Conservation laws / invariants
# ---------------------------------------------------------------------------
class TestConservation:

    def test_brier_bounded_in_pipeline(self) -> None:
        report = run_market(n_predictions=50, sols=100, seeds=[42])
        assert 0.0 <= report["mean_brier"] <= 1.0

    def test_accuracy_bounded(self) -> None:
        report = run_market(n_predictions=50, sols=100, seeds=[42])
        assert 0.0 <= report["accuracy"] <= 1.0

    def test_resolved_plus_unresolved_equals_total(self) -> None:
        report = run_market(n_predictions=50, sols=100, seeds=[42])
        assert report["resolved"] + report["unresolved"] == report["total_predictions"]

    def test_payout_nonnegative_in_pipeline(self) -> None:
        preds = generate_predictions(50, seed=42)
        results = run_terrarium_ensemble(sols=100, seeds=[42])
        resolve_predictions(preds, results)
        score_predictions(preds)
        for p in preds:
            if p.payout is not None:
                assert p.payout >= 0.0

    def test_outcomes_are_boolean_or_none(self) -> None:
        preds = generate_predictions(50, seed=42)
        results = run_terrarium_ensemble(sols=100, seeds=[42])
        resolve_predictions(preds, results)
        for p in preds:
            assert p.outcome is None or isinstance(p.outcome, bool)
