"""Tests for Mars-100 post-simulation analysis module."""
from __future__ import annotations

import pytest
from src.mars100.engine import Mars100Engine
from src.mars100.analysis import (
    analyze_governance_stability,
    analyze_meta_emergence,
    analyze_subsim_effectiveness,
    analyze_value_convergence,
    compute_fitness_score,
    extract_amendment_proposal,
    full_analysis,
)


@pytest.fixture
def short_sim() -> dict:
    """Run a 15-year sim (fast, deterministic)."""
    engine = Mars100Engine(seed=42, total_years=15)
    return engine.run().to_dict()


@pytest.fixture
def medium_sim() -> dict:
    """Run a 50-year sim for richer governance data."""
    engine = Mars100Engine(seed=42, total_years=50)
    return engine.run().to_dict()


@pytest.fixture
def full_sim() -> dict:
    """Run the full 100-year sim."""
    engine = Mars100Engine(seed=42, total_years=100)
    return engine.run().to_dict()


@pytest.fixture
def empty_sim() -> dict:
    """Minimal dict simulating an empty run."""
    return {"years": [], "final_colonists": [], "final_resources": {},
            "final_governance": {"gov_type": "anarchy"},
            "summary": {}, "_meta": {"total_years": 0, "engine": "mars-100"}}


# ---- Value Convergence ----

class TestValueConvergence:
    def test_returns_all_stat_names(self, short_sim: dict) -> None:
        result = analyze_value_convergence(short_sim)
        expected_stats = {"resolve", "improvisation", "empathy",
                         "hoarding", "faith", "paranoia"}
        assert set(result["convergence_scores"].keys()) == expected_stats
        assert set(result["stat_trajectories"].keys()) == expected_stats

    def test_convergence_scores_bounded(self, short_sim: dict) -> None:
        result = analyze_value_convergence(short_sim)
        for stat, score in result["convergence_scores"].items():
            assert -10.0 < score < 10.0, f"{stat} score out of bounds: {score}"

    def test_verdict_is_valid(self, short_sim: dict) -> None:
        result = analyze_value_convergence(short_sim)
        assert result["verdict"] in ("converging", "diverging", "stable")

    def test_trajectories_match_years(self, short_sim: dict) -> None:
        result = analyze_value_convergence(short_sim)
        n_labels = len(result["year_labels"])
        for stat, traj in result["stat_trajectories"].items():
            assert len(traj) == n_labels, f"{stat}: {len(traj)} != {n_labels}"

    def test_stdev_non_negative(self, short_sim: dict) -> None:
        result = analyze_value_convergence(short_sim)
        for stat, traj in result["stat_trajectories"].items():
            for val in traj:
                assert val >= 0.0, f"{stat} stdev negative: {val}"

    def test_empty_sim(self, empty_sim: dict) -> None:
        result = analyze_value_convergence(empty_sim)
        assert result["verdict"] in ("converging", "diverging", "stable")
        assert result["overall_convergence"] == 0.0

    def test_deterministic(self) -> None:
        a = Mars100Engine(seed=77, total_years=10).run().to_dict()
        b = Mars100Engine(seed=77, total_years=10).run().to_dict()
        ra = analyze_value_convergence(a)
        rb = analyze_value_convergence(b)
        assert ra["convergence_scores"] == rb["convergence_scores"]


# ---- Governance Stability ----

class TestGovernanceStability:
    def test_periods_cover_all_years(self, medium_sim: dict) -> None:
        result = analyze_governance_stability(medium_sim)
        total_duration = sum(p["duration"] for p in result["periods"])
        n_years = len(medium_sim["years"])
        assert total_duration == n_years

    def test_transitions_match_periods(self, medium_sim: dict) -> None:
        result = analyze_governance_stability(medium_sim)
        assert result["transitions"] == len(result["periods"]) - 1

    def test_attractor_is_valid_type(self, medium_sim: dict) -> None:
        result = analyze_governance_stability(medium_sim)
        valid = {"anarchy", "council", "dictator", "lottery", "consensus",
                 "ai_governor", "none"}
        assert result["attractor"] in valid

    def test_longest_period_exists(self, medium_sim: dict) -> None:
        result = analyze_governance_stability(medium_sim)
        if result["longest_period"]:
            assert result["longest_period"]["duration"] > 0
            assert result["longest_period"]["type"] in result["type_durations"]

    def test_type_durations_sum(self, medium_sim: dict) -> None:
        result = analyze_governance_stability(medium_sim)
        total = sum(result["type_durations"].values())
        n_years = len(medium_sim["years"])
        assert total == n_years

    def test_empty_sim(self, empty_sim: dict) -> None:
        result = analyze_governance_stability(empty_sim)
        assert result["periods"] == []
        assert result["transitions"] == 0

    def test_single_year(self) -> None:
        sim = Mars100Engine(seed=42, total_years=1).run().to_dict()
        result = analyze_governance_stability(sim)
        assert len(result["periods"]) == 1
        assert result["transitions"] == 0

    def test_period_years_ascending(self, medium_sim: dict) -> None:
        result = analyze_governance_stability(medium_sim)
        for i in range(1, len(result["periods"])):
            assert result["periods"][i]["start_year"] > result["periods"][i - 1]["start_year"]


# ---- Subsim Effectiveness ----

class TestSubsimEffectiveness:
    def test_rates_bounded(self, medium_sim: dict) -> None:
        result = analyze_subsim_effectiveness(medium_sim)
        assert 0.0 <= result["backed_pass_rate"] <= 1.0
        assert 0.0 <= result["unbacked_pass_rate"] <= 1.0

    def test_total_subsims_non_negative(self, medium_sim: dict) -> None:
        result = analyze_subsim_effectiveness(medium_sim)
        assert result["total_subsims"] >= 0

    def test_depth_counts_positive(self, medium_sim: dict) -> None:
        result = analyze_subsim_effectiveness(medium_sim)
        for depth, count in result["depth_counts"].items():
            assert depth >= 1
            assert count > 0

    def test_proposals_add_up(self, medium_sim: dict) -> None:
        result = analyze_subsim_effectiveness(medium_sim)
        # Sanity: backed + unbacked should equal total proposals
        total_proposals = result["backed_proposals"] + result["unbacked_proposals"]
        actual_proposals = sum(1 for yr in medium_sim["years"] if yr.get("governance") is not None)
        assert total_proposals == actual_proposals

    def test_effectiveness_delta_bounded(self, medium_sim: dict) -> None:
        result = analyze_subsim_effectiveness(medium_sim)
        assert -1.0 <= result["effectiveness_delta"] <= 1.0

    def test_empty_sim(self, empty_sim: dict) -> None:
        result = analyze_subsim_effectiveness(empty_sim)
        assert result["total_subsims"] == 0
        assert result["backed_pass_rate"] == 0.0
        assert result["max_depth_reached"] == 0

    def test_max_depth_within_limits(self, full_sim: dict) -> None:
        result = analyze_subsim_effectiveness(full_sim)
        assert result["max_depth_reached"] <= 3


# ---- Meta Emergence ----

class TestMetaEmergence:
    def test_cumulative_non_decreasing(self, full_sim: dict) -> None:
        result = analyze_meta_emergence(full_sim)
        for i in range(1, len(result["curve"])):
            assert result["curve"][i]["cumulative"] >= result["curve"][i - 1]["cumulative"]

    def test_total_events_matches_curve(self, full_sim: dict) -> None:
        result = analyze_meta_emergence(full_sim)
        if result["curve"]:
            assert result["total_events"] == result["curve"][-1]["cumulative"]

    def test_first_year_before_total(self, full_sim: dict) -> None:
        result = analyze_meta_emergence(full_sim)
        if result["first_year"] is not None:
            assert result["first_year"] >= 1
            assert result["total_events"] > 0

    def test_unique_colonists_bounded(self, full_sim: dict) -> None:
        result = analyze_meta_emergence(full_sim)
        # 10 founding + births; bound by total population that ever lived
        total_ever = 10 + full_sim["summary"].get("total_births", 0)
        assert result["unique_colonists_aware"] <= total_ever

    def test_empty_sim(self, empty_sim: dict) -> None:
        result = analyze_meta_emergence(empty_sim)
        assert result["total_events"] == 0
        assert result["first_year"] is None
        assert result["curve"] == []


# ---- Amendment Proposal ----

class TestAmendmentProposal:
    def test_full_sim_proposes_something(self, full_sim: dict) -> None:
        result = extract_amendment_proposal(full_sim)
        # With 100 years, something should emerge
        assert isinstance(result, dict)
        assert "proposed" in result

    def test_amendment_has_required_fields(self, full_sim: dict) -> None:
        result = extract_amendment_proposal(full_sim)
        if result["proposed"]:
            amend = result["amendment"]
            assert "id" in amend
            assert "title" in amend
            assert "text" in amend
            assert "rationale" in amend
            assert "strength" in amend
            assert 0.0 <= amend["strength"] <= 1.0

    def test_all_candidates_have_strength(self, full_sim: dict) -> None:
        result = extract_amendment_proposal(full_sim)
        if result["proposed"]:
            for c in result["all_candidates"]:
                assert 0.0 <= c["strength"] <= 1.0

    def test_strongest_is_best(self, full_sim: dict) -> None:
        result = extract_amendment_proposal(full_sim)
        if result["proposed"] and len(result["all_candidates"]) > 1:
            best_strength = result["amendment"]["strength"]
            for c in result["all_candidates"]:
                assert c["strength"] <= best_strength + 1e-9

    def test_empty_sim_no_proposal(self, empty_sim: dict) -> None:
        result = extract_amendment_proposal(empty_sim)
        assert result["proposed"] is False

    def test_deterministic(self) -> None:
        a = Mars100Engine(seed=42, total_years=50).run().to_dict()
        b = Mars100Engine(seed=42, total_years=50).run().to_dict()
        ra = extract_amendment_proposal(a)
        rb = extract_amendment_proposal(b)
        assert ra["proposed"] == rb["proposed"]
        if ra["proposed"]:
            assert ra["amendment"]["id"] == rb["amendment"]["id"]


# ---- Fitness Score ----

class TestFitnessScore:
    def test_components_bounded(self, short_sim: dict) -> None:
        result = compute_fitness_score(short_sim)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_composite_is_weighted_average(self, short_sim: dict) -> None:
        result = compute_fitness_score(short_sim)
        weights = {
            "survival_rate": 0.25, "resource_health": 0.20,
            "social_cohesion": 0.20, "governance_stability": 0.20,
            "cultural_richness": 0.15,
        }
        expected = sum(result[k] * w for k, w in weights.items())
        assert abs(result["composite"] - expected) < 1e-9

    def test_empty_sim(self, empty_sim: dict) -> None:
        result = compute_fitness_score(empty_sim)
        assert "composite" in result
        assert 0.0 <= result["composite"] <= 1.0


# ---- Full Analysis ----

class TestFullAnalysis:
    def test_has_all_sections(self, short_sim: dict) -> None:
        result = full_analysis(short_sim)
        expected_keys = {"value_convergence", "governance_stability",
                        "subsim_effectiveness", "meta_emergence",
                        "amendment_proposal", "fitness"}
        assert set(result.keys()) == expected_keys

    def test_is_serializable(self, short_sim: dict) -> None:
        import json
        result = full_analysis(short_sim)
        # Should not raise
        json.dumps(result)

    def test_deterministic(self) -> None:
        a = Mars100Engine(seed=42, total_years=10).run().to_dict()
        b = Mars100Engine(seed=42, total_years=10).run().to_dict()
        ra = full_analysis(a)
        rb = full_analysis(b)
        assert ra["fitness"] == rb["fitness"]
        assert ra["value_convergence"]["convergence_scores"] == rb["value_convergence"]["convergence_scores"]
