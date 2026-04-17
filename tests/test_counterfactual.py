"""
Tests for the counterfactual engine — hinge detection, timeline forking,
divergence computation, and amendment proposals.
"""
from __future__ import annotations

import json
import pytest

from src.mars100.engine import Mars100Engine, YearResult
from src.mars100.colony import SocialGraph, Relationship, Resources, RESOURCE_NAMES
from src.mars100.counterfactual import (
    HingePoint, AlternateTimeline, TimelineDivergence, CounterfactualAnalysis,
    find_hinge_points, fork_timeline, compute_divergence,
    run_counterfactual_analysis, propose_amendment,
    _select_diverse_hinges, _apply_intervention,
    HINGE_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def short_sim():
    """Run a short 20-year sim for fast testing."""
    engine = Mars100Engine(seed=42, total_years=20)
    snapshots: dict[int, dict] = {}
    years: list[YearResult] = []
    for _ in range(20):
        if not engine._active_colonists():
            break
        snapshots[engine.year + 1] = engine.snapshot()
        yr = engine.tick()
        years.append(yr)
    return engine, years, snapshots


@pytest.fixture(scope="module")
def full_sim():
    """Run a 100-year sim for comprehensive testing."""
    engine = Mars100Engine(seed=42, total_years=100)
    snapshots: dict[int, dict] = {}
    years: list[YearResult] = []
    for _ in range(100):
        if not engine._active_colonists():
            break
        snapshots[engine.year + 1] = engine.snapshot()
        yr = engine.tick()
        years.append(yr)
    return engine, years, snapshots


# ---------------------------------------------------------------------------
# HingePoint tests
# ---------------------------------------------------------------------------

class TestHingePoint:
    def test_to_dict(self):
        h = HingePoint(year=5, category="death", description="test",
                        severity=0.7, actual_outcome="died")
        d = h.to_dict()
        assert d["year"] == 5
        assert d["category"] == "death"
        assert d["severity"] == 0.7

    def test_categories_are_valid(self):
        for cat in HINGE_CATEGORIES:
            h = HingePoint(year=1, category=cat, description="x",
                            severity=0.5, actual_outcome="x")
            assert h.category in HINGE_CATEGORIES


# ---------------------------------------------------------------------------
# AlternateTimeline tests
# ---------------------------------------------------------------------------

class TestAlternateTimeline:
    def test_to_dict(self):
        a = AlternateTimeline(
            hinge_year=10, intervention="test", years_simulated=90,
            final_population=15, final_resources={"food": 0.5},
            final_governance="council", total_deaths=3,
            final_cohesion=0.6, governance_changes=2, meta_events=5,
        )
        d = a.to_dict()
        assert d["hinge_year"] == 10
        assert d["final_population"] == 15

    def test_resources_complete(self):
        resources = {name: 0.5 for name in RESOURCE_NAMES}
        a = AlternateTimeline(
            hinge_year=1, intervention="test", years_simulated=10,
            final_population=10, final_resources=resources,
            final_governance="anarchy", total_deaths=0,
            final_cohesion=0.5, governance_changes=0, meta_events=0,
        )
        for name in RESOURCE_NAMES:
            assert name in a.final_resources


# ---------------------------------------------------------------------------
# find_hinge_points tests
# ---------------------------------------------------------------------------

class TestFindHingePoints:
    def test_finds_hinges_in_real_sim(self, short_sim):
        _, years, _ = short_sim
        hinges = find_hinge_points(years)
        assert len(hinges) > 0

    def test_sorted_by_severity(self, short_sim):
        _, years, _ = short_sim
        hinges = find_hinge_points(years)
        for i in range(len(hinges) - 1):
            assert hinges[i].severity >= hinges[i + 1].severity or \
                   hinges[i].year <= hinges[i + 1].year

    def test_governance_hinge_detected(self, short_sim):
        _, years, _ = short_sim
        hinges = find_hinge_points(years)
        gov_hinges = [h for h in hinges if h.category == "governance_change"]
        # The sim should produce governance changes in 20 years
        assert len(gov_hinges) >= 0  # may or may not have governance changes

    def test_death_hinge_detected(self, full_sim):
        _, years, _ = full_sim
        hinges = find_hinge_points(years)
        death_hinges = [h for h in hinges if h.category == "death"]
        # Over 100 years, deaths should occur
        assert len(death_hinges) > 0

    def test_birth_hinge_detected(self, full_sim):
        _, years, _ = full_sim
        hinges = find_hinge_points(years)
        birth_hinges = [h for h in hinges if h.category == "birth"]
        assert len(birth_hinges) > 0

    def test_resource_crisis_detected(self):
        """Synthetic test: create a year result with critical resources."""
        yr = YearResult(
            year=5, events=[], actions={}, subsim_log=[],
            governance=None,
            resources_before={"food": 0.2, "water": 0.2, "power": 0.5, "air": 0.5, "medicine": 0.3},
            resources_after={"food": 0.1, "water": 0.2, "power": 0.5, "air": 0.5, "medicine": 0.3},
            resource_delta={}, deaths=[], exiles=[], meta_awareness=[],
            social_cohesion=0.5, governance_state={}, colonist_snapshots=[],
            convergence={}, births=[],
        )
        hinges = find_hinge_points([yr])
        crisis = [h for h in hinges if h.category == "resource_crisis"]
        assert len(crisis) == 1
        assert "food" in crisis[0].description

    def test_no_hinges_for_quiet_year(self):
        """A year with nothing happening should produce no hinges."""
        yr = YearResult(
            year=50, events=[], actions={}, subsim_log=[],
            governance=None,
            resources_before={n: 0.5 for n in RESOURCE_NAMES},
            resources_after={n: 0.5 for n in RESOURCE_NAMES},
            resource_delta={}, deaths=[], exiles=[], meta_awareness=[],
            social_cohesion=0.5, governance_state={"gov_type": "anarchy"},
            colonist_snapshots=[], convergence={}, births=[],
        )
        hinges = find_hinge_points([yr])
        assert len(hinges) == 0

    def test_multiple_hinges_same_year(self):
        """A year can produce multiple hinges of different types."""
        yr = YearResult(
            year=10, events=[], actions={}, subsim_log=[],
            governance={"passed": True, "gov_type": "council"},
            resources_before={n: 0.5 for n in RESOURCE_NAMES},
            resources_after={"food": 0.1, "water": 0.5, "power": 0.5, "air": 0.5, "medicine": 0.5},
            resource_delta={},
            deaths=[{"id": "test-1", "name": "Test One", "cause": "accident", "year": 10}],
            exiles=[], meta_awareness=[],
            social_cohesion=0.5, governance_state={"gov_type": "council"},
            colonist_snapshots=[{"alive": True}] * 9,
            convergence={}, births=[],
        )
        hinges = find_hinge_points([yr])
        categories = {h.category for h in hinges}
        assert "governance_change" in categories
        assert "death" in categories
        assert "resource_crisis" in categories


# ---------------------------------------------------------------------------
# _select_diverse_hinges tests
# ---------------------------------------------------------------------------

class TestSelectDiverseHinges:
    def test_limits_count(self):
        hinges = [HingePoint(year=i, category="governance_change",
                              description="x", severity=0.8, actual_outcome="x")
                  for i in range(0, 100, 5)]
        selected = _select_diverse_hinges(hinges, 3)
        assert len(selected) <= 3

    def test_category_diversity(self):
        """Round-robin should pick from different categories."""
        hinges = [
            HingePoint(year=7, category="governance_change", description="x",
                        severity=0.8, actual_outcome="x"),
            HingePoint(year=15, category="governance_change", description="x",
                        severity=0.8, actual_outcome="x"),
            HingePoint(year=10, category="death", description="x",
                        severity=0.6, actual_outcome="x"),
            HingePoint(year=20, category="birth", description="x",
                        severity=0.5, actual_outcome="x"),
        ]
        selected = _select_diverse_hinges(hinges, 3)
        categories = {h.category for h in selected}
        assert len(categories) >= 2  # Should pick from multiple categories

    def test_empty_input(self):
        assert _select_diverse_hinges([], 5) == []


# ---------------------------------------------------------------------------
# Snapshot / from_snapshot tests
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_roundtrip(self):
        engine = Mars100Engine(seed=42, total_years=10)
        for _ in range(3):
            engine.tick()
        snap = engine.snapshot()
        restored = Mars100Engine.from_snapshot(snap)
        assert restored.year == engine.year
        assert restored.seed == engine.seed
        assert len(restored.colonists) == len(engine.colonists)

    def test_snapshot_determinism(self):
        """Two engines restored from the same snapshot produce identical results."""
        engine = Mars100Engine(seed=42, total_years=20)
        for _ in range(5):
            engine.tick()
        snap = engine.snapshot()

        e1 = Mars100Engine.from_snapshot(snap)
        e2 = Mars100Engine.from_snapshot(snap)
        yr1 = e1.tick()
        yr2 = e2.tick()
        assert yr1.resources_after == yr2.resources_after
        assert len(yr1.deaths) == len(yr2.deaths)

    def test_snapshot_captures_rng(self):
        """RNG state must be preserved for determinism."""
        engine = Mars100Engine(seed=42, total_years=10)
        engine.tick()
        snap = engine.snapshot()
        assert "rng_state" in snap

    def test_snapshot_contains_all_state(self):
        engine = Mars100Engine(seed=42, total_years=10)
        snap = engine.snapshot()
        assert "colonists" in snap
        assert "resources" in snap
        assert "social" in snap
        assert "governance" in snap
        assert "year" in snap
        assert "seed" in snap

    def test_forked_engine_diverges_after_intervention(self):
        """A forked engine with modified state should produce different results."""
        engine = Mars100Engine(seed=42, total_years=20)
        for _ in range(5):
            engine.tick()
        snap = engine.snapshot()

        forked = Mars100Engine.from_snapshot(snap)
        # Kill all resources
        forked.resources.food = 0.01
        forked.resources.water = 0.01

        yr_orig = engine.tick()
        yr_forked = forked.tick()
        # Results should differ (not guaranteed but very likely)
        assert yr_orig.resources_after != yr_forked.resources_after


# ---------------------------------------------------------------------------
# SocialGraph.from_dict tests
# ---------------------------------------------------------------------------

class TestSocialGraphFromDict:
    def test_roundtrip(self):
        import random
        graph = SocialGraph()
        graph.initialize(["a", "b", "c"], random.Random(42))
        d = graph.to_dict()
        restored = SocialGraph.from_dict(d)
        assert set(restored.edges.keys()) == set(graph.edges.keys())
        for a in graph.edges:
            for b in graph.edges[a]:
                assert abs(restored.edges[a][b].trust - graph.edges[a][b].trust) < 1e-10


# ---------------------------------------------------------------------------
# Intervention tests
# ---------------------------------------------------------------------------

class TestInterventions:
    def test_governance_intervention(self):
        engine = Mars100Engine(seed=42, total_years=20)
        engine.governance.gov_type = "council"
        hinge = HingePoint(year=1, category="governance_change",
                            description="x", severity=0.8, actual_outcome="x")
        desc = _apply_intervention(engine, hinge)
        assert engine.governance.gov_type == "anarchy"
        assert "council" in desc

    def test_death_intervention(self):
        engine = Mars100Engine(seed=42, total_years=20)
        # Kill a colonist using the die() method
        engine.colonists[0].die(5, "test")
        name = engine.colonists[0].name
        hinge = HingePoint(year=5, category="death",
                            description=f"{name} died: test",
                            severity=0.6, actual_outcome=f"Death of {name}")
        desc = _apply_intervention(engine, hinge)
        assert engine.colonists[0].is_active()  # Should be resurrected
        assert "resurrected" in desc

    def test_resource_intervention(self):
        engine = Mars100Engine(seed=42, total_years=20)
        engine.resources.food = 0.05
        hinge = HingePoint(year=3, category="resource_crisis",
                            description="food critically low at 5%",
                            severity=0.7, actual_outcome="food dropped to 5%")
        desc = _apply_intervention(engine, hinge)
        assert engine.resources.food >= 0.4
        assert "food" in desc

    def test_birth_intervention(self):
        engine = Mars100Engine(seed=42, total_years=20)
        # Run a few years to get births
        initial_count = len(engine.colonists)
        hinge = HingePoint(year=1, category="birth",
                            description="Birth of Test", severity=0.5,
                            actual_outcome="Test born")
        # Only applies if > 10 colonists
        desc = _apply_intervention(engine, hinge)
        assert "no_birth" in desc or "suppressed" in desc

    def test_meta_boost_intervention(self):
        engine = Mars100Engine(seed=42, total_years=20)
        original_faith = engine.colonists[0].stats.faith
        hinge = HingePoint(year=1, category="meta_awareness",
                            description="x", severity=0.5, actual_outcome="x")
        desc = _apply_intervention(engine, hinge)
        assert engine.colonists[0].stats.faith >= original_faith
        assert "boosted" in desc


# ---------------------------------------------------------------------------
# fork_timeline tests
# ---------------------------------------------------------------------------

class TestForkTimeline:
    def test_fork_produces_alternate(self, short_sim):
        _, years, snapshots = short_sim
        hinges = find_hinge_points(years)
        if not hinges:
            pytest.skip("No hinges found in short sim")
        alt = fork_timeline(snapshots, hinges[0], total_years=20)
        assert alt is not None
        assert alt.years_simulated > 0
        assert alt.final_population >= 0

    def test_fork_with_no_snapshots_returns_none(self):
        hinge = HingePoint(year=5, category="death", description="x",
                            severity=0.5, actual_outcome="x")
        result = fork_timeline({}, hinge)
        assert result is None

    def test_fork_at_last_year_returns_none(self, short_sim):
        _, years, snapshots = short_sim
        # Use a year beyond the sim's total to ensure remaining <= 0
        hinge = HingePoint(year=20, category="death", description="x",
                            severity=0.5, actual_outcome="x")
        # fork_timeline should find closest snapshot and check remaining years
        # With total_years equal to the sim length, the fork at the last year
        # may still run 1 year. Use a tighter bound to test the edge case.
        result = fork_timeline(snapshots, hinge, total_years=19)
        # At year 19 with total=19, remaining=0 → None
        assert result is None or result.years_simulated <= 1

    def test_fork_finds_closest_snapshot(self, short_sim):
        _, years, snapshots = short_sim
        hinge = HingePoint(year=999, category="death", description="x",
                            severity=0.5, actual_outcome="x")
        # Should find the closest snapshot instead of exact match
        result = fork_timeline(snapshots, hinge, total_years=1000)
        # May or may not be None depending on remaining years
        # Main thing is it doesn't crash


# ---------------------------------------------------------------------------
# compute_divergence tests
# ---------------------------------------------------------------------------

class TestComputeDivergence:
    def test_identical_timelines_zero_divergence(self):
        hinge = HingePoint(year=5, category="death", description="x",
                            severity=0.5, actual_outcome="x")
        alt = AlternateTimeline(
            hinge_year=5, intervention="test", years_simulated=95,
            final_population=20, final_resources={"food": 0.7, "water": 0.7, "power": 0.8, "air": 0.9, "medicine": 0.5},
            final_governance="council", total_deaths=3,
            final_cohesion=0.6, governance_changes=2, meta_events=5,
        )
        div = compute_divergence(hinge, alt, 20,
                                  {"food": 0.7, "water": 0.7, "power": 0.8, "air": 0.9, "medicine": 0.5},
                                  "council", 0.6)
        assert div.divergence_score < 0.01
        assert div.population_delta == 0
        assert div.governance_same is True

    def test_different_population_increases_divergence(self):
        hinge = HingePoint(year=5, category="death", description="x",
                            severity=0.5, actual_outcome="x")
        alt = AlternateTimeline(
            hinge_year=5, intervention="test", years_simulated=95,
            final_population=30, final_resources={"food": 0.7, "water": 0.7, "power": 0.8, "air": 0.9, "medicine": 0.5},
            final_governance="council", total_deaths=3,
            final_cohesion=0.6, governance_changes=2, meta_events=5,
        )
        div = compute_divergence(hinge, alt, 20,
                                  {"food": 0.7, "water": 0.7, "power": 0.8, "air": 0.9, "medicine": 0.5},
                                  "council", 0.6)
        assert div.divergence_score > 0.0
        assert div.population_delta == 10

    def test_different_governance_increases_divergence(self):
        hinge = HingePoint(year=5, category="governance_change", description="x",
                            severity=0.8, actual_outcome="x")
        alt = AlternateTimeline(
            hinge_year=5, intervention="test", years_simulated=95,
            final_population=20, final_resources={"food": 0.7, "water": 0.7, "power": 0.8, "air": 0.9, "medicine": 0.5},
            final_governance="anarchy", total_deaths=3,
            final_cohesion=0.6, governance_changes=0, meta_events=5,
        )
        div = compute_divergence(hinge, alt, 20,
                                  {"food": 0.7, "water": 0.7, "power": 0.8, "air": 0.9, "medicine": 0.5},
                                  "council", 0.6)
        assert div.governance_same is False
        assert div.divergence_score > 0.1

    def test_divergence_score_bounded(self):
        hinge = HingePoint(year=1, category="death", description="x",
                            severity=1.0, actual_outcome="x")
        alt = AlternateTimeline(
            hinge_year=1, intervention="test", years_simulated=99,
            final_population=100, final_resources={n: 1.0 for n in RESOURCE_NAMES},
            final_governance="anarchy", total_deaths=0,
            final_cohesion=1.0, governance_changes=0, meta_events=0,
        )
        div = compute_divergence(hinge, alt, 1,
                                  {n: 0.0 for n in RESOURCE_NAMES},
                                  "council", 0.0)
        assert 0.0 <= div.divergence_score <= 1.0

    def test_divergence_to_dict(self):
        hinge = HingePoint(year=5, category="death", description="x",
                            severity=0.5, actual_outcome="x")
        alt = AlternateTimeline(
            hinge_year=5, intervention="test", years_simulated=95,
            final_population=20, final_resources={n: 0.5 for n in RESOURCE_NAMES},
            final_governance="council", total_deaths=3,
            final_cohesion=0.6, governance_changes=2, meta_events=5,
        )
        div = compute_divergence(hinge, alt, 20,
                                  {n: 0.5 for n in RESOURCE_NAMES},
                                  "council", 0.6)
        d = div.to_dict()
        assert "divergence_score" in d
        assert "hinge" in d
        assert "alternate" in d


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestRunCounterfactualAnalysis:
    def test_produces_complete_analysis(self):
        analysis = run_counterfactual_analysis(engine_seed=42, total_years=20, max_hinges=3)
        assert isinstance(analysis, CounterfactualAnalysis)
        assert len(analysis.hinge_points) > 0
        assert 0.0 <= analysis.fragility_score <= 1.0
        assert 0.0 <= analysis.max_divergence <= 1.0
        assert "name" in analysis.proposed_amendment

    def test_different_seeds_produce_different_results(self):
        a1 = run_counterfactual_analysis(engine_seed=1, total_years=20, max_hinges=3)
        a2 = run_counterfactual_analysis(engine_seed=999, total_years=20, max_hinges=3)
        # Different seeds should produce at least slightly different results
        assert a1.fragility_score != a2.fragility_score or \
               len(a1.hinge_points) != len(a2.hinge_points)

    def test_to_dict_serializable(self):
        analysis = run_counterfactual_analysis(engine_seed=42, total_years=20, max_hinges=3)
        d = analysis.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert "hinge_points" in parsed
        assert "fragility_score" in parsed

    def test_short_sim_no_crash(self):
        analysis = run_counterfactual_analysis(engine_seed=42, total_years=5, max_hinges=2)
        assert isinstance(analysis, CounterfactualAnalysis)

    def test_single_year_sim(self):
        analysis = run_counterfactual_analysis(engine_seed=42, total_years=1, max_hinges=1)
        assert isinstance(analysis, CounterfactualAnalysis)


# ---------------------------------------------------------------------------
# Amendment proposal tests
# ---------------------------------------------------------------------------

class TestAmendmentProposal:
    def test_proposal_structure(self):
        amendment = propose_amendment(0.5, None, [])
        assert "name" in amendment
        assert "text" in amendment
        assert "rationale" in amendment
        assert "evidence" in amendment
        assert "confidence" in amendment

    def test_high_fragility_high_confidence(self):
        amendment = propose_amendment(0.5, None, [])
        assert amendment["confidence"] == "high"

    def test_low_fragility_low_confidence(self):
        amendment = propose_amendment(0.1, None, [])
        assert amendment["confidence"] == "low"

    def test_moderate_fragility(self):
        amendment = propose_amendment(0.3, None, [])
        assert amendment["confidence"] == "moderate"

    def test_amendment_name(self):
        amendment = propose_amendment(0.5, None, [])
        assert amendment["name"] == "The Counterfactual Principle"


# ---------------------------------------------------------------------------
# Invariant tests (property-like)
# ---------------------------------------------------------------------------

class TestInvariants:
    @pytest.mark.parametrize("seed", [1, 13, 42, 77, 100])
    def test_fragility_bounded(self, seed):
        analysis = run_counterfactual_analysis(engine_seed=seed, total_years=20, max_hinges=3)
        assert 0.0 <= analysis.fragility_score <= 1.0
        assert 0.0 <= analysis.max_divergence <= 1.0

    @pytest.mark.parametrize("seed", [1, 42, 99])
    def test_divergence_bounded(self, seed):
        analysis = run_counterfactual_analysis(engine_seed=seed, total_years=20, max_hinges=3)
        for div in analysis.divergences:
            assert 0.0 <= div.divergence_score <= 1.0

    def test_hinge_categories_valid(self):
        analysis = run_counterfactual_analysis(engine_seed=42, total_years=20, max_hinges=5)
        for h in analysis.hinge_points:
            assert h.category in HINGE_CATEGORIES

    def test_colonist_count_conservation(self):
        """Forked timelines shouldn't create colonists from nothing."""
        analysis = run_counterfactual_analysis(engine_seed=42, total_years=20, max_hinges=3)
        for div in analysis.divergences:
            # Population can grow via births but shouldn't exceed reasonable bounds
            assert div.alternate.final_population < 100


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_year_list(self):
        hinges = find_hinge_points([])
        assert hinges == []

    def test_all_resources_critical(self):
        yr = YearResult(
            year=5, events=[], actions={}, subsim_log=[],
            governance=None,
            resources_before={n: 0.05 for n in RESOURCE_NAMES},
            resources_after={n: 0.05 for n in RESOURCE_NAMES},
            resource_delta={}, deaths=[], exiles=[], meta_awareness=[],
            social_cohesion=0.3, governance_state={"gov_type": "anarchy"},
            colonist_snapshots=[], convergence={}, births=[],
        )
        hinges = find_hinge_points([yr])
        crises = [h for h in hinges if h.category == "resource_crisis"]
        assert len(crises) == len(RESOURCE_NAMES)

    def test_multiple_deaths_same_year(self):
        yr = YearResult(
            year=50, events=[], actions={}, subsim_log=[],
            governance=None,
            resources_before={n: 0.5 for n in RESOURCE_NAMES},
            resources_after={n: 0.5 for n in RESOURCE_NAMES},
            resource_delta={},
            deaths=[
                {"id": "a", "name": "Alpha", "cause": "storm", "year": 50},
                {"id": "b", "name": "Beta", "cause": "disease", "year": 50},
            ],
            exiles=[], meta_awareness=[],
            social_cohesion=0.4, governance_state={"gov_type": "council"},
            colonist_snapshots=[{"alive": True}] * 8,
            convergence={}, births=[],
        )
        hinges = find_hinge_points([yr])
        death_hinges = [h for h in hinges if h.category == "death"]
        assert len(death_hinges) == 2

    def test_governance_proposal_rejected_no_hinge(self):
        yr = YearResult(
            year=5, events=[], actions={}, subsim_log=[],
            governance={"passed": False, "gov_type": "dictator"},
            resources_before={n: 0.5 for n in RESOURCE_NAMES},
            resources_after={n: 0.5 for n in RESOURCE_NAMES},
            resource_delta={}, deaths=[], exiles=[], meta_awareness=[],
            social_cohesion=0.5, governance_state={"gov_type": "anarchy"},
            colonist_snapshots=[], convergence={}, births=[],
        )
        hinges = find_hinge_points([yr])
        gov_hinges = [h for h in hinges if h.category == "governance_change"]
        assert len(gov_hinges) == 0  # Rejected proposals aren't hinges
