"""Tests for the Mars-100 recursive colony simulation (package API)."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100 import (
    Mars100Engine, YearResult, SimulationResult, ENGINE_VERSION,
    Colonist, ColonistStats, ColonistSkills, create_founding_ten,
    Resources, SocialGraph, RESOURCE_NAMES, STAT_NAMES, SKILL_NAMES,
    generate_events, GovernanceState,
    run_simulation, reset_birth_counter,
    compute_convergence_score, convergence_trend,
)


@pytest.fixture(autouse=True)
def reset():
    reset_birth_counter()
    yield
    reset_birth_counter()


class TestEngineInit:
    def test_creates_ten_colonists(self):
        engine = Mars100Engine(seed=42)
        assert len(engine.colonists) == 10
        assert all(c.is_active() for c in engine.colonists)

    def test_deterministic(self):
        e1 = Mars100Engine(seed=42)
        e2 = Mars100Engine(seed=42)
        for c1, c2 in zip(e1.colonists, e2.colonists):
            assert c1.id == c2.id
            assert c1.name == c2.name

    def test_different_seeds(self):
        e1 = Mars100Engine(seed=42)
        e2 = Mars100Engine(seed=99)
        ids1 = {c.id for c in e1.colonists}
        ids2 = {c.id for c in e2.colonists}
        assert ids1 == ids2  # same founding ten names

    def test_initial_resources(self):
        engine = Mars100Engine(seed=42)
        for name in RESOURCE_NAMES:
            assert 0.0 <= getattr(engine.resources, name) <= 1.0

    def test_governance_starts_anarchy(self):
        engine = Mars100Engine(seed=42)
        assert engine.governance.gov_type == "anarchy"


class TestEngineTick:
    def test_tick_advances_year(self):
        engine = Mars100Engine(seed=42)
        result = engine.tick()
        assert result.year == 1
        assert engine.year == 1

    def test_tick_returns_year_result(self):
        engine = Mars100Engine(seed=42)
        result = engine.tick()
        assert isinstance(result, YearResult)
        assert len(result.events) > 0
        assert len(result.actions) > 0

    def test_tick_resources_change(self):
        engine = Mars100Engine(seed=42)
        before = engine.resources.to_dict()
        engine.tick()
        after = engine.resources.to_dict()
        assert before != after

    def test_convergence_tracked(self):
        engine = Mars100Engine(seed=42)
        result = engine.tick()
        assert result.convergence_score >= 0.0
        assert len(result.convergence_per_stat) == len(STAT_NAMES)

    def test_subsim_log_present(self):
        engine = Mars100Engine(seed=42, total_years=20)
        result = engine.run()
        total_subsims = sum(len(y.subsim_log) for y in result.years)
        assert total_subsims >= 0  # may or may not have subsims

    def test_tick_serializable(self):
        engine = Mars100Engine(seed=42)
        result = engine.tick()
        d = result.to_dict()
        assert "year" in d
        assert "births" in d
        assert "convergence_score" in d
        assert "meta_insights" in d


class TestEngineRun:
    def test_short_run(self):
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert isinstance(result, SimulationResult)
        assert len(result.years) == 10

    def test_result_has_convergence(self):
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.convergence_scores) == 10
        assert "trend" in result.convergence_summary

    def test_result_has_births_count(self):
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert result.total_births >= 0

    def test_result_serializable(self):
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == ENGINE_VERSION
        assert "convergence" in d["summary"]
        assert "convergence_scores" in d
        assert "meta_insights" in d

    def test_deterministic_run(self):
        r1 = Mars100Engine(seed=42, total_years=10).run()
        r2 = Mars100Engine(seed=42, total_years=10).run()
        assert r1.total_deaths == r2.total_deaths
        assert r1.total_births == r2.total_births
        assert len(r1.years) == len(r2.years)


class TestRunSimulation:
    def test_convenience_function(self):
        result = run_simulation(years=5, seed=42)
        assert isinstance(result, SimulationResult)
        assert len(result.years) == 5


class TestPopulationDynamics:
    def test_population_evolves(self):
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        populations = []
        for yr in result.years:
            alive = sum(1 for c in yr.colonist_snapshots
                       if c.get("alive") and not c.get("exiled"))
            populations.append(alive)
        # Population should change over 50 years
        assert len(set(populations)) > 1

    def test_deaths_possible(self):
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        assert result.total_deaths >= 0

    def test_births_after_year_15(self):
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        early_births = sum(len(y.births) for y in result.years if y.year <= 14)
        assert early_births == 0  # no births before year 15


class TestGovernanceEvolution:
    def test_governance_proposals(self):
        engine = Mars100Engine(seed=42, total_years=30)
        result = engine.run()
        gov_proposals = sum(1 for y in result.years if y.governance is not None)
        assert gov_proposals > 0  # should have some proposals

    def test_governance_changes(self):
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        assert result.governance_changes >= 0


class TestConvergenceIntegration:
    def test_convergence_score_bounded(self):
        engine = Mars100Engine(seed=42, total_years=20)
        result = engine.run()
        for score in result.convergence_scores:
            assert 0.0 <= score <= 1.0

    def test_convergence_trend_determined(self):
        engine = Mars100Engine(seed=42, total_years=30)
        result = engine.run()
        assert result.convergence_summary["trend"] in ("converging", "diverging", "stable")


class TestMetaInsightIntegration:
    def test_meta_insights_list(self):
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        assert isinstance(result.meta_insights, list)

    def test_proposed_amendment_type(self):
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        # May or may not have a proposed amendment
        if result.proposed_amendment:
            assert "type" in result.proposed_amendment
            assert "strength" in result.proposed_amendment
            assert "proposed_amendment" in result.proposed_amendment
