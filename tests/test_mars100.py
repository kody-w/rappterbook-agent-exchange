"""Tests for the Mars-100 recursive colony simulation package.

Tests the class-based API exposed via src.mars100.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100 import (
    COLONIST_NAMES, ELEMENTS, STAT_NAMES, SKILL_NAMES, RESOURCE_NAMES,
    Colonist, ColonistStats, ColonistSkills, MemoryEntry,
    Resources, SocialGraph, Relationship,
    Mars100Engine, YearResult, SimulationResult,
    Event, generate_events, EVENT_TEMPLATES,
    GovernanceProposal, GovernanceState, resolve_vote,
    SubSimBudget, SubSimResult, spawn_subsim,
    LispyError, lispy_run, make_env,
    create_founding_ten, create_child,
    compute_value_convergence, tick_resources,
)


# ──────────────────────────────────────────────────────────────
# Colonist tests
# ──────────────────────────────────────────────────────────────

class TestColonist:
    """Tests for colonist creation and mutation."""

    def test_founding_ten_count(self):
        colonists = create_founding_ten(seed=42)
        assert len(colonists) == 10

    def test_founding_ten_unique_ids(self):
        colonists = create_founding_ten()
        ids = [c.id for c in colonists]
        assert len(set(ids)) == 10

    def test_founding_ten_all_active(self):
        for c in create_founding_ten():
            assert c.is_active()

    def test_colonist_elements_valid(self):
        for c in create_founding_ten():
            assert c.element in ELEMENTS

    def test_colonist_stats_bounded(self):
        for c in create_founding_ten():
            for name in STAT_NAMES:
                val = getattr(c.stats, name)
                assert 0.0 <= val <= 1.0, f"{c.id}.{name} = {val}"

    def test_colonist_deterministic(self):
        a = create_founding_ten(seed=42)
        b = create_founding_ten(seed=42)
        for ca, cb in zip(a, b):
            assert ca.name == cb.name
            assert ca.stats.to_dict() == cb.stats.to_dict()

    def test_colonist_different_seeds_differ(self):
        a = create_founding_ten(seed=42)
        b = create_founding_ten(seed=99)
        assert a[0].stats.to_dict() != b[0].stats.to_dict()

    def test_die_marks_inactive(self):
        c = create_founding_ten()[0]
        c.die(5, "test cause")
        assert not c.is_active()
        assert c.death_year == 5
        assert c.death_cause == "test cause"

    def test_exile_marks_inactive(self):
        c = create_founding_ten()[0]
        c.exile(10)
        assert not c.is_active()
        assert c.exile_year == 10

    def test_to_dict_roundtrip(self):
        c = create_founding_ten()[0]
        d = c.to_dict()
        c2 = Colonist.from_dict(d)
        assert c2.id == c.id
        assert c2.stats.to_dict() == c.stats.to_dict()

    def test_lispy_bindings_complete(self):
        c = create_founding_ten()[0]
        bindings = c.lispy_bindings()
        for name in STAT_NAMES:
            assert name in bindings
        for name in SKILL_NAMES:
            assert name in bindings

    def test_memory_capped_at_50(self):
        c = create_founding_ten()[0]
        for i in range(60):
            c.add_memory(i, f"event-{i}", 0.5)
        assert len(c.memories) == 50

    def test_evolve_stats_bounded(self):
        c = create_founding_ten()[0]
        rng = random.Random(42)
        for _ in range(200):
            c.evolve_stats("dust_storm", rng)
        for name in STAT_NAMES:
            val = getattr(c.stats, name)
            assert 0.0 <= val <= 1.0

    def test_evolve_skills_improves(self):
        c = create_founding_ten()[0]
        rng = random.Random(42)
        before = c.skills.terraforming
        for _ in range(50):
            c.evolve_skills("terraform", rng)
        assert c.skills.terraforming > before

    def test_birth_year_default(self):
        c = create_founding_ten()[0]
        assert c.birth_year == 0

    def test_to_dict_includes_birth_year(self):
        c = create_founding_ten()[0]
        d = c.to_dict()
        assert "birth_year" in d


# ──────────────────────────────────────────────────────────────
# Birth tests
# ──────────────────────────────────────────────────────────────

class TestBirths:
    """Tests for child creation."""

    def test_create_child_inherits_element(self):
        parents = create_founding_ten(seed=42)[:2]
        rng = random.Random(99)
        child = create_child(parents[0], parents[1], "child-0", 15, rng)
        assert child.element in (parents[0].element, parents[1].element)

    def test_create_child_stats_bounded(self):
        parents = create_founding_ten(seed=42)[:2]
        rng = random.Random(99)
        child = create_child(parents[0], parents[1], "child-0", 15, rng)
        for name in STAT_NAMES:
            val = getattr(child.stats, name)
            assert 0.0 <= val <= 1.0

    def test_create_child_skills_near_zero(self):
        parents = create_founding_ten(seed=42)[:2]
        rng = random.Random(99)
        child = create_child(parents[0], parents[1], "child-0", 15, rng)
        for name in SKILL_NAMES:
            val = getattr(child.skills, name)
            assert val < 0.3

    def test_create_child_birth_year_set(self):
        parents = create_founding_ten()[:2]
        child = create_child(parents[0], parents[1], "child-1", 25, random.Random(1))
        assert child.birth_year == 25

    def test_create_child_archetype_is_child(self):
        parents = create_founding_ten()[:2]
        child = create_child(parents[0], parents[1], "child-2", 20, random.Random(2))
        assert child.archetype == "child"

    def test_colonist_names_pool_exists(self):
        assert len(COLONIST_NAMES) >= 10


# ──────────────────────────────────────────────────────────────
# Resources tests
# ──────────────────────────────────────────────────────────────

class TestResources:
    """Tests for the resource model."""

    def test_default_resources(self):
        r = Resources()
        assert r.food == pytest.approx(0.7)
        assert r.air == pytest.approx(0.9)

    def test_clamp_enforced(self):
        r = Resources(food=1.5, water=-0.3)
        r.clamp()
        assert r.food == 1.0
        assert r.water == 0.0

    def test_critical_resources(self):
        r = Resources(food=0.1, water=0.05)
        crits = r.critical()
        assert "water" in crits
        assert "food" in crits

    def test_tick_resources_returns_delta(self):
        r = Resources()
        delta = tick_resources(r, 10, {}, {})
        for name in RESOURCE_NAMES:
            assert name in delta

    def test_tick_resources_bounded(self):
        r = Resources()
        for _ in range(200):
            tick_resources(r, 10, {}, {})
        for name in RESOURCE_NAMES:
            val = getattr(r, name)
            assert 0.0 <= val <= 1.0


# ──────────────────────────────────────────────────────────────
# Social graph tests
# ──────────────────────────────────────────────────────────────

class TestSocialGraph:
    """Tests for the social graph."""

    def test_initialize(self):
        sg = SocialGraph()
        ids = ["a", "b", "c"]
        sg.initialize(ids, random.Random(42))
        assert "a" in sg.edges
        assert "b" in sg.edges["a"]
        assert "a" not in sg.edges["a"]

    def test_add_colonist(self):
        sg = SocialGraph()
        ids = ["a", "b"]
        sg.initialize(ids, random.Random(42))
        sg.add_colonist("c", ["a", "b", "c"], random.Random(42))
        assert "c" in sg.edges
        assert "a" in sg.edges["c"]
        assert "c" in sg.edges["a"]

    def test_colony_cohesion_range(self):
        sg = SocialGraph()
        ids = ["a", "b", "c"]
        sg.initialize(ids, random.Random(42))
        coh = sg.colony_cohesion(ids)
        assert 0.0 <= coh <= 1.0

    def test_most_trusted_by(self):
        sg = SocialGraph()
        ids = ["a", "b", "c"]
        sg.initialize(ids, random.Random(42))
        best = sg.most_trusted_by("a", ids)
        assert best in ("b", "c")


# ──────────────────────────────────────────────────────────────
# Convergence tests
# ──────────────────────────────────────────────────────────────

class TestConvergence:
    """Tests for value convergence tracking."""

    def test_convergence_returns_all_stats(self):
        colonists = create_founding_ten()
        result = compute_value_convergence(colonists)
        for name in STAT_NAMES:
            assert name in result
        assert "convergence_score" in result

    def test_convergence_score_positive(self):
        colonists = create_founding_ten()
        result = compute_value_convergence(colonists)
        assert result["convergence_score"] >= 0.0

    def test_convergence_single_colonist(self):
        colonists = create_founding_ten()[:1]
        result = compute_value_convergence(colonists)
        assert result["convergence_score"] == 0.0

    def test_convergence_identical_colonists_zero(self):
        c = create_founding_ten()[0]
        c2 = Colonist.from_dict(c.to_dict())
        c2.id = "clone"
        result = compute_value_convergence([c, c2])
        assert result["convergence_score"] == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────
# Events tests
# ──────────────────────────────────────────────────────────────

class TestEvents:
    """Tests for event generation."""

    def test_generate_events_returns_list(self):
        events = generate_events(1, random.Random(42))
        assert isinstance(events, list)

    def test_events_have_required_fields(self):
        for ev in generate_events(5, random.Random(42)):
            assert ev.name
            assert 0.0 <= ev.severity <= 1.0
            assert isinstance(ev.effects, dict)

    def test_event_templates_nonempty(self):
        assert len(EVENT_TEMPLATES) > 0


# ──────────────────────────────────────────────────────────────
# Governance tests
# ──────────────────────────────────────────────────────────────

class TestGovernance:
    """Tests for governance proposals and voting."""

    def test_governance_state_defaults(self):
        gs = GovernanceState()
        assert gs.gov_type is not None

    def test_resolve_vote_passes(self):
        p = GovernanceProposal(id="p1", year=1, proposer_id="a",
                               gov_type="council", parameters={},
                               rationale="test",
                               votes_for=["a", "b", "c"],
                               votes_against=["d"])
        assert resolve_vote(p, 4) is True

    def test_resolve_vote_fails(self):
        p = GovernanceProposal(id="p2", year=1, proposer_id="a",
                               gov_type="council", parameters={},
                               rationale="test",
                               votes_for=["a"],
                               votes_against=["b", "c", "d"])
        assert resolve_vote(p, 4) is False


# ──────────────────────────────────────────────────────────────
# LisPy VM tests
# ──────────────────────────────────────────────────────────────

class TestLispyVM:
    """Tests for the safe-eval LisPy interpreter."""

    def test_basic_arithmetic(self):
        assert lispy_run("(+ 1 2)") == 3

    def test_let_binding(self):
        assert lispy_run("(let ((x 5)) x)") == 5

    def test_if_expression(self):
        assert lispy_run("(if (> 3 2) 10 20)") == 10

    def test_custom_bindings(self):
        result = lispy_run("(+ resolve empathy)",
                           extra_bindings={"resolve": 0.5, "empathy": 0.3})
        assert result == pytest.approx(0.8)

    def test_colonist_bindings_work(self):
        c = create_founding_ten()[0]
        result = lispy_run(c.decision_expr, extra_bindings=c.lispy_bindings())
        assert isinstance(result, (int, float))


# ──────────────────────────────────────────────────────────────
# Sub-simulation tests
# ──────────────────────────────────────────────────────────────

class TestSubSim:
    """Tests for sub-simulation spawning."""

    def test_spawn_subsim_basic(self):
        budget = SubSimBudget(year=1)
        log: list[SubSimResult] = []
        bindings = {"resolve": 0.5, "empathy": 0.3, "faith": 0.4}
        result = spawn_subsim(
            expression="(+ resolve empathy)",
            colonist_id="test", year=1,
            bindings=bindings, depth=1,
            budget=budget, log=log,
        )
        assert result.succeeded
        assert result.result == pytest.approx(0.8)

    def test_subsim_depth_limit(self):
        budget = SubSimBudget(year=1)
        log: list[SubSimResult] = []
        result = spawn_subsim(
            expression="(+ 1 1)", colonist_id="test", year=1,
            bindings={}, depth=4, budget=budget, log=log,
        )
        assert not result.succeeded

    def test_subsim_budget_tracking(self):
        budget = SubSimBudget(year=1)
        assert budget.can_spawn("test")


# ──────────────────────────────────────────────────────────────
# Engine integration tests
# ──────────────────────────────────────────────────────────────

class TestEngine:
    """Integration tests for the Mars100Engine."""

    def test_engine_creates(self):
        engine = Mars100Engine(seed=42, total_years=10)
        assert engine.year == 0
        assert len(engine.colonists) == 10

    def test_single_tick(self):
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.tick()
        assert result.year == 1
        assert isinstance(result.convergence, dict)
        assert isinstance(result.births, list)

    def test_ten_year_run(self):
        engine = Mars100Engine(seed=42, total_years=10)
        sim = engine.run()
        assert len(sim.years) == 10
        assert sim.total_deaths >= 0
        assert sim.total_births >= 0
        assert sim.convergence_trend in ("converging", "diverging", "stable")

    def test_full_hundred_year_run(self):
        engine = Mars100Engine(seed=42, total_years=100)
        sim = engine.run()
        assert len(sim.years) <= 100
        assert sim.total_subsims >= 0
        assert isinstance(sim.promoted_insights, list)

    def test_deterministic_runs(self):
        a = Mars100Engine(seed=42, total_years=20).run()
        b = Mars100Engine(seed=42, total_years=20).run()
        assert a.total_deaths == b.total_deaths
        assert a.total_subsims == b.total_subsims
        assert a.years[0].actions == b.years[0].actions

    def test_different_seeds_diverge(self):
        a = Mars100Engine(seed=42, total_years=20).run()
        b = Mars100Engine(seed=99, total_years=20).run()
        assert a.years[0].actions != b.years[0].actions

    def test_resources_stay_bounded(self):
        engine = Mars100Engine(seed=42, total_years=100)
        sim = engine.run()
        for year in sim.years:
            for name in RESOURCE_NAMES:
                assert 0.0 <= year.resources_after[name] <= 1.0

    def test_convergence_tracked(self):
        engine = Mars100Engine(seed=42, total_years=50)
        sim = engine.run()
        for year in sim.years:
            assert "convergence_score" in year.convergence

    def test_to_dict_serializable(self):
        import json
        engine = Mars100Engine(seed=42, total_years=10)
        sim = engine.run()
        d = sim.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 100

    def test_year_result_to_dict(self):
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.tick()
        d = result.to_dict()
        assert "convergence" in d
        assert "births" in d

    def test_insight_queue_initialized(self):
        engine = Mars100Engine(seed=42)
        assert engine.insight_queue == []
        assert engine.promoted_insights == []

    def test_callback_invoked(self):
        engine = Mars100Engine(seed=42, total_years=5)
        years_seen: list[int] = []
        engine.run(callback=lambda r: years_seen.append(r.year))
        assert len(years_seen) == 5

    def test_simulation_result_summary(self):
        engine = Mars100Engine(seed=42, total_years=10)
        sim = engine.run()
        d = sim.to_dict()
        assert "summary" in d
        assert "total_births" in d["summary"]
        assert "convergence_trend" in d["summary"]
        assert d["_meta"]["version"] == "12.1"


# ──────────────────────────────────────────────────────────────
# Property-based invariants
# ──────────────────────────────────────────────────────────────

class TestInvariants:
    """Property checks: physical bounds, conservation, determinism."""

    @pytest.mark.parametrize("seed", [1, 42, 99, 777, 12345])
    def test_stats_bounded_after_full_run(self, seed: int):
        engine = Mars100Engine(seed=seed, total_years=100)
        sim = engine.run()
        for cd in sim.final_colonists:
            for name in STAT_NAMES:
                val = cd["stats"][name]
                assert 0.0 <= val <= 1.0, f"stat {name} = {val} for {cd['id']}"

    @pytest.mark.parametrize("seed", [1, 42, 99])
    def test_resources_bounded_every_year(self, seed: int):
        engine = Mars100Engine(seed=seed, total_years=50)
        sim = engine.run()
        for yr in sim.years:
            for name in RESOURCE_NAMES:
                assert 0.0 <= yr.resources_after[name] <= 1.0

    def test_deaths_monotonically_increase(self):
        engine = Mars100Engine(seed=42, total_years=100)
        sim = engine.run()
        cumulative = 0
        for yr in sim.years:
            cumulative += len(yr.deaths)
        assert cumulative == sim.total_deaths

    def test_births_monotonically_increase(self):
        engine = Mars100Engine(seed=42, total_years=100)
        sim = engine.run()
        cumulative = 0
        for yr in sim.years:
            cumulative += len(yr.births)
        assert cumulative == sim.total_births

    def test_colonist_count_consistent(self):
        """Total colonists = founding 10 + births."""
        engine = Mars100Engine(seed=42, total_years=100)
        sim = engine.run()
        assert len(sim.final_colonists) == 10 + sim.total_births + sim.total_immigrants
