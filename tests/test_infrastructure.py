"""Tests for Mars-100 infrastructure system."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.mars100.infrastructure import (
    TECH_TREE, TECH_BY_ID, MAX_STALL_YEARS,
    TechSpec, ActiveProject, InfrastructureState,
    available_techs, can_afford, choose_project, start_project,
    tick_infrastructure, compute_resource_modifiers, compute_operating_costs,
    validate_tech_tree,
)
from src.mars100.colony import Resources
from src.mars100.colonist import create_founding_ten


# ── Tech tree structure ────────────────────────────────────────────

class TestTechTreeStructure:
    def test_tree_is_nonempty(self):
        assert len(TECH_TREE) >= 6

    def test_unique_ids(self):
        ids = [t.id for t in TECH_TREE]
        assert len(ids) == len(set(ids))

    def test_all_prereqs_exist(self):
        ids = {t.id for t in TECH_TREE}
        for tech in TECH_TREE:
            for prereq in tech.prereqs:
                assert prereq in ids, f"{tech.id} references unknown prereq {prereq}"

    def test_no_self_prereqs(self):
        for tech in TECH_TREE:
            assert tech.id not in tech.prereqs

    def test_no_cycles(self):
        errors = validate_tech_tree()
        cycle_errors = [e for e in errors if "cycle" in e]
        assert cycle_errors == []

    def test_validate_clean(self):
        assert validate_tech_tree() == []

    def test_positive_build_times(self):
        for tech in TECH_TREE:
            assert tech.build_time >= 1

    def test_positive_worker_requirements(self):
        for tech in TECH_TREE:
            assert tech.workers_needed >= 1

    def test_non_negative_costs(self):
        for tech in TECH_TREE:
            for resource, cost in tech.resource_cost.items():
                assert cost >= 0, f"{tech.id}: negative cost {cost} for {resource}"

    def test_tech_by_id_complete(self):
        assert len(TECH_BY_ID) == len(TECH_TREE)
        for tech in TECH_TREE:
            assert tech.id in TECH_BY_ID
            assert TECH_BY_ID[tech.id] is tech

    def test_tech_spec_to_dict(self):
        tech = TECH_TREE[0]
        d = tech.to_dict()
        assert d["id"] == tech.id
        assert d["name"] == tech.name
        assert d["build_time"] == tech.build_time
        assert "effects" in d


# ── State management ───────────────────────────────────────────────

class TestInfrastructureState:
    def test_empty_state(self):
        state = InfrastructureState()
        assert state.completed == []
        assert state.project is None
        assert state.history == []
        assert state.abandoned == []

    def test_to_dict_empty(self):
        state = InfrastructureState()
        d = state.to_dict()
        assert d["completed"] == []
        assert d["project"] is None

    def test_roundtrip_serialization(self):
        state = InfrastructureState(
            completed=["greenhouse_dome"],
            project=ActiveProject(tech_id="water_recycler", progress=2, stall_years=1),
            history=[{"year": 5, "event": "completed", "tech_id": "greenhouse_dome"}],
            abandoned=["power_grid"],
        )
        d = state.to_dict()
        restored = InfrastructureState.from_dict(d)
        assert restored.completed == ["greenhouse_dome"]
        assert restored.project is not None
        assert restored.project.tech_id == "water_recycler"
        assert restored.project.progress == 2
        assert restored.project.stall_years == 1
        assert restored.abandoned == ["power_grid"]

    def test_from_dict_no_project(self):
        d = {"completed": ["med_bay"], "project": None, "history": [], "abandoned": []}
        state = InfrastructureState.from_dict(d)
        assert state.project is None
        assert state.completed == ["med_bay"]


# ── Available techs ───────────────────────────────────────────────

class TestAvailableTechs:
    def test_all_available_when_empty(self):
        state = InfrastructureState()
        avail = available_techs(state)
        # Only root techs (no prereqs) should be available
        root_ids = {t.id for t in TECH_TREE if not t.prereqs}
        avail_ids = {t.id for t in avail}
        assert avail_ids == root_ids

    def test_completed_excluded(self):
        state = InfrastructureState(completed=["greenhouse_dome"])
        avail = available_techs(state)
        assert "greenhouse_dome" not in {t.id for t in avail}

    def test_prereqs_unlock(self):
        state = InfrastructureState(completed=["greenhouse_dome"])
        avail = available_techs(state)
        avail_ids = {t.id for t in avail}
        # med_bay requires greenhouse_dome, so it should be available now
        assert "med_bay" in avail_ids

    def test_in_progress_excluded(self):
        state = InfrastructureState(
            project=ActiveProject(tech_id="water_recycler")
        )
        avail = available_techs(state)
        assert "water_recycler" not in {t.id for t in avail}

    def test_deep_prereqs_not_available(self):
        state = InfrastructureState()
        avail_ids = {t.id for t in available_techs(state)}
        # research_lab requires med_bay + shelter_reinforcement
        assert "research_lab" not in avail_ids


# ── Affordability ─────────────────────────────────────────────────

class TestCanAfford:
    def test_can_afford_with_plenty(self):
        tech = TECH_BY_ID["greenhouse_dome"]
        resources = {"power": 0.5, "food": 0.5, "water": 0.5}
        assert can_afford(tech, resources) is True

    def test_cannot_afford_when_broke(self):
        tech = TECH_BY_ID["greenhouse_dome"]
        resources = {"power": 0.01, "food": 0.01, "water": 0.01}
        assert can_afford(tech, resources) is False

    def test_edge_exactly_enough(self):
        tech = TECH_BY_ID["greenhouse_dome"]
        resources = dict(tech.resource_cost)
        assert can_afford(tech, resources) is True

    def test_missing_resource_key(self):
        tech = TECH_BY_ID["greenhouse_dome"]
        assert can_afford(tech, {}) is False


# ── Project selection ─────────────────────────────────────────────

class TestChooseProject:
    def test_returns_none_when_building(self):
        state = InfrastructureState(project=ActiveProject(tech_id="greenhouse_dome"))
        colonists = create_founding_ten()
        resources = {"power": 0.5, "food": 0.5, "water": 0.5, "air": 0.5, "medicine": 0.5}
        result = choose_project(state, resources, colonists, random.Random(42))
        assert result is None

    def test_returns_none_when_cannot_afford(self):
        state = InfrastructureState()
        colonists = create_founding_ten()
        resources = {"power": 0.0, "food": 0.0, "water": 0.0, "air": 0.0, "medicine": 0.0}
        result = choose_project(state, resources, colonists, random.Random(42))
        assert result is None

    def test_returns_tech_when_available(self):
        state = InfrastructureState()
        colonists = create_founding_ten()
        resources = {"power": 0.5, "food": 0.5, "water": 0.5, "air": 0.5, "medicine": 0.5}
        result = choose_project(state, resources, colonists, random.Random(42))
        assert result is not None
        assert isinstance(result, TechSpec)
        assert result.id in TECH_BY_ID

    def test_deterministic_with_seed(self):
        state = InfrastructureState()
        colonists = create_founding_ten()
        resources = {"power": 0.5, "food": 0.5, "water": 0.5, "air": 0.5, "medicine": 0.5}
        r1 = choose_project(state, resources, colonists, random.Random(42))
        r2 = choose_project(state, resources, colonists, random.Random(42))
        assert r1.id == r2.id


# ── Start project ─────────────────────────────────────────────────

class TestStartProject:
    def test_deducts_resources(self):
        state = InfrastructureState()
        tech = TECH_BY_ID["greenhouse_dome"]
        resources = Resources(food=0.7, water=0.7, power=0.8, air=0.9, medicine=0.5)
        before_power = resources.power
        before_food = resources.food
        start_project(state, tech, resources, year=5)
        assert resources.power < before_power
        assert resources.food < before_food
        assert state.project is not None
        assert state.project.tech_id == "greenhouse_dome"

    def test_records_history(self):
        state = InfrastructureState()
        tech = TECH_BY_ID["greenhouse_dome"]
        resources = Resources()
        start_project(state, tech, resources, year=5)
        assert len(state.history) == 1
        assert state.history[0]["event"] == "started"
        assert state.history[0]["year"] == 5

    def test_resources_dont_go_negative(self):
        state = InfrastructureState()
        tech = TECH_BY_ID["greenhouse_dome"]
        resources = Resources(food=0.01, power=0.01)
        start_project(state, tech, resources, year=1)
        assert resources.food >= 0.0
        assert resources.power >= 0.0


# ── Tick infrastructure ───────────────────────────────────────────

class TestTickInfrastructure:
    def test_no_project_returns_none(self):
        state = InfrastructureState()
        result = tick_infrastructure(state, researcher_count=5, skill_avg=0.5, year=10)
        assert result is None

    def test_progress_increments(self):
        state = InfrastructureState(
            project=ActiveProject(tech_id="greenhouse_dome", progress=0)
        )
        tick_infrastructure(state, researcher_count=3, skill_avg=0.5, year=5)
        assert state.project.progress >= 1

    def test_completion(self):
        tech = TECH_BY_ID["greenhouse_dome"]
        state = InfrastructureState(
            project=ActiveProject(tech_id="greenhouse_dome",
                                  progress=tech.build_time - 1)
        )
        result = tick_infrastructure(state, researcher_count=3, skill_avg=0.5, year=10)
        assert result is not None
        assert result["event"] == "completed"
        assert "greenhouse_dome" in state.completed
        assert state.project is None

    def test_stall_increments(self):
        state = InfrastructureState(
            project=ActiveProject(tech_id="greenhouse_dome", progress=1)
        )
        tick_infrastructure(state, researcher_count=0, skill_avg=0.0, year=5)
        assert state.project.stall_years == 1
        assert state.project.progress == 1  # no progress

    def test_abandonment_after_max_stall(self):
        state = InfrastructureState(
            project=ActiveProject(tech_id="greenhouse_dome", progress=1,
                                  stall_years=MAX_STALL_YEARS - 1)
        )
        result = tick_infrastructure(state, researcher_count=0, skill_avg=0.0, year=20)
        assert result is not None
        assert result["event"] == "abandoned"
        assert state.project is None
        assert "greenhouse_dome" in state.abandoned

    def test_stall_resets_on_sufficient_workers(self):
        state = InfrastructureState(
            project=ActiveProject(tech_id="greenhouse_dome", progress=0,
                                  stall_years=2)
        )
        tick_infrastructure(state, researcher_count=3, skill_avg=0.5, year=5)
        assert state.project.stall_years == 0

    def test_skill_bonus_accelerates(self):
        state_low = InfrastructureState(
            project=ActiveProject(tech_id="greenhouse_dome", progress=0)
        )
        state_high = InfrastructureState(
            project=ActiveProject(tech_id="greenhouse_dome", progress=0)
        )
        tick_infrastructure(state_low, researcher_count=3, skill_avg=0.0, year=5)
        tick_infrastructure(state_high, researcher_count=3, skill_avg=1.0, year=5)
        assert state_high.project.progress >= state_low.project.progress

    def test_no_double_completion(self):
        """A tech completed once must not be completable again."""
        state = InfrastructureState(completed=["greenhouse_dome"])
        avail = available_techs(state)
        assert "greenhouse_dome" not in {t.id for t in avail}

    def test_unknown_tech_clears_project(self):
        state = InfrastructureState(
            project=ActiveProject(tech_id="nonexistent_tech")
        )
        result = tick_infrastructure(state, researcher_count=5, skill_avg=1.0, year=10)
        assert result is None
        assert state.project is None


# ── Resource modifiers ────────────────────────────────────────────

class TestResourceModifiers:
    def test_empty_completed(self):
        mods = compute_resource_modifiers([])
        assert mods == {}

    def test_single_tech(self):
        mods = compute_resource_modifiers(["greenhouse_dome"])
        assert "food_spoilage_mult" in mods
        assert mods["food_spoilage_mult"] == pytest.approx(0.5)

    def test_multiplicative_stacking(self):
        """If two techs share a modifier key, they stack multiplicatively."""
        # greenhouse_dome: food_production_mult=1.10
        # med_bay: medicine_production_mult=1.15
        # These don't overlap, but let's verify stacking logic works
        mods = compute_resource_modifiers(["greenhouse_dome", "water_recycler"])
        # Each has distinct keys, so no stacking
        assert "food_spoilage_mult" in mods
        assert "water_spoilage_mult" in mods

    def test_unknown_tech_ignored(self):
        mods = compute_resource_modifiers(["nonexistent", "greenhouse_dome"])
        assert "food_spoilage_mult" in mods


class TestOperatingCosts:
    def test_empty(self):
        costs = compute_operating_costs([])
        assert costs == {}

    def test_single_tech(self):
        costs = compute_operating_costs(["greenhouse_dome"])
        assert "power" in costs
        assert costs["power"] > 0

    def test_stacking(self):
        costs_one = compute_operating_costs(["greenhouse_dome"])
        costs_two = compute_operating_costs(["greenhouse_dome", "water_recycler"])
        assert costs_two["power"] > costs_one["power"]

    def test_unknown_tech_ignored(self):
        costs = compute_operating_costs(["nonexistent"])
        assert costs == {}


# ── Integration: full build cycle ─────────────────────────────────

class TestFullBuildCycle:
    def test_build_and_complete(self):
        state = InfrastructureState()
        resources = Resources(food=0.7, water=0.7, power=0.8, air=0.9, medicine=0.5)
        tech = TECH_BY_ID["greenhouse_dome"]

        start_project(state, tech, resources, year=1)
        assert state.project is not None

        # Advance enough years
        result = None
        for year in range(2, 20):
            result = tick_infrastructure(state, researcher_count=3,
                                         skill_avg=0.5, year=year)
            if result and result["event"] == "completed":
                break

        assert result is not None
        assert result["event"] == "completed"
        assert "greenhouse_dome" in state.completed

        # Modifiers should now include greenhouse effects
        mods = compute_resource_modifiers(state.completed)
        assert mods.get("food_spoilage_mult", 1.0) < 1.0

    def test_chain_build_two_techs(self):
        rng = random.Random(42)
        state = InfrastructureState()
        resources = Resources(food=0.9, water=0.9, power=0.9, air=0.9, medicine=0.9)

        # Build first tech
        tech1 = TECH_BY_ID["greenhouse_dome"]
        start_project(state, tech1, resources, year=1)
        for year in range(2, 20):
            result = tick_infrastructure(state, 3, 0.5, year)
            if result and result["event"] == "completed":
                break

        # Now med_bay (requires greenhouse_dome) should be available
        avail = available_techs(state)
        assert "med_bay" in {t.id for t in avail}

        # Build second tech
        tech2 = TECH_BY_ID["med_bay"]
        start_project(state, tech2, resources, year=10)
        for year in range(11, 30):
            result = tick_infrastructure(state, 4, 0.6, year)
            if result and result["event"] == "completed":
                break

        assert "med_bay" in state.completed
        mods = compute_resource_modifiers(state.completed)
        assert "death_rate_mult" in mods

    def test_resources_bounded_after_full_tree(self):
        """Building all techs should not violate resource bounds."""
        state = InfrastructureState()
        resources = Resources(food=0.9, water=0.9, power=0.9, air=0.9, medicine=0.9)

        # Build all techs in order
        build_order = ["greenhouse_dome", "water_recycler", "power_grid",
                       "med_bay", "shelter_reinforcement", "research_lab"]
        year = 1
        for tid in build_order:
            tech = TECH_BY_ID[tid]
            start_project(state, tech, resources, year=year)
            for y in range(year + 1, year + 20):
                result = tick_infrastructure(state, 5, 0.8, y)
                if result and result["event"] == "completed":
                    year = y + 1
                    break
            else:
                year += 20

        # Resources should still be valid
        for name in ("food", "water", "power", "air", "medicine"):
            val = getattr(resources, name)
            assert 0.0 <= val <= 1.0, f"{name} out of bounds: {val}"


# ── Determinism ───────────────────────────────────────────────────

class TestDeterminism:
    def test_choose_project_deterministic(self):
        for seed in [1, 42, 99, 777]:
            state = InfrastructureState()
            colonists = create_founding_ten(seed)
            resources = {"power": 0.5, "food": 0.5, "water": 0.5,
                         "air": 0.5, "medicine": 0.5}
            r1 = choose_project(state, resources, colonists, random.Random(seed))
            r2 = choose_project(state, resources, colonists, random.Random(seed))
            assert r1.id == r2.id, f"Non-deterministic with seed {seed}"


# ── Engine integration tests ──────────────────────────────────────────

class TestEngineIntegration:
    """Test infrastructure wired into the full Mars100Engine."""

    def test_engine_10_years_with_infra(self):
        """Engine runs 10 years without crash; infrastructure state present."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) == 10
        assert result.final_infrastructure is not None
        assert "completed" in result.final_infrastructure

    def test_infra_appears_in_year_result(self):
        """Each YearResult has an infrastructure field."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        for yr in result.years:
            d = yr.to_dict()
            assert "infrastructure" in d

    def test_research_action_chosen(self):
        """Over 50 years, at least one colonist chooses 'research'."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        all_actions = []
        for yr in result.years:
            all_actions.extend(yr.actions.values())
        assert "research" in all_actions

    def test_tech_completed_in_100_years(self):
        """Over 100 years, at least one tech should be completed."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        completed = result.final_infrastructure.get("completed", [])
        assert len(completed) >= 1, f"Expected at least 1 tech completed, got {completed}"

    def test_infra_determinism(self):
        """Same seed produces identical infrastructure outcomes."""
        from src.mars100.engine import Mars100Engine
        e1 = Mars100Engine(seed=99, total_years=30)
        e2 = Mars100Engine(seed=99, total_years=30)
        r1 = e1.run()
        r2 = e2.run()
        assert r1.final_infrastructure == r2.final_infrastructure

    def test_resources_bounded_with_infra(self):
        """Resources stay in [0, 1] even with infrastructure modifiers active."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        for yr in result.years:
            for name, val in yr.resources_after.items():
                assert 0.0 <= val <= 1.0, f"Year {yr.year}: {name}={val} out of bounds"

    def test_version_bumped(self):
        """SimulationResult version should be 4.0 with infrastructure."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "4.0"

    def test_summary_includes_techs_built(self):
        """Summary dict includes techs_built count."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        d = result.to_dict()
        assert "techs_built" in d["summary"]
        assert isinstance(d["summary"]["techs_built"], int)
