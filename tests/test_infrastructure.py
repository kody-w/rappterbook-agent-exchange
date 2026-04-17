"""Tests for the infrastructure / tech-tree module."""
from __future__ import annotations

import random
import pytest

from src.mars100.infrastructure import (
    TECH_BY_ID,
    TECH_TREE,
    ActiveProject,
    InfrastructureState,
    TechSpec,
    available_techs,
    can_afford,
    choose_project,
    compute_operating_costs,
    compute_resource_modifiers,
    start_project,
    tick_infrastructure,
    validate_tech_tree,
)


class _FakeColonist:
    def __init__(self, best_skill="coding"):
        self.skills = self
        self._best = best_skill
    def best_skill(self):
        return self._best


class _FakeResources:
    def __init__(self, food=0.5, water=0.5, power=0.5, air=0.5, medicine=0.5):
        self.food = food
        self.water = water
        self.power = power
        self.air = air
        self.medicine = medicine


RICH = {"food": 0.9, "water": 0.9, "power": 0.9, "air": 0.9, "medicine": 0.9}
POOR = {"food": 0.0, "water": 0.0, "power": 0.0, "air": 0.0, "medicine": 0.0}


class TestTechTreeStructure:
    def test_six_techs(self):
        assert len(TECH_TREE) == 7
    def test_unique_ids(self):
        ids = [t.id for t in TECH_TREE]
        assert len(ids) == len(set(ids))
    def test_by_id_lookup(self):
        for t in TECH_TREE:
            assert TECH_BY_ID[t.id] is t
    def test_validate_passes(self):
        assert validate_tech_tree() == []
    def test_positive_build_time(self):
        for t in TECH_TREE:
            assert t.build_time >= 1
    def test_positive_workers(self):
        for t in TECH_TREE:
            assert t.workers_needed >= 1
    def test_resource_costs_bounded(self):
        for t in TECH_TREE:
            for res, cost in t.resource_cost.items():
                assert 0 <= cost <= 1, f"{t.id}.{res}={cost}"
    def test_effects_non_empty(self):
        for t in TECH_TREE:
            assert len(t.effects) > 0
    def test_to_dict_roundtrip(self):
        for t in TECH_TREE:
            d = t.to_dict()
            assert d["id"] == t.id
            assert d["build_time"] == t.build_time


class TestInfrastructureState:
    def test_default_empty(self):
        s = InfrastructureState()
        assert s.completed == []
        assert s.project is None
        assert s.abandoned == []
    def test_to_dict(self):
        s = InfrastructureState(completed=["greenhouse_dome"])
        d = s.to_dict()
        assert d["completed"] == ["greenhouse_dome"]
        assert d["active_project"] is None
    def test_to_dict_with_project(self):
        s = InfrastructureState(project=ActiveProject(tech_id="med_bay", progress=2))
        d = s.to_dict()
        assert d["active_project"]["tech_id"] == "med_bay"
        assert d["active_project"]["progress"] == 2


class TestAvailableTechs:
    def test_all_available_initially(self):
        assert len(available_techs(InfrastructureState())) == 7
    def test_excludes_completed(self):
        s = InfrastructureState(completed=["greenhouse_dome"])
        ids = [t.id for t in available_techs(s)]
        assert "greenhouse_dome" not in ids
        assert len(ids) == 6
    def test_excludes_abandoned(self):
        s = InfrastructureState(abandoned=["power_grid"])
        ids = [t.id for t in available_techs(s)]
        assert "power_grid" not in ids


class TestCanAfford:
    def test_rich_can_afford_all(self):
        for t in TECH_TREE:
            assert can_afford(t, RICH)
    def test_poor_cannot_afford(self):
        for t in TECH_TREE:
            assert not can_afford(t, POOR)
    def test_exact_amount(self):
        tech = TECH_BY_ID["greenhouse_dome"]
        resources = dict(tech.resource_cost)
        assert can_afford(tech, resources)
    def test_one_short(self):
        tech = TECH_BY_ID["greenhouse_dome"]
        resources = {k: v - 0.01 for k, v in tech.resource_cost.items()}
        assert not can_afford(tech, resources)


class TestChooseProject:
    def test_returns_none_when_active(self):
        s = InfrastructureState(project=ActiveProject(tech_id="med_bay"))
        result = choose_project(s, RICH, [_FakeColonist()], random.Random(42))
        assert result is None
    def test_returns_none_when_poor(self):
        s = InfrastructureState()
        result = choose_project(s, POOR, [_FakeColonist()], random.Random(42))
        assert result is None
    def test_returns_tech_when_rich(self):
        s = InfrastructureState()
        result = choose_project(s, RICH, [_FakeColonist()], random.Random(42))
        assert result is not None
        assert result.id in TECH_BY_ID
    def test_skill_affinity_matters(self):
        s = InfrastructureState()
        coders = [_FakeColonist("coding") for _ in range(5)]
        results = set()
        for seed in range(50):
            t = choose_project(s, RICH, coders, random.Random(seed))
            if t:
                results.add(t.id)
        assert "research_lab" in results or "power_grid" in results
    def test_returns_none_all_complete(self):
        s = InfrastructureState(completed=[t.id for t in TECH_TREE])
        result = choose_project(s, RICH, [_FakeColonist()], random.Random(42))
        assert result is None


class TestStartProject:
    def test_deducts_resources(self):
        s = InfrastructureState()
        tech = TECH_BY_ID["greenhouse_dome"]
        res = _FakeResources(power=0.5)
        start_project(s, tech, res, year=10)
        assert res.power < 0.5
        assert s.project is not None
        assert s.project.tech_id == "greenhouse_dome"
    def test_history_logged(self):
        s = InfrastructureState()
        tech = TECH_BY_ID["med_bay"]
        start_project(s, tech, _FakeResources(), year=5)
        assert len(s.history) == 1
        assert s.history[0]["event"] == "started"
    def test_does_not_go_below_zero(self):
        s = InfrastructureState()
        tech = TECH_BY_ID["power_grid"]
        res = _FakeResources(power=0.01, air=0.01)
        start_project(s, tech, res, year=1)
        assert res.power >= 0.0
        assert res.air >= 0.0


class TestTickInfrastructure:
    def test_noop_without_project(self):
        s = InfrastructureState()
        result = tick_infrastructure(s, researcher_count=3, skill_avg=0.5, year=1)
        assert result is None
    def test_progress_advances(self):
        s = InfrastructureState(project=ActiveProject(tech_id="water_recycler"))
        tick_infrastructure(s, researcher_count=5, skill_avg=0.5, year=1)
        assert s.project is not None
        assert s.project.progress >= 1
    def test_completion(self):
        tech = TECH_BY_ID["water_recycler"]
        proj = ActiveProject(tech_id="water_recycler", progress=tech.build_time - 1)
        s = InfrastructureState(project=proj)
        result = tick_infrastructure(s, researcher_count=5, skill_avg=0.5, year=10)
        assert result is not None
        assert result["event"] == "completed"
        assert "water_recycler" in s.completed
        assert s.project is None
    def test_stall_and_abandon(self):
        s = InfrastructureState(
            project=ActiveProject(tech_id="power_grid", stall_years=2))
        result = tick_infrastructure(s, researcher_count=0, skill_avg=0.0, year=20)
        assert result is not None
        assert result["event"] == "abandoned"
        assert "power_grid" in s.abandoned
        assert s.project is None
    def test_stall_resets_on_staff(self):
        s = InfrastructureState(
            project=ActiveProject(tech_id="power_grid", stall_years=2))
        tick_infrastructure(s, researcher_count=5, skill_avg=0.5, year=20)
        assert s.project is not None
        assert s.project.stall_years == 0
    def test_skill_bonus(self):
        s1 = InfrastructureState(project=ActiveProject(tech_id="research_lab"))
        s2 = InfrastructureState(project=ActiveProject(tech_id="research_lab"))
        tick_infrastructure(s1, researcher_count=5, skill_avg=0.0, year=1)
        tick_infrastructure(s2, researcher_count=5, skill_avg=1.0, year=1)
        assert s2.project.progress >= s1.project.progress


class TestResourceModifiers:
    def test_empty(self):
        assert compute_resource_modifiers([]) == {}
    def test_single_tech(self):
        mods = compute_resource_modifiers(["greenhouse_dome"])
        assert mods["food_spoilage_mult"] == pytest.approx(0.5)
    def test_stacking(self):
        mods = compute_resource_modifiers(["greenhouse_dome", "med_bay"])
        assert "food_spoilage_mult" in mods
        assert "death_rate_mult" in mods
    def test_all_completed(self):
        mods = compute_resource_modifiers([t.id for t in TECH_TREE])
        for v in mods.values():
            assert 0 < v < 2.0


class TestOperatingCosts:
    def test_empty(self):
        assert compute_operating_costs([]) == {}
    def test_single_tech(self):
        costs = compute_operating_costs(["greenhouse_dome"])
        assert costs["power"] > 0
    def test_additive(self):
        c1 = compute_operating_costs(["greenhouse_dome"])
        c2 = compute_operating_costs(["greenhouse_dome", "water_recycler"])
        assert c2.get("power", 0) >= c1.get("power", 0)


class TestFullBuildCycle:
    def test_build_to_completion(self):
        s = InfrastructureState()
        tech = TECH_BY_ID["water_recycler"]
        res = _FakeResources(power=0.5, water=0.5, air=0.5)
        start_project(s, tech, res, year=1)
        for year in range(2, 20):
            result = tick_infrastructure(s, researcher_count=5, skill_avg=0.8, year=year)
            if result and result["event"] == "completed":
                break
        assert "water_recycler" in s.completed
        assert s.project is None
    def test_multiple_completions(self):
        s = InfrastructureState()
        rng = random.Random(42)
        colonists = [_FakeColonist("coding") for _ in range(5)]
        res = _FakeResources(food=0.99, water=0.99, power=0.99, air=0.99, medicine=0.99)
        for year in range(1, 80):
            if s.project is None:
                tech = choose_project(s, {"food": res.food, "water": res.water,
                                          "power": res.power, "air": res.air,
                                          "medicine": res.medicine}, colonists, rng)
                if tech:
                    start_project(s, tech, res, year=year)
            tick_infrastructure(s, researcher_count=5, skill_avg=0.8, year=year)
        assert len(s.completed) >= 1, f"Completed none: {s.completed}"


class TestDeterminism:
    def test_same_seed_same_result(self):
        colonists = [_FakeColonist("hydroponics") for _ in range(3)]
        results = []
        for _ in range(3):
            s = InfrastructureState()
            rng = random.Random(12345)
            t = choose_project(s, RICH, colonists, rng)
            results.append(t.id if t else None)
        assert results[0] == results[1] == results[2]


class TestEngineIntegration:
    def _engine(self):
        try:
            from src.mars100.engine import Mars100Engine
            return Mars100Engine
        except ImportError:
            pytest.skip("engine not available")

    def test_engine_has_infra(self):
        E = self._engine()
        engine = E(seed=42)
        assert hasattr(engine, "infra")

    def test_10_year_run(self):
        E = self._engine()
        engine = E(seed=42, total_years=10)
        result = engine.run()
        d = result.to_dict()
        assert len(d["years"]) == 10

    def test_100_year_completions(self):
        E = self._engine()
        engine = E(seed=42, total_years=100)
        result = engine.run()
        d = result.to_dict()
        infra = d.get("infrastructure", {})
        completed = infra.get("completed", [])
        assert len(completed) >= 1, "100-year sim should complete at least 1 tech"

    def test_infra_in_year_results(self):
        E = self._engine()
        engine = E(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        for yr in d["years"]:
            assert "infrastructure" in yr

    def test_research_action_exists(self):
        E = self._engine()
        engine = E(seed=42, total_years=20)
        result = engine.run()
        d = result.to_dict()
        all_actions = []
        for yr in d["years"]:
            for cid, action in yr.get("actions", {}).items():
                all_actions.append(action)
        assert "research" in all_actions, "research action should appear in 20 years"

    def test_death_rate_reduced_by_med_bay(self):
        mods = compute_resource_modifiers(["med_bay"])
        assert mods["death_rate_mult"] < 1.0

    def test_event_damage_reduced_by_shelter(self):
        mods = compute_resource_modifiers(["shelter_reinforcement"])
        assert mods["event_damage_mult"] < 1.0

    def test_subsim_budget_boosted_by_lab(self):
        mods = compute_resource_modifiers(["research_lab"])
        assert mods["subsim_budget_mult"] > 1.0
