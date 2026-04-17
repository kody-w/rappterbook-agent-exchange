"""Tests for the ecology organ (engine v9.0)."""
from __future__ import annotations

import random

import pytest

from src.mars100.ecology import (
    EcologyState,
    EcologyTickResult,
    MILESTONES,
    TERRAFORM_ATMOSPHERE_DELTA,
    TERRAFORM_SOIL_DELTA,
    FARM_SOIL_DELTA,
    EXPLORE_WATER_DELTA,
    WATER_EXTRACTION_PER_CAPITA,
    WATER_RECYCLER_EXTRACTION_REDUCTION,
    GREENHOUSE_SOIL_BONUS,
    _clamp,
    check_milestones,
    compute_ecology_production_modifiers,
    get_milestone_label,
    tick_ecology,
)


# ---------------------------------------------------------------------------
# EcologyState basics
# ---------------------------------------------------------------------------

class TestEcologyState:
    def test_defaults(self) -> None:
        s = EcologyState()
        assert s.atmosphere_pressure == 0.01
        assert s.soil_fertility == 0.02
        assert s.water_table == 0.10
        assert s.radiation_level == 0.90
        assert s.milestones_achieved == []

    def test_biodiversity_zero_below_threshold(self) -> None:
        s = EcologyState()
        assert s.biodiversity == 0.0

    def test_biodiversity_positive_above_threshold(self) -> None:
        s = EcologyState(atmosphere_pressure=0.06, soil_fertility=0.12)
        assert s.biodiversity > 0.0

    def test_habitability_bounds(self) -> None:
        for _ in range(50):
            s = EcologyState(
                atmosphere_pressure=random.random(),
                soil_fertility=random.random(),
                water_table=random.random(),
                radiation_level=random.random(),
            )
            assert 0.0 <= s.habitability <= 1.0

    def test_habitability_improves_with_better_state(self) -> None:
        bad = EcologyState()
        good = EcologyState(atmosphere_pressure=0.10, soil_fertility=0.20,
                            water_table=0.20, radiation_level=0.60)
        assert good.habitability > bad.habitability

    def test_to_dict_roundtrip(self) -> None:
        s = EcologyState(atmosphere_pressure=0.05, soil_fertility=0.10,
                         water_table=0.15, radiation_level=0.80,
                         milestones_achieved=["first_microbes"])
        d = s.to_dict()
        s2 = EcologyState.from_dict(d)
        assert abs(s2.atmosphere_pressure - s.atmosphere_pressure) < 1e-4
        assert abs(s2.soil_fertility - s.soil_fertility) < 1e-4
        assert s2.milestones_achieved == s.milestones_achieved

    def test_from_dict_defaults(self) -> None:
        s = EcologyState.from_dict({})
        assert s.atmosphere_pressure == 0.01
        assert s.radiation_level == 0.90


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_clamp_in_range(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_clamp_below(self) -> None:
        assert _clamp(-0.1) == 0.0

    def test_clamp_above(self) -> None:
        assert _clamp(1.5) == 1.0

    def test_milestone_label_known(self) -> None:
        label = get_milestone_label("first_microbes")
        assert "microbe" in label.lower()

    def test_milestone_label_unknown(self) -> None:
        assert get_milestone_label("nonexistent") == "nonexistent"


# ---------------------------------------------------------------------------
# Milestone detection
# ---------------------------------------------------------------------------

class TestMilestones:
    def test_no_milestones_at_start(self) -> None:
        s = EcologyState()
        assert check_milestones(s) == []

    def test_first_microbes_achieved(self) -> None:
        s = EcologyState(atmosphere_pressure=0.05, soil_fertility=0.10)
        ms = check_milestones(s)
        assert "first_microbes" in ms

    def test_milestones_not_repeated(self) -> None:
        s = EcologyState(atmosphere_pressure=0.05, soil_fertility=0.10,
                         milestones_achieved=["first_microbes"])
        ms = check_milestones(s)
        assert "first_microbes" not in ms

    def test_radiation_milestone(self) -> None:
        s = EcologyState(radiation_level=0.60)
        ms = check_milestones(s)
        assert "radiation_safe_outdoors" in ms

    def test_atmosphere_thickening(self) -> None:
        s = EcologyState(atmosphere_pressure=0.07)
        ms = check_milestones(s)
        assert "atmosphere_thickening" in ms

    def test_multiple_milestones_at_once(self) -> None:
        s = EcologyState(atmosphere_pressure=0.12, soil_fertility=0.25,
                         water_table=0.25, radiation_level=0.60)
        ms = check_milestones(s)
        assert len(ms) >= 3  # microbes, lichen, atmosphere, water, radiation


# ---------------------------------------------------------------------------
# Production modifiers
# ---------------------------------------------------------------------------

class TestProductionModifiers:
    def test_default_state_near_baseline(self) -> None:
        s = EcologyState()
        mods = compute_ecology_production_modifiers(s)
        # At default, soil is low → food bonus small
        assert 0.9 <= mods["food_production_mult"] <= 1.1
        assert 0.8 <= mods["power_production_mult"] <= 1.1

    def test_good_soil_boosts_food(self) -> None:
        s = EcologyState(soil_fertility=0.25)
        mods = compute_ecology_production_modifiers(s)
        assert mods["food_production_mult"] > 1.1

    def test_high_water_table_boosts_water(self) -> None:
        s = EcologyState(water_table=0.25)
        mods = compute_ecology_production_modifiers(s)
        assert mods["water_production_mult"] > 1.1

    def test_biodiversity_boosts_medicine(self) -> None:
        s = EcologyState(atmosphere_pressure=0.08, soil_fertility=0.15)
        assert s.biodiversity > 0
        mods = compute_ecology_production_modifiers(s)
        assert mods["medicine_production_mult"] > 1.0

    def test_no_biodiversity_no_medicine_bonus(self) -> None:
        s = EcologyState()
        mods = compute_ecology_production_modifiers(s)
        assert mods["medicine_production_mult"] == 1.0

    def test_all_modifiers_bounded(self) -> None:
        for _ in range(100):
            s = EcologyState(
                atmosphere_pressure=random.random(),
                soil_fertility=random.random(),
                water_table=random.random(),
                radiation_level=random.random(),
            )
            mods = compute_ecology_production_modifiers(s)
            for v in mods.values():
                assert 0.5 <= v <= 2.0, f"modifier out of bounds: {v}"


# ---------------------------------------------------------------------------
# tick_ecology
# ---------------------------------------------------------------------------

class TestTickEcology:
    def _make_actions(self, n_terraform: int = 0, n_farm: int = 0,
                      n_explore: int = 0, n_research: int = 0,
                      n_rest: int = 0) -> dict[str, str]:
        actions: dict[str, str] = {}
        idx = 0
        for _ in range(n_terraform):
            actions[f"c-{idx}"] = "terraform"
            idx += 1
        for _ in range(n_farm):
            actions[f"c-{idx}"] = "farm"
            idx += 1
        for _ in range(n_explore):
            actions[f"c-{idx}"] = "explore"
            idx += 1
        for _ in range(n_research):
            actions[f"c-{idx}"] = "research"
            idx += 1
        for _ in range(n_rest):
            actions[f"c-{idx}"] = "rest"
            idx += 1
        return actions

    def test_terraform_increases_atmosphere(self) -> None:
        s = EcologyState()
        rng = random.Random(42)
        tick_ecology(s, self._make_actions(n_terraform=3), 10, [], 1, rng)
        assert s.atmosphere_pressure > 0.01

    def test_farm_increases_soil(self) -> None:
        s = EcologyState()
        rng = random.Random(42)
        tick_ecology(s, self._make_actions(n_farm=3), 10, [], 1, rng)
        assert s.soil_fertility > 0.02

    def test_explore_increases_water(self) -> None:
        # With minimal population, exploration gains outweigh extraction
        s = EcologyState()
        rng = random.Random(42)
        tick_ecology(s, self._make_actions(n_explore=5), 1, [], 1, rng)
        assert s.water_table > 0.10

    def test_water_depletes_with_large_population(self) -> None:
        s = EcologyState()
        rng = random.Random(42)
        tick_ecology(s, self._make_actions(n_rest=30), 30, [], 1, rng)
        assert s.water_table < 0.10

    def test_water_recycler_reduces_depletion(self) -> None:
        s1 = EcologyState()
        s2 = EcologyState()
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        tick_ecology(s1, self._make_actions(n_rest=20), 20, [], 1, rng1)
        tick_ecology(s2, self._make_actions(n_rest=20), 20, ["water_recycler"], 1, rng2)
        # With recycler, water table should be higher (less depletion)
        assert s2.water_table > s1.water_table

    def test_greenhouse_boosts_soil(self) -> None:
        s1 = EcologyState()
        s2 = EcologyState()
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        tick_ecology(s1, self._make_actions(n_farm=2), 10, [], 1, rng1)
        tick_ecology(s2, self._make_actions(n_farm=2), 10, ["greenhouse_dome"], 1, rng2)
        assert s2.soil_fertility > s1.soil_fertility

    def test_radiation_decreases_over_time(self) -> None:
        s = EcologyState(atmosphere_pressure=0.05)
        rng = random.Random(42)
        for yr in range(1, 21):
            tick_ecology(s, self._make_actions(n_research=2), 10, [], yr, rng)
        assert s.radiation_level < 0.90

    def test_all_values_clamped(self) -> None:
        """Property: all ecology values stay in [0, 1] over 100 ticks."""
        s = EcologyState()
        rng = random.Random(99)
        for yr in range(1, 101):
            actions = self._make_actions(n_terraform=3, n_farm=3,
                                         n_explore=2, n_research=2)
            tick_ecology(s, actions, 10, ["greenhouse_dome", "water_recycler"], yr, rng)
            assert 0.0 <= s.atmosphere_pressure <= 1.0
            assert 0.0 <= s.soil_fertility <= 1.0
            assert 0.0 <= s.water_table <= 1.0
            assert 0.0 <= s.radiation_level <= 1.0

    def test_milestones_accumulate(self) -> None:
        s = EcologyState()
        rng = random.Random(42)
        for yr in range(1, 101):
            actions = self._make_actions(n_terraform=4, n_farm=4, n_explore=1, n_research=1)
            result = tick_ecology(s, actions, 10, ["greenhouse_dome"], yr, rng)
        assert len(s.milestones_achieved) > 0

    def test_result_contains_resource_modifiers(self) -> None:
        s = EcologyState()
        rng = random.Random(42)
        result = tick_ecology(s, self._make_actions(n_terraform=2), 10, [], 1, rng)
        assert "food_production_mult" in result.resource_modifiers

    def test_tick_result_to_dict(self) -> None:
        s = EcologyState()
        rng = random.Random(42)
        result = tick_ecology(s, self._make_actions(n_terraform=1), 5, [], 1, rng)
        d = result.to_dict()
        assert "state_before" in d
        assert "state_after" in d
        assert "habitability" in d

    def test_diminishing_returns_on_atmosphere(self) -> None:
        """Higher atmosphere → slower gains (diminishing returns)."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        s_low = EcologyState(atmosphere_pressure=0.02)
        s_high = EcologyState(atmosphere_pressure=0.15)
        actions = self._make_actions(n_terraform=3)
        tick_ecology(s_low, actions, 10, [], 1, rng1)
        tick_ecology(s_high, actions, 10, [], 1, rng2)
        gain_low = s_low.atmosphere_pressure - 0.02
        gain_high = s_high.atmosphere_pressure - 0.15
        assert gain_low > gain_high  # diminishing returns


# ---------------------------------------------------------------------------
# Integration: 10-year smoke test
# ---------------------------------------------------------------------------

class TestEcologySmoke:
    def test_ten_year_run(self) -> None:
        """Ecology should be MORE alive after 10 years of terraforming."""
        s = EcologyState()
        rng = random.Random(42)
        initial_hab = s.habitability
        for yr in range(1, 11):
            actions = {f"c-{i}": "terraform" for i in range(5)}
            actions.update({f"c-{i}": "farm" for i in range(5, 8)})
            actions.update({f"c-{i}": "explore" for i in range(8, 10)})
            tick_ecology(s, actions, 10, [], yr, rng)
        assert s.habitability > initial_hab
        assert s.atmosphere_pressure > 0.01
        assert s.soil_fertility > 0.02
