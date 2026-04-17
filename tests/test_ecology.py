"""Tests for the ecology organ (engine v10.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.ecology import (
    Atmosphere, SoilState, WaterCycle, EcologyState,
    EcologyTickResult, BIOME_NAMES, BIOME_THRESHOLDS, BIOME_GATES,
    BIOME_FOOD_SPOILAGE_MULT, BIOME_AIR_MAINTENANCE_MULT,
    BIOME_WATER_MAINTENANCE_MULT,
    compute_biome_level, compute_ecology_modifiers,
    compute_ecology_psych_pressure, tick_ecology,
    _check_biome_gate,
    TERRAFORM_SCORE_PER_ACTION, FARM_SCORE_CONTRIBUTION,
    RESEARCH_SCORE_CONTRIBUTION, EVENT_ECOLOGY_DAMAGE,
)


class TestAtmosphere:
    def test_defaults(self):
        a = Atmosphere()
        assert a.o2_fraction == pytest.approx(0.001)
        assert a.co2_fraction == pytest.approx(0.96)

    def test_roundtrip(self):
        a = Atmosphere(o2_fraction=0.05, co2_fraction=0.90,
                       temperature_c=-40.0, pressure_kpa=1.2)
        a2 = Atmosphere.from_dict(a.to_dict())
        assert a2.o2_fraction == pytest.approx(a.o2_fraction)
        assert a2.co2_fraction == pytest.approx(a.co2_fraction)

    def test_from_empty_dict(self):
        a = Atmosphere.from_dict({})
        assert a.o2_fraction == pytest.approx(0.001)


class TestSoilState:
    def test_defaults(self):
        s = SoilState()
        assert s.fertility == pytest.approx(0.05)
        assert s.perchlorate_index == pytest.approx(0.60)

    def test_roundtrip(self):
        s = SoilState(fertility=0.3, perchlorate_index=0.2, water_content=0.15)
        s2 = SoilState.from_dict(s.to_dict())
        assert s2.fertility == pytest.approx(s.fertility)

    def test_from_empty_dict(self):
        s = SoilState.from_dict({})
        assert s.fertility == pytest.approx(0.05)


class TestWaterCycle:
    def test_defaults(self):
        w = WaterCycle()
        assert w.ice_reserves == pytest.approx(0.80)

    def test_roundtrip(self):
        w = WaterCycle(ice_reserves=0.5, liquid_available=0.3,
                       recapture_efficiency=0.7)
        w2 = WaterCycle.from_dict(w.to_dict())
        assert w2.ice_reserves == pytest.approx(w.ice_reserves)


class TestEcologyState:
    def test_defaults(self):
        e = EcologyState()
        assert e.biome_level == 0
        assert e.biodiversity == pytest.approx(0.0)
        assert e.terraforming_score == pytest.approx(0.0)

    def test_roundtrip(self):
        e = EcologyState()
        e.biome_level = 2
        e.terraforming_score = 0.25
        e.biodiversity = 0.3
        e2 = EcologyState.from_dict(e.to_dict())
        assert e2.biome_level == 2
        assert e2.terraforming_score == pytest.approx(0.25)

    def test_clamp_within_bounds(self):
        e = EcologyState()
        e.atmosphere.o2_fraction = 1.5
        e.soil.fertility = -0.1
        e.water.ice_reserves = 2.0
        e.biodiversity = -0.5
        e.terraforming_score = 1.5
        e.clamp()
        assert e.atmosphere.o2_fraction == pytest.approx(1.0)
        assert e.soil.fertility == pytest.approx(0.0)
        assert e.water.ice_reserves == pytest.approx(1.0)
        assert e.biodiversity == pytest.approx(0.0)
        assert e.terraforming_score == pytest.approx(1.0)

    def test_to_dict_contains_biome_name(self):
        e = EcologyState()
        e.biome_level = 3
        assert e.to_dict()["biome_name"] == "moss"


class TestBiomeGate:
    def test_barren_always_passes(self):
        assert _check_biome_gate(0, EcologyState(), [])

    def test_microbial_passes_by_default(self):
        assert _check_biome_gate(1, EcologyState(), [])

    def test_lichen_requires_soil_fertility(self):
        e = EcologyState()
        e.soil.fertility = 0.05
        assert not _check_biome_gate(2, e, [])
        e.soil.fertility = 0.15
        e.soil.perchlorate_index = 0.40
        assert _check_biome_gate(2, e, [])

    def test_greenhouse_requires_tech(self):
        e = EcologyState()
        e.soil.fertility = 0.35
        e.soil.perchlorate_index = 0.20
        assert not _check_biome_gate(4, e, [])
        assert _check_biome_gate(4, e, ["greenhouse_dome"])

    def test_outdoor_requires_pressure_and_o2(self):
        e = EcologyState()
        e.soil.fertility = 0.50
        e.soil.perchlorate_index = 0.10
        e.atmosphere.o2_fraction = 0.01
        e.atmosphere.pressure_kpa = 0.5
        assert not _check_biome_gate(5, e, [])
        e.atmosphere.o2_fraction = 0.06
        e.atmosphere.pressure_kpa = 1.5
        assert _check_biome_gate(5, e, [])

    def test_out_of_range_returns_false(self):
        assert not _check_biome_gate(99, EcologyState(), [])


class TestComputeBiomeLevel:
    def test_barren_at_zero(self):
        assert compute_biome_level(EcologyState(), []) == 0

    def test_microbial_at_threshold(self):
        e = EcologyState()
        e.terraforming_score = 0.09
        assert compute_biome_level(e, []) == 1

    def test_cannot_skip_levels(self):
        e = EcologyState()
        e.terraforming_score = 0.90
        e.soil.fertility = 0.50
        e.soil.perchlorate_index = 0.10
        e.atmosphere.o2_fraction = 0.10
        e.atmosphere.pressure_kpa = 2.0
        level = compute_biome_level(e, [])
        assert level == 3  # stuck at moss (no greenhouse_dome)

    def test_full_progression_with_techs(self):
        e = EcologyState()
        e.terraforming_score = 0.90
        e.soil.fertility = 0.50
        e.soil.perchlorate_index = 0.10
        e.atmosphere.o2_fraction = 0.10
        e.atmosphere.pressure_kpa = 2.0
        level = compute_biome_level(e, ["greenhouse_dome"])
        assert level == 5


class TestComputeEcologyModifiers:
    def test_barren_no_bonus(self):
        mods = compute_ecology_modifiers(EcologyState())
        assert mods["food_spoilage_mult"] == pytest.approx(1.0)

    def test_greenhouse_crops_bonus(self):
        e = EcologyState()
        e.biome_level = 4
        mods = compute_ecology_modifiers(e)
        assert mods["food_spoilage_mult"] < 1.0
        assert mods["air_maintenance_mult"] < 1.0

    def test_outdoor_crops_best_bonus(self):
        e = EcologyState()
        e.biome_level = 5
        mods = compute_ecology_modifiers(e)
        assert mods["food_spoilage_mult"] < 0.75

    def test_keys_are_multipliers(self):
        for key in compute_ecology_modifiers(EcologyState()):
            assert key.endswith("_mult")


class TestComputeEcologyPsychPressure:
    def test_barren_high_perchlorate(self):
        e = EcologyState()
        p = compute_ecology_psych_pressure(e)
        assert p["stress"] > 0
        assert p["purpose"] == pytest.approx(0.0)

    def test_high_biome_low_perchlorate(self):
        e = EcologyState()
        e.biome_level = 4
        e.soil.perchlorate_index = 0.1
        p = compute_ecology_psych_pressure(e)
        assert p["stress"] < 0
        assert p["purpose"] > 0


class TestTickEcology:
    def _make_state(self):
        return EcologyState()

    def _tick(self, state=None, **kwargs):
        if state is None:
            state = self._make_state()
        defaults = dict(
            year=1, terraforming_count=2, avg_terraform_skill=0.5,
            farming_count=3, research_count=1, event_damage=0.0,
            infra_completed=[], rng=random.Random(42),
        )
        defaults.update(kwargs)
        return tick_ecology(state, **defaults)

    def test_basic_tick(self):
        result = self._tick()
        assert isinstance(result, EcologyTickResult)

    def test_terraforming_increases_score(self):
        state = self._make_state()
        score_before = state.terraforming_score
        self._tick(state, terraforming_count=5, avg_terraform_skill=0.8)
        assert state.terraforming_score > score_before

    def test_no_actions_minimal_change(self):
        state = self._make_state()
        self._tick(state, terraforming_count=0, farming_count=0,
                   research_count=0)
        assert abs(state.terraforming_score) < 0.01

    def test_o2_co2_conservation(self):
        state = self._make_state()
        o2_before = state.atmosphere.o2_fraction
        co2_before = state.atmosphere.co2_fraction
        self._tick(state, terraforming_count=5)
        o2_delta = state.atmosphere.o2_fraction - o2_before
        co2_delta = state.atmosphere.co2_fraction - co2_before
        assert abs(o2_delta + co2_delta) < 0.001

    def test_soil_improves_with_farming(self):
        state = self._make_state()
        fert_before = state.soil.fertility
        self._tick(state, farming_count=5)
        assert state.soil.fertility > fert_before

    def test_perchlorate_decreases(self):
        state = self._make_state()
        perc_before = state.soil.perchlorate_index
        self._tick(state)
        assert state.soil.perchlorate_index < perc_before

    def test_ice_drains_to_liquid(self):
        state = self._make_state()
        ice_before = state.water.ice_reserves
        self._tick(state)
        assert state.water.ice_reserves < ice_before

    def test_water_recycler_improves_recapture(self):
        state = self._make_state()
        eff_before = state.water.recapture_efficiency
        self._tick(state, infra_completed=["water_recycler"])
        assert state.water.recapture_efficiency > eff_before

    def test_event_damage_reduces_score(self):
        state_a = self._make_state()
        state_b = self._make_state()
        self._tick(state_a, event_damage=0.0, rng=random.Random(42))
        self._tick(state_b, event_damage=5.0, rng=random.Random(42))
        assert state_b.terraforming_score < state_a.terraforming_score

    def test_shelter_reduces_event_damage(self):
        state_a = self._make_state()
        state_b = self._make_state()
        self._tick(state_a, event_damage=3.0, rng=random.Random(42))
        self._tick(state_b, event_damage=3.0,
                   infra_completed=["shelter_reinforcement"],
                   rng=random.Random(42))
        assert state_b.terraforming_score > state_a.terraforming_score

    def test_biome_transition_detected(self):
        state = self._make_state()
        state.terraforming_score = BIOME_THRESHOLDS[1] - 0.001
        result = self._tick(state, terraforming_count=5,
                            avg_terraform_skill=0.8)
        if state.terraforming_score >= BIOME_THRESHOLDS[1]:
            assert result.biome_changed
            assert result.tipping_point is not None

    def test_all_values_bounded_after_50_ticks(self):
        state = self._make_state()
        rng = random.Random(77)
        for yr in range(1, 51):
            tick_ecology(state, yr,
                         terraforming_count=rng.randint(0, 5),
                         avg_terraform_skill=rng.random(),
                         farming_count=rng.randint(0, 5),
                         research_count=rng.randint(0, 3),
                         event_damage=rng.random() * 2,
                         infra_completed=[], rng=rng)
        assert 0.0 <= state.atmosphere.o2_fraction <= 1.0
        assert 0.0 <= state.atmosphere.co2_fraction <= 1.0
        assert 0.0 <= state.soil.fertility <= 1.0
        assert 0.0 <= state.soil.perchlorate_index <= 1.0
        assert 0.0 <= state.water.ice_reserves <= 1.0
        assert 0.0 <= state.water.liquid_available <= 1.0
        assert 0.0 <= state.biodiversity <= 1.0
        assert 0.0 <= state.terraforming_score <= 1.0

    def test_determinism(self):
        state_a = self._make_state()
        state_b = self._make_state()
        tick_ecology(state_a, 1, 3, 0.6, 2, 1, 0.5, [], random.Random(42))
        tick_ecology(state_b, 1, 3, 0.6, 2, 1, 0.5, [], random.Random(42))
        assert state_a.atmosphere.o2_fraction == pytest.approx(
            state_b.atmosphere.o2_fraction)

    def test_infra_boost(self):
        state_a = self._make_state()
        state_b = self._make_state()
        self._tick(state_a, infra_completed=[], rng=random.Random(42))
        self._tick(state_b,
                   infra_completed=["greenhouse_dome", "water_recycler", "air_recycler"],
                   rng=random.Random(42))
        assert state_b.terraforming_score > state_a.terraforming_score


class TestEcologyTickResult:
    def test_to_dict_basic(self):
        r = EcologyTickResult()
        d = r.to_dict()
        assert d["biome_changed"] is False
        assert "tipping_point" not in d

    def test_to_dict_with_tipping(self):
        r = EcologyTickResult(biome_changed=True, old_biome=0, new_biome=1,
                              tipping_point="Biome advanced")
        d = r.to_dict()
        assert "tipping_point" in d


class TestEcologyEngineIntegration:
    def test_10_year_engine_run(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) > 0
        for yr in result.years:
            d = yr.to_dict()
            assert "ecology" in d

    def test_ecology_progresses_over_50_years(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        eco = engine.ecology
        assert eco.terraforming_score > 0.0
        assert eco.soil.fertility > 0.05

    def test_ecology_bounded_100_years(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=99, total_years=100)
        result = engine.run()
        eco = engine.ecology
        assert 0.0 <= eco.atmosphere.o2_fraction <= 1.0
        assert 0.0 <= eco.atmosphere.co2_fraction <= 1.0
        assert 0.0 <= eco.soil.fertility <= 1.0
        assert 0.0 <= eco.biodiversity <= 1.0

    def test_ecology_determinism(self):
        from src.mars100.engine import Mars100Engine
        a = Mars100Engine(seed=42, total_years=20)
        b = Mars100Engine(seed=42, total_years=20)
        a.run()
        b.run()
        assert a.ecology.terraforming_score == pytest.approx(
            b.ecology.terraforming_score)
        assert a.ecology.biome_level == b.ecology.biome_level


class TestConstants:
    def test_biome_thresholds_monotonic(self):
        for i in range(1, len(BIOME_THRESHOLDS)):
            assert BIOME_THRESHOLDS[i] > BIOME_THRESHOLDS[i - 1]

    def test_biome_names_match_thresholds(self):
        assert len(BIOME_NAMES) == len(BIOME_THRESHOLDS)

    def test_biome_gates_match_names(self):
        assert len(BIOME_GATES) == len(BIOME_NAMES)

    def test_modifier_tuples_match_biomes(self):
        assert len(BIOME_FOOD_SPOILAGE_MULT) == len(BIOME_NAMES)
        assert len(BIOME_AIR_MAINTENANCE_MULT) == len(BIOME_NAMES)
        assert len(BIOME_WATER_MAINTENANCE_MULT) == len(BIOME_NAMES)

    def test_modifiers_monotonically_decrease(self):
        for tup in (BIOME_FOOD_SPOILAGE_MULT, BIOME_AIR_MAINTENANCE_MULT,
                     BIOME_WATER_MAINTENANCE_MULT):
            for i in range(1, len(tup)):
                assert tup[i] <= tup[i - 1]

    def test_modifiers_positive(self):
        for tup in (BIOME_FOOD_SPOILAGE_MULT, BIOME_AIR_MAINTENANCE_MULT,
                     BIOME_WATER_MAINTENANCE_MULT):
            for val in tup:
                assert val > 0.0
