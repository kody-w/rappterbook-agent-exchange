"""Tests for the ecology organ (engine v10.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.ecology import (
    EcologyState, EcologyYearContext, EcologyTickResult,
    BIOME_NAMES, BIOME_UP_THRESHOLDS, BIOME_DOWN_THRESHOLDS,
    MAX_FOOD_BONUS, MAX_AIR_BONUS, MAX_WATER_BONUS,
    MAX_NATURE_STRESS_REDUCTION,
    PRESSURE_GATE_KPA, TEMPERATURE_GATE_C, PERCHLORATE_GATE,
    compute_atmosphere_score, compute_soil_score, compute_flora_score,
    compute_biosphere_index, outdoor_habitable, has_greenhouse_tech,
    compute_ecology_bonuses, compute_resource_modifiers,
    compute_nature_stress_reduction,
    update_biome_level, tick_ecology, compute_biome_level,
)
from src.mars100.engine import Mars100Engine


# ---------- helpers ----------

def make_eco(**kw) -> EcologyState:
    """Create an EcologyState with overrides."""
    return EcologyState(**kw)


def make_ctx(**kw) -> EcologyYearContext:
    """Create a context with sensible defaults."""
    defaults = dict(year=1, terraform_count=0, farm_count=0,
                    research_count=0, population=10,
                    infrastructure_completed=[])
    defaults.update(kw)
    return EcologyYearContext(**defaults)


# ---------- EcologyState dataclass ----------

class TestEcologyState:
    def test_defaults(self):
        eco = EcologyState()
        assert eco.pressure_kpa == 0.6
        assert eco.o2_kpa == 0.001
        assert eco.temperature_c == -60.0
        assert eco.perchlorate == 0.8
        assert eco.biome_level == 0
        assert eco.biosphere_index == 0.0

    def test_to_dict_roundtrip(self):
        eco = make_eco(pressure_kpa=5.0, o2_kpa=2.0, temperature_c=-20.0,
                       perchlorate=0.3, soil_organic=0.5, soil_moisture=0.4,
                       greenhouse_crops=0.3, outdoor_plants=0.1,
                       biome_level=2, biosphere_index=0.35)
        d = eco.to_dict()
        eco2 = EcologyState.from_dict(d)
        assert abs(eco2.pressure_kpa - eco.pressure_kpa) < 0.001
        assert eco2.biome_level == eco.biome_level

    def test_to_dict_has_biome_name(self):
        eco = make_eco(biome_level=3)
        assert eco.to_dict()["biome_name"] == "grassland"

    def test_clamp_enforces_bounds(self):
        eco = make_eco(pressure_kpa=-1.0, o2_kpa=999.0, temperature_c=100.0,
                       perchlorate=2.0, soil_organic=-0.5, biome_level=99)
        eco.clamp()
        assert eco.pressure_kpa == 0.0
        assert eco.o2_kpa == 0.0  # clamped to pressure_kpa
        assert eco.temperature_c == 30.0
        assert eco.perchlorate == 1.0
        assert eco.soil_organic == 0.0
        assert eco.biome_level == len(BIOME_NAMES) - 1

    def test_health_alias(self):
        eco = make_eco(biosphere_index=0.42)
        assert eco.health() == 0.42

    def test_from_dict_defaults(self):
        eco = EcologyState.from_dict({})
        assert eco.pressure_kpa == 0.6
        assert eco.biome_unlocks == []


# ---------- scoring functions ----------

class TestScoring:
    def test_atmosphere_score_zero_at_start(self):
        eco = EcologyState()
        score = compute_atmosphere_score(eco)
        assert 0.0 <= score <= 1.0
        assert score < 0.1  # Mars baseline is very low

    def test_atmosphere_score_increases_with_pressure(self):
        lo = compute_atmosphere_score(make_eco(pressure_kpa=1.0))
        hi = compute_atmosphere_score(make_eco(pressure_kpa=25.0))
        assert hi > lo

    def test_soil_score_improves_with_less_perchlorate(self):
        bad = compute_soil_score(make_eco(perchlorate=0.9))
        good = compute_soil_score(make_eco(perchlorate=0.1))
        assert good > bad

    def test_flora_score_depends_on_crops(self):
        none_ = compute_flora_score(EcologyState())
        crops = compute_flora_score(make_eco(greenhouse_crops=0.5))
        assert crops > none_
        assert 0.0 <= crops <= 1.0

    def test_biosphere_index_bounded(self):
        for pressure in [0.6, 10.0, 50.0]:
            for organic in [0.0, 0.5, 1.0]:
                eco = make_eco(pressure_kpa=pressure, soil_organic=organic)
                idx = compute_biosphere_index(eco)
                assert 0.0 <= idx <= 1.0

    def test_biosphere_index_increases_with_terraforming(self):
        barren = compute_biosphere_index(EcologyState())
        green = compute_biosphere_index(make_eco(
            pressure_kpa=20.0, o2_kpa=5.0, temperature_c=0.0,
            perchlorate=0.1, soil_organic=0.8, greenhouse_crops=0.6))
        assert green > barren


# ---------- outdoor_habitable ----------

class TestOutdoorHabitable:
    def test_mars_baseline_not_habitable(self):
        assert not outdoor_habitable(EcologyState())

    def test_habitable_when_conditions_met(self):
        eco = make_eco(pressure_kpa=6.0, temperature_c=-30.0, perchlorate=0.2)
        assert outdoor_habitable(eco)

    def test_not_habitable_low_pressure(self):
        eco = make_eco(pressure_kpa=4.0, temperature_c=-30.0, perchlorate=0.2)
        assert not outdoor_habitable(eco)

    def test_not_habitable_too_cold(self):
        eco = make_eco(pressure_kpa=6.0, temperature_c=-50.0, perchlorate=0.2)
        assert not outdoor_habitable(eco)

    def test_not_habitable_high_perchlorate(self):
        eco = make_eco(pressure_kpa=6.0, temperature_c=-30.0, perchlorate=0.5)
        assert not outdoor_habitable(eco)


# ---------- greenhouse tech ----------

class TestGreenhouseTech:
    def test_no_tech(self):
        assert not has_greenhouse_tech([])

    def test_has_greenhouse(self):
        assert has_greenhouse_tech(["basic_greenhouse"])

    def test_case_insensitive(self):
        assert has_greenhouse_tech(["Advanced_Greenhouse_Mk2"])


# ---------- resource integration ----------

class TestResourceIntegration:
    def test_bonuses_zero_at_start(self):
        bonuses = compute_ecology_bonuses(EcologyState())
        assert bonuses["food"] == 0.0
        assert bonuses["water"] >= 0.0

    def test_bonuses_increase_with_crops(self):
        eco = make_eco(greenhouse_crops=0.5, outdoor_plants=0.2)
        bonuses = compute_ecology_bonuses(eco)
        assert bonuses["food"] > 0.0
        assert bonuses["food"] <= MAX_FOOD_BONUS * 1.5

    def test_modifiers_at_zero_index(self):
        mods = compute_resource_modifiers(EcologyState())
        assert mods["food_spoilage_mult"] == 1.0

    def test_modifiers_improve_with_index(self):
        eco = make_eco(biosphere_index=0.5)
        mods = compute_resource_modifiers(eco)
        assert mods["food_spoilage_mult"] < 1.0
        assert mods["food_spoilage_mult"] >= 0.5

    def test_stress_reduction_zero_at_start(self):
        assert compute_nature_stress_reduction(EcologyState()) == 0.0

    def test_stress_reduction_scales_with_index(self):
        eco = make_eco(biosphere_index=1.0)
        assert compute_nature_stress_reduction(eco) == MAX_NATURE_STRESS_REDUCTION


# ---------- biome transitions ----------

class TestBiomeTransitions:
    def test_no_transition_at_start(self):
        eco = EcologyState()
        assert update_biome_level(eco) is None

    def test_promotion(self):
        eco = make_eco(biosphere_index=0.10, biome_level=0)
        result = update_biome_level(eco)
        assert result is not None
        assert result["direction"] == "up"
        assert result["to_name"] == "lichen"
        assert result["first_time"] is True
        assert eco.biome_level == 1

    def test_demotion(self):
        eco = make_eco(biosphere_index=0.03, biome_level=1)
        result = update_biome_level(eco)
        assert result is not None
        assert result["direction"] == "down"
        assert eco.biome_level == 0

    def test_hysteresis_prevents_oscillation(self):
        # At threshold between levels — should NOT demote
        eco = make_eco(biosphere_index=0.06, biome_level=1)
        assert update_biome_level(eco) is None

    def test_multiple_promotions(self):
        eco = make_eco(biosphere_index=0.55, biome_level=0)
        result = update_biome_level(eco)
        assert result is not None
        assert eco.biome_level >= 4  # should jump to shrubland

    def test_compute_biome_level_standalone(self):
        assert compute_biome_level(0.0, 0) == 0
        assert compute_biome_level(0.10, 0) == 1
        assert compute_biome_level(0.75, 0) == 5

    def test_unlocks_tracked(self):
        eco = make_eco(biosphere_index=0.10, biome_level=0, biome_unlocks=[])
        update_biome_level(eco)
        assert 1 in eco.biome_unlocks
        # Second time at same level: not first_time
        eco.biome_level = 0
        eco.biosphere_index = 0.10
        r2 = update_biome_level(eco)
        assert r2["first_time"] is False


# ---------- tick_ecology ----------

class TestTickEcology:
    def test_basic_tick(self):
        eco = EcologyState()
        ctx = make_ctx(terraform_count=2, farm_count=3)
        rng = random.Random(42)
        result = tick_ecology(eco, ctx, rng)
        assert isinstance(result, EcologyTickResult)
        assert result.resource_bonuses is not None

    def test_terraforming_increases_pressure(self):
        eco = EcologyState()
        initial = eco.pressure_kpa
        tick_ecology(eco, make_ctx(terraform_count=5), random.Random(42))
        assert eco.pressure_kpa > initial

    def test_farming_reduces_perchlorate(self):
        eco = EcologyState()
        initial = eco.perchlorate
        tick_ecology(eco, make_ctx(farm_count=5), random.Random(42))
        assert eco.perchlorate < initial

    def test_greenhouse_growth_with_farming(self):
        eco = EcologyState()
        tick_ecology(eco, make_ctx(farm_count=3), random.Random(42))
        assert eco.greenhouse_crops > 0.0

    def test_greenhouse_decay_without_farming(self):
        eco = make_eco(greenhouse_crops=0.1)
        tick_ecology(eco, make_ctx(farm_count=0), random.Random(42))
        assert eco.greenhouse_crops < 0.1

    def test_outdoor_plants_die_without_habitable(self):
        eco = make_eco(outdoor_plants=0.1)  # Mars baseline: not habitable
        tick_ecology(eco, make_ctx(), random.Random(42))
        assert eco.outdoor_plants < 0.1

    def test_outdoor_plants_grow_when_habitable(self):
        eco = make_eco(pressure_kpa=10.0, temperature_c=-20.0,
                       perchlorate=0.1, greenhouse_crops=0.3,
                       soil_organic=0.5, soil_moisture=0.3)
        tick_ecology(eco, make_ctx(farm_count=2), random.Random(42))
        assert eco.outdoor_plants > 0.0

    def test_biosphere_index_updated(self):
        eco = EcologyState()
        tick_ecology(eco, make_ctx(terraform_count=3, farm_count=3), random.Random(42))
        assert eco.biosphere_index > 0.0

    def test_clamp_applied(self):
        eco = EcologyState()
        for _ in range(200):
            tick_ecology(eco, make_ctx(terraform_count=10, farm_count=10), random.Random(42))
        assert eco.temperature_c <= 30.0
        assert eco.perchlorate >= 0.0
        assert eco.biosphere_index <= 1.0

    def test_deterministic_with_same_seed(self):
        eco1 = EcologyState()
        eco2 = EcologyState()
        ctx = make_ctx(terraform_count=2, farm_count=2)
        tick_ecology(eco1, ctx, random.Random(99))
        tick_ecology(eco2, ctx, random.Random(99))
        assert eco1.to_dict() == eco2.to_dict()

    def test_tick_result_to_dict(self):
        eco = EcologyState()
        result = tick_ecology(eco, make_ctx(), random.Random(42))
        d = result.to_dict()
        assert "resource_bonuses" in d
        assert "nature_stress_reduction" in d

    def test_one_year_lag_bonuses(self):
        """Bonuses should reflect pre-mutation state."""
        eco = make_eco(greenhouse_crops=0.5)
        ctx = make_ctx(farm_count=0)
        result = tick_ecology(eco, ctx, random.Random(42))
        # Bonuses computed from 0.5 crops, not post-mutation value
        assert result.resource_bonuses["food"] > 0.0


# ---------- property-based: physical bounds over many seeds ----------

class TestPropertyBased:
    @pytest.mark.parametrize("seed", range(10))
    def test_50_years_all_in_bounds(self, seed):
        """Run 50 years with different seeds — all values stay physical."""
        eco = EcologyState()
        rng = random.Random(seed)
        for year in range(50):
            ctx = EcologyYearContext(
                year=year,
                terraform_count=rng.randint(0, 5),
                farm_count=rng.randint(0, 5),
                research_count=rng.randint(0, 3),
                population=rng.randint(5, 15),
                infrastructure_completed=(
                    ["basic_greenhouse"] if year > 10 else []),
            )
            tick_ecology(eco, ctx, rng)
            assert 0.0 <= eco.pressure_kpa
            assert 0.0 <= eco.o2_kpa <= eco.pressure_kpa
            assert -80.0 <= eco.temperature_c <= 30.0
            assert 0.0 <= eco.perchlorate <= 1.0
            assert 0.0 <= eco.soil_organic <= 1.0
            assert 0.0 <= eco.soil_moisture <= 1.0
            assert 0.0 <= eco.greenhouse_crops <= 1.0
            assert 0.0 <= eco.outdoor_plants <= 1.0
            assert 0.0 <= eco.biosphere_index <= 1.0
            assert 0 <= eco.biome_level < len(BIOME_NAMES)


# ---------- engine integration ----------

class TestEngineIntegration:
    def test_engine_runs_with_ecology(self):
        """Smoke test: engine completes 10 years with ecology."""
        e = Mars100Engine(seed=42, total_years=10)
        r = e.run()
        d = r.to_dict()
        assert d["_meta"]["version"] == "10.0"
        assert len(r.years) == 10
        assert "biome_name" in d["final_ecology"]

    def test_ecology_in_year_result(self):
        e = Mars100Engine(seed=42, total_years=3)
        r = e.run()
        for y in r.years:
            assert "resource_bonuses" in y.ecology
            assert "nature_stress_reduction" in y.ecology

    def test_final_ecology_populated(self):
        e = Mars100Engine(seed=42, total_years=5)
        r = e.run()
        fe = r.to_dict()["final_ecology"]
        assert "pressure_kpa" in fe
        assert "biosphere_index" in fe
        assert "biome_name" in fe

    @pytest.mark.parametrize("seed", [42, 99, 1234])
    def test_deterministic_across_seeds(self, seed):
        """Same seed -> same ecology outcome."""
        e1 = Mars100Engine(seed=seed, total_years=5)
        e2 = Mars100Engine(seed=seed, total_years=5)
        r1 = e1.run()
        r2 = e2.run()
        assert r1.to_dict()["final_ecology"] == r2.to_dict()["final_ecology"]

    def test_ecology_evolves_over_time(self):
        """Ecology should change over 20 years."""
        e = Mars100Engine(seed=42, total_years=20)
        r = e.run()
        y1 = r.years[0].ecology
        y20 = r.years[-1].ecology
        # At least one field should differ
        assert y1 != y20
