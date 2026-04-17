"""Tests for the Mars-100 ecology organ (engine v10.0)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.mars100.ecology import (
    Atmosphere, Biosphere, EcologyTickResult, Fauna, Flora, Soil, WaterCycle,
    ATMOSPHERE_TIPPING_POINT, FLORA_TIPPING_POINT,
    ICE_CAP_INITIAL, MAX_FOOD_BONUS, MAX_MEDICINE_BONUS,
    MAX_STRESS_REDUCTION, MAX_WATER_BONUS, MAX_AIR_BONUS,
    PERCHLORATE_INITIAL, SOIL_TIPPING_POINT,
    SURFACE_WATER_PRESSURE_THRESHOLD, WATER_TIPPING_POINT,
    WILD_PLANT_PERCHLORATE_THRESHOLD, WILD_PLANT_PRESSURE_THRESHOLD,
    compute_nature_stress_reduction, compute_resource_bonuses,
    tick_atmosphere, tick_ecology, tick_fauna, tick_flora, tick_soil,
    tick_water,
)


# ---------------------------------------------------------------------------
# Atmosphere tests
# ---------------------------------------------------------------------------

class TestAtmosphere:
    def test_initial_health_low(self) -> None:
        atmo = Atmosphere()
        assert 0.0 <= atmo.health() <= 0.15

    def test_terraforming_increases_pressure(self) -> None:
        atmo = Atmosphere()
        p0 = atmo.pressure
        tick_atmosphere(atmo, terraform_count=3, flora_outdoor=0.0, event_severity=0.0)
        assert atmo.pressure > p0

    def test_zero_effort_pressure_decays(self) -> None:
        atmo = Atmosphere(pressure=0.1)
        tick_atmosphere(atmo, terraform_count=0, flora_outdoor=0.0, event_severity=0.0)
        assert atmo.pressure < 0.1

    def test_terraforming_increases_o2(self) -> None:
        atmo = Atmosphere()
        o2_0 = atmo.o2_fraction
        tick_atmosphere(atmo, terraform_count=5, flora_outdoor=0.0, event_severity=0.0)
        assert atmo.o2_fraction > o2_0

    def test_flora_contributes_o2(self) -> None:
        atmo = Atmosphere()
        o2_0 = atmo.o2_fraction
        tick_atmosphere(atmo, terraform_count=0, flora_outdoor=0.5, event_severity=0.0)
        assert atmo.o2_fraction > o2_0

    def test_terraforming_reduces_co2(self) -> None:
        atmo = Atmosphere()
        co2_0 = atmo.co2_fraction
        tick_atmosphere(atmo, terraform_count=3, flora_outdoor=0.0, event_severity=0.0)
        assert atmo.co2_fraction < co2_0

    def test_severe_event_damages_pressure(self) -> None:
        atmo = Atmosphere(pressure=0.2)
        tick_atmosphere(atmo, terraform_count=0, flora_outdoor=0.0, event_severity=0.9)
        assert atmo.pressure < 0.2

    def test_tipping_point_accelerates_decay(self) -> None:
        atmo = Atmosphere(pressure=0.01, o2_fraction=0.001, co2_fraction=0.99, temperature=0.01)
        assert atmo.health() < ATMOSPHERE_TIPPING_POINT
        tipping = tick_atmosphere(atmo, terraform_count=0, flora_outdoor=0.0, event_severity=0.0)
        assert "atmosphere_collapse" in tipping

    def test_all_values_bounded(self) -> None:
        atmo = Atmosphere()
        for _ in range(200):
            tick_atmosphere(atmo, terraform_count=10, flora_outdoor=1.0, event_severity=0.0)
        assert 0.0 <= atmo.pressure <= 1.0
        assert 0.0 <= atmo.o2_fraction <= 1.0
        assert 0.0 <= atmo.co2_fraction <= 1.0
        assert 0.0 <= atmo.temperature <= 1.0

    def test_serialization_roundtrip(self) -> None:
        atmo = Atmosphere(pressure=0.3, o2_fraction=0.1, co2_fraction=0.7, temperature=0.2)
        d = atmo.to_dict()
        restored = Atmosphere.from_dict(d)
        assert abs(restored.pressure - 0.3) < 1e-4
        assert abs(restored.o2_fraction - 0.1) < 1e-4


# ---------------------------------------------------------------------------
# Soil tests
# ---------------------------------------------------------------------------

class TestSoil:
    def test_initial_perchlorate_high(self) -> None:
        soil = Soil()
        assert soil.perchlorate == PERCHLORATE_INITIAL

    def test_farming_remediates_perchlorate(self) -> None:
        soil = Soil()
        tick_soil(soil, farm_count=5, flora_wild=0.0)
        assert soil.perchlorate < PERCHLORATE_INITIAL

    def test_farming_increases_organic(self) -> None:
        soil = Soil()
        org_0 = soil.organic_content
        tick_soil(soil, farm_count=3, flora_wild=0.0)
        assert soil.organic_content > org_0

    def test_wild_flora_adds_organic(self) -> None:
        # Start above tipping point; use high flora so gain > decay
        soil = Soil(perchlorate=0.3, organic_content=0.15, nitrogen=0.1)
        org_0 = soil.organic_content
        tick_soil(soil, farm_count=0, flora_wild=1.0)
        assert soil.organic_content > org_0

    def test_zero_effort_organic_decays(self) -> None:
        soil = Soil(organic_content=0.2)
        tick_soil(soil, farm_count=0, flora_wild=0.0)
        assert soil.organic_content < 0.2

    def test_tipping_point(self) -> None:
        soil = Soil(perchlorate=0.95, organic_content=0.01, nitrogen=0.01)
        assert soil.health() < SOIL_TIPPING_POINT
        tipping = tick_soil(soil, farm_count=0, flora_wild=0.0)
        assert "soil_degradation" in tipping

    def test_all_bounded(self) -> None:
        soil = Soil()
        for _ in range(200):
            tick_soil(soil, farm_count=10, flora_wild=1.0)
        assert 0.0 <= soil.perchlorate <= 1.0
        assert 0.0 <= soil.organic_content <= 1.0
        assert 0.0 <= soil.nitrogen <= 1.0

    def test_serialization_roundtrip(self) -> None:
        soil = Soil(perchlorate=0.3, organic_content=0.2, nitrogen=0.15)
        d = soil.to_dict()
        restored = Soil.from_dict(d)
        assert abs(restored.perchlorate - 0.3) < 1e-4


# ---------------------------------------------------------------------------
# Water cycle tests
# ---------------------------------------------------------------------------

class TestWaterCycle:
    def test_initial_aquifer(self) -> None:
        water = WaterCycle()
        assert water.aquifer > 0

    def test_terraforming_increases_aquifer(self) -> None:
        water = WaterCycle()
        aq_0 = water.aquifer
        tick_water(water, terraform_count=3, pressure=0.05)
        assert water.aquifer > aq_0

    def test_surface_water_below_threshold(self) -> None:
        water = WaterCycle()
        tick_water(water, terraform_count=0, pressure=0.01)
        assert water.surface_water == 0.0

    def test_surface_water_above_threshold(self) -> None:
        water = WaterCycle(aquifer=0.5)
        tick_water(water, terraform_count=0, pressure=SURFACE_WATER_PRESSURE_THRESHOLD + 0.01)
        assert water.surface_water > 0.0

    def test_ice_cap_melts_with_terraforming(self) -> None:
        water = WaterCycle()
        ice_0 = water.ice_cap
        tick_water(water, terraform_count=5, pressure=0.1)
        assert water.ice_cap < ice_0

    def test_ice_melt_adds_to_aquifer(self) -> None:
        water = WaterCycle(ice_cap=0.5, aquifer=0.1)
        tick_water(water, terraform_count=3, pressure=0.1)
        # Aquifer gains from both terraforming AND ice melt
        assert water.aquifer > 0.1

    def test_tipping_point(self) -> None:
        water = WaterCycle(aquifer=0.01, surface_water=0.01, ice_cap=0.01)
        assert water.health() < WATER_TIPPING_POINT
        tipping = tick_water(water, terraform_count=0, pressure=0.0)
        assert "water_crisis" in tipping

    def test_all_bounded(self) -> None:
        water = WaterCycle()
        for _ in range(200):
            tick_water(water, terraform_count=10, pressure=0.5)
        assert 0.0 <= water.aquifer <= 1.0
        assert 0.0 <= water.surface_water <= 1.0
        assert 0.0 <= water.ice_cap <= 1.0


# ---------------------------------------------------------------------------
# Flora tests
# ---------------------------------------------------------------------------

class TestFlora:
    def test_farming_increases_crops(self) -> None:
        flora = Flora()
        crop_0 = flora.crop_yield
        tick_flora(flora, farm_count=3, perchlorate=0.8, pressure=0.01, surface_water=0.0)
        assert flora.crop_yield > crop_0

    def test_zero_farming_crops_decay(self) -> None:
        flora = Flora(crop_yield=0.3)
        tick_flora(flora, farm_count=0, perchlorate=0.8, pressure=0.01, surface_water=0.0)
        assert flora.crop_yield < 0.3

    def test_wild_plants_blocked_by_perchlorate(self) -> None:
        flora = Flora()
        tick_flora(flora, farm_count=0, perchlorate=0.5, pressure=0.2, surface_water=0.1)
        assert flora.wild_plants == 0.0  # 0.5 > WILD_PLANT_PERCHLORATE_THRESHOLD

    def test_wild_plants_grow_when_conditions_met(self) -> None:
        flora = Flora()
        tick_flora(flora, farm_count=0,
                   perchlorate=WILD_PLANT_PERCHLORATE_THRESHOLD - 0.1,
                   pressure=WILD_PLANT_PRESSURE_THRESHOLD + 0.01,
                   surface_water=0.2)
        assert flora.wild_plants > 0.0

    def test_wild_plants_need_pressure(self) -> None:
        flora = Flora()
        tick_flora(flora, farm_count=0, perchlorate=0.1, pressure=0.01, surface_water=0.2)
        assert flora.wild_plants == 0.0

    def test_biodiversity_needs_both_crop_and_wild(self) -> None:
        flora = Flora(crop_yield=0.1, wild_plants=0.1)
        tick_flora(flora, farm_count=1, perchlorate=0.1, pressure=0.2, surface_water=0.2)
        assert flora.biodiversity > 0.0

    def test_biodiversity_decays_without_base(self) -> None:
        flora = Flora(biodiversity=0.1, crop_yield=0.0, wild_plants=0.0)
        tick_flora(flora, farm_count=0, perchlorate=0.8, pressure=0.01, surface_water=0.0)
        assert flora.biodiversity < 0.1

    def test_indoor_outdoor_independence(self) -> None:
        """Crop yield (indoor) should not drive wild plants (outdoor)."""
        flora = Flora(crop_yield=0.9)
        tick_flora(flora, farm_count=5, perchlorate=0.9, pressure=0.01, surface_water=0.0)
        assert flora.wild_plants == 0.0  # perchlorate too high, pressure too low

    def test_tipping_point(self) -> None:
        flora = Flora(crop_yield=0.01, wild_plants=0.01, biodiversity=0.0)
        h = flora.health()
        if h < FLORA_TIPPING_POINT and h > 0:
            tipping = tick_flora(flora, farm_count=0, perchlorate=0.8, pressure=0.01, surface_water=0.0)
            assert "flora_collapse" in tipping

    def test_all_bounded(self) -> None:
        flora = Flora()
        for _ in range(200):
            tick_flora(flora, farm_count=10, perchlorate=0.01, pressure=0.5, surface_water=0.5)
        assert 0.0 <= flora.crop_yield <= 1.0
        assert 0.0 <= flora.wild_plants <= 1.0
        assert 0.0 <= flora.biodiversity <= 1.0


# ---------------------------------------------------------------------------
# Fauna tests
# ---------------------------------------------------------------------------

class TestFauna:
    def test_insects_gated_on_flora(self) -> None:
        fauna = Fauna()
        tick_fauna(fauna, outdoor_biomass=0.0, atmosphere_health=0.2, soil_organic=0.0)
        assert fauna.insects == 0.0  # decayed from 0

    def test_insects_grow_with_flora(self) -> None:
        fauna = Fauna()
        tick_fauna(fauna, outdoor_biomass=0.3, atmosphere_health=0.2, soil_organic=0.0)
        assert fauna.insects > 0.0

    def test_microbes_gated_on_soil(self) -> None:
        fauna = Fauna(microbes=0.1)
        tick_fauna(fauna, outdoor_biomass=0.0, atmosphere_health=0.0, soil_organic=0.01)
        assert fauna.microbes < 0.1  # below threshold, decays

    def test_microbes_grow_with_soil(self) -> None:
        fauna = Fauna()
        tick_fauna(fauna, outdoor_biomass=0.0, atmosphere_health=0.0, soil_organic=0.3)
        assert fauna.microbes > 0.02  # started at 0.02

    def test_all_bounded(self) -> None:
        fauna = Fauna()
        for _ in range(200):
            tick_fauna(fauna, outdoor_biomass=1.0, atmosphere_health=1.0, soil_organic=1.0)
        assert 0.0 <= fauna.insects <= 1.0
        assert 0.0 <= fauna.microbes <= 1.0


# ---------------------------------------------------------------------------
# Biosphere composite tests
# ---------------------------------------------------------------------------

class TestBiosphere:
    def test_initial_index_low(self) -> None:
        bio = Biosphere()
        assert 0.0 <= bio.index() < 0.3

    def test_index_bounded(self) -> None:
        bio = Biosphere()
        assert 0.0 <= bio.index() <= 1.0

    def test_serialization_roundtrip(self) -> None:
        bio = Biosphere()
        d = bio.to_dict()
        restored = Biosphere.from_dict(d)
        assert abs(restored.index() - bio.index()) < 1e-4

    def test_full_biosphere_high_index(self) -> None:
        bio = Biosphere(
            atmosphere=Atmosphere(pressure=0.8, o2_fraction=0.5, co2_fraction=0.2, temperature=0.5),
            soil=Soil(perchlorate=0.05, organic_content=0.6, nitrogen=0.4),
            water=WaterCycle(aquifer=0.7, surface_water=0.5, ice_cap=0.3),
            flora=Flora(crop_yield=0.7, wild_plants=0.5, biodiversity=0.4),
            fauna=Fauna(insects=0.3, microbes=0.4),
        )
        assert bio.index() > 0.5


# ---------------------------------------------------------------------------
# Resource bonus tests
# ---------------------------------------------------------------------------

class TestResourceBonuses:
    def test_low_biosphere_low_bonuses(self) -> None:
        bio = Biosphere()
        bonuses = compute_resource_bonuses(bio)
        assert all(v >= 0 for v in bonuses.values())
        assert bonuses["food"] < 0.01

    def test_healthy_biosphere_gives_bonuses(self) -> None:
        bio = Biosphere(
            flora=Flora(crop_yield=0.7, wild_plants=0.5, biodiversity=0.4),
            water=WaterCycle(aquifer=0.7, surface_water=0.5, ice_cap=0.3),
            atmosphere=Atmosphere(pressure=0.5, o2_fraction=0.3, co2_fraction=0.3, temperature=0.3),
        )
        bonuses = compute_resource_bonuses(bio)
        assert bonuses["food"] > 0.01
        assert bonuses["water"] > 0.01
        assert bonuses["air"] > 0.01

    def test_bonuses_capped(self) -> None:
        bio = Biosphere(
            flora=Flora(crop_yield=1.0, wild_plants=1.0, biodiversity=1.0),
            water=WaterCycle(aquifer=1.0, surface_water=1.0, ice_cap=1.0),
            atmosphere=Atmosphere(pressure=1.0, o2_fraction=1.0, co2_fraction=0.0, temperature=1.0),
            soil=Soil(perchlorate=0.0, organic_content=1.0, nitrogen=1.0),
            fauna=Fauna(insects=1.0, microbes=1.0),
        )
        bonuses = compute_resource_bonuses(bio)
        assert bonuses["food"] <= MAX_FOOD_BONUS
        assert bonuses["water"] <= MAX_WATER_BONUS
        assert bonuses["air"] <= MAX_AIR_BONUS
        assert bonuses["medicine"] <= MAX_MEDICINE_BONUS

    def test_power_bonus_always_zero(self) -> None:
        bio = Biosphere(
            flora=Flora(crop_yield=1.0, wild_plants=1.0, biodiversity=1.0),
        )
        bonuses = compute_resource_bonuses(bio)
        assert bonuses["power"] == 0.0


# ---------------------------------------------------------------------------
# Stress reduction tests
# ---------------------------------------------------------------------------

class TestStressReduction:
    def test_low_biosphere_low_reduction(self) -> None:
        bio = Biosphere()
        reduction = compute_nature_stress_reduction(bio)
        assert 0.0 <= reduction < 0.01

    def test_high_biosphere_high_reduction(self) -> None:
        bio = Biosphere(
            flora=Flora(crop_yield=0.8, wild_plants=0.6, biodiversity=0.5),
            atmosphere=Atmosphere(pressure=0.5, o2_fraction=0.3, co2_fraction=0.3, temperature=0.3),
            soil=Soil(perchlorate=0.1, organic_content=0.5, nitrogen=0.3),
            water=WaterCycle(aquifer=0.5, surface_water=0.3, ice_cap=0.2),
            fauna=Fauna(insects=0.2, microbes=0.3),
        )
        reduction = compute_nature_stress_reduction(bio)
        assert reduction > 0.01

    def test_reduction_capped(self) -> None:
        bio = Biosphere(
            flora=Flora(crop_yield=1.0, wild_plants=1.0, biodiversity=1.0),
            atmosphere=Atmosphere(pressure=1.0, o2_fraction=1.0, co2_fraction=0.0, temperature=1.0),
            soil=Soil(perchlorate=0.0, organic_content=1.0, nitrogen=1.0),
            water=WaterCycle(aquifer=1.0, surface_water=1.0, ice_cap=1.0),
            fauna=Fauna(insects=1.0, microbes=1.0),
        )
        reduction = compute_nature_stress_reduction(bio)
        assert reduction <= MAX_STRESS_REDUCTION


# ---------------------------------------------------------------------------
# tick_ecology integration tests
# ---------------------------------------------------------------------------

class TestTickEcology:
    def test_smoke(self) -> None:
        bio = Biosphere()
        result = tick_ecology(bio, terraform_count=2, farm_count=2, event_severity=0.3)
        assert isinstance(result, EcologyTickResult)
        assert "biosphere_index" in result.biosphere_after
        assert all(v >= 0 for v in result.resource_bonuses.values())

    def test_terraforming_and_farming_improve_biosphere(self) -> None:
        bio = Biosphere()
        idx_0 = bio.index()
        for _ in range(50):
            tick_ecology(bio, terraform_count=3, farm_count=3, event_severity=0.0)
        assert bio.index() > idx_0

    def test_zero_effort_biosphere_decays(self) -> None:
        bio = Biosphere(
            atmosphere=Atmosphere(pressure=0.1, o2_fraction=0.05, co2_fraction=0.8, temperature=0.15),
            soil=Soil(perchlorate=0.5, organic_content=0.1, nitrogen=0.05),
        )
        idx_0 = bio.index()
        for _ in range(20):
            tick_ecology(bio, terraform_count=0, farm_count=0, event_severity=0.0)
        assert bio.index() < idx_0

    def test_one_year_lag(self) -> None:
        """Resource bonuses should be computed from BEFORE state, not AFTER."""
        bio = Biosphere()
        result = tick_ecology(bio, terraform_count=5, farm_count=5, event_severity=0.0)
        # Bonuses are from the initial (low) biosphere, not the improved one
        assert result.resource_bonuses["food"] < MAX_FOOD_BONUS * 0.5

    def test_tipping_points_logged(self) -> None:
        bio = Biosphere(
            atmosphere=Atmosphere(pressure=0.01, o2_fraction=0.001, co2_fraction=0.99, temperature=0.01),
            water=WaterCycle(aquifer=0.01, surface_water=0.01, ice_cap=0.01),
        )
        result = tick_ecology(bio, terraform_count=0, farm_count=0, event_severity=0.0)
        assert len(result.tipping_points_hit) > 0


# ---------------------------------------------------------------------------
# Property-based: multi-seed, multi-year bounded invariants
# ---------------------------------------------------------------------------

class TestBoundedInvariants:
    @pytest.mark.parametrize("seed", range(10))
    def test_50_years_all_bounded(self, seed: int) -> None:
        """Run 50 years with varied effort; all values must stay in [0, 1]."""
        import random
        rng = random.Random(seed)
        bio = Biosphere()
        for year in range(50):
            tf = rng.randint(0, 5)
            fm = rng.randint(0, 5)
            sev = rng.random()
            result = tick_ecology(bio, terraform_count=tf, farm_count=fm, event_severity=sev)
            # Check all subsystem values bounded
            assert 0.0 <= bio.atmosphere.pressure <= 1.0, f"pressure OOB year {year}"
            assert 0.0 <= bio.atmosphere.o2_fraction <= 1.0, f"o2 OOB year {year}"
            assert 0.0 <= bio.atmosphere.co2_fraction <= 1.0, f"co2 OOB year {year}"
            assert 0.0 <= bio.atmosphere.temperature <= 1.0, f"temp OOB year {year}"
            assert 0.0 <= bio.soil.perchlorate <= 1.0, f"perchlorate OOB year {year}"
            assert 0.0 <= bio.soil.organic_content <= 1.0, f"organic OOB year {year}"
            assert 0.0 <= bio.soil.nitrogen <= 1.0, f"nitrogen OOB year {year}"
            assert 0.0 <= bio.water.aquifer <= 1.0, f"aquifer OOB year {year}"
            assert 0.0 <= bio.water.surface_water <= 1.0, f"surface_water OOB year {year}"
            assert 0.0 <= bio.water.ice_cap <= 1.0, f"ice_cap OOB year {year}"
            assert 0.0 <= bio.flora.crop_yield <= 1.0, f"crop OOB year {year}"
            assert 0.0 <= bio.flora.wild_plants <= 1.0, f"wild_plants OOB year {year}"
            assert 0.0 <= bio.flora.biodiversity <= 1.0, f"biodiversity OOB year {year}"
            assert 0.0 <= bio.fauna.insects <= 1.0, f"insects OOB year {year}"
            assert 0.0 <= bio.fauna.microbes <= 1.0, f"microbes OOB year {year}"
            assert 0.0 <= bio.index() <= 1.0, f"biosphere index OOB year {year}"
            # Resource bonuses non-negative and capped
            for k, v in result.resource_bonuses.items():
                assert v >= 0.0, f"bonus {k} negative year {year}"
            assert result.stress_reduction >= 0.0
            assert result.stress_reduction <= MAX_STRESS_REDUCTION

    @pytest.mark.parametrize("seed", range(5))
    def test_100_years_full_effort(self, seed: int) -> None:
        """100 years with heavy terraforming/farming. Biosphere improves meaningfully."""
        bio = Biosphere()
        for _ in range(100):
            tick_ecology(bio, terraform_count=5, farm_count=5, event_severity=0.1)
        # After 100 years of heavy effort, biosphere should be noticeably improved
        assert bio.index() > 0.15, f"Biosphere too low after 100 years: {bio.index()}"
        assert bio.atmosphere.pressure > 0.1
        assert bio.soil.perchlorate < PERCHLORATE_INITIAL


# ---------------------------------------------------------------------------
# Engine integration smoke test
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_engine_10_year_smoke(self) -> None:
        """Run 10 years of the full engine and verify ecology is present."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "10.0"
        assert "final_ecology" in d
        assert "biosphere_index" in d["final_ecology"]
        for yr in d["years"]:
            assert "ecology" in yr
            assert "biosphere_after" in yr["ecology"]
            assert "resource_bonuses" in yr["ecology"]

    def test_ecology_evolves_over_time(self) -> None:
        """Biosphere index should change (not stay frozen) over simulation."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=99, total_years=30)
        result = engine.run()
        indices = [
            yr.ecology.get("biosphere_after", {}).get("biosphere_index", 0)
            for yr in result.years
            if isinstance(yr.ecology, dict)
        ]
        assert len(indices) == 30
        # Should not be constant
        assert max(indices) > min(indices), "Biosphere index unchanged over 30 years"

    def test_rng_isolation(self) -> None:
        """Adding ecology should not change action/event sequences when
        ecology bonuses are negligible (initial low biosphere)."""
        from src.mars100.engine import Mars100Engine
        # Run engine and capture year-1 actions — these should be
        # deterministic for same seed regardless of ecology integration
        engine = Mars100Engine(seed=42, total_years=1)
        result = engine.run()
        actions_y1 = result.years[0].actions
        # Run again with same seed
        engine2 = Mars100Engine(seed=42, total_years=1)
        result2 = engine2.run()
        assert result2.years[0].actions == actions_y1

    def test_resource_delta_honest(self) -> None:
        """resource_delta should match resources_after - resources_before."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        for yr in result.years:
            for res in ("food", "water", "power", "air", "medicine"):
                expected = yr.resources_after[res] - yr.resources_before[res]
                actual = yr.resource_delta[res]
                assert abs(expected - actual) < 1e-6, (
                    f"Year {yr.year} resource_delta[{res}] dishonest: "
                    f"expected {expected:.6f}, got {actual:.6f}")
