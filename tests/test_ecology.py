"""Tests for the ecology organ (engine v10.0)."""
from __future__ import annotations

import random
import pytest
from src.mars100.ecology import (
    Atmosphere, SoilState, WaterCycle, Flora, Fauna,
    Biosphere, EcologyTickResult, tick_ecology,
    compute_ecology_resource_bonus, compute_ecology_psych_pressure,
    compute_ecology_modifiers, compute_ecology_upkeep,
)


# --------------- Dataclass basics ---------------

class TestAtmosphere:
    def test_defaults(self):
        a = Atmosphere()
        assert a.pressure_kpa == pytest.approx(0.6)
        assert 0.0 <= a.health() <= 1.0

    def test_clamp(self):
        a = Atmosphere(pressure_kpa=-5.0, o2_fraction=0.5, co2_fraction=2.0)
        a.clamp()
        assert a.pressure_kpa >= 0.6
        assert a.o2_fraction <= 0.25
        assert a.co2_fraction <= 1.0

    def test_roundtrip(self):
        a = Atmosphere(pressure_kpa=8.0, o2_fraction=0.05, co2_fraction=0.6)
        a2 = Atmosphere.from_dict(a.to_dict())
        assert a2.pressure_kpa == pytest.approx(a.pressure_kpa, abs=1e-3)


class TestSoilState:
    def test_defaults(self):
        s = SoilState()
        assert s.perchlorate == pytest.approx(0.75)
        assert 0.0 <= s.health() <= 1.0

    def test_clamp(self):
        s = SoilState(perchlorate=-1.0, organic_content=5.0, moisture=3.0)
        s.clamp()
        assert 0.0 <= s.perchlorate <= 1.0
        assert 0.0 <= s.organic_content <= 1.0

    def test_roundtrip(self):
        s = SoilState(perchlorate=0.3, organic_content=0.2, moisture=0.15)
        s2 = SoilState.from_dict(s.to_dict())
        assert s2.perchlorate == pytest.approx(s.perchlorate, abs=1e-3)


class TestWaterCycle:
    def test_defaults(self):
        w = WaterCycle()
        assert w.aquifer == pytest.approx(0.6)
        assert 0.0 <= w.health() <= 1.0

    def test_roundtrip(self):
        w = WaterCycle(aquifer=0.4, surface_water=0.1, ice_reserves=0.5)
        w2 = WaterCycle.from_dict(w.to_dict())
        assert w2.aquifer == pytest.approx(w.aquifer, abs=1e-3)


class TestFlora:
    def test_defaults(self):
        f = Flora()
        assert f.crops == pytest.approx(0.1)

    def test_roundtrip(self):
        f = Flora(crops=0.3, wild_plants=0.1, biomass=0.2)
        f2 = Flora.from_dict(f.to_dict())
        assert f2.crops == pytest.approx(f.crops, abs=1e-3)


class TestFauna:
    def test_defaults(self):
        f = Fauna()
        assert f.microbes == pytest.approx(0.02)

    def test_roundtrip(self):
        f = Fauna(insects=0.1, microbes=0.05)
        f2 = Fauna.from_dict(f.to_dict())
        assert f2.insects == pytest.approx(f.insects, abs=1e-3)


class TestBiosphere:
    def test_defaults(self):
        b = Biosphere()
        assert 0.0 <= b.biosphere_index() <= 1.0
        assert b.health() == b.biosphere_index()

    def test_clamp(self):
        b = Biosphere()
        b.atmosphere.pressure_kpa = -100
        b.clamp()
        assert b.atmosphere.pressure_kpa >= 0.6

    def test_roundtrip(self):
        b = Biosphere()
        d = b.to_dict()
        b2 = Biosphere.from_dict(d)
        assert b2.biosphere_index() == pytest.approx(b.biosphere_index(), abs=1e-3)


# --------------- tick_ecology ---------------

class TestTickEcology:
    def test_basic_tick(self):
        bio = Biosphere()
        rng = random.Random(42)
        result = tick_ecology(bio, 1, rng=rng)
        assert isinstance(result, EcologyTickResult)
        assert result.biosphere_before != {}
        assert result.biosphere_after != {}

    def test_terraformers_increase_pressure(self):
        bio = Biosphere()
        initial_p = bio.atmosphere.pressure_kpa
        tick_ecology(bio, 1, terraformers=3, rng=random.Random(42))
        assert bio.atmosphere.pressure_kpa > initial_p

    def test_farmers_remediate_perchlorate(self):
        bio = Biosphere()
        initial_perch = bio.soil.perchlorate
        tick_ecology(bio, 1, farmers=5, rng=random.Random(42))
        assert bio.soil.perchlorate < initial_perch

    def test_farmers_grow_crops(self):
        bio = Biosphere()
        initial_crops = bio.flora.crops
        tick_ecology(bio, 1, farmers=3, rng=random.Random(42))
        assert bio.flora.crops > initial_crops

    def test_saboteurs_damage_crops(self):
        bio = Biosphere()
        bio.flora.crops = 0.5
        tick_ecology(bio, 1, saboteurs=2, rng=random.Random(42))
        assert bio.flora.crops < 0.5

    def test_event_damage_lowers_pressure(self):
        bio = Biosphere()
        bio.atmosphere.pressure_kpa = 10.0
        tick_ecology(bio, 1, event_damage=0.8, rng=random.Random(42))
        assert bio.atmosphere.pressure_kpa < 10.0

    def test_infra_protects_from_damage(self):
        """shelter_reinforcement should reduce event damage."""
        bio1 = Biosphere()
        bio1.atmosphere.pressure_kpa = 10.0
        bio2 = Biosphere()
        bio2.atmosphere.pressure_kpa = 10.0
        tick_ecology(bio1, 1, event_damage=0.8, rng=random.Random(42))
        tick_ecology(bio2, 1, event_damage=0.8,
                     infra_completed=["shelter_reinforcement"],
                     rng=random.Random(42))
        # With shelter, less pressure loss
        assert bio2.atmosphere.pressure_kpa >= bio1.atmosphere.pressure_kpa

    def test_greenhouse_boosts_crops(self):
        bio1 = Biosphere()
        bio2 = Biosphere()
        tick_ecology(bio1, 1, farmers=3, rng=random.Random(42))
        tick_ecology(bio2, 1, farmers=3,
                     infra_completed=["greenhouse_dome"],
                     rng=random.Random(42))
        assert bio2.flora.crops >= bio1.flora.crops

    def test_aquifer_drain(self):
        bio = Biosphere()
        initial_aquifer = bio.water.aquifer
        tick_ecology(bio, 1, rng=random.Random(42))
        assert bio.water.aquifer < initial_aquifer

    def test_100_year_stability(self):
        """Run 100 years — biosphere stays bounded [0,1]."""
        bio = Biosphere()
        rng = random.Random(42)
        for year in range(1, 101):
            tick_ecology(bio, year, terraformers=2, farmers=3,
                         researchers=1, rng=rng)
        idx = bio.biosphere_index()
        assert 0.0 <= idx <= 1.0
        # After 100 years of effort, should have improved
        assert idx > Biosphere().biosphere_index()

    def test_result_serialization(self):
        bio = Biosphere()
        result = tick_ecology(bio, 1, rng=random.Random(42))
        d = result.to_dict()
        assert "biosphere_before" in d
        assert "biosphere_after" in d
        assert "events" in d


# --------------- Property invariants ---------------

class TestEcologyInvariants:
    @pytest.mark.parametrize("seed", range(10))
    def test_all_values_bounded(self, seed):
        """All ecology values stay in [0, 1] regardless of inputs."""
        bio = Biosphere()
        rng = random.Random(seed)
        for year in range(1, 51):
            t = rng.randint(0, 5)
            f = rng.randint(0, 5)
            r = rng.randint(0, 3)
            s = rng.randint(0, 2)
            d = rng.uniform(0, 1)
            tick_ecology(bio, year, terraformers=t, farmers=f,
                         researchers=r, saboteurs=s, event_damage=d, rng=rng)
        # Check all sub-values bounded
        assert 0.0 <= bio.atmosphere.o2_fraction <= 0.25
        assert 0.0 <= bio.atmosphere.co2_fraction <= 1.0
        assert bio.atmosphere.pressure_kpa >= 0.6
        assert 0.0 <= bio.soil.perchlorate <= 1.0
        assert 0.0 <= bio.soil.organic_content <= 1.0
        assert 0.0 <= bio.soil.moisture <= 1.0
        assert 0.0 <= bio.water.aquifer <= 1.0
        assert 0.0 <= bio.water.surface_water <= 1.0
        assert 0.0 <= bio.water.ice_reserves <= 1.0
        assert 0.0 <= bio.flora.crops <= 1.0
        assert 0.0 <= bio.flora.wild_plants <= 1.0
        assert 0.0 <= bio.flora.biomass <= 1.0
        assert 0.0 <= bio.fauna.insects <= 1.0
        assert 0.0 <= bio.fauna.microbes <= 1.0

    @pytest.mark.parametrize("seed", range(5))
    def test_biosphere_index_bounded(self, seed):
        bio = Biosphere()
        rng = random.Random(seed)
        for _ in range(100):
            tick_ecology(bio, 1, terraformers=rng.randint(0, 5),
                         farmers=rng.randint(0, 5), rng=rng)
        assert 0.0 <= bio.biosphere_index() <= 1.0


# --------------- Resource bonus / modifier / upkeep ---------------

class TestResourceBonus:
    def test_all_non_negative(self):
        bio = Biosphere()
        bonus = compute_ecology_resource_bonus(bio)
        for v in bonus.values():
            assert v >= 0.0

    def test_capped(self):
        bio = Biosphere()
        bio.flora.crops = 1.0
        bio.flora.wild_plants = 1.0
        bio.flora.biomass = 1.0
        bio.water.surface_water = 1.0
        bio.fauna.microbes = 1.0
        bonus = compute_ecology_resource_bonus(bio)
        assert bonus["food"] <= 0.03
        assert bonus["water"] <= 0.02
        assert bonus["air"] <= 0.025
        assert bonus["medicine"] <= 0.01


class TestModifiers:
    def test_neutral_at_zero(self):
        """Empty biosphere → modifiers near 1.0 (no benefit)."""
        bio = Biosphere()
        # Set everything to minimum
        bio.atmosphere = Atmosphere(pressure_kpa=0.6, o2_fraction=0.0, co2_fraction=1.0)
        bio.soil = SoilState(perchlorate=1.0, organic_content=0.0, moisture=0.0)
        bio.water = WaterCycle(aquifer=0.0, surface_water=0.0, ice_reserves=0.0)
        bio.flora = Flora(crops=0.0, wild_plants=0.0, biomass=0.0)
        bio.fauna = Fauna(insects=0.0, microbes=0.0)
        mods = compute_ecology_modifiers(bio)
        assert mods["food_spoilage_mult"] == pytest.approx(1.0, abs=0.01)

    def test_green_biosphere_reduces_spoilage(self):
        bio = Biosphere()
        # Max out everything
        bio.atmosphere = Atmosphere(pressure_kpa=30.0, o2_fraction=0.21, co2_fraction=0.04)
        bio.soil = SoilState(perchlorate=0.0, organic_content=1.0, moisture=1.0)
        bio.water = WaterCycle(aquifer=1.0, surface_water=1.0, ice_reserves=1.0)
        bio.flora = Flora(crops=1.0, wild_plants=1.0, biomass=1.0)
        bio.fauna = Fauna(insects=1.0, microbes=1.0)
        mods = compute_ecology_modifiers(bio)
        assert mods["food_spoilage_mult"] < 1.0
        assert mods["air_maintenance_mult"] < 1.0

    def test_all_multipliers_at_least_floor(self):
        bio = Biosphere()
        mods = compute_ecology_modifiers(bio)
        assert mods["food_spoilage_mult"] >= 0.75
        assert mods["air_maintenance_mult"] >= 0.80
        assert mods["water_maintenance_mult"] >= 0.85


class TestUpkeep:
    def test_minimal_at_start(self):
        bio = Biosphere()
        upkeep = compute_ecology_upkeep(bio)
        assert upkeep["power"] >= 0.0
        assert upkeep["water"] >= 0.0

    def test_scales_with_flora(self):
        bio1 = Biosphere()
        bio2 = Biosphere()
        bio2.flora.crops = 0.8
        bio2.fauna.insects = 0.5
        u1 = compute_ecology_upkeep(bio1)
        u2 = compute_ecology_upkeep(bio2)
        assert u2["power"] > u1["power"]


class TestPsychPressure:
    def test_stress_negative_or_zero(self):
        bio = Biosphere()
        pp = compute_ecology_psych_pressure(bio)
        assert pp["stress"] <= 0.0

    def test_purpose_positive_or_zero(self):
        bio = Biosphere()
        pp = compute_ecology_psych_pressure(bio)
        assert pp["purpose"] >= 0.0

    def test_scales_with_health(self):
        bio = Biosphere()
        bio.flora = Flora(crops=1.0, wild_plants=1.0, biomass=1.0)
        bio.fauna = Fauna(insects=1.0, microbes=1.0)
        pp = compute_ecology_psych_pressure(bio)
        assert pp["purpose"] > 0.0
        assert pp["stress"] < 0.0


# --------------- Golden path: 100-year terraforming ---------------

class TestGoldenPath:
    def test_flora_reachable_within_100_years(self):
        """With sustained terraforming + farming, wild plants should emerge."""
        bio = Biosphere()
        rng = random.Random(42)
        for year in range(1, 101):
            tick_ecology(bio, year, terraformers=3, farmers=4,
                         researchers=1, rng=rng)
        # Wild plants should have grown (perchlorate remediated, pressure up)
        assert bio.flora.wild_plants > 0.01

    def test_fauna_reachable_within_100_years(self):
        """With sustained effort, insects/microbes should grow."""
        bio = Biosphere()
        rng = random.Random(42)
        for year in range(1, 101):
            tick_ecology(bio, year, terraformers=3, farmers=4,
                         researchers=1, rng=rng)
        assert bio.fauna.microbes > 0.02  # above initial

    def test_biosphere_improves_with_effort(self):
        bio = Biosphere()
        initial_idx = bio.biosphere_index()
        rng = random.Random(42)
        for year in range(1, 101):
            tick_ecology(bio, year, terraformers=2, farmers=3, rng=rng)
        assert bio.biosphere_index() > initial_idx

    def test_no_effort_biosphere_decays(self):
        bio = Biosphere()
        initial_idx = bio.biosphere_index()
        rng = random.Random(42)
        for year in range(1, 51):
            tick_ecology(bio, year, rng=rng)
        # Without effort, biosphere should not improve (may stay same or decay)
        assert bio.biosphere_index() <= initial_idx + 0.05
