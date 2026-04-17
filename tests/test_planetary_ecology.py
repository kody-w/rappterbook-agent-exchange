"""Tests for the planetary ecology organ (ecology.py).

Covers both layers:
  - Biosphere: biomass, soil, crop diversity, air/food deltas
  - MarsEcology: atmosphere, dust, water_ice, radiation, terraform
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.mars100.ecology import (
    Biosphere, EcologyDelta, tick_biosphere,
    MarsEcology, ECOLOGY_VARS, tick_ecology,
    compute_ecology_resource_modifiers,
    compute_ecology_death_modifier,
    _clamp,
)


def _eco() -> MarsEcology:
    return MarsEcology()


class TestClamp:
    def test_within(self):
        assert _clamp(0.5) == 0.5

    def test_below(self):
        assert _clamp(-1.0) == 0.0

    def test_above(self):
        assert _clamp(2.0) == 1.0

    def test_custom_bounds(self):
        assert _clamp(3.0, 0.5, 2.0) == 2.0
        assert _clamp(0.1, 0.5, 2.0) == 0.5


class TestBiosphere:
    def test_defaults_bounded(self):
        b = Biosphere()
        for attr in ("biomass", "soil_health", "crop_diversity"):
            assert 0.0 <= getattr(b, attr) <= 1.0

    def test_to_dict(self):
        d = Biosphere(biomass=0.12345).to_dict()
        assert d["biomass"] == round(0.12345, 4)

    def test_tick_no_inputs(self):
        b = Biosphere()
        d = tick_biosphere(b, 0, 0.0)
        assert isinstance(d, EcologyDelta)
        assert d.air_delta >= 0
        assert d.food_delta >= 0

    def test_farmers_boost(self):
        b1, b2 = Biosphere(), Biosphere()
        tick_biosphere(b1, 0, 0.0)
        tick_biosphere(b2, 5, 0.0)
        assert b2.biomass >= b1.biomass
        assert b2.soil_health >= b1.soil_health

    def test_event_damage(self):
        b1, b2 = Biosphere(), Biosphere()
        tick_biosphere(b1, 0, 0.0)
        tick_biosphere(b2, 0, 1.0)
        assert b2.biomass <= b1.biomass

    def test_bounds_200_good_ticks(self):
        b = Biosphere()
        for _ in range(200):
            tick_biosphere(b, 10, 0.0)
        for attr in ("biomass", "soil_health", "crop_diversity"):
            assert 0.0 <= getattr(b, attr) <= 1.0

    def test_bounds_200_bad_ticks(self):
        b = Biosphere()
        for _ in range(200):
            tick_biosphere(b, 0, 1.0)
        for attr in ("biomass", "soil_health", "crop_diversity"):
            assert 0.0 <= getattr(b, attr) <= 1.0

    def test_food_scales_with_farmers(self):
        b1 = Biosphere(soil_health=0.5, crop_diversity=0.5)
        b2 = Biosphere(soil_health=0.5, crop_diversity=0.5)
        d1 = tick_biosphere(b1, 1, 0.0)
        d2 = tick_biosphere(b2, 5, 0.0)
        assert d2.food_delta > d1.food_delta

    def test_delta_to_dict(self):
        d = EcologyDelta(air_delta=0.001).to_dict()
        assert "air_delta" in d and "food_delta" in d


class TestMarsEcology:
    def test_defaults_bounded(self):
        eco = MarsEcology()
        for v in ECOLOGY_VARS:
            assert 0.0 <= getattr(eco, v) <= 1.0

    def test_to_dict_keys(self):
        assert set(MarsEcology().to_dict().keys()) == set(ECOLOGY_VARS)

    def test_habitability_bounded(self):
        assert 0.0 <= MarsEcology().habitability() <= 1.0

    def test_perfect_planet(self):
        eco = MarsEcology(atmosphere=1, dust=0, water_ice=1, radiation=0, terraform=1)
        assert eco.habitability() == 1.0

    def test_dead_planet(self):
        eco = MarsEcology(atmosphere=0, dust=1, water_ice=0, radiation=1, terraform=0)
        assert eco.habitability() == 0.0

    def test_harshness_inverse(self):
        eco = MarsEcology()
        assert abs(eco.habitability() + eco.harshness() - 1.0) < 1e-10


class TestTickEcology:
    def test_keys(self):
        r = tick_ecology(_eco(), 10, 2, [])
        for k in ("before", "after", "deltas", "habitability", "harshness"):
            assert k in r

    def test_bounds(self):
        eco = _eco()
        tick_ecology(eco, 10, 5, [], 0.5)
        for v in ECOLOGY_VARS:
            assert 0.0 <= getattr(eco, v) <= 1.0

    def test_terraform_increases(self):
        e1, e2 = _eco(), _eco()
        tick_ecology(e1, 10, 0, [])
        tick_ecology(e2, 10, 5, [])
        assert e2.terraform > e1.terraform

    def test_diminishing_returns(self):
        e1, e2 = _eco(), _eco()
        tick_ecology(e1, 10, 1, [])
        tick_ecology(e2, 10, 9, [])
        assert e2.terraform / max(e1.terraform, 1e-10) < 9.0

    def test_water_depletes(self):
        eco = _eco()
        w0 = eco.water_ice
        tick_ecology(eco, 10, 0, [])
        assert eco.water_ice < w0

    def test_atmosphere_reduces_dust(self):
        eco = MarsEcology(atmosphere=0.5, dust=0.5)
        tick_ecology(eco, 5, 0, [])
        assert eco.dust < 0.5

    def test_atmosphere_reduces_radiation(self):
        eco = MarsEcology(atmosphere=0.5)
        tick_ecology(eco, 5, 0, [])
        assert eco.radiation < 0.7

    def test_greenhouse_dome(self):
        e1, e2 = _eco(), _eco()
        tick_ecology(e1, 10, 0, [])
        tick_ecology(e2, 10, 0, ["greenhouse_dome"])
        assert e2.atmosphere > e1.atmosphere

    def test_water_recycler(self):
        e1, e2 = _eco(), _eco()
        tick_ecology(e1, 10, 0, [])
        tick_ecology(e2, 10, 0, ["water_recycler"])
        assert e2.water_ice > e1.water_ice

    def test_shelter(self):
        e1, e2 = _eco(), _eco()
        tick_ecology(e1, 10, 0, [])
        tick_ecology(e2, 10, 0, ["shelter_reinforcement"])
        assert e2.dust < e1.dust

    def test_100_years_bounded(self):
        eco = _eco()
        for _ in range(100):
            tick_ecology(eco, 15, 3, ["greenhouse_dome", "water_recycler"], 0.5)
        for v in ECOLOGY_VARS:
            assert 0.0 <= getattr(eco, v) <= 1.0

    def test_extreme_depletion_bounded(self):
        eco = _eco()
        for _ in range(100):
            tick_ecology(eco, 50, 0, [], -1.0)
        for v in ECOLOGY_VARS:
            assert 0.0 <= getattr(eco, v) <= 1.0

    def test_determinism(self):
        e1, e2 = _eco(), _eco()
        r1 = tick_ecology(e1, 10, 3, [], 0.5)
        r2 = tick_ecology(e2, 10, 3, [], 0.5)
        assert e1.to_dict() == e2.to_dict()
        assert r1 == r2

    def test_deltas_consistent(self):
        eco = _eco()
        r = tick_ecology(eco, 10, 2, [])
        for v in ECOLOGY_VARS:
            expected = r["after"][v] - r["before"][v]
            assert abs(r["deltas"][v] - round(expected, 6)) < 1e-5


class TestResourceModifiers:
    def test_keys_and_range(self):
        mods = compute_ecology_resource_modifiers(_eco())
        for k in ("water", "power", "food", "air"):
            assert k in mods
            assert 0.5 <= mods[k] <= 1.5

    def test_perfect_high(self):
        eco = MarsEcology(atmosphere=1, dust=0, water_ice=1, terraform=1)
        mods = compute_ecology_resource_modifiers(eco)
        for k in ("water", "power", "food", "air"):
            assert mods[k] >= 1.0

    def test_dead_low(self):
        eco = MarsEcology(atmosphere=0, dust=1, water_ice=0, terraform=0)
        mods = compute_ecology_resource_modifiers(eco)
        for k in ("water", "power", "food", "air"):
            assert mods[k] <= 1.0

    def test_water_scales_with_ice(self):
        hi = compute_ecology_resource_modifiers(MarsEcology(water_ice=1.0))["water"]
        lo = compute_ecology_resource_modifiers(MarsEcology(water_ice=0.0))["water"]
        assert hi > lo

    def test_power_inverse_dust(self):
        clear = compute_ecology_resource_modifiers(MarsEcology(dust=0.0))["power"]
        dusty = compute_ecology_resource_modifiers(MarsEcology(dust=1.0))["power"]
        assert clear > dusty


class TestDeathModifier:
    def test_bounded_sweep(self):
        for a in (0.0, 0.5, 1.0):
            for d in (0.0, 0.5, 1.0):
                for w in (0.0, 0.5, 1.0):
                    eco = MarsEcology(atmosphere=a, dust=d, water_ice=w)
                    mod = compute_ecology_death_modifier(eco)
                    assert 0.5 <= mod <= 2.0

    def test_harsh_high(self):
        eco = MarsEcology(atmosphere=0, dust=1, water_ice=0, radiation=1, terraform=0)
        assert compute_ecology_death_modifier(eco) == 2.0

    def test_perfect_low(self):
        eco = MarsEcology(atmosphere=1, dust=0, water_ice=1, radiation=0, terraform=1)
        assert compute_ecology_death_modifier(eco) == 0.5

    def test_neutral_near_one(self):
        eco = MarsEcology(atmosphere=0.5, dust=0.5, water_ice=0.5, radiation=0.5, terraform=0)
        assert 0.8 <= compute_ecology_death_modifier(eco) <= 1.5


class TestEngineIntegration:
    def test_year_result_has_ecology(self):
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(seed=42, total_years=3)
        r = e.run()
        yr = r.years[0]
        assert yr.ecology and "habitability" in yr.ecology
        assert yr.biosphere and "biomass" in yr.biosphere

    def test_sim_result_has_ecology(self):
        from src.mars100.engine import Mars100Engine
        r = Mars100Engine(seed=42, total_years=3).run()
        assert r.final_ecology
        for v in ECOLOGY_VARS:
            assert v in r.final_ecology

    def test_to_dict(self):
        from src.mars100.engine import Mars100Engine
        d = Mars100Engine(seed=42, total_years=3).run().to_dict()
        assert "final_ecology" in d
        assert "ecology" in d["years"][0]
        assert "biosphere" in d["years"][0]

    def test_version(self):
        from src.mars100.engine import Mars100Engine
        d = Mars100Engine(seed=42, total_years=1).run().to_dict()
        assert d["_meta"]["version"] == "5.0"

    def test_100_year_smoke(self):
        from src.mars100.engine import Mars100Engine
        r = Mars100Engine(seed=42, total_years=100).run()
        assert len(r.years) > 0
        for yr in r.years:
            assert 0.0 <= yr.ecology["habitability"] <= 1.0

    def test_ecology_evolves(self):
        from src.mars100.engine import Mars100Engine
        r = Mars100Engine(seed=42, total_years=20).run()
        f, l = r.years[0].ecology, r.years[-1].ecology
        assert any(abs(f["after"][v] - l["after"][v]) > 0.001 for v in ECOLOGY_VARS)

    def test_biosphere_evolves(self):
        from src.mars100.engine import Mars100Engine
        r = Mars100Engine(seed=42, total_years=20).run()
        assert r.years[0].biosphere != r.years[-1].biosphere

    def test_deterministic(self):
        from src.mars100.engine import Mars100Engine
        r1 = Mars100Engine(seed=99, total_years=10).run()
        r2 = Mars100Engine(seed=99, total_years=10).run()
        for i in range(len(r1.years)):
            assert r1.years[i].ecology == r2.years[i].ecology
