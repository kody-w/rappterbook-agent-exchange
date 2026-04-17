"""Tests for the ecology organ (v10.0)."""
from __future__ import annotations

import random
import pytest
from src.mars100.ecology import (
    Atmosphere, SoilState, WaterCycle, Flora, Fauna,
    Biosphere, EcologyTickResult,
    tick_atmosphere, tick_soil, tick_water_cycle, tick_flora, tick_fauna,
    tick_ecology, compute_ecology_resource_bonus, compute_ecology_psych_bonus,
    MARS_BASE_CO2, PERCHLORATE_INITIAL, PSYCH_STRESS_REDUCTION_MAX,
)


class TestAtmosphere:
    """Tests for Atmosphere dataclass."""

    def test_defaults(self) -> None:
        a = Atmosphere()
        assert a.co2_fraction == pytest.approx(MARS_BASE_CO2)
        assert a.o2_fraction == pytest.approx(0.001)
        assert a.pressure == pytest.approx(0.006)

    def test_health_score_initial_near_zero(self) -> None:
        a = Atmosphere()
        assert 0.0 <= a.health_score() < 0.01

    def test_health_score_earthlike(self) -> None:
        a = Atmosphere(o2_fraction=0.21, pressure=101.3)
        assert a.health_score() == pytest.approx(1.0)

    def test_to_dict(self) -> None:
        a = Atmosphere()
        d = a.to_dict()
        assert "co2_fraction" in d and "o2_fraction" in d and "pressure" in d


class TestSoilState:
    """Tests for SoilState dataclass."""

    def test_defaults(self) -> None:
        s = SoilState()
        assert s.perchlorate_level == PERCHLORATE_INITIAL
        assert s.organic_content == 0.0

    def test_health_score_toxic(self) -> None:
        s = SoilState()
        assert s.health_score() == pytest.approx(0.1, abs=0.05)

    def test_health_score_remediated(self) -> None:
        s = SoilState(perchlorate_level=0.0, organic_content=0.3)
        assert s.health_score() == pytest.approx(1.0)


class TestWaterCycle:
    """Tests for WaterCycle dataclass."""

    def test_defaults(self) -> None:
        w = WaterCycle()
        assert w.ice_reserves == 0.5
        assert w.aquifer_level == 0.0
        assert w.surface_water == 0.0

    def test_health_score_range(self) -> None:
        w = WaterCycle()
        assert 0.0 <= w.health_score() <= 1.0


class TestFlora:
    """Tests for Flora dataclass."""

    def test_defaults(self) -> None:
        f = Flora()
        assert f.crop_yield == 0.0
        assert f.wild_coverage == 0.0

    def test_health_score_zero_initially(self) -> None:
        f = Flora()
        assert f.health_score() == 0.0

    def test_health_score_capped(self) -> None:
        f = Flora(crop_yield=2.0, wild_coverage=2.0, biodiversity=2.0)
        assert f.health_score() <= 1.0


class TestFauna:
    """Tests for Fauna dataclass."""

    def test_defaults(self) -> None:
        f = Fauna()
        assert f.population == 0.0

    def test_health_score_range(self) -> None:
        f = Fauna(population=0.5, diversity=0.3, ecosystem_stability=0.2)
        assert 0.0 <= f.health_score() <= 1.0


class TestBiosphere:
    """Tests for Biosphere composite."""

    def test_defaults(self) -> None:
        b = Biosphere()
        assert isinstance(b.atmosphere, Atmosphere)
        assert isinstance(b.soil, SoilState)
        assert isinstance(b.water, WaterCycle)
        assert isinstance(b.flora, Flora)
        assert isinstance(b.fauna, Fauna)

    def test_biosphere_index_range(self) -> None:
        b = Biosphere()
        idx = b.biosphere_index()
        assert 0.0 <= idx <= 1.0

    def test_biosphere_index_low_initially(self) -> None:
        b = Biosphere()
        assert b.biosphere_index() < 0.15

    def test_to_dict_has_all_keys(self) -> None:
        b = Biosphere()
        d = b.to_dict()
        assert "atmosphere" in d
        assert "soil" in d
        assert "water" in d
        assert "flora" in d
        assert "fauna" in d
        assert "biosphere_index" in d


class TestTickAtmosphere:
    """Tests for tick_atmosphere."""

    def test_o2_increases(self) -> None:
        a = Atmosphere()
        rng = random.Random(42)
        tick_atmosphere(a, 2, False, rng)
        assert a.o2_fraction > 0.001

    def test_co2_decreases(self) -> None:
        a = Atmosphere()
        initial_co2 = a.co2_fraction
        rng = random.Random(42)
        tick_atmosphere(a, 2, False, rng)
        assert a.co2_fraction < initial_co2

    def test_dust_storm_reduces_pressure(self) -> None:
        a = Atmosphere(pressure=0.05)
        rng = random.Random(42)
        events = tick_atmosphere(a, 0, True, rng)
        assert "dust_storm_pressure_loss" in events

    def test_pressure_stays_positive(self) -> None:
        a = Atmosphere(pressure=0.002)
        rng = random.Random(42)
        tick_atmosphere(a, 0, True, rng)
        assert a.pressure > 0


class TestTickSoil:
    """Tests for tick_soil."""

    def test_perchlorate_decreases(self) -> None:
        s = SoilState()
        rng = random.Random(42)
        tick_soil(s, 2, False, rng)
        assert s.perchlorate_level < PERCHLORATE_INITIAL

    def test_organic_increases(self) -> None:
        s = SoilState()
        rng = random.Random(42)
        tick_soil(s, 2, True, rng)
        assert s.organic_content > 0.0

    def test_perchlorate_bounded(self) -> None:
        s = SoilState(perchlorate_level=0.01)
        rng = random.Random(42)
        for _ in range(100):
            tick_soil(s, 5, True, rng)
        assert s.perchlorate_level >= 0.0


class TestTickWaterCycle:
    """Tests for tick_water_cycle."""

    def test_aquifer_increases(self) -> None:
        w = WaterCycle()
        rng = random.Random(42)
        tick_water_cycle(w, 2, 0.006, rng)
        assert w.aquifer_level > 0.0

    def test_no_surface_water_at_low_pressure(self) -> None:
        w = WaterCycle(surface_water=0.01)
        rng = random.Random(42)
        tick_water_cycle(w, 0, 0.006, rng)
        assert w.surface_water < 0.01

    def test_surface_water_at_high_pressure(self) -> None:
        w = WaterCycle()
        rng = random.Random(42)
        tick_water_cycle(w, 2, 1.0, rng)
        assert w.surface_water > 0.0


class TestTickFlora:
    """Tests for tick_flora."""

    def test_crops_grow_with_farming(self) -> None:
        f = Flora()
        s = SoilState(perchlorate_level=0.3, organic_content=0.1)
        w = WaterCycle(aquifer_level=0.5)
        rng = random.Random(42)
        tick_flora(f, s, w, 0.1, 3, rng)
        assert f.crop_yield > 0.0

    def test_no_wild_with_high_perchlorate(self) -> None:
        f = Flora()
        s = SoilState(perchlorate_level=0.8, organic_content=0.0)
        w = WaterCycle()
        rng = random.Random(42)
        tick_flora(f, s, w, 0.1, 0, rng)
        assert f.wild_coverage == 0.0

    def test_wild_grows_with_low_perchlorate(self) -> None:
        f = Flora()
        s = SoilState(perchlorate_level=0.2, organic_content=0.1)
        w = WaterCycle(aquifer_level=0.3)
        rng = random.Random(42)
        tick_flora(f, s, w, 0.1, 2, rng)
        assert f.wild_coverage > 0.0


class TestTickFauna:
    """Tests for tick_fauna."""

    def test_no_growth_without_flora(self) -> None:
        fa = Fauna()
        fl = Flora(crop_yield=0.0, wild_coverage=0.0)
        rng = random.Random(42)
        tick_fauna(fa, fl, rng)
        assert fa.population == 0.0

    def test_growth_with_flora(self) -> None:
        fa = Fauna()
        fl = Flora(crop_yield=0.2, wild_coverage=0.1)
        rng = random.Random(42)
        tick_fauna(fa, fl, rng)
        assert fa.population > 0.0

    def test_decline_without_support(self) -> None:
        fa = Fauna(population=0.3, diversity=0.1)
        fl = Flora(crop_yield=0.0, wild_coverage=0.0)
        rng = random.Random(42)
        events = tick_fauna(fa, fl, rng)
        assert fa.population < 0.3
        assert "fauna_declining" in events


class TestTickEcology:
    """Integration tests for the full ecology tick."""

    def test_returns_result(self) -> None:
        b = Biosphere()
        result = tick_ecology(b, 1, 2, 10, rng=random.Random(42))
        assert isinstance(result, EcologyTickResult)
        assert result.year == 1
        assert isinstance(result.events, list)

    def test_biosphere_mutated(self) -> None:
        b = Biosphere()
        before_o2 = b.atmosphere.o2_fraction
        tick_ecology(b, 1, 2, 10, rng=random.Random(42))
        assert b.atmosphere.o2_fraction != before_o2

    def test_dust_storm_flag(self) -> None:
        b = Biosphere()
        result = tick_ecology(b, 1, 0, 10, has_dust_storm=True, rng=random.Random(42))
        assert "dust_storm_pressure_loss" in result.events

    def test_solar_event_damages(self) -> None:
        b = Biosphere()
        b.flora.wild_coverage = 0.5
        tick_ecology(b, 50, 0, 10, has_solar_event=True, rng=random.Random(42))
        assert b.flora.wild_coverage < 0.5

    def test_result_serializes(self) -> None:
        b = Biosphere()
        result = tick_ecology(b, 1, 2, 10, rng=random.Random(42))
        d = result.to_dict()
        assert "year" in d
        assert "biosphere_before" in d
        assert "biosphere_after" in d
        assert "biosphere_index" in d

    def test_100_year_integration(self) -> None:
        """Run 100 years and verify biosphere evolves within bounds."""
        b = Biosphere()
        rng = random.Random(42)
        for year in range(1, 101):
            effort = min(5, year // 10)
            result = tick_ecology(b, year, effort, 10, rng=rng)
            assert 0.0 <= result.biosphere_index <= 1.0
        # After 100 years of effort, biosphere should have improved
        assert b.biosphere_index() > 0.05
        assert b.atmosphere.o2_fraction > 0.01
        assert b.soil.perchlorate_level < PERCHLORATE_INITIAL


class TestResourceBonus:
    """Tests for compute_ecology_resource_bonus."""

    def test_initial_near_zero(self) -> None:
        b = Biosphere()
        bonus = compute_ecology_resource_bonus(b)
        assert bonus["food"] >= 0.0
        assert bonus["water"] >= 0.0
        assert bonus["air_maintenance_reduction"] >= 0.0

    def test_bonus_increases_with_development(self) -> None:
        b = Biosphere()
        b.flora.crop_yield = 0.5
        b.water.aquifer_level = 0.3
        b.atmosphere.o2_fraction = 0.05
        bonus = compute_ecology_resource_bonus(b)
        assert bonus["food"] > 0.0
        assert bonus["water"] > 0.0
        assert bonus["air_maintenance_reduction"] > 0.0


class TestPsychBonus:
    """Tests for compute_ecology_psych_bonus."""

    def test_initial_near_zero(self) -> None:
        b = Biosphere()
        bonus = compute_ecology_psych_bonus(b)
        assert 0.0 <= bonus <= PSYCH_STRESS_REDUCTION_MAX

    def test_bonus_increases(self) -> None:
        b = Biosphere()
        b.flora.crop_yield = 0.5
        b.water.aquifer_level = 0.5
        b.atmosphere.o2_fraction = 0.1
        b.soil.perchlorate_level = 0.1
        b.soil.organic_content = 0.2
        bonus = compute_ecology_psych_bonus(b)
        assert bonus > 0.0

    def test_bounded(self) -> None:
        b = Biosphere()
        b.flora.crop_yield = 1.0
        b.flora.wild_coverage = 1.0
        b.flora.biodiversity = 1.0
        b.fauna.population = 1.0
        b.fauna.diversity = 1.0
        b.fauna.ecosystem_stability = 1.0
        b.atmosphere.o2_fraction = 0.21
        b.atmosphere.pressure = 101.3
        b.soil.perchlorate_level = 0.0
        b.soil.organic_content = 0.3
        b.water.aquifer_level = 1.0
        b.water.surface_water = 1.0
        bonus = compute_ecology_psych_bonus(b)
        assert bonus <= PSYCH_STRESS_REDUCTION_MAX


class TestPropertyInvariants:
    """Property-based invariants that must hold for any input."""

    @pytest.mark.parametrize("seed", range(10))
    def test_biosphere_index_bounded(self, seed: int) -> None:
        b = Biosphere()
        rng = random.Random(seed)
        for year in range(50):
            tick_ecology(b, year, rng.randint(0, 5), 10, rng=rng)
        assert 0.0 <= b.biosphere_index() <= 1.0

    @pytest.mark.parametrize("seed", range(10))
    def test_all_values_bounded(self, seed: int) -> None:
        b = Biosphere()
        rng = random.Random(seed)
        for year in range(50):
            tick_ecology(b, year, rng.randint(0, 5), 10,
                         has_dust_storm=rng.random() < 0.2,
                         has_solar_event=rng.random() < 0.1,
                         rng=rng)
        assert 0.0 <= b.atmosphere.o2_fraction <= 1.0
        assert 0.0 <= b.atmosphere.co2_fraction <= 1.0
        assert b.atmosphere.pressure >= 0.001
        assert 0.0 <= b.soil.perchlorate_level <= 1.0
        assert 0.0 <= b.soil.organic_content <= 1.0
        assert 0.0 <= b.water.ice_reserves <= 1.0
        assert 0.0 <= b.water.aquifer_level <= 1.0
        assert 0.0 <= b.water.surface_water <= 1.0
        assert 0.0 <= b.flora.crop_yield <= 1.0
        assert 0.0 <= b.flora.wild_coverage <= 1.0
        assert 0.0 <= b.fauna.population <= 1.0
