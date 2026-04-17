"""Tests for Mars-100 biosphere ecology organ."""
from __future__ import annotations

import random
import pytest
from src.mars100.ecology import (
    Biosphere, EcologyDelta, tick_biosphere,
    _light_factor, _farming_intensity,
    BIOMASS_GROWTH_BASE, BIOMASS_DECAY_STRESS, SOIL_REGEN_RATE,
    SOIL_DEGRADE_RATE, PHOTOSYNTHESIS_RATE, CROP_YIELD_RATE,
    BLIGHT_THRESHOLD, BLIGHT_BIOMASS_LOSS, BLIGHT_PROBABILITY,
    GREENHOUSE_GROWTH_MULT, GREENHOUSE_FOOD_MULT,
    RESEARCH_LAB_DIVERSITY_BOOST, MARS_AMBIENT_LIGHT,
    MIN_WATER_FOR_GROWTH, MIN_POWER_FOR_LIGHTS,
    DIVERSITY_DECAY_RATE, DIVERSITY_FARMING_PENALTY,
)


# ── Biosphere dataclass ──────────────────────────────────────────

class TestBiosphere:
    def test_defaults(self):
        b = Biosphere()
        assert b.biomass == 0.1
        assert b.soil_health == 0.5
        assert b.crop_diversity == 0.5

    def test_to_dict(self):
        b = Biosphere(biomass=0.12345678, soil_health=0.99, crop_diversity=0.01)
        d = b.to_dict()
        assert d["biomass"] == 0.1235  # rounded to 4
        assert d["soil_health"] == 0.99
        assert d["crop_diversity"] == 0.01

    def test_clone_independent(self):
        b = Biosphere(biomass=0.5)
        c = b.clone()
        c.biomass = 0.9
        assert b.biomass == 0.5


# ── Helper functions ──────────────────────────────────────────────

class TestHelpers:
    def test_light_factor_no_power(self):
        assert _light_factor(0.0) == pytest.approx(MARS_AMBIENT_LIGHT)

    def test_light_factor_full_power(self):
        result = _light_factor(1.0)
        assert result >= MARS_AMBIENT_LIGHT
        assert result <= 1.0

    def test_light_factor_below_threshold(self):
        assert _light_factor(MIN_POWER_FOR_LIGHTS - 0.01) == pytest.approx(MARS_AMBIENT_LIGHT)

    def test_farming_intensity_no_farmers(self):
        assert _farming_intensity(0, 10, 0.5) == 0.0

    def test_farming_intensity_no_colonists(self):
        assert _farming_intensity(0, 0, 0.5) == 0.0

    def test_farming_intensity_clamped(self):
        result = _farming_intensity(10, 10, 1.0)
        assert 0.0 <= result <= 1.0

    def test_farming_intensity_increases_with_skill(self):
        low = _farming_intensity(1, 10, 0.1)
        high = _farming_intensity(1, 10, 0.9)
        assert high > low


# ── Bounds invariants ─────────────────────────────────────────────

class TestBoundsInvariants:
    """All biosphere fields must stay in [0, 1] under any input."""

    @pytest.mark.parametrize("water", [0.0, 0.05, 0.5, 1.0])
    @pytest.mark.parametrize("power", [0.0, 0.5, 1.0])
    def test_bounds_maintained(self, water, power):
        bio = Biosphere(biomass=0.5, soil_health=0.5, crop_diversity=0.5)
        tick_biosphere(bio, water_level=water, power_level=power,
                       farmer_count=3, active_count=10, avg_hydroponics=0.5,
                       rng=random.Random(42))
        assert 0.0 <= bio.biomass <= 1.0
        assert 0.0 <= bio.soil_health <= 1.0
        assert 0.0 <= bio.crop_diversity <= 1.0

    def test_extreme_biomass_high(self):
        bio = Biosphere(biomass=1.0, soil_health=1.0, crop_diversity=1.0)
        tick_biosphere(bio, water_level=1.0, power_level=1.0,
                       farmer_count=10, active_count=10, avg_hydroponics=1.0,
                       has_greenhouse=True, has_research_lab=True,
                       rng=random.Random(42))
        assert bio.biomass <= 1.0

    def test_extreme_biomass_low(self):
        bio = Biosphere(biomass=0.0, soil_health=0.0, crop_diversity=0.0)
        tick_biosphere(bio, water_level=0.0, power_level=0.0,
                       farmer_count=0, active_count=0, avg_hydroponics=0.0,
                       rng=random.Random(42))
        assert bio.biomass >= 0.0


# ── Stress decay ──────────────────────────────────────────────────

class TestStressDecay:
    def test_no_water_causes_decay(self):
        bio = Biosphere(biomass=0.5)
        tick_biosphere(bio, water_level=0.0, power_level=0.5,
                       farmer_count=3, active_count=10, avg_hydroponics=0.5,
                       rng=random.Random(42))
        assert bio.biomass < 0.5

    def test_no_farmers_causes_decay(self):
        bio = Biosphere(biomass=0.5)
        tick_biosphere(bio, water_level=0.5, power_level=0.5,
                       farmer_count=0, active_count=10, avg_hydroponics=0.0,
                       rng=random.Random(42))
        assert bio.biomass < 0.5

    def test_good_conditions_cause_growth(self):
        bio = Biosphere(biomass=0.3, soil_health=0.7, crop_diversity=0.6)
        tick_biosphere(bio, water_level=0.8, power_level=0.8,
                       farmer_count=4, active_count=10, avg_hydroponics=0.7,
                       rng=random.Random(42))
        assert bio.biomass > 0.3


# ── O2 and food output ───────────────────────────────────────────

class TestO2AndFood:
    def test_air_delta_positive_with_biomass(self):
        bio = Biosphere(biomass=0.5, soil_health=0.5)
        delta = tick_biosphere(bio, water_level=0.5, power_level=0.5,
                               farmer_count=3, active_count=10,
                               avg_hydroponics=0.5, rng=random.Random(42))
        assert delta.air_delta > 0

    def test_food_delta_positive_with_biomass(self):
        bio = Biosphere(biomass=0.5, soil_health=0.5)
        delta = tick_biosphere(bio, water_level=0.5, power_level=0.5,
                               farmer_count=3, active_count=10,
                               avg_hydroponics=0.5, rng=random.Random(42))
        assert delta.food_delta > 0

    def test_zero_biomass_zero_output(self):
        bio = Biosphere(biomass=0.0, soil_health=0.5)
        delta = tick_biosphere(bio, water_level=0.0, power_level=0.5,
                               farmer_count=0, active_count=10,
                               avg_hydroponics=0.0, rng=random.Random(42))
        assert delta.air_delta == 0.0
        assert delta.food_delta == 0.0


# ── Blight ────────────────────────────────────────────────────────

class TestBlight:
    def test_blight_only_below_threshold(self):
        """Blight can only happen when diversity < BLIGHT_THRESHOLD."""
        bio = Biosphere(biomass=0.5, crop_diversity=0.5)
        delta = tick_biosphere(bio, water_level=0.5, power_level=0.5,
                               farmer_count=3, active_count=10,
                               avg_hydroponics=0.5, rng=random.Random(42))
        assert not delta.blight_occurred

    def test_blight_devastates_biomass(self):
        """Force blight with a rigged RNG that returns low values."""
        rng = random.Random()
        rng.random = lambda: 0.01  # always triggers blight
        bio = Biosphere(biomass=0.8, crop_diversity=0.1)
        tick_biosphere(bio, water_level=0.5, power_level=0.5,
                       farmer_count=3, active_count=10,
                       avg_hydroponics=0.5, rng=rng)
        assert bio.biomass < 0.8 * (1 - BLIGHT_BIOMASS_LOSS) + 0.1

    def test_no_blight_with_zero_biomass(self):
        """Blight has no effect on zero biomass."""
        rng = random.Random()
        rng.random = lambda: 0.01
        bio = Biosphere(biomass=0.0, crop_diversity=0.1)
        delta = tick_biosphere(bio, water_level=0.5, power_level=0.5,
                               farmer_count=0, active_count=10,
                               avg_hydroponics=0.0, rng=rng)
        assert not delta.blight_occurred


# ── Soil dynamics ─────────────────────────────────────────────────

class TestSoilDynamics:
    def test_farming_degrades_soil(self):
        bio = Biosphere(soil_health=0.8)
        tick_biosphere(bio, water_level=0.5, power_level=0.5,
                       farmer_count=5, active_count=10,
                       avg_hydroponics=0.5, rng=random.Random(42))
        # Net effect depends on farming intensity vs regen rate
        # With high farming, degradation should dominate
        assert bio.soil_health < 0.8

    def test_no_farming_regenerates_soil(self):
        bio = Biosphere(soil_health=0.3)
        tick_biosphere(bio, water_level=0.0, power_level=0.5,
                       farmer_count=0, active_count=10,
                       avg_hydroponics=0.0, rng=random.Random(42))
        assert bio.soil_health > 0.3


# ── Tech integration ──────────────────────────────────────────────

class TestTechIntegration:
    def test_greenhouse_boosts_growth(self):
        bio_base = Biosphere(biomass=0.3, soil_health=0.6, crop_diversity=0.5)
        bio_green = bio_base.clone()
        tick_biosphere(bio_base, water_level=0.7, power_level=0.7,
                       farmer_count=3, active_count=10, avg_hydroponics=0.5,
                       rng=random.Random(42))
        tick_biosphere(bio_green, water_level=0.7, power_level=0.7,
                       farmer_count=3, active_count=10, avg_hydroponics=0.5,
                       has_greenhouse=True, rng=random.Random(42))
        assert bio_green.biomass > bio_base.biomass

    def test_greenhouse_boosts_food(self):
        bio_base = Biosphere(biomass=0.5, soil_health=0.5)
        bio_green = bio_base.clone()
        d1 = tick_biosphere(bio_base, water_level=0.5, power_level=0.5,
                            farmer_count=3, active_count=10, avg_hydroponics=0.5,
                            rng=random.Random(42))
        d2 = tick_biosphere(bio_green, water_level=0.5, power_level=0.5,
                            farmer_count=3, active_count=10, avg_hydroponics=0.5,
                            has_greenhouse=True, rng=random.Random(42))
        assert d2.food_delta > d1.food_delta

    def test_research_lab_boosts_diversity(self):
        bio_base = Biosphere(crop_diversity=0.4)
        bio_lab = bio_base.clone()
        tick_biosphere(bio_base, water_level=0.5, power_level=0.5,
                       farmer_count=3, active_count=10, avg_hydroponics=0.5,
                       rng=random.Random(42))
        tick_biosphere(bio_lab, water_level=0.5, power_level=0.5,
                       farmer_count=3, active_count=10, avg_hydroponics=0.5,
                       has_research_lab=True, rng=random.Random(42))
        assert bio_lab.crop_diversity > bio_base.crop_diversity


# ── EcologyDelta ──────────────────────────────────────────────────

class TestEcologyDelta:
    def test_to_dict_keys(self):
        d = EcologyDelta(air_delta=0.01, food_delta=0.02).to_dict()
        assert set(d.keys()) == {"air_delta", "food_delta", "blight_occurred",
                                  "biomass_before", "biomass_after"}

    def test_records_before_after(self):
        bio = Biosphere(biomass=0.5)
        delta = tick_biosphere(bio, water_level=0.5, power_level=0.5,
                               farmer_count=3, active_count=10,
                               avg_hydroponics=0.5, rng=random.Random(42))
        assert delta.biomass_before == 0.5
        assert delta.biomass_after == bio.biomass


# ── Smoke run ─────────────────────────────────────────────────────

class TestSmokeRun:
    def test_100_year_run_no_crash(self):
        """Run 100 ticks without crashing. The most important test."""
        bio = Biosphere()
        rng = random.Random(42)
        for year in range(100):
            farmer_count = rng.randint(0, 5)
            active = rng.randint(max(1, farmer_count), 10)
            delta = tick_biosphere(
                bio,
                water_level=rng.uniform(0, 1),
                power_level=rng.uniform(0, 1),
                farmer_count=farmer_count,
                active_count=active,
                avg_hydroponics=rng.uniform(0, 1),
                has_greenhouse=(year > 30),
                has_research_lab=(year > 50),
                rng=rng,
            )
            assert 0.0 <= bio.biomass <= 1.0
            assert 0.0 <= bio.soil_health <= 1.0
            assert 0.0 <= bio.crop_diversity <= 1.0
            assert delta.air_delta >= 0.0
            assert delta.food_delta >= 0.0

    def test_colony_death_scenario(self):
        """Simulate total neglect — all values should decay toward 0."""
        bio = Biosphere(biomass=0.8, soil_health=0.8, crop_diversity=0.8)
        rng = random.Random(42)
        for _ in range(50):
            tick_biosphere(bio, water_level=0.0, power_level=0.0,
                           farmer_count=0, active_count=0,
                           avg_hydroponics=0.0, rng=rng)
        assert bio.biomass < 0.05


# ── Calibration ───────────────────────────────────────────────────

class TestCalibration:
    """Verify deltas are meaningful but not dominating vs colony resource scale."""

    def test_max_air_delta_bounded(self):
        """Peak air delta should be < 0.05 (resources are 0-1 scale)."""
        bio = Biosphere(biomass=1.0, soil_health=1.0)
        delta = tick_biosphere(bio, water_level=1.0, power_level=1.0,
                               farmer_count=5, active_count=10,
                               avg_hydroponics=1.0, has_greenhouse=True,
                               rng=random.Random(42))
        assert delta.air_delta < 0.05

    def test_max_food_delta_bounded(self):
        """Peak food delta should be < 0.05."""
        bio = Biosphere(biomass=1.0, soil_health=1.0)
        delta = tick_biosphere(bio, water_level=1.0, power_level=1.0,
                               farmer_count=5, active_count=10,
                               avg_hydroponics=1.0, has_greenhouse=True,
                               rng=random.Random(42))
        assert delta.food_delta < 0.05

    def test_typical_deltas_positive(self):
        """Under typical conditions, both deltas should be meaningfully positive."""
        bio = Biosphere(biomass=0.4, soil_health=0.5, crop_diversity=0.5)
        delta = tick_biosphere(bio, water_level=0.5, power_level=0.5,
                               farmer_count=3, active_count=10,
                               avg_hydroponics=0.5, rng=random.Random(42))
        assert delta.air_delta > 0.001
        assert delta.food_delta > 0.001
