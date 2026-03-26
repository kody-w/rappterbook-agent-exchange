"""Tests for src/greenhouse.py — Mars agricultural module.

65 tests covering:
  - CropBed and Greenhouse dataclasses
  - Light and temperature efficiency curves
  - Crop growth, health dynamics, and harvest
  - Gas exchange (CO2 uptake, O2 release)
  - Water and power demand calculations
  - Food coverage metric
  - Full tick_greenhouse integration
  - Physical invariants (non-negative, bounded health/growth)
  - Smoke tests: full Mars year, starvation, abundance
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
from src.greenhouse import (
    CropBed,
    Greenhouse,
    light_efficiency,
    temperature_efficiency,
    grow_bed,
    harvest_bed,
    greenhouse_gas_exchange,
    greenhouse_water_demand,
    greenhouse_power_demand,
    food_coverage,
    tick_greenhouse,
    create_starter_greenhouse,
    CROP_DATA,
    CALORIES_PER_PERSON_SOL,
    CO2_UPTAKE_KG_M2_SOL,
    O2_RELEASE_KG_M2_SOL,
    WATER_NET_L_M2_SOL,
    LED_KWH_M2_SOL,
    MIN_LIGHT_FRACTION,
    COLD_STRESS_TEMP_C,
)


# ===================================================================
# CropBed dataclass
# ===================================================================

class TestCropBed:
    def test_valid_creation(self) -> None:
        bed = CropBed(crop="lettuce", area_m2=10.0)
        assert bed.crop == "lettuce"
        assert bed.area_m2 == 10.0
        assert bed.day_in_cycle == 0
        assert bed.health == 1.0

    def test_all_crop_types(self) -> None:
        for crop in CROP_DATA:
            bed = CropBed(crop=crop, area_m2=5.0)
            assert bed.cycle_days == CROP_DATA[crop][0]
            assert bed.yield_kg_m2 == CROP_DATA[crop][1]
            assert bed.kcal_per_kg == CROP_DATA[crop][2]

    def test_invalid_crop_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown crop"):
            CropBed(crop="banana", area_m2=5.0)

    def test_negative_area_clamped(self) -> None:
        bed = CropBed(crop="potato", area_m2=-10.0)
        assert bed.area_m2 == 0.0

    def test_health_clamped(self) -> None:
        bed = CropBed(crop="wheat", area_m2=5.0, health=1.5)
        assert bed.health == 1.0
        bed2 = CropBed(crop="wheat", area_m2=5.0, health=-0.5)
        assert bed2.health == 0.0

    def test_growth_fraction(self) -> None:
        bed = CropBed(crop="lettuce", area_m2=5.0, day_in_cycle=14)
        assert bed.growth_fraction() == pytest.approx(14.0 / 28.0)

    def test_growth_fraction_at_maturity(self) -> None:
        bed = CropBed(crop="lettuce", area_m2=5.0, day_in_cycle=28)
        assert bed.growth_fraction() == 1.0

    def test_growth_fraction_over_cycle(self) -> None:
        bed = CropBed(crop="lettuce", area_m2=5.0, day_in_cycle=35)
        assert bed.growth_fraction() == 1.0  # capped

    def test_is_mature(self) -> None:
        bed = CropBed(crop="potato", area_m2=5.0, day_in_cycle=89)
        assert not bed.is_mature()
        bed.day_in_cycle = 90
        assert bed.is_mature()


# ===================================================================
# Greenhouse dataclass
# ===================================================================

class TestGreenhouse:
    def test_empty_greenhouse(self) -> None:
        gh = Greenhouse()
        assert len(gh.beds) == 0
        assert gh.active_area_m2() == 0.0

    def test_add_bed(self) -> None:
        gh = Greenhouse()
        bed = gh.add_bed("lettuce", 10.0)
        assert len(gh.beds) == 1
        assert bed.crop == "lettuce"
        assert gh.active_area_m2() == 10.0

    def test_multiple_beds_area(self) -> None:
        gh = Greenhouse()
        gh.add_bed("lettuce", 10.0)
        gh.add_bed("potato", 20.0)
        assert gh.active_area_m2() == 30.0


# ===================================================================
# light_efficiency
# ===================================================================

class TestLightEfficiency:
    def test_full_light(self) -> None:
        assert light_efficiency(1.0) == 1.0

    def test_saturation_above_06(self) -> None:
        assert light_efficiency(0.6) == 1.0
        assert light_efficiency(0.8) == 1.0

    def test_darkness(self) -> None:
        assert light_efficiency(0.0) == 0.0

    def test_below_min_threshold(self) -> None:
        assert light_efficiency(MIN_LIGHT_FRACTION - 0.01) == 0.0

    def test_midrange(self) -> None:
        eff = light_efficiency(0.4)
        assert 0.0 < eff < 1.0

    def test_monotonic(self) -> None:
        """Efficiency increases with more light."""
        prev = light_efficiency(0.0)
        for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
            curr = light_efficiency(frac)
            assert curr >= prev
            prev = curr


# ===================================================================
# temperature_efficiency
# ===================================================================

class TestTemperatureEfficiency:
    def test_optimal(self) -> None:
        eff = temperature_efficiency(22.0)
        assert eff >= 0.9  # near-peak

    def test_extreme_cold(self) -> None:
        assert temperature_efficiency(-30.0) == 0.0

    def test_cold_stress_boundary(self) -> None:
        assert temperature_efficiency(COLD_STRESS_TEMP_C - 1) == 0.0
        assert temperature_efficiency(COLD_STRESS_TEMP_C + 1) > 0.0

    def test_extreme_heat(self) -> None:
        assert temperature_efficiency(50.0) == 0.0

    def test_moderate_cold(self) -> None:
        eff = temperature_efficiency(0.0)
        assert 0.0 < eff < 1.0

    def test_bounded(self) -> None:
        for temp in range(-50, 60, 5):
            eff = temperature_efficiency(float(temp))
            assert 0.0 <= eff <= 1.0


# ===================================================================
# grow_bed
# ===================================================================

class TestGrowBed:
    def test_normal_growth(self) -> None:
        bed = CropBed(crop="lettuce", area_m2=5.0)
        grow_bed(bed, light_eff=0.8, temp_eff=0.9)
        assert bed.day_in_cycle == 1

    def test_good_conditions_heal(self) -> None:
        bed = CropBed(crop="lettuce", area_m2=5.0, health=0.7)
        grow_bed(bed, light_eff=1.0, temp_eff=1.0)
        assert bed.health > 0.7

    def test_bad_conditions_damage(self) -> None:
        bed = CropBed(crop="lettuce", area_m2=5.0, health=1.0)
        grow_bed(bed, light_eff=0.1, temp_eff=0.1)
        assert bed.health < 1.0

    def test_zero_light_no_growth(self) -> None:
        bed = CropBed(crop="potato", area_m2=5.0)
        grow_bed(bed, light_eff=0.0, temp_eff=1.0)
        assert bed.day_in_cycle == 0  # no growth in darkness

    def test_dead_crop_no_growth(self) -> None:
        bed = CropBed(crop="potato", area_m2=5.0, health=0.05)
        grow_bed(bed, light_eff=0.0, temp_eff=0.0)
        assert bed.day_in_cycle == 0

    def test_health_bounded(self) -> None:
        bed = CropBed(crop="lettuce", area_m2=5.0, health=0.99)
        for _ in range(100):
            grow_bed(bed, light_eff=1.0, temp_eff=1.0)
        assert bed.health <= 1.0


# ===================================================================
# harvest_bed
# ===================================================================

class TestHarvestBed:
    def test_harvest_mature_lettuce(self) -> None:
        bed = CropBed(crop="lettuce", area_m2=10.0, day_in_cycle=28, health=1.0)
        food_kg, cal = harvest_bed(bed)
        expected_kg = 3.5 * 10.0 * 1.0  # 35 kg
        expected_cal = 35.0 * 150.0       # 5250 kcal
        assert food_kg == pytest.approx(expected_kg, rel=0.01)
        assert cal == pytest.approx(expected_cal, rel=0.01)
        assert bed.day_in_cycle == 0  # reset
        assert bed.harvests == 1

    def test_harvest_damaged_crop(self) -> None:
        bed = CropBed(crop="potato", area_m2=10.0, day_in_cycle=90, health=0.5)
        food_kg, cal = harvest_bed(bed)
        expected_kg = 5.0 * 10.0 * 0.5  # 25 kg (50% health)
        assert food_kg == pytest.approx(expected_kg, rel=0.01)

    def test_no_harvest_if_immature(self) -> None:
        bed = CropBed(crop="wheat", area_m2=10.0, day_in_cycle=60)
        food_kg, cal = harvest_bed(bed)
        assert food_kg == 0.0
        assert cal == 0.0

    def test_harvest_increments_counter(self) -> None:
        bed = CropBed(crop="lettuce", area_m2=5.0, day_in_cycle=28)
        harvest_bed(bed)
        assert bed.harvests == 1
        bed.day_in_cycle = 28
        harvest_bed(bed)
        assert bed.harvests == 2


# ===================================================================
# greenhouse_gas_exchange
# ===================================================================

class TestGasExchange:
    def test_basic_exchange(self) -> None:
        gas = greenhouse_gas_exchange(100.0)
        assert gas["co2_absorbed_kg"] == pytest.approx(
            CO2_UPTAKE_KG_M2_SOL * 100.0, rel=0.01)
        assert gas["o2_released_kg"] == pytest.approx(
            O2_RELEASE_KG_M2_SOL * 100.0, rel=0.01)

    def test_zero_area(self) -> None:
        gas = greenhouse_gas_exchange(0.0)
        assert gas["co2_absorbed_kg"] == 0.0
        assert gas["o2_released_kg"] == 0.0

    def test_scales_linearly(self) -> None:
        gas1 = greenhouse_gas_exchange(10.0)
        gas2 = greenhouse_gas_exchange(20.0)
        assert gas2["co2_absorbed_kg"] == pytest.approx(
            2 * gas1["co2_absorbed_kg"], rel=0.01)


# ===================================================================
# greenhouse_water_demand / greenhouse_power_demand
# ===================================================================

class TestDemands:
    def test_water_demand(self) -> None:
        demand = greenhouse_water_demand(50.0)
        assert demand == pytest.approx(WATER_NET_L_M2_SOL * 50.0, rel=0.01)

    def test_power_demand(self) -> None:
        demand = greenhouse_power_demand(50.0)
        assert demand == pytest.approx(LED_KWH_M2_SOL * 50.0, rel=0.01)

    def test_zero_area(self) -> None:
        assert greenhouse_water_demand(0.0) == 0.0
        assert greenhouse_power_demand(0.0) == 0.0


# ===================================================================
# food_coverage
# ===================================================================

class TestFoodCoverage:
    def test_full_coverage(self) -> None:
        cov = food_coverage(CALORIES_PER_PERSON_SOL * 4, crew_count=4)
        assert cov == pytest.approx(1.0)

    def test_half_coverage(self) -> None:
        cov = food_coverage(CALORIES_PER_PERSON_SOL * 2, crew_count=4)
        assert cov == pytest.approx(0.5)

    def test_surplus(self) -> None:
        cov = food_coverage(CALORIES_PER_PERSON_SOL * 8, crew_count=4)
        assert cov == pytest.approx(2.0)

    def test_zero_crew(self) -> None:
        assert food_coverage(1000.0, crew_count=0) == 1.0
        assert food_coverage(0.0, crew_count=0) == 0.0

    def test_no_food(self) -> None:
        assert food_coverage(0.0, crew_count=4) == 0.0


# ===================================================================
# tick_greenhouse — integration
# ===================================================================

class TestTickGreenhouse:
    def test_returns_all_keys(self) -> None:
        gh = create_starter_greenhouse()
        result = tick_greenhouse(gh, solar_flux_fraction=0.7,
                                 greenhouse_temp_c=22.0)
        expected = {
            "light_efficiency", "temperature_efficiency", "active_area_m2",
            "food_harvested_kg", "calories_harvested", "harvests_this_sol",
            "total_food_kg", "total_calories", "co2_absorbed_kg",
            "o2_released_kg", "water_demand_kg", "power_demand_kwh", "beds",
        }
        assert set(result.keys()) == expected

    def test_beds_grow(self) -> None:
        gh = create_starter_greenhouse()
        tick_greenhouse(gh, solar_flux_fraction=0.7, greenhouse_temp_c=22.0)
        for bed in gh.beds:
            assert bed.day_in_cycle >= 1

    def test_lettuce_harvests_at_28(self) -> None:
        """Lettuce should harvest after 28 sols of good conditions."""
        gh = Greenhouse()
        gh.add_bed("lettuce", 10.0)
        total_cal = 0.0
        for _ in range(29):
            result = tick_greenhouse(gh, solar_flux_fraction=0.7,
                                     greenhouse_temp_c=22.0)
            total_cal += result["calories_harvested"]
        # Should have harvested once
        assert gh.total_harvests >= 1
        assert total_cal > 0

    def test_no_harvest_in_darkness(self) -> None:
        """No light = no growth = no harvest."""
        gh = Greenhouse()
        gh.add_bed("lettuce", 10.0)
        for _ in range(50):
            tick_greenhouse(gh, solar_flux_fraction=0.0,
                            greenhouse_temp_c=22.0)
        assert gh.total_harvests == 0

    def test_gas_exchange_positive(self) -> None:
        gh = create_starter_greenhouse()
        result = tick_greenhouse(gh, solar_flux_fraction=0.7,
                                 greenhouse_temp_c=22.0)
        assert result["co2_absorbed_kg"] > 0
        assert result["o2_released_kg"] > 0

    def test_resource_demands_positive(self) -> None:
        gh = create_starter_greenhouse()
        result = tick_greenhouse(gh, solar_flux_fraction=0.7,
                                 greenhouse_temp_c=22.0)
        assert result["water_demand_kg"] > 0
        assert result["power_demand_kwh"] > 0

    def test_cold_stops_growth(self) -> None:
        gh = Greenhouse()
        gh.add_bed("lettuce", 10.0)
        for _ in range(50):
            tick_greenhouse(gh, solar_flux_fraction=0.7,
                            greenhouse_temp_c=-30.0)
        assert gh.total_harvests == 0

    def test_bed_status_in_result(self) -> None:
        gh = create_starter_greenhouse()
        result = tick_greenhouse(gh, solar_flux_fraction=0.7,
                                 greenhouse_temp_c=22.0)
        assert len(result["beds"]) == 4
        for bed_info in result["beds"]:
            assert "crop" in bed_info
            assert "health" in bed_info
            assert "growth" in bed_info


# ===================================================================
# Physical invariants
# ===================================================================

class TestPhysicalInvariants:
    def test_food_never_negative(self) -> None:
        gh = create_starter_greenhouse()
        for _ in range(200):
            result = tick_greenhouse(gh, solar_flux_fraction=0.5,
                                     greenhouse_temp_c=20.0)
            assert result["food_harvested_kg"] >= 0.0
            assert result["total_food_kg"] >= 0.0

    def test_health_bounded(self) -> None:
        gh = create_starter_greenhouse()
        for _ in range(500):
            tick_greenhouse(gh, solar_flux_fraction=0.1,
                            greenhouse_temp_c=5.0)
            for bed in gh.beds:
                assert 0.0 <= bed.health <= 1.0

    def test_growth_fraction_bounded(self) -> None:
        gh = create_starter_greenhouse()
        for _ in range(200):
            tick_greenhouse(gh, solar_flux_fraction=0.7,
                            greenhouse_temp_c=22.0)
            for bed in gh.beds:
                assert 0.0 <= bed.growth_fraction() <= 1.0

    def test_gas_exchange_never_negative(self) -> None:
        gh = create_starter_greenhouse()
        for _ in range(100):
            result = tick_greenhouse(gh, solar_flux_fraction=0.5,
                                     greenhouse_temp_c=22.0)
            assert result["co2_absorbed_kg"] >= 0.0
            assert result["o2_released_kg"] >= 0.0

    def test_demands_never_negative(self) -> None:
        gh = create_starter_greenhouse()
        for _ in range(100):
            result = tick_greenhouse(gh, solar_flux_fraction=0.5,
                                     greenhouse_temp_c=22.0)
            assert result["water_demand_kg"] >= 0.0
            assert result["power_demand_kwh"] >= 0.0


# ===================================================================
# Smoke tests
# ===================================================================

class TestSmoke:
    def test_full_mars_year(self) -> None:
        """668 sols, good conditions, no crash."""
        gh = create_starter_greenhouse()
        for _ in range(668):
            result = tick_greenhouse(gh, solar_flux_fraction=0.6,
                                     greenhouse_temp_c=22.0)
            assert result["total_food_kg"] >= 0.0
        # Should have harvested lettuce ~23 times, potatoes ~7, wheat ~5, soy ~6
        assert gh.total_harvests >= 20

    def test_harsh_year(self) -> None:
        """668 sols, dust storm (low light), cold greenhouse."""
        gh = create_starter_greenhouse()
        for _ in range(668):
            result = tick_greenhouse(gh, solar_flux_fraction=0.1,
                                     greenhouse_temp_c=0.0)
        # Some beds may have grown slowly, but should not crash
        assert result["total_food_kg"] >= 0.0

    def test_perfect_conditions(self) -> None:
        """200 sols, perfect light and temp."""
        gh = create_starter_greenhouse()
        for _ in range(200):
            result = tick_greenhouse(gh, solar_flux_fraction=1.0,
                                     greenhouse_temp_c=22.0)
        assert gh.total_food_kg > 0
        assert gh.total_calories > 0

    def test_empty_greenhouse_no_crash(self) -> None:
        """No beds, 100 sols, nothing happens."""
        gh = Greenhouse()
        for _ in range(100):
            result = tick_greenhouse(gh, solar_flux_fraction=0.7,
                                     greenhouse_temp_c=22.0)
        assert result["active_area_m2"] == 0.0
        assert result["food_harvested_kg"] == 0.0
