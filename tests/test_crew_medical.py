"""
Tests for crew_medical.py — Mars colony crew health and medical system.

Coverage:
  - Radiation damage (chronic and acute thresholds)
  - Nutrition damage (calorie deficit, grace period, starvation)
  - Oxygen damage (hypoxia onset, impairment, lethal)
  - CO₂ damage (safe → headache → dangerous → lethal)
  - Psychological damage (morale, isolation, crowding)
  - Injury probability (base, storm, EVA multipliers)
  - Healing rates (treated vs natural)
  - Medical tick integration (all stressors combined)
  - Physical bounds (health always in [0,1], non-negative counts)
  - Multi-sol smoke tests (10/100 sols without crash)
  - Edge cases (zero population, extreme inputs)
  - Conservation (crew counts sum to population)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crew_medical import (
    CrewHealth,
    MedicalReport,
    tick_medical,
    radiation_damage,
    nutrition_damage,
    oxygen_damage,
    co2_damage,
    psychological_damage,
    injury_chance,
    healing_rate,
    RAD_CAREER_LIMIT_MSV,
    RAD_NAUSEA_THRESHOLD_MSV,
    RAD_LETHAL_MSV,
    O2_NORMAL_KPA,
    O2_HYPOXIA_ONSET_KPA,
    O2_IMPAIRMENT_KPA,
    O2_LETHAL_KPA,
    CO2_SAFE_KPA,
    CO2_HEADACHE_KPA,
    CO2_DANGEROUS_KPA,
    CO2_LETHAL_KPA,
    CALORIES_DAILY_NEED,
    CALORIE_DEFICIT_GRACE_SOLS,
    BASE_INJURY_RATE,
    STORM_INJURY_MULTIPLIER,
    EVA_INJURY_MULTIPLIER,
    NATURAL_HEALING_RATE,
    TREATED_HEALING_RATE,
    CRITICAL_THRESHOLD,
    ISOLATION_STRESS_ONSET_SOLS,
    CROWDING_THRESHOLD,
)


# ===================================================================
# Radiation damage tests
# ===================================================================

class TestRadiationDamage:
    """Verify radiation damage thresholds match NASA data."""

    def test_zero_dose_zero_damage(self):
        assert radiation_damage(0.0, 0.0) == 0.0

    def test_normal_dose_below_career_limit(self):
        """0.67 mSv/sol (Mars surface baseline) should cause zero chronic damage."""
        damage = radiation_damage(0.67, 100.0)
        assert damage == 0.0

    def test_chronic_above_career_limit(self):
        """Cumulative dose above 600 mSv → chronic damage."""
        damage = radiation_damage(0.67, 800.0)
        assert damage > 0

    def test_acute_mild(self):
        """Single sol >10 mSv → mild acute damage."""
        damage = radiation_damage(50.0, 100.0)
        assert damage > 0

    def test_acute_nausea_threshold(self):
        """700+ mSv single event → significant acute damage."""
        damage = radiation_damage(800.0, 100.0)
        assert damage >= 0.15

    def test_acute_lethal(self):
        """4000+ mSv → near-lethal damage."""
        damage = radiation_damage(5000.0, 100.0)
        assert damage >= 0.5

    def test_damage_capped_at_one(self):
        damage = radiation_damage(100000.0, 100000.0)
        assert damage <= 1.0

    def test_negative_dose_clamped(self):
        damage = radiation_damage(-10.0, -10.0)
        assert damage == 0.0

    def test_monotonic_with_cumulative(self):
        """More cumulative dose → more damage."""
        d1 = radiation_damage(0.67, 500.0)
        d2 = radiation_damage(0.67, 1000.0)
        assert d2 >= d1


# ===================================================================
# Nutrition damage tests
# ===================================================================

class TestNutritionDamage:
    """Verify nutrition damage tracks caloric deficits."""

    def test_adequate_food_no_damage(self):
        damage = nutrition_damage(2500.0 * 100, 100, 0)
        assert damage == 0.0

    def test_excess_food_no_damage(self):
        damage = nutrition_damage(5000.0 * 100, 100, 0)
        assert damage == 0.0

    def test_zero_food_causes_damage(self):
        damage = nutrition_damage(0.0, 100, 10)
        assert damage > 0

    def test_deficit_within_grace_period(self):
        """Within 30-sol grace, damage is mild."""
        damage = nutrition_damage(1000.0 * 100, 100, 10)
        assert 0 < damage < 0.01

    def test_deficit_beyond_grace_period(self):
        """After 30 sols, damage escalates."""
        d_early = nutrition_damage(1000.0 * 100, 100, 10)
        d_late = nutrition_damage(1000.0 * 100, 100, 60)
        assert d_late > d_early

    def test_total_starvation_severe(self):
        """90 sols of zero food → severe damage."""
        damage = nutrition_damage(0.0, 100, 90)
        assert damage >= 0.04

    def test_zero_population_no_damage(self):
        damage = nutrition_damage(0.0, 0, 10)
        assert damage == 0.0

    def test_negative_calories_clamped(self):
        damage = nutrition_damage(-1000.0, 100, 10)
        assert damage > 0  # treated as zero calories


# ===================================================================
# Oxygen damage tests
# ===================================================================

class TestOxygenDamage:
    """Verify hypoxia thresholds match medical data."""

    def test_normal_o2_no_damage(self):
        assert oxygen_damage(21.3) == 0.0

    def test_above_threshold_no_damage(self):
        assert oxygen_damage(18.0) == 0.0

    def test_mild_hypoxia(self):
        """Between 14-16 kPa → mild damage."""
        damage = oxygen_damage(15.0)
        assert 0 < damage <= 0.02

    def test_impairment_zone(self):
        """Between 10-14 kPa → significant damage."""
        damage = oxygen_damage(12.0)
        assert damage > 0.02

    def test_lethal_o2(self):
        """Below 10 kPa → severe damage."""
        damage = oxygen_damage(5.0)
        assert damage >= 0.2

    def test_zero_o2(self):
        damage = oxygen_damage(0.0)
        assert damage == 0.5

    def test_monotonic_decreasing(self):
        """Lower O₂ → more damage."""
        d_high = oxygen_damage(15.0)
        d_low = oxygen_damage(12.0)
        assert d_low >= d_high

    def test_negative_clamped(self):
        damage = oxygen_damage(-5.0)
        assert damage == 0.5  # treated as 0 kPa


# ===================================================================
# CO₂ damage tests
# ===================================================================

class TestCO2Damage:
    """Verify CO₂ toxicity thresholds."""

    def test_safe_co2_no_damage(self):
        assert co2_damage(0.04) == 0.0

    def test_at_safe_limit_no_damage(self):
        assert co2_damage(CO2_SAFE_KPA) == 0.0

    def test_headache_zone(self):
        """0.5-2.0 kPa → mild damage."""
        damage = co2_damage(1.0)
        assert 0 < damage <= 0.01

    def test_dangerous_zone(self):
        """2.0-5.0 kPa → moderate damage."""
        damage = co2_damage(3.0)
        assert 0.01 < damage < 0.10

    def test_very_dangerous_zone(self):
        """5.0-8.0 kPa → severe damage."""
        damage = co2_damage(6.0)
        assert damage >= 0.10

    def test_lethal_co2(self):
        """Above 8 kPa → near-lethal."""
        damage = co2_damage(10.0)
        assert damage >= 0.5

    def test_monotonic_increasing(self):
        """Higher CO₂ → more damage."""
        d_low = co2_damage(1.0)
        d_high = co2_damage(5.0)
        assert d_high > d_low

    def test_zero_co2_no_damage(self):
        assert co2_damage(0.0) == 0.0


# ===================================================================
# Psychological damage tests
# ===================================================================

class TestPsychDamage:
    """Verify psychological stress factors."""

    def test_perfect_conditions_minimal(self):
        damage = psychological_damage(1.0, 100, 50, 5000.0)
        assert damage < 0.005

    def test_low_morale_damage(self):
        damage = psychological_damage(0.2, 100, 50, 5000.0)
        assert damage > 0

    def test_long_isolation_damage(self):
        damage = psychological_damage(0.7, 500, 50, 5000.0)
        assert damage > 0

    def test_crowding_damage(self):
        """100 people in 1000 m² = 10 m²/person (below 25 threshold)."""
        damage = psychological_damage(0.7, 100, 100, 1000.0)
        assert damage > 0

    def test_no_crowding_above_threshold(self):
        """50 m²/person → no crowding stress."""
        d_spacious = psychological_damage(0.7, 100, 50, 5000.0)
        d_cramped = psychological_damage(0.7, 100, 50, 500.0)
        assert d_cramped > d_spacious

    def test_damage_capped(self):
        """Even worst case, psych damage capped at 0.1."""
        damage = psychological_damage(0.0, 2000, 500, 100.0)
        assert damage <= 0.1

    def test_zero_population_no_crowding(self):
        damage = psychological_damage(0.7, 100, 0, 0.0)
        assert damage >= 0  # no crash


# ===================================================================
# Injury probability tests
# ===================================================================

class TestInjuryChance:
    """Verify injury probability calculations."""

    def test_base_rate(self):
        rate = injury_chance(BASE_INJURY_RATE, False, False)
        assert rate == BASE_INJURY_RATE

    def test_storm_multiplier(self):
        rate = injury_chance(BASE_INJURY_RATE, True, False)
        assert rate == BASE_INJURY_RATE * STORM_INJURY_MULTIPLIER

    def test_eva_multiplier(self):
        rate = injury_chance(BASE_INJURY_RATE, False, True)
        assert rate == BASE_INJURY_RATE * EVA_INJURY_MULTIPLIER

    def test_both_multipliers(self):
        rate = injury_chance(BASE_INJURY_RATE, True, True)
        expected = BASE_INJURY_RATE * STORM_INJURY_MULTIPLIER * EVA_INJURY_MULTIPLIER
        assert abs(rate - expected) < 1e-10

    def test_capped_at_one(self):
        rate = injury_chance(0.9, True, True)
        assert rate <= 1.0

    def test_negative_clamped(self):
        rate = injury_chance(-0.1, False, False)
        assert rate == 0.0


# ===================================================================
# Healing rate tests
# ===================================================================

class TestHealingRate:
    """Verify healing rate calculations."""

    def test_with_supplies(self):
        rate = healing_rate(True)
        assert rate == TREATED_HEALING_RATE

    def test_without_supplies(self):
        rate = healing_rate(False)
        assert rate == NATURAL_HEALING_RATE

    def test_treated_faster_than_natural(self):
        assert TREATED_HEALING_RATE > NATURAL_HEALING_RATE


# ===================================================================
# CrewHealth dataclass tests
# ===================================================================

class TestCrewHealth:
    """Verify CrewHealth dataclass constraints."""

    def test_defaults(self):
        h = CrewHealth()
        assert h.avg_health == 1.0
        assert h.min_health == 1.0
        assert h.medical_supplies_kg == 500.0

    def test_clamped_above_one(self):
        h = CrewHealth(avg_health=1.5, min_health=2.0)
        assert h.avg_health == 1.0
        assert h.min_health == 1.0

    def test_clamped_below_zero(self):
        h = CrewHealth(avg_health=-0.5, min_health=-1.0)
        assert h.avg_health == 0.0
        assert h.min_health == 0.0

    def test_negative_counts_clamped(self):
        h = CrewHealth(healthy_count=-5, injured_count=-3)
        assert h.healthy_count == 0
        assert h.injured_count == 0

    def test_negative_supplies_clamped(self):
        h = CrewHealth(medical_supplies_kg=-100)
        assert h.medical_supplies_kg == 0.0


# ===================================================================
# tick_medical integration tests
# ===================================================================

class TestTickMedical:
    """Test the main medical tick function."""

    def _healthy_conditions(self, population: int = 100) -> dict:
        """Return kwargs for a healthy colony."""
        return dict(
            population=population,
            radiation_msv_sol=0.67,
            o2_kpa=21.3,
            co2_kpa=0.04,
            calories_available=CALORIES_DAILY_NEED * population,
            morale=0.8,
            habitat_area_m2=population * 50,
        )

    def test_healthy_conditions_stable(self):
        """Under perfect conditions, health should remain near 1.0."""
        crew = CrewHealth()
        crew.healthy_count = 100
        report = tick_medical(crew, **self._healthy_conditions())
        assert crew.avg_health > 0.95

    def test_zero_population_no_crash(self):
        crew = CrewHealth()
        report = tick_medical(
            crew, population=0, radiation_msv_sol=0.0,
            o2_kpa=21.3, co2_kpa=0.04, calories_available=0.0,
            morale=0.5,
        )
        assert crew.healthy_count == 0
        assert crew.injured_count == 0
        assert crew.critical_count == 0

    def test_radiation_spike_damages_health(self):
        crew = CrewHealth()
        crew.healthy_count = 100
        conditions = self._healthy_conditions()
        conditions["radiation_msv_sol"] = 100.0
        report = tick_medical(crew, **conditions)
        assert report.rad_damage > 0

    def test_food_shortage_damages_health(self):
        crew = CrewHealth()
        crew.healthy_count = 100
        crew.calorie_deficit_sols = 40
        conditions = self._healthy_conditions()
        conditions["calories_available"] = 0.0
        report = tick_medical(crew, **conditions)
        assert report.nutrition_damage > 0

    def test_low_o2_damages_health(self):
        crew = CrewHealth()
        crew.healthy_count = 100
        conditions = self._healthy_conditions()
        conditions["o2_kpa"] = 12.0
        report = tick_medical(crew, **conditions)
        assert report.o2_damage > 0

    def test_high_co2_damages_health(self):
        crew = CrewHealth()
        crew.healthy_count = 100
        conditions = self._healthy_conditions()
        conditions["co2_kpa"] = 3.0
        report = tick_medical(crew, **conditions)
        assert report.co2_damage > 0

    def test_dust_storm_increases_injury(self):
        crew = CrewHealth()
        crew.healthy_count = 100
        conditions = self._healthy_conditions()
        report_calm = tick_medical(crew, **conditions)

        crew2 = CrewHealth()
        crew2.healthy_count = 100
        conditions["dust_storm_active"] = True
        report_storm = tick_medical(crew2, **conditions)
        assert report_storm.injury_damage > report_calm.injury_damage

    def test_medical_supplies_consumed(self):
        crew = CrewHealth(medical_supplies_kg=100.0, injured_count=10)
        conditions = self._healthy_conditions()
        report = tick_medical(crew, **conditions)
        assert crew.medical_supplies_kg < 100.0

    def test_no_supplies_still_heals(self):
        """Natural healing works even without supplies."""
        crew = CrewHealth(avg_health=0.5, medical_supplies_kg=0.0)
        crew.healthy_count = 50
        crew.injured_count = 50
        conditions = self._healthy_conditions()
        report = tick_medical(crew, **conditions)
        assert report.healing > 0

    def test_isolation_tracked(self):
        crew = CrewHealth(isolation_sols=0)
        crew.healthy_count = 100
        conditions = self._healthy_conditions()
        tick_medical(crew, **conditions)
        assert crew.isolation_sols == 1
        tick_medical(crew, **conditions)
        assert crew.isolation_sols == 2

    def test_radiation_accumulates(self):
        crew = CrewHealth(cumulative_rad_msv=0.0)
        crew.healthy_count = 100
        conditions = self._healthy_conditions()
        conditions["radiation_msv_sol"] = 1.0
        tick_medical(crew, **conditions)
        assert crew.cumulative_rad_msv == 1.0
        tick_medical(crew, **conditions)
        assert crew.cumulative_rad_msv == 2.0

    def test_calorie_deficit_tracked(self):
        crew = CrewHealth(calorie_deficit_sols=0)
        crew.healthy_count = 100
        conditions = self._healthy_conditions()
        conditions["calories_available"] = 0.0
        tick_medical(crew, **conditions)
        assert crew.calorie_deficit_sols == 1

    def test_calorie_surplus_resets_deficit(self):
        crew = CrewHealth(calorie_deficit_sols=20)
        crew.healthy_count = 100
        conditions = self._healthy_conditions()
        tick_medical(crew, **conditions)
        assert crew.calorie_deficit_sols == 0


# ===================================================================
# Physical bounds / invariants
# ===================================================================

class TestPhysicalBounds:
    """All outputs must be physically realistic."""

    def test_health_always_in_range(self):
        crew = CrewHealth()
        crew.healthy_count = 100
        for _ in range(100):
            tick_medical(
                crew, population=100, radiation_msv_sol=5.0,
                o2_kpa=15.0, co2_kpa=1.0, calories_available=200000.0,
                morale=0.5,
            )
        assert 0.0 <= crew.avg_health <= 1.0
        assert 0.0 <= crew.min_health <= 1.0

    def test_counts_sum_to_population(self):
        crew = CrewHealth()
        crew.healthy_count = 100
        pop = 100
        tick_medical(
            crew, population=pop, radiation_msv_sol=0.67,
            o2_kpa=21.3, co2_kpa=0.04, calories_available=250000.0,
            morale=0.8,
        )
        total = crew.healthy_count + crew.injured_count + crew.critical_count
        assert total == pop

    def test_counts_nonneg(self):
        crew = CrewHealth()
        crew.healthy_count = 50
        tick_medical(
            crew, population=50, radiation_msv_sol=100.0,
            o2_kpa=12.0, co2_kpa=3.0, calories_available=0.0,
            morale=0.1,
        )
        assert crew.healthy_count >= 0
        assert crew.injured_count >= 0
        assert crew.critical_count >= 0

    def test_supplies_never_negative(self):
        crew = CrewHealth(medical_supplies_kg=1.0, injured_count=100)
        for _ in range(50):
            tick_medical(
                crew, population=100, radiation_msv_sol=10.0,
                o2_kpa=15.0, co2_kpa=1.0, calories_available=100000.0,
                morale=0.3,
            )
        assert crew.medical_supplies_kg >= 0.0

    def test_report_fields_nonneg(self):
        crew = CrewHealth()
        crew.healthy_count = 100
        report = tick_medical(
            crew, population=100, radiation_msv_sol=0.67,
            o2_kpa=21.3, co2_kpa=0.04, calories_available=250000.0,
            morale=0.8,
        )
        assert report.rad_damage >= 0
        assert report.nutrition_damage >= 0
        assert report.o2_damage >= 0
        assert report.co2_damage >= 0
        assert report.psych_damage >= 0
        assert report.injury_damage >= 0
        assert report.healing >= 0
        assert report.supplies_used_kg >= 0
        assert report.deaths_medical >= 0


# ===================================================================
# Multi-sol smoke tests
# ===================================================================

class TestMultiSolSmoke:
    """Run simulation without crash."""

    def test_10_sols_healthy(self):
        crew = CrewHealth()
        crew.healthy_count = 100
        for _ in range(10):
            tick_medical(
                crew, population=100, radiation_msv_sol=0.67,
                o2_kpa=21.3, co2_kpa=0.04,
                calories_available=250000.0, morale=0.8,
            )
        assert crew.avg_health > 0.8

    def test_100_sols_degrading(self):
        """100 sols of moderate stress — health degrades but colony survives."""
        crew = CrewHealth()
        crew.healthy_count = 100
        for sol in range(100):
            tick_medical(
                crew, population=100, radiation_msv_sol=2.0,
                o2_kpa=18.0, co2_kpa=0.3,
                calories_available=200000.0, morale=0.5,
                habitat_area_m2=2000.0,
            )
        assert crew.avg_health > 0.0  # survived
        assert crew.cumulative_rad_msv == 200.0

    def test_100_sols_random_conditions(self):
        """100 sols with randomized conditions — no crash."""
        import random
        rng = random.Random(42)
        crew = CrewHealth()
        crew.healthy_count = 80
        for _ in range(100):
            tick_medical(
                crew, population=80,
                radiation_msv_sol=rng.uniform(0, 5),
                o2_kpa=rng.uniform(14, 22),
                co2_kpa=rng.uniform(0, 2),
                calories_available=rng.uniform(100000, 250000),
                morale=rng.uniform(0.3, 0.9),
                habitat_area_m2=rng.uniform(1000, 5000),
                dust_storm_active=rng.random() < 0.1,
                eva_active=rng.random() < 0.2,
            )
        assert 0 <= crew.avg_health <= 1.0

    def test_365_sols_full_year(self):
        """Full Mars year — the colony endures."""
        crew = CrewHealth()
        crew.healthy_count = 120
        for sol in range(365):
            tick_medical(
                crew, population=120, radiation_msv_sol=0.67,
                o2_kpa=21.0, co2_kpa=0.1,
                calories_available=300000.0, morale=0.7,
                habitat_area_m2=6000.0,
            )
        assert crew.avg_health > 0.5
        assert crew.isolation_sols == 365
        assert abs(crew.cumulative_rad_msv - 0.67 * 365) < 0.01


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Boundary condition tests."""

    def test_single_person_colony(self):
        crew = CrewHealth()
        crew.healthy_count = 1
        report = tick_medical(
            crew, population=1, radiation_msv_sol=0.67,
            o2_kpa=21.3, co2_kpa=0.04, calories_available=2500.0,
            morale=0.7,
        )
        total = crew.healthy_count + crew.injured_count + crew.critical_count
        assert total == 1

    def test_huge_population(self):
        crew = CrewHealth()
        crew.healthy_count = 10000
        report = tick_medical(
            crew, population=10000, radiation_msv_sol=0.67,
            o2_kpa=21.3, co2_kpa=0.04,
            calories_available=25000000.0, morale=0.8,
            habitat_area_m2=500000.0,
        )
        total = crew.healthy_count + crew.injured_count + crew.critical_count
        assert total == 10000

    def test_all_damage_sources_simultaneous(self):
        """Everything goes wrong at once — colony takes massive damage."""
        crew = CrewHealth()
        crew.healthy_count = 50
        report = tick_medical(
            crew, population=50,
            radiation_msv_sol=100.0,
            o2_kpa=11.0,
            co2_kpa=6.0,
            calories_available=0.0,
            morale=0.1,
            habitat_area_m2=100.0,
            dust_storm_active=True,
            eva_active=True,
        )
        assert report.rad_damage > 0
        assert report.o2_damage > 0
        assert report.co2_damage > 0
        assert report.psych_damage > 0
        assert report.injury_damage > 0
        assert report.health_delta < 0  # net negative

    def test_default_report(self):
        r = MedicalReport()
        assert r.health_delta == 0.0
        assert r.deaths_medical == 0
