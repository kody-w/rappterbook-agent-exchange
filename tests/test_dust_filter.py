"""Tests for dust_filter.py — Mars Habitat Perchlorate Dust Filtration."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dust_filter import (
    FilterState, FilterTickResult,
    dust_ingress, perchlorate_from_dust, pressure_drop_pa,
    fan_power_watts, electrostatic_capture, hepa_capture,
    langmuir_adsorption, airborne_concentration_mg_m3, assess_alert,
    hepa_life_remaining, carbon_life_remaining, tick_filter,
    create_filter_system,
    DUST_PER_EVA_GRAM, PERCHLORATE_FRACTION, HEPA_MAX_LOAD_GRAM,
    HEPA_CLEAN_PRESSURE_DROP_PA, HEPA_LOADED_PRESSURE_DROP_PA,
    HEPA_CAPTURE_RATE, ELECTROSTATIC_CAPTURE, HABITAT_VOLUME_M3,
    CARBON_BED_MASS_GRAM, CARBON_CAPACITY_MG_PER_G,
    PERCHLORATE_SAFE_MG_M3, PERCHLORATE_WARNING_MG_M3,
    PERCHLORATE_DANGER_MG_M3, FAN_BASE_POWER_W,
    SETTLING_RATE_PER_SOL, MAINTENANCE_HEPA_RESTORE,
)


# ── FilterState clamping ───────────────────────────────────────────

class TestFilterState:
    def test_defaults(self):
        s = FilterState()
        assert s.hepa_load_gram == 0.0
        assert s.carbon_adsorbed_mg == 0.0
        assert s.airborne_dust_gram == 0.0
        assert s.alert_level == "nominal"

    def test_hepa_load_clamped_high(self):
        s = FilterState(hepa_load_gram=9999.0)
        assert s.hepa_load_gram == HEPA_MAX_LOAD_GRAM

    def test_hepa_load_clamped_low(self):
        s = FilterState(hepa_load_gram=-10.0)
        assert s.hepa_load_gram == 0.0

    def test_negative_airborne_clamped(self):
        s = FilterState(airborne_dust_gram=-5.0)
        assert s.airborne_dust_gram == 0.0

    def test_negative_perchlorate_clamped(self):
        s = FilterState(airborne_perchlorate_mg=-1.0)
        assert s.airborne_perchlorate_mg == 0.0

    def test_negative_power_clamped(self):
        s = FilterState(power_draw_w=-100.0)
        assert s.power_draw_w == 0.0


# ── dust_ingress ───────────────────────────────────────────────────

class TestDustIngress:
    def test_zero_evas(self):
        assert dust_ingress(0, 2) == 0.0

    def test_one_eva_two_crew(self):
        result = dust_ingress(1, 2)
        assert result == 2 * DUST_PER_EVA_GRAM

    def test_multiple_evas(self):
        result = dust_ingress(3, 2)
        assert result == 6 * DUST_PER_EVA_GRAM

    def test_dust_storm_multiplier(self):
        normal = dust_ingress(1, 2, 1.0)
        storm = dust_ingress(1, 2, 3.0)
        assert storm == 3.0 * normal

    def test_storm_factor_clamped(self):
        result = dust_ingress(1, 2, 10.0)
        expected = dust_ingress(1, 2, 5.0)
        assert result == expected

    def test_negative_evas_clamped(self):
        assert dust_ingress(-3, 2) == 0.0

    def test_negative_crew_clamped(self):
        assert dust_ingress(2, -1) == 0.0

    def test_always_non_negative(self):
        for evas in range(-2, 10):
            for crew in range(-2, 6):
                for storm in [-1.0, 0.0, 1.0, 3.0, 7.0]:
                    assert dust_ingress(evas, crew, storm) >= 0.0


# ── perchlorate_from_dust ──────────────────────────────────────────

class TestPerchlorateFromDust:
    def test_zero_dust(self):
        assert perchlorate_from_dust(0.0) == 0.0

    def test_known_value(self):
        # 100 g dust × 0.007 fraction × 1000 mg/g = 700 mg
        result = perchlorate_from_dust(100.0)
        assert abs(result - 700.0) < 0.01

    def test_proportional(self):
        r1 = perchlorate_from_dust(50.0)
        r2 = perchlorate_from_dust(100.0)
        assert abs(r2 - 2.0 * r1) < 0.01

    def test_negative_dust_clamped(self):
        assert perchlorate_from_dust(-10.0) == 0.0


# ── pressure_drop_pa ──────────────────────────────────────────────

class TestPressureDrop:
    def test_clean_filter(self):
        dp = pressure_drop_pa(0.0)
        assert dp == HEPA_CLEAN_PRESSURE_DROP_PA

    def test_fully_loaded(self):
        dp = pressure_drop_pa(HEPA_MAX_LOAD_GRAM)
        assert dp == HEPA_LOADED_PRESSURE_DROP_PA

    def test_half_loaded(self):
        dp = pressure_drop_pa(HEPA_MAX_LOAD_GRAM / 2.0)
        expected = (HEPA_CLEAN_PRESSURE_DROP_PA
                    + 0.5 * (HEPA_LOADED_PRESSURE_DROP_PA
                             - HEPA_CLEAN_PRESSURE_DROP_PA))
        assert abs(dp - expected) < 0.01

    def test_monotonic_increase(self):
        prev = 0.0
        for load in [0, 200, 500, 1000, 1500, 2000]:
            dp = pressure_drop_pa(float(load))
            assert dp >= prev
            prev = dp

    def test_over_max_clamped(self):
        dp = pressure_drop_pa(HEPA_MAX_LOAD_GRAM * 2)
        assert dp == HEPA_LOADED_PRESSURE_DROP_PA

    def test_negative_clamped(self):
        dp = pressure_drop_pa(-100.0)
        assert dp == HEPA_CLEAN_PRESSURE_DROP_PA


# ── fan_power_watts ────────────────────────────────────────────────

class TestFanPower:
    def test_minimum_power(self):
        p = fan_power_watts(0.0)
        assert p >= FAN_BASE_POWER_W

    def test_increases_with_pressure_drop(self):
        p_low = fan_power_watts(50.0)
        p_high = fan_power_watts(400.0)
        assert p_high > p_low

    def test_always_positive(self):
        for dp in [-10.0, 0.0, 50.0, 200.0, 500.0]:
            assert fan_power_watts(dp) > 0.0

    def test_negative_dp_handled(self):
        p = fan_power_watts(-100.0)
        assert p >= FAN_BASE_POWER_W


# ── electrostatic_capture ──────────────────────────────────────────

class TestElectrostaticCapture:
    def test_zero_dust(self):
        assert electrostatic_capture(0.0) == 0.0

    def test_capture_fraction(self):
        result = electrostatic_capture(100.0)
        assert abs(result - 80.0) < 0.01  # 80% of 100g

    def test_always_less_than_input(self):
        for dust in [1.0, 50.0, 200.0, 1000.0]:
            assert electrostatic_capture(dust) <= dust

    def test_negative_clamped(self):
        assert electrostatic_capture(-10.0) == 0.0


# ── hepa_capture ───────────────────────────────────────────────────

class TestHepaCapture:
    def test_clean_filter_high_capture(self):
        captured = hepa_capture(100.0, 0.0)
        assert captured >= 99.0  # ≥99% for clean filter

    def test_mid_life_still_good(self):
        captured = hepa_capture(100.0, 0.5)
        assert captured >= 99.0  # cake effect helps

    def test_near_death_degrades(self):
        captured_clean = hepa_capture(100.0, 0.0)
        captured_spent = hepa_capture(100.0, 0.99)
        assert captured_spent < captured_clean

    def test_zero_dust(self):
        assert hepa_capture(0.0, 0.5) == 0.0

    def test_negative_dust_clamped(self):
        assert hepa_capture(-10.0, 0.5) == 0.0

    def test_capture_never_exceeds_input(self):
        for dust in [0.1, 1.0, 10.0, 100.0]:
            for load in [0.0, 0.3, 0.6, 0.9, 0.99]:
                assert hepa_capture(dust, load) <= dust + 0.001

    def test_load_fraction_clamped(self):
        # Over 1.0 should not crash
        result = hepa_capture(100.0, 1.5)
        assert result >= 0.0


# ── langmuir_adsorption ───────────────────────────────────────────

class TestLangmuirAdsorption:
    def test_zero_perchlorate(self):
        assert langmuir_adsorption(0.0, 0.0) == 0.0

    def test_fresh_carbon_adsorbs(self):
        result = langmuir_adsorption(100.0, 0.0)
        assert result > 0.0
        assert result <= 100.0

    def test_exhausted_carbon_no_adsorption(self):
        max_cap = CARBON_BED_MASS_GRAM * CARBON_CAPACITY_MG_PER_G
        result = langmuir_adsorption(100.0, max_cap)
        assert result == 0.0

    def test_adsorption_decreases_with_loading(self):
        fresh = langmuir_adsorption(100.0, 0.0)
        half = langmuir_adsorption(100.0, CARBON_BED_MASS_GRAM
                                   * CARBON_CAPACITY_MG_PER_G * 0.5)
        assert half <= fresh

    def test_never_exceeds_input(self):
        result = langmuir_adsorption(5.0, 0.0)
        assert result <= 5.0

    def test_never_exceeds_remaining_capacity(self):
        max_cap = CARBON_BED_MASS_GRAM * CARBON_CAPACITY_MG_PER_G
        nearly_full = max_cap - 1.0
        result = langmuir_adsorption(1000.0, nearly_full)
        assert result <= 1.0 + 0.001

    def test_negative_inputs_clamped(self):
        assert langmuir_adsorption(-10.0, 0.0) == 0.0
        assert langmuir_adsorption(10.0, -5.0) >= 0.0


# ── airborne_concentration ─────────────────────────────────────────

class TestAirborneConcentration:
    def test_zero_mass(self):
        assert airborne_concentration_mg_m3(0.0) == 0.0

    def test_known_value(self):
        # 500 mg in 500 m³ = 1.0 mg/m³
        result = airborne_concentration_mg_m3(500.0, 500.0)
        assert abs(result - 1.0) < 0.001

    def test_always_non_negative(self):
        assert airborne_concentration_mg_m3(-10.0) == 0.0

    def test_small_volume_high_concentration(self):
        result = airborne_concentration_mg_m3(100.0, 1.0)
        assert result == 100.0


# ── assess_alert ───────────────────────────────────────────────────

class TestAssessAlert:
    def test_nominal(self):
        assert assess_alert(0.0) == "nominal"
        assert assess_alert(0.05) == "nominal"

    def test_warning(self):
        assert assess_alert(PERCHLORATE_SAFE_MG_M3) == "warning"
        assert assess_alert(0.3) == "warning"

    def test_danger(self):
        assert assess_alert(PERCHLORATE_WARNING_MG_M3) == "danger"
        assert assess_alert(1.5) == "danger"

    def test_critical(self):
        assert assess_alert(PERCHLORATE_DANGER_MG_M3) == "critical"
        assert assess_alert(10.0) == "critical"


# ── hepa_life_remaining ───────────────────────────────────────────

class TestHepaLife:
    def test_new_filter(self):
        assert hepa_life_remaining(0.0) == 1.0

    def test_half_life(self):
        result = hepa_life_remaining(HEPA_MAX_LOAD_GRAM / 2.0)
        assert abs(result - 0.5) < 0.001

    def test_spent(self):
        assert hepa_life_remaining(HEPA_MAX_LOAD_GRAM) == 0.0

    def test_over_max_clamped(self):
        assert hepa_life_remaining(HEPA_MAX_LOAD_GRAM * 2) == 0.0


# ── carbon_life_remaining ─────────────────────────────────────────

class TestCarbonLife:
    def test_fresh_bed(self):
        assert carbon_life_remaining(0.0) == 1.0

    def test_half_exhausted(self):
        half = CARBON_BED_MASS_GRAM * CARBON_CAPACITY_MG_PER_G * 0.5
        result = carbon_life_remaining(half)
        assert abs(result - 0.5) < 0.001

    def test_fully_exhausted(self):
        full = CARBON_BED_MASS_GRAM * CARBON_CAPACITY_MG_PER_G
        assert carbon_life_remaining(full) == 0.0

    def test_zero_bed_mass(self):
        assert carbon_life_remaining(0.0, 0.0) == 0.0


# ── tick_filter integration ────────────────────────────────────────

class TestTickFilter:
    def test_one_tick_nominal(self):
        state = create_filter_system()
        state, result = tick_filter(state, eva_count=2, crew_per_eva=2)
        assert result.dust_ingress_gram == 4 * DUST_PER_EVA_GRAM
        assert result.dust_captured_gram > 0.0
        assert result.power_consumed_wh > 0.0
        assert state.sols_since_hepa_change == 1

    def test_no_evas_no_ingress(self):
        state = create_filter_system()
        state, result = tick_filter(state, eva_count=0)
        assert result.dust_ingress_gram == 0.0
        assert result.perchlorate_ingress_mg == 0.0

    def test_mass_conservation_dust(self):
        """Dust in = dust captured + dust airborne + dust settled."""
        state = create_filter_system()
        state, result = tick_filter(state, eva_count=3, crew_per_eva=3)
        ingress = result.dust_ingress_gram
        captured = result.dust_captured_gram
        airborne = state.airborne_dust_gram
        # Ingress = captured + airborne + settled (settled is lost)
        # So ingress >= captured + airborne (settled is positive)
        assert ingress >= captured + airborne - 0.01

    def test_perchlorate_never_negative(self):
        state = create_filter_system()
        for _ in range(50):
            state, result = tick_filter(state, eva_count=1, crew_per_eva=1)
        assert state.airborne_perchlorate_mg >= 0.0
        assert result.airborne_perchlorate_mg_m3 >= 0.0

    def test_dust_storm_increases_contamination(self):
        s_normal = create_filter_system()
        s_storm = create_filter_system()
        s_normal, r_normal = tick_filter(s_normal, eva_count=2, dust_storm_factor=1.0)
        s_storm, r_storm = tick_filter(s_storm, eva_count=2, dust_storm_factor=3.0)
        assert r_storm.dust_ingress_gram > r_normal.dust_ingress_gram

    def test_hepa_degrades_over_time(self):
        state = create_filter_system()
        initial_life = 1.0
        for _ in range(100):
            state, result = tick_filter(state, eva_count=4, crew_per_eva=3)
        assert result.hepa_life_fraction < initial_life

    def test_pressure_drop_increases_with_loading(self):
        state = create_filter_system()
        _, r0 = tick_filter(FilterState(), eva_count=1)
        dp_initial = r0.pressure_drop_pa

        state = create_filter_system()
        for _ in range(200):
            state, result = tick_filter(state, eva_count=4, crew_per_eva=3)
        assert result.pressure_drop_pa > dp_initial

    def test_power_increases_with_loading(self):
        s1 = create_filter_system()
        _, r1 = tick_filter(s1, eva_count=1)

        s2 = FilterState(hepa_load_gram=1500.0)
        _, r2 = tick_filter(s2, eva_count=1)
        assert r2.power_consumed_wh > r1.power_consumed_wh

    def test_maintenance_reduces_hepa_load(self):
        state = create_filter_system()
        for _ in range(50):
            state, _ = tick_filter(state, eva_count=4, crew_per_eva=3)
        load_before = state.hepa_load_gram
        state, _ = tick_filter(state, eva_count=0, maintenance=True)
        assert state.hepa_load_gram < load_before

    def test_maintenance_resets_counters(self):
        state = create_filter_system()
        for _ in range(30):
            state, _ = tick_filter(state, eva_count=2)
        assert state.sols_since_hepa_change == 30
        state, _ = tick_filter(state, maintenance=True)
        # After maintenance sol, counter is 1 (incremented after reset)
        assert state.sols_since_hepa_change == 1
        assert state.maintenance_events == 1

    def test_alert_escalation(self):
        """With a spent filter, alerts should escalate."""
        state = FilterState(
            hepa_load_gram=HEPA_MAX_LOAD_GRAM,
            carbon_adsorbed_mg=CARBON_BED_MASS_GRAM * CARBON_CAPACITY_MG_PER_G,
        )
        state, result = tick_filter(state, eva_count=4, crew_per_eva=4)
        assert result.alert in ("warning", "danger", "critical")

    def test_spent_hepa_critical(self):
        state = FilterState(hepa_load_gram=HEPA_MAX_LOAD_GRAM)
        state, result = tick_filter(state, eva_count=0)
        assert result.alert == "critical"

    def test_cumulative_tracking(self):
        state = create_filter_system()
        total_ingress = 0.0
        total_captured = 0.0
        for _ in range(10):
            state, result = tick_filter(state, eva_count=2, crew_per_eva=2)
            total_ingress += result.dust_ingress_gram
            total_captured += result.dust_captured_gram
        assert abs(state.total_dust_ingress_gram - total_ingress) < 0.01
        assert abs(state.total_dust_captured_gram - total_captured) < 0.01


# ── Property-based invariants ──────────────────────────────────────

class TestPhysicalInvariants:
    """Conservation laws and physical bounds that must always hold."""

    def test_capture_never_exceeds_ingress(self):
        """Cannot capture more dust than enters."""
        state = create_filter_system()
        for _ in range(100):
            state, result = tick_filter(state, eva_count=5, crew_per_eva=4,
                                        dust_storm_factor=3.0)
            assert result.dust_captured_gram <= result.dust_ingress_gram + 0.01

    def test_hepa_load_bounded(self):
        """HEPA load never exceeds physical capacity."""
        state = create_filter_system()
        for _ in range(500):
            state, _ = tick_filter(state, eva_count=5, crew_per_eva=4)
        assert state.hepa_load_gram <= HEPA_MAX_LOAD_GRAM

    def test_airborne_always_non_negative(self):
        """No negative mass in the air — ever."""
        state = create_filter_system()
        for i in range(200):
            eva = 4 if i % 3 == 0 else 0
            maint = (i % 50 == 49)
            state, result = tick_filter(state, eva_count=eva,
                                        maintenance=maint)
            assert state.airborne_dust_gram >= 0.0
            assert state.airborne_perchlorate_mg >= 0.0
            assert result.airborne_dust_mg_m3 >= 0.0
            assert result.airborne_perchlorate_mg_m3 >= 0.0

    def test_power_always_positive_when_running(self):
        """Filtration always draws some power."""
        state = create_filter_system()
        for _ in range(50):
            state, result = tick_filter(state, eva_count=2)
            assert result.power_consumed_wh > 0.0

    def test_settling_reduces_airborne(self):
        """Without new ingress, airborne should decrease."""
        state = create_filter_system()
        # Contaminate the air first
        for _ in range(10):
            state, _ = tick_filter(state, eva_count=4, crew_per_eva=3)
        dust_before = state.airborne_dust_gram
        # Now no EVAs — air should clear
        state, _ = tick_filter(state, eva_count=0)
        assert state.airborne_dust_gram < dust_before

    def test_long_run_no_crash(self):
        """Smoke test: 1000 sols with varied inputs, no crash."""
        state = create_filter_system()
        for sol in range(1000):
            eva = sol % 5
            storm = 1.0 + (sol % 100 == 0) * 3.0
            maint = (sol % 100 == 99)
            state, result = tick_filter(state, eva_count=eva,
                                        crew_per_eva=2,
                                        dust_storm_factor=storm,
                                        maintenance=maint)
        # Should still be a valid state
        assert state.hepa_load_gram >= 0.0
        assert state.total_dust_ingress_gram > 0.0
        assert state.maintenance_events >= 9  # at least 9 maintenance cycles

    def test_total_captured_leq_total_ingress(self):
        """Cannot capture more total dust than has ever entered."""
        state = create_filter_system()
        for _ in range(200):
            state, _ = tick_filter(state, eva_count=3, crew_per_eva=2)
        assert state.total_dust_captured_gram <= state.total_dust_ingress_gram + 0.01


# ── create_filter_system ───────────────────────────────────────────

class TestCreateFilterSystem:
    def test_pristine_state(self):
        state = create_filter_system()
        assert state.hepa_load_gram == 0.0
        assert state.carbon_adsorbed_mg == 0.0
        assert state.airborne_dust_gram == 0.0
        assert state.maintenance_events == 0
        assert state.alert_level == "nominal"
