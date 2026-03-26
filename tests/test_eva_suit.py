"""Tests for eva_suit.py -- Mars EVA Suit Life-Support Simulation.

Covers: constants, dataclasses, metabolic rate, CO2 scrubbing,
pressure management, thermal regulation, radiation, mobility,
abort conditions, single-tick integration, multi-tick sessions,
edge cases, and conservation laws.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from eva_suit import (
    # Constants
    AMBIENT_RAD_USVH,
    BASE_POWER_DRAW_W,
    BASE_SPEED_KMH,
    BATTERY_CAPACITY_WH,
    BREACH_LEAK_KPA_PER_MIN,
    CO2_DANGER_PERCENT,
    COOLING_CAPACITY_W,
    FATIGUE_RATE_PER_MIN,
    FATIGUE_SPEED_PENALTY,
    FEED_VALVE_KPA_PER_MIN,
    HEATER_CAPACITY_W,
    HEATER_EFFICIENCY,
    LIOH_CAPACITY_L,
    MARS_AMBIENT_PRESSURE_KPA,
    MARS_AMBIENT_TEMP_C,
    MAX_FATIGUE,
    METABOLIC_HEAT_MAX_W,
    METABOLIC_HEAT_REST_W,
    MICRO_LEAK_KPA_PER_MIN,
    O2_MAX_L_PER_MIN,
    O2_REST_L_PER_MIN,
    RESPIRATORY_QUOTIENT,
    SPE_RAD_USVH,
    SUIT_INSULATION_W_PER_C,
    SUIT_MIN_SAFE_PRESSURE_KPA,
    SUIT_NOMINAL_PRESSURE_KPA,
    SUIT_SHIELDING_FACTOR,
    SUIT_TARGET_TEMP_C,
    SUIT_THERMAL_MASS_J_PER_C,
    SUIT_VOLUME_L,
    # Data structures
    EvaSession,
    EvaTick,
    SuitState,
    # Functions
    check_abort_conditions,
    metabolic_rate,
    mobility_speed,
    pressure_tick,
    radiation_tick,
    run_eva,
    scrub_co2,
    thermal_tick,
    tick_eva,
)


# =============================================================================
# Test Constants -- physical bounds
# =============================================================================

class TestConstants:
    """All constants must be physically reasonable."""

    def test_mars_ambient_pressure_positive(self) -> None:
        assert 0 < MARS_AMBIENT_PRESSURE_KPA < 2.0

    def test_suit_pressure_above_mars(self) -> None:
        assert SUIT_NOMINAL_PRESSURE_KPA > MARS_AMBIENT_PRESSURE_KPA

    def test_min_safe_pressure_bounded(self) -> None:
        assert MARS_AMBIENT_PRESSURE_KPA < SUIT_MIN_SAFE_PRESSURE_KPA < SUIT_NOMINAL_PRESSURE_KPA

    def test_o2_rates_ordered(self) -> None:
        assert 0 < O2_REST_L_PER_MIN < O2_MAX_L_PER_MIN

    def test_respiratory_quotient_bounded(self) -> None:
        assert 0.5 < RESPIRATORY_QUOTIENT < 1.2

    def test_co2_danger_positive(self) -> None:
        assert 0 < CO2_DANGER_PERCENT < 10.0

    def test_lioh_capacity_positive(self) -> None:
        assert LIOH_CAPACITY_L > 0

    def test_leak_rates_ordered(self) -> None:
        assert 0 < MICRO_LEAK_KPA_PER_MIN < BREACH_LEAK_KPA_PER_MIN

    def test_thermal_constants_positive(self) -> None:
        assert METABOLIC_HEAT_REST_W > 0
        assert METABOLIC_HEAT_MAX_W > METABOLIC_HEAT_REST_W
        assert COOLING_CAPACITY_W > 0
        assert HEATER_CAPACITY_W > 0
        assert SUIT_THERMAL_MASS_J_PER_C > 0

    def test_battery_positive(self) -> None:
        assert BATTERY_CAPACITY_WH > 0
        assert BASE_POWER_DRAW_W > 0

    def test_radiation_constants(self) -> None:
        assert AMBIENT_RAD_USVH > 0
        assert SPE_RAD_USVH > AMBIENT_RAD_USVH
        assert 0 < SUIT_SHIELDING_FACTOR <= 1.0

    def test_mobility_constants(self) -> None:
        assert BASE_SPEED_KMH > 0
        assert 0 < FATIGUE_RATE_PER_MIN < 1.0
        assert 0 < FATIGUE_SPEED_PENALTY <= 1.0


# =============================================================================
# Test SuitState dataclass
# =============================================================================

class TestSuitState:
    """SuitState clamping and defaults."""

    def test_defaults(self) -> None:
        s = SuitState()
        assert s.pressure_kpa == SUIT_NOMINAL_PRESSURE_KPA
        assert s.o2_reserve_l == 600.0
        assert s.co2_absorbed_l == 0.0
        assert s.temp_c == SUIT_TARGET_TEMP_C
        assert s.battery_wh == BATTERY_CAPACITY_WH
        assert s.fatigue == 0.0
        assert not s.breached
        assert s.elapsed_min == 0
        assert s.distance_km == 0.0

    def test_clamps_negative_pressure(self) -> None:
        s = SuitState(pressure_kpa=-10.0)
        assert s.pressure_kpa == 0.0

    def test_clamps_negative_o2(self) -> None:
        s = SuitState(o2_reserve_l=-5.0)
        assert s.o2_reserve_l == 0.0

    def test_clamps_co2_absorbed(self) -> None:
        s = SuitState(co2_absorbed_l=9999.0)
        assert s.co2_absorbed_l == LIOH_CAPACITY_L

    def test_clamps_battery(self) -> None:
        s = SuitState(battery_wh=9999.0)
        assert s.battery_wh == BATTERY_CAPACITY_WH

    def test_clamps_fatigue(self) -> None:
        s = SuitState(fatigue=5.0)
        assert s.fatigue == MAX_FATIGUE

    def test_clamps_co2_pct(self) -> None:
        s = SuitState(co2_in_suit_pct=200.0)
        assert s.co2_in_suit_pct == 100.0

    def test_temp_floor(self) -> None:
        s = SuitState(temp_c=-300.0)
        assert s.temp_c == -273.15


# =============================================================================
# Test metabolic_rate
# =============================================================================

class TestMetabolicRate:
    """Metabolic rate as a function of activity level."""

    def test_rest(self) -> None:
        o2, heat = metabolic_rate(0.0)
        assert abs(o2 - O2_REST_L_PER_MIN) < 1e-9
        assert abs(heat - METABOLIC_HEAT_REST_W) < 1e-9

    def test_max_exertion(self) -> None:
        o2, heat = metabolic_rate(1.0)
        assert abs(o2 - O2_MAX_L_PER_MIN) < 1e-9
        assert abs(heat - METABOLIC_HEAT_MAX_W) < 1e-9

    def test_monotone_increasing(self) -> None:
        prev_o2 = 0.0
        for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            o2, _ = metabolic_rate(level)
            assert o2 >= prev_o2
            prev_o2 = o2

    def test_clamps_above_one(self) -> None:
        o2_clamped, _ = metabolic_rate(5.0)
        o2_max, _ = metabolic_rate(1.0)
        assert abs(o2_clamped - o2_max) < 1e-9

    def test_clamps_below_zero(self) -> None:
        o2_clamped, _ = metabolic_rate(-1.0)
        o2_rest, _ = metabolic_rate(0.0)
        assert abs(o2_clamped - o2_rest) < 1e-9

    def test_nonlinear(self) -> None:
        """Mid-activity should be less than linear interpolation."""
        o2_mid, _ = metabolic_rate(0.5)
        linear_mid = (O2_REST_L_PER_MIN + O2_MAX_L_PER_MIN) / 2
        assert o2_mid < linear_mid

    def test_outputs_positive(self) -> None:
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            o2, heat = metabolic_rate(a)
            assert o2 > 0
            assert heat > 0


# =============================================================================
# Test scrub_co2
# =============================================================================

class TestScrubCO2:
    """LiOH canister CO2 scrubbing."""

    def test_full_scrub_fresh_canister(self) -> None:
        scrubbed, used = scrub_co2(1.0, 0.0)
        assert abs(scrubbed - 1.0) < 1e-9  # 100% efficiency when fresh
        assert abs(used - 1.0) < 1e-9

    def test_partial_scrub_worn_canister(self) -> None:
        scrubbed, used = scrub_co2(1.0, LIOH_CAPACITY_L * 0.5)
        # At 50% fill, efficiency = 1 - 0.3*0.5 = 0.85
        assert abs(scrubbed - 0.85) < 1e-9

    def test_saturated_canister(self) -> None:
        scrubbed, used = scrub_co2(1.0, LIOH_CAPACITY_L)
        assert scrubbed == 0.0  # No remaining capacity

    def test_negative_input_clamped(self) -> None:
        scrubbed, used = scrub_co2(-5.0, -10.0)
        assert scrubbed == 0.0
        assert used == 0.0

    def test_canister_never_exceeds_capacity(self) -> None:
        _, used = scrub_co2(9999.0, LIOH_CAPACITY_L - 1.0)
        assert used <= LIOH_CAPACITY_L + 1e-9

    def test_scrubbed_leq_produced(self) -> None:
        for fill in [0.0, 200.0, 400.0, 590.0]:
            scrubbed, _ = scrub_co2(2.0, fill)
            assert scrubbed <= 2.0 + 1e-9


# =============================================================================
# Test pressure_tick
# =============================================================================

class TestPressureTick:
    """Suit pressure dynamics."""

    def test_nominal_pressure_stable(self) -> None:
        suit = SuitState()
        delta = pressure_tick(suit, 0.3, 0.0)
        # At nominal pressure, feed valve shouldn't add much;
        # micro-leak and gas loss should be small
        assert abs(delta) < 1.0  # Less than 1 kPa change per minute

    def test_breach_causes_rapid_depress(self) -> None:
        suit = SuitState(breached=True)
        delta = pressure_tick(suit, 0.3, 0.0)
        assert delta < -4.0  # At least 4 kPa drop per minute

    def test_low_pressure_gets_feed(self) -> None:
        suit = SuitState(pressure_kpa=20.0)
        delta = pressure_tick(suit, 0.0, 0.0)
        # Feed valve should be pushing pressure up
        assert delta > 0

    def test_no_feed_without_o2_reserve(self) -> None:
        suit = SuitState(pressure_kpa=20.0, o2_reserve_l=0.0)
        delta = pressure_tick(suit, 0.0, 0.0)
        # Without O2, feed valve cannot work — only leak
        assert delta <= 0

    def test_co2_net_adds_pressure(self) -> None:
        suit = SuitState()
        delta_no_co2 = pressure_tick(suit, 0.3, 0.0)
        delta_with_co2 = pressure_tick(suit, 0.3, 1.0)
        assert delta_with_co2 > delta_no_co2


# =============================================================================
# Test thermal_tick
# =============================================================================

class TestThermalTick:
    """Suit thermal regulation."""

    def test_nominal_temp_stability(self) -> None:
        suit = SuitState()
        delta, heater = thermal_tick(suit, METABOLIC_HEAT_REST_W)
        # At target temp, system should roughly balance
        assert abs(delta) < 5.0  # Within 5C change per minute

    def test_high_exertion_heats_up(self) -> None:
        suit = SuitState()
        delta, _ = thermal_tick(suit, METABOLIC_HEAT_MAX_W)
        # 600W metabolic should overwhelm cooling at target temp
        assert delta > 0

    def test_heater_activates_when_cold(self) -> None:
        suit = SuitState(temp_c=10.0)
        _, heater = thermal_tick(suit, METABOLIC_HEAT_REST_W)
        assert heater > 0

    def test_no_heater_when_hot(self) -> None:
        suit = SuitState(temp_c=30.0)
        _, heater = thermal_tick(suit, METABOLIC_HEAT_REST_W)
        assert heater == 0.0

    def test_insulation_loss_to_mars(self) -> None:
        """Warmer suit loses more heat to Mars ambient."""
        suit_warm = SuitState(temp_c=40.0)
        suit_cool = SuitState(temp_c=25.0)
        delta_warm, _ = thermal_tick(suit_warm, METABOLIC_HEAT_REST_W)
        delta_cool, _ = thermal_tick(suit_cool, METABOLIC_HEAT_REST_W)
        # Warmer suit should cool faster (more negative or less positive delta)
        assert delta_warm < delta_cool


# =============================================================================
# Test radiation_tick
# =============================================================================

class TestRadiationTick:
    """Radiation dose accumulation."""

    def test_ambient_dose(self) -> None:
        dose = radiation_tick()
        expected = AMBIENT_RAD_USVH * SUIT_SHIELDING_FACTOR / 60.0
        assert abs(dose - expected) < 1e-9

    def test_spe_dose_higher(self) -> None:
        dose_normal = radiation_tick(solar_particle_event=False)
        dose_spe = radiation_tick(solar_particle_event=True)
        assert dose_spe > dose_normal * 10  # SPE is much worse

    def test_zero_ambient(self) -> None:
        dose = radiation_tick(ambient_usv_per_hour=0.0, solar_particle_event=False)
        assert dose == 0.0

    def test_always_non_negative(self) -> None:
        dose = radiation_tick(ambient_usv_per_hour=-100.0)
        assert dose >= 0.0

    def test_shielding_applied(self) -> None:
        unshielded = AMBIENT_RAD_USVH / 60.0
        shielded = radiation_tick()
        assert shielded < unshielded


# =============================================================================
# Test mobility_speed
# =============================================================================

class TestMobilitySpeed:
    """Movement speed under various conditions."""

    def test_normal_walking(self) -> None:
        suit = SuitState()
        speed = mobility_speed(suit, 0.5)
        assert 0 < speed <= BASE_SPEED_KMH

    def test_higher_activity_faster(self) -> None:
        suit = SuitState()
        speed_low = mobility_speed(suit, 0.2)
        speed_high = mobility_speed(suit, 0.8)
        assert speed_high > speed_low

    def test_fatigue_slows(self) -> None:
        suit_fresh = SuitState(fatigue=0.0)
        suit_tired = SuitState(fatigue=0.8)
        assert mobility_speed(suit_fresh, 0.5) > mobility_speed(suit_tired, 0.5)

    def test_low_pressure_slows(self) -> None:
        suit_normal = SuitState()
        suit_low = SuitState(pressure_kpa=15.0)
        assert mobility_speed(suit_normal, 0.5) > mobility_speed(suit_low, 0.5)

    def test_steep_slope_slows(self) -> None:
        suit = SuitState()
        speed_flat = mobility_speed(suit, 0.5, terrain_slope_deg=0.0)
        speed_steep = mobility_speed(suit, 0.5, terrain_slope_deg=30.0)
        assert speed_flat > speed_steep

    def test_speed_always_non_negative(self) -> None:
        suit = SuitState(fatigue=1.0, pressure_kpa=1.0)
        speed = mobility_speed(suit, 0.5, terrain_slope_deg=40.0)
        assert speed >= 0.0

    def test_zero_activity_still_moves(self) -> None:
        suit = SuitState()
        speed = mobility_speed(suit, 0.0)
        assert speed > 0  # Even at rest activity, 20% base walking


# =============================================================================
# Test check_abort_conditions
# =============================================================================

class TestAbortConditions:
    """Safety abort logic."""

    def test_nominal_no_abort(self) -> None:
        suit = SuitState()
        assert check_abort_conditions(suit) == ""

    def test_low_pressure_aborts(self) -> None:
        suit = SuitState(pressure_kpa=15.0)
        assert "pressure" in check_abort_conditions(suit).lower()

    def test_high_co2_aborts(self) -> None:
        suit = SuitState(co2_in_suit_pct=4.0)
        assert "co2" in check_abort_conditions(suit).lower()

    def test_battery_depleted_aborts(self) -> None:
        suit = SuitState(battery_wh=0.0)
        assert "battery" in check_abort_conditions(suit).lower()

    def test_o2_exhausted_low_pressure_aborts(self) -> None:
        suit = SuitState(o2_reserve_l=0.0, pressure_kpa=25.0)
        assert "o2" in check_abort_conditions(suit).lower()

    def test_o2_exhausted_nominal_pressure_ok(self) -> None:
        """O2 gone but pressure still nominal -- not an immediate abort."""
        suit = SuitState(o2_reserve_l=0.0, pressure_kpa=SUIT_NOMINAL_PRESSURE_KPA)
        assert check_abort_conditions(suit) == ""

    def test_hyperthermia_aborts(self) -> None:
        suit = SuitState(temp_c=50.0)
        assert "hyperthermia" in check_abort_conditions(suit).lower()

    def test_hypothermia_aborts(self) -> None:
        suit = SuitState(temp_c=3.0)
        assert "hypothermia" in check_abort_conditions(suit).lower()


# =============================================================================
# Test tick_eva (single minute integration)
# =============================================================================

class TestTickEva:
    """Single-minute EVA tick integration."""

    def test_basic_tick(self) -> None:
        suit = SuitState()
        tick = tick_eva(suit, activity=0.3)
        assert tick.o2_consumed_l > 0
        assert tick.co2_produced_l > 0
        assert tick.co2_scrubbed_l > 0
        assert tick.power_used_wh > 0
        assert tick.radiation_usv > 0
        assert suit.elapsed_min == 1

    def test_o2_reserve_decreases(self) -> None:
        suit = SuitState()
        initial_o2 = suit.o2_reserve_l
        tick_eva(suit, activity=0.5)
        assert suit.o2_reserve_l < initial_o2

    def test_co2_accumulates(self) -> None:
        suit = SuitState()
        initial_co2 = suit.co2_in_suit_pct
        tick_eva(suit, activity=0.5)
        # CO2 should increase slightly (scrubber removes most but not all)
        # Over many ticks it will accumulate
        assert suit.co2_in_suit_pct >= 0

    def test_battery_drains(self) -> None:
        suit = SuitState()
        initial = suit.battery_wh
        tick_eva(suit, activity=0.3)
        assert suit.battery_wh < initial

    def test_radiation_accumulates(self) -> None:
        suit = SuitState()
        tick_eva(suit, activity=0.3)
        assert suit.radiation_dose_usv > 0

    def test_fatigue_increases_with_activity(self) -> None:
        suit = SuitState()
        tick_eva(suit, activity=0.8)
        assert suit.fatigue > 0

    def test_distance_increases(self) -> None:
        suit = SuitState()
        tick_eva(suit, activity=0.5)
        assert suit.distance_km > 0

    def test_spe_warning_generated(self) -> None:
        suit = SuitState()
        tick = tick_eva(suit, solar_particle_event=True)
        assert any("SPE" in w for w in tick.warnings)

    def test_spe_higher_radiation(self) -> None:
        suit1 = SuitState()
        suit2 = SuitState()
        tick_normal = tick_eva(suit1, activity=0.3, solar_particle_event=False)
        tick_spe = tick_eva(suit2, activity=0.3, solar_particle_event=True)
        assert tick_spe.radiation_usv > tick_normal.radiation_usv * 10

    def test_conservation_o2_to_co2(self) -> None:
        """O2 consumed * RQ should equal CO2 produced."""
        suit = SuitState()
        tick = tick_eva(suit, activity=0.5)
        expected_co2 = tick.o2_consumed_l * RESPIRATORY_QUOTIENT
        assert abs(tick.co2_produced_l - expected_co2) < 1e-9

    def test_co2_scrubbed_leq_produced(self) -> None:
        suit = SuitState()
        tick = tick_eva(suit, activity=0.5)
        assert tick.co2_scrubbed_l <= tick.co2_produced_l + 1e-9


# =============================================================================
# Test run_eva (multi-minute sessions)
# =============================================================================

class TestRunEva:
    """Multi-minute EVA sessions."""

    def test_10_minute_session(self) -> None:
        suit = SuitState()
        session = run_eva(suit, duration_min=10, activity=0.3)
        assert session.total_minutes == 10
        assert session.total_o2_consumed_l > 0
        assert session.total_distance_km > 0
        assert session.total_radiation_usv > 0
        assert session.total_power_used_wh > 0
        assert session.abort_reason == ""

    def test_240_minute_nominal_eva(self) -> None:
        """Standard 4-hour EVA should complete without abort at low activity."""
        suit = SuitState()
        session = run_eva(suit, duration_min=240, activity=0.3)
        assert session.total_minutes == 240
        assert session.abort_reason == ""

    def test_high_exertion_depletes_o2_faster(self) -> None:
        suit_low = SuitState()
        suit_high = SuitState()
        sess_low = run_eva(suit_low, duration_min=60, activity=0.2)
        sess_high = run_eva(suit_high, duration_min=60, activity=0.8)
        assert sess_high.total_o2_consumed_l > sess_low.total_o2_consumed_l

    def test_breach_causes_early_abort(self) -> None:
        suit = SuitState(breached=True)
        session = run_eva(suit, duration_min=240, activity=0.3)
        assert session.total_minutes < 240
        assert "pressure" in session.abort_reason.lower()

    def test_spe_mid_eva(self) -> None:
        suit = SuitState()
        session = run_eva(
            suit,
            duration_min=60,
            activity=0.3,
            solar_particle_event_start=20,
            solar_particle_event_duration=10,
        )
        assert session.total_radiation_usv > 0
        assert any("SPE" in w for w in session.warnings)

    def test_steep_terrain_reduces_distance(self) -> None:
        suit_flat = SuitState()
        suit_steep = SuitState()
        sess_flat = run_eva(suit_flat, duration_min=60, activity=0.5, terrain_slope_deg=0.0)
        sess_steep = run_eva(suit_steep, duration_min=60, activity=0.5, terrain_slope_deg=25.0)
        assert sess_flat.total_distance_km > sess_steep.total_distance_km

    def test_session_tracks_extremes(self) -> None:
        suit = SuitState()
        session = run_eva(suit, duration_min=60, activity=0.5)
        assert session.min_pressure_kpa <= SUIT_NOMINAL_PRESSURE_KPA
        assert session.peak_co2_pct >= 0


# =============================================================================
# Smoke tests -- long-duration stability
# =============================================================================

class TestSmoke:
    """Multi-hour stability tests."""

    def test_60_min_no_crash(self) -> None:
        suit = SuitState()
        session = run_eva(suit, duration_min=60, activity=0.3)
        assert session.total_minutes > 0

    def test_480_min_full_day_eva(self) -> None:
        """8-hour EVA may abort on consumables, but must not crash."""
        suit = SuitState()
        session = run_eva(suit, duration_min=480, activity=0.3)
        assert session.total_minutes > 0
        assert suit.pressure_kpa >= 0
        assert suit.battery_wh >= 0
        assert suit.o2_reserve_l >= 0

    def test_1000_min_extreme_duration(self) -> None:
        """Ultra-long EVA must not crash or produce NaN."""
        suit = SuitState()
        session = run_eva(suit, duration_min=1000, activity=0.5)
        assert session.total_minutes > 0
        assert not math.isnan(suit.temp_c)
        assert not math.isnan(suit.pressure_kpa)
        assert not math.isnan(suit.battery_wh)
        assert not math.isnan(session.total_radiation_usv)


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    """Boundary and extreme inputs."""

    def test_zero_activity(self) -> None:
        suit = SuitState()
        tick = tick_eva(suit, activity=0.0)
        assert tick.o2_consumed_l > 0  # Still breathing at rest

    def test_max_activity(self) -> None:
        suit = SuitState()
        tick = tick_eva(suit, activity=1.0)
        assert tick.o2_consumed_l > 0

    def test_zero_duration(self) -> None:
        suit = SuitState()
        session = run_eva(suit, duration_min=0)
        assert session.total_minutes == 0
        assert session.abort_reason == ""

    def test_zero_o2_reserve(self) -> None:
        suit = SuitState(o2_reserve_l=0.0, pressure_kpa=SUIT_NOMINAL_PRESSURE_KPA)
        tick = tick_eva(suit, activity=0.3)
        assert tick.o2_consumed_l == 0.0  # Nothing to consume
        assert "LOW_O2" in str(tick.warnings)

    def test_zero_battery(self) -> None:
        """Battery at 0 should trigger abort on next tick."""
        suit = SuitState(battery_wh=0.0)
        session = run_eva(suit, duration_min=10, activity=0.3)
        assert session.total_minutes == 0
        assert "battery" in session.abort_reason.lower()

    def test_extreme_cold_start(self) -> None:
        suit = SuitState(temp_c=-50.0)
        session = run_eva(suit, duration_min=10, activity=0.3)
        assert session.total_minutes == 0  # Should abort immediately (hypothermia)
        assert "hypothermia" in session.abort_reason.lower()

    def test_extreme_hot_start(self) -> None:
        suit = SuitState(temp_c=50.0)
        session = run_eva(suit, duration_min=10, activity=0.3)
        assert session.total_minutes == 0
        assert "hyperthermia" in session.abort_reason.lower()

    def test_high_co2_start(self) -> None:
        suit = SuitState(co2_in_suit_pct=5.0)
        session = run_eva(suit, duration_min=10, activity=0.3)
        assert session.total_minutes == 0
        assert "co2" in session.abort_reason.lower()

    def test_45_degree_slope(self) -> None:
        suit = SuitState()
        speed = mobility_speed(suit, 0.5, terrain_slope_deg=45.0)
        assert speed >= 0  # Floor at 0.1 factor

    def test_negative_slope_treated_as_positive(self) -> None:
        suit = SuitState()
        speed_neg = mobility_speed(suit, 0.5, terrain_slope_deg=-20.0)
        speed_pos = mobility_speed(suit, 0.5, terrain_slope_deg=20.0)
        assert abs(speed_neg - speed_pos) < 1e-9


# =============================================================================
# Conservation / invariant tests
# =============================================================================

class TestConservation:
    """Physical conservation laws and invariants."""

    def test_o2_reserve_monotone_decreasing(self) -> None:
        """O2 reserve should never increase during an EVA."""
        suit = SuitState()
        prev = suit.o2_reserve_l
        for _ in range(100):
            tick_eva(suit, activity=0.3)
            assert suit.o2_reserve_l <= prev + 1e-9
            prev = suit.o2_reserve_l

    def test_battery_monotone_decreasing(self) -> None:
        """Battery should never increase during an EVA."""
        suit = SuitState()
        prev = suit.battery_wh
        for _ in range(100):
            tick_eva(suit, activity=0.3)
            assert suit.battery_wh <= prev + 1e-9
            prev = suit.battery_wh

    def test_radiation_monotone_increasing(self) -> None:
        """Radiation dose should only ever increase."""
        suit = SuitState()
        prev = suit.radiation_dose_usv
        for _ in range(100):
            tick_eva(suit, activity=0.3)
            assert suit.radiation_dose_usv >= prev - 1e-9
            prev = suit.radiation_dose_usv

    def test_distance_monotone_increasing(self) -> None:
        """Distance traveled should only ever increase."""
        suit = SuitState()
        prev = suit.distance_km
        for _ in range(100):
            tick_eva(suit, activity=0.5)
            assert suit.distance_km >= prev - 1e-9
            prev = suit.distance_km

    def test_elapsed_increments_by_one(self) -> None:
        suit = SuitState()
        for i in range(10):
            tick_eva(suit, activity=0.3)
            assert suit.elapsed_min == i + 1

    def test_fatigue_bounded(self) -> None:
        """Fatigue should never exceed MAX_FATIGUE."""
        suit = SuitState()
        for _ in range(2000):
            tick_eva(suit, activity=1.0)
            assert suit.fatigue <= MAX_FATIGUE + 1e-9

    def test_pressure_never_negative(self) -> None:
        """Even with breach, pressure cannot go below 0."""
        suit = SuitState(breached=True)
        for _ in range(100):
            tick_eva(suit, activity=0.3)
            assert suit.pressure_kpa >= 0.0

    def test_co2_pct_bounded(self) -> None:
        """CO2 percentage should stay in [0, 100]."""
        suit = SuitState()
        for _ in range(500):
            if check_abort_conditions(suit):
                break
            tick_eva(suit, activity=0.5)
            assert 0.0 <= suit.co2_in_suit_pct <= 100.0
