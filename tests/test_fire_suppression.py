"""
Tests for fire_suppression.py -- Mars Habitat Fire Detection and Suppression.

75 tests across 12 test classes.  Every physics function, edge case, and
invariant tested.  Fire is the colony's deadliest enemy in a sealed habitat.

Run: python -m pytest tests/test_fire_suppression.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.fire_suppression import (
    FireState, FireMinute, t_squared_hrr, o2_consumed_kg, smoke_density,
    co_concentration, compartment_temp_rise, agent_needed_kg,
    check_smoke_detector, check_heat_detector, tick_fire, create_fire_system,
    NOMINAL_O2_FRACTION, AMBIENT_TEMP_C, FLASHOVER_TEMP_C,
    O2_EXTINCTION_THRESHOLD, T_SQUARED_ALPHA_SLOW, T_SQUARED_ALPHA_MEDIUM,
    T_SQUARED_ALPHA_FAST, O2_ENHANCEMENT_FACTOR,
    HEAT_OF_COMBUSTION_MJ_PER_KG_O2, AIR_DENSITY_KG_PER_M3,
    SMOKE_YIELD_PER_MW, SMOKE_DETECTOR_THRESHOLD_OD,
    HEAT_DETECTOR_RATE_C_PER_MIN, HEAT_DETECTOR_FIXED_TEMP_C,
    CO_BASE_YIELD_PPM_PER_KW, CO_LOW_O2_MULTIPLIER, CO_LETHAL_PPM,
    CO2_AGENT_DENSITY_KG_PER_M3, AGENT_DISCHARGE_RATE_KG_PER_MIN,
    DEFAULT_COMPARTMENT_VOLUME_M3, VENT_O2_SUPPLY_KG_PER_MIN,
    SMOKE_LETHAL_OD,
)


class TestFireState:
    def test_defaults(self) -> None:
        s = FireState()
        assert s.minute == 0
        assert s.fire_active is False
        assert s.o2_fraction == NOMINAL_O2_FRACTION
        assert s.compartment_temp_c == AMBIENT_TEMP_C
        assert s.agent_remaining_kg == 50.0
        assert s.casualties is False

    def test_o2_mass_initialized(self) -> None:
        s = FireState()
        expected = DEFAULT_COMPARTMENT_VOLUME_M3 * AIR_DENSITY_KG_PER_M3 * NOMINAL_O2_FRACTION
        assert abs(s.o2_mass_kg - expected) < 0.01

    def test_clamping_o2_high(self) -> None:
        assert FireState(o2_fraction=2.0).o2_fraction <= 1.0

    def test_clamping_o2_negative(self) -> None:
        assert FireState(o2_fraction=-0.5).o2_fraction >= 0.0

    def test_clamping_agent(self) -> None:
        assert FireState(agent_remaining_kg=-10.0).agent_remaining_kg >= 0.0

    def test_clamping_volume(self) -> None:
        assert FireState(volume_m3=0.0).volume_m3 >= 1.0

    def test_clamping_hrr(self) -> None:
        assert FireState(fire_hrr_kw=-5.0).fire_hrr_kw >= 0.0

    def test_clamping_smoke(self) -> None:
        assert FireState(smoke_od=-1.0).smoke_od >= 0.0

    def test_clamping_co(self) -> None:
        assert FireState(co_ppm=-100.0).co_ppm >= 0.0


class TestTSquaredHRR:
    def test_zero_time(self) -> None:
        assert t_squared_hrr(0.0, T_SQUARED_ALPHA_MEDIUM, 0.21) == 0.0

    def test_negative_time(self) -> None:
        assert t_squared_hrr(-1.0, T_SQUARED_ALPHA_MEDIUM, 0.21) == 0.0

    def test_zero_alpha(self) -> None:
        assert t_squared_hrr(5.0, 0.0, 0.21) == 0.0

    def test_positive_hrr(self) -> None:
        assert t_squared_hrr(5.0, T_SQUARED_ALPHA_MEDIUM, 0.21) > 0

    def test_grows_with_time(self) -> None:
        h1 = t_squared_hrr(1.0, T_SQUARED_ALPHA_MEDIUM, 0.21)
        h5 = t_squared_hrr(5.0, T_SQUARED_ALPHA_MEDIUM, 0.21)
        assert h5 > h1

    def test_faster_alpha(self) -> None:
        slow = t_squared_hrr(5.0, T_SQUARED_ALPHA_SLOW, 0.21)
        fast = t_squared_hrr(5.0, T_SQUARED_ALPHA_FAST, 0.21)
        assert fast > slow

    def test_enriched_o2(self) -> None:
        normal = t_squared_hrr(5.0, T_SQUARED_ALPHA_MEDIUM, 0.21)
        enriched = t_squared_hrr(5.0, T_SQUARED_ALPHA_MEDIUM, 0.26)
        assert enriched > normal

    def test_low_o2_no_enhancement(self) -> None:
        low = t_squared_hrr(5.0, T_SQUARED_ALPHA_MEDIUM, 0.18)
        normal = t_squared_hrr(5.0, T_SQUARED_ALPHA_MEDIUM, 0.21)
        assert abs(low - normal) < 0.01


class TestO2Consumed:
    def test_zero_hrr(self) -> None:
        assert o2_consumed_kg(0.0, 1.0) == 0.0

    def test_zero_duration(self) -> None:
        assert o2_consumed_kg(100.0, 0.0) == 0.0

    def test_positive(self) -> None:
        assert o2_consumed_kg(100.0, 1.0) > 0

    def test_proportional_to_hrr(self) -> None:
        c1 = o2_consumed_kg(100.0, 1.0)
        assert abs(o2_consumed_kg(200.0, 1.0) - 2 * c1) < 1e-6

    def test_proportional_to_duration(self) -> None:
        c1 = o2_consumed_kg(100.0, 1.0)
        assert abs(o2_consumed_kg(100.0, 5.0) - 5 * c1) < 1e-6


class TestSmokeDensity:
    def test_zero(self) -> None:
        assert smoke_density(0.0) == 0.0

    def test_positive(self) -> None:
        assert smoke_density(1000.0) > 0

    def test_proportional(self) -> None:
        s1 = smoke_density(500.0)
        assert abs(smoke_density(1000.0) - 2 * s1) < 1e-6


class TestCOConcentration:
    def test_zero_hrr(self) -> None:
        assert co_concentration(0.0, 0.21, 50.0) == 0.0

    def test_zero_volume(self) -> None:
        assert co_concentration(100.0, 0.21, 0.0) == 0.0

    def test_positive(self) -> None:
        assert co_concentration(100.0, 0.21, 50.0) > 0

    def test_low_o2_multiplier(self) -> None:
        normal = co_concentration(100.0, 0.21, 50.0)
        assert co_concentration(100.0, 0.15, 50.0) > normal

    def test_smaller_volume_higher(self) -> None:
        big = co_concentration(100.0, 0.21, 100.0)
        assert co_concentration(100.0, 0.21, 25.0) > big


class TestTempRise:
    def test_no_fire_at_ambient(self) -> None:
        assert abs(compartment_temp_rise(0.0, AMBIENT_TEMP_C, 50.0)) < 0.01

    def test_fire_heats_room(self) -> None:
        assert compartment_temp_rise(100.0, AMBIENT_TEMP_C, 50.0) > 0

    def test_hot_room_loses_heat(self) -> None:
        assert compartment_temp_rise(0.0, 200.0, 50.0) < 0

    def test_larger_volume_slower(self) -> None:
        small = compartment_temp_rise(100.0, AMBIENT_TEMP_C, 25.0)
        large = compartment_temp_rise(100.0, AMBIENT_TEMP_C, 100.0)
        assert small > large


class TestAgentNeeded:
    def test_below_threshold(self) -> None:
        assert agent_needed_kg(50.0, O2_EXTINCTION_THRESHOLD) == 0.0

    def test_positive(self) -> None:
        assert agent_needed_kg(50.0, NOMINAL_O2_FRACTION) > 0

    def test_higher_o2(self) -> None:
        assert agent_needed_kg(50.0, 0.26) > agent_needed_kg(50.0, 0.21)

    def test_larger_volume(self) -> None:
        assert agent_needed_kg(100.0, NOMINAL_O2_FRACTION) > agent_needed_kg(25.0, NOMINAL_O2_FRACTION)


class TestSmokeDetector:
    def test_below(self) -> None:
        assert check_smoke_detector(0.05) is False

    def test_at(self) -> None:
        assert check_smoke_detector(SMOKE_DETECTOR_THRESHOLD_OD) is True

    def test_above(self) -> None:
        assert check_smoke_detector(1.0) is True


class TestHeatDetector:
    def test_no_change(self) -> None:
        assert check_heat_detector(22.0, 22.0) is False

    def test_rate_of_rise(self) -> None:
        assert check_heat_detector(30.0, 22.0) is True

    def test_fixed_temp(self) -> None:
        assert check_heat_detector(HEAT_DETECTOR_FIXED_TEMP_C, 22.0) is True

    def test_slow_rise(self) -> None:
        assert check_heat_detector(25.0, 22.0) is False


class TestTickFire:
    def test_no_fire_stable(self) -> None:
        state = create_fire_system()
        result = tick_fire(state)
        assert result.fire_active is False
        assert result.temp_c == AMBIENT_TEMP_C

    def test_ignition(self) -> None:
        state = create_fire_system()
        result = tick_fire(state, ignite=True)
        assert state.fire_active is True
        assert any("FIRE_IGNITED" in w for w in result.warnings)

    def test_minute_counter(self) -> None:
        state = create_fire_system()
        tick_fire(state)
        assert state.minute == 1
        tick_fire(state)
        assert state.minute == 2

    def test_fire_grows(self) -> None:
        state = create_fire_system("storage")
        tick_fire(state, ignite=True)
        r1 = tick_fire(state)
        r2 = tick_fire(state)
        r3 = tick_fire(state)
        assert r3.hrr_kw > r1.hrr_kw

    def test_fire_consumes_o2(self) -> None:
        state = create_fire_system()
        initial = state.o2_fraction
        tick_fire(state, ignite=True)
        for _ in range(10):
            tick_fire(state)
        assert state.o2_fraction < initial

    def test_fire_produces_smoke(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        for _ in range(5):
            tick_fire(state)
        assert state.smoke_od > 0

    def test_fire_produces_co(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        for _ in range(5):
            tick_fire(state)
        assert state.co_ppm > 0

    def test_fire_heats(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        for _ in range(10):
            tick_fire(state)
        assert state.compartment_temp_c > AMBIENT_TEMP_C

    def test_smoke_alarm(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        for _ in range(20):
            tick_fire(state)
        assert state.smoke_alarm is True

    def test_manual_suppression(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        for _ in range(3):
            tick_fire(state)
        tick_fire(state, activate_suppression=True)
        assert state.suppression_active is True
        assert state.agent_deployed_kg > 0

    def test_seal_compartment(self) -> None:
        state = create_fire_system()
        tick_fire(state, seal_compartment=True)
        assert state.ventilation_open is False

    def test_self_extinguish(self) -> None:
        state = create_fire_system()
        state.ventilation_open = False
        state.volume_m3 = 10.0
        state.o2_mass_kg = state.volume_m3 * AIR_DENSITY_KG_PER_M3 * state.o2_fraction
        tick_fire(state, ignite=True)
        for _ in range(60):
            tick_fire(state)
        assert state.fire_suppressed or state.o2_fraction <= O2_EXTINCTION_THRESHOLD

    def test_agent_depletes(self) -> None:
        state = create_fire_system()
        state.agent_remaining_kg = 5.0
        tick_fire(state, ignite=True)
        tick_fire(state, activate_suppression=True)
        assert state.agent_remaining_kg < 5.0

    def test_no_double_ignition(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        state.fire_suppressed = True
        state.fire_active = False
        tick_fire(state, ignite=True)
        assert state.fire_active is False

    def test_suppression_reduces_o2(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        o2_before = state.o2_fraction
        tick_fire(state, activate_suppression=True)
        assert state.o2_fraction < o2_before

    def test_peak_tracking(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        for _ in range(10):
            tick_fire(state)
        assert state.peak_hrr_kw > 0
        assert state.peak_temp_c >= AMBIENT_TEMP_C


class TestCreateFireSystem:
    def test_standard(self) -> None:
        s = create_fire_system("standard")
        assert s.o2_fraction == NOMINAL_O2_FRACTION
        assert s.fire_growth_alpha == T_SQUARED_ALPHA_MEDIUM

    def test_enriched_o2(self) -> None:
        s = create_fire_system("enriched_o2")
        assert s.o2_fraction == 0.26
        assert s.fire_growth_alpha == T_SQUARED_ALPHA_FAST

    def test_storage(self) -> None:
        s = create_fire_system("storage")
        assert s.volume_m3 == 100.0
        assert s.fire_growth_alpha == T_SQUARED_ALPHA_SLOW

    def test_unknown(self) -> None:
        assert create_fire_system("x").o2_fraction == NOMINAL_O2_FRACTION


class TestInvariants:
    def test_10_min_no_crash(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        for _ in range(10):
            tick_fire(state)

    def test_30_min_no_crash(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        for _ in range(30):
            tick_fire(state)

    def test_o2_never_negative(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        for _ in range(60):
            tick_fire(state)
            assert state.o2_fraction >= 0.0
            assert state.o2_mass_kg >= 0.0

    def test_smoke_monotonic(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        prev = 0.0
        for _ in range(15):
            tick_fire(state)
            assert state.smoke_od >= prev
            prev = state.smoke_od

    def test_co_monotonic(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        prev = 0.0
        for _ in range(15):
            tick_fire(state)
            assert state.co_ppm >= prev
            prev = state.co_ppm

    def test_agent_never_negative(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        for _ in range(30):
            tick_fire(state, activate_suppression=True)
            assert state.agent_remaining_kg >= 0.0

    def test_agent_conservation(self) -> None:
        state = create_fire_system()
        initial = state.agent_remaining_kg
        tick_fire(state, ignite=True)
        for _ in range(10):
            tick_fire(state, activate_suppression=True)
        assert abs(state.agent_remaining_kg + state.agent_deployed_kg - initial) < 0.01

    def test_suppression_works(self) -> None:
        state = create_fire_system()
        state.agent_remaining_kg = 200.0
        state.ventilation_open = False
        tick_fire(state, ignite=True)
        for _ in range(30):
            tick_fire(state, activate_suppression=True)
        assert state.fire_suppressed is True

    def test_all_bounded(self) -> None:
        state = create_fire_system()
        tick_fire(state, ignite=True)
        for _ in range(20):
            result = tick_fire(state)
            assert 0.0 <= state.o2_fraction <= 1.0
            assert state.co_ppm >= 0
            assert state.smoke_od >= 0
            assert state.agent_remaining_kg >= 0
            assert state.fire_hrr_kw >= 0

    def test_enriched_more_dangerous(self) -> None:
        s1 = create_fire_system("standard")
        s2 = create_fire_system("enriched_o2")
        tick_fire(s1, ignite=True)
        tick_fire(s2, ignite=True)
        for _ in range(10):
            tick_fire(s1)
            tick_fire(s2)
        assert s2.peak_hrr_kw > s1.peak_hrr_kw
