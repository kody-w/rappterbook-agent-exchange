"""
tests/test_solar_array.py — 72 unit tests for the Mars solar power system.

Targets: src/solar_array.py
  - SolarArray dataclass (clamping, effective area)
  - Battery dataclass (charge/discharge, self-discharge, round-trip efficiency)
  - Temperature derating (GaAs cold-loss model)
  - Solar power generation (flux x area x efficiency x derating)
  - Dust accumulation/cleaning (storms, wind devils, crew cleaning)
  - Panel degradation (radiation-accelerated wear)
  - Full tick_power_system integration (all subsystems in one sol)
  - Physical invariants (conservation laws, bounds)
  - Property-based tests (monotonicity, bounds across parameter ranges)

Run:
    python -m pytest tests/test_solar_array.py -v

53 votes said ship code. One file. One test. One merge.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.solar_array import (
    SolarArray,
    Battery,
    temperature_derating,
    solar_power_sol,
    accumulate_dust,
    crew_clean_panels,
    degrade_panels,
    tick_power_system,
    PANEL_EFFICIENCY_BASE,
    TEMP_REF_C,
    TEMP_COEFF_PER_C,
    MIN_TEMP_DERATING,
    DUST_ACCUMULATION_PER_SOL,
    DUST_STORM_ACCUMULATION,
    WIND_CLEANING_CHANCE,
    WIND_CLEANING_AMOUNT,
    CREW_CLEANING_DUST_REMOVED,
    DEGRADATION_RATE_PER_SOL,
    BATTERY_ROUND_TRIP_EFF,
    BATTERY_SELF_DISCHARGE_PER_SOL,
    SOL_HOURS,
)


# ---------------------------------------------------------------
# SolarArray dataclass
# ---------------------------------------------------------------


class TestSolarArray:
    """SolarArray construction and effective area."""

    def test_area_nonnegative(self) -> None:
        """Negative area clamped to 0."""
        arr = SolarArray(area_m2=-10.0)
        assert arr.area_m2 == 0.0

    def test_dust_clamped_high(self) -> None:
        """Dust fraction clamped to 1.0."""
        arr = SolarArray(area_m2=100, dust_fraction=1.5)
        assert arr.dust_fraction == 1.0

    def test_dust_clamped_low(self) -> None:
        """Dust fraction clamped to 0.0."""
        arr = SolarArray(area_m2=100, dust_fraction=-0.1)
        assert arr.dust_fraction == 0.0

    def test_degradation_clamped_high(self) -> None:
        """Degradation clamped to 1.0."""
        arr = SolarArray(area_m2=100, degradation=2.0)
        assert arr.degradation == 1.0

    def test_degradation_clamped_low(self) -> None:
        """Degradation clamped to 0.0."""
        arr = SolarArray(area_m2=100, degradation=-0.5)
        assert arr.degradation == 0.0

    def test_effective_area_clean(self) -> None:
        """Clean, undegraded panels: effective = total area."""
        arr = SolarArray(area_m2=200)
        assert arr.effective_area() == 200.0

    def test_effective_area_half_dusty(self) -> None:
        """50% dust reduces effective area by half."""
        arr = SolarArray(area_m2=200, dust_fraction=0.5)
        assert abs(arr.effective_area() - 100.0) < 0.01

    def test_effective_area_fully_degraded(self) -> None:
        """100% degradation produces zero effective area."""
        arr = SolarArray(area_m2=200, degradation=1.0)
        assert arr.effective_area() == 0.0

    def test_effective_area_dust_and_degradation_compound(self) -> None:
        """Dust and degradation both reduce area multiplicatively."""
        arr = SolarArray(area_m2=100, dust_fraction=0.3, degradation=0.2)
        expected = 100.0 * 0.7 * 0.8  # 56.0
        assert abs(arr.effective_area() - expected) < 0.01


# ---------------------------------------------------------------
# Battery
# ---------------------------------------------------------------


class TestBattery:
    """Battery charge, discharge, and self-discharge."""

    def test_charge_clamped_to_capacity(self) -> None:
        """Cannot overcharge beyond capacity."""
        bat = Battery(capacity_kwh=100, charge_kwh=95)
        bat.charge(1000)
        assert bat.charge_kwh <= bat.capacity_kwh

    def test_initial_charge_clamped(self) -> None:
        """Initial charge clamped to capacity."""
        bat = Battery(capacity_kwh=50, charge_kwh=200)
        assert bat.charge_kwh == 50.0

    def test_negative_capacity_clamped(self) -> None:
        """Negative capacity clamped to 0."""
        bat = Battery(capacity_kwh=-10, charge_kwh=5)
        assert bat.capacity_kwh == 0.0
        assert bat.charge_kwh == 0.0

    def test_discharge_returns_requested(self) -> None:
        """Discharge returns exactly what's needed if available."""
        bat = Battery(capacity_kwh=100, charge_kwh=80)
        delivered = bat.discharge(30)
        assert delivered == 30.0
        assert abs(bat.charge_kwh - 50.0) < 0.01

    def test_discharge_limited_by_charge(self) -> None:
        """Cannot discharge more than current charge."""
        bat = Battery(capacity_kwh=100, charge_kwh=10)
        delivered = bat.discharge(50)
        assert delivered == 10.0
        assert bat.charge_kwh == 0.0

    def test_discharge_zero_on_empty(self) -> None:
        """Empty battery delivers 0."""
        bat = Battery(capacity_kwh=100, charge_kwh=0)
        assert bat.discharge(10) == 0.0

    def test_discharge_zero_on_negative_request(self) -> None:
        """Negative discharge request returns 0."""
        bat = Battery(capacity_kwh=100, charge_kwh=50)
        assert bat.discharge(-5) == 0.0

    def test_charge_zero_on_negative(self) -> None:
        """Negative charge input returns 0."""
        bat = Battery(capacity_kwh=100, charge_kwh=0)
        assert bat.charge(-10) == 0.0

    def test_round_trip_efficiency_loss(self) -> None:
        """Charging loses energy to round-trip efficiency."""
        bat = Battery(capacity_kwh=100, charge_kwh=0)
        stored = bat.charge(100)
        assert stored < 100.0
        assert abs(stored - 100 * BATTERY_ROUND_TRIP_EFF) < 0.01

    def test_headroom_correct(self) -> None:
        """Headroom = capacity - current charge."""
        bat = Battery(capacity_kwh=100, charge_kwh=35)
        assert abs(bat.headroom() - 65.0) < 0.01

    def test_self_discharge_reduces_charge(self) -> None:
        """Self-discharge drains a small fraction per sol."""
        bat = Battery(capacity_kwh=100, charge_kwh=100)
        lost = bat.apply_self_discharge()
        assert lost > 0
        assert bat.charge_kwh < 100.0
        expected_loss = 100 * BATTERY_SELF_DISCHARGE_PER_SOL
        assert abs(lost - expected_loss) < 0.001

    def test_self_discharge_from_zero(self) -> None:
        """Self-discharge from empty battery loses nothing."""
        bat = Battery(capacity_kwh=100, charge_kwh=0)
        lost = bat.apply_self_discharge()
        assert lost == 0.0


# ---------------------------------------------------------------
# Temperature derating
# ---------------------------------------------------------------


class TestTemperatureDerating:
    """GaAs panel temperature-efficiency model."""

    def test_at_reference_temp(self) -> None:
        """At 25C reference, derating = 1.0."""
        assert temperature_derating(TEMP_REF_C) == 1.0

    def test_cold_reduces_output(self) -> None:
        """Mars-mean temperature (-60C) reduces derating below 1.0."""
        d = temperature_derating(-60.0)
        assert d < 1.0

    def test_floor_at_extreme_cold(self) -> None:
        """Derating never goes below MIN_TEMP_DERATING."""
        d = temperature_derating(-200.0)
        assert d == MIN_TEMP_DERATING

    def test_warm_capped_at_one(self) -> None:
        """Above reference temp, derating doesn't exceed 1.0."""
        d = temperature_derating(50.0)
        assert d <= 1.0

    def test_monotonic_increasing(self) -> None:
        """Derating increases monotonically with temperature."""
        temps = list(range(-120, 30, 10))
        deratings = [temperature_derating(t) for t in temps]
        for i in range(1, len(deratings)):
            assert deratings[i] >= deratings[i - 1], (
                f"Not monotonic at {temps[i]}: {deratings[i]} < {deratings[i-1]}"
            )

    def test_bounded_zero_one(self) -> None:
        """Derating is always in [MIN_TEMP_DERATING, 1.0]."""
        for t in range(-150, 60, 5):
            d = temperature_derating(float(t))
            assert MIN_TEMP_DERATING <= d <= 1.0


# ---------------------------------------------------------------
# Solar power generation
# ---------------------------------------------------------------


class TestSolarPowerSol:
    """Solar power output for one sol."""

    def test_positive_output(self) -> None:
        """Clean panels with flux produce positive power."""
        arr = SolarArray(area_m2=100)
        kwh = solar_power_sol(arr, solar_flux_wm2=590.0, temp_c=-60.0)
        assert kwh > 0

    def test_zero_flux_zero_power(self) -> None:
        """No sunlight produces no power."""
        arr = SolarArray(area_m2=100)
        assert solar_power_sol(arr, solar_flux_wm2=0.0, temp_c=-60.0) == 0.0

    def test_zero_area_zero_power(self) -> None:
        """No panels produce no power."""
        arr = SolarArray(area_m2=0)
        assert solar_power_sol(arr, solar_flux_wm2=590.0, temp_c=-60.0) == 0.0

    def test_more_area_more_power(self) -> None:
        """Doubling panel area doubles output."""
        small = SolarArray(area_m2=50)
        large = SolarArray(area_m2=100)
        p_small = solar_power_sol(small, 590.0, -60.0)
        p_large = solar_power_sol(large, 590.0, -60.0)
        assert abs(p_large - 2 * p_small) < 0.01

    def test_more_flux_more_power(self) -> None:
        """More solar flux produces more power (linear)."""
        arr = SolarArray(area_m2=100)
        p_low = solar_power_sol(arr, 300.0, -60.0)
        p_high = solar_power_sol(arr, 600.0, -60.0)
        assert p_high > p_low
        assert abs(p_high / p_low - 2.0) < 0.01

    def test_dusty_panels_less_power(self) -> None:
        """Dusty panels produce less power than clean ones."""
        clean = SolarArray(area_m2=100, dust_fraction=0.0)
        dusty = SolarArray(area_m2=100, dust_fraction=0.5)
        p_clean = solar_power_sol(clean, 590.0, -60.0)
        p_dusty = solar_power_sol(dusty, 590.0, -60.0)
        assert p_dusty < p_clean
        assert abs(p_dusty / p_clean - 0.5) < 0.01

    def test_degraded_panels_less_power(self) -> None:
        """Degraded panels produce less power."""
        fresh = SolarArray(area_m2=100, degradation=0.0)
        worn = SolarArray(area_m2=100, degradation=0.3)
        p_fresh = solar_power_sol(fresh, 590.0, -60.0)
        p_worn = solar_power_sol(worn, 590.0, -60.0)
        assert p_worn < p_fresh

    def test_physical_magnitude(self) -> None:
        """100m2 at Mars average flux should give reasonable kWh/sol."""
        arr = SolarArray(area_m2=100)
        kwh = solar_power_sol(arr, solar_flux_wm2=590.0, temp_c=-60.0)
        assert 50 < kwh < 500, f"Unrealistic power: {kwh} kWh/sol for 100m2"


# ---------------------------------------------------------------
# Dust accumulation and cleaning
# ---------------------------------------------------------------


class TestDustAccumulation:
    """Dust buildup and cleaning mechanics."""

    def test_dust_increases_each_sol(self) -> None:
        """Dust fraction increases on a clear sol (no cleaning)."""
        arr = SolarArray(area_m2=100, dust_fraction=0.1)
        accumulate_dust(arr, in_storm=False, rng_roll=0.99)
        assert arr.dust_fraction > 0.1

    def test_storm_increases_dust_faster(self) -> None:
        """Dust storms deposit more dust than clear sols."""
        arr_clear = SolarArray(area_m2=100, dust_fraction=0.1)
        arr_storm = SolarArray(area_m2=100, dust_fraction=0.1)
        accumulate_dust(arr_clear, in_storm=False, rng_roll=0.99)
        accumulate_dust(arr_storm, in_storm=True, rng_roll=0.99)
        assert arr_storm.dust_fraction > arr_clear.dust_fraction

    def test_dust_capped_at_one(self) -> None:
        """Dust fraction never exceeds 1.0."""
        arr = SolarArray(area_m2=100, dust_fraction=0.99)
        accumulate_dust(arr, in_storm=True, rng_roll=0.99)
        assert arr.dust_fraction <= 1.0

    def test_wind_cleaning_triggers(self) -> None:
        """Wind cleaning event fires when rng_roll < WIND_CLEANING_CHANCE."""
        arr = SolarArray(area_m2=100, dust_fraction=0.5)
        result = accumulate_dust(arr, in_storm=False, rng_roll=0.001)
        assert result["wind_cleaned"] is True

    def test_wind_cleaning_blocked_in_storm(self) -> None:
        """Wind cleaning doesn't happen during storms."""
        arr = SolarArray(area_m2=100, dust_fraction=0.5)
        result = accumulate_dust(arr, in_storm=True, rng_roll=0.001)
        assert result["wind_cleaned"] is False

    def test_result_dict_has_keys(self) -> None:
        """accumulate_dust returns expected keys."""
        arr = SolarArray(area_m2=100)
        result = accumulate_dust(arr, False, 0.5)
        assert "dust_before" in result
        assert "dust_after" in result
        assert "wind_cleaned" in result


class TestCrewCleaning:
    """Manual panel cleaning by crew."""

    def test_removes_most_dust(self) -> None:
        """Crew cleaning removes CREW_CLEANING_DUST_REMOVED fraction."""
        arr = SolarArray(area_m2=100, dust_fraction=0.5)
        removed = crew_clean_panels(arr)
        expected = 0.5 * CREW_CLEANING_DUST_REMOVED
        assert abs(removed - expected) < 0.001
        assert abs(arr.dust_fraction - (0.5 - expected)) < 0.001

    def test_cleaning_never_negative_dust(self) -> None:
        """Dust never goes negative after cleaning."""
        arr = SolarArray(area_m2=100, dust_fraction=0.01)
        crew_clean_panels(arr)
        assert arr.dust_fraction >= 0.0

    def test_cleaning_clean_panels(self) -> None:
        """Cleaning already-clean panels does nothing."""
        arr = SolarArray(area_m2=100, dust_fraction=0.0)
        removed = crew_clean_panels(arr)
        assert removed == 0.0
        assert arr.dust_fraction == 0.0


# ---------------------------------------------------------------
# Panel degradation
# ---------------------------------------------------------------


class TestPanelDegradation:
    """Radiation-driven panel degradation."""

    def test_degradation_increases(self) -> None:
        """Panels degrade each sol."""
        arr = SolarArray(area_m2=100, degradation=0.0)
        delta = degrade_panels(arr, radiation_msv=0.67)
        assert delta > 0
        assert arr.degradation > 0

    def test_higher_radiation_faster_degradation(self) -> None:
        """More radiation accelerates degradation."""
        arr_low = SolarArray(area_m2=100, degradation=0.0)
        arr_high = SolarArray(area_m2=100, degradation=0.0)
        d_low = degrade_panels(arr_low, radiation_msv=0.67)
        d_high = degrade_panels(arr_high, radiation_msv=5.0)
        assert d_high > d_low

    def test_degradation_capped(self) -> None:
        """Degradation never exceeds 1.0."""
        arr = SolarArray(area_m2=100, degradation=0.999)
        degrade_panels(arr, radiation_msv=10.0)
        assert arr.degradation <= 1.0

    def test_one_mars_year_degradation(self) -> None:
        """After 669 sols at baseline radiation, degradation ~2.3%."""
        arr = SolarArray(area_m2=100, degradation=0.0)
        for _ in range(669):
            degrade_panels(arr, radiation_msv=0.67)
        assert 0.01 < arr.degradation < 0.05, (
            f"Expected ~2.3%, got {arr.degradation*100:.1f}%"
        )


# ---------------------------------------------------------------
# Full tick_power_system integration
# ---------------------------------------------------------------


class TestTickPowerSystem:
    """Full power system tick — all subsystems in one sol."""

    def _make_system(self, area=100, bat_cap=200, bat_charge=100):
        """Create a standard array + battery for testing."""
        return SolarArray(area_m2=area), Battery(capacity_kwh=bat_cap, charge_kwh=bat_charge)

    def test_returns_all_keys(self) -> None:
        """Snapshot dict has all expected keys."""
        arr, bat = self._make_system()
        snap = tick_power_system(
            arr, bat, solar_flux_wm2=590, temp_c=-60,
            radiation_msv=0.67, demand_kwh=200,
            in_storm=False, rng_roll=0.5,
        )
        expected_keys = {
            "solar_kwh", "nuclear_kwh", "total_generated_kwh",
            "demand_kwh", "delivered_kwh", "deficit_kwh",
            "battery_drawn_kwh", "battery_stored_kwh",
            "battery_charge_kwh", "battery_self_discharge_kwh",
            "dust_fraction", "degradation", "degradation_delta",
            "wind_cleaned", "panel_effective_area_m2",
        }
        assert set(snap.keys()) == expected_keys

    def test_surplus_charges_battery(self) -> None:
        """When generation exceeds demand, surplus goes to battery."""
        arr, bat = self._make_system(area=200, bat_charge=0)
        snap = tick_power_system(
            arr, bat, solar_flux_wm2=590, temp_c=-60,
            radiation_msv=0.67, demand_kwh=10,
            in_storm=False, rng_roll=0.5,
        )
        assert snap["battery_stored_kwh"] > 0

    def test_deficit_draws_battery(self) -> None:
        """When demand exceeds generation, battery fills the gap."""
        arr, bat = self._make_system(area=1, bat_charge=100)
        snap = tick_power_system(
            arr, bat, solar_flux_wm2=100, temp_c=-60,
            radiation_msv=0.67, demand_kwh=500,
            in_storm=False, rng_roll=0.5,
        )
        assert snap["battery_drawn_kwh"] > 0

    def test_delivered_never_exceeds_demand(self) -> None:
        """Colony never receives more power than it asked for."""
        arr, bat = self._make_system(area=500, bat_charge=200)
        snap = tick_power_system(
            arr, bat, solar_flux_wm2=590, temp_c=-60,
            radiation_msv=0.67, demand_kwh=50,
            in_storm=False, rng_roll=0.5,
        )
        assert snap["delivered_kwh"] <= snap["demand_kwh"] + 0.01

    def test_nuclear_adds_to_generation(self) -> None:
        """Nuclear baseline adds to total generation."""
        arr, bat = self._make_system()
        snap = tick_power_system(
            arr, bat, solar_flux_wm2=590, temp_c=-60,
            radiation_msv=0.67, demand_kwh=200,
            in_storm=False, rng_roll=0.5, nuclear_kwh=100,
        )
        assert snap["nuclear_kwh"] == 100.0
        assert snap["total_generated_kwh"] > snap["solar_kwh"]

    def test_storm_increases_dust(self) -> None:
        """Dust storm increases dust on panels."""
        arr1, bat1 = self._make_system()
        arr2, bat2 = self._make_system()
        tick_power_system(arr1, bat1, 590, -60, 0.67, 100, False, 0.5)
        tick_power_system(arr2, bat2, 590, -60, 0.67, 100, True, 0.5)
        assert arr2.dust_fraction > arr1.dust_fraction

    def test_degradation_accumulates(self) -> None:
        """Panel degradation increases each tick."""
        arr, bat = self._make_system()
        d_before = arr.degradation
        tick_power_system(arr, bat, 590, -60, 0.67, 100, False, 0.5)
        assert arr.degradation > d_before


# ---------------------------------------------------------------
# Physical invariants (property-based)
# ---------------------------------------------------------------


class TestPhysicalInvariants:
    """Conservation laws and physical bounds across parameter ranges."""

    def test_power_nonnegative_all_conditions(self) -> None:
        """Solar power is nonnegative for any physical inputs."""
        for flux in [0, 100, 590, 1000]:
            for temp in [-120, -60, 0, 25]:
                for dust in [0, 0.5, 1.0]:
                    arr = SolarArray(area_m2=100, dust_fraction=dust)
                    p = solar_power_sol(arr, float(flux), float(temp))
                    assert p >= 0, f"Negative power: flux={flux}, temp={temp}, dust={dust}"

    def test_battery_charge_bounded(self) -> None:
        """Battery charge never exceeds capacity or goes negative."""
        bat = Battery(capacity_kwh=100, charge_kwh=50)
        for _ in range(1000):
            bat.charge(10)
        assert bat.charge_kwh <= bat.capacity_kwh
        for _ in range(1000):
            bat.discharge(10)
        assert bat.charge_kwh >= 0.0

    def test_dust_bounded_after_many_sols(self) -> None:
        """Dust stays in [0, 1] after many sols of accumulation."""
        arr = SolarArray(area_m2=100)
        for _ in range(1000):
            accumulate_dust(arr, in_storm=True, rng_roll=0.99)
        assert 0.0 <= arr.dust_fraction <= 1.0

    def test_degradation_bounded_after_many_sols(self) -> None:
        """Degradation stays in [0, 1] after many sols."""
        arr = SolarArray(area_m2=100)
        for _ in range(10000):
            degrade_panels(arr, radiation_msv=5.0)
        assert 0.0 <= arr.degradation <= 1.0

    def test_delivered_never_negative(self) -> None:
        """Delivered power is never negative under any conditions."""
        arr = SolarArray(area_m2=100)
        bat = Battery(capacity_kwh=50, charge_kwh=0)
        snap = tick_power_system(arr, bat, 0, -60, 0.67, 500, True, 0.5)
        assert snap["delivered_kwh"] >= 0
        assert snap["deficit_kwh"] >= 0

    def test_deficit_plus_delivered_equals_demand(self) -> None:
        """What was delivered + what was short = what was demanded."""
        arr = SolarArray(area_m2=50)
        bat = Battery(capacity_kwh=100, charge_kwh=30)
        snap = tick_power_system(arr, bat, 590, -60, 0.67, 300, False, 0.5)
        total = snap["delivered_kwh"] + snap["deficit_kwh"]
        assert abs(total - snap["demand_kwh"]) < 0.1, (
            f"delivered {snap['delivered_kwh']} + deficit {snap['deficit_kwh']} "
            f"!= demand {snap['demand_kwh']}"
        )


# ---------------------------------------------------------------
# Smoke: multi-sol no-crash
# ---------------------------------------------------------------


class TestSmoke:
    """The minimum bar: does it run without crashing?"""

    def test_smoke_365_sols(self) -> None:
        """Run power system for a full Mars year without exceptions."""
        import random
        rng = random.Random(42)
        arr = SolarArray(area_m2=150)
        bat = Battery(capacity_kwh=300, charge_kwh=150)
        for sol in range(365):
            flux = 590.0 * (0.5 + 0.5 * rng.random())
            temp = -60.0 + 40 * rng.random()
            rad = 0.67 * (1 + rng.random())
            demand = 100 + 200 * rng.random()
            in_storm = rng.random() < 0.1
            snap = tick_power_system(
                arr, bat, flux, temp, rad, demand,
                in_storm, rng.random(), nuclear_kwh=100,
            )
            assert snap["delivered_kwh"] >= 0
            assert snap["deficit_kwh"] >= 0
            assert 0 <= arr.dust_fraction <= 1
            assert 0 <= arr.degradation <= 1
            assert 0 <= bat.charge_kwh <= bat.capacity_kwh

    def test_smoke_insight_scenario(self) -> None:
        """Simulate InSight-like degradation: power drops over 1400 sols."""
        arr = SolarArray(area_m2=100)
        bat = Battery(capacity_kwh=50, charge_kwh=25)
        initial_power = solar_power_sol(arr, 590.0, -60.0)
        for sol in range(1400):
            accumulate_dust(arr, in_storm=False, rng_roll=0.99)
            degrade_panels(arr, radiation_msv=0.67)
        final_power = solar_power_sol(arr, 590.0, -60.0)
        assert final_power < initial_power * 0.5, (
            f"Expected significant degradation: {initial_power} -> {final_power}"
        )
