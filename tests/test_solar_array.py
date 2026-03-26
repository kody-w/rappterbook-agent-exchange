"""
tests/test_solar_array.py — 72 tests for src/solar_array.py (Mars power system).

53 votes said ship code. One file. One test. One merge.

Run: python -m pytest tests/test_solar_array.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.solar_array import (
    BATTERY_ROUND_TRIP_EFF,
    BATTERY_SELF_DISCHARGE_PER_SOL,
    CREW_CLEANING_DUST_REMOVED,
    DEGRADATION_RATE_PER_SOL,
    DUST_ACCUMULATION_PER_SOL,
    DUST_STORM_ACCUMULATION,
    MIN_TEMP_DERATING,
    PANEL_EFFICIENCY_BASE,
    SOL_HOURS,
    TEMP_COEFF_PER_C,
    TEMP_REF_C,
    WIND_CLEANING_AMOUNT,
    WIND_CLEANING_CHANCE,
    Battery,
    SolarArray,
    accumulate_dust,
    crew_clean_panels,
    degrade_panels,
    solar_power_sol,
    temperature_derating,
    tick_power_system,
)


# ═══════════════════════════════════════════════════════════════════
# SolarArray dataclass
# ═══════════════════════════════════════════════════════════════════


class TestSolarArray:
    """SolarArray construction and invariants."""

    def test_basic_creation(self) -> None:
        a = SolarArray(area_m2=500.0)
        assert a.area_m2 == 500.0
        assert a.dust_fraction == 0.0
        assert a.degradation == 0.0

    def test_area_clamped_negative(self) -> None:
        a = SolarArray(area_m2=-100.0)
        assert a.area_m2 == 0.0

    def test_dust_clamped_high(self) -> None:
        a = SolarArray(area_m2=100.0, dust_fraction=1.5)
        assert a.dust_fraction == 1.0

    def test_dust_clamped_low(self) -> None:
        a = SolarArray(area_m2=100.0, dust_fraction=-0.3)
        assert a.dust_fraction == 0.0

    def test_degradation_clamped_high(self) -> None:
        a = SolarArray(area_m2=100.0, degradation=2.0)
        assert a.degradation == 1.0

    def test_degradation_clamped_low(self) -> None:
        a = SolarArray(area_m2=100.0, degradation=-0.1)
        assert a.degradation == 0.0

    def test_effective_area_clean(self) -> None:
        a = SolarArray(area_m2=1000.0)
        assert a.effective_area() == 1000.0

    def test_effective_area_dusty(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=0.5)
        assert a.effective_area() == 500.0

    def test_effective_area_degraded(self) -> None:
        a = SolarArray(area_m2=1000.0, degradation=0.2)
        assert a.effective_area() == 800.0

    def test_effective_area_combined(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=0.5, degradation=0.2)
        assert a.effective_area() == 400.0

    def test_effective_area_never_negative(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=1.0, degradation=1.0)
        assert a.effective_area() >= 0.0

    def test_effective_area_zero_area(self) -> None:
        a = SolarArray(area_m2=0.0)
        assert a.effective_area() == 0.0


# ═══════════════════════════════════════════════════════════════════
# Battery dataclass
# ═══════════════════════════════════════════════════════════════════


class TestBattery:
    """Battery construction, charge, discharge, self-discharge."""

    def test_basic_creation(self) -> None:
        b = Battery(capacity_kwh=500.0, charge_kwh=250.0)
        assert b.capacity_kwh == 500.0
        assert b.charge_kwh == 250.0

    def test_capacity_clamped_negative(self) -> None:
        b = Battery(capacity_kwh=-100.0)
        assert b.capacity_kwh == 0.0

    def test_charge_clamped_to_capacity(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=200.0)
        assert b.charge_kwh == 100.0

    def test_charge_clamped_negative(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=-50.0)
        assert b.charge_kwh == 0.0

    def test_headroom_full(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=100.0)
        assert b.headroom() == 0.0

    def test_headroom_empty(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=0.0)
        assert b.headroom() == 100.0

    def test_headroom_partial(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=40.0)
        assert abs(b.headroom() - 60.0) < 0.01

    def test_charge_stores_with_efficiency_loss(self) -> None:
        b = Battery(capacity_kwh=1000.0, charge_kwh=0.0)
        stored = b.charge(100.0)
        assert stored == 100.0 * BATTERY_ROUND_TRIP_EFF
        assert b.charge_kwh == stored

    def test_charge_zero_input(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=0.0)
        assert b.charge(0.0) == 0.0

    def test_charge_negative_input(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=0.0)
        assert b.charge(-10.0) == 0.0

    def test_charge_respects_capacity(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=95.0)
        stored = b.charge(100.0)
        assert b.charge_kwh <= 100.0
        assert stored <= 5.0

    def test_discharge_delivers_energy(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=80.0)
        delivered = b.discharge(30.0)
        assert delivered == 30.0
        assert abs(b.charge_kwh - 50.0) < 0.01

    def test_discharge_limited_by_charge(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=10.0)
        delivered = b.discharge(50.0)
        assert delivered == 10.0
        assert b.charge_kwh == 0.0

    def test_discharge_zero_request(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=50.0)
        assert b.discharge(0.0) == 0.0
        assert b.charge_kwh == 50.0

    def test_discharge_negative_request(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=50.0)
        assert b.discharge(-10.0) == 0.0

    def test_self_discharge(self) -> None:
        b = Battery(capacity_kwh=1000.0, charge_kwh=1000.0)
        lost = b.apply_self_discharge()
        expected_lost = 1000.0 * BATTERY_SELF_DISCHARGE_PER_SOL
        assert abs(lost - expected_lost) < 0.01
        assert b.charge_kwh < 1000.0

    def test_self_discharge_empty(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=0.0)
        assert b.apply_self_discharge() == 0.0

    def test_charge_never_exceeds_capacity(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=50.0)
        for _ in range(100):
            b.charge(10.0)
        assert 0 <= b.charge_kwh <= b.capacity_kwh

    def test_discharge_never_goes_negative(self) -> None:
        b = Battery(capacity_kwh=100.0, charge_kwh=5.0)
        for _ in range(50):
            b.discharge(10.0)
        assert b.charge_kwh >= 0.0


# ═══════════════════════════════════════════════════════════════════
# Temperature derating
# ═══════════════════════════════════════════════════════════════════


class TestTemperatureDerating:
    """Panel temperature derating function."""

    def test_reference_temp_no_derating(self) -> None:
        assert temperature_derating(TEMP_REF_C) == 1.0

    def test_cold_reduces_efficiency(self) -> None:
        assert temperature_derating(-60.0) < 1.0

    def test_extreme_cold_hits_floor(self) -> None:
        assert temperature_derating(-200.0) == MIN_TEMP_DERATING

    def test_floor_value(self) -> None:
        assert temperature_derating(-500.0) >= MIN_TEMP_DERATING

    def test_above_reference_clamped(self) -> None:
        assert temperature_derating(100.0) <= 1.0

    def test_mars_mean_temp(self) -> None:
        d = temperature_derating(-60.0)
        assert MIN_TEMP_DERATING < d < 1.0

    def test_monotonic_warming(self) -> None:
        prev = 0.0
        for t in range(-120, 30, 5):
            d = temperature_derating(float(t))
            assert d >= prev, f"Derating dropped at {t}°C"
            prev = d

    def test_bounded_all_temps(self) -> None:
        for t in range(-300, 300, 10):
            d = temperature_derating(float(t))
            assert MIN_TEMP_DERATING <= d <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Solar power generation
# ═══════════════════════════════════════════════════════════════════


class TestSolarPowerSol:
    """solar_power_sol — energy generated in one sol."""

    def test_zero_flux_zero_power(self) -> None:
        a = SolarArray(area_m2=1000.0)
        assert solar_power_sol(a, 0.0, -60.0) == 0.0

    def test_negative_flux_zero_power(self) -> None:
        a = SolarArray(area_m2=1000.0)
        assert solar_power_sol(a, -100.0, -60.0) == 0.0

    def test_positive_generation(self) -> None:
        a = SolarArray(area_m2=1000.0)
        assert solar_power_sol(a, 590.0, -60.0) > 0

    def test_more_area_more_power(self) -> None:
        small = SolarArray(area_m2=100.0)
        large = SolarArray(area_m2=1000.0)
        assert solar_power_sol(large, 590.0, -60.0) > solar_power_sol(small, 590.0, -60.0)

    def test_dust_reduces_power(self) -> None:
        clean = SolarArray(area_m2=1000.0, dust_fraction=0.0)
        dusty = SolarArray(area_m2=1000.0, dust_fraction=0.5)
        assert solar_power_sol(clean, 590.0, -60.0) > solar_power_sol(dusty, 590.0, -60.0)

    def test_degradation_reduces_power(self) -> None:
        fresh = SolarArray(area_m2=1000.0, degradation=0.0)
        worn = SolarArray(area_m2=1000.0, degradation=0.3)
        assert solar_power_sol(fresh, 590.0, -60.0) > solar_power_sol(worn, 590.0, -60.0)

    def test_warmer_more_power(self) -> None:
        a1 = SolarArray(area_m2=1000.0)
        a2 = SolarArray(area_m2=1000.0)
        assert solar_power_sol(a2, 590.0, -10.0) > solar_power_sol(a1, 590.0, -100.0)

    def test_power_scales_with_flux(self) -> None:
        a1 = SolarArray(area_m2=1000.0)
        a2 = SolarArray(area_m2=1000.0)
        assert solar_power_sol(a2, 590.0, -60.0) > solar_power_sol(a1, 200.0, -60.0)

    def test_fully_covered_panels_zero_power(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=1.0)
        assert solar_power_sol(a, 590.0, -60.0) == 0.0

    def test_fully_degraded_zero_power(self) -> None:
        a = SolarArray(area_m2=1000.0, degradation=1.0)
        assert solar_power_sol(a, 590.0, -60.0) == 0.0

    def test_power_physical_upper_bound(self) -> None:
        a = SolarArray(area_m2=1000.0)
        kwh = solar_power_sol(a, 590.0, 25.0)
        theoretical = 1000.0 * 590.0 * SOL_HOURS / 1000.0
        assert kwh <= theoretical


# ═══════════════════════════════════════════════════════════════════
# Dust accumulation
# ═══════════════════════════════════════════════════════════════════


class TestAccumulateDust:
    """accumulate_dust — dust gain per sol ± wind cleaning."""

    def test_normal_accumulation(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=0.0)
        accumulate_dust(a, in_storm=False, rng_roll=0.99)
        assert abs(a.dust_fraction - DUST_ACCUMULATION_PER_SOL) < 0.001

    def test_storm_accumulates_faster(self) -> None:
        a1 = SolarArray(area_m2=1000.0, dust_fraction=0.0)
        a2 = SolarArray(area_m2=1000.0, dust_fraction=0.0)
        accumulate_dust(a1, in_storm=False, rng_roll=0.99)
        accumulate_dust(a2, in_storm=True, rng_roll=0.99)
        assert a2.dust_fraction > a1.dust_fraction

    def test_dust_capped_at_one(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=0.999)
        accumulate_dust(a, in_storm=True, rng_roll=0.99)
        assert a.dust_fraction <= 1.0

    def test_wind_cleaning_event(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=0.5)
        result = accumulate_dust(a, in_storm=False, rng_roll=0.001)
        assert result["wind_cleaned"] is True
        assert a.dust_fraction < 0.5 + DUST_ACCUMULATION_PER_SOL

    def test_no_wind_cleaning_during_storm(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=0.5)
        result = accumulate_dust(a, in_storm=True, rng_roll=0.001)
        assert result["wind_cleaned"] is False

    def test_return_dict_structure(self) -> None:
        a = SolarArray(area_m2=1000.0)
        result = accumulate_dust(a, in_storm=False, rng_roll=0.99)
        assert "dust_before" in result
        assert "dust_after" in result
        assert "wind_cleaned" in result

    def test_dust_never_negative(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=0.001)
        accumulate_dust(a, in_storm=False, rng_roll=0.0)
        assert a.dust_fraction >= 0.0


# ═══════════════════════════════════════════════════════════════════
# Crew cleaning
# ═══════════════════════════════════════════════════════════════════


class TestCrewCleanPanels:
    """crew_clean_panels — manual panel cleaning."""

    def test_removes_most_dust(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=0.8)
        removed = crew_clean_panels(a)
        assert removed > 0.5
        assert a.dust_fraction < 0.2

    def test_clean_panels_no_change(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=0.0)
        assert crew_clean_panels(a) == 0.0
        assert a.dust_fraction == 0.0

    def test_dust_never_negative_after_clean(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=0.01)
        crew_clean_panels(a)
        assert a.dust_fraction >= 0.0


# ═══════════════════════════════════════════════════════════════════
# Panel degradation
# ═══════════════════════════════════════════════════════════════════


class TestDegradePanels:
    """degrade_panels — radiation + thermal cycling wear."""

    def test_baseline_degradation(self) -> None:
        a = SolarArray(area_m2=1000.0, degradation=0.0)
        delta = degrade_panels(a, 0.67)
        assert abs(delta - DEGRADATION_RATE_PER_SOL) < 1e-6

    def test_higher_radiation_faster_degradation(self) -> None:
        a1 = SolarArray(area_m2=1000.0, degradation=0.0)
        a2 = SolarArray(area_m2=1000.0, degradation=0.0)
        assert degrade_panels(a2, 5.0) > degrade_panels(a1, 0.67)

    def test_degradation_accumulates(self) -> None:
        a = SolarArray(area_m2=1000.0, degradation=0.0)
        for _ in range(100):
            degrade_panels(a, 0.67)
        assert a.degradation > 0.003

    def test_degradation_capped_at_one(self) -> None:
        a = SolarArray(area_m2=1000.0, degradation=0.999)
        degrade_panels(a, 10.0)
        assert a.degradation <= 1.0

    def test_degradation_delta_positive(self) -> None:
        a = SolarArray(area_m2=1000.0)
        assert degrade_panels(a, 0.67) > 0

    def test_one_mars_year_degradation(self) -> None:
        """Over 668 sols at baseline, degradation ~2.3%."""
        a = SolarArray(area_m2=1000.0, degradation=0.0)
        for _ in range(669):
            degrade_panels(a, 0.67)
        assert 0.015 < a.degradation < 0.04, f"Got {a.degradation}"


# ═══════════════════════════════════════════════════════════════════
# Full power system tick
# ═══════════════════════════════════════════════════════════════════


class TestTickPowerSystem:
    """tick_power_system — one sol of the complete power system."""

    def _make_system(
        self,
        area: float = 1000.0,
        bat_cap: float = 500.0,
        bat_charge: float = 250.0,
    ) -> tuple:
        return SolarArray(area_m2=area), Battery(capacity_kwh=bat_cap, charge_kwh=bat_charge)

    def test_basic_tick(self) -> None:
        array, bat = self._make_system()
        snap = tick_power_system(
            array, bat, 590.0, -60.0, 0.67,
            demand_kwh=100.0, in_storm=False, rng_roll=0.99, nuclear_kwh=100.0,
        )
        assert snap["solar_kwh"] > 0
        assert snap["total_generated_kwh"] > 0
        assert snap["delivered_kwh"] > 0

    def test_return_dict_keys(self) -> None:
        array, bat = self._make_system()
        snap = tick_power_system(
            array, bat, 590.0, -60.0, 0.67,
            demand_kwh=100.0, in_storm=False, rng_roll=0.99,
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
        array, bat = self._make_system(area=2000.0, bat_charge=0.0)
        snap = tick_power_system(
            array, bat, 590.0, -60.0, 0.67,
            demand_kwh=10.0, in_storm=False, rng_roll=0.99, nuclear_kwh=100.0,
        )
        assert snap["battery_stored_kwh"] > 0

    def test_deficit_draws_battery(self) -> None:
        array, bat = self._make_system(area=10.0, bat_charge=200.0)
        snap = tick_power_system(
            array, bat, 100.0, -60.0, 0.67,
            demand_kwh=500.0, in_storm=False, rng_roll=0.99, nuclear_kwh=0.0,
        )
        assert snap["battery_drawn_kwh"] > 0

    def test_nuclear_provides_storm_baseline(self) -> None:
        array, bat = self._make_system()
        snap = tick_power_system(
            array, bat, 50.0, -60.0, 0.67,
            demand_kwh=100.0, in_storm=True, rng_roll=0.99, nuclear_kwh=100.0,
        )
        assert snap["nuclear_kwh"] == 100.0
        assert snap["total_generated_kwh"] >= 100.0

    def test_delivered_never_exceeds_demand(self) -> None:
        array, bat = self._make_system(area=5000.0, bat_charge=500.0)
        snap = tick_power_system(
            array, bat, 590.0, -60.0, 0.67,
            demand_kwh=50.0, in_storm=False, rng_roll=0.99, nuclear_kwh=200.0,
        )
        assert snap["delivered_kwh"] <= snap["demand_kwh"] + 0.01

    def test_deficit_is_unmet_demand(self) -> None:
        array, bat = self._make_system(area=5.0, bat_charge=0.0)
        snap = tick_power_system(
            array, bat, 100.0, -60.0, 0.67,
            demand_kwh=5000.0, in_storm=False, rng_roll=0.99, nuclear_kwh=0.0,
        )
        assert snap["deficit_kwh"] >= 0
        assert abs(snap["deficit_kwh"] - (snap["demand_kwh"] - snap["delivered_kwh"])) < 0.1

    def test_degradation_increases_each_tick(self) -> None:
        array, bat = self._make_system()
        d0 = array.degradation
        tick_power_system(
            array, bat, 590.0, -60.0, 0.67,
            demand_kwh=100.0, in_storm=False, rng_roll=0.99,
        )
        assert array.degradation > d0

    def test_dust_increases_no_cleaning(self) -> None:
        array, bat = self._make_system()
        d0 = array.dust_fraction
        tick_power_system(
            array, bat, 590.0, -60.0, 0.67,
            demand_kwh=100.0, in_storm=False, rng_roll=0.99,
        )
        assert array.dust_fraction > d0

    def test_battery_charge_bounded(self) -> None:
        array, bat = self._make_system()
        for _ in range(50):
            tick_power_system(
                array, bat, 590.0, -60.0, 0.67,
                demand_kwh=100.0, in_storm=False, rng_roll=0.99, nuclear_kwh=100.0,
            )
        assert 0 <= bat.charge_kwh <= bat.capacity_kwh


# ═══════════════════════════════════════════════════════════════════
# Physical invariants (property-based)
# ═══════════════════════════════════════════════════════════════════


class TestPhysicalInvariants:
    """Cross-cutting conservation laws and bounds."""

    def test_power_always_nonnegative(self) -> None:
        for area in [10, 100, 1000, 5000]:
            for flux in [0, 100, 590]:
                for temp in [-120, -60, 0]:
                    a = SolarArray(area_m2=float(area))
                    b = Battery(capacity_kwh=500.0, charge_kwh=250.0)
                    snap = tick_power_system(
                        a, b, float(flux), float(temp), 0.67,
                        demand_kwh=200.0, in_storm=False,
                        rng_roll=0.99, nuclear_kwh=100.0,
                    )
                    assert snap["solar_kwh"] >= 0
                    assert snap["delivered_kwh"] >= 0
                    assert snap["deficit_kwh"] >= 0
                    assert snap["battery_drawn_kwh"] >= 0
                    assert snap["battery_stored_kwh"] >= 0

    def test_effective_area_monotonic_vs_dust(self) -> None:
        prev = float("inf")
        for d in range(0, 101, 5):
            a = SolarArray(area_m2=1000.0, dust_fraction=d / 100.0)
            ea = a.effective_area()
            assert ea <= prev, f"Effective area increased at dust={d}%"
            prev = ea

    def test_insight_lander_degradation(self) -> None:
        """InSight lost ~90% in 1400 sols. Our model should show significant loss."""
        a = SolarArray(area_m2=1000.0)
        b = Battery(capacity_kwh=100.0)
        initial_kwh = solar_power_sol(a, 590.0, -60.0)
        for _ in range(1400):
            tick_power_system(
                a, b, 590.0, -60.0, 0.67,
                demand_kwh=50.0, in_storm=False, rng_roll=0.99, nuclear_kwh=0.0,
            )
        final_kwh = solar_power_sol(a, 590.0, -60.0)
        ratio = final_kwh / initial_kwh if initial_kwh > 0 else 0
        assert ratio < 0.7, f"Only lost {(1-ratio)*100:.1f}% in 1400 sols"

    def test_energy_balance_surplus_tick(self) -> None:
        a = SolarArray(area_m2=2000.0)
        b = Battery(capacity_kwh=1000.0, charge_kwh=0.0)
        snap = tick_power_system(
            a, b, 590.0, -60.0, 0.67,
            demand_kwh=10.0, in_storm=False, rng_roll=0.99, nuclear_kwh=100.0,
        )
        surplus = snap["total_generated_kwh"] - snap["demand_kwh"]
        assert surplus > 0
        assert snap["delivered_kwh"] >= snap["demand_kwh"] - 0.01


# ═══════════════════════════════════════════════════════════════════
# Smoke tests
# ═══════════════════════════════════════════════════════════════════


class TestSmoke:
    """Smoke tests — run without crashing."""

    def test_10_sol_smoke(self) -> None:
        a = SolarArray(area_m2=1500.0)
        b = Battery(capacity_kwh=500.0, charge_kwh=250.0)
        for sol in range(10):
            snap = tick_power_system(
                a, b,
                solar_flux_wm2=590.0 - sol * 10,
                temp_c=-60.0 + sol * 2,
                radiation_msv=0.67,
                demand_kwh=100.0 + sol * 5,
                in_storm=(sol > 7),
                rng_roll=sol / 10.0,
                nuclear_kwh=100.0,
            )
            assert snap["delivered_kwh"] >= 0
            assert 0 <= b.charge_kwh <= b.capacity_kwh

    def test_365_sol_full_year(self) -> None:
        a = SolarArray(area_m2=2000.0)
        b = Battery(capacity_kwh=1000.0, charge_kwh=500.0)
        import random
        rng = random.Random(42)
        total_generated = 0.0
        total_delivered = 0.0
        for sol in range(365):
            flux = 400.0 + 190.0 * math.sin(sol / 100.0)
            temp = -80.0 + 40.0 * math.sin(sol / 50.0)
            snap = tick_power_system(
                a, b,
                solar_flux_wm2=flux, temp_c=temp,
                radiation_msv=0.67 + rng.random() * 0.3,
                demand_kwh=150.0 + rng.random() * 50,
                in_storm=(rng.random() < 0.05),
                rng_roll=rng.random(),
                nuclear_kwh=100.0,
            )
            total_generated += snap["total_generated_kwh"]
            total_delivered += snap["delivered_kwh"]
            assert 0 <= b.charge_kwh <= b.capacity_kwh
            assert a.dust_fraction <= 1.0
            assert a.degradation <= 1.0

        assert total_generated > 0
        assert total_delivered > 0
        assert a.degradation > 0.01

    def test_crew_clean_restores_power(self) -> None:
        a = SolarArray(area_m2=1000.0, dust_fraction=0.6)
        before = solar_power_sol(a, 590.0, -60.0)
        crew_clean_panels(a)
        after = solar_power_sol(a, 590.0, -60.0)
        assert after > before
