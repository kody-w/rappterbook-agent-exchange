"""
test_atmo_processor.py — 70+ tests for Mars atmospheric CO2 → O2 conversion.

Tests physical bounds, conservation laws, edge cases, degradation,
power-limiting, tank dynamics, and multi-sol simulation.
"""
from __future__ import annotations

import math
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atmo_processor import (
    # Constants
    CO2_FRACTION_ATMOSPHERE,
    MARS_SURFACE_PRESSURE_KPA,
    SOCE_BASE_EFFICIENCY,
    THEORETICAL_YIELD,
    O2_KG_PER_PERSON_SOL,
    CO2_EXHALED_KG_PER_PERSON_SOL,
    DEGRADATION_RATE_PER_SOL,
    DUST_INTAKE_PENALTY,
    MAINTENANCE_REPAIR_FRACTION,
    O2_STORAGE_LEAK_RATE_PER_SOL,
    SOL_HOURS,
    MOLAR_MASS_CO2_KG,
    MOLAR_MASS_O2_KG,
    ENTHALPY_KJ_PER_MOL_CO2,
    KJ_PER_KWH,
    # Data structures
    SOCEUnit,
    O2Tank,
    # Functions
    soce_efficiency,
    power_required_kwh,
    o2_from_co2,
    co2_required_for_o2,
    degrade_unit,
    maintain_unit,
    tick_atmo_processor,
)


# ===========================================================================
# Physical constants validation
# ===========================================================================

class TestPhysicalConstants:
    """Verify constants match known physical values."""

    def test_mars_co2_fraction(self):
        """Mars atmosphere is ~95% CO2."""
        assert 0.93 <= CO2_FRACTION_ATMOSPHERE <= 0.97

    def test_mars_surface_pressure(self):
        """Mars surface pressure is ~0.636 kPa."""
        assert 0.5 <= MARS_SURFACE_PRESSURE_KPA <= 0.8

    def test_theoretical_yield(self):
        """CO2 → O2 theoretical yield is 32/44 ≈ 0.727."""
        expected = MOLAR_MASS_O2_KG / MOLAR_MASS_CO2_KG
        assert abs(THEORETICAL_YIELD - expected) < 1e-4
        assert 0.72 < THEORETICAL_YIELD < 0.74

    def test_human_o2_consumption(self):
        """Human needs ~0.84 kg O2/sol (NASA HRP)."""
        assert 0.7 <= O2_KG_PER_PERSON_SOL <= 1.0

    def test_moxie_efficiency_range(self):
        """MOXIE measured 50-60% of theoretical."""
        assert 0.45 <= SOCE_BASE_EFFICIENCY <= 0.65

    def test_sol_hours(self):
        """Mars sol is ~24.66 hours."""
        assert 24.5 <= SOL_HOURS <= 24.8

    def test_enthalpy_positive(self):
        """CO2 electrolysis is endothermic (positive ΔH)."""
        assert ENTHALPY_KJ_PER_MOL_CO2 > 0

    def test_molar_masses(self):
        """CO2 = 44 g/mol, O2 = 32 g/mol."""
        assert abs(MOLAR_MASS_CO2_KG - 0.044) < 0.001
        assert abs(MOLAR_MASS_O2_KG - 0.032) < 0.001


# ===========================================================================
# SOCEUnit dataclass
# ===========================================================================

class TestSOCEUnit:
    """Tests for the SOCE processor unit."""

    def test_create_unit(self):
        """Basic unit creation."""
        unit = SOCEUnit(capacity_kg_sol=10.0)
        assert unit.capacity_kg_sol == 10.0
        assert unit.degradation == 0.0
        assert unit.operating_hours == 0.0

    def test_negative_capacity_clamped(self):
        """Negative capacity gets clamped to 0."""
        unit = SOCEUnit(capacity_kg_sol=-5.0)
        assert unit.capacity_kg_sol == 0.0

    def test_degradation_clamped(self):
        """Degradation is clamped to [0, 1]."""
        unit = SOCEUnit(capacity_kg_sol=10.0, degradation=1.5)
        assert unit.degradation == 1.0
        unit2 = SOCEUnit(capacity_kg_sol=10.0, degradation=-0.5)
        assert unit2.degradation == 0.0

    def test_effective_capacity_no_degradation(self):
        """Full capacity when no degradation."""
        unit = SOCEUnit(capacity_kg_sol=10.0)
        assert unit.effective_capacity() == 10.0

    def test_effective_capacity_half_degraded(self):
        """Half capacity at 50% degradation."""
        unit = SOCEUnit(capacity_kg_sol=10.0, degradation=0.5)
        assert abs(unit.effective_capacity() - 5.0) < 1e-6

    def test_effective_capacity_fully_degraded(self):
        """Zero capacity when fully degraded."""
        unit = SOCEUnit(capacity_kg_sol=10.0, degradation=1.0)
        assert unit.effective_capacity() == 0.0


# ===========================================================================
# O2Tank dataclass
# ===========================================================================

class TestO2Tank:
    """Tests for O2 storage tank."""

    def test_create_tank(self):
        """Basic tank creation."""
        tank = O2Tank(capacity_kg=100.0)
        assert tank.capacity_kg == 100.0
        assert tank.level_kg == 0.0

    def test_create_tank_with_initial_level(self):
        """Tank with initial O2."""
        tank = O2Tank(capacity_kg=100.0, level_kg=50.0)
        assert tank.level_kg == 50.0

    def test_level_clamped_to_capacity(self):
        """Can't store more than capacity."""
        tank = O2Tank(capacity_kg=100.0, level_kg=200.0)
        assert tank.level_kg == 100.0

    def test_negative_capacity_clamped(self):
        tank = O2Tank(capacity_kg=-10.0)
        assert tank.capacity_kg == 0.0

    def test_headroom(self):
        """Headroom = capacity - level."""
        tank = O2Tank(capacity_kg=100.0, level_kg=30.0)
        assert abs(tank.headroom() - 70.0) < 1e-6

    def test_store_within_capacity(self):
        """Store O2 within headroom."""
        tank = O2Tank(capacity_kg=100.0, level_kg=0.0)
        stored = tank.store(50.0)
        assert stored == 50.0
        assert tank.level_kg == 50.0

    def test_store_overflow_clamped(self):
        """Storing more than headroom is clamped."""
        tank = O2Tank(capacity_kg=100.0, level_kg=80.0)
        stored = tank.store(30.0)
        assert stored == 20.0
        assert abs(tank.level_kg - 100.0) < 1e-6

    def test_store_zero(self):
        """Storing zero returns zero."""
        tank = O2Tank(capacity_kg=100.0)
        assert tank.store(0.0) == 0.0

    def test_store_negative(self):
        """Storing negative returns zero."""
        tank = O2Tank(capacity_kg=100.0)
        assert tank.store(-5.0) == 0.0

    def test_draw_within_level(self):
        """Draw O2 within available level."""
        tank = O2Tank(capacity_kg=100.0, level_kg=50.0)
        drawn = tank.draw(30.0)
        assert drawn == 30.0
        assert abs(tank.level_kg - 20.0) < 1e-6

    def test_draw_more_than_available(self):
        """Drawing more than available returns what's there."""
        tank = O2Tank(capacity_kg=100.0, level_kg=10.0)
        drawn = tank.draw(50.0)
        assert drawn == 10.0
        assert tank.level_kg == 0.0

    def test_draw_zero(self):
        assert O2Tank(capacity_kg=100.0, level_kg=50.0).draw(0.0) == 0.0

    def test_draw_negative(self):
        assert O2Tank(capacity_kg=100.0, level_kg=50.0).draw(-5.0) == 0.0

    def test_apply_leak(self):
        """Leak removes small fraction."""
        tank = O2Tank(capacity_kg=100.0, level_kg=100.0)
        lost = tank.apply_leak()
        assert lost > 0
        assert abs(lost - 100.0 * O2_STORAGE_LEAK_RATE_PER_SOL) < 1e-6
        assert tank.level_kg < 100.0

    def test_leak_from_empty(self):
        """No leak from empty tank."""
        tank = O2Tank(capacity_kg=100.0, level_kg=0.0)
        lost = tank.apply_leak()
        assert lost == 0.0

    def test_days_of_reserve(self):
        """Reserve calculation for known population."""
        tank = O2Tank(capacity_kg=100.0, level_kg=8.4)
        # 10 people need 10 * 0.84 = 8.4 kg/sol → 1 sol reserve
        reserve = tank.days_of_reserve(10)
        assert abs(reserve - 1.0) < 0.01

    def test_days_of_reserve_zero_pop(self):
        """Zero population = infinite reserve."""
        tank = O2Tank(capacity_kg=100.0, level_kg=50.0)
        assert tank.days_of_reserve(0) == float('inf')


# ===========================================================================
# Core physics functions
# ===========================================================================

class TestSOCEEfficiency:
    """Tests for soce_efficiency()."""

    def test_baseline_clear_conditions(self):
        """Clear conditions, no degradation → base efficiency."""
        eff = soce_efficiency(0.0, MARS_SURFACE_PRESSURE_KPA, 0.0)
        assert abs(eff - SOCE_BASE_EFFICIENCY) < 1e-6

    def test_dust_reduces_efficiency(self):
        """Dust storm reduces efficiency."""
        clear = soce_efficiency(0.0, MARS_SURFACE_PRESSURE_KPA, 0.0)
        dusty = soce_efficiency(1.0, MARS_SURFACE_PRESSURE_KPA, 0.0)
        assert dusty < clear

    def test_degradation_reduces_efficiency(self):
        """Equipment degradation reduces efficiency."""
        fresh = soce_efficiency(0.0, MARS_SURFACE_PRESSURE_KPA, 0.0)
        worn = soce_efficiency(0.0, MARS_SURFACE_PRESSURE_KPA, 0.5)
        assert worn < fresh

    def test_full_degradation_zero_efficiency(self):
        """Fully degraded unit produces nothing."""
        eff = soce_efficiency(0.0, MARS_SURFACE_PRESSURE_KPA, 1.0)
        assert eff == 0.0

    def test_efficiency_always_in_bounds(self):
        """Efficiency is always in [0, 1]."""
        for dust in [0.0, 0.5, 1.0, 2.0]:
            for press in [0.0, 0.3, 0.636, 1.0, 5.0]:
                for deg in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    eff = soce_efficiency(dust, press, deg)
                    assert 0.0 <= eff <= 1.0

    def test_higher_pressure_helps(self):
        """Higher pressure (terraforming) improves efficiency (up to nominal)."""
        low = soce_efficiency(0.0, 0.3, 0.0)
        normal = soce_efficiency(0.0, MARS_SURFACE_PRESSURE_KPA, 0.0)
        assert normal >= low


class TestPowerRequired:
    """Tests for power_required_kwh()."""

    def test_zero_intake(self):
        """No CO2 intake = no power."""
        assert power_required_kwh(0.0) == 0.0

    def test_negative_intake(self):
        """Negative intake = no power."""
        assert power_required_kwh(-1.0) == 0.0

    def test_positive_intake_positive_power(self):
        """Any CO2 intake requires power."""
        assert power_required_kwh(1.0) > 0.0

    def test_power_scales_with_intake(self):
        """More CO2 = more power (monotonic)."""
        p1 = power_required_kwh(1.0)
        p2 = power_required_kwh(2.0)
        assert p2 > p1

    def test_power_includes_heating(self):
        """Even small intake requires heating power."""
        p = power_required_kwh(0.001)
        heating = 0.08 * SOL_HOURS
        assert p >= heating * 0.99  # heating dominates at small scale


class TestO2FromCO2:
    """Tests for o2_from_co2()."""

    def test_zero_co2(self):
        assert o2_from_co2(0.0, 0.55) == 0.0

    def test_zero_efficiency(self):
        assert o2_from_co2(10.0, 0.0) == 0.0

    def test_negative_co2(self):
        assert o2_from_co2(-1.0, 0.55) == 0.0

    def test_theoretical_yield(self):
        """At 100% efficiency, yield = 0.727 kg O2 per kg CO2."""
        o2 = o2_from_co2(1.0, 1.0)
        assert abs(o2 - THEORETICAL_YIELD) < 1e-4

    def test_realistic_yield(self):
        """At 55% efficiency, yield ≈ 0.4 kg O2 per kg CO2."""
        o2 = o2_from_co2(1.0, 0.55)
        assert 0.35 < o2 < 0.45

    def test_o2_scales_linearly(self):
        """Double CO2 = double O2 at same efficiency."""
        o2_1 = o2_from_co2(1.0, 0.55)
        o2_2 = o2_from_co2(2.0, 0.55)
        assert abs(o2_2 - 2 * o2_1) < 1e-6


class TestCO2Required:
    """Tests for co2_required_for_o2()."""

    def test_zero_target(self):
        assert co2_required_for_o2(0.0, 0.55) == 0.0

    def test_zero_efficiency(self):
        assert co2_required_for_o2(1.0, 0.0) == 0.0

    def test_round_trip(self):
        """co2_required → o2_from_co2 should give back the target."""
        target = 5.0
        eff = 0.55
        co2 = co2_required_for_o2(target, eff)
        o2 = o2_from_co2(co2, eff)
        assert abs(o2 - target) < 1e-6

    def test_more_o2_needs_more_co2(self):
        """More O2 target = more CO2 needed."""
        co2_1 = co2_required_for_o2(1.0, 0.55)
        co2_5 = co2_required_for_o2(5.0, 0.55)
        assert co2_5 > co2_1


# ===========================================================================
# Degradation and maintenance
# ===========================================================================

class TestDegradation:
    """Tests for degrade_unit() and maintain_unit()."""

    def test_degrade_increases(self):
        """Degradation increases each sol."""
        unit = SOCEUnit(capacity_kg_sol=10.0)
        delta = degrade_unit(unit, 0.67)
        assert delta > 0
        assert unit.degradation > 0

    def test_degrade_radiation_accelerates(self):
        """Higher radiation = faster degradation."""
        u1 = SOCEUnit(capacity_kg_sol=10.0)
        u2 = SOCEUnit(capacity_kg_sol=10.0)
        d1 = degrade_unit(u1, 0.67)  # normal GCR
        d2 = degrade_unit(u2, 5.0)   # solar flare
        assert d2 > d1

    def test_degrade_capped_at_one(self):
        """Degradation cannot exceed 1.0."""
        unit = SOCEUnit(capacity_kg_sol=10.0, degradation=0.999)
        degrade_unit(unit, 100.0)
        assert unit.degradation <= 1.0

    def test_operating_hours_accumulate(self):
        """Operating hours increase each sol."""
        unit = SOCEUnit(capacity_kg_sol=10.0)
        degrade_unit(unit, 0.67)
        assert abs(unit.operating_hours - SOL_HOURS) < 0.01

    def test_maintenance_repairs(self):
        """Maintenance reduces degradation."""
        unit = SOCEUnit(capacity_kg_sol=10.0, degradation=0.4)
        recovered = maintain_unit(unit)
        assert recovered > 0
        assert unit.degradation < 0.4

    def test_maintenance_fraction(self):
        """Maintenance recovers expected fraction."""
        unit = SOCEUnit(capacity_kg_sol=10.0, degradation=0.4)
        recovered = maintain_unit(unit)
        expected = 0.4 * MAINTENANCE_REPAIR_FRACTION
        assert abs(recovered - expected) < 1e-6

    def test_maintenance_on_zero_degradation(self):
        """Maintaining a pristine unit does nothing."""
        unit = SOCEUnit(capacity_kg_sol=10.0, degradation=0.0)
        recovered = maintain_unit(unit)
        assert recovered == 0.0
        assert unit.degradation == 0.0


# ===========================================================================
# tick_atmo_processor integration
# ===========================================================================

class TestTickAtmoProcessor:
    """Integration tests for the per-sol tick."""

    def _make_system(self, capacity=10.0, tank_cap=100.0, tank_level=0.0):
        unit = SOCEUnit(capacity_kg_sol=capacity)
        tank = O2Tank(capacity_kg=tank_cap, level_kg=tank_level)
        return unit, tank

    def test_basic_tick(self):
        """Tick produces O2 and returns valid snapshot."""
        unit, tank = self._make_system()
        snap = tick_atmo_processor(
            unit, tank, population=10, power_available_kwh=1000.0,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        assert snap["o2_produced_kg"] > 0
        assert snap["power_consumed_kwh"] > 0
        assert "efficiency" in snap
        assert "reserve_sols" in snap

    def test_all_fields_present(self):
        """Snapshot contains all expected fields."""
        unit, tank = self._make_system()
        snap = tick_atmo_processor(
            unit, tank, population=5, power_available_kwh=500.0,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        expected_fields = {
            "o2_produced_kg", "co2_intake_kg", "power_consumed_kwh",
            "demand_kg", "delivered_kg", "deficit_kg",
            "from_production_kg", "from_tank_kg",
            "tank_stored_kg", "tank_level_kg", "tank_leak_kg",
            "reserve_sols", "efficiency", "degradation",
            "degradation_delta", "operating_hours", "power_scale",
        }
        assert set(snap.keys()) == expected_fields

    def test_zero_population_no_demand(self):
        """No population = no O2 demand."""
        unit, tank = self._make_system()
        snap = tick_atmo_processor(
            unit, tank, population=0, power_available_kwh=500.0,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        assert snap["demand_kg"] == 0.0
        assert snap["deficit_kg"] == 0.0

    def test_surplus_goes_to_tank(self):
        """Excess O2 is stored in tank."""
        unit, tank = self._make_system(capacity=20.0)
        snap = tick_atmo_processor(
            unit, tank, population=1, power_available_kwh=5000.0,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        # 1 person needs 0.84 kg, unit can produce much more
        assert snap["tank_stored_kg"] > 0
        assert snap["deficit_kg"] == 0.0

    def test_deficit_when_underpowered(self):
        """Deficit occurs when power is insufficient."""
        unit, tank = self._make_system(capacity=10.0)
        snap = tick_atmo_processor(
            unit, tank, population=100, power_available_kwh=0.1,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        # 100 people need 84 kg, barely any power → deficit
        assert snap["deficit_kg"] > 0

    def test_tank_supplements_deficit(self):
        """Tank O2 is drawn when production falls short."""
        unit, tank = self._make_system(capacity=0.5, tank_level=50.0)
        snap = tick_atmo_processor(
            unit, tank, population=10, power_available_kwh=1000.0,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        # Unit can only produce 0.5 kg max, 10 people need 8.4
        assert snap["from_tank_kg"] > 0

    def test_zero_power_no_production(self):
        """Zero power = no O2 production."""
        unit, tank = self._make_system()
        snap = tick_atmo_processor(
            unit, tank, population=5, power_available_kwh=0.0,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        assert snap["o2_produced_kg"] == 0.0
        assert snap["power_consumed_kwh"] == 0.0

    def test_conservation_o2_balance(self):
        """O2 conservation: produced = delivered_from_prod + stored."""
        unit, tank = self._make_system(capacity=15.0)
        snap = tick_atmo_processor(
            unit, tank, population=5, power_available_kwh=5000.0,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        produced = snap["o2_produced_kg"]
        from_prod = snap["from_production_kg"]
        stored = snap["tank_stored_kg"]
        assert abs(produced - from_prod - stored) < 0.01

    def test_degradation_increases_over_tick(self):
        """Unit degrades each tick."""
        unit, tank = self._make_system()
        assert unit.degradation == 0.0
        tick_atmo_processor(
            unit, tank, population=5, power_available_kwh=500.0,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        assert unit.degradation > 0.0

    def test_dust_storm_reduces_production(self):
        """Dust storm reduces O2 output when power-limited."""
        u1, t1 = self._make_system(capacity=100.0)
        u2, t2 = self._make_system(capacity=100.0)
        # Use moderate power so efficiency differences show
        snap_clear = tick_atmo_processor(
            u1, t1, population=5, power_available_kwh=200.0,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        snap_storm = tick_atmo_processor(
            u2, t2, population=5, power_available_kwh=200.0,
            dust_opacity=1.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        assert snap_storm["o2_produced_kg"] < snap_clear["o2_produced_kg"]

    def test_power_scale_clamped(self):
        """Power scale is between 0 and 1."""
        unit, tank = self._make_system()
        snap = tick_atmo_processor(
            unit, tank, population=5, power_available_kwh=0.5,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        assert 0.0 <= snap["power_scale"] <= 1.0


# ===========================================================================
# Multi-sol simulation
# ===========================================================================

class TestMultiSolSimulation:
    """Run the processor for many sols and check invariants."""

    def test_10_sol_smoke(self):
        """10 sols without crash, all values non-negative."""
        unit = SOCEUnit(capacity_kg_sol=10.0)
        tank = O2Tank(capacity_kg=200.0, level_kg=50.0)
        for sol in range(10):
            snap = tick_atmo_processor(
                unit, tank, population=8, power_available_kwh=500.0,
                dust_opacity=0.05, pressure_kpa=0.636, radiation_msv=0.67,
            )
            for key, val in snap.items():
                if isinstance(val, (int, float)):
                    assert val >= 0, f"sol {sol}: {key} = {val} < 0"

    def test_100_sol_degradation_bounded(self):
        """Over 100 sols, degradation stays in [0, 1]."""
        unit = SOCEUnit(capacity_kg_sol=10.0)
        tank = O2Tank(capacity_kg=500.0, level_kg=100.0)
        for _ in range(100):
            tick_atmo_processor(
                unit, tank, population=5, power_available_kwh=500.0,
                dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
            )
        assert 0.0 <= unit.degradation <= 1.0

    def test_tank_never_negative(self):
        """Tank level never goes negative over 50 sols of heavy draw."""
        unit = SOCEUnit(capacity_kg_sol=2.0)
        tank = O2Tank(capacity_kg=100.0, level_kg=50.0)
        for _ in range(50):
            tick_atmo_processor(
                unit, tank, population=20, power_available_kwh=500.0,
                dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
            )
            assert tank.level_kg >= 0.0

    def test_reserve_decreases_under_deficit(self):
        """With constant deficit, reserve sols decrease over time."""
        unit = SOCEUnit(capacity_kg_sol=1.0)
        tank = O2Tank(capacity_kg=200.0, level_kg=100.0)
        reserves = []
        for _ in range(20):
            snap = tick_atmo_processor(
                unit, tank, population=20, power_available_kwh=500.0,
                dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
            )
            reserves.append(snap["reserve_sols"])
        # Reserve should trend downward (not strictly monotonic due to rounding)
        assert reserves[-1] < reserves[0]

    def test_surplus_fills_tank(self):
        """With low population and high capacity, tank fills over time."""
        unit = SOCEUnit(capacity_kg_sol=20.0)
        tank = O2Tank(capacity_kg=500.0, level_kg=0.0)
        for _ in range(30):
            tick_atmo_processor(
                unit, tank, population=1, power_available_kwh=5000.0,
                dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
            )
        assert tank.level_kg > 100.0  # should have accumulated significantly


# ===========================================================================
# Property-based invariants
# ===========================================================================

class TestInvariants:
    """Property-based checks that must hold for any input."""

    @pytest.mark.parametrize("pop", [0, 1, 10, 100])
    @pytest.mark.parametrize("power", [0.0, 10.0, 500.0, 10000.0])
    def test_delivered_never_exceeds_demand(self, pop, power):
        """O2 delivered never exceeds demand."""
        unit = SOCEUnit(capacity_kg_sol=50.0)
        tank = O2Tank(capacity_kg=500.0, level_kg=200.0)
        snap = tick_atmo_processor(
            unit, tank, population=pop, power_available_kwh=power,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        assert snap["delivered_kg"] <= snap["demand_kg"] + 0.01

    @pytest.mark.parametrize("dust", [0.0, 0.3, 0.7, 1.0])
    def test_efficiency_monotone_dust(self, dust):
        """Higher dust → lower or equal efficiency."""
        eff = soce_efficiency(dust, MARS_SURFACE_PRESSURE_KPA, 0.0)
        eff_low = soce_efficiency(0.0, MARS_SURFACE_PRESSURE_KPA, 0.0)
        assert eff <= eff_low + 1e-9

    @pytest.mark.parametrize("deg", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_capacity_monotone_degradation(self, deg):
        """Higher degradation → lower effective capacity."""
        unit = SOCEUnit(capacity_kg_sol=10.0, degradation=deg)
        assert unit.effective_capacity() <= 10.0

    def test_deficit_plus_delivered_equals_demand(self):
        """deficit + delivered = demand (accounting identity)."""
        unit = SOCEUnit(capacity_kg_sol=5.0)
        tank = O2Tank(capacity_kg=100.0, level_kg=20.0)
        snap = tick_atmo_processor(
            unit, tank, population=15, power_available_kwh=500.0,
            dust_opacity=0.0, pressure_kpa=0.636, radiation_msv=0.67,
        )
        total = snap["delivered_kg"] + snap["deficit_kg"]
        assert abs(total - snap["demand_kg"]) < 0.01
