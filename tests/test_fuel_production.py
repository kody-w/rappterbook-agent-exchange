"""
Tests for fuel_production.py — Sabatier ISRU propellant factory.

Coverage:
  - PropellantTank dataclass (clamping, add/remove, boiloff, fill_fraction)
  - SabatierReactor dataclass (operational check, catalyst degradation, rates)
  - Stoichiometric mass ratios (conservation of mass)
  - Warmup energy (ambient temperature effects)
  - Electrolysis calculations (water→H2, energy costs)
  - CO2 capture energy
  - sabatier_products (mass balance)
  - tick_fuel_production (full integration — all limiters)
  - Resource limiting (water, power, tank headroom)
  - sols_to_full_load estimation
  - propellant_status readiness check
  - Physical invariants (mass conservation, energy bounds, no negative masses)
  - Property-based sweeps (monotonicity, bounds across parameter ranges)
  - Multi-sol smoke test (100 sols without crash)

53 votes said ship code. One file. One test. One merge.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.fuel_production import (
    AMBIENT_EFFECT_FACTOR,
    CATALYST_DEGRADATION_PER_SOL,
    CATALYST_FRESH_EFFICIENCY,
    CATALYST_MIN_EFFICIENCY,
    CO2_CAPTURE_KWH_PER_KG,
    CO2_PER_KG_CH4,
    DEFAULT_CH4_TANK_KG,
    DEFAULT_O2_TANK_KG,
    ELECTROLYSIS_KWH_PER_KG_H2,
    H2_PER_KG_CH4,
    H2O_PER_KG_CH4,
    H2O_PER_KG_H2,
    MW_CH4,
    MW_CO2,
    MW_H2,
    MW_H2O,
    MW_O2,
    O2_FROM_ELECTROLYSIS_PER_KG_H2,
    O2_PER_KG_CH4,
    OPTIMAL_TEMP_LOW_C,
    REACTOR_WARMUP_KWH,
    STARSHIP_CH4_KG,
    STARSHIP_O2_KG,
    PropellantTank,
    SabatierReactor,
    co2_capture_energy_kwh,
    electrolysis_energy_kwh,
    electrolysis_water_needed_kg,
    propellant_status,
    sabatier_products,
    sols_to_full_load,
    tick_fuel_production,
    warmup_energy_kwh,
)


# ===================================================================
# PropellantTank
# ===================================================================

class TestPropellantTank:
    """PropellantTank construction, clamping, and operations."""

    def test_capacity_nonneg(self):
        """Negative capacity clamped to 0."""
        t = PropellantTank(capacity_kg=-100)
        assert t.capacity_kg == 0.0

    def test_level_clamped_to_capacity(self):
        """Level cannot exceed capacity."""
        t = PropellantTank(capacity_kg=100, level_kg=200)
        assert t.level_kg == 100.0

    def test_level_nonneg(self):
        """Negative level clamped to 0."""
        t = PropellantTank(capacity_kg=100, level_kg=-50)
        assert t.level_kg == 0.0

    def test_boiloff_rate_clamped(self):
        """Boiloff rate clamped to [0, 1]."""
        t = PropellantTank(capacity_kg=100, boiloff_rate_per_sol=2.0)
        assert t.boiloff_rate_per_sol == 1.0
        t2 = PropellantTank(capacity_kg=100, boiloff_rate_per_sol=-0.5)
        assert t2.boiloff_rate_per_sol == 0.0

    def test_headroom(self):
        """Headroom is capacity minus level."""
        t = PropellantTank(capacity_kg=100, level_kg=40)
        assert t.headroom() == pytest.approx(60.0)

    def test_headroom_full_tank(self):
        """Full tank has zero headroom."""
        t = PropellantTank(capacity_kg=100, level_kg=100)
        assert t.headroom() == pytest.approx(0.0)

    def test_add_within_headroom(self):
        """Add less than headroom stores all."""
        t = PropellantTank(capacity_kg=100, level_kg=40)
        added = t.add(30)
        assert added == pytest.approx(30.0)
        assert t.level_kg == pytest.approx(70.0)

    def test_add_exceeds_headroom(self):
        """Add more than headroom caps at capacity."""
        t = PropellantTank(capacity_kg=100, level_kg=80)
        added = t.add(50)
        assert added == pytest.approx(20.0)
        assert t.level_kg == pytest.approx(100.0)

    def test_add_negative_returns_zero(self):
        """Adding negative kg does nothing."""
        t = PropellantTank(capacity_kg=100, level_kg=50)
        added = t.add(-10)
        assert added == 0.0
        assert t.level_kg == pytest.approx(50.0)

    def test_remove_within_level(self):
        """Remove less than level succeeds fully."""
        t = PropellantTank(capacity_kg=100, level_kg=60)
        removed = t.remove(30)
        assert removed == pytest.approx(30.0)
        assert t.level_kg == pytest.approx(30.0)

    def test_remove_exceeds_level(self):
        """Remove more than level drains to zero."""
        t = PropellantTank(capacity_kg=100, level_kg=20)
        removed = t.remove(50)
        assert removed == pytest.approx(20.0)
        assert t.level_kg == pytest.approx(0.0)

    def test_remove_negative_returns_zero(self):
        """Removing negative kg does nothing."""
        t = PropellantTank(capacity_kg=100, level_kg=50)
        removed = t.remove(-10)
        assert removed == 0.0

    def test_boiloff_reduces_level(self):
        """Boiloff reduces level by rate fraction."""
        t = PropellantTank(capacity_kg=1000, level_kg=500, boiloff_rate_per_sol=0.01)
        lost = t.apply_boiloff()
        assert lost == pytest.approx(5.0)
        assert t.level_kg == pytest.approx(495.0)

    def test_boiloff_empty_tank(self):
        """Boiloff on empty tank loses nothing."""
        t = PropellantTank(capacity_kg=1000, level_kg=0)
        lost = t.apply_boiloff()
        assert lost == 0.0

    def test_fill_fraction(self):
        """Fill fraction is level/capacity."""
        t = PropellantTank(capacity_kg=200, level_kg=50)
        assert t.fill_fraction() == pytest.approx(0.25)

    def test_fill_fraction_zero_capacity(self):
        """Zero capacity returns 0 fill fraction (no div by zero)."""
        t = PropellantTank(capacity_kg=0)
        assert t.fill_fraction() == 0.0

    def test_fill_fraction_full(self):
        """Full tank has fill fraction 1.0."""
        t = PropellantTank(capacity_kg=100, level_kg=100)
        assert t.fill_fraction() == pytest.approx(1.0)


# ===================================================================
# SabatierReactor
# ===================================================================

class TestSabatierReactor:
    """SabatierReactor construction, operational checks, degradation."""

    def test_default_fresh_efficiency(self):
        """Fresh reactor has full catalyst efficiency."""
        r = SabatierReactor()
        assert r.catalyst_efficiency == pytest.approx(CATALYST_FRESH_EFFICIENCY)

    def test_operational_above_threshold(self):
        """Reactor is operational when catalyst above minimum."""
        r = SabatierReactor(catalyst_efficiency=0.50)
        assert r.is_operational() is True

    def test_not_operational_below_threshold(self):
        """Reactor is NOT operational when catalyst below minimum."""
        r = SabatierReactor(catalyst_efficiency=CATALYST_MIN_EFFICIENCY - 0.01)
        assert r.is_operational() is False

    def test_not_operational_at_zero(self):
        """Zero catalyst = dead reactor."""
        r = SabatierReactor(catalyst_efficiency=0.0)
        assert r.is_operational() is False

    def test_degradation_reduces_efficiency(self):
        """One sol of degradation reduces catalyst efficiency."""
        r = SabatierReactor(catalyst_efficiency=0.90)
        loss = r.degrade_catalyst()
        assert loss == pytest.approx(CATALYST_DEGRADATION_PER_SOL)
        assert r.catalyst_efficiency == pytest.approx(0.90 - CATALYST_DEGRADATION_PER_SOL)

    def test_degradation_stops_when_dead(self):
        """Dead reactor doesn't degrade further."""
        r = SabatierReactor(catalyst_efficiency=0.10)
        loss = r.degrade_catalyst()
        assert loss == 0.0

    def test_effective_rate_proportional_to_efficiency(self):
        """Effective rate scales with catalyst efficiency."""
        r = SabatierReactor(max_ch4_per_sol_kg=10, catalyst_efficiency=0.80)
        assert r.effective_rate_kg() == pytest.approx(8.0)

    def test_effective_rate_dead_reactor(self):
        """Dead reactor has zero rate."""
        r = SabatierReactor(catalyst_efficiency=0.0)
        assert r.effective_rate_kg() == 0.0

    def test_clamping_negative_rate(self):
        """Negative max rate clamped to 0."""
        r = SabatierReactor(max_ch4_per_sol_kg=-5)
        assert r.max_ch4_per_sol_kg == 0.0

    def test_clamping_oversize_efficiency(self):
        """Efficiency clamped to 1.0."""
        r = SabatierReactor(catalyst_efficiency=1.5)
        assert r.catalyst_efficiency == 1.0


# ===================================================================
# Stoichiometric constants (chemistry self-consistency)
# ===================================================================

class TestStoichiometry:
    """Verify the stoichiometric mass ratios are chemically correct."""

    def test_co2_per_ch4_ratio(self):
        """CO2 + 4H2 → CH4: 1 mol CO2 per 1 mol CH4 → MW ratio."""
        expected = MW_CO2 / MW_CH4
        assert CO2_PER_KG_CH4 == pytest.approx(expected)

    def test_h2_per_ch4_ratio(self):
        """4 mol H2 per 1 mol CH4."""
        expected = (4 * MW_H2) / MW_CH4
        assert H2_PER_KG_CH4 == pytest.approx(expected)

    def test_h2o_per_ch4_ratio(self):
        """2 mol H2O produced per 1 mol CH4."""
        expected = (2 * MW_H2O) / MW_CH4
        assert H2O_PER_KG_CH4 == pytest.approx(expected)

    def test_o2_per_ch4_ratio(self):
        """2 mol O2 per 1 mol CH4 (overall reaction)."""
        expected = (2 * MW_O2) / MW_CH4
        assert O2_PER_KG_CH4 == pytest.approx(expected)

    def test_h2o_per_h2_electrolysis(self):
        """H2O → H2 + 0.5 O2: 1 mol H2O per 1 mol H2."""
        expected = MW_H2O / MW_H2
        assert H2O_PER_KG_H2 == pytest.approx(expected)

    def test_sabatier_mass_balance(self):
        """Sabatier reaction conserves mass: reactants = products."""
        # CO2 + 4H2 → CH4 + 2H2O (molar)
        reactants = MW_CO2 + 4 * MW_H2
        products = MW_CH4 + 2 * MW_H2O
        assert reactants == pytest.approx(products, rel=1e-3)

    def test_electrolysis_mass_balance(self):
        """Electrolysis conserves mass: H2O → H2 + 0.5 O2."""
        reactant = MW_H2O
        product = MW_H2 + 0.5 * MW_O2
        assert reactant == pytest.approx(product, rel=1e-3)


# ===================================================================
# Warmup energy
# ===================================================================

class TestWarmupEnergy:
    """Reactor warmup energy calculations."""

    def test_warmup_at_reference_temp(self):
        """Warmup at reactor operating temp is base energy."""
        e = warmup_energy_kwh(OPTIMAL_TEMP_LOW_C)
        assert e == pytest.approx(REACTOR_WARMUP_KWH)

    def test_warmup_colder_needs_more(self):
        """Colder ambient needs more warmup energy."""
        e_cold = warmup_energy_kwh(-100.0)
        e_warm = warmup_energy_kwh(-20.0)
        assert e_cold > e_warm

    def test_warmup_always_positive(self):
        """Warmup energy is always positive."""
        for temp in [-120, -80, -40, 0, 20, 300]:
            assert warmup_energy_kwh(temp) > 0


# ===================================================================
# Electrolysis calculations
# ===================================================================

class TestElectrolysis:
    """Water electrolysis helper functions."""

    def test_water_needed_positive(self):
        """Positive H2 requires positive water."""
        w = electrolysis_water_needed_kg(1.0)
        assert w == pytest.approx(H2O_PER_KG_H2)

    def test_water_needed_zero(self):
        """Zero H2 needs zero water."""
        assert electrolysis_water_needed_kg(0.0) == 0.0

    def test_water_needed_negative(self):
        """Negative H2 returns zero water."""
        assert electrolysis_water_needed_kg(-5.0) == 0.0

    def test_energy_positive(self):
        """Positive H2 requires positive energy."""
        e = electrolysis_energy_kwh(1.0)
        assert e == pytest.approx(ELECTROLYSIS_KWH_PER_KG_H2)

    def test_energy_zero(self):
        """Zero H2 needs zero energy."""
        assert electrolysis_energy_kwh(0.0) == 0.0

    def test_energy_negative(self):
        """Negative H2 returns zero energy."""
        assert electrolysis_energy_kwh(-1.0) == 0.0


# ===================================================================
# CO2 capture energy
# ===================================================================

class TestCO2Capture:
    """CO2 atmospheric capture energy calculations."""

    def test_capture_positive(self):
        """Positive CO2 mass costs positive energy."""
        e = co2_capture_energy_kwh(10.0)
        assert e == pytest.approx(10.0 * CO2_CAPTURE_KWH_PER_KG)

    def test_capture_zero(self):
        assert co2_capture_energy_kwh(0.0) == 0.0

    def test_capture_negative(self):
        assert co2_capture_energy_kwh(-5.0) == 0.0


# ===================================================================
# sabatier_products
# ===================================================================

class TestSabatierProducts:
    """Sabatier reaction product calculations."""

    def test_one_kg_ch4(self):
        """1 kg CH4 requires correct stoichiometric inputs."""
        p = sabatier_products(1.0)
        assert p["ch4_kg"] == pytest.approx(1.0)
        assert p["co2_consumed_kg"] == pytest.approx(CO2_PER_KG_CH4)
        assert p["h2_consumed_kg"] == pytest.approx(H2_PER_KG_CH4)
        assert p["h2o_produced_kg"] == pytest.approx(H2O_PER_KG_CH4)

    def test_zero_ch4(self):
        """Zero CH4 produces all zeros."""
        p = sabatier_products(0.0)
        assert all(v == 0.0 for v in p.values())

    def test_negative_ch4(self):
        """Negative CH4 produces all zeros."""
        p = sabatier_products(-1.0)
        assert all(v == 0.0 for v in p.values())

    def test_mass_balance(self):
        """Mass in = mass out for any positive CH4."""
        for ch4 in [0.1, 1.0, 10.0, 100.0]:
            p = sabatier_products(ch4)
            mass_in = p["co2_consumed_kg"] + p["h2_consumed_kg"]
            mass_out = p["ch4_kg"] + p["h2o_produced_kg"]
            assert mass_in == pytest.approx(mass_out, rel=1e-3)

    def test_products_scale_linearly(self):
        """Doubling CH4 doubles all products/inputs."""
        p1 = sabatier_products(5.0)
        p2 = sabatier_products(10.0)
        for key in p1:
            assert p2[key] == pytest.approx(2 * p1[key], rel=1e-6)


# ===================================================================
# tick_fuel_production — integration
# ===================================================================

def make_standard_setup(
    ch4_level: float = 0.0,
    o2_level: float = 0.0,
    water: float = 500.0,
    power: float = 500.0,
    temp: float = -60.0,
    efficiency: float = CATALYST_FRESH_EFFICIENCY,
    max_rate: float = 10.0,
) -> tuple:
    """Create a standard reactor/tank setup for testing."""
    reactor = SabatierReactor(max_ch4_per_sol_kg=max_rate, catalyst_efficiency=efficiency)
    ch4_tank = PropellantTank(capacity_kg=DEFAULT_CH4_TANK_KG, level_kg=ch4_level)
    o2_tank = PropellantTank(capacity_kg=DEFAULT_O2_TANK_KG, level_kg=o2_level)
    return reactor, ch4_tank, o2_tank, water, power, temp


class TestTickFuelProduction:
    """Full tick integration tests."""

    def test_basic_production(self):
        """With sufficient resources, produces CH4 and O2."""
        r, ch4, o2, w, p, t = make_standard_setup()
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["operational"] is True
        assert result["ch4_produced_kg"] > 0
        assert result["o2_produced_kg"] > 0

    def test_power_consumed_positive(self):
        """Active production consumes positive power."""
        r, ch4, o2, w, p, t = make_standard_setup()
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["power_consumed_kwh"] > 0

    def test_power_consumed_within_budget(self):
        """Power consumed never exceeds available budget."""
        r, ch4, o2, w, p, t = make_standard_setup(power=100.0)
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["power_consumed_kwh"] <= 100.0 + 0.001

    def test_water_consumed_positive(self):
        """Production consumes water (for electrolysis)."""
        r, ch4, o2, w, p, t = make_standard_setup()
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["water_consumed_kg"] > 0

    def test_water_returned_positive(self):
        """Sabatier byproduct returns some water."""
        r, ch4, o2, w, p, t = make_standard_setup()
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["water_returned_kg"] > 0

    def test_co2_consumed_positive(self):
        """CO2 is consumed from atmosphere."""
        r, ch4, o2, w, p, t = make_standard_setup()
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["co2_consumed_kg"] > 0

    def test_dead_reactor_no_production(self):
        """Dead catalyst → zero production."""
        r, ch4, o2, w, p, t = make_standard_setup(efficiency=0.0)
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["operational"] is False
        assert result["ch4_produced_kg"] == 0
        assert result["limited_by"] == "catalyst_dead"

    def test_zero_power_no_production(self):
        """Zero power → only warmup consumed, no production."""
        r, ch4, o2, w, p, t = make_standard_setup(power=0.0)
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["ch4_produced_kg"] == 0
        assert result["limited_by"] == "power"

    def test_zero_water_no_production(self):
        """Zero water → no electrolysis → no H2 → no CH4."""
        r, ch4, o2, w, p, t = make_standard_setup(water=0.0)
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["ch4_produced_kg"] == 0.0

    def test_tank_levels_increase(self):
        """After production, tank levels are higher than starting."""
        r, ch4, o2, w, p, t = make_standard_setup()
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["ch4_tank_kg"] > 0
        assert result["o2_tank_kg"] > 0

    def test_catalyst_degrades_after_tick(self):
        """Catalyst efficiency drops after one sol of operation."""
        r, ch4, o2, w, p, t = make_standard_setup()
        initial_eff = r.catalyst_efficiency
        tick_fuel_production(r, ch4, o2, w, p, t)
        assert r.catalyst_efficiency < initial_eff

    def test_total_sols_incremented(self):
        """Reactor sol counter increments."""
        r, ch4, o2, w, p, t = make_standard_setup()
        tick_fuel_production(r, ch4, o2, w, p, t)
        assert r.total_sols_run == 1

    def test_boiloff_applied(self):
        """Boiloff reduces tank levels slightly."""
        r = SabatierReactor(max_ch4_per_sol_kg=0)  # no production
        ch4 = PropellantTank(capacity_kg=10000, level_kg=1000, boiloff_rate_per_sol=0.01)
        o2 = PropellantTank(capacity_kg=10000, level_kg=1000, boiloff_rate_per_sol=0.01)
        result = tick_fuel_production(r, ch4, o2, 100, 100, -60)
        assert result["ch4_boiloff_kg"] > 0
        assert result["o2_boiloff_kg"] > 0


# ===================================================================
# Resource limiting
# ===================================================================

class TestResourceLimiting:
    """Verify production scales down when limited by resources."""

    def test_water_limited_produces_less(self):
        """Scarce water → less CH4 than abundant water."""
        r1, ch4_1, o2_1, _, p, t = make_standard_setup(water=500)
        res1 = tick_fuel_production(r1, ch4_1, o2_1, 500, p, t)

        r2, ch4_2, o2_2, _, _, _ = make_standard_setup(water=10)
        res2 = tick_fuel_production(r2, ch4_2, o2_2, 10, p, t)

        assert res2["ch4_produced_kg"] < res1["ch4_produced_kg"]

    def test_power_limited_produces_less(self):
        """Less power → less CH4."""
        r1, ch4_1, o2_1, w, _, t = make_standard_setup(power=500)
        res1 = tick_fuel_production(r1, ch4_1, o2_1, w, 500, t)

        r2, ch4_2, o2_2, _, _, _ = make_standard_setup(power=10)
        res2 = tick_fuel_production(r2, ch4_2, o2_2, w, 10, t)

        assert res2["ch4_produced_kg"] < res1["ch4_produced_kg"]

    def test_ch4_tank_full_caps_production(self):
        """Full CH4 tank → production limited."""
        r, ch4, o2, w, p, t = make_standard_setup(
            ch4_level=DEFAULT_CH4_TANK_KG - 1.0
        )
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["ch4_produced_kg"] <= 1.0 + 0.001

    def test_o2_tank_full_caps_production(self):
        """Full O2 tank → production limited."""
        r, ch4, o2, w, p, t = make_standard_setup(
            o2_level=DEFAULT_O2_TANK_KG - 1.0
        )
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        assert result["o2_produced_kg"] <= 1.0 + 0.01


# ===================================================================
# Physical invariants
# ===================================================================

class TestPhysicalInvariants:
    """Conservation laws and physical bounds that must ALWAYS hold."""

    def test_no_negative_masses(self):
        """No mass value in result is ever negative."""
        for water in [0, 10, 100, 1000]:
            for power in [0, 10, 100, 1000]:
                r, ch4, o2, _, _, t = make_standard_setup()
                result = tick_fuel_production(r, ch4, o2, water, power, t)
                for key, val in result.items():
                    if isinstance(val, (int, float)) and "kwh" not in key and key != "catalyst_efficiency":
                        assert val >= 0, f"{key} = {val} < 0 at water={water}, power={power}"

    def test_tank_levels_never_exceed_capacity(self):
        """Tank levels stay within [0, capacity]."""
        for _ in range(50):
            r, ch4, o2, w, p, t = make_standard_setup()
            tick_fuel_production(r, ch4, o2, w, p, t)
            assert 0 <= ch4.level_kg <= ch4.capacity_kg + 0.001
            assert 0 <= o2.level_kg <= o2.capacity_kg + 0.001

    def test_catalyst_bounded_0_1(self):
        """Catalyst efficiency stays in [0, 1]."""
        r = SabatierReactor(catalyst_efficiency=CATALYST_FRESH_EFFICIENCY)
        for _ in range(20000):
            r.degrade_catalyst()
        assert 0 <= r.catalyst_efficiency <= 1.0

    def test_energy_accounting(self):
        """Total energy consumed = warmup + electrolysis + CO2 capture."""
        r, ch4, o2, w, p, t = make_standard_setup()
        result = tick_fuel_production(r, ch4, o2, w, p, t)
        expected = result["warmup_kwh"] + result["electrolysis_kwh"] + result["co2_capture_kwh"]
        assert result["power_consumed_kwh"] == pytest.approx(expected, rel=1e-3)


# ===================================================================
# Property-based sweeps
# ===================================================================

class TestPropertySweeps:
    """Monotonicity and bound checks across parameter ranges."""

    def test_more_water_means_more_or_equal_ch4(self):
        """CH4 production is monotonically non-decreasing with water supply."""
        prev_ch4 = -1.0
        for water in [0, 1, 5, 10, 50, 100, 500]:
            r, ch4t, o2t, _, p, t = make_standard_setup(water=water)
            result = tick_fuel_production(r, ch4t, o2t, water, p, t)
            assert result["ch4_produced_kg"] >= prev_ch4 - 0.001
            prev_ch4 = result["ch4_produced_kg"]

    def test_more_power_means_more_or_equal_ch4(self):
        """CH4 production is monotonically non-decreasing with power."""
        prev_ch4 = -1.0
        for power in [0, 5, 10, 50, 100, 500]:
            r, ch4t, o2t, w, _, t = make_standard_setup(power=power)
            result = tick_fuel_production(r, ch4t, o2t, w, power, t)
            assert result["ch4_produced_kg"] >= prev_ch4 - 0.001
            prev_ch4 = result["ch4_produced_kg"]

    def test_higher_efficiency_means_more_ch4(self):
        """Better catalyst → more CH4."""
        prev_ch4 = -1.0
        for eff in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
            r, ch4t, o2t, w, p, t = make_standard_setup(efficiency=eff)
            result = tick_fuel_production(r, ch4t, o2t, w, p, t)
            assert result["ch4_produced_kg"] >= prev_ch4 - 0.001
            prev_ch4 = result["ch4_produced_kg"]

    def test_temperature_affects_warmup(self):
        """Colder temps increase warmup energy monotonically."""
        prev_warmup = 999999
        for temp in [-120, -100, -60, -20, 0, 20]:
            r, ch4t, o2t, w, p, _ = make_standard_setup()
            result = tick_fuel_production(r, ch4t, o2t, w, p, temp)
            assert result["warmup_kwh"] <= prev_warmup + 0.001
            prev_warmup = result["warmup_kwh"]


# ===================================================================
# sols_to_full_load
# ===================================================================

class TestSolsToFullLoad:
    """Propellant load time estimation."""

    def test_fresh_reactor_finite(self):
        """Fresh reactor can produce full load in finite time."""
        r = SabatierReactor(max_ch4_per_sol_kg=10)
        sols = sols_to_full_load(r)
        assert sols > 0
        assert sols < 100_000

    def test_dead_reactor_returns_negative(self):
        """Dead reactor returns -1."""
        r = SabatierReactor(catalyst_efficiency=0.0)
        assert sols_to_full_load(r) == -1

    def test_bigger_reactor_faster(self):
        """Higher production rate → fewer sols."""
        r_small = SabatierReactor(max_ch4_per_sol_kg=20)
        r_big = SabatierReactor(max_ch4_per_sol_kg=100)
        s_small = sols_to_full_load(r_small)
        s_big = sols_to_full_load(r_big)
        assert s_small > 0, "small reactor must finish before catalyst dies"
        assert s_big > 0, "big reactor must finish before catalyst dies"
        assert s_big < s_small

    def test_custom_target(self):
        """Custom propellant target works."""
        r = SabatierReactor(max_ch4_per_sol_kg=100)
        sols = sols_to_full_load(r, ch4_target_kg=100, o2_target_kg=100)
        assert sols > 0
        assert sols < 50


# ===================================================================
# propellant_status
# ===================================================================

class TestPropellantStatus:
    """Return-trip readiness checks."""

    def test_empty_tanks_not_ready(self):
        """Empty tanks → not launch ready."""
        ch4 = PropellantTank(capacity_kg=60000, level_kg=0)
        o2 = PropellantTank(capacity_kg=200000, level_kg=0)
        status = propellant_status(ch4, o2)
        assert status["launch_ready"] is False
        assert status["overall_percent"] == 0.0

    def test_full_tanks_ready(self):
        """Full tanks → launch ready."""
        ch4 = PropellantTank(capacity_kg=60000, level_kg=STARSHIP_CH4_KG)
        o2 = PropellantTank(capacity_kg=200000, level_kg=STARSHIP_O2_KG)
        status = propellant_status(ch4, o2)
        assert status["launch_ready"] is True
        assert status["overall_percent"] >= 100.0

    def test_half_ch4_half_status(self):
        """Half CH4 → ~50% CH4 readiness."""
        ch4 = PropellantTank(capacity_kg=60000, level_kg=STARSHIP_CH4_KG / 2)
        o2 = PropellantTank(capacity_kg=200000, level_kg=STARSHIP_O2_KG)
        status = propellant_status(ch4, o2)
        assert 49 < status["ch4_percent"] < 51
        assert status["launch_ready"] is False

    def test_overall_is_minimum(self):
        """Overall readiness is the minimum of CH4 and O2 percentages."""
        ch4 = PropellantTank(capacity_kg=60000, level_kg=STARSHIP_CH4_KG * 0.3)
        o2 = PropellantTank(capacity_kg=200000, level_kg=STARSHIP_O2_KG * 0.8)
        status = propellant_status(ch4, o2)
        assert status["overall_percent"] == pytest.approx(status["ch4_percent"])


# ===================================================================
# Multi-sol smoke test
# ===================================================================

class TestMultiSolSmoke:
    """Run the simulation for many sols — must not crash."""

    def test_100_sol_run(self):
        """100 consecutive sols without crash, tanks accumulate."""
        reactor = SabatierReactor(max_ch4_per_sol_kg=10)
        ch4_tank = PropellantTank(capacity_kg=DEFAULT_CH4_TANK_KG)
        o2_tank = PropellantTank(capacity_kg=DEFAULT_O2_TANK_KG)

        for sol in range(100):
            temp = -60 + 30 * math.sin(sol * 2 * math.pi / 668)
            result = tick_fuel_production(
                reactor, ch4_tank, o2_tank,
                water_available_kg=200.0,
                power_available_kwh=300.0,
                ambient_temp_c=temp,
            )
            assert result["ch4_tank_kg"] >= 0
            assert result["o2_tank_kg"] >= 0

        # After 100 sols, should have meaningful propellant
        assert ch4_tank.level_kg > 0
        assert o2_tank.level_kg > 0
        assert reactor.total_sols_run == 100

    def test_catalyst_lifecycle(self):
        """Run until catalyst dies — reactor must shut down gracefully."""
        reactor = SabatierReactor(max_ch4_per_sol_kg=10, catalyst_efficiency=0.42)
        ch4_tank = PropellantTank(capacity_kg=DEFAULT_CH4_TANK_KG)
        o2_tank = PropellantTank(capacity_kg=DEFAULT_O2_TANK_KG)

        operational_sols = 0
        for sol in range(10000):
            result = tick_fuel_production(
                reactor, ch4_tank, o2_tank, 200, 300, -60
            )
            if result["operational"]:
                operational_sols += 1
            else:
                break

        assert operational_sols > 0
        assert not reactor.is_operational()

    def test_long_run_propellant_readiness(self):
        """With a large reactor, should approach launch readiness over time."""
        reactor = SabatierReactor(max_ch4_per_sol_kg=100)
        ch4_tank = PropellantTank(capacity_kg=DEFAULT_CH4_TANK_KG)
        o2_tank = PropellantTank(capacity_kg=DEFAULT_O2_TANK_KG)

        for sol in range(2000):
            tick_fuel_production(
                reactor, ch4_tank, o2_tank, 5000, 10000, -60
            )

        status = propellant_status(ch4_tank, o2_tank)
        # A 100 kg/sol reactor running 2000 sols should produce significant CH4
        assert status["ch4_percent"] > 10  # at least some progress
