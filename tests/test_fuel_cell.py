"""test_fuel_cell.py -- 75+ tests for Mars Colony PEM Fuel Cell.

Tests cover:
  - Physical constants and stoichiometric ratios
  - Nernst equation voltage calculations
  - Pure physics functions (consumption, energy, heat, water production)
  - Mass conservation (H₂ + O₂ in = H₂O out)
  - Energy conservation (1st law: chemical = electric + heat)
  - Stack degradation over time
  - Cold start behavior
  - End-of-life shutdown
  - Fuel/oxidizer limiting
  - Emergency mode triggers
  - Factory profiles
  - Multi-sol simulation (10+ ticks without crash)
  - Property-based invariants (outputs in physical bounds)
"""
from __future__ import annotations

import math
import pytest
import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fuel_cell import (
    # Constants
    E_REVERSIBLE_V, FARADAY_C, N_ELECTRONS, GIBBS_FREE_ENERGY_J,
    H2_MOLAR_MASS_KG, O2_MOLAR_MASS_KG, H2O_MOLAR_MASS_KG,
    HHV_H2_KWH_KG, LHV_H2_KWH_KG, O2_PER_H2_MASS, H2O_PER_H2_MASS,
    NOMINAL_CELL_VOLTAGE_V, MIN_CELL_VOLTAGE_V, CELLS_PER_STACK,
    EOL_VOLTAGE_FRACTION, MARS_AMBIENT_C, SOL_HOURS,
    # Classes
    FuelCellStack, FuelCellState, TickResult,
    # Functions
    nernst_voltage, h2_consumption_rate, energy_from_h2, heat_from_h2,
    water_from_h2, o2_for_h2, membrane_warmup, tick, create_fuel_cell,
)


# =========================================================================
# SECTION 1: Physical constants validation
# =========================================================================

class TestPhysicalConstants:
    """Verify constants are within known physical ranges."""

    def test_reversible_voltage_range(self):
        """E° for H₂/O₂ fuel cell should be ~1.229 V."""
        assert 1.22 < E_REVERSIBLE_V < 1.24

    def test_reversible_voltage_from_gibbs(self):
        """E° = ΔG / (n·F) should match stored constant."""
        computed = GIBBS_FREE_ENERGY_J / (N_ELECTRONS * FARADAY_C)
        assert abs(computed - E_REVERSIBLE_V) < 0.001

    def test_faraday_constant(self):
        """Faraday constant ~96485 C/mol."""
        assert 96_400 < FARADAY_C < 96_600

    def test_h2_molar_mass(self):
        """H₂ molar mass ~2.016 g/mol."""
        assert abs(H2_MOLAR_MASS_KG - 0.002016) < 1e-6

    def test_o2_molar_mass(self):
        """O₂ molar mass ~32 g/mol."""
        assert abs(O2_MOLAR_MASS_KG - 0.032) < 1e-6

    def test_h2o_molar_mass(self):
        """H₂O molar mass ~18.015 g/mol."""
        assert abs(H2O_MOLAR_MASS_KG - 0.018015) < 1e-6

    def test_stoichiometric_o2_per_h2(self):
        """1 kg H₂ needs ~7.937 kg O₂ (from 2H₂ + O₂ → 2H₂O)."""
        expected = O2_MOLAR_MASS_KG / (2 * H2_MOLAR_MASS_KG)
        assert abs(O2_PER_H2_MASS - expected) < 0.01

    def test_stoichiometric_h2o_per_h2(self):
        """1 kg H₂ produces ~8.937 kg H₂O."""
        expected = (2 * H2O_MOLAR_MASS_KG) / (2 * H2_MOLAR_MASS_KG)
        assert abs(H2O_PER_H2_MASS - expected) < 0.01

    def test_mass_balance_stoichiometry(self):
        """H₂ + O₂ = H₂O by mass (1 + 7.937 = 8.937)."""
        assert abs((1.0 + O2_PER_H2_MASS) - H2O_PER_H2_MASS) < 0.01

    def test_hhv_gt_lhv(self):
        """Higher heating value > lower heating value for H₂."""
        assert HHV_H2_KWH_KG > LHV_H2_KWH_KG

    def test_nominal_voltage_below_reversible(self):
        """Operating voltage must be below thermodynamic limit."""
        assert NOMINAL_CELL_VOLTAGE_V < E_REVERSIBLE_V

    def test_min_voltage_positive(self):
        """Minimum cell voltage must be positive."""
        assert MIN_CELL_VOLTAGE_V > 0

    def test_sol_hours(self):
        """Mars sol is ~24.66 Earth hours."""
        assert 24.6 < SOL_HOURS < 24.7


# =========================================================================
# SECTION 2: Nernst equation
# =========================================================================

class TestNernstVoltage:
    """Test the Nernst voltage calculation."""

    def test_stp_voltage(self):
        """At 25°C, 1 atm H₂, 0.21 atm O₂: voltage near E°."""
        v = nernst_voltage(25.0, 1.0, 0.21)
        # Should be close to E° but slightly lower due to low pO2
        assert 1.15 < v < 1.25

    def test_higher_pressure_higher_voltage(self):
        """Higher reactant pressure → higher voltage (Le Chatelier)."""
        v_low = nernst_voltage(25.0, 1.0, 0.21)
        v_high = nernst_voltage(25.0, 5.0, 1.0)
        assert v_high > v_low

    def test_higher_temp_lower_voltage(self):
        """Higher temp → slightly lower reversible voltage for H₂/O₂."""
        v_cold = nernst_voltage(25.0, 1.0, 1.0)
        v_hot = nernst_voltage(80.0, 1.0, 1.0)
        # Temperature coefficient is negative for H₂/O₂ fuel cell
        assert v_hot < v_cold

    def test_zero_kelvin_returns_zero(self):
        """At absolute zero (or below), return 0."""
        assert nernst_voltage(-273.15, 1.0, 1.0) == 0.0
        assert nernst_voltage(-300.0, 1.0, 1.0) == 0.0

    def test_zero_pressure_returns_zero(self):
        """Zero pressure → no reaction possible."""
        assert nernst_voltage(25.0, 0.0, 1.0) == 0.0
        assert nernst_voltage(25.0, 1.0, 0.0) == 0.0

    def test_negative_pressure_returns_zero(self):
        """Negative pressure is unphysical."""
        assert nernst_voltage(25.0, -1.0, 1.0) == 0.0

    def test_voltage_always_non_negative(self):
        """Nernst voltage should never go negative."""
        for t in [-50, 0, 25, 100, 200]:
            for p in [0.01, 0.1, 1.0, 10.0]:
                assert nernst_voltage(t, p, p) >= 0.0


# =========================================================================
# SECTION 3: Pure physics functions
# =========================================================================

class TestH2ConsumptionRate:
    """Test hydrogen consumption rate calculation."""

    def test_positive_power_positive_consumption(self):
        """Drawing power should consume hydrogen."""
        rate = h2_consumption_rate(10.0, 0.7)
        assert rate > 0

    def test_zero_power_zero_consumption(self):
        """No power demand → no hydrogen consumed."""
        assert h2_consumption_rate(0.0, 0.7) == 0.0

    def test_higher_power_more_consumption(self):
        """More power → more H₂ consumed."""
        r1 = h2_consumption_rate(5.0, 0.7)
        r2 = h2_consumption_rate(10.0, 0.7)
        assert r2 > r1

    def test_lower_voltage_more_consumption(self):
        """Lower efficiency (voltage) → more H₂ per kWh."""
        r_high_v = h2_consumption_rate(10.0, 0.8)
        r_low_v = h2_consumption_rate(10.0, 0.5)
        assert r_low_v > r_high_v

    def test_zero_voltage_zero_consumption(self):
        """Dead cell → can't produce power → no consumption."""
        assert h2_consumption_rate(10.0, 0.0) == 0.0

    def test_negative_power_zero_consumption(self):
        """Negative power demand is meaningless."""
        assert h2_consumption_rate(-5.0, 0.7) == 0.0


class TestEnergyFromH2:
    """Test electrical energy production from hydrogen."""

    def test_positive_h2_positive_energy(self):
        """Consuming H₂ should produce energy."""
        assert energy_from_h2(1.0, 0.7) > 0

    def test_zero_h2_zero_energy(self):
        """No fuel → no energy."""
        assert energy_from_h2(0.0, 0.7) == 0.0

    def test_energy_bounded_by_lhv(self):
        """Energy per kg can't exceed LHV × 100% efficiency."""
        e = energy_from_h2(1.0, 0.7)
        assert e <= LHV_H2_KWH_KG * 1.0  # can't exceed 100% of LHV

    def test_higher_voltage_more_energy(self):
        """Higher cell voltage → more electrical energy per kg H₂."""
        e_low = energy_from_h2(1.0, 0.5)
        e_high = energy_from_h2(1.0, 0.8)
        assert e_high > e_low

    def test_negative_h2_zero_energy(self):
        """Negative fuel is unphysical."""
        assert energy_from_h2(-1.0, 0.7) == 0.0


class TestHeatFromH2:
    """Test waste heat production."""

    def test_positive_h2_positive_heat(self):
        """Reaction always produces some waste heat."""
        assert heat_from_h2(1.0, 0.7) > 0

    def test_zero_h2_zero_heat(self):
        """No fuel → no heat."""
        assert heat_from_h2(0.0, 0.7) == 0.0

    def test_energy_plus_heat_equals_chemical(self):
        """First law: electrical + heat = chemical energy (LHV)."""
        h2 = 2.5
        v = 0.65
        e = energy_from_h2(h2, v)
        q = heat_from_h2(h2, v)
        chemical = h2 * LHV_H2_KWH_KG
        assert abs(e + q - chemical) < 1e-6

    def test_lower_voltage_more_heat(self):
        """Less efficient cell → more waste heat."""
        q_high_v = heat_from_h2(1.0, 0.8)
        q_low_v = heat_from_h2(1.0, 0.5)
        assert q_low_v > q_high_v


class TestWaterAndO2:
    """Test stoichiometric calculations."""

    def test_water_from_h2_positive(self):
        """H₂ consumption produces water."""
        assert water_from_h2(1.0) > 0

    def test_water_mass_balance(self):
        """1 kg H₂ + O₂ needed = water produced (mass conservation)."""
        h2 = 3.0
        o2 = o2_for_h2(h2)
        water = water_from_h2(h2)
        assert abs((h2 + o2) - water) < 1e-6

    def test_o2_for_h2_ratio(self):
        """O₂:H₂ mass ratio should be ~7.937."""
        assert abs(o2_for_h2(1.0) - O2_PER_H2_MASS) < 1e-6

    def test_water_per_kg_h2(self):
        """1 kg H₂ → ~8.937 kg H₂O."""
        assert abs(water_from_h2(1.0) - H2O_PER_H2_MASS) < 1e-6

    def test_zero_h2_zero_products(self):
        """No fuel → no products."""
        assert water_from_h2(0.0) == 0.0
        assert o2_for_h2(0.0) == 0.0


# =========================================================================
# SECTION 4: FuelCellStack
# =========================================================================

class TestFuelCellStack:
    """Test stack state and degradation model."""

    def test_new_stack_full_health(self):
        """Fresh stack has health = 1.0."""
        s = FuelCellStack()
        assert s.health() == 1.0

    def test_new_stack_not_eol(self):
        """Fresh stack is not end-of-life."""
        assert not FuelCellStack().is_eol()

    def test_cell_voltage_degrades(self):
        """Voltage drops with operating time."""
        s = FuelCellStack()
        v0 = s.cell_voltage()
        s.operating_sols = 100
        v100 = s.cell_voltage()
        assert v100 < v0

    def test_cell_voltage_floor(self):
        """Voltage never drops below MIN_CELL_VOLTAGE_V."""
        s = FuelCellStack(operating_sols=100_000)
        assert s.cell_voltage() >= MIN_CELL_VOLTAGE_V

    def test_stack_voltage_is_cells_times_cell_v(self):
        """Stack voltage = num_cells × cell_voltage."""
        s = FuelCellStack()
        assert abs(s.stack_voltage() - s.num_cells * s.cell_voltage()) < 1e-9

    def test_eol_after_long_operation(self):
        """Stack reaches end-of-life after enough sols."""
        s = FuelCellStack()
        # health < 0.80 → EOL.  nominal=0.70, decay=0.15mV/sol
        # 0.70 × 0.80 = 0.56V threshold.  decay = 0.00015V/sol
        # sols to EOL = (0.70 - 0.56) / 0.00015 = 933 sols
        s.operating_sols = 1000
        assert s.is_eol()

    def test_electrical_efficiency_range(self):
        """Electrical efficiency: 0 < η_e < 1 for valid stack."""
        s = FuelCellStack()
        eta = s.electrical_efficiency()
        assert 0 < eta < 1.0

    def test_chp_efficiency_gt_electrical(self):
        """CHP efficiency always exceeds electrical efficiency."""
        s = FuelCellStack()
        assert s.chp_efficiency() > s.electrical_efficiency()

    def test_chp_efficiency_capped(self):
        """CHP efficiency never exceeds 95%."""
        s = FuelCellStack()
        assert s.chp_efficiency() <= 0.95


# =========================================================================
# SECTION 5: FuelCellState
# =========================================================================

class TestFuelCellState:
    """Test system-level state."""

    def test_full_tanks(self):
        """Fresh state has full H₂ and O₂."""
        s = FuelCellState()
        assert s.h2_fill() == 1.0
        assert s.o2_fill() == 1.0

    def test_runtime_estimate_positive(self):
        """Runtime at moderate power should be many hours."""
        s = FuelCellState()
        hours = s.runtime_hours(5.0)
        assert hours > 10  # 50 kg H₂ at 5 kW → many hours

    def test_runtime_zero_power_infinite(self):
        """Zero power demand → infinite runtime."""
        s = FuelCellState()
        assert s.runtime_hours(0.0) == float("inf")

    def test_empty_tanks_zero_runtime(self):
        """Empty tanks → zero runtime."""
        s = FuelCellState(h2_supply_kg=0.0, o2_supply_kg=0.0)
        assert s.runtime_hours(5.0) == 0.0


# =========================================================================
# SECTION 6: Tick function (integration tests)
# =========================================================================

class TestTick:
    """Test the main simulation tick."""

    def test_idle_tick_no_consumption(self):
        """Zero power demand → no fuel consumed."""
        state = create_fuel_cell()
        result = tick(state, power_demand_kw=0.0)
        assert result.h2_consumed_kg == 0.0
        assert result.o2_consumed_kg == 0.0
        assert result.energy_produced_kwh == 0.0

    def test_active_tick_produces_energy(self):
        """Positive power demand → energy produced."""
        state = create_fuel_cell()
        result = tick(state, power_demand_kw=5.0)
        assert result.energy_produced_kwh > 0

    def test_active_tick_consumes_h2(self):
        """Active cell consumes hydrogen."""
        state = create_fuel_cell()
        h2_before = state.h2_supply_kg
        tick(state, power_demand_kw=5.0)
        assert state.h2_supply_kg < h2_before

    def test_active_tick_consumes_o2(self):
        """Active cell consumes oxygen."""
        state = create_fuel_cell()
        o2_before = state.o2_supply_kg
        tick(state, power_demand_kw=5.0)
        assert state.o2_supply_kg < o2_before

    def test_active_tick_produces_water(self):
        """Fuel cell produces water as byproduct."""
        state = create_fuel_cell()
        result = tick(state, power_demand_kw=5.0)
        assert result.water_produced_kg > 0

    def test_active_tick_produces_heat(self):
        """Fuel cell produces waste heat."""
        state = create_fuel_cell()
        result = tick(state, power_demand_kw=5.0)
        assert result.heat_produced_kwh > 0

    def test_mass_conservation_per_tick(self):
        """H₂ + O₂ consumed = H₂O produced (every tick)."""
        state = create_fuel_cell()
        result = tick(state, power_demand_kw=8.0)
        mass_in = result.h2_consumed_kg + result.o2_consumed_kg
        mass_out = result.water_produced_kg
        assert abs(mass_in - mass_out) < 1e-5

    def test_energy_conservation_per_tick(self):
        """Electric + heat = chemical energy (1st law, every tick)."""
        state = create_fuel_cell()
        result = tick(state, power_demand_kw=8.0)
        chemical = result.h2_consumed_kg * LHV_H2_KWH_KG
        total_out = result.energy_produced_kwh + result.heat_produced_kwh
        assert abs(chemical - total_out) < 1e-4

    def test_cold_start_from_mars_ambient(self):
        """Starting from Mars ambient triggers cold start heater."""
        state = create_fuel_cell()
        state.stack.membrane_temp_c = MARS_AMBIENT_C
        result = tick(state, power_demand_kw=5.0)
        assert any("COLD_START" in w for w in result.warnings)
        assert state.cold_start_energy_kwh > 0

    def test_h2_limited_operation(self):
        """When H₂ is low, cell is fuel-limited."""
        state = create_fuel_cell()
        state.h2_supply_kg = 0.1  # very low
        result = tick(state, power_demand_kw=10.0)
        assert any("H2_L" in w for w in result.warnings)
        # Should have consumed all available H₂
        assert state.h2_supply_kg < 0.01

    def test_o2_limited_operation(self):
        """When O₂ is low, cell is oxidizer-limited."""
        state = create_fuel_cell()
        state.o2_supply_kg = 0.5  # not enough for full H₂
        state.h2_supply_kg = 50.0
        result = tick(state, power_demand_kw=10.0)
        assert any("O2_L" in w for w in result.warnings)

    def test_eol_stack_refuses_to_run(self):
        """End-of-life stack shuts down."""
        state = create_fuel_cell()
        state.stack.operating_sols = 2000  # way past EOL
        result = tick(state, power_demand_kw=5.0)
        assert any("END_OF_LIFE" in w for w in result.warnings)
        assert result.energy_produced_kwh == 0.0

    def test_empty_fuel_no_crash(self):
        """Completely empty tanks → graceful zero output."""
        state = create_fuel_cell()
        state.h2_supply_kg = 0.0
        state.o2_supply_kg = 0.0
        result = tick(state, power_demand_kw=5.0)
        assert result.energy_produced_kwh == 0.0
        assert result.h2_consumed_kg == 0.0

    def test_sol_fraction(self):
        """Half a sol produces roughly half the energy of a full sol."""
        s1 = create_fuel_cell()
        s2 = create_fuel_cell()
        r_full = tick(s1, power_demand_kw=5.0, sol_fraction=1.0)
        r_half = tick(s2, power_demand_kw=5.0, sol_fraction=0.5)
        ratio = r_half.energy_produced_kwh / r_full.energy_produced_kwh
        assert 0.45 < ratio < 0.55

    def test_degradation_increases_per_tick(self):
        """Operating sols increase after each active tick."""
        state = create_fuel_cell()
        sols_before = state.stack.operating_sols
        tick(state, power_demand_kw=5.0)
        assert state.stack.operating_sols > sols_before

    def test_idle_no_degradation(self):
        """Idle ticks don't degrade the stack."""
        state = create_fuel_cell()
        sols_before = state.stack.operating_sols
        tick(state, power_demand_kw=0.0)
        assert state.stack.operating_sols == sols_before


# =========================================================================
# SECTION 7: Factory profiles
# =========================================================================

class TestFactory:
    """Test create_fuel_cell profiles."""

    def test_colony_profile(self):
        """Colony profile has 50 kg H₂."""
        s = create_fuel_cell("colony")
        assert s.h2_capacity_kg == 50.0

    def test_rover_profile(self):
        """Rover profile has 5 kg H₂, fewer cells."""
        s = create_fuel_cell("rover")
        assert s.h2_capacity_kg == 5.0
        assert s.stack.num_cells == 60

    def test_emergency_profile(self):
        """Emergency profile has 10 kg H₂, minimal stack."""
        s = create_fuel_cell("emergency")
        assert s.h2_capacity_kg == 10.0
        assert s.stack.num_cells == 30

    def test_unknown_profile_defaults_to_colony(self):
        """Unknown profile name falls back to colony."""
        s = create_fuel_cell("unknown")
        assert s.h2_capacity_kg == 50.0


# =========================================================================
# SECTION 8: Multi-sol simulation (smoke test)
# =========================================================================

class TestMultiSolSimulation:
    """Run the fuel cell for many sols — the organism must survive."""

    def test_10_sol_smoke(self):
        """10 sols at 5 kW — no crash, monotone fuel decrease."""
        state = create_fuel_cell()
        prev_h2 = state.h2_supply_kg
        for sol in range(10):
            result = tick(state, power_demand_kw=5.0)
            assert state.h2_supply_kg <= prev_h2
            assert result.energy_produced_kwh >= 0
            prev_h2 = state.h2_supply_kg

    def test_50_sol_run_to_exhaustion(self):
        """Run until fuel runs out — cell should gracefully deplete."""
        state = create_fuel_cell("emergency")  # small tank
        total_energy = 0.0
        total_water = 0.0
        for sol in range(200):
            result = tick(state, power_demand_kw=3.0)
            total_energy += result.energy_produced_kwh
            total_water += result.water_produced_kg
            if state.h2_supply_kg <= 0:
                break
        # Should have produced meaningful energy before exhaustion
        assert total_energy > 50
        # Water produced should match stoichiometry
        total_h2_used = state.stack.total_h2_consumed_kg
        expected_water = water_from_h2(total_h2_used)
        assert abs(total_water - expected_water) < 0.01

    def test_variable_load_profile(self):
        """Simulate day/night cycling with variable power demands."""
        state = create_fuel_cell()
        loads = [10, 8, 5, 2, 0, 0, 2, 5, 8, 10]  # kW per sol
        for load in loads:
            result = tick(state, power_demand_kw=float(load))
            assert result.energy_produced_kwh >= 0
            assert result.stack_health > 0

    def test_cumulative_water_matches_h2(self):
        """Over multiple ticks, total water = stoichiometric from total H₂."""
        state = create_fuel_cell()
        for _ in range(20):
            tick(state, power_demand_kw=4.0)
        expected_water = water_from_h2(state.stack.total_h2_consumed_kg)
        assert abs(state.water_produced_kg - expected_water) < 0.01


# =========================================================================
# SECTION 9: Property-based invariants
# =========================================================================

class TestInvariants:
    """Property-based checks — outputs must stay in physical bounds."""

    @pytest.mark.parametrize("power_kw", [0, 1, 5, 10, 20, 50])
    def test_efficiency_in_bounds(self, power_kw):
        """Electrical efficiency must be 0–1."""
        state = create_fuel_cell()
        result = tick(state, power_demand_kw=float(power_kw))
        assert 0.0 <= result.electrical_efficiency <= 1.0

    @pytest.mark.parametrize("power_kw", [0, 1, 5, 10, 20])
    def test_stack_health_in_bounds(self, power_kw):
        """Stack health must be 0–1."""
        state = create_fuel_cell()
        result = tick(state, power_demand_kw=float(power_kw))
        assert 0.0 < result.stack_health <= 1.0

    @pytest.mark.parametrize("power_kw", [0, 5, 15])
    def test_non_negative_outputs(self, power_kw):
        """All outputs must be non-negative."""
        state = create_fuel_cell()
        result = tick(state, power_demand_kw=float(power_kw))
        assert result.energy_produced_kwh >= 0
        assert result.heat_produced_kwh >= 0
        assert result.h2_consumed_kg >= 0
        assert result.o2_consumed_kg >= 0
        assert result.water_produced_kg >= 0

    @pytest.mark.parametrize("power_kw", [1, 5, 10])
    def test_fuel_monotonically_decreasing(self, power_kw):
        """H₂ supply can only decrease under load."""
        state = create_fuel_cell()
        for _ in range(5):
            prev = state.h2_supply_kg
            tick(state, power_demand_kw=float(power_kw))
            assert state.h2_supply_kg <= prev

    def test_h2_supply_never_negative(self):
        """H₂ supply must never go below zero."""
        state = create_fuel_cell("emergency")
        for _ in range(500):
            tick(state, power_demand_kw=10.0)
            assert state.h2_supply_kg >= -1e-9

    def test_o2_supply_never_negative(self):
        """O₂ supply must never go below zero."""
        state = create_fuel_cell("emergency")
        for _ in range(500):
            tick(state, power_demand_kw=10.0)
            assert state.o2_supply_kg >= -1e-9


# =========================================================================
# SECTION 10: Membrane warmup
# =========================================================================

class TestMembraneWarmup:
    """Test thermal model for membrane temperature."""

    def test_heat_warms_membrane(self):
        """Adding heat should raise temperature."""
        t0 = 25.0
        t1 = membrane_warmup(t0, 5.0)
        assert t1 > t0

    def test_no_heat_cools_toward_ambient(self):
        """Without heat input, membrane cools toward Mars ambient."""
        t0 = 50.0
        t1 = membrane_warmup(t0, 0.0)
        assert t1 < t0

    def test_at_ambient_no_cooling(self):
        """At Mars ambient with no heat, temperature stays."""
        t1 = membrane_warmup(MARS_AMBIENT_C, 0.0)
        assert abs(t1 - MARS_AMBIENT_C) < 0.01

    def test_zero_thermal_mass_no_change(self):
        """Zero thermal mass → no temperature change."""
        t1 = membrane_warmup(25.0, 5.0, thermal_mass_kwh_per_c=0.0)
        assert t1 == 25.0
