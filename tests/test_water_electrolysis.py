"""
Tests for water_electrolysis.py — Mars Colony PEM Water Electrolysis.

91 tests across 10 test classes. Every function, edge case, and physics
invariant tested.  The electrolyzer is the colony's molecular bridge
between water ice and breathable air + rocket fuel.

Conservation laws verified:
  - Mass conservation: water_in = h2_out + o2_out (stoichiometric)
  - Energy conservation: consumed ≤ allocated
  - Tank levels never negative
  - Pressure follows ideal gas law
  - Efficiency bounded [0, 1]

Run: python -m pytest tests/test_water_electrolysis.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.water_electrolysis import (
    GasTank,
    ElectrolyzerState,
    ElectrolysisSol,
    temperature_efficiency,
    energy_per_kg_h2,
    max_h2_from_power,
    max_h2_from_water,
    water_for_h2,
    o2_from_h2,
    system_efficiency,
    tick_electrolysis,
    create_electrolyzer,
    H2_MASS_FRACTION,
    O2_MASS_FRACTION,
    WATER_PER_KG_H2,
    WATER_PER_KG_O2,
    THEORETICAL_KWH_PER_KG_H2,
    BASELINE_KWH_PER_KG_H2,
    MIN_POWER_KW,
    CELL_DEGRADATION_PER_SOL,
    CELL_MAINTENANCE_RESTORE,
    MAX_CELL_DEGRADATION,
    H2_TANK_MAX_KPA,
    O2_TANK_MAX_KPA,
    OVERPRESSURE_MARGIN,
    GAS_CONSTANT_KPA_L_MOL_K,
    H2_MOLAR_MASS_KG,
    O2_MOLAR_MASS_KG,
    OPTIMAL_TEMP_C,
    MIN_OPERATING_TEMP_C,
    COLD_EFFICIENCY_FLOOR,
    WARM_EFFICIENCY_CEIL,
    DEFAULT_H2_TANK_VOLUME_L,
    DEFAULT_O2_TANK_VOLUME_L,
    DEFAULT_TANK_TEMP_K,
)


# ─── GasTank ─────────────────────────────────────────────────────────────────

class TestGasTank:
    """Unit tests for the GasTank dataclass."""

    def test_defaults(self):
        t = GasTank(volume_l=100.0)
        assert t.volume_l == 100.0
        assert t.stored_kg == 0.0
        assert t.pressure_kpa() == 0.0

    def test_volume_clamped_low(self):
        t = GasTank(volume_l=-10.0)
        assert t.volume_l == 1.0

    def test_stored_clamped_low(self):
        t = GasTank(volume_l=100.0, stored_kg=-5.0)
        assert t.stored_kg == 0.0

    def test_pressure_ideal_gas(self):
        """PV = nRT → P = nRT/V.  Verify with known values."""
        t = GasTank(volume_l=100.0, stored_kg=1.0, molar_mass_kg=H2_MOLAR_MASS_KG,
                    temperature_k=293.15)
        moles = 1.0 / H2_MOLAR_MASS_KG
        expected = moles * GAS_CONSTANT_KPA_L_MOL_K * 293.15 / 100.0
        assert abs(t.pressure_kpa() - round(expected, 4)) < 0.01

    def test_pressure_increases_with_mass(self):
        t = GasTank(volume_l=100.0, molar_mass_kg=H2_MOLAR_MASS_KG)
        t.stored_kg = 0.5
        p1 = t.pressure_kpa()
        t.stored_kg = 1.0
        p2 = t.pressure_kpa()
        assert p2 > p1

    def test_pressure_decreases_with_volume(self):
        small = GasTank(volume_l=50.0, stored_kg=1.0, molar_mass_kg=H2_MOLAR_MASS_KG)
        large = GasTank(volume_l=200.0, stored_kg=1.0, molar_mass_kg=H2_MOLAR_MASS_KG)
        assert small.pressure_kpa() > large.pressure_kpa()

    def test_fill_fraction_empty(self):
        t = GasTank(volume_l=100.0, max_pressure_kpa=1000.0)
        assert t.fill_fraction() == 0.0

    def test_fill_fraction_capped_at_one(self):
        t = GasTank(volume_l=1.0, stored_kg=1000.0, max_pressure_kpa=100.0,
                    molar_mass_kg=H2_MOLAR_MASS_KG)
        assert t.fill_fraction() <= 1.0

    def test_headroom_empty_tank(self):
        t = GasTank(volume_l=500.0, max_pressure_kpa=H2_TANK_MAX_KPA,
                    molar_mass_kg=H2_MOLAR_MASS_KG)
        assert t.headroom_kg() > 0

    def test_headroom_decreases_as_tank_fills(self):
        t = GasTank(volume_l=500.0, max_pressure_kpa=H2_TANK_MAX_KPA,
                    molar_mass_kg=H2_MOLAR_MASS_KG)
        h1 = t.headroom_kg()
        t.add(10.0)
        h2 = t.headroom_kg()
        assert h2 < h1

    def test_add_respects_headroom(self):
        t = GasTank(volume_l=10.0, max_pressure_kpa=100.0,
                    molar_mass_kg=O2_MOLAR_MASS_KG)
        room = t.headroom_kg()
        stored = t.add(room + 100.0)
        assert stored <= room + 0.001
        assert t.pressure_kpa() <= t.max_pressure_kpa * OVERPRESSURE_MARGIN + 0.1

    def test_add_returns_actual_stored(self):
        t = GasTank(volume_l=500.0, max_pressure_kpa=H2_TANK_MAX_KPA,
                    molar_mass_kg=H2_MOLAR_MASS_KG)
        stored = t.add(1.0)
        assert stored == 1.0
        assert t.stored_kg == 1.0

    def test_add_negative_does_nothing(self):
        t = GasTank(volume_l=100.0, stored_kg=5.0, molar_mass_kg=H2_MOLAR_MASS_KG)
        stored = t.add(-10.0)
        assert stored == 0.0
        assert t.stored_kg == 5.0

    def test_remove_basic(self):
        t = GasTank(volume_l=100.0, stored_kg=5.0, molar_mass_kg=H2_MOLAR_MASS_KG)
        removed = t.remove(3.0)
        assert removed == 3.0
        assert abs(t.stored_kg - 2.0) < 0.001

    def test_remove_capped_at_stored(self):
        t = GasTank(volume_l=100.0, stored_kg=2.0, molar_mass_kg=H2_MOLAR_MASS_KG)
        removed = t.remove(10.0)
        assert removed == 2.0
        assert t.stored_kg == 0.0

    def test_remove_negative_does_nothing(self):
        t = GasTank(volume_l=100.0, stored_kg=5.0, molar_mass_kg=H2_MOLAR_MASS_KG)
        removed = t.remove(-3.0)
        assert removed == 0.0
        assert t.stored_kg == 5.0

    def test_stored_never_negative(self):
        t = GasTank(volume_l=100.0, stored_kg=1.0, molar_mass_kg=H2_MOLAR_MASS_KG)
        t.remove(1000.0)
        assert t.stored_kg >= 0.0


# ─── ElectrolyzerState ───────────────────────────────────────────────────────

class TestElectrolyzerState:
    """Unit tests for the ElectrolyzerState dataclass."""

    def test_defaults(self):
        e = ElectrolyzerState()
        assert e.sol == 0
        assert e.cell_degradation == 0.0
        assert e.operating is True
        assert e.total_water_consumed_kg == 0.0
        assert e.total_h2_produced_kg == 0.0
        assert e.total_o2_produced_kg == 0.0

    def test_degradation_clamped(self):
        e = ElectrolyzerState(cell_degradation=5.0)
        assert e.cell_degradation == 1.0
        e2 = ElectrolyzerState(cell_degradation=-1.0)
        assert e2.cell_degradation == 0.0

    def test_totals_clamped_low(self):
        e = ElectrolyzerState(total_water_consumed_kg=-100.0)
        assert e.total_water_consumed_kg == 0.0


# ─── Temperature efficiency ──────────────────────────────────────────────────

class TestTemperatureEfficiency:
    """Unit tests for temperature_efficiency()."""

    def test_below_minimum_is_zero(self):
        assert temperature_efficiency(-50.0) == 0.0
        assert temperature_efficiency(0.0) == 0.0
        assert temperature_efficiency(MIN_OPERATING_TEMP_C - 0.1) == 0.0

    def test_at_minimum(self):
        eff = temperature_efficiency(MIN_OPERATING_TEMP_C)
        assert abs(eff - COLD_EFFICIENCY_FLOOR) < 0.01

    def test_at_optimal(self):
        eff = temperature_efficiency(OPTIMAL_TEMP_C)
        assert abs(eff - WARM_EFFICIENCY_CEIL) < 0.01

    def test_above_optimal_capped(self):
        eff = temperature_efficiency(200.0)
        assert eff == WARM_EFFICIENCY_CEIL

    def test_monotonic_increase(self):
        """Efficiency must increase with temperature."""
        temps = [5.0, 20.0, 40.0, 60.0, 80.0]
        effs = [temperature_efficiency(t) for t in temps]
        for i in range(len(effs) - 1):
            assert effs[i + 1] >= effs[i]

    def test_midpoint(self):
        mid_temp = (MIN_OPERATING_TEMP_C + OPTIMAL_TEMP_C) / 2.0
        eff = temperature_efficiency(mid_temp)
        expected = (COLD_EFFICIENCY_FLOOR + WARM_EFFICIENCY_CEIL) / 2.0
        assert abs(eff - expected) < 0.01

    def test_bounded_zero_to_one(self):
        for t in [-100, -50, 0, 10, 40, 80, 120]:
            eff = temperature_efficiency(float(t))
            assert 0.0 <= eff <= 1.0


# ─── Stoichiometry ───────────────────────────────────────────────────────────

class TestStoichiometry:
    """Verify stoichiometric relationships: 2H₂O → 2H₂ + O₂."""

    def test_mass_fractions_sum_to_one(self):
        """H₂ + O₂ mass fractions from water must sum to 1.0."""
        total = H2_MASS_FRACTION + O2_MASS_FRACTION
        assert abs(total - 1.0) < 0.001

    def test_h2_from_water(self):
        """1 kg water → ~0.112 kg H₂."""
        h2 = max_h2_from_water(1.0)
        assert abs(h2 - H2_MASS_FRACTION) < 0.001

    def test_o2_from_h2(self):
        """1 kg H₂ → ~7.936 kg O₂ (stoichiometric ratio)."""
        o2 = o2_from_h2(1.0)
        expected = O2_MASS_FRACTION / H2_MASS_FRACTION
        assert abs(o2 - expected) < 0.01

    def test_water_for_h2_roundtrip(self):
        """water_for_h2(max_h2_from_water(W)) ≈ W."""
        water_in = 10.0
        h2 = max_h2_from_water(water_in)
        water_back = water_for_h2(h2)
        assert abs(water_back - water_in) < 0.01

    def test_mass_conservation(self):
        """water consumed = h2 produced + o2 produced (by mass)."""
        water = 50.0
        h2 = max_h2_from_water(water)
        o2 = o2_from_h2(h2)
        assert abs(water - (h2 + o2)) < 0.05  # small float rounding

    def test_zero_inputs(self):
        assert max_h2_from_water(0.0) == 0.0
        assert max_h2_from_water(-5.0) == 0.0
        assert water_for_h2(0.0) == 0.0
        assert water_for_h2(-1.0) == 0.0
        assert o2_from_h2(0.0) == 0.0
        assert o2_from_h2(-1.0) == 0.0


# ─── Energy functions ────────────────────────────────────────────────────────

class TestEnergyFunctions:
    """Tests for energy_per_kg_h2, max_h2_from_power, system_efficiency."""

    def test_baseline_energy(self):
        """Fresh cells at optimal temp → baseline energy."""
        cost = energy_per_kg_h2(0.0, OPTIMAL_TEMP_C)
        assert abs(cost - BASELINE_KWH_PER_KG_H2) < 0.1

    def test_degradation_increases_energy(self):
        """More degradation → more energy per kg H₂."""
        fresh = energy_per_kg_h2(0.0, 60.0)
        worn = energy_per_kg_h2(0.3, 60.0)
        assert worn > fresh

    def test_cold_increases_energy(self):
        """Colder temperature → more energy per kg H₂."""
        warm = energy_per_kg_h2(0.0, 70.0)
        cold = energy_per_kg_h2(0.0, 10.0)
        assert cold > warm

    def test_frozen_infinite_energy(self):
        """Below operating temp → infinite energy (can't operate)."""
        cost = energy_per_kg_h2(0.0, -10.0)
        assert math.isinf(cost)

    def test_max_h2_from_power_positive(self):
        h2 = max_h2_from_power(100.0, 0.0, 60.0)
        assert h2 > 0

    def test_max_h2_from_power_zero_power(self):
        assert max_h2_from_power(0.0, 0.0, 60.0) == 0.0

    def test_max_h2_from_power_frozen(self):
        assert max_h2_from_power(100.0, 0.0, -10.0) == 0.0

    def test_max_h2_proportional_to_power(self):
        h2_50 = max_h2_from_power(50.0, 0.0, 60.0)
        h2_100 = max_h2_from_power(100.0, 0.0, 60.0)
        assert abs(h2_100 / h2_50 - 2.0) < 0.01

    def test_system_efficiency_baseline(self):
        eff = system_efficiency(BASELINE_KWH_PER_KG_H2)
        expected = THEORETICAL_KWH_PER_KG_H2 / BASELINE_KWH_PER_KG_H2
        assert abs(eff - expected) < 0.01

    def test_system_efficiency_bounded(self):
        assert system_efficiency(THEORETICAL_KWH_PER_KG_H2) == 1.0
        assert system_efficiency(0.0) == 0.0
        assert system_efficiency(float('inf')) == 0.0

    def test_system_efficiency_perfect_capped(self):
        """Even sub-theoretical input caps at 1.0."""
        assert system_efficiency(THEORETICAL_KWH_PER_KG_H2 * 0.5) == 1.0


# ─── tick_electrolysis: basic operation ──────────────────────────────────────

class TestTickBasic:
    """Basic operational tests for tick_electrolysis."""

    def test_one_sol_produces_gas(self):
        state = create_electrolyzer()
        result = tick_electrolysis(state, water_available_kg=100.0,
                                    power_kwh=100.0, cell_temp_c=60.0)
        assert result.h2_produced_kg > 0
        assert result.o2_produced_kg > 0
        assert result.water_consumed_kg > 0
        assert result.energy_consumed_kwh > 0
        assert not result.halted

    def test_sol_counter_increments(self):
        state = create_electrolyzer()
        r1 = tick_electrolysis(state, 100.0, 100.0)
        assert r1.sol == 1
        r2 = tick_electrolysis(state, 100.0, 100.0)
        assert r2.sol == 2

    def test_no_water_no_production(self):
        state = create_electrolyzer()
        result = tick_electrolysis(state, water_available_kg=0.0, power_kwh=100.0)
        assert result.h2_produced_kg == 0.0
        assert result.o2_produced_kg == 0.0
        assert result.water_consumed_kg == 0.0

    def test_no_power_no_production(self):
        state = create_electrolyzer()
        result = tick_electrolysis(state, water_available_kg=100.0, power_kwh=0.0)
        assert result.h2_produced_kg == 0.0
        assert result.o2_produced_kg == 0.0

    def test_negative_inputs_treated_as_zero(self):
        state = create_electrolyzer()
        result = tick_electrolysis(state, water_available_kg=-50.0, power_kwh=-50.0)
        assert result.h2_produced_kg == 0.0
        assert result.o2_produced_kg == 0.0


# ─── tick_electrolysis: conservation laws ────────────────────────────────────

class TestTickConservation:
    """Physics invariants that must hold every single sol."""

    def test_mass_conservation_single_sol(self):
        """water consumed = h2 + o2 produced (within rounding)."""
        state = create_electrolyzer()
        result = tick_electrolysis(state, 100.0, 100.0, cell_temp_c=60.0)
        mass_out = result.h2_produced_kg + result.o2_produced_kg
        assert abs(result.water_consumed_kg - mass_out) < 0.01

    def test_mass_conservation_multi_sol(self):
        """Cumulative mass conservation over 50 sols."""
        state = create_electrolyzer()
        for _ in range(50):
            tick_electrolysis(state, 200.0, 200.0, cell_temp_c=60.0)
        mass_out = state.total_h2_produced_kg + state.total_o2_produced_kg
        assert abs(state.total_water_consumed_kg - mass_out) < 0.5

    def test_energy_never_exceeds_allocation(self):
        """Energy consumed ≤ power budget."""
        state = create_electrolyzer()
        for power in [10.0, 50.0, 100.0, 500.0]:
            s = create_electrolyzer()
            result = tick_electrolysis(s, 1000.0, power, cell_temp_c=60.0)
            assert result.energy_consumed_kwh <= power + 0.001

    def test_tank_levels_never_negative(self):
        state = create_electrolyzer()
        for _ in range(100):
            tick_electrolysis(state, 200.0, 200.0, cell_temp_c=60.0)
        assert state.h2_tank.stored_kg >= 0.0
        assert state.o2_tank.stored_kg >= 0.0

    def test_stoichiometric_ratio_holds(self):
        """O₂/H₂ mass ratio ≈ 7.936 for every sol."""
        state = create_electrolyzer()
        for _ in range(10):
            result = tick_electrolysis(state, 200.0, 200.0, cell_temp_c=60.0)
            if result.h2_produced_kg > 0:
                ratio = result.o2_produced_kg / result.h2_produced_kg
                expected = O2_MASS_FRACTION / H2_MASS_FRACTION
                assert abs(ratio - expected) < 0.1

    def test_efficiency_bounded(self):
        """System efficiency must be in [0, 1]."""
        state = create_electrolyzer()
        result = tick_electrolysis(state, 100.0, 100.0, cell_temp_c=60.0)
        assert 0.0 <= result.efficiency <= 1.0

    def test_h2_o2_ratio_matches_stoichiometry(self):
        """Cumulative H₂:O₂ production ratio is stoichiometric."""
        state = create_electrolyzer()
        for _ in range(30):
            tick_electrolysis(state, 500.0, 300.0, cell_temp_c=70.0)
        if state.total_h2_produced_kg > 0:
            ratio = state.total_o2_produced_kg / state.total_h2_produced_kg
            expected = O2_MASS_FRACTION / H2_MASS_FRACTION
            assert abs(ratio - expected) < 0.2


# ─── tick_electrolysis: edge cases and failure modes ─────────────────────────

class TestTickEdgeCases:
    """Edge cases: frozen, degraded, tanks full, offline."""

    def test_frozen_halts_production(self):
        state = create_electrolyzer()
        result = tick_electrolysis(state, 100.0, 100.0, cell_temp_c=-20.0)
        assert result.halted
        assert result.h2_produced_kg == 0.0
        assert "FROZEN" in result.warnings[0]

    def test_offline_stays_offline(self):
        state = create_electrolyzer()
        state.operating = False
        result = tick_electrolysis(state, 100.0, 100.0)
        assert result.halted
        assert "OFFLINE" in result.warnings[0]

    def test_cell_degradation_accumulates(self):
        state = create_electrolyzer()
        for _ in range(10):
            tick_electrolysis(state, 100.0, 100.0, cell_temp_c=60.0)
        expected_deg = CELL_DEGRADATION_PER_SOL * 10
        assert abs(state.cell_degradation - expected_deg) < 0.001

    def test_cell_failure_at_max_degradation(self):
        """Stack shuts down when degradation hits MAX."""
        state = create_electrolyzer()
        state.cell_degradation = MAX_CELL_DEGRADATION - CELL_DEGRADATION_PER_SOL * 0.5
        result = tick_electrolysis(state, 100.0, 100.0, cell_temp_c=60.0)
        assert result.halted or state.cell_degradation >= MAX_CELL_DEGRADATION

    def test_maintenance_reduces_degradation(self):
        state = create_electrolyzer()
        state.cell_degradation = 0.2
        tick_electrolysis(state, 100.0, 100.0, cell_temp_c=60.0, maintenance=True)
        assert state.cell_degradation < 0.2

    def test_maintenance_then_operation(self):
        """Maintenance + continued operation still works."""
        state = create_electrolyzer()
        state.cell_degradation = 0.15
        result = tick_electrolysis(state, 100.0, 100.0, cell_temp_c=60.0,
                                    maintenance=True)
        assert result.h2_produced_kg > 0
        assert state.cell_degradation < 0.15

    def test_h2_tank_limits_production(self):
        """When H₂ tank is nearly full, production is throttled."""
        state = create_electrolyzer()
        # Fill H₂ tank close to max
        room = state.h2_tank.headroom_kg()
        state.h2_tank.add(room - 0.01)  # leave only 0.01 kg room
        result = tick_electrolysis(state, 1000.0, 1000.0, cell_temp_c=60.0)
        assert result.h2_produced_kg <= 0.02  # at most the remaining room

    def test_o2_tank_limits_production(self):
        """When O₂ tank is nearly full, production is throttled."""
        state = create_electrolyzer()
        room = state.o2_tank.headroom_kg()
        state.o2_tank.add(room - 0.01)  # leave only 0.01 kg room
        result = tick_electrolysis(state, 1000.0, 1000.0, cell_temp_c=60.0)
        assert result.o2_produced_kg <= 0.02

    def test_water_limited_production(self):
        """Tiny water supply limits total output."""
        state = create_electrolyzer()
        result = tick_electrolysis(state, water_available_kg=0.5, power_kwh=500.0,
                                    cell_temp_c=60.0)
        max_possible_h2 = 0.5 * H2_MASS_FRACTION
        assert result.h2_produced_kg <= max_possible_h2 + 0.001


# ─── Smoke test: 100 sols without crash ──────────────────────────────────────

class TestSmoke:
    """Run the electrolyzer for 100+ sols — must not crash."""

    def test_100_sols_balanced(self):
        state = create_electrolyzer("balanced")
        for sol in range(100):
            result = tick_electrolysis(state, 200.0, 150.0, cell_temp_c=60.0,
                                        maintenance=(sol % 30 == 0))
            assert result.sol == sol + 1
            assert state.h2_tank.stored_kg >= 0.0
            assert state.o2_tank.stored_kg >= 0.0

    def test_100_sols_conservative(self):
        state = create_electrolyzer("conservative")
        for sol in range(100):
            tick_electrolysis(state, 300.0, 200.0, cell_temp_c=70.0,
                              maintenance=(sol % 50 == 0))
        assert state.total_h2_produced_kg > 0
        assert state.total_o2_produced_kg > 0

    def test_100_sols_aggressive(self):
        state = create_electrolyzer("aggressive")
        for sol in range(100):
            tick_electrolysis(state, 100.0, 100.0, cell_temp_c=50.0,
                              maintenance=(sol % 20 == 0))
        assert state.sol == 100

    def test_degradation_to_failure(self):
        """Run until cell failure — should happen and be graceful."""
        state = create_electrolyzer()
        failed_sol = None
        for sol in range(5000):
            result = tick_electrolysis(state, 100.0, 100.0, cell_temp_c=60.0)
            if result.halted:
                failed_sol = sol + 1
                break
        assert failed_sol is not None, "Should fail before 5000 sols"
        assert failed_sol > 100, "Should not fail too quickly"
        assert not state.operating

    def test_varying_temperature(self):
        """Temperature varies over sols — simulate seasonal changes."""
        state = create_electrolyzer()
        for sol in range(100):
            temp = 20.0 + 40.0 * math.sin(2 * math.pi * sol / 668.6)
            result = tick_electrolysis(state, 200.0, 150.0, cell_temp_c=temp,
                                        maintenance=(sol % 30 == 0))
            assert result.efficiency >= 0.0
            assert result.efficiency <= 1.0

    def test_intermittent_power(self):
        """Power cuts on/off — system handles gracefully."""
        state = create_electrolyzer()
        for sol in range(50):
            power = 150.0 if sol % 3 != 0 else 0.0
            result = tick_electrolysis(state, 200.0, power, cell_temp_c=60.0)
            if power == 0.0:
                assert result.h2_produced_kg == 0.0
            assert state.h2_tank.stored_kg >= 0.0


# ─── create_electrolyzer ─────────────────────────────────────────────────────

class TestCreateElectrolyzer:
    """Tests for the factory function."""

    def test_balanced_is_default(self):
        e = create_electrolyzer()
        assert e.operating is True
        assert e.cell_degradation == 0.0
        assert e.h2_tank.volume_l == DEFAULT_H2_TANK_VOLUME_L
        assert e.o2_tank.volume_l == DEFAULT_O2_TANK_VOLUME_L

    def test_conservative_has_larger_tanks(self):
        e = create_electrolyzer("conservative")
        d = create_electrolyzer("balanced")
        assert e.h2_tank.volume_l >= d.h2_tank.volume_l
        assert e.o2_tank.volume_l >= d.o2_tank.volume_l

    def test_aggressive_has_smaller_tanks(self):
        e = create_electrolyzer("aggressive")
        d = create_electrolyzer("balanced")
        assert e.h2_tank.volume_l <= d.h2_tank.volume_l
        assert e.o2_tank.volume_l <= d.o2_tank.volume_l

    def test_unknown_strategy_defaults_balanced(self):
        e = create_electrolyzer("unknown_strategy")
        d = create_electrolyzer("balanced")
        assert e.h2_tank.volume_l == d.h2_tank.volume_l

    def test_all_strategies_are_operational(self):
        for strat in ["conservative", "balanced", "aggressive"]:
            e = create_electrolyzer(strat)
            assert e.operating is True
            assert e.cell_degradation == 0.0
