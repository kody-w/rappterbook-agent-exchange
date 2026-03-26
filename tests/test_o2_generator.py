"""
test_o2_generator.py — Tests for Mars ISRU oxygen production model.

Covers: SOEC physics, mass conservation, energy bounds, filter
degradation, crew demand, storage dynamics, and multi-sol integration.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.o2_generator import (
    # Constants
    O2_PER_CO2_MASS,
    CO_PER_CO2_MASS,
    MM_CO2,
    MM_O2,
    MM_CO,
    MARS_SURFACE_PRESSURE_KPA,
    ENERGY_KWH_PER_KG_O2,
    SOEC_OPERATING_TEMP_C,
    SOEC_MIN_TEMP_C,
    SOEC_EFFICIENCY_AT_NOMINAL,
    O2_KG_PER_PERSON_SOL,
    SOL_HOURS,
    DUST_CLOG_RATE_PER_SOL,
    DUST_STORM_CLOG_RATE,
    FILTER_CLEANING_RESTORES,
    STORAGE_LEAK_RATE_PER_SOL,
    MOXIE_O2_RATE_G_HR,
    MOXIE_POWER_W,
    # Functions
    co2_intake_kg_hr,
    soec_efficiency,
    produce_o2_sol,
    clean_filter,
    crew_o2_demand,
    o2_sufficiency,
    tick_o2_system,
    create_colony_o2_system,
    # Data structures
    SOECStack,
    O2Storage,
)


# ===================================================================
# Constants & stoichiometry
# ===================================================================

class TestStoichiometry:
    """Mass conservation of the CO2 → CO + O2 reaction."""

    def test_mass_ratios_sum_to_one(self):
        """O2 fraction + CO fraction of input CO2 mass must equal 1."""
        assert abs(O2_PER_CO2_MASS + CO_PER_CO2_MASS - 1.0) < 1e-6

    def test_o2_per_co2_value(self):
        """2 CO2 → 2 CO + O2: O2/CO2 mass ratio = 32/88."""
        expected = MM_O2 / (2 * MM_CO2)
        assert abs(O2_PER_CO2_MASS - expected) < 1e-6

    def test_co_per_co2_value(self):
        """2 CO2 → 2 CO + O2: CO/CO2 mass ratio = 56/88."""
        expected = (2 * MM_CO) / (2 * MM_CO2)
        assert abs(CO_PER_CO2_MASS - expected) < 1e-6

    def test_molar_balance(self):
        """2 mol CO2 = 2 mol CO + 1 mol O2 by mass."""
        lhs = 2 * MM_CO2
        rhs = 2 * MM_CO + MM_O2
        assert abs(lhs - rhs) < 0.1  # within rounding of atomic masses


class TestMOXIEDerivedConstants:
    """Verify MOXIE-derived energy constants are self-consistent."""

    def test_energy_per_kg_o2(self):
        """300W / 10 g/hr = 30 kWh/kg."""
        expected = MOXIE_POWER_W / MOXIE_O2_RATE_G_HR
        assert abs(ENERGY_KWH_PER_KG_O2 - expected) < 1e-6

    def test_human_o2_rate(self):
        """NASA HRP: 0.84 kg O2/person/sol."""
        assert O2_KG_PER_PERSON_SOL == 0.84


# ===================================================================
# SOECStack dataclass
# ===================================================================

class TestSOECStack:
    """Test SOEC stack construction and basic properties."""

    def test_valid_construction(self):
        s = SOECStack(n_cells=10)
        assert s.n_cells == 10
        assert s.filter_health == 1.0
        assert s.cumulative_runtime_hrs == 0.0

    def test_negative_cells_raises(self):
        with pytest.raises(ValueError):
            SOECStack(n_cells=-1)

    def test_filter_health_clamped(self):
        s = SOECStack(n_cells=5, filter_health=1.5)
        assert s.filter_health == 1.0
        s2 = SOECStack(n_cells=5, filter_health=-0.3)
        assert s2.filter_health == 0.0

    def test_max_o2_rate(self):
        s = SOECStack(n_cells=10, filter_health=1.0)
        # 10 cells × 10 g/hr × 1.0 filter / 1000 = 0.1 kg/hr
        assert abs(s.max_o2_rate_kg_hr() - 0.1) < 1e-6

    def test_max_o2_rate_degraded_filter(self):
        s = SOECStack(n_cells=10, filter_health=0.5)
        assert abs(s.max_o2_rate_kg_hr() - 0.05) < 1e-6

    def test_power_demand(self):
        s = SOECStack(n_cells=10)
        # 10 × 300W = 3000W = 3.0 kW
        assert abs(s.power_demand_kw() - 3.0) < 1e-6

    def test_zero_cells(self):
        s = SOECStack(n_cells=0)
        assert s.max_o2_rate_kg_hr() == 0.0
        assert s.power_demand_kw() == 0.0


# ===================================================================
# O2Storage dataclass
# ===================================================================

class TestO2Storage:
    """Test O2 storage tank operations."""

    def test_valid_construction(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=50.0)
        assert t.capacity_kg == 100.0
        assert t.stored_kg == 50.0

    def test_stored_clamped_to_capacity(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=150.0)
        assert t.stored_kg == 100.0

    def test_negative_stored_clamped(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=-10.0)
        assert t.stored_kg == 0.0

    def test_headroom(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=30.0)
        assert abs(t.headroom_kg() - 70.0) < 1e-6

    def test_store_within_capacity(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=80.0)
        actual = t.store(10.0)
        assert abs(actual - 10.0) < 1e-6
        assert abs(t.stored_kg - 90.0) < 1e-6

    def test_store_overflow(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=95.0)
        actual = t.store(20.0)
        assert abs(actual - 5.0) < 1e-6
        assert abs(t.stored_kg - 100.0) < 1e-6

    def test_store_negative_ignored(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=50.0)
        actual = t.store(-5.0)
        assert actual == 0.0
        assert t.stored_kg == 50.0

    def test_draw_within_stored(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=50.0)
        actual = t.draw(20.0)
        assert abs(actual - 20.0) < 1e-6
        assert abs(t.stored_kg - 30.0) < 1e-6

    def test_draw_exceeds_stored(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=10.0)
        actual = t.draw(50.0)
        assert abs(actual - 10.0) < 1e-6
        assert abs(t.stored_kg) < 1e-6

    def test_draw_negative_ignored(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=50.0)
        actual = t.draw(-5.0)
        assert actual == 0.0

    def test_apply_leak(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=100.0)
        lost = t.apply_leak()
        expected_loss = 100.0 * STORAGE_LEAK_RATE_PER_SOL
        assert abs(lost - expected_loss) < 1e-6
        assert abs(t.stored_kg - (100.0 - expected_loss)) < 1e-6

    def test_leak_from_empty(self):
        t = O2Storage(capacity_kg=100.0, stored_kg=0.0)
        lost = t.apply_leak()
        assert lost == 0.0


# ===================================================================
# CO2 intake function
# ===================================================================

class TestCO2Intake:
    """Test atmospheric CO2 intake rate."""

    def test_nominal_pressure(self):
        rate = co2_intake_kg_hr(MARS_SURFACE_PRESSURE_KPA)
        # At nominal, one cell needs enough CO2 for 10 g O2/hr
        assert rate > 0

    def test_zero_pressure(self):
        assert co2_intake_kg_hr(0.0) == 0.0

    def test_negative_pressure(self):
        assert co2_intake_kg_hr(-1.0) == 0.0

    def test_double_pressure_doubles_intake(self):
        rate1 = co2_intake_kg_hr(MARS_SURFACE_PRESSURE_KPA)
        rate2 = co2_intake_kg_hr(MARS_SURFACE_PRESSURE_KPA * 2)
        assert abs(rate2 / rate1 - 2.0) < 1e-6

    def test_terraformed_pressure_increases_intake(self):
        """Higher pressure (terraforming) means more CO2 available."""
        rate_base = co2_intake_kg_hr(0.636)
        rate_terra = co2_intake_kg_hr(5.0)
        assert rate_terra > rate_base


# ===================================================================
# SOEC efficiency curve
# ===================================================================

class TestSOECEfficiency:
    """Test SOEC temperature-dependent efficiency."""

    def test_below_minimum(self):
        assert soec_efficiency(500.0) == 0.0

    def test_at_minimum(self):
        """At 600°C, efficiency should be ~0.20."""
        eff = soec_efficiency(SOEC_MIN_TEMP_C)
        assert abs(eff - 0.20) < 1e-6

    def test_at_nominal(self):
        """At 800°C, efficiency should be SOEC_EFFICIENCY_AT_NOMINAL."""
        eff = soec_efficiency(SOEC_OPERATING_TEMP_C)
        assert abs(eff - SOEC_EFFICIENCY_AT_NOMINAL) < 1e-6

    def test_above_nominal_better(self):
        """Between 800-1000°C, efficiency is slightly higher."""
        eff_nom = soec_efficiency(800.0)
        eff_900 = soec_efficiency(900.0)
        assert eff_900 > eff_nom

    def test_above_1000_degrades(self):
        """Above 1000°C, thermal damage reduces efficiency."""
        eff_1000 = soec_efficiency(1000.0)
        eff_1200 = soec_efficiency(1200.0)
        assert eff_1200 < eff_1000

    def test_extreme_heat_clamps_to_zero(self):
        """At extreme temperature, efficiency hits floor of 0."""
        eff = soec_efficiency(2500.0)
        assert eff >= 0.0

    def test_monotone_600_to_800(self):
        """Efficiency monotonically increases from 600 to 800°C."""
        temps = [600, 650, 700, 750, 800]
        effs = [soec_efficiency(t) for t in temps]
        for i in range(len(effs) - 1):
            assert effs[i + 1] >= effs[i]

    def test_efficiency_bounded(self):
        """Efficiency is always in [0, 1]."""
        for t in range(0, 2000, 50):
            eff = soec_efficiency(float(t))
            assert 0.0 <= eff <= 1.0


# ===================================================================
# produce_o2_sol — core production function
# ===================================================================

class TestProduceO2Sol:
    """Test single-sol O2 production."""

    def _make_stack(self, n_cells: int = 10) -> SOECStack:
        return SOECStack(n_cells=n_cells)

    def test_basic_production(self):
        stack = self._make_stack()
        result = produce_o2_sol(stack, power_kwh=100.0, pressure_kpa=0.636)
        assert result["o2_produced_kg"] > 0
        assert result["co2_consumed_kg"] > 0
        assert result["co_produced_kg"] > 0

    def test_mass_conservation(self):
        """CO2 consumed = O2 produced + CO produced (mass balance)."""
        stack = self._make_stack()
        r = produce_o2_sol(stack, power_kwh=100.0, pressure_kpa=0.636)
        total_out = r["o2_produced_kg"] + r["co_produced_kg"]
        assert abs(r["co2_consumed_kg"] - total_out) < 1e-4

    def test_stoichiometric_ratio(self):
        """O2/CO2 mass ratio matches stoichiometry."""
        stack = self._make_stack()
        r = produce_o2_sol(stack, power_kwh=100.0, pressure_kpa=0.636)
        if r["co2_consumed_kg"] > 0:
            ratio = r["o2_produced_kg"] / r["co2_consumed_kg"]
            assert abs(ratio - O2_PER_CO2_MASS) < 1e-4

    def test_power_conservation(self):
        """Power consumed never exceeds power allocated."""
        stack = self._make_stack()
        r = produce_o2_sol(stack, power_kwh=50.0, pressure_kpa=0.636)
        assert r["power_consumed_kwh"] <= 50.0 + 1e-6

    def test_zero_power(self):
        stack = self._make_stack()
        r = produce_o2_sol(stack, power_kwh=0.0, pressure_kpa=0.636)
        assert r["o2_produced_kg"] == 0.0

    def test_zero_pressure(self):
        stack = self._make_stack()
        r = produce_o2_sol(stack, power_kwh=100.0, pressure_kpa=0.0)
        assert r["o2_produced_kg"] == 0.0

    def test_zero_cells(self):
        stack = SOECStack(n_cells=0)
        r = produce_o2_sol(stack, power_kwh=100.0, pressure_kpa=0.636)
        assert r["o2_produced_kg"] == 0.0

    def test_filter_degrades(self):
        """Filter health decreases after a production sol."""
        stack = self._make_stack()
        before = stack.filter_health
        produce_o2_sol(stack, power_kwh=100.0, pressure_kpa=0.636)
        assert stack.filter_health < before

    def test_dust_storm_degrades_filter_faster(self):
        """High dust opacity degrades filter faster."""
        s1 = self._make_stack()
        s2 = self._make_stack()
        produce_o2_sol(s1, power_kwh=100.0, pressure_kpa=0.636, dust_opacity=0.0)
        produce_o2_sol(s2, power_kwh=100.0, pressure_kpa=0.636, dust_opacity=0.8)
        assert s2.filter_health < s1.filter_health

    def test_runtime_accumulates(self):
        stack = self._make_stack()
        produce_o2_sol(stack, power_kwh=100.0, pressure_kpa=0.636)
        assert abs(stack.cumulative_runtime_hrs - SOL_HOURS) < 1e-6

    def test_dead_filter_produces_nothing(self):
        stack = SOECStack(n_cells=10, filter_health=0.0)
        r = produce_o2_sol(stack, power_kwh=100.0, pressure_kpa=0.636)
        assert r["o2_produced_kg"] == 0.0

    def test_more_cells_more_production(self):
        """More cells → more O2 (up to power/pressure limits)."""
        s5 = SOECStack(n_cells=5)
        s20 = SOECStack(n_cells=20)
        r5 = produce_o2_sol(s5, power_kwh=500.0, pressure_kpa=0.636)
        r20 = produce_o2_sol(s20, power_kwh=500.0, pressure_kpa=0.636)
        assert r20["o2_produced_kg"] >= r5["o2_produced_kg"]

    def test_more_power_more_production(self):
        """More power → more O2 (up to intake limit)."""
        s1 = self._make_stack()
        s2 = self._make_stack()
        r1 = produce_o2_sol(s1, power_kwh=10.0, pressure_kpa=0.636)
        r2 = produce_o2_sol(s2, power_kwh=200.0, pressure_kpa=0.636)
        assert r2["o2_produced_kg"] >= r1["o2_produced_kg"]

    def test_higher_pressure_more_production(self):
        """Terraformed atmosphere yields more O2."""
        s1 = self._make_stack()
        s2 = self._make_stack()
        r1 = produce_o2_sol(s1, power_kwh=100.0, pressure_kpa=0.636)
        r2 = produce_o2_sol(s2, power_kwh=100.0, pressure_kpa=5.0)
        assert r2["o2_produced_kg"] >= r1["o2_produced_kg"]


# ===================================================================
# Filter cleaning
# ===================================================================

class TestCleanFilter:
    """Test manual filter cleaning."""

    def test_clean_restores_health(self):
        stack = SOECStack(n_cells=10, filter_health=0.5)
        restored = clean_filter(stack)
        assert restored > 0
        assert stack.filter_health > 0.5

    def test_clean_amount_correct(self):
        stack = SOECStack(n_cells=10, filter_health=0.5)
        restored = clean_filter(stack)
        expected = (1.0 - 0.5) * FILTER_CLEANING_RESTORES
        assert abs(restored - expected) < 1e-4

    def test_clean_perfect_filter_no_change(self):
        stack = SOECStack(n_cells=10, filter_health=1.0)
        restored = clean_filter(stack)
        assert abs(restored) < 1e-6

    def test_clean_never_exceeds_one(self):
        stack = SOECStack(n_cells=10, filter_health=0.95)
        clean_filter(stack)
        assert stack.filter_health <= 1.0


# ===================================================================
# Crew O2 demand
# ===================================================================

class TestCrewO2Demand:
    """Test crew oxygen consumption."""

    def test_single_person(self):
        assert abs(crew_o2_demand(1) - O2_KG_PER_PERSON_SOL) < 1e-6

    def test_hundred_people(self):
        assert abs(crew_o2_demand(100) - 84.0) < 1e-6

    def test_zero_population(self):
        assert crew_o2_demand(0) == 0.0


# ===================================================================
# O2 sufficiency
# ===================================================================

class TestO2Sufficiency:
    """Test stored O2 vs demand ratio."""

    def test_sufficient(self):
        ratio = o2_sufficiency(stored_kg=10.0, population=1, days=1)
        expected = 10.0 / O2_KG_PER_PERSON_SOL
        assert abs(ratio - expected) < 1e-4

    def test_zero_population_infinite(self):
        ratio = o2_sufficiency(stored_kg=10.0, population=0)
        assert ratio == float("inf")

    def test_zero_stored_zero_demand(self):
        ratio = o2_sufficiency(stored_kg=0.0, population=0)
        assert ratio == 1.0

    def test_multiday(self):
        ratio = o2_sufficiency(stored_kg=10.0, population=1, days=10)
        expected = 10.0 / (O2_KG_PER_PERSON_SOL * 10)
        assert abs(ratio - expected) < 1e-4


# ===================================================================
# tick_o2_system — full system integration
# ===================================================================

class TestTickO2System:
    """Test the complete O2 life support tick."""

    def _make_system(self, n_cells: int = 20, population: int = 10):
        stack = SOECStack(n_cells=n_cells)
        storage = O2Storage(capacity_kg=100.0, stored_kg=50.0)
        return stack, storage

    def test_basic_tick(self):
        stack, storage = self._make_system()
        r = tick_o2_system(
            stack, storage,
            power_kwh=200.0,
            pressure_kpa=0.636,
            dust_opacity=0.0,
            population=10,
        )
        assert r["o2_produced_kg"] > 0
        assert r["o2_demand_kg"] > 0
        assert r["o2_delivered_kg"] > 0

    def test_demand_equals_population_times_rate(self):
        stack, storage = self._make_system()
        r = tick_o2_system(
            stack, storage,
            power_kwh=200.0,
            pressure_kpa=0.636,
            dust_opacity=0.0,
            population=10,
        )
        expected = 10 * O2_KG_PER_PERSON_SOL
        assert abs(r["o2_demand_kg"] - expected) < 1e-4

    def test_surplus_goes_to_storage(self):
        """When production > demand, surplus stored."""
        stack, storage = self._make_system(n_cells=100, population=1)
        storage_before = storage.stored_kg
        r = tick_o2_system(
            stack, storage,
            power_kwh=500.0,
            pressure_kpa=0.636,
            dust_opacity=0.0,
            population=1,
        )
        # Storage should increase (minus leak)
        assert r["o2_surplus_stored_kg"] > 0

    def test_deficit_draws_from_storage(self):
        """When production < demand, storage is tapped."""
        stack = SOECStack(n_cells=1)  # tiny stack
        storage = O2Storage(capacity_kg=100.0, stored_kg=50.0)
        r = tick_o2_system(
            stack, storage,
            power_kwh=1.0,  # very little power
            pressure_kpa=0.636,
            dust_opacity=0.0,
            population=50,  # huge demand
        )
        assert storage.stored_kg < 50.0

    def test_shortfall_when_both_exhausted(self):
        """When production + storage < demand, shortfall occurs."""
        stack = SOECStack(n_cells=1)
        storage = O2Storage(capacity_kg=10.0, stored_kg=0.5)
        r = tick_o2_system(
            stack, storage,
            power_kwh=0.5,
            pressure_kpa=0.636,
            dust_opacity=0.0,
            population=100,
        )
        assert r["o2_shortfall_kg"] > 0

    def test_zero_population_no_draw(self):
        stack, storage = self._make_system()
        before = storage.stored_kg
        r = tick_o2_system(
            stack, storage,
            power_kwh=100.0,
            pressure_kpa=0.636,
            dust_opacity=0.0,
            population=0,
        )
        assert r["o2_demand_kg"] == 0.0
        assert r["o2_shortfall_kg"] == 0.0
        # Storage should increase from production (minus leak)
        assert storage.stored_kg >= before - 1.0  # allow for leak

    def test_leak_occurs(self):
        """Storage leak happens every sol."""
        stack, storage = self._make_system()
        r = tick_o2_system(
            stack, storage,
            power_kwh=200.0,
            pressure_kpa=0.636,
            dust_opacity=0.0,
            population=10,
        )
        assert r["o2_leaked_kg"] > 0

    def test_sufficiency_reported(self):
        stack, storage = self._make_system()
        r = tick_o2_system(
            stack, storage,
            power_kwh=200.0,
            pressure_kpa=0.636,
            dust_opacity=0.0,
            population=10,
        )
        assert r["sufficiency_ratio"] > 0

    def test_mass_conservation_in_tick(self):
        """CO2 in = CO out + O2 out (from production sub-result)."""
        stack, storage = self._make_system()
        r = tick_o2_system(
            stack, storage,
            power_kwh=200.0,
            pressure_kpa=0.636,
            dust_opacity=0.0,
            population=10,
        )
        total_out = r["o2_produced_kg"] + r["co_produced_kg"]
        assert abs(r["co2_consumed_kg"] - total_out) < 1e-4


# ===================================================================
# create_colony_o2_system — factory function
# ===================================================================

class TestCreateColonyO2System:
    """Test O2 system factory for colony strategies."""

    @pytest.mark.parametrize("strategy", ["conservative", "balanced", "aggressive"])
    def test_creates_valid_system(self, strategy):
        stack, storage = create_colony_o2_system(strategy, population=100)
        assert stack.n_cells > 0
        assert storage.capacity_kg > 0
        assert storage.stored_kg >= 0

    def test_conservative_has_more_cells(self):
        sc, _ = create_colony_o2_system("conservative", population=100)
        sa, _ = create_colony_o2_system("aggressive", population=100)
        assert sc.n_cells >= sa.n_cells

    def test_conservative_has_more_storage(self):
        _, tc = create_colony_o2_system("conservative", population=100)
        _, ta = create_colony_o2_system("aggressive", population=100)
        assert tc.capacity_kg >= ta.capacity_kg

    def test_scales_with_population(self):
        s50, t50 = create_colony_o2_system("balanced", population=50)
        s200, t200 = create_colony_o2_system("balanced", population=200)
        assert s200.n_cells >= s50.n_cells
        assert t200.capacity_kg >= t50.capacity_kg

    def test_zero_population_still_works(self):
        stack, storage = create_colony_o2_system("balanced", population=0)
        assert stack.n_cells >= 0
        assert storage.capacity_kg >= 0

    def test_unknown_strategy_defaults_to_balanced(self):
        stack, storage = create_colony_o2_system("unknown", population=50)
        sb, tb = create_colony_o2_system("balanced", population=50)
        assert stack.n_cells == sb.n_cells
        assert storage.capacity_kg == tb.capacity_kg


# ===================================================================
# Multi-sol integration: smoke test + invariants
# ===================================================================

class TestMultiSolIntegration:
    """Run the O2 system for multiple sols and verify invariants."""

    def test_10_sol_smoke(self):
        """System runs 10 sols without crash."""
        stack, storage = create_colony_o2_system("balanced", population=80)
        for sol in range(10):
            dust = 0.1 if sol < 5 else 0.5
            result = tick_o2_system(
                stack, storage,
                power_kwh=150.0,
                pressure_kpa=0.636,
                dust_opacity=dust,
                population=80,
            )
            assert result["o2_produced_kg"] >= 0
            assert result["o2_delivered_kg"] >= 0
            assert result["filter_health"] >= 0
            assert storage.stored_kg >= 0

    def test_filter_degrades_over_100_sols(self):
        """Filter health drops significantly over 100 sols."""
        stack, storage = create_colony_o2_system("balanced", population=80)
        initial_health = stack.filter_health
        for _ in range(100):
            tick_o2_system(
                stack, storage,
                power_kwh=150.0,
                pressure_kpa=0.636,
                dust_opacity=0.2,
                population=80,
            )
        assert stack.filter_health < initial_health * 0.8

    def test_storage_never_negative(self):
        """Storage never goes below zero, even under extreme demand."""
        stack = SOECStack(n_cells=2)  # undersized
        storage = O2Storage(capacity_kg=50.0, stored_kg=50.0)
        for _ in range(50):
            tick_o2_system(
                stack, storage,
                power_kwh=10.0,
                pressure_kpa=0.636,
                dust_opacity=0.0,
                population=200,  # huge demand
            )
            assert storage.stored_kg >= 0.0

    def test_storage_never_exceeds_capacity(self):
        """Storage never exceeds capacity, even with massive production."""
        stack = SOECStack(n_cells=500)  # oversized
        storage = O2Storage(capacity_kg=100.0, stored_kg=0.0)
        for _ in range(50):
            tick_o2_system(
                stack, storage,
                power_kwh=10000.0,
                pressure_kpa=0.636,
                dust_opacity=0.0,
                population=1,  # tiny demand
            )
            assert storage.stored_kg <= storage.capacity_kg + 1e-6

    def test_filter_cleaning_extends_production(self):
        """Periodic filter cleaning keeps production alive longer."""
        s1, t1 = create_colony_o2_system("balanced", population=80)
        s2, t2 = create_colony_o2_system("balanced", population=80)
        total_o2_no_clean = 0.0
        total_o2_with_clean = 0.0
        for sol in range(100):
            r1 = tick_o2_system(s1, t1, 150.0, 0.636, 0.3, 80)
            r2 = tick_o2_system(s2, t2, 150.0, 0.636, 0.3, 80)
            total_o2_no_clean += r1["o2_produced_kg"]
            total_o2_with_clean += r2["o2_produced_kg"]
            if sol % 20 == 0:
                clean_filter(s2)
        assert total_o2_with_clean > total_o2_no_clean

    def test_production_physically_bounded(self):
        """O2 output per sol is within physical limits at nominal pressure."""
        stack = SOECStack(n_cells=10)
        # At nominal pressure (0.636 kPa), intake-limited:
        # 10 cells × 10g/hr × 24.66hr / 1000 = 2.466 kg max raw intake O2-equiv
        # Efficiency and filter reduce this further
        max_theoretical = 10 * MOXIE_O2_RATE_G_HR * SOL_HOURS / 1000.0
        result = produce_o2_sol(stack, power_kwh=1000.0, pressure_kpa=0.636)
        assert result["o2_produced_kg"] <= max_theoretical + 0.01

    def test_co_byproduct_always_positive_with_production(self):
        """Whenever O2 is produced, CO is also produced."""
        stack = SOECStack(n_cells=10)
        r = produce_o2_sol(stack, power_kwh=100.0, pressure_kpa=0.636)
        if r["o2_produced_kg"] > 0:
            assert r["co_produced_kg"] > 0


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_negative_power(self):
        stack = SOECStack(n_cells=10)
        r = produce_o2_sol(stack, power_kwh=-10.0, pressure_kpa=0.636)
        assert r["o2_produced_kg"] == 0.0

    def test_tiny_power(self):
        stack = SOECStack(n_cells=10)
        r = produce_o2_sol(stack, power_kwh=0.001, pressure_kpa=0.636)
        assert r["o2_produced_kg"] >= 0.0

    def test_extreme_pressure(self):
        """Even at high terraformed pressure, no division by zero."""
        stack = SOECStack(n_cells=10)
        r = produce_o2_sol(stack, power_kwh=100.0, pressure_kpa=100.0)
        assert r["o2_produced_kg"] >= 0.0

    def test_single_cell_produces_o2(self):
        """Even one cell can produce some O2."""
        stack = SOECStack(n_cells=1)
        r = produce_o2_sol(stack, power_kwh=50.0, pressure_kpa=0.636)
        assert r["o2_produced_kg"] > 0
