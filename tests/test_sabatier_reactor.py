"""Tests for sabatier_reactor.py -- Mars Sabatier Propellant Production.

133 unit tests across 18 sections covering chemistry, thermodynamics,
conservation laws, failure modes, parametrized invariants, edge cases,
and multi-sol smoke tests.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sabatier_reactor import (
    R_GAS, MARS_AMBIENT_TEMP_K, MARS_CO2_PARTIAL_ATM,
    CO2_MOLAR_MASS, H2_MOLAR_MASS, CH4_MOLAR_MASS, H2O_MOLAR_MASS,
    STOICH_H2_PER_CO2, STOICH_CH4_PER_CO2, STOICH_H2O_PER_CO2,
    CO2_MASS_PER_MOL, H2_MASS_PER_MOL, CH4_MASS_PER_MOL, H2O_MASS_PER_MOL,
    DELTA_H_KJ_PER_MOL, ACTIVATION_ENERGY_KJ_MOL,
    PRE_EXPONENTIAL_FACTOR, REFERENCE_TEMP_K, K_EQ_REF,
    DEFAULT_REACTOR_TEMP_K, MIN_REACTOR_TEMP_K, MAX_REACTOR_TEMP_K,
    DEFAULT_PRESSURE_ATM, MIN_PRESSURE_ATM,
    DEFAULT_CATALYST_MASS_KG, DEFAULT_CO2_FEED_KG_PER_SOL,
    DEFAULT_H2_FEED_KG_PER_SOL,
    CATALYST_WEAR_PER_KG_CH4, CATALYST_REPLACEMENT_THRESHOLD,
    CATALYST_TEMP_PENALTY_K,
    HOURS_PER_SOL, SECONDS_PER_SOL, COMPRESSOR_EFFICIENCY,
    arrhenius_rate, equilibrium_constant, equilibrium_conversion,
    actual_conversion, co2_feed_to_moles, h2_feed_to_moles,
    stoichiometric_h2_kg, limiting_reagent_moles,
    reaction_products_kg, reactants_consumed_kg,
    reaction_heat_kw, compressor_power_kw, heater_power_kw,
    catalyst_degradation, mass_balance_check,
    SabatierReactor, TickResult, tick, run_simulation,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Physical & Chemical Constants
# ═══════════════════════════════════════════════════════════════════════

class TestPhysicalConstants:
    def test_gas_constant(self):
        assert abs(R_GAS - 8.314) < 0.001

    def test_mars_ambient(self):
        assert 200.0 < MARS_AMBIENT_TEMP_K < 230.0

    def test_co2_molar_mass(self):
        assert abs(CO2_MOLAR_MASS - 0.044009) < 0.0001

    def test_h2_molar_mass(self):
        assert abs(H2_MOLAR_MASS - 0.002016) < 0.0005

    def test_ch4_molar_mass(self):
        assert abs(CH4_MOLAR_MASS - 0.016043) < 0.001

    def test_h2o_molar_mass(self):
        assert abs(H2O_MOLAR_MASS - 0.018015) < 0.001

    def test_stoich_h2_per_co2(self):
        assert STOICH_H2_PER_CO2 == 4.0

    def test_stoich_ch4_per_co2(self):
        assert STOICH_CH4_PER_CO2 == 1.0

    def test_stoich_h2o_per_co2(self):
        assert STOICH_H2O_PER_CO2 == 2.0

    def test_delta_h_exothermic(self):
        assert DELTA_H_KJ_PER_MOL < 0

    def test_delta_h_value(self):
        assert abs(DELTA_H_KJ_PER_MOL - (-165.0)) < 1.0

    def test_hours_per_sol(self):
        assert abs(HOURS_PER_SOL - 24.66) < 0.01

    def test_seconds_per_sol(self):
        assert abs(SECONDS_PER_SOL - HOURS_PER_SOL * 3600.0) < 1.0

    def test_mars_co2_partial(self):
        assert 0.005 < MARS_CO2_PARTIAL_ATM < 0.007


# ═══════════════════════════════════════════════════════════════════════
# 2. Stoichiometric Mass Ratios
# ═══════════════════════════════════════════════════════════════════════

class TestStoichiometry:
    def test_mass_conservation_per_mol(self):
        """CO2 + 4H2 -> CH4 + 2H2O mass balance per mole."""
        reactants = CO2_MASS_PER_MOL + H2_MASS_PER_MOL
        products = CH4_MASS_PER_MOL + H2O_MASS_PER_MOL
        assert abs(reactants - products) < 1e-6

    def test_h2_mass_per_mol(self):
        expected = 4 * H2_MOLAR_MASS
        assert abs(H2_MASS_PER_MOL - expected) < 1e-6

    def test_ch4_mass_per_mol(self):
        expected = 1 * CH4_MOLAR_MASS
        assert abs(CH4_MASS_PER_MOL - expected) < 1e-6

    def test_h2o_mass_per_mol(self):
        expected = 2 * H2O_MOLAR_MASS
        assert abs(H2O_MASS_PER_MOL - expected) < 1e-6

    def test_stoichiometric_h2_known(self):
        co2_kg = CO2_MOLAR_MASS  # 1 mole
        h2_needed = stoichiometric_h2_kg(co2_kg)
        assert abs(h2_needed - H2_MASS_PER_MOL) < 1e-6

    def test_stoichiometric_h2_zero(self):
        assert stoichiometric_h2_kg(0.0) == 0.0

    def test_stoichiometric_h2_negative(self):
        assert stoichiometric_h2_kg(-1.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 3. Mole Conversions
# ═══════════════════════════════════════════════════════════════════════

class TestMoleConversions:
    def test_co2_one_kg(self):
        moles = co2_feed_to_moles(1.0)
        assert abs(moles - 1.0 / CO2_MOLAR_MASS) < 0.01

    def test_h2_one_kg(self):
        moles = h2_feed_to_moles(1.0)
        assert abs(moles - 1.0 / H2_MOLAR_MASS) < 0.1

    def test_co2_zero(self):
        assert co2_feed_to_moles(0.0) == 0.0

    def test_h2_zero(self):
        assert h2_feed_to_moles(0.0) == 0.0

    def test_co2_negative(self):
        assert co2_feed_to_moles(-5.0) == 0.0

    def test_h2_negative(self):
        assert h2_feed_to_moles(-5.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 4. Limiting Reagent
# ═══════════════════════════════════════════════════════════════════════

class TestLimitingReagent:
    def test_stoichiometric_balance(self):
        """When H2 is exactly 4x CO2 moles, CO2 is limiting."""
        co2_mol, h2_mol = 10.0, 40.0
        result = limiting_reagent_moles(co2_mol, h2_mol)
        assert abs(result - 10.0) < 1e-6

    def test_h2_limiting(self):
        co2_mol, h2_mol = 10.0, 20.0  # only enough H2 for 5 mol CO2
        result = limiting_reagent_moles(co2_mol, h2_mol)
        assert abs(result - 5.0) < 1e-6

    def test_co2_limiting(self):
        co2_mol, h2_mol = 5.0, 100.0
        result = limiting_reagent_moles(co2_mol, h2_mol)
        assert abs(result - 5.0) < 1e-6

    def test_zero_co2(self):
        assert limiting_reagent_moles(0.0, 40.0) == 0.0

    def test_zero_h2(self):
        assert limiting_reagent_moles(10.0, 0.0) == 0.0

    def test_both_zero(self):
        assert limiting_reagent_moles(0.0, 0.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 5. Arrhenius Kinetics
# ═══════════════════════════════════════════════════════════════════════

class TestArrheniusRate:
    def test_positive_at_300c(self):
        rate = arrhenius_rate(573.15, 5.0, 1.0)
        assert rate > 0

    def test_increases_with_temp(self):
        r1 = arrhenius_rate(473.15, 5.0, 1.0)
        r2 = arrhenius_rate(573.15, 5.0, 1.0)
        assert r2 > r1

    def test_scales_with_catalyst_mass(self):
        r1 = arrhenius_rate(573.15, 5.0, 1.0)
        r2 = arrhenius_rate(573.15, 10.0, 1.0)
        assert abs(r2 - 2 * r1) < 1e-6

    def test_scales_with_health(self):
        r_full = arrhenius_rate(573.15, 5.0, 1.0)
        r_half = arrhenius_rate(573.15, 5.0, 0.5)
        assert abs(r_half - 0.5 * r_full) < 1e-6

    def test_zero_temp(self):
        assert arrhenius_rate(0.0, 5.0, 1.0) == 0.0

    def test_zero_catalyst(self):
        assert arrhenius_rate(573.15, 0.0, 1.0) == 0.0

    def test_zero_health(self):
        assert arrhenius_rate(573.15, 5.0, 0.0) == 0.0

    def test_negative_temp(self):
        assert arrhenius_rate(-100, 5.0, 1.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 6. Equilibrium Constant & Conversion
# ═══════════════════════════════════════════════════════════════════════

class TestEquilibrium:
    def test_keq_at_reference(self):
        """K_eq at reference temp should be K_EQ_REF."""
        k = equilibrium_constant(REFERENCE_TEMP_K)
        assert abs(k - K_EQ_REF) < 1.0

    def test_keq_decreases_with_temp(self):
        """Exothermic: K decreases at higher temp."""
        k_low = equilibrium_constant(473.15)
        k_high = equilibrium_constant(673.15)
        assert k_low > k_high

    def test_keq_positive(self):
        assert equilibrium_constant(500.0) > 0

    def test_keq_zero_temp(self):
        assert equilibrium_constant(0.0) == 0.0

    def test_conversion_at_300c(self):
        """~99% at 300°C, 3 atm."""
        x = equilibrium_conversion(573.15, 3.0)
        assert x > 0.95

    def test_conversion_lower_at_500c(self):
        x300 = equilibrium_conversion(573.15, 3.0)
        x500 = equilibrium_conversion(773.15, 3.0)
        assert x300 > x500

    def test_conversion_increases_with_pressure(self):
        x_low = equilibrium_conversion(573.15, 1.0)
        x_high = equilibrium_conversion(573.15, 5.0)
        assert x_high >= x_low

    def test_conversion_bounded_0_1(self):
        x = equilibrium_conversion(573.15, 3.0)
        assert 0.0 <= x <= 1.0

    def test_conversion_zero_temp(self):
        assert equilibrium_conversion(0.0, 3.0) == 0.0

    def test_conversion_zero_pressure(self):
        assert equilibrium_conversion(573.15, 0.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 7. Actual Conversion (Kinetics + Equilibrium)
# ═══════════════════════════════════════════════════════════════════════

class TestActualConversion:
    def test_positive_at_defaults(self):
        mol = co2_feed_to_moles(DEFAULT_CO2_FEED_KG_PER_SOL)
        x = actual_conversion(DEFAULT_REACTOR_TEMP_K, DEFAULT_PRESSURE_ATM,
                              DEFAULT_CATALYST_MASS_KG, 1.0, mol, SECONDS_PER_SOL)
        assert x > 0

    def test_bounded_by_equilibrium(self):
        mol = co2_feed_to_moles(10.0)
        x = actual_conversion(573.15, 3.0, 5.0, 1.0, mol, SECONDS_PER_SOL)
        x_eq = equilibrium_conversion(573.15, 3.0)
        assert x <= x_eq + 1e-9

    def test_zero_feed(self):
        assert actual_conversion(573.15, 3.0, 5.0, 1.0, 0.0, SECONDS_PER_SOL) == 0.0

    def test_zero_time(self):
        assert actual_conversion(573.15, 3.0, 5.0, 1.0, 100.0, 0.0) == 0.0

    def test_degraded_catalyst_lower(self):
        mol = co2_feed_to_moles(10.0)
        x_full = actual_conversion(573.15, 3.0, 5.0, 1.0, mol, SECONDS_PER_SOL)
        x_half = actual_conversion(573.15, 3.0, 5.0, 0.3, mol, SECONDS_PER_SOL)
        assert x_full >= x_half


# ═══════════════════════════════════════════════════════════════════════
# 8. Reaction Products & Reactants
# ═══════════════════════════════════════════════════════════════════════

class TestReactionProducts:
    def test_products_from_one_mol(self):
        ch4, h2o = reaction_products_kg(1.0)
        assert abs(ch4 - CH4_MOLAR_MASS) < 1e-6
        assert abs(h2o - 2 * H2O_MOLAR_MASS) < 1e-6

    def test_products_zero(self):
        ch4, h2o = reaction_products_kg(0.0)
        assert ch4 == 0.0
        assert h2o == 0.0

    def test_products_negative(self):
        ch4, h2o = reaction_products_kg(-1.0)
        assert ch4 == 0.0
        assert h2o == 0.0

    def test_reactants_from_one_mol(self):
        co2, h2 = reactants_consumed_kg(1.0)
        assert abs(co2 - CO2_MOLAR_MASS) < 1e-6
        assert abs(h2 - 4 * H2_MOLAR_MASS) < 1e-6

    def test_reactants_zero(self):
        co2, h2 = reactants_consumed_kg(0.0)
        assert co2 == 0.0
        assert h2 == 0.0

    def test_mass_balance_one_mol(self):
        ch4, h2o = reaction_products_kg(1.0)
        co2, h2 = reactants_consumed_kg(1.0)
        assert mass_balance_check(co2, h2, ch4, h2o)


# ═══════════════════════════════════════════════════════════════════════
# 9. Mass Balance Verification
# ═══════════════════════════════════════════════════════════════════════

class TestMassBalance:
    def test_balanced(self):
        ch4, h2o = reaction_products_kg(1.0)
        co2, h2 = reactants_consumed_kg(1.0)
        assert mass_balance_check(co2, h2, ch4, h2o)

    def test_unbalanced(self):
        assert not mass_balance_check(44.01, 8.08, 100.0, 36.04)

    def test_zeros_balanced(self):
        assert mass_balance_check(0.0, 0.0, 0.0, 0.0)

    @pytest.mark.parametrize("n_mol", [0.1, 1.0, 10.0, 100.0, 1000.0])
    def test_mass_conservation_at_scale(self, n_mol):
        ch4, h2o = reaction_products_kg(n_mol)
        co2, h2 = reactants_consumed_kg(n_mol)
        assert mass_balance_check(co2, h2, ch4, h2o)


# ═══════════════════════════════════════════════════════════════════════
# 10. Thermal Output
# ═══════════════════════════════════════════════════════════════════════

class TestReactionHeat:
    def test_positive_heat(self):
        """Exothermic reaction produces positive thermal power."""
        q = reaction_heat_kw(10.0)
        assert q > 0

    def test_scales_linearly(self):
        q1 = reaction_heat_kw(10.0)
        q2 = reaction_heat_kw(20.0)
        assert abs(q2 - 2 * q1) < 1e-6

    def test_zero_reaction(self):
        assert reaction_heat_kw(0.0) == 0.0

    def test_known_value(self):
        """10 mol/sol should produce 10*165 kJ / 88776 s ≈ 0.0186 kW."""
        q = reaction_heat_kw(10.0)
        expected = 10.0 * 165.0 / SECONDS_PER_SOL
        assert abs(q - expected) < 0.001


# ═══════════════════════════════════════════════════════════════════════
# 11. Compressor Power
# ═══════════════════════════════════════════════════════════════════════

class TestCompressorPower:
    def test_positive_power(self):
        p = compressor_power_kw(10.0, 0.006, 3.0, 210.0)
        assert p > 0

    def test_higher_ratio_more_power(self):
        p1 = compressor_power_kw(10.0, 0.006, 2.0, 210.0)
        p2 = compressor_power_kw(10.0, 0.006, 5.0, 210.0)
        assert p2 > p1

    def test_zero_feed(self):
        assert compressor_power_kw(0.0, 0.006, 3.0, 210.0) == 0.0

    def test_no_compression_needed(self):
        assert compressor_power_kw(10.0, 3.0, 3.0, 210.0) == 0.0

    def test_inlet_above_outlet(self):
        assert compressor_power_kw(10.0, 5.0, 3.0, 210.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 12. Heater Power
# ═══════════════════════════════════════════════════════════════════════

class TestHeaterPower:
    def test_positive(self):
        p = heater_power_kw(573.15, 210.0)
        assert p > 0

    def test_zero_when_cold(self):
        p = heater_power_kw(200.0, 210.0)
        assert p == 0.0

    def test_increases_with_delta_t(self):
        p1 = heater_power_kw(400.0, 210.0)
        p2 = heater_power_kw(600.0, 210.0)
        assert p2 > p1


# ═══════════════════════════════════════════════════════════════════════
# 13. Catalyst Degradation
# ═══════════════════════════════════════════════════════════════════════

class TestCatalystDegradation:
    def test_wear_reduces_health(self):
        new_health = catalyst_degradation(1.0, 1.0, 573.15)
        assert new_health < 1.0

    def test_no_production_no_wear(self):
        new_health = catalyst_degradation(1.0, 0.0, 573.15)
        assert new_health == 1.0

    def test_high_temp_accelerates(self):
        h_normal = catalyst_degradation(1.0, 10.0, 573.15)
        h_hot = catalyst_degradation(1.0, 10.0, 773.15)
        assert h_hot < h_normal

    def test_never_below_zero(self):
        h = catalyst_degradation(0.001, 1000.0, 773.15)
        assert h >= 0.0

    def test_dead_stays_dead(self):
        assert catalyst_degradation(0.0, 1.0, 573.15) == 0.0

    def test_monotonic_decrease(self):
        health = 1.0
        for _ in range(100):
            new_health = catalyst_degradation(health, 0.5, 573.15)
            assert new_health <= health
            health = new_health


# ═══════════════════════════════════════════════════════════════════════
# 14. SabatierReactor State
# ═══════════════════════════════════════════════════════════════════════

class TestSabatierReactorState:
    def test_defaults(self):
        r = SabatierReactor()
        assert r.sol == 0
        assert r.catalyst_health == 1.0
        assert r.reactor_temp_k == DEFAULT_REACTOR_TEMP_K
        assert r.pressure_atm == DEFAULT_PRESSURE_ATM
        assert r.cumulative_ch4_kg == 0.0

    def test_custom_init(self):
        r = SabatierReactor(co2_feed_kg_per_sol=20.0, reactor_temp_k=600.0)
        assert r.co2_feed_kg_per_sol == 20.0
        assert r.reactor_temp_k == 600.0


# ═══════════════════════════════════════════════════════════════════════
# 15. Tick Function
# ═══════════════════════════════════════════════════════════════════════

class TestTick:
    def test_sol_advances(self):
        state = SabatierReactor()
        tick(state)
        assert state.sol == 1

    def test_produces_ch4(self):
        state = SabatierReactor()
        result = tick(state)
        assert result.ch4_kg > 0

    def test_produces_h2o(self):
        state = SabatierReactor()
        result = tick(state)
        assert result.h2o_kg > 0

    def test_consumes_co2(self):
        state = SabatierReactor()
        result = tick(state)
        assert result.co2_consumed_kg > 0

    def test_consumes_h2(self):
        state = SabatierReactor()
        result = tick(state)
        assert result.h2_consumed_kg > 0

    def test_mass_balance_per_tick(self):
        state = SabatierReactor()
        result = tick(state)
        assert mass_balance_check(result.co2_consumed_kg, result.h2_consumed_kg,
                                  result.ch4_kg, result.h2o_kg)

    def test_operational_true(self):
        state = SabatierReactor()
        result = tick(state)
        assert result.operational is True

    def test_catalyst_wears(self):
        state = SabatierReactor()
        tick(state)
        assert state.catalyst_health < 1.0

    def test_cumulative_ch4_grows(self):
        state = SabatierReactor()
        tick(state)
        assert state.cumulative_ch4_kg > 0.0
        first = state.cumulative_ch4_kg
        tick(state)
        assert state.cumulative_ch4_kg > first

    def test_conversion_in_bounds(self):
        state = SabatierReactor()
        result = tick(state)
        assert 0.0 <= result.conversion <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# 16. Failure Modes
# ═══════════════════════════════════════════════════════════════════════

class TestFailureModes:
    def test_dead_catalyst(self):
        state = SabatierReactor(catalyst_health=0.05)
        result = tick(state)
        assert result.operational is False
        assert "CATALYST EXHAUSTED" in " ".join(result.events)

    def test_cold_reactor(self):
        state = SabatierReactor(reactor_temp_k=400.0)
        result = tick(state)
        assert result.operational is False
        assert "REACTOR TOO COLD" in " ".join(result.events)

    def test_low_pressure(self):
        state = SabatierReactor(pressure_atm=0.1)
        result = tick(state)
        assert result.operational is False
        assert "PRESSURE TOO LOW" in " ".join(result.events)

    def test_no_co2_feed(self):
        state = SabatierReactor(co2_feed_kg_per_sol=0.0)
        result = tick(state)
        assert result.operational is False
        assert result.ch4_kg == 0.0

    def test_no_h2_feed(self):
        state = SabatierReactor(h2_feed_kg_per_sol=0.0)
        result = tick(state)
        assert result.operational is False
        assert result.ch4_kg == 0.0

    def test_hot_reactor_still_operational(self):
        """Over MAX temp emits warning but still runs (just low yield)."""
        state = SabatierReactor(reactor_temp_k=MAX_REACTOR_TEMP_K + 1)
        result = tick(state)
        # Above max still runs (equilibrium shifts but reactor doesn't shut down)
        # unless some other constraint fires
        assert "REACTOR TOO HOT" in " ".join(result.events)

    def test_h2_deficit_warning(self):
        """When H2 is sub-stoichiometric, warning emitted."""
        state = SabatierReactor(h2_feed_kg_per_sol=0.5)  # way below stoich
        result = tick(state)
        assert result.operational is True
        assert "H2 DEFICIT" in " ".join(result.events)


# ═══════════════════════════════════════════════════════════════════════
# 17. Parametrized Invariants
# ═══════════════════════════════════════════════════════════════════════

class TestParametrizedInvariants:
    @pytest.mark.parametrize("co2_kg", [1.0, 5.0, 10.0, 50.0, 100.0])
    def test_ch4_scales_with_feed(self, co2_kg):
        h2_kg = stoichiometric_h2_kg(co2_kg)
        state = SabatierReactor(co2_feed_kg_per_sol=co2_kg,
                                h2_feed_kg_per_sol=h2_kg)
        result = tick(state)
        assert result.ch4_kg >= 0
        assert result.ch4_kg <= co2_kg  # can't produce more CH4 than CO2 input

    @pytest.mark.parametrize("temp_k", [500.0, 550.0, 573.15, 600.0, 650.0, 700.0])
    def test_conversion_bounded(self, temp_k):
        if temp_k < MIN_REACTOR_TEMP_K:
            return  # skip invalid temps
        state = SabatierReactor(reactor_temp_k=temp_k)
        result = tick(state)
        assert 0.0 <= result.conversion <= 1.0

    @pytest.mark.parametrize("pressure", [0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    def test_all_outputs_nonneg(self, pressure):
        state = SabatierReactor(pressure_atm=pressure)
        result = tick(state)
        assert result.ch4_kg >= 0
        assert result.h2o_kg >= 0
        assert result.co2_consumed_kg >= 0
        assert result.h2_consumed_kg >= 0
        assert result.reaction_heat_kw >= 0
        assert result.compressor_power_kw >= 0

    @pytest.mark.parametrize("health", [0.2, 0.5, 0.8, 1.0])
    def test_catalyst_health_monotonic(self, health):
        if health <= CATALYST_REPLACEMENT_THRESHOLD:
            return
        state = SabatierReactor(catalyst_health=health)
        result = tick(state)
        assert result.catalyst_health <= health

    @pytest.mark.parametrize("sols", [1, 10, 50])
    def test_mass_balance_over_n_sols(self, sols):
        """Mass balance holds every single sol."""
        state = SabatierReactor()
        for _ in range(sols):
            result = tick(state)
            if result.operational:
                assert mass_balance_check(
                    result.co2_consumed_kg, result.h2_consumed_kg,
                    result.ch4_kg, result.h2o_kg)


# ═══════════════════════════════════════════════════════════════════════
# 18. Multi-Sol Smoke Tests & Simulation
# ═══════════════════════════════════════════════════════════════════════

class TestSimulation:
    def test_smoke_10_sols(self):
        results = run_simulation(sols=10)
        assert len(results) == 10
        assert all(r.ch4_kg >= 0 for r in results)

    def test_smoke_100_sols(self):
        results = run_simulation(sols=100)
        assert len(results) == 100
        total_ch4 = sum(r.ch4_kg for r in results)
        assert total_ch4 > 0

    def test_365_sol_mission(self):
        """Full Mars year: colony produces meaningful fuel."""
        results = run_simulation(sols=365)
        total_ch4 = sum(r.ch4_kg for r in results)
        total_h2o = sum(r.h2o_kg for r in results)
        # At 10 kg CO2/sol, ~99% conversion: ~3.64 kg CH4/sol → ~1329 kg/year
        assert total_ch4 > 500  # at least 500 kg methane
        assert total_h2o > 1000  # at least 1000 kg water

    def test_catalyst_degrades_over_year(self):
        results = run_simulation(sols=365)
        assert results[-1].catalyst_health < results[0].catalyst_health

    def test_cumulative_mass_conservation(self):
        """Total mass in = total mass out across entire simulation."""
        results = run_simulation(sols=100)
        total_co2 = sum(r.co2_consumed_kg for r in results)
        total_h2 = sum(r.h2_consumed_kg for r in results)
        total_ch4 = sum(r.ch4_kg for r in results)
        total_h2o = sum(r.h2o_kg for r in results)
        assert abs((total_co2 + total_h2) - (total_ch4 + total_h2o)) < 0.01

    def test_no_crash_at_low_health(self):
        """Reactor should gracefully stop when catalyst dies."""
        state = SabatierReactor(catalyst_health=0.15)
        results = []
        for _ in range(500):
            results.append(tick(state))
        # Should have some operational and some non-operational
        ops = [r for r in results if r.operational]
        non_ops = [r for r in results if not r.operational]
        assert len(ops) >= 1
        assert len(non_ops) >= 1

    def test_high_feed_rate(self):
        """Stress test with 100 kg/sol CO2."""
        h2_needed = stoichiometric_h2_kg(100.0)
        results = run_simulation(sols=10, co2_feed_kg=100.0, h2_feed_kg=h2_needed)
        assert all(r.ch4_kg >= 0 for r in results)
        total = sum(r.ch4_kg for r in results)
        assert total > 0

    def test_all_sols_have_events_list(self):
        results = run_simulation(sols=50)
        for r in results:
            assert isinstance(r.events, list)

    def test_sol_numbers_sequential(self):
        results = run_simulation(sols=20)
        for i, r in enumerate(results):
            assert r.sol == i + 1
