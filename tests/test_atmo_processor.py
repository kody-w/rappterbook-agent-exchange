"""
Tests for atmo_processor.py — Mars ISRU atmospheric processing.

Covers: MOXIE electrolysis, Sabatier methanation, water electrolysis,
O₂ budget tracking, propellant accumulation, dust degradation,
temperature efficiency, conservation of mass, and multi-sol smoke tests.

61 tests across 12 test classes.
"""
from __future__ import annotations

import math
import sys
import os
import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atmo_processor import (
    # Constants
    O2_KG_PER_PERSON_SOL,
    MOXIE_O2_PER_KG_CO2,
    MOXIE_CO_PER_KG_CO2,
    MOXIE_KWH_PER_KG_O2,
    MOXIE_RATED_KG_O2_SOL,
    SABATIER_H2_PER_KG_CO2,
    SABATIER_CH4_PER_KG_CO2,
    SABATIER_H2O_PER_KG_CO2,
    ELECTROLYSIS_H2_PER_KG_H2O,
    ELECTROLYSIS_O2_PER_KG_H2O,
    ELECTROLYSIS_KWH_PER_KG_H2O,
    COLD_PENALTY_FLOOR,
    WARM_EFFICIENCY_CEIL,
    MAV_PROPELLANT_TARGET_KG,
    DUST_FILTER_CLOG_RATE,
    FILTER_MAINTENANCE_CLEAR,
    # Data structures
    MoxieBank,
    SabatierBank,
    AtmoState,
    TickResult,
    # Functions
    temperature_efficiency,
    dust_throughput_factor,
    moxie_output,
    sabatier_output,
    electrolyze_water,
    colony_o2_demand,
    mav_progress_fraction,
    tick_atmo,
    create_isru,
    perform_maintenance,
)


# =========================================================================
# 1. Stoichiometric constant validation
# =========================================================================

class TestStoichiometry:
    """Verify mass ratios obey conservation of mass."""

    def test_moxie_mass_balance(self):
        """CO₂ → CO + ½O₂: products must equal input mass."""
        assert abs(MOXIE_O2_PER_KG_CO2 + MOXIE_CO_PER_KG_CO2 - 1.0) < 1e-10

    def test_sabatier_mass_balance(self):
        """CO₂ + 4H₂ → CH₄ + 2H₂O: input mass == output mass."""
        input_mass = 1.0 + SABATIER_H2_PER_KG_CO2
        output_mass = SABATIER_CH4_PER_KG_CO2 + SABATIER_H2O_PER_KG_CO2
        assert abs(input_mass - output_mass) < 1e-10

    def test_electrolysis_mass_balance(self):
        """2H₂O → 2H₂ + O₂: products must equal input mass."""
        assert abs(ELECTROLYSIS_H2_PER_KG_H2O + ELECTROLYSIS_O2_PER_KG_H2O - 1.0) < 1e-10

    def test_moxie_o2_ratio_physical(self):
        """O₂ output per kg CO₂ matches molecular weight ratio 16/44."""
        assert abs(MOXIE_O2_PER_KG_CO2 - 16.0 / 44.0) < 1e-10

    def test_sabatier_ch4_ratio_physical(self):
        """CH₄ output per kg CO₂ matches molecular weight ratio 16/44."""
        assert abs(SABATIER_CH4_PER_KG_CO2 - 16.0 / 44.0) < 1e-10

    def test_electrolysis_h2_ratio_physical(self):
        """H₂ output per kg H₂O matches molecular weight ratio 4/36."""
        assert abs(ELECTROLYSIS_H2_PER_KG_H2O - 4.0 / 36.0) < 1e-10


# =========================================================================
# 2. Temperature efficiency
# =========================================================================

class TestTemperatureEfficiency:
    """Verify temperature-dependent ISRU efficiency curve."""

    def test_at_zero_c(self):
        """At 0°C (warm Mars), full efficiency."""
        assert abs(temperature_efficiency(0.0) - WARM_EFFICIENCY_CEIL) < 1e-10

    def test_at_negative_120(self):
        """At -120°C (extreme cold), minimum efficiency."""
        assert abs(temperature_efficiency(-120.0) - COLD_PENALTY_FLOOR) < 1e-10

    def test_at_minus_60(self):
        """At -60°C (Mars mean), halfway between floor and ceiling."""
        expected = COLD_PENALTY_FLOOR + 0.5 * (WARM_EFFICIENCY_CEIL - COLD_PENALTY_FLOOR)
        assert abs(temperature_efficiency(-60.0) - expected) < 1e-10

    def test_monotonically_increasing(self):
        """Warmer temperatures → higher efficiency."""
        temps = list(range(-120, 1, 10))
        effs = [temperature_efficiency(t) for t in temps]
        for i in range(1, len(effs)):
            assert effs[i] >= effs[i - 1]

    def test_clamped_above_zero(self):
        """Temperatures above 0°C clamp to ceiling."""
        assert abs(temperature_efficiency(50.0) - WARM_EFFICIENCY_CEIL) < 1e-10

    def test_clamped_below_minus_120(self):
        """Temperatures below -120°C clamp to floor."""
        assert abs(temperature_efficiency(-200.0) - COLD_PENALTY_FLOOR) < 1e-10

    def test_always_positive(self):
        """Efficiency is always > 0 at any temperature."""
        for t in range(-200, 100, 5):
            assert temperature_efficiency(float(t)) > 0


# =========================================================================
# 3. Dust throughput factor
# =========================================================================

class TestDustThroughput:
    """Verify dust clogging effect on throughput."""

    def test_clean_filter(self):
        """No clogging → full throughput."""
        assert abs(dust_throughput_factor(0.0) - 1.0) < 1e-10

    def test_fully_clogged(self):
        """Fully clogged → zero throughput."""
        assert abs(dust_throughput_factor(1.0)) < 1e-10

    def test_quadratic_shape(self):
        """Light dust has little effect (quadratic curve)."""
        light = dust_throughput_factor(0.1)
        heavy = dust_throughput_factor(0.9)
        assert light > 0.95  # 0.1² = 0.01, so ~0.99
        assert heavy < 0.25  # 0.9² = 0.81, so ~0.19

    def test_monotonically_decreasing(self):
        """More clogging → less throughput."""
        clogs = [i / 20.0 for i in range(21)]
        factors = [dust_throughput_factor(c) for c in clogs]
        for i in range(1, len(factors)):
            assert factors[i] <= factors[i - 1] + 1e-10

    def test_clamped_range(self):
        """Throughput always in [0, 1] even with out-of-range inputs."""
        assert 0.0 <= dust_throughput_factor(-0.5) <= 1.0
        assert 0.0 <= dust_throughput_factor(1.5) <= 1.0


# =========================================================================
# 4. MOXIE output
# =========================================================================

class TestMoxieOutput:
    """Verify MOXIE CO₂ electrolysis computations."""

    def test_zero_units(self):
        """Zero MOXIE units produce nothing."""
        bank = MoxieBank(units=0)
        o2, co, co2, pwr = moxie_output(bank, 1000.0, -20.0)
        assert o2 == 0.0 and co == 0.0 and co2 == 0.0 and pwr == 0.0

    def test_zero_power(self):
        """Zero power produces nothing."""
        bank = MoxieBank(units=3)
        o2, co, co2, pwr = moxie_output(bank, 0.0, -20.0)
        assert o2 == 0.0

    def test_mass_conservation(self):
        """CO₂ consumed == O₂ produced + CO produced."""
        bank = MoxieBank(units=2)
        o2, co, co2, pwr = moxie_output(bank, 500.0, -30.0)
        assert abs(co2 - o2 - co) < 1e-10

    def test_power_limited(self):
        """When power is scarce, O₂ production is capped."""
        bank = MoxieBank(units=10)  # many units
        tiny_power = 10.0  # only enough for 0.4 kg O₂
        o2, _, _, pwr = moxie_output(bank, tiny_power, 0.0)
        assert o2 <= tiny_power / MOXIE_KWH_PER_KG_O2 + 1e-10
        assert pwr <= tiny_power + 1e-10

    def test_rated_output_at_ideal(self):
        """At 0°C, no dust, unlimited power → rated output."""
        bank = MoxieBank(units=1, filter_clog=0.0)
        o2, _, _, _ = moxie_output(bank, 10000.0, 0.0)
        assert abs(o2 - MOXIE_RATED_KG_O2_SOL) < 1e-10

    def test_dust_reduces_output(self):
        """Clogged filters reduce O₂ production."""
        clean = MoxieBank(units=1, filter_clog=0.0)
        dirty = MoxieBank(units=1, filter_clog=0.5)
        o2_clean, _, _, _ = moxie_output(clean, 10000.0, 0.0)
        o2_dirty, _, _, _ = moxie_output(dirty, 10000.0, 0.0)
        assert o2_dirty < o2_clean

    def test_cold_reduces_output(self):
        """Cold temperature reduces efficiency."""
        bank = MoxieBank(units=1)
        o2_warm, _, _, _ = moxie_output(bank, 10000.0, 0.0)
        o2_cold, _, _, _ = moxie_output(bank, 10000.0, -100.0)
        assert o2_cold < o2_warm


# =========================================================================
# 5. Sabatier output
# =========================================================================

class TestSabatierOutput:
    """Verify Sabatier reactor computations."""

    def test_zero_units(self):
        """Zero reactors produce nothing."""
        bank = SabatierBank(units=0)
        ch4, h2o, co2, h2, heat = sabatier_output(bank, 10.0, -20.0)
        assert ch4 == 0.0 and h2o == 0.0 and heat == 0.0

    def test_zero_hydrogen(self):
        """No hydrogen feedstock → no production."""
        bank = SabatierBank(units=1)
        ch4, h2o, co2, h2, heat = sabatier_output(bank, 0.0, -20.0)
        assert ch4 == 0.0

    def test_mass_conservation(self):
        """CO₂ consumed + H₂ consumed == CH₄ produced + H₂O produced."""
        bank = SabatierBank(units=1)
        ch4, h2o, co2, h2, heat = sabatier_output(bank, 5.0, -30.0)
        input_mass = co2 + h2
        output_mass = ch4 + h2o
        assert abs(input_mass - output_mass) < 1e-10

    def test_h2_limited(self):
        """When hydrogen is scarce, production scales down."""
        bank = SabatierBank(units=10, rated_kg_co2_sol=100.0)
        tiny_h2 = 0.1
        ch4, h2o, co2, h2_consumed, _ = sabatier_output(bank, tiny_h2, 0.0)
        assert h2_consumed <= tiny_h2 + 1e-10

    def test_exothermic(self):
        """Sabatier reaction produces heat (positive kJ)."""
        bank = SabatierBank(units=1)
        _, _, _, _, heat = sabatier_output(bank, 5.0, -20.0)
        assert heat > 0


# =========================================================================
# 6. Water electrolysis
# =========================================================================

class TestElectrolyzeWater:
    """Verify water electrolysis computations."""

    def test_zero_water(self):
        """No water → no products."""
        h2, o2, h2o, pwr = electrolyze_water(0.0, 100.0)
        assert h2 == 0.0 and o2 == 0.0

    def test_zero_power(self):
        """No power → no products."""
        h2, o2, h2o, pwr = electrolyze_water(100.0, 0.0)
        assert h2 == 0.0 and o2 == 0.0

    def test_mass_conservation(self):
        """H₂O consumed == H₂ + O₂ produced."""
        h2, o2, h2o, _ = electrolyze_water(10.0, 100.0)
        assert abs(h2o - h2 - o2) < 1e-10

    def test_power_limited(self):
        """Power limits water throughput."""
        tiny_power = 1.0  # ~0.19 kg H₂O
        h2, o2, h2o, pwr = electrolyze_water(1000.0, tiny_power)
        assert h2o <= tiny_power / ELECTROLYSIS_KWH_PER_KG_H2O + 1e-10

    def test_target_h2_cap(self):
        """target_h2_kg caps hydrogen production."""
        h2, o2, h2o, _ = electrolyze_water(100.0, 1000.0, target_h2_kg=0.05)
        assert h2 <= 0.05 + 1e-10

    def test_target_h2_preserves_ratio(self):
        """Capping H₂ still preserves H₂:O₂ mass ratio."""
        h2, o2, h2o, _ = electrolyze_water(100.0, 1000.0, target_h2_kg=0.05)
        if h2 > 1e-15:
            ratio = o2 / h2
            expected_ratio = ELECTROLYSIS_O2_PER_KG_H2O / ELECTROLYSIS_H2_PER_KG_H2O
            assert abs(ratio - expected_ratio) < 1e-8


# =========================================================================
# 7. O₂ demand
# =========================================================================

class TestO2Demand:
    """Verify colony oxygen demand calculations."""

    def test_single_person(self):
        """One person needs 0.84 kg O₂/sol."""
        assert abs(colony_o2_demand(1) - O2_KG_PER_PERSON_SOL) < 1e-10

    def test_hundred_people(self):
        """100 people scale linearly."""
        assert abs(colony_o2_demand(100) - 100 * O2_KG_PER_PERSON_SOL) < 1e-10

    def test_zero_population(self):
        """Zero people → zero demand."""
        assert colony_o2_demand(0) == 0.0

    def test_negative_population(self):
        """Negative population clamped to zero."""
        assert colony_o2_demand(-5) == 0.0


# =========================================================================
# 8. MAV progress
# =========================================================================

class TestMavProgress:
    """Verify Mars Ascent Vehicle propellant tracking."""

    def test_zero_ch4(self):
        """No methane → 0% progress."""
        assert mav_progress_fraction(0.0) == 0.0

    def test_full_target(self):
        """Full CH₄ target → 100%."""
        assert mav_progress_fraction(MAV_PROPELLANT_TARGET_KG) == 1.0

    def test_clamped_at_one(self):
        """Excess CH₄ still clamps to 100%."""
        assert mav_progress_fraction(MAV_PROPELLANT_TARGET_KG * 2) == 1.0

    def test_monotonic(self):
        """More CH₄ → higher progress."""
        vals = [mav_progress_fraction(i * 100) for i in range(100)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]


# =========================================================================
# 9. Data structure validation
# =========================================================================

class TestDataStructures:
    """Verify dataclass defaults and clamping."""

    def test_moxie_bank_defaults(self):
        """Default MoxieBank has 1 unit, clean filter."""
        bank = MoxieBank()
        assert bank.units == 1
        assert bank.filter_clog == 0.0

    def test_moxie_bank_clamps_negative(self):
        """Negative values are clamped."""
        bank = MoxieBank(units=-1, filter_clog=-0.5, age_sols=-10)
        assert bank.units == 0
        assert bank.filter_clog == 0.0
        assert bank.age_sols == 0

    def test_atmo_state_clamps(self):
        """Negative stockpiles are clamped to zero."""
        state = AtmoState(o2_buffer_kg=-10, ch4_stockpile_kg=-5)
        assert state.o2_buffer_kg == 0.0
        assert state.ch4_stockpile_kg == 0.0

    def test_sabatier_bank_clamps(self):
        """Sabatier bank clamps negative values."""
        bank = SabatierBank(units=-3, rated_kg_co2_sol=-1.0)
        assert bank.units == 0
        assert bank.rated_kg_co2_sol == 0.0

    def test_create_isru_conservative(self):
        """Conservative ISRU has most units and largest buffer."""
        moxie_c, sab_c, state_c = create_isru("conservative")
        moxie_a, sab_a, state_a = create_isru("aggressive")
        assert moxie_c.units > moxie_a.units
        assert state_c.o2_buffer_kg > state_a.o2_buffer_kg

    def test_create_isru_unknown_strategy(self):
        """Unknown strategy falls back to balanced."""
        moxie, sab, state = create_isru("unknown")
        moxie_b, sab_b, state_b = create_isru("balanced")
        assert moxie.units == moxie_b.units


# =========================================================================
# 10. Tick integration
# =========================================================================

class TestTickAtmo:
    """Integration tests for tick_atmo — one sol of processing."""

    def test_basic_tick_produces_o2(self):
        """A colony with MOXIE should produce O₂."""
        moxie = MoxieBank(units=2)
        sabatier = SabatierBank(units=0)
        state = AtmoState(o2_buffer_kg=100.0)
        result = tick_atmo(moxie, sabatier, state, 10, 500.0, 0.0, -30.0, 0.0)
        assert result.o2_produced_kg > 0

    def test_o2_consumed_matches_population(self):
        """O₂ consumption equals population × daily rate."""
        moxie = MoxieBank(units=5)
        sabatier = SabatierBank(units=0)
        state = AtmoState(o2_buffer_kg=1000.0)
        pop = 50
        result = tick_atmo(moxie, sabatier, state, pop, 5000.0, 0.0, -30.0, 0.0)
        assert abs(result.o2_consumed_kg - pop * O2_KG_PER_PERSON_SOL) < 1e-10

    def test_deficit_tracking(self):
        """When production + buffer < demand, deficit is tracked."""
        moxie = MoxieBank(units=0)
        sabatier = SabatierBank(units=0)
        state = AtmoState(o2_buffer_kg=1.0)  # tiny buffer, no production
        result = tick_atmo(moxie, sabatier, state, 100, 0.0, 0.0, -30.0, 0.0)
        assert result.o2_deficit_kg > 0
        assert state.deficit_sols == 1
        assert state.o2_buffer_kg == 0.0

    def test_no_deficit_with_enough_production(self):
        """Sufficient MOXIE prevents O₂ deficit."""
        moxie = MoxieBank(units=5)
        sabatier = SabatierBank(units=0)
        state = AtmoState(o2_buffer_kg=500.0)
        # 5 units × 5 kg/sol = 25 kg O₂, demand = 20 × 0.84 = 16.8
        result = tick_atmo(moxie, sabatier, state, 20, 10000.0, 0.0, 0.0, 0.0)
        assert result.o2_deficit_kg == 0.0

    def test_sabatier_produces_ch4(self):
        """With H₂ stockpile and Sabatier, CH₄ is produced."""
        moxie = MoxieBank(units=1)
        sabatier = SabatierBank(units=1)
        state = AtmoState(o2_buffer_kg=100.0, h2_stockpile_kg=5.0)
        result = tick_atmo(moxie, sabatier, state, 10, 500.0, 0.0, -20.0, 0.0)
        assert result.ch4_produced_kg > 0

    def test_electrolysis_feeds_sabatier(self):
        """Water electrolysis provides H₂ for Sabatier."""
        moxie = MoxieBank(units=1)
        sabatier = SabatierBank(units=1)
        state = AtmoState(o2_buffer_kg=100.0, h2_stockpile_kg=0.0)
        # Provide water but no H₂ stockpile
        result = tick_atmo(moxie, sabatier, state, 10, 1000.0, 50.0, -20.0, 0.0)
        # Electrolysis should make H₂, Sabatier should use it
        assert result.h2_produced_kg > 0
        assert result.ch4_produced_kg > 0

    def test_dust_clogs_filters(self):
        """Dust opacity increases filter clogging over time."""
        moxie = MoxieBank(units=1, filter_clog=0.0)
        sabatier = SabatierBank(units=0)
        state = AtmoState(o2_buffer_kg=1000.0)
        tick_atmo(moxie, sabatier, state, 10, 500.0, 0.0, -30.0, 0.8)
        assert moxie.filter_clog > 0

    def test_ch4_accumulates(self):
        """CH₄ stockpile grows over multiple ticks."""
        moxie = MoxieBank(units=2)
        sabatier = SabatierBank(units=1)
        state = AtmoState(o2_buffer_kg=500.0, h2_stockpile_kg=20.0)
        for _ in range(10):
            tick_atmo(moxie, sabatier, state, 10, 1000.0, 10.0, -20.0, 0.0)
        assert state.ch4_stockpile_kg > 0
        assert state.total_ch4_produced_kg > 0


# =========================================================================
# 11. Physical invariants (property-based)
# =========================================================================

class TestPhysicalInvariants:
    """Property-based tests: physics must hold across any input."""

    @pytest.mark.parametrize("units", [0, 1, 5, 20])
    @pytest.mark.parametrize("temp", [-120.0, -60.0, 0.0])
    def test_o2_non_negative(self, units, temp):
        """O₂ production is never negative."""
        bank = MoxieBank(units=units)
        o2, _, _, _ = moxie_output(bank, 500.0, temp)
        assert o2 >= 0

    @pytest.mark.parametrize("h2", [0.0, 0.5, 5.0, 50.0])
    def test_sabatier_conservation_parametric(self, h2):
        """Sabatier mass conservation across input ranges."""
        bank = SabatierBank(units=2)
        ch4, h2o, co2, h2_c, _ = sabatier_output(bank, h2, -30.0)
        assert abs((co2 + h2_c) - (ch4 + h2o)) < 1e-10

    def test_moxie_conservation_sweep(self):
        """MOXIE mass conservation across 50 parameter combinations."""
        for units in [1, 3, 5]:
            for power in [10.0, 100.0, 1000.0]:
                for temp in [-100.0, -50.0, 0.0]:
                    for clog in [0.0, 0.3, 0.7]:
                        bank = MoxieBank(units=units, filter_clog=clog)
                        o2, co, co2, _ = moxie_output(bank, power, temp)
                        assert abs(co2 - o2 - co) < 1e-10

    def test_electrolysis_conservation_sweep(self):
        """Water electrolysis mass conservation across inputs."""
        for water in [0.1, 1.0, 10.0, 100.0]:
            for power in [0.5, 5.0, 50.0, 500.0]:
                h2, o2, h2o, _ = electrolyze_water(water, power)
                assert abs(h2o - h2 - o2) < 1e-10

    def test_buffer_never_negative(self):
        """O₂ buffer never goes negative, even under extreme deficit."""
        moxie = MoxieBank(units=0)
        sabatier = SabatierBank(units=0)
        state = AtmoState(o2_buffer_kg=5.0)
        for _ in range(100):
            tick_atmo(moxie, sabatier, state, 1000, 0.0, 0.0, -60.0, 0.5)
        assert state.o2_buffer_kg >= 0.0

    def test_power_consumed_never_exceeds_budget(self):
        """Total power consumed never exceeds allocated budget."""
        moxie = MoxieBank(units=10)
        sabatier = SabatierBank(units=5)
        state = AtmoState(o2_buffer_kg=100.0, h2_stockpile_kg=50.0)
        budget = 200.0
        result = tick_atmo(moxie, sabatier, state, 50, budget, 100.0, -30.0, 0.0)
        assert result.power_consumed_kwh <= budget + 1e-10


# =========================================================================
# 12. Smoke test — 365-sol continuous run
# =========================================================================

class TestSmoke:
    """Multi-sol integration smoke tests."""

    def test_365_sol_run(self):
        """Run ISRU for a full Mars year without crash or negative state."""
        moxie, sabatier, state = create_isru("balanced")
        pop = 80

        for sol in range(365):
            temp = -60.0 + 50.0 * math.sin(2 * math.pi * sol / 668.6)
            dust = 0.1 + 0.05 * math.sin(2 * math.pi * sol / 100)
            dust = max(0.0, min(1.0, dust))

            result = tick_atmo(
                moxie, sabatier, state, pop,
                power_budget_kwh=800.0,
                water_budget_kg=20.0,
                temp_c=temp,
                dust_opacity=dust,
            )

            # Invariants every sol
            assert state.o2_buffer_kg >= 0.0
            assert state.ch4_stockpile_kg >= 0.0
            assert state.h2_stockpile_kg >= 0.0
            assert result.power_consumed_kwh >= 0.0

            # Periodic maintenance every 30 sols
            if sol % 30 == 29:
                perform_maintenance(moxie)

        # After a year: cumulative production should be meaningful
        assert state.total_o2_produced_kg > 0
        assert state.total_power_consumed_kwh > 0

    def test_conservative_colony_no_deficit(self):
        """Conservative colony (30 MOXIE units) should never run out of O₂."""
        moxie, sabatier, state = create_isru("conservative")
        pop = 120
        # Power budget: 120 × 0.84 kg O₂ × 25 kWh/kg = 2520 kWh minimum
        # Plus margin for electrolysis: allocate 4000 kWh
        for sol in range(365):
            temp = -60.0 + 50.0 * math.sin(2 * math.pi * sol / 668.6)
            result = tick_atmo(
                moxie, sabatier, state, pop,
                power_budget_kwh=4000.0,
                water_budget_kg=30.0,
                temp_c=temp,
                dust_opacity=0.1,
            )
            if sol % 30 == 29:
                perform_maintenance(moxie)

        # With 30 units × 5 kg/sol = 150 kg O₂ capacity
        # At ~85% avg efficiency ≈ 127 kg/sol > demand of 100.8 kg/sol
        # 4000 kWh budget supports ~160 kg O₂ production
        assert state.deficit_sols == 0

    def test_maintenance_clears_clog(self):
        """Maintenance reduces filter clogging."""
        moxie = MoxieBank(units=1, filter_clog=0.4)
        cleared = perform_maintenance(moxie)
        assert cleared > 0
        assert moxie.filter_clog < 0.4

    def test_aggressive_colony_propellant(self):
        """Aggressive colony with Sabatier accumulates CH₄ over a year."""
        moxie, sabatier, state = create_isru("aggressive")
        sabatier.units = 2
        pop = 60
        # Power: 60 × 0.84 × 25 = 1260 kWh for O₂ + margin for electrolysis
        for sol in range(365):
            temp = -60.0 + 30.0 * math.sin(2 * math.pi * sol / 668.6)
            tick_atmo(
                moxie, sabatier, state, pop,
                power_budget_kwh=2500.0,
                water_budget_kg=15.0,
                temp_c=temp,
                dust_opacity=0.05,
            )
            if sol % 30 == 29:
                perform_maintenance(moxie)

        assert state.ch4_stockpile_kg > 0
        assert state.total_ch4_produced_kg > 0
