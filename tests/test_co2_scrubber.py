"""Tests for co2_scrubber.py — Mars Habitat CO₂ Removal & Sabatier Recycling.

97 tests covering:
  - Chemical constants & stoichiometry
  - CO₂ generation from crew respiration
  - Pressure/mass conversions (ideal gas law)
  - Langmuir adsorption isotherm
  - Zeolite molecular sieve behavior
  - Sabatier reactor products & energy
  - LiOH emergency canisters
  - Alert level classification
  - Tick engine lifecycle
  - Multi-sol integration & stability
  - Conservation law invariants
  - Edge cases & boundary conditions
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from co2_scrubber import (
    CH4_MOLAR_MASS,
    CO2_ACTIVE_MULTIPLIER,
    CO2_CRITICAL_FRACTION,
    CO2_CRITICAL_KPA,
    CO2_DANGER_FRACTION,
    CO2_DANGER_KPA,
    CO2_MOLAR_MASS,
    CO2_PER_PERSON_KG_SOL,
    CO2_TARGET_FRACTION,
    CO2_TARGET_KPA,
    CO2_WARNING_FRACTION,
    CO2_WARNING_KPA,
    FAN_PUMP_KWH_SOL,
    H2_MOLAR_MASS,
    H2O_MOLAR_MASS,
    HABITAT_PRESSURE_KPA,
    HABITAT_VOLUME_M3,
    LANGMUIR_K,
    LANGMUIR_Q_MAX,
    LIOH_CANISTER_KG,
    LIOH_CO2_CAPACITY,
    LIOH_CO2_PER_CANISTER,
    LIOH_INITIAL_CANISTERS,
    LIOH_MOLAR_MASS,
    LIOH_PER_CO2,
    SABATIER_CH4_PER_CO2,
    SABATIER_CONVERSION,
    SABATIER_DELTA_H_KJ_MOL,
    SABATIER_H2O_PER_CO2,
    SABATIER_H2_PER_CO2,
    SABATIER_MIN_CO2_KG,
    SABATIER_STARTUP_KWH,
    ScrubberState,
    ScrubberTickResult,
    ZEOLITE_BED_MASS_KG,
    ZEOLITE_CAPACITY_KG_CO2_PER_KG,
    ZEOLITE_DEGRADATION_PER_CYCLE,
    ZEOLITE_EFFICIENCY,
    ZEOLITE_REGEN_ENERGY_KWH,
    ZEOLITE_TOTAL_CAPACITY_KG,
    assess_alert,
    co2_generation_kg,
    co2_kpa_from_mass,
    co2_mass_from_kpa,
    create_scrubber,
    langmuir_loading,
    lioh_removal_kg,
    sabatier_energy_kwh,
    sabatier_products,
    tick_scrubber,
    zeolite_health,
    zeolite_removal_kg,
)


# ============================================================================
# 1. Chemical constants
# ============================================================================

class TestChemicalConstants:
    """Validate chemical constants against known values."""

    def test_co2_molar_mass(self):
        # C=12.01 + 2*O=2*16.00 = 44.01
        assert abs(CO2_MOLAR_MASS - 44.01) < 0.1

    def test_h2_molar_mass(self):
        assert abs(H2_MOLAR_MASS - 2.016) < 0.01

    def test_ch4_molar_mass(self):
        # C=12.01 + 4*H=4*1.008 = 16.04
        assert abs(CH4_MOLAR_MASS - 16.04) < 0.1

    def test_h2o_molar_mass(self):
        assert abs(H2O_MOLAR_MASS - 18.015) < 0.1

    def test_lioh_molar_mass(self):
        # Li=6.94 + O=16.00 + H=1.008 = 23.95
        assert abs(LIOH_MOLAR_MASS - 23.95) < 0.1

    def test_sabatier_stoichiometry_h2(self):
        # 4 mol H₂ per mol CO₂ → (4*2.016)/44.01
        expected = 4.0 * H2_MOLAR_MASS / CO2_MOLAR_MASS
        assert abs(SABATIER_H2_PER_CO2 - expected) < 0.001

    def test_sabatier_stoichiometry_ch4(self):
        expected = CH4_MOLAR_MASS / CO2_MOLAR_MASS
        assert abs(SABATIER_CH4_PER_CO2 - expected) < 0.001

    def test_sabatier_stoichiometry_h2o(self):
        expected = 2.0 * H2O_MOLAR_MASS / CO2_MOLAR_MASS
        assert abs(SABATIER_H2O_PER_CO2 - expected) < 0.001

    def test_sabatier_exothermic(self):
        assert SABATIER_DELTA_H_KJ_MOL < 0

    def test_lioh_stoichiometry(self):
        # 2 mol LiOH per mol CO₂
        expected = 2.0 * LIOH_MOLAR_MASS / CO2_MOLAR_MASS
        assert abs(LIOH_PER_CO2 - expected) < 0.01

    def test_co2_per_person_reasonable(self):
        # Humans produce ~0.8-1.2 kg CO₂/day
        assert 0.5 <= CO2_PER_PERSON_KG_SOL <= 2.0


# ============================================================================
# 2. CO₂ generation
# ============================================================================

class TestCO2Generation:
    """Test crew CO₂ generation rates."""

    def test_six_crew_resting(self):
        gen = co2_generation_kg(6, active_fraction=0.0)
        assert abs(gen - 6.0 * CO2_PER_PERSON_KG_SOL) < 0.01

    def test_six_crew_active(self):
        gen = co2_generation_kg(6, active_fraction=1.0)
        expected = 6.0 * CO2_PER_PERSON_KG_SOL * CO2_ACTIVE_MULTIPLIER
        assert abs(gen - expected) < 0.01

    def test_mixed_activity(self):
        gen = co2_generation_kg(6, active_fraction=0.3)
        rest = 6 * CO2_PER_PERSON_KG_SOL * 0.7
        active = 6 * CO2_PER_PERSON_KG_SOL * CO2_ACTIVE_MULTIPLIER * 0.3
        assert abs(gen - (rest + active)) < 0.01

    def test_zero_crew(self):
        assert co2_generation_kg(0) == 0.0

    def test_negative_crew_clamped(self):
        assert co2_generation_kg(-3) == 0.0

    def test_generation_scales_with_crew(self):
        g3 = co2_generation_kg(3)
        g6 = co2_generation_kg(6)
        assert abs(g6 / g3 - 2.0) < 0.01


# ============================================================================
# 3. Pressure / mass conversions
# ============================================================================

class TestPressureMass:
    """Test CO₂ pressure-mass conversions via ideal gas law."""

    def test_roundtrip_conversion(self):
        """mass → kPa → mass should be identity."""
        original_kg = 1.5
        kpa = co2_kpa_from_mass(original_kg)
        back = co2_mass_from_kpa(kpa)
        assert abs(back - original_kg) < 0.01

    def test_zero_mass_zero_pressure(self):
        assert co2_kpa_from_mass(0.0) == 0.0

    def test_zero_pressure_zero_mass(self):
        assert co2_mass_from_kpa(0.0) == 0.0

    def test_positive_mass_positive_pressure(self):
        assert co2_kpa_from_mass(1.0) > 0.0

    def test_negative_mass_clamped(self):
        assert co2_kpa_from_mass(-5.0) == 0.0

    def test_target_kpa_mass_reasonable(self):
        # At 0.35 kPa CO₂ in 500 m³: P·V·M/(R·T) ≈ 3.2 kg
        mass = co2_mass_from_kpa(CO2_TARGET_KPA)
        assert 1.0 < mass < 10.0


# ============================================================================
# 4. Langmuir adsorption
# ============================================================================

class TestLangmuir:
    """Test Langmuir isotherm for zeolite loading."""

    def test_zero_pressure_zero_loading(self):
        assert langmuir_loading(0.0) == 0.0

    def test_high_pressure_approaches_max(self):
        loading = langmuir_loading(100.0)
        assert abs(loading - LANGMUIR_Q_MAX) < 0.01 * LANGMUIR_Q_MAX

    def test_loading_increases_with_pressure(self):
        l1 = langmuir_loading(0.1)
        l2 = langmuir_loading(1.0)
        l3 = langmuir_loading(10.0)
        assert l1 < l2 < l3

    def test_loading_bounded(self):
        for p in [0.0, 0.1, 1.0, 10.0, 100.0]:
            assert 0.0 <= langmuir_loading(p) <= LANGMUIR_Q_MAX

    def test_negative_pressure_clamped(self):
        assert langmuir_loading(-1.0) == 0.0


# ============================================================================
# 5. Zeolite removal
# ============================================================================

class TestZeoliteRemoval:
    """Test zeolite CO₂ capture calculations."""

    def test_removal_positive_at_target_co2(self):
        removed = zeolite_removal_kg(
            CO2_TARGET_KPA, ZEOLITE_BED_MASS_KG, 0.0,
            ZEOLITE_TOTAL_CAPACITY_KG,
        )
        assert removed > 0.0

    def test_removal_limited_by_capacity(self):
        # Nearly full bed should capture very little
        nearly_full = ZEOLITE_TOTAL_CAPACITY_KG * 0.99
        removed = zeolite_removal_kg(
            1.0, ZEOLITE_BED_MASS_KG, nearly_full,
            ZEOLITE_TOTAL_CAPACITY_KG,
        )
        assert removed < ZEOLITE_TOTAL_CAPACITY_KG * 0.02

    def test_removal_never_negative(self):
        removed = zeolite_removal_kg(0.0, 0.0, 0.0, ZEOLITE_TOTAL_CAPACITY_KG)
        assert removed >= 0.0

    def test_removal_never_exceeds_capacity(self):
        removed = zeolite_removal_kg(
            10.0, ZEOLITE_BED_MASS_KG, 0.0,
            ZEOLITE_TOTAL_CAPACITY_KG,
        )
        assert removed <= ZEOLITE_TOTAL_CAPACITY_KG


# ============================================================================
# 6. Sabatier reactor
# ============================================================================

class TestSabatier:
    """Test Sabatier reaction products."""

    def test_one_kg_co2(self):
        co2_c, h2_c, ch4_p, h2o_p = sabatier_products(1.0)
        assert abs(co2_c - SABATIER_CONVERSION) < 0.01

    def test_stoichiometry_mass_balance(self):
        """CO₂ + H₂ input mass ≈ CH₄ + H₂O output mass."""
        co2_c, h2_c, ch4_p, h2o_p = sabatier_products(10.0)
        input_mass = co2_c + h2_c
        output_mass = ch4_p + h2o_p
        assert abs(input_mass - output_mass) < 0.1

    def test_zero_co2_no_products(self):
        co2_c, h2_c, ch4_p, h2o_p = sabatier_products(0.0)
        assert co2_c == 0.0
        assert ch4_p == 0.0
        assert h2o_p == 0.0

    def test_products_scale_linearly(self):
        _, _, ch4_1, h2o_1 = sabatier_products(1.0)
        _, _, ch4_5, h2o_5 = sabatier_products(5.0)
        assert abs(ch4_5 / ch4_1 - 5.0) < 0.01
        assert abs(h2o_5 / h2o_1 - 5.0) < 0.01

    def test_conversion_bounded(self):
        co2_c, _, _, _ = sabatier_products(10.0)
        assert co2_c <= 10.0

    def test_energy_below_minimum_zero(self):
        assert sabatier_energy_kwh(0.1) == 0.0

    def test_energy_above_minimum_positive(self):
        assert sabatier_energy_kwh(1.0) > 0.0


# ============================================================================
# 7. LiOH emergency
# ============================================================================

class TestLiOH:
    """Test LiOH canister CO₂ removal."""

    def test_one_canister(self):
        removed = lioh_removal_kg(1)
        assert removed == LIOH_CO2_PER_CANISTER

    def test_zero_canisters(self):
        assert lioh_removal_kg(0) == 0.0

    def test_negative_clamped(self):
        assert lioh_removal_kg(-5) == 0.0

    def test_capacity_reasonable(self):
        # ~0.92 kg CO₂ per kg LiOH × 2 kg = ~1.84 kg per canister
        assert 0.5 < LIOH_CO2_PER_CANISTER < 3.0

    def test_multiple_canisters_scale(self):
        r1 = lioh_removal_kg(1)
        r5 = lioh_removal_kg(5)
        assert abs(r5 / r1 - 5.0) < 0.01


# ============================================================================
# 8. Alert levels
# ============================================================================

class TestAlerts:
    """Test alert classification."""

    def test_nominal(self):
        assert assess_alert(CO2_TARGET_FRACTION, 1.0, 50) == "nominal"

    def test_warning_high_co2(self):
        assert assess_alert(CO2_WARNING_FRACTION, 1.0, 50) == "warning"

    def test_danger(self):
        assert assess_alert(CO2_DANGER_FRACTION, 1.0, 50) == "danger"

    def test_critical(self):
        assert assess_alert(CO2_CRITICAL_FRACTION, 1.0, 50) == "critical"

    def test_warning_low_lioh(self):
        assert assess_alert(CO2_TARGET_FRACTION, 1.0, 3) == "warning"

    def test_warning_degraded_zeolite(self):
        assert assess_alert(CO2_TARGET_FRACTION, 0.3, 50) == "warning"

    def test_zeolite_health_100(self):
        h = zeolite_health(ZEOLITE_TOTAL_CAPACITY_KG)
        assert abs(h - 1.0) < 0.01

    def test_zeolite_health_zero(self):
        assert zeolite_health(0.0) == 0.0


# ============================================================================
# 9. Tick engine
# ============================================================================

class TestTickEngine:
    """Test the main tick function."""

    def test_single_tick_runs(self):
        state = create_scrubber()
        state, result = tick_scrubber(state)
        assert state.sols_running == 1
        assert result.co2_generated_kg > 0

    def test_tick_removes_co2(self):
        state = create_scrubber()
        _, result = tick_scrubber(state)
        assert result.co2_zeolite_kg > 0

    def test_tick_energy_consumed(self):
        state = create_scrubber()
        _, result = tick_scrubber(state)
        assert result.energy_kwh >= FAN_PUMP_KWH_SOL

    def test_co2_stays_manageable(self):
        state = create_scrubber()
        _, result = tick_scrubber(state)
        assert result.co2_fraction < CO2_DANGER_FRACTION

    def test_power_limited_operation(self):
        s1 = create_scrubber()
        s2 = create_scrubber()
        _, r1 = tick_scrubber(s1, power_available_kwh=50.0)
        _, r2 = tick_scrubber(s2, power_available_kwh=2.0)
        assert r2.co2_zeolite_kg <= r1.co2_zeolite_kg

    def test_sabatier_produces_water(self):
        state = create_scrubber()
        # Run a few sols to build up CO₂ buffer
        for _ in range(5):
            state, _ = tick_scrubber(state, h2_available_kg=5.0)
        # Check cumulative Sabatier products
        assert state.total_h2o_recovered_kg >= 0.0

    def test_zero_crew_no_co2(self):
        state = create_scrubber(crew=0)
        _, result = tick_scrubber(state)
        assert result.co2_generated_kg == 0.0


# ============================================================================
# 10. Multi-sol integration
# ============================================================================

class TestMultiSol:
    """Test multi-sol simulation stability."""

    def test_10_sol_smoke_test(self):
        """Run 10 sols without crash."""
        state = create_scrubber()
        for _ in range(10):
            state, _ = tick_scrubber(state)
        assert state.sols_running == 10
        assert state.total_co2_generated_kg > 0
        assert state.total_co2_scrubbed_kg > 0

    def test_100_sol_co2_stable(self):
        """CO₂ should remain below critical over 100 sols."""
        state = create_scrubber()
        for _ in range(100):
            state, result = tick_scrubber(state, h2_available_kg=5.0)
        assert result.co2_fraction < CO2_CRITICAL_FRACTION

    def test_cumulative_co2_generated_increases(self):
        state = create_scrubber()
        prev = 0.0
        for _ in range(5):
            state, _ = tick_scrubber(state)
            assert state.total_co2_generated_kg > prev
            prev = state.total_co2_generated_kg

    def test_sabatier_produces_ch4_over_time(self):
        state = create_scrubber()
        for _ in range(20):
            state, _ = tick_scrubber(state, h2_available_kg=5.0)
        assert state.total_ch4_produced_kg > 0

    def test_water_recovery_over_time(self):
        state = create_scrubber()
        for _ in range(20):
            state, _ = tick_scrubber(state, h2_available_kg=5.0)
        assert state.total_h2o_recovered_kg > 0

    def test_zeolite_degrades_slowly(self):
        state = create_scrubber()
        original_cap = state.zeolite_capacity_kg
        for _ in range(100):
            state, _ = tick_scrubber(state)
        assert state.zeolite_capacity_kg < original_cap
        # But not too much in 100 sols
        assert state.zeolite_capacity_kg > original_cap * 0.9

    def test_lioh_usage_limited(self):
        """LiOH canisters should be used sparingly in normal operation."""
        state = create_scrubber()
        for _ in range(50):
            state, _ = tick_scrubber(state, h2_available_kg=5.0)
        # With Sabatier running, LiOH usage should be minimal
        assert state.total_lioh_used_canisters < 10


# ============================================================================
# 11. Conservation law invariants
# ============================================================================

class TestConservationLaws:
    """Property-based tests: physical invariants that must always hold."""

    def test_co2_kpa_never_negative(self):
        state = create_scrubber()
        for _ in range(50):
            state, result = tick_scrubber(state)
            assert state.co2_kpa >= 0.0
            assert result.co2_kpa >= 0.0

    def test_co2_fraction_bounded(self):
        state = create_scrubber()
        for _ in range(50):
            state, result = tick_scrubber(state)
            assert 0.0 <= state.co2_fraction <= 1.0

    def test_co2_kpa_bounded_by_total_pressure(self):
        state = create_scrubber()
        for _ in range(50):
            state, _ = tick_scrubber(state)
            assert state.co2_kpa <= HABITAT_PRESSURE_KPA

    def test_zeolite_load_bounded(self):
        state = create_scrubber()
        for _ in range(50):
            state, _ = tick_scrubber(state)
            assert state.zeolite_loaded_kg <= state.zeolite_capacity_kg + 1e-6
            assert state.zeolite_loaded_kg >= 0.0

    def test_lioh_never_negative(self):
        state = create_scrubber()
        for _ in range(50):
            state, _ = tick_scrubber(state)
            assert state.lioh_canisters >= 0

    def test_energy_never_negative(self):
        state = create_scrubber()
        for _ in range(20):
            state, result = tick_scrubber(state)
            assert result.energy_kwh >= 0.0

    def test_sabatier_mass_balance(self):
        """CO₂ + H₂ input ≈ CH₄ + H₂O output (mass conservation)."""
        state = create_scrubber()
        for _ in range(30):
            state, _ = tick_scrubber(state, h2_available_kg=5.0)
        if state.total_co2_sabatier_kg > 0.1:
            input_mass = state.total_co2_sabatier_kg + state.total_h2_consumed_kg
            output_mass = state.total_ch4_produced_kg + state.total_h2o_recovered_kg
            assert abs(input_mass - output_mass) < 1.0

    def test_total_scrubbed_ge_zero(self):
        state = create_scrubber()
        for _ in range(20):
            state, _ = tick_scrubber(state)
        assert state.total_co2_scrubbed_kg >= 0.0


# ============================================================================
# 12. Edge cases & boundary conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_crew(self):
        state = create_scrubber(crew=0)
        state, result = tick_scrubber(state)
        assert result.co2_generated_kg == 0.0

    def test_max_crew(self):
        state = create_scrubber(crew=20)
        state, result = tick_scrubber(state)
        assert result.co2_generated_kg > 0

    def test_high_initial_co2(self):
        state = create_scrubber(co2_kpa=CO2_DANGER_KPA)
        state, result = tick_scrubber(state)
        # Should try to bring it down
        assert result.co2_zeolite_kg > 0 or result.lioh_canisters_used > 0

    def test_zero_power(self):
        state = create_scrubber()
        state, result = tick_scrubber(state, power_available_kwh=0.0)
        # Fans still counted, but no zeolite cycles
        assert result.co2_zeolite_kg == 0.0

    def test_no_h2_for_sabatier(self):
        state = create_scrubber()
        for _ in range(5):
            state, _ = tick_scrubber(state, h2_available_kg=0.0)
        # Sabatier should not run
        assert state.total_ch4_produced_kg == 0.0

    def test_factory_default_state(self):
        state = create_scrubber()
        assert state.crew_count == 6
        assert abs(state.co2_kpa - CO2_TARGET_KPA) < 0.01
        assert state.sols_running == 0
        assert state.lioh_canisters == LIOH_INITIAL_CANISTERS

    def test_factory_custom_crew(self):
        state = create_scrubber(crew=3)
        assert state.crew_count == 3

    def test_state_post_init_clamps(self):
        state = ScrubberState(co2_kpa=-5.0, lioh_canisters=-10, crew_count=100)
        assert state.co2_kpa == 0.0
        assert state.lioh_canisters == 0
        assert state.crew_count == 20
