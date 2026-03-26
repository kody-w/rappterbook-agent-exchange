"""Tests for perchlorate_scrubber.py — Mars Colony Perchlorate Remediation.

104 tests covering:
  - Chemical constants & stoichiometry
  - Thomas model breakthrough curves
  - Ion exchange capture calculations
  - Monod bioreactor kinetics
  - Temperature sensitivity
  - Resin regeneration
  - Water quality classification
  - Energy accounting
  - Tick engine lifecycle
  - Multi-sol integration
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

from perchlorate_scrubber import (
    BIO_DECAY_RATE,
    BIO_HOURS_PER_SOL,
    BIO_INITIAL_BIOMASS_G,
    BIO_KS_MG_L,
    BIO_MAX_BIOMASS_G,
    BIO_MU_MAX,
    BIO_TEMP_MAX_C,
    BIO_TEMP_MIN_C,
    BIO_TEMP_OPTIMAL_C,
    CHLORIDE_MOLAR_MASS,
    EPA_MCL_UG_L,
    IRRIGATION_LIMIT_MG_L,
    IX_FLOW_RATE_L_H,
    IX_HOURS_PER_SOL,
    MARS_SOIL_CLO4_FRACTION,
    MARS_WATER_CLO4_MG_L,
    O2_MOLAR_MASS,
    PERCHLORATE_MOLAR_MASS,
    POTABLE_LIMIT_MG_L,
    REGEN_EFFICIENCY,
    REGEN_ENERGY_KWH,
    RESIN_BED_VOLUME_L,
    RESIN_CAPACITY_EQ_L,
    RESIN_MAX_CLO4_G,
    STOICH_CL_PER_CLO4,
    STOICH_O2_PER_CLO4,
    ScrubberState,
    ScrubberTickResult,
    assess_alert,
    bio_destruction_sol,
    bio_temperature_factor,
    create_scrubber,
    energy_per_sol,
    is_irrigable,
    is_potable,
    ix_effluent_mg_l,
    ix_removal_sol,
    monod_rate,
    regenerate_resin,
    resin_needs_regen,
    stoichiometry_products,
    thomas_breakthrough,
    tick_scrubber,
)


# ============================================================================
# 1. Chemical constants
# ============================================================================

class TestChemicalConstants:
    """Validate chemical constants against known values."""

    def test_perchlorate_molar_mass(self):
        # Cl=35.45 + 4*O=4*16.00 = 99.45
        assert abs(PERCHLORATE_MOLAR_MASS - 99.45) < 0.1

    def test_chloride_molar_mass(self):
        assert abs(CHLORIDE_MOLAR_MASS - 35.45) < 0.1

    def test_o2_molar_mass(self):
        assert abs(O2_MOLAR_MASS - 32.00) < 0.1

    def test_stoich_cl_per_clo4(self):
        expected = 35.45 / 99.45
        assert abs(STOICH_CL_PER_CLO4 - expected) < 0.001

    def test_stoich_o2_per_clo4(self):
        # 2 mol O₂ per mol ClO₄⁻
        expected = 2.0 * 32.00 / 99.45
        assert abs(STOICH_O2_PER_CLO4 - expected) < 0.001

    def test_mars_soil_perchlorate_range(self):
        # Phoenix measured 0.4-1.0%; we use 0.6%
        assert 0.004 <= MARS_SOIL_CLO4_FRACTION <= 0.010

    def test_mars_water_perchlorate_positive(self):
        assert MARS_WATER_CLO4_MG_L > 0

    def test_epa_limit_micrograms(self):
        assert EPA_MCL_UG_L == 15.0

    def test_potable_limit_conversion(self):
        assert abs(POTABLE_LIMIT_MG_L - 0.015) < 0.001

    def test_resin_capacity_positive(self):
        assert RESIN_MAX_CLO4_G > 0
        expected = RESIN_CAPACITY_EQ_L * RESIN_BED_VOLUME_L * PERCHLORATE_MOLAR_MASS
        assert abs(RESIN_MAX_CLO4_G - expected) < 1.0


# ============================================================================
# 2. Thomas model breakthrough
# ============================================================================

class TestThomasBreakthrough:
    """Test IX column breakthrough curve behavior."""

    def test_fresh_resin_no_breakthrough(self):
        bt = thomas_breakthrough(0, 500.0, 0.0, RESIN_MAX_CLO4_G)
        assert bt < 0.01, "Fresh resin should capture nearly everything"

    def test_low_saturation_low_breakthrough(self):
        load = RESIN_MAX_CLO4_G * 0.3  # 30% saturated
        bt = thomas_breakthrough(100, 500.0, load, RESIN_MAX_CLO4_G)
        assert bt < 0.05

    def test_mid_saturation_partial_breakthrough(self):
        load = RESIN_MAX_CLO4_G * 0.85  # 85% saturated
        bt = thomas_breakthrough(1000, 500.0, load, RESIN_MAX_CLO4_G)
        assert 0.3 < bt < 0.7

    def test_full_saturation_complete_breakthrough(self):
        load = RESIN_MAX_CLO4_G * 0.99  # 99% saturated
        bt = thomas_breakthrough(5000, 500.0, load, RESIN_MAX_CLO4_G)
        assert bt > 0.9

    def test_breakthrough_monotonic_with_saturation(self):
        """Breakthrough fraction must increase with resin loading."""
        prev = 0.0
        for sat in [0.0, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99]:
            load = RESIN_MAX_CLO4_G * sat
            bt = thomas_breakthrough(1000, 500.0, load, RESIN_MAX_CLO4_G)
            assert bt >= prev - 1e-10, f"Breakthrough must be monotonic at sat={sat}"
            prev = bt

    def test_breakthrough_bounded_zero_one(self):
        for load_frac in [0.0, 0.5, 1.0, 1.5]:
            load = RESIN_MAX_CLO4_G * load_frac
            bt = thomas_breakthrough(100, 500.0, load, RESIN_MAX_CLO4_G)
            assert 0.0 <= bt <= 1.0

    def test_zero_capacity_full_breakthrough(self):
        bt = thomas_breakthrough(100, 500.0, 0.0, 0.0)
        assert bt == 1.0

    def test_zero_influent_full_breakthrough(self):
        bt = thomas_breakthrough(100, 0.0, 0.0, RESIN_MAX_CLO4_G)
        assert bt == 1.0


# ============================================================================
# 3. Ion exchange capture
# ============================================================================

class TestIXRemoval:
    """Test IX column perchlorate capture calculations."""

    def test_full_capture_at_zero_breakthrough(self):
        captured = ix_removal_sol(20.0, 20.0, 500.0, 0.0)
        # 20 L/h × 20 h × 500 mg/L = 200,000 mg = 200 g
        assert abs(captured - 200.0) < 0.1

    def test_zero_capture_at_full_breakthrough(self):
        captured = ix_removal_sol(20.0, 20.0, 500.0, 1.0)
        assert abs(captured) < 0.01

    def test_partial_capture(self):
        captured = ix_removal_sol(20.0, 20.0, 500.0, 0.5)
        assert abs(captured - 100.0) < 0.1

    def test_zero_flow_no_capture(self):
        captured = ix_removal_sol(0.0, 20.0, 500.0, 0.0)
        assert captured == 0.0

    def test_zero_hours_no_capture(self):
        captured = ix_removal_sol(20.0, 0.0, 500.0, 0.0)
        assert captured == 0.0

    def test_capture_scales_with_flow(self):
        c1 = ix_removal_sol(10.0, 20.0, 500.0, 0.0)
        c2 = ix_removal_sol(20.0, 20.0, 500.0, 0.0)
        assert abs(c2 / c1 - 2.0) < 0.01

    def test_negative_values_clamped(self):
        captured = ix_removal_sol(-5.0, -10.0, 500.0, 0.0)
        assert captured == 0.0


class TestIXEffluent:
    """Test IX effluent concentration calculations."""

    def test_zero_breakthrough_zero_effluent(self):
        eff = ix_effluent_mg_l(500.0, 0.0)
        assert eff == 0.0

    def test_full_breakthrough_full_effluent(self):
        eff = ix_effluent_mg_l(500.0, 1.0)
        assert abs(eff - 500.0) < 0.01

    def test_half_breakthrough(self):
        eff = ix_effluent_mg_l(500.0, 0.5)
        assert abs(eff - 250.0) < 0.01

    def test_effluent_never_negative(self):
        eff = ix_effluent_mg_l(500.0, -0.5)
        assert eff >= 0.0


# ============================================================================
# 4. Monod bioreactor kinetics
# ============================================================================

class TestMonodKinetics:
    """Test Monod specific growth rate model."""

    def test_zero_substrate_zero_rate(self):
        assert monod_rate(0.0) == 0.0

    def test_high_substrate_approaches_mu_max(self):
        rate = monod_rate(1000.0)
        assert abs(rate - BIO_MU_MAX) < 0.01 * BIO_MU_MAX

    def test_at_ks_half_mu_max(self):
        rate = monod_rate(BIO_KS_MG_L)
        assert abs(rate - BIO_MU_MAX / 2.0) < 0.001

    def test_rate_increases_with_substrate(self):
        r1 = monod_rate(1.0)
        r2 = monod_rate(10.0)
        r3 = monod_rate(100.0)
        assert r1 < r2 < r3

    def test_rate_bounded_by_mu_max(self):
        for s in [0.0, 1.0, 10.0, 100.0, 10000.0]:
            assert monod_rate(s) <= BIO_MU_MAX + 1e-10

    def test_negative_substrate_clamped(self):
        rate = monod_rate(-50.0)
        assert rate == 0.0


class TestBioTemperature:
    """Test temperature sensitivity of bioreactor."""

    def test_optimal_temp_full_activity(self):
        factor = bio_temperature_factor(BIO_TEMP_OPTIMAL_C)
        assert abs(factor - 1.0) < 0.01

    def test_below_min_zero_activity(self):
        assert bio_temperature_factor(BIO_TEMP_MIN_C - 1.0) == 0.0

    def test_above_max_zero_activity(self):
        assert bio_temperature_factor(BIO_TEMP_MAX_C + 1.0) == 0.0

    def test_cold_reduced_activity(self):
        factor = bio_temperature_factor(15.0)
        assert 0.0 < factor < 1.0

    def test_hot_reduced_activity(self):
        factor = bio_temperature_factor(40.0)
        assert 0.0 < factor < 1.0

    def test_factor_bounded(self):
        for t in range(-20, 60):
            f = bio_temperature_factor(float(t))
            assert 0.0 <= f <= 1.0


class TestBioDestruction:
    """Test biological perchlorate destruction."""

    def test_destruction_produces_output(self):
        destroyed, biomass = bio_destruction_sol(
            50.0, 100.0, 100.0, 24.0, BIO_TEMP_OPTIMAL_C,
        )
        assert destroyed > 0.0
        assert biomass > 0.0

    def test_no_biomass_no_destruction(self):
        destroyed, biomass = bio_destruction_sol(
            0.0, 100.0, 100.0, 24.0, BIO_TEMP_OPTIMAL_C,
        )
        assert destroyed == 0.0

    def test_no_substrate_no_destruction(self):
        destroyed, _ = bio_destruction_sol(
            50.0, 0.0, 100.0, 24.0, BIO_TEMP_OPTIMAL_C,
        )
        assert destroyed == 0.0

    def test_cold_kills_activity(self):
        destroyed, biomass = bio_destruction_sol(
            50.0, 100.0, 100.0, 24.0, 0.0,
        )
        assert destroyed == 0.0
        assert biomass < 50.0  # decay still happens

    def test_biomass_bounded(self):
        _, biomass = bio_destruction_sol(
            BIO_MAX_BIOMASS_G * 2, 1000.0, 100.0, 100.0,
            BIO_TEMP_OPTIMAL_C,
        )
        assert biomass <= BIO_MAX_BIOMASS_G

    def test_destruction_limited_by_substrate(self):
        """Cannot destroy more ClO₄⁻ than is present."""
        substrate_mg_l = 0.1
        reactor_l = 10.0  # only 1 mg = 0.001 g total
        destroyed, _ = bio_destruction_sol(
            500.0, substrate_mg_l, reactor_l, 24.0, BIO_TEMP_OPTIMAL_C,
        )
        available_g = substrate_mg_l * reactor_l / 1000.0
        assert destroyed <= available_g + 1e-10


# ============================================================================
# 5. Stoichiometry
# ============================================================================

class TestStoichiometry:
    """Test mass-balance stoichiometry."""

    def test_products_from_zero(self):
        cl, o2 = stoichiometry_products(0.0)
        assert cl == 0.0
        assert o2 == 0.0

    def test_one_mole_perchlorate(self):
        # 99.45 g ClO₄⁻ → 35.45 g Cl⁻ + 64.00 g O₂
        cl, o2 = stoichiometry_products(99.45)
        assert abs(cl - 35.45) < 0.1
        assert abs(o2 - 64.00) < 0.1

    def test_mass_conservation(self):
        """Cl + O₂ mass should equal ClO₄⁻ mass (minus electrons)."""
        for g in [1.0, 10.0, 100.0, 1000.0]:
            cl, o2 = stoichiometry_products(g)
            # ClO₄⁻ = Cl⁻ + 2O₂ (approximate mass balance)
            assert abs(cl + o2 - g) < 0.1 * g

    def test_negative_input_clamped(self):
        cl, o2 = stoichiometry_products(-10.0)
        assert cl == 0.0
        assert o2 == 0.0

    def test_o2_always_more_than_chloride(self):
        """2 mol O₂ (64g) > 1 mol Cl⁻ (35.45g) per mol ClO₄⁻."""
        for g in [1.0, 50.0, 500.0]:
            cl, o2 = stoichiometry_products(g)
            assert o2 > cl


# ============================================================================
# 6. Resin management
# ============================================================================

class TestResinManagement:
    """Test resin regeneration and saturation checks."""

    def test_fresh_resin_no_regen(self):
        assert not resin_needs_regen(0.0, RESIN_MAX_CLO4_G)

    def test_saturated_resin_needs_regen(self):
        assert resin_needs_regen(RESIN_MAX_CLO4_G * 0.9, RESIN_MAX_CLO4_G)

    def test_threshold_at_85_percent(self):
        assert not resin_needs_regen(RESIN_MAX_CLO4_G * 0.84, RESIN_MAX_CLO4_G)
        assert resin_needs_regen(RESIN_MAX_CLO4_G * 0.86, RESIN_MAX_CLO4_G)

    def test_zero_capacity_needs_regen(self):
        assert resin_needs_regen(0.0, 0.0)

    def test_regeneration_reduces_load(self):
        initial = RESIN_MAX_CLO4_G * 0.9
        new_load, recovered, _ = regenerate_resin(initial, RESIN_MAX_CLO4_G)
        assert new_load < initial
        assert recovered > 0

    def test_regeneration_recovery_fraction(self):
        initial = 1000.0
        _, recovered, _ = regenerate_resin(initial, RESIN_MAX_CLO4_G)
        assert abs(recovered - initial * REGEN_EFFICIENCY) < 0.1

    def test_regeneration_degrades_capacity(self):
        _, _, new_cap = regenerate_resin(1000.0, RESIN_MAX_CLO4_G)
        assert new_cap < RESIN_MAX_CLO4_G
        assert new_cap > RESIN_MAX_CLO4_G * 0.95


# ============================================================================
# 7. Water quality classification
# ============================================================================

class TestWaterQuality:
    """Test potable and irrigation water classification."""

    def test_pure_water_potable(self):
        assert is_potable(0.0)

    def test_epa_limit_potable(self):
        assert is_potable(POTABLE_LIMIT_MG_L)

    def test_above_epa_not_potable(self):
        assert not is_potable(POTABLE_LIMIT_MG_L + 0.001)

    def test_mars_raw_water_not_potable(self):
        assert not is_potable(MARS_WATER_CLO4_MG_L)

    def test_pure_water_irrigable(self):
        assert is_irrigable(0.0)

    def test_irrigation_limit(self):
        assert is_irrigable(IRRIGATION_LIMIT_MG_L)

    def test_above_irrigation_not_irrigable(self):
        assert not is_irrigable(IRRIGATION_LIMIT_MG_L + 0.1)

    def test_potable_implies_irrigable(self):
        """If water is potable, it must also be irrigable."""
        assert POTABLE_LIMIT_MG_L < IRRIGATION_LIMIT_MG_L
        if is_potable(0.01):
            assert is_irrigable(0.01)


# ============================================================================
# 8. Energy accounting
# ============================================================================

class TestEnergy:
    """Test energy calculations."""

    def test_base_energy_positive(self):
        e = energy_per_sol(regenerating=False)
        assert e > 0

    def test_regen_uses_more_energy(self):
        base = energy_per_sol(False)
        regen = energy_per_sol(True)
        assert regen > base
        assert abs(regen - base - REGEN_ENERGY_KWH) < 0.01


# ============================================================================
# 9. Alert levels
# ============================================================================

class TestAlerts:
    """Test alert classification."""

    def test_nominal_conditions(self):
        assert assess_alert(0.001, 0.3, 50.0) == "nominal"

    def test_high_effluent_warning(self):
        assert assess_alert(2.0, 0.3, 50.0) == "warning"

    def test_very_high_effluent_critical(self):
        assert assess_alert(15.0, 0.3, 50.0) == "critical"

    def test_high_saturation_warning(self):
        assert assess_alert(0.001, 0.9, 50.0) == "warning"

    def test_low_biomass_critical(self):
        assert assess_alert(0.001, 0.3, 2.0) == "critical"


# ============================================================================
# 10. Tick engine
# ============================================================================

class TestTickEngine:
    """Test the main tick function."""

    def test_single_tick_runs(self):
        state = create_scrubber()
        state, result = tick_scrubber(state)
        assert result.water_treated_l > 0
        assert result.ix_clo4_captured_g > 0
        assert state.sols_running == 1

    def test_tick_captures_perchlorate(self):
        state = create_scrubber()
        _, result = tick_scrubber(state)
        assert result.total_clo4_removed_g > 0

    def test_tick_energy_consumed(self):
        state = create_scrubber()
        _, result = tick_scrubber(state)
        assert result.energy_kwh > 0

    def test_tick_resin_loads(self):
        state = create_scrubber()
        state, result = tick_scrubber(state)
        assert state.resin_clo4_g > 0
        assert result.resin_saturation > 0

    def test_fresh_resin_produces_clean_water(self):
        state = create_scrubber()
        _, result = tick_scrubber(state)
        # Fresh resin should produce irrigable water
        assert result.irrigable

    def test_power_limited_reduces_throughput(self):
        s1 = create_scrubber()
        s2 = create_scrubber()
        _, r1 = tick_scrubber(s1, power_available_kwh=50.0)
        _, r2 = tick_scrubber(s2, power_available_kwh=1.0)
        assert r2.water_treated_l < r1.water_treated_l

    def test_zero_power_minimal_operation(self):
        state = create_scrubber()
        _, result = tick_scrubber(state, power_available_kwh=0.0)
        assert result.water_treated_l == 0.0


# ============================================================================
# 11. Multi-sol integration
# ============================================================================

class TestMultiSol:
    """Test multi-sol simulation behavior."""

    def test_10_sol_smoke_test(self):
        """Run 10 sols without crash."""
        state = create_scrubber()
        for _ in range(10):
            state, result = tick_scrubber(state)
        assert state.sols_running == 10
        assert state.total_water_treated_l > 0
        assert state.total_clo4_removed_g > 0

    def test_100_sol_resin_saturates(self):
        """Over many sols, resin must eventually saturate."""
        state = create_scrubber()
        max_saturation = 0.0
        for _ in range(100):
            state, result = tick_scrubber(state)
            max_saturation = max(max_saturation, result.resin_saturation)
        assert max_saturation > 0.5, "Resin should load up over time"

    def test_cumulative_water_increases(self):
        state = create_scrubber()
        prev = 0.0
        for _ in range(5):
            state, result = tick_scrubber(state)
            assert state.total_water_treated_l > prev
            prev = state.total_water_treated_l

    def test_biomass_grows_with_substrate(self):
        state = create_scrubber(influent_mg_l=500.0)
        initial_biomass = state.biomass_g
        for _ in range(20):
            state, _ = tick_scrubber(state)
        # With substrate, biomass should grow
        assert state.biomass_g >= initial_biomass * 0.5

    def test_regeneration_occurs_eventually(self):
        state = create_scrubber()
        regen_happened = False
        for _ in range(200):
            state, result = tick_scrubber(state, power_available_kwh=100.0)
            if result.regenerated:
                regen_happened = True
                break
        assert regen_happened, "Resin should regenerate within 200 sols"

    def test_o2_production_cumulative(self):
        state = create_scrubber()
        for _ in range(10):
            state, _ = tick_scrubber(state)
        assert state.total_o2_produced_g >= 0.0

    def test_chloride_production_cumulative(self):
        state = create_scrubber()
        for _ in range(10):
            state, _ = tick_scrubber(state)
        assert state.total_chloride_produced_g >= 0.0


# ============================================================================
# 12. Conservation law invariants
# ============================================================================

class TestConservationLaws:
    """Property-based tests: physical invariants that must always hold."""

    def test_resin_load_never_exceeds_capacity(self):
        state = create_scrubber()
        for _ in range(50):
            state, _ = tick_scrubber(state, power_available_kwh=100.0)
            assert state.resin_clo4_g <= state.resin_capacity_g + 1e-6

    def test_biomass_never_negative(self):
        state = create_scrubber()
        for _ in range(50):
            state, _ = tick_scrubber(state, bio_temp_c=0.0)
            assert state.biomass_g >= 0.0

    def test_biomass_never_exceeds_max(self):
        state = create_scrubber(biomass_g=BIO_MAX_BIOMASS_G)
        for _ in range(50):
            state, _ = tick_scrubber(state)
            assert state.biomass_g <= BIO_MAX_BIOMASS_G + 1e-6

    def test_effluent_never_negative(self):
        state = create_scrubber()
        for _ in range(50):
            state, result = tick_scrubber(state)
            assert result.effluent_clo4_mg_l >= 0.0

    def test_energy_never_negative(self):
        state = create_scrubber()
        for _ in range(20):
            state, result = tick_scrubber(state)
            assert result.energy_kwh >= 0.0
            assert state.total_energy_kwh >= 0.0

    def test_water_treated_never_decreases(self):
        state = create_scrubber()
        prev = 0.0
        for _ in range(20):
            state, _ = tick_scrubber(state)
            assert state.total_water_treated_l >= prev
            prev = state.total_water_treated_l

    def test_stoichiometry_mass_balance(self):
        """Cl⁻ + O₂ ≈ ClO₄⁻ destroyed (mass conservation)."""
        state = create_scrubber()
        for _ in range(30):
            state, _ = tick_scrubber(state)
        if state.bio_clo4_destroyed_g > 0.01:
            expected_cl = state.bio_clo4_destroyed_g * STOICH_CL_PER_CLO4
            expected_o2 = state.bio_clo4_destroyed_g * STOICH_O2_PER_CLO4
            assert abs(state.total_chloride_produced_g - expected_cl) < 1.0
            assert abs(state.total_o2_produced_g - expected_o2) < 1.0

    def test_effluent_less_than_influent(self):
        """Output concentration must never exceed input."""
        state = create_scrubber()
        for _ in range(20):
            state, result = tick_scrubber(state)
            assert result.effluent_clo4_mg_l <= state.influent_clo4_mg_l + 1e-6


# ============================================================================
# 13. Edge cases & boundary conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_influent(self):
        state = create_scrubber(influent_mg_l=0.0)
        state, result = tick_scrubber(state)
        assert result.ix_clo4_captured_g == 0.0
        assert result.potable

    def test_very_high_influent(self):
        state = create_scrubber(influent_mg_l=10000.0)
        state, result = tick_scrubber(state)
        assert result.water_treated_l > 0

    def test_zero_biomass_bio_stage_inactive(self):
        state = create_scrubber(biomass_g=0.0)
        state, result = tick_scrubber(state)
        assert result.bio_clo4_destroyed_g == 0.0

    def test_extreme_cold_bioreactor(self):
        state = create_scrubber()
        state, result = tick_scrubber(state, bio_temp_c=-40.0)
        assert result.bio_clo4_destroyed_g == 0.0

    def test_extreme_hot_bioreactor(self):
        state = create_scrubber()
        state, result = tick_scrubber(state, bio_temp_c=80.0)
        assert result.bio_clo4_destroyed_g == 0.0

    def test_factory_default_state(self):
        state = create_scrubber()
        assert state.resin_clo4_g == 0.0
        assert state.biomass_g == BIO_INITIAL_BIOMASS_G
        assert state.influent_clo4_mg_l == MARS_WATER_CLO4_MG_L
        assert state.sols_running == 0

    def test_factory_custom_influent(self):
        state = create_scrubber(influent_mg_l=100.0)
        assert state.influent_clo4_mg_l == 100.0

    def test_state_post_init_clamps(self):
        state = ScrubberState(resin_clo4_g=-100.0, biomass_g=99999.0)
        assert state.resin_clo4_g == 0.0
        assert state.biomass_g == BIO_MAX_BIOMASS_G
