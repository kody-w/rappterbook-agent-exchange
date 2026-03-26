"""Tests for perchlorate_scrubber.py -- Mars Perchlorate Remediation System.

91 tests covering:
  - Physical constants & stoichiometry
  - Perchlorate mass calculations
  - Thermal decomposition physics
  - Bioremediation culture dynamics
  - Ion exchange degradation
  - Mass conservation invariants
  - Oxygen recovery accounting
  - Full tick lifecycle
  - Multi-sol integration
  - Edge cases and boundary conditions
  - Power-limited operation
  - Resource exhaustion scenarios
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from perchlorate_scrubber import (
    BIO_BASE_RATE_KG_SOL,
    BIO_CULTURE_DECAY_PER_SOL,
    BIO_CULTURE_GROWTH_PER_SOL,
    BIO_ELECTRON_DONOR_KG_PER_KG,
    BIO_KWH_PER_KG,
    BIO_MAX_TEMP_C,
    BIO_MIN_TEMP_C,
    BIO_OPTIMAL_TEMP_C,
    CL_PER_PERCHLORATE_KG,
    CHLORIDE_MOLAR_MASS,
    IX_KWH_PER_KG,
    IX_REMOVAL_EFFICIENCY,
    IX_RESIN_CYCLES,
    IX_RESIN_DEGRADE_PER_CYCLE,
    IX_THROUGHPUT_KG_SOL,
    IX_WATER_L_PER_KG,
    O2_MOLAR_MASS,
    O2_PER_PERCHLORATE_KG,
    PERCHLORATE_MARS_HIGH,
    PERCHLORATE_MARS_LOW,
    PERCHLORATE_MARS_MEAN,
    PERCHLORATE_MOLAR_MASS,
    REGOLITH_DENSITY_KG_M3,
    SAFE_AGRICULTURE_LIMIT,
    SAFE_DRINKING_WATER_UG_L,
    THERMAL_KWH_PER_KG,
    THERMAL_MAX_TEMP_C,
    THERMAL_MIN_TEMP_C,
    THERMAL_OPTIMAL_TEMP_C,
    THERMAL_REMOVAL_EFFICIENCY,
    THERMAL_THROUGHPUT_KG_SOL,
    ScrubberResult,
    ScrubberState,
    bio_culture_factor,
    bio_process_sol,
    bio_update_health,
    chloride_from_perchlorate_kg,
    create_scrubber,
    is_safe_for_agriculture,
    ix_current_efficiency,
    ix_process_sol,
    mass_balance_check,
    o2_from_perchlorate_kg,
    perchlorate_in_regolith_kg,
    residual_perchlorate_wt,
    scrubber_power_kwh,
    thermal_efficiency,
    thermal_process_sol,
    tick_scrubber,
)


# ============================================================================
# 1. Physical constants and stoichiometry
# ============================================================================

class TestPhysicalConstants:
    """Validate physical constants against known values."""

    def test_perchlorate_molar_mass(self):
        # Cl=35.45, O=16.00*4=64.00 -> 99.45
        assert abs(PERCHLORATE_MOLAR_MASS - 99.45) < 0.1

    def test_chloride_molar_mass(self):
        assert abs(CHLORIDE_MOLAR_MASS - 35.45) < 0.1

    def test_o2_molar_mass(self):
        assert abs(O2_MOLAR_MASS - 32.0) < 0.1

    def test_o2_per_perchlorate_stoichiometry(self):
        # ClO4- -> Cl- + 2O2: ratio = 2*32/99.45 ~ 0.6435
        expected = (2 * 32.0) / 99.45
        assert abs(O2_PER_PERCHLORATE_KG - expected) < 0.001

    def test_cl_per_perchlorate_stoichiometry(self):
        expected = 35.45 / 99.45
        assert abs(CL_PER_PERCHLORATE_KG - expected) < 0.001

    def test_stoichiometry_sums_to_one(self):
        # O2 fraction + Cl fraction should equal 1 (mass balance)
        total = O2_PER_PERCHLORATE_KG + CL_PER_PERCHLORATE_KG
        assert abs(total - 1.0) < 0.001

    def test_mars_perchlorate_range(self):
        assert 0.0 < PERCHLORATE_MARS_LOW < PERCHLORATE_MARS_MEAN
        assert PERCHLORATE_MARS_MEAN < PERCHLORATE_MARS_HIGH
        assert PERCHLORATE_MARS_HIGH <= 0.02  # max ~2%

    def test_safe_limits_are_strict(self):
        assert SAFE_AGRICULTURE_LIMIT < PERCHLORATE_MARS_LOW
        assert SAFE_DRINKING_WATER_UG_L > 0

    def test_regolith_density(self):
        assert 1000 < REGOLITH_DENSITY_KG_M3 < 2000


# ============================================================================
# 2. Perchlorate mass calculations
# ============================================================================

class TestPerchlorateMass:
    """Test perchlorate content functions."""

    def test_perchlorate_in_regolith_typical(self):
        result = perchlorate_in_regolith_kg(1000.0, 0.007)
        assert abs(result - 7.0) < 0.001

    def test_perchlorate_in_regolith_zero_mass(self):
        assert perchlorate_in_regolith_kg(0.0) == 0.0

    def test_perchlorate_in_regolith_negative_clamped(self):
        result = perchlorate_in_regolith_kg(100.0, -0.5)
        assert result == 0.0

    def test_perchlorate_concentration_capped(self):
        # Concentration > 1.0 should be clamped
        result = perchlorate_in_regolith_kg(100.0, 1.5)
        assert result == 100.0  # capped at 100%

    def test_o2_from_perchlorate(self):
        result = o2_from_perchlorate_kg(1.0)
        expected = O2_PER_PERCHLORATE_KG
        assert abs(result - expected) < 0.001

    def test_o2_from_perchlorate_zero(self):
        assert o2_from_perchlorate_kg(0.0) == 0.0

    def test_o2_from_perchlorate_negative(self):
        assert o2_from_perchlorate_kg(-5.0) == 0.0

    def test_chloride_from_perchlorate(self):
        result = chloride_from_perchlorate_kg(1.0)
        expected = CL_PER_PERCHLORATE_KG
        assert abs(result - expected) < 0.001

    def test_chloride_from_perchlorate_zero(self):
        assert chloride_from_perchlorate_kg(0.0) == 0.0

    def test_o2_plus_cl_equals_perchlorate(self):
        """Conservation: O2 + Cl- mass must equal original perchlorate mass."""
        perc = 10.0
        o2 = o2_from_perchlorate_kg(perc)
        cl = chloride_from_perchlorate_kg(perc)
        assert abs((o2 + cl) - perc) < 0.01


# ============================================================================
# 3. Residual concentration and safety
# ============================================================================

class TestResidualAndSafety:
    """Test residual perchlorate and safety checks."""

    def test_residual_full_removal(self):
        result = residual_perchlorate_wt(0.007, 1.0)
        assert result == 0.0

    def test_residual_no_removal(self):
        result = residual_perchlorate_wt(0.007, 0.0)
        assert abs(result - 0.007) < 1e-9

    def test_residual_partial_removal(self):
        result = residual_perchlorate_wt(0.007, 0.995)
        assert result < 0.007
        assert result > 0.0

    def test_thermal_makes_soil_safe(self):
        # 99.5% removal of 0.7% -> 0.0035% = 0.000035 wt
        residual = residual_perchlorate_wt(0.007, THERMAL_REMOVAL_EFFICIENCY)
        assert is_safe_for_agriculture(residual)

    def test_safe_below_limit(self):
        assert is_safe_for_agriculture(0.00005)

    def test_unsafe_above_limit(self):
        assert not is_safe_for_agriculture(0.001)

    def test_safe_at_exactly_limit(self):
        assert is_safe_for_agriculture(SAFE_AGRICULTURE_LIMIT)


# ============================================================================
# 4. Thermal decomposition
# ============================================================================

class TestThermalDecomposition:
    """Test thermal kiln pathway."""

    def test_efficiency_below_threshold(self):
        assert thermal_efficiency(300.0) == 0.0
        assert thermal_efficiency(399.9) == 0.0

    def test_efficiency_at_threshold(self):
        assert thermal_efficiency(400.0) == 0.0

    def test_efficiency_midpoint(self):
        eff = thermal_efficiency(500.0)
        expected = 0.5 * THERMAL_REMOVAL_EFFICIENCY
        assert abs(eff - expected) < 0.001

    def test_efficiency_at_optimal(self):
        eff = thermal_efficiency(600.0)
        assert abs(eff - THERMAL_REMOVAL_EFFICIENCY) < 0.001

    def test_efficiency_above_optimal(self):
        eff = thermal_efficiency(700.0)
        assert abs(eff - THERMAL_REMOVAL_EFFICIENCY) < 0.001

    def test_efficiency_monotonic(self):
        temps = [400, 450, 500, 550, 600, 650, 700]
        effs = [thermal_efficiency(t) for t in temps]
        for i in range(len(effs) - 1):
            assert effs[i] <= effs[i + 1]

    def test_thermal_process_normal(self):
        result = thermal_process_sol(1000.0, 1000.0, 600.0, 0.007)
        assert result["processed_kg"] > 0
        assert result["clean_kg"] > 0
        assert result["perchlorate_removed_kg"] > 0
        assert result["o2_recovered_kg"] > 0
        assert result["energy_used_kwh"] > 0

    def test_thermal_process_cold_kiln(self):
        result = thermal_process_sol(1000.0, 1000.0, 200.0)
        assert result["processed_kg"] == 0.0

    def test_thermal_process_no_power(self):
        result = thermal_process_sol(1000.0, 0.0, 600.0)
        assert result["processed_kg"] == 0.0

    def test_thermal_process_no_material(self):
        result = thermal_process_sol(0.0, 1000.0, 600.0)
        assert result["processed_kg"] == 0.0

    def test_thermal_power_limited(self):
        # Only 7 kWh -> can process 7/3.5 = 2 kg
        result = thermal_process_sol(1000.0, 7.0, 600.0)
        assert abs(result["processed_kg"] - 2.0) < 0.01

    def test_thermal_throughput_capped(self):
        result = thermal_process_sol(10000.0, 100000.0, 600.0)
        assert result["processed_kg"] <= THERMAL_THROUGHPUT_KG_SOL

    def test_thermal_mass_conservation(self):
        result = thermal_process_sol(500.0, 5000.0, 600.0, 0.007)
        proc = result["processed_kg"]
        clean = result["clean_kg"]
        perc = result["perchlorate_removed_kg"]
        assert abs((clean + perc) - proc) < 0.01

    def test_thermal_o2_matches_stoichiometry(self):
        result = thermal_process_sol(500.0, 5000.0, 600.0, 0.007)
        perc = result["perchlorate_removed_kg"]
        o2 = result["o2_recovered_kg"]
        expected_o2 = perc * O2_PER_PERCHLORATE_KG
        assert abs(o2 - expected_o2) < 0.001


# ============================================================================
# 5. Bioremediation
# ============================================================================

class TestBioremediation:
    """Test bioreactor pathway."""

    def test_culture_factor_optimal(self):
        factor = bio_culture_factor(1.0, 30.0)
        assert abs(factor - 1.0) < 0.01

    def test_culture_factor_cold(self):
        factor = bio_culture_factor(1.0, 5.0)
        assert factor == 0.0

    def test_culture_factor_hot(self):
        factor = bio_culture_factor(1.0, 50.0)
        assert factor == 0.0

    def test_culture_factor_dead_culture(self):
        factor = bio_culture_factor(0.0, 30.0)
        assert factor == 0.0

    def test_culture_factor_suboptimal_temp(self):
        factor = bio_culture_factor(1.0, 20.0)
        assert 0.0 < factor < 1.0

    def test_culture_factor_bounded(self):
        for h in [0.0, 0.5, 1.0]:
            for t in [10, 20, 30, 40]:
                f = bio_culture_factor(h, t)
                assert 0.0 <= f <= 1.0

    def test_health_growth_good_conditions(self):
        new = bio_update_health(0.5, 30.0, True)
        assert new > 0.5

    def test_health_decay_bad_conditions(self):
        new = bio_update_health(0.5, 5.0, True)
        assert new < 0.5

    def test_health_decay_no_donor(self):
        new = bio_update_health(0.5, 30.0, False)
        assert new < 0.5

    def test_health_clamped_max(self):
        new = bio_update_health(0.99, 30.0, True)
        assert new <= 1.0

    def test_health_clamped_min(self):
        new = bio_update_health(0.01, 5.0, False)
        assert new >= 0.0

    def test_bio_process_normal(self):
        result = bio_process_sol(500.0, 100.0, 0.8, 30.0, 50.0, 0.007)
        assert result["processed_kg"] > 0
        assert result["perchlorate_removed_kg"] > 0
        assert result["donor_consumed_kg"] > 0

    def test_bio_process_no_material(self):
        result = bio_process_sol(0.0, 100.0, 0.8, 30.0, 50.0)
        assert result["processed_kg"] == 0.0

    def test_bio_process_no_power(self):
        result = bio_process_sol(500.0, 0.0, 0.8, 30.0, 50.0)
        assert result["processed_kg"] == 0.0

    def test_bio_process_cold_reactor(self):
        result = bio_process_sol(500.0, 100.0, 0.8, 5.0, 50.0)
        assert result["processed_kg"] == 0.0

    def test_bio_process_no_donor(self):
        result = bio_process_sol(500.0, 100.0, 0.8, 30.0, 0.0)
        assert result["processed_kg"] == 0.0

    def test_bio_donor_consumption(self):
        result = bio_process_sol(100.0, 100.0, 1.0, 30.0, 50.0, 0.007)
        proc = result["processed_kg"]
        donor = result["donor_consumed_kg"]
        expected = proc * BIO_ELECTRON_DONOR_KG_PER_KG
        assert abs(donor - expected) < 0.01


# ============================================================================
# 6. Ion exchange
# ============================================================================

class TestIonExchange:
    """Test IX wash pathway."""

    def test_ix_fresh_efficiency(self):
        eff = ix_current_efficiency(0)
        assert abs(eff - IX_REMOVAL_EFFICIENCY) < 0.001

    def test_ix_degradation(self):
        eff = ix_current_efficiency(100)
        expected = IX_REMOVAL_EFFICIENCY - 100 * IX_RESIN_DEGRADE_PER_CYCLE
        assert abs(eff - expected) < 0.001

    def test_ix_fully_degraded(self):
        eff = ix_current_efficiency(IX_RESIN_CYCLES)
        assert eff >= 0.0

    def test_ix_over_degraded_clamped(self):
        eff = ix_current_efficiency(IX_RESIN_CYCLES + 1000)
        assert eff == 0.0

    def test_ix_process_normal(self):
        result = ix_process_sol(500.0, 200.0, 500.0, 0, 0.007)
        assert result["processed_kg"] > 0
        assert result["perchlorate_removed_kg"] > 0
        assert result["water_used_l"] > 0

    def test_ix_process_no_water(self):
        result = ix_process_sol(500.0, 200.0, 0.0, 0)
        assert result["processed_kg"] == 0.0

    def test_ix_process_no_power(self):
        result = ix_process_sol(500.0, 0.0, 500.0, 0)
        assert result["processed_kg"] == 0.0

    def test_ix_process_worn_resin(self):
        # Resin fully degrades at ceil(IX_REMOVAL_EFFICIENCY / IX_RESIN_DEGRADE_PER_CYCLE)
        dead_cycles = int(IX_REMOVAL_EFFICIENCY / IX_RESIN_DEGRADE_PER_CYCLE) + 1
        result = ix_process_sol(500.0, 200.0, 500.0, dead_cycles)
        assert result["processed_kg"] == 0.0

    def test_ix_water_consumption(self):
        result = ix_process_sol(100.0, 500.0, 1000.0, 0, 0.007)
        proc = result["processed_kg"]
        water = result["water_used_l"]
        expected = proc * IX_WATER_L_PER_KG
        assert abs(water - expected) < 0.01

    def test_ix_cycle_incremented(self):
        result = ix_process_sol(100.0, 500.0, 1000.0, 5)
        assert result["cycles_used"] == 6

    def test_ix_throughput_capped(self):
        result = ix_process_sol(10000.0, 100000.0, 100000.0, 0)
        assert result["processed_kg"] <= IX_THROUGHPUT_KG_SOL


# ============================================================================
# 7. Mass balance check
# ============================================================================

class TestMassBalance:
    """Test mass conservation helper."""

    def test_perfect_balance(self):
        err = mass_balance_check(100.0, 0.007, 99.3, 0.7)
        assert err < 0.01

    def test_imbalanced(self):
        err = mass_balance_check(100.0, 0.007, 50.0, 0.7)
        assert err > 1.0


# ============================================================================
# 8. Full tick lifecycle
# ============================================================================

class TestTickScrubber:
    """Test the integrated tick function."""

    def test_tick_basic(self):
        state = create_scrubber()
        state, result = tick_scrubber(state, regolith_input_kg=500.0)
        assert result.total_processed_kg > 0
        assert result.clean_kg_produced > 0
        assert result.energy_used_kwh > 0
        assert state.sols_running == 1

    def test_tick_no_input_no_stock(self):
        state = create_scrubber()
        state, result = tick_scrubber(state, regolith_input_kg=0.0)
        assert result.total_processed_kg == 0.0

    def test_tick_accumulates_clean(self):
        state = create_scrubber()
        state, _ = tick_scrubber(state, regolith_input_kg=500.0)
        clean1 = state.clean_kg
        state, _ = tick_scrubber(state, regolith_input_kg=500.0)
        assert state.clean_kg > clean1

    def test_tick_o2_recovery(self):
        state = create_scrubber()
        state, result = tick_scrubber(state, regolith_input_kg=1000.0)
        assert result.o2_recovered_kg > 0
        assert state.total_o2_recovered_kg > 0

    def test_tick_mass_conservation(self):
        """Total input = clean + perchlorate_removed + remaining contaminated."""
        state = create_scrubber()
        input_kg = 1000.0
        state, result = tick_scrubber(state, regolith_input_kg=input_kg,
                                       concentration=0.007)
        total_out = result.clean_kg_produced + result.perchlorate_removed_kg + state.contaminated_kg
        assert abs(total_out - input_kg) < 0.1

    def test_tick_energy_positive(self):
        state = create_scrubber()
        state, result = tick_scrubber(state, regolith_input_kg=500.0)
        assert result.energy_used_kwh >= 0

    def test_tick_custom_allocation(self):
        state = create_scrubber()
        alloc = {"thermal": 1.0, "bio": 0.0, "ix": 0.0}
        state, result = tick_scrubber(state, regolith_input_kg=500.0,
                                       allocation=alloc)
        assert result.thermal_processed_kg > 0
        assert result.bio_processed_kg == 0.0
        assert result.ix_processed_kg == 0.0

    def test_tick_bio_only(self):
        state = create_scrubber()
        state.bio_culture_health = 1.0
        alloc = {"thermal": 0.0, "bio": 1.0, "ix": 0.0}
        state, result = tick_scrubber(state, regolith_input_kg=200.0,
                                       power_budget_kwh=100.0,
                                       allocation=alloc)
        assert result.bio_processed_kg > 0
        assert result.thermal_processed_kg == 0.0

    def test_tick_ix_only(self):
        state = create_scrubber()
        alloc = {"thermal": 0.0, "bio": 0.0, "ix": 1.0}
        state, result = tick_scrubber(state, regolith_input_kg=200.0,
                                       water_available_l=1000.0,
                                       allocation=alloc)
        assert result.ix_processed_kg > 0
        assert result.thermal_processed_kg == 0.0

    def test_tick_sols_increment(self):
        state = create_scrubber()
        for i in range(5):
            state, _ = tick_scrubber(state, regolith_input_kg=100.0)
        assert state.sols_running == 5

    def test_tick_warning_backlog(self):
        state = create_scrubber()
        state.contaminated_kg = 6000.0
        state, result = tick_scrubber(state, power_budget_kwh=1.0)
        assert result.alert == "warning"

    def test_tick_warning_sick_culture(self):
        state = create_scrubber()
        state.bio_culture_health = 0.1
        state, result = tick_scrubber(state, regolith_input_kg=100.0)
        assert result.alert == "warning"

    def test_tick_zero_power(self):
        state = create_scrubber()
        state, result = tick_scrubber(state, regolith_input_kg=500.0,
                                       power_budget_kwh=0.0)
        assert result.total_processed_kg == 0.0
        assert state.contaminated_kg == 500.0


# ============================================================================
# 9. Multi-sol integration (smoke test)
# ============================================================================

class TestMultiSol:
    """Run the scrubber for many sols -- the colony endures."""

    def test_30_sol_run(self):
        """Run 30 sols feeding 200 kg/sol.  Must not crash."""
        state = create_scrubber()
        total_input = 0.0
        for sol in range(30):
            state, result = tick_scrubber(
                state,
                regolith_input_kg=200.0,
                power_budget_kwh=80.0,
                water_available_l=500.0,
            )
            total_input += 200.0
            assert result.energy_used_kwh >= 0
            assert state.clean_kg >= 0
            assert state.contaminated_kg >= 0

        # After 30 sols we should have processed significant material
        assert state.total_processed_kg > 0
        assert state.clean_kg > 0
        assert state.total_o2_recovered_kg > 0
        assert state.sols_running == 30

    def test_100_sol_convergence(self):
        """Over 100 sols, contaminated backlog should stabilize."""
        state = create_scrubber()
        backlogs = []
        for sol in range(100):
            state, _ = tick_scrubber(
                state,
                regolith_input_kg=100.0,
                power_budget_kwh=100.0,
                water_available_l=500.0,
            )
            backlogs.append(state.contaminated_kg)

        # Backlog should not grow without bound
        assert backlogs[-1] < backlogs[50] + 10000

    def test_drain_stockpile(self):
        """Feed 1000 kg once, then tick with no input until drained."""
        state = create_scrubber()
        state, _ = tick_scrubber(state, regolith_input_kg=1000.0,
                                  power_budget_kwh=200.0,
                                  water_available_l=1000.0)
        for _ in range(50):
            state, _ = tick_scrubber(state, regolith_input_kg=0.0,
                                      power_budget_kwh=200.0,
                                      water_available_l=1000.0)
        assert state.contaminated_kg < 1.0  # nearly all processed

    def test_electron_donor_depletion(self):
        """Bioreactor stops when acetate runs out."""
        state = create_scrubber(electron_donor_kg=5.0)
        state.bio_culture_health = 1.0
        alloc = {"thermal": 0.0, "bio": 1.0, "ix": 0.0}
        bio_totals = []
        for _ in range(20):
            state, result = tick_scrubber(
                state, regolith_input_kg=100.0,
                power_budget_kwh=50.0,
                allocation=alloc,
            )
            bio_totals.append(result.bio_processed_kg)

        # Should have some zeros after donor runs out
        assert any(b == 0.0 for b in bio_totals[-5:])


# ============================================================================
# 10. Factory and utility
# ============================================================================

class TestFactory:
    """Test factory and utility functions."""

    def test_create_default(self):
        state = create_scrubber()
        assert state.kiln_temp_c == 600.0
        assert state.bio_temp_c == 30.0
        assert state.bio_culture_health == 0.5
        assert state.sols_running == 0

    def test_create_custom(self):
        state = create_scrubber(kiln_temp_c=500.0, bio_temp_c=25.0,
                                 electron_donor_kg=100.0)
        assert state.kiln_temp_c == 500.0
        assert state.bio_temp_c == 25.0
        assert state.electron_donor_kg == 100.0

    def test_create_clamps_kiln_temp(self):
        state = create_scrubber(kiln_temp_c=2000.0)
        assert state.kiln_temp_c <= THERMAL_MAX_TEMP_C

    def test_create_clamps_negative_donor(self):
        state = create_scrubber(electron_donor_kg=-10.0)
        assert state.electron_donor_kg == 0.0

    def test_scrubber_power_estimate(self):
        state = create_scrubber()
        state.contaminated_kg = 1000.0
        power = scrubber_power_kwh(state)
        assert power > 0

    def test_scrubber_power_empty(self):
        state = create_scrubber()
        state.contaminated_kg = 0.0
        power = scrubber_power_kwh(state)
        assert power == 0.0


# ============================================================================
# 11. Property-based invariants
# ============================================================================

class TestInvariants:
    """Physical invariants that must hold under all conditions."""

    def test_clean_never_negative(self):
        state = create_scrubber()
        for _ in range(10):
            state, result = tick_scrubber(state, regolith_input_kg=50.0)
            assert state.clean_kg >= 0
            assert result.clean_kg_produced >= 0

    def test_contaminated_never_negative(self):
        state = create_scrubber()
        state.contaminated_kg = 10.0
        for _ in range(100):
            state, _ = tick_scrubber(state, regolith_input_kg=0.0,
                                      power_budget_kwh=1000.0,
                                      water_available_l=10000.0)
            assert state.contaminated_kg >= 0

    def test_culture_health_bounded(self):
        state = create_scrubber()
        state.bio_culture_health = 0.5
        for _ in range(200):
            state, _ = tick_scrubber(state, regolith_input_kg=10.0)
            assert 0.0 <= state.bio_culture_health <= 1.0

    def test_energy_monotonic(self):
        state = create_scrubber()
        for _ in range(10):
            prev = state.total_energy_kwh
            state, _ = tick_scrubber(state, regolith_input_kg=100.0)
            assert state.total_energy_kwh >= prev

    def test_o2_monotonic(self):
        state = create_scrubber()
        for _ in range(10):
            prev = state.total_o2_recovered_kg
            state, _ = tick_scrubber(state, regolith_input_kg=100.0)
            assert state.total_o2_recovered_kg >= prev

    def test_processed_equals_clean_plus_perchlorate_plus_remaining(self):
        """Across multiple ticks, mass must be conserved."""
        state = create_scrubber()
        total_input = 0.0
        total_clean = 0.0
        total_perc_removed = 0.0
        for _ in range(20):
            inp = 150.0
            total_input += inp
            state, result = tick_scrubber(state, regolith_input_kg=inp)
            total_clean += result.clean_kg_produced
            total_perc_removed += result.perchlorate_removed_kg

        # total_input = total_clean + total_perc_removed + remaining
        balance = total_clean + total_perc_removed + state.contaminated_kg
        assert abs(balance - total_input) < 1.0  # within 1 kg tolerance
