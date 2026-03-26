"""Tests for perchlorate_scrubber.py — Mars Perchlorate Remediation System.

Covers:
  - Physical constants sanity (8 tests)
  - Thermal energy calculations (8 tests)
  - Thermal decomposition sigmoid (8 tests)
  - Stoichiometry / mass balance (10 tests)
  - Iron reduction rate (6 tests)
  - Residual PPM (5 tests)
  - Alert classification (5 tests)
  - Basic tick behaviour (10 tests)
  - Power-limited tick (5 tests)
  - Chemical-only tick (5 tests)
  - Multi-sol tick sequences (5 tests)
  - Conservation laws (8 tests)
  - Edge cases (5 tests)
  - Factory function (3 tests)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.perchlorate_scrubber import (
    # Constants
    PERCHLORATE_MOLAR_MASS,
    NACLO4_MOLAR_MASS,
    NACL_MOLAR_MASS,
    O2_MOLAR_MASS,
    THERMAL_DECOMP_TEMP_C,
    THERMAL_DECOMP_MIN_C,
    MARS_AMBIENT_TEMP_C,
    REGOLITH_SPECIFIC_HEAT,
    HEATING_EFFICIENCY,
    SAFE_SOIL_PERCHLORATE_PPM,
    SAFE_WATER_PERCHLORATE_UG_L,
    MARS_REGOLITH_PERCHLORATE_FRACTION,
    IRON_PER_KG_PERCHLORATE,
    WATER_PER_KG_REGOLITH,
    MAX_BATCH_KG,
    UV_EFFICIENCY_FACTOR,
    EQUIPMENT_DEGRADATION_PER_SOL,
    CRITICAL_HEALTH,
    WARNING_HEALTH,
    # Dataclasses
    PerchlorateState,
    ScrubberTickResult,
    # Functions
    thermal_energy_kwh,
    thermal_decomp_fraction,
    perchlorate_mass_kg,
    o2_from_perchlorate_kg,
    salt_from_perchlorate_kg,
    iron_reduction_rate,
    residual_perchlorate_ppm,
    assess_alert,
    tick_perchlorate,
    create_scrubber,
)

import pytest


# =====================================================================
# 1. TestPhysicalConstants
# =====================================================================

class TestPhysicalConstants:
    """Verify that physical constants are within known physical bounds."""

    def test_perchlorate_molar_mass_range(self) -> None:
        assert 95.0 < PERCHLORATE_MOLAR_MASS < 105.0

    def test_naclo4_molar_mass_range(self) -> None:
        assert 120.0 < NACLO4_MOLAR_MASS < 125.0

    def test_nacl_molar_mass_range(self) -> None:
        assert 56.0 < NACL_MOLAR_MASS < 60.0

    def test_o2_molar_mass(self) -> None:
        assert O2_MOLAR_MASS == pytest.approx(32.0, abs=0.1)

    def test_thermal_decomp_temp_above_min(self) -> None:
        assert THERMAL_DECOMP_TEMP_C > THERMAL_DECOMP_MIN_C

    def test_mars_ambient_temperature_negative(self) -> None:
        assert MARS_AMBIENT_TEMP_C < 0.0

    def test_perchlorate_fraction_in_range(self) -> None:
        assert 0.001 < MARS_REGOLITH_PERCHLORATE_FRACTION < 0.02

    def test_stoichiometric_mass_balance(self) -> None:
        """NaClO₄ = NaCl + 2×O₂ within 0.1 g/mol."""
        assert NACLO4_MOLAR_MASS == pytest.approx(
            NACL_MOLAR_MASS + 2.0 * O2_MOLAR_MASS, abs=0.1,
        )


# =====================================================================
# 2. TestThermalEnergy
# =====================================================================

class TestThermalEnergy:
    """Test thermal_energy_kwh — heating regolith."""

    def test_zero_mass_zero_energy(self) -> None:
        assert thermal_energy_kwh(0.0, -60.0, 450.0) == 0.0

    def test_positive_mass_positive_energy(self) -> None:
        assert thermal_energy_kwh(100.0, -60.0, 450.0) > 0.0

    def test_higher_mass_more_energy(self) -> None:
        e1 = thermal_energy_kwh(100.0, -60.0, 450.0)
        e2 = thermal_energy_kwh(200.0, -60.0, 450.0)
        assert e2 > e1

    def test_larger_delta_t_more_energy(self) -> None:
        e1 = thermal_energy_kwh(100.0, -60.0, 400.0)
        e2 = thermal_energy_kwh(100.0, -60.0, 500.0)
        assert e2 > e1

    def test_mars_ambient_to_target_realistic(self) -> None:
        """200 kg from −60→450 °C ≈ 32 kWh (order of magnitude)."""
        e = thermal_energy_kwh(200.0, MARS_AMBIENT_TEMP_C, THERMAL_DECOMP_TEMP_C)
        assert 10.0 < e < 100.0

    def test_negative_mass_clamped(self) -> None:
        assert thermal_energy_kwh(-50.0, -60.0, 450.0) == 0.0

    def test_target_below_start_zero_energy(self) -> None:
        assert thermal_energy_kwh(100.0, 450.0, -60.0) == 0.0

    def test_efficiency_factor_increases_energy(self) -> None:
        """Energy should be higher than naive specific-heat calculation."""
        mass, dt = 100.0, 510.0
        naive_kwh = mass * REGOLITH_SPECIFIC_HEAT * dt / 3600.0
        actual = thermal_energy_kwh(mass, -60.0, 450.0)
        assert actual > naive_kwh


# =====================================================================
# 3. TestThermalDecomp
# =====================================================================

class TestThermalDecomp:
    """Test thermal_decomp_fraction — sigmoid decomposition curve."""

    def test_below_400_returns_zero(self) -> None:
        assert thermal_decomp_fraction(399.9) == 0.0

    def test_at_400_small_nonzero(self) -> None:
        f = thermal_decomp_fraction(400.0)
        assert 0.0 < f < 0.10

    def test_at_450_around_half(self) -> None:
        f = thermal_decomp_fraction(450.0)
        assert 0.40 < f < 0.60

    def test_at_500_above_90_percent(self) -> None:
        f = thermal_decomp_fraction(500.0)
        assert f > 0.90

    def test_at_600_near_99_percent(self) -> None:
        f = thermal_decomp_fraction(600.0)
        assert f > 0.98

    def test_monotonically_increasing(self) -> None:
        temps = [400.0, 420.0, 450.0, 500.0, 550.0, 600.0]
        fractions = [thermal_decomp_fraction(t) for t in temps]
        for i in range(len(fractions) - 1):
            assert fractions[i + 1] >= fractions[i]

    def test_never_exceeds_max(self) -> None:
        for t in [500.0, 700.0, 1000.0, 2000.0]:
            assert thermal_decomp_fraction(t) <= 1.0

    def test_at_minus_60_returns_zero(self) -> None:
        assert thermal_decomp_fraction(-60.0) == 0.0


# =====================================================================
# 4. TestStoichiometry
# =====================================================================

class TestStoichiometry:
    """Test o2_from_perchlorate_kg and salt_from_perchlorate_kg."""

    def test_o2_from_zero(self) -> None:
        assert o2_from_perchlorate_kg(0.0) == 0.0

    def test_o2_positive(self) -> None:
        assert o2_from_perchlorate_kg(1.0) > 0.0

    def test_o2_proportional(self) -> None:
        assert o2_from_perchlorate_kg(2.0) == pytest.approx(
            2.0 * o2_from_perchlorate_kg(1.0),
        )

    def test_o2_stoichiometric_ratio(self) -> None:
        """1 kg NaClO₄ → (2×32/122.44) kg O₂."""
        expected = 2.0 * O2_MOLAR_MASS / NACLO4_MOLAR_MASS
        assert o2_from_perchlorate_kg(1.0) == pytest.approx(expected)

    def test_salt_from_zero(self) -> None:
        assert salt_from_perchlorate_kg(0.0) == 0.0

    def test_salt_positive(self) -> None:
        assert salt_from_perchlorate_kg(1.0) > 0.0

    def test_salt_proportional(self) -> None:
        assert salt_from_perchlorate_kg(3.0) == pytest.approx(
            3.0 * salt_from_perchlorate_kg(1.0),
        )

    def test_salt_stoichiometric_ratio(self) -> None:
        """1 kg NaClO₄ → (58.44/122.44) kg NaCl."""
        expected = NACL_MOLAR_MASS / NACLO4_MOLAR_MASS
        assert salt_from_perchlorate_kg(1.0) == pytest.approx(expected)

    def test_o2_plus_salt_equals_input(self) -> None:
        """Conservation: NaClO₄ → NaCl + 2 O₂.  Mass in ≈ mass out."""
        p = 5.0
        o2 = o2_from_perchlorate_kg(p)
        nacl = salt_from_perchlorate_kg(p)
        assert o2 + nacl == pytest.approx(p, rel=1e-6)

    def test_negative_perchlorate_clamped(self) -> None:
        assert o2_from_perchlorate_kg(-1.0) == 0.0
        assert salt_from_perchlorate_kg(-1.0) == 0.0


# =====================================================================
# 5. TestIronReduction
# =====================================================================

class TestIronReduction:
    """Test iron_reduction_rate."""

    def test_no_iron_returns_zero(self) -> None:
        assert iron_reduction_rate(0.0, 500.0, 200.0) == 0.0

    def test_no_water_returns_zero(self) -> None:
        assert iron_reduction_rate(50.0, 0.0, 200.0) == 0.0

    def test_no_regolith_returns_zero(self) -> None:
        assert iron_reduction_rate(50.0, 500.0, 0.0) == 0.0

    def test_abundant_resources_near_max(self) -> None:
        """With excess iron and water, rate should be near IRON_MAX_EFFICIENCY."""
        r = iron_reduction_rate(1000.0, 10000.0, 100.0)
        assert r == pytest.approx(0.85, abs=0.01)

    def test_limited_water_reduces_rate(self) -> None:
        r_full = iron_reduction_rate(50.0, 10000.0, 200.0)
        r_half = iron_reduction_rate(50.0, 200.0, 200.0)
        assert r_half < r_full

    def test_monotonic_with_iron(self) -> None:
        rates = [iron_reduction_rate(fe, 500.0, 200.0) for fe in [1, 5, 20, 50]]
        for i in range(len(rates) - 1):
            assert rates[i + 1] >= rates[i]


# =====================================================================
# 6. TestResidualPPM
# =====================================================================

class TestResidualPPM:
    """Test residual_perchlorate_ppm."""

    def test_zero_removal_full_ppm(self) -> None:
        assert residual_perchlorate_ppm(0.007, 0.0) == pytest.approx(7000.0)

    def test_full_removal_zero_ppm(self) -> None:
        assert residual_perchlorate_ppm(0.007, 1.0) == pytest.approx(0.0)

    def test_half_removal(self) -> None:
        assert residual_perchlorate_ppm(0.007, 0.5) == pytest.approx(3500.0)

    def test_default_perchlorate_gives_7000ppm(self) -> None:
        ppm = residual_perchlorate_ppm(MARS_REGOLITH_PERCHLORATE_FRACTION, 0.0)
        assert ppm == pytest.approx(7000.0)

    def test_negative_fraction_clamped(self) -> None:
        assert residual_perchlorate_ppm(0.007, -0.5) == pytest.approx(7000.0)


# =====================================================================
# 7. TestAlerts
# =====================================================================

class TestAlerts:
    """Test assess_alert."""

    def test_nominal_healthy(self) -> None:
        assert assess_alert(50.0, 0.9) == "NOMINAL"

    def test_warning_low_health(self) -> None:
        assert assess_alert(50.0, 0.2) == "WARNING"

    def test_warning_high_ppm(self) -> None:
        assert assess_alert(6000.0, 0.9) == "WARNING"

    def test_critical_very_low_health(self) -> None:
        assert assess_alert(50.0, 0.05) == "CRITICAL"

    def test_critical_overrides_ppm(self) -> None:
        """CRITICAL from health even when ppm is fine."""
        assert assess_alert(10.0, 0.05) == "CRITICAL"


# =====================================================================
# 8. TestTickBasic
# =====================================================================

class TestTickBasic:
    """Test single-tick behaviour with default parameters."""

    def _default_tick(self) -> tuple[PerchlorateState, ScrubberTickResult]:
        s = create_scrubber()
        return tick_perchlorate(s)

    def test_produces_output(self) -> None:
        _, r = self._default_tick()
        assert r.regolith_in_kg > 0.0

    def test_regolith_processed_accumulates(self) -> None:
        s, r = self._default_tick()
        assert s.regolith_processed_kg == pytest.approx(r.regolith_in_kg)

    def test_perchlorate_destroyed_positive(self) -> None:
        _, r = self._default_tick()
        assert r.perchlorate_removed_kg > 0.0

    def test_clean_soil_less_than_input(self) -> None:
        """O₂ escaped as gas → clean soil < regolith input."""
        _, r = self._default_tick()
        assert r.clean_soil_out_kg < r.regolith_in_kg

    def test_o2_released_positive(self) -> None:
        _, r = self._default_tick()
        assert r.o2_released_kg > 0.0

    def test_salt_produced_positive(self) -> None:
        _, r = self._default_tick()
        assert r.salt_produced_kg > 0.0

    def test_energy_used_positive(self) -> None:
        _, r = self._default_tick()
        assert r.energy_used_kwh > 0.0

    def test_equipment_health_decreases(self) -> None:
        s, _ = self._default_tick()
        assert s.equipment_health < 1.0

    def test_sol_counter_increments(self) -> None:
        s, _ = self._default_tick()
        assert s.sols_running == 1

    def test_default_alert_nominal(self) -> None:
        _, r = self._default_tick()
        assert r.alert == "NOMINAL"


# =====================================================================
# 9. TestTickPowerLimited
# =====================================================================

class TestTickPowerLimited:
    """Test behaviour when power is insufficient for full thermal."""

    def test_zero_power_no_thermal_removal(self) -> None:
        """Zero power → thermal path produces nothing (UV still works)."""
        s = create_scrubber()
        _, r = tick_perchlorate(s, power_available_kwh=0.0)
        # Only chemical + UV contribute; thermal fraction should be 0
        assert r.thermal_fraction == 0.0

    def test_half_power_lower_removal(self) -> None:
        s1 = create_scrubber()
        _, r_full = tick_perchlorate(s1, power_available_kwh=50.0)
        s2 = create_scrubber()
        _, r_half = tick_perchlorate(s2, power_available_kwh=10.0)
        assert r_half.perchlorate_removed_kg <= r_full.perchlorate_removed_kg

    def test_full_power_higher_removal(self) -> None:
        s1 = create_scrubber()
        _, r_low = tick_perchlorate(s1, power_available_kwh=5.0)
        s2 = create_scrubber()
        _, r_high = tick_perchlorate(s2, power_available_kwh=100.0)
        assert r_high.perchlorate_removed_kg >= r_low.perchlorate_removed_kg

    def test_energy_never_exceeds_available(self) -> None:
        s = create_scrubber()
        _, r = tick_perchlorate(s, power_available_kwh=10.0)
        assert r.energy_used_kwh <= 10.0 + 1e-9

    def test_very_low_power_still_some_removal(self) -> None:
        """UV is free; should still remove some perchlorate."""
        s = create_scrubber()
        _, r = tick_perchlorate(
            s, power_available_kwh=0.0, use_thermal=False, use_chemical=False,
        )
        assert r.perchlorate_removed_kg > 0.0


# =====================================================================
# 10. TestTickChemicalOnly
# =====================================================================

class TestTickChemicalOnly:
    """Test iron-only pathway (thermal disabled)."""

    def _chemical_tick(self) -> tuple[PerchlorateState, ScrubberTickResult]:
        s = create_scrubber()
        return tick_perchlorate(s, use_thermal=False)

    def test_still_removes_perchlorate(self) -> None:
        _, r = self._chemical_tick()
        assert r.perchlorate_removed_kg > 0.0

    def test_no_thermal_no_o2(self) -> None:
        """Thermal fraction = 0 → O₂ released = 0."""
        _, r = self._chemical_tick()
        assert r.thermal_fraction == 0.0
        assert r.o2_released_kg == 0.0

    def test_iron_consumed(self) -> None:
        _, r = self._chemical_tick()
        assert r.iron_consumed_kg > 0.0

    def test_water_used(self) -> None:
        _, r = self._chemical_tick()
        assert r.water_used_L > 0.0

    def test_chemical_fraction_is_one(self) -> None:
        _, r = self._chemical_tick()
        assert r.chemical_fraction == pytest.approx(1.0)


# =====================================================================
# 11. TestTickMultiSol
# =====================================================================

class TestTickMultiSol:
    """Test running multiple ticks in sequence."""

    def test_10_sols_monotonic_accumulation(self) -> None:
        s = create_scrubber()
        prev_processed = 0.0
        for _ in range(10):
            s, _ = tick_perchlorate(s)
            assert s.regolith_processed_kg >= prev_processed
            prev_processed = s.regolith_processed_kg

    def test_50_sols_iron_depletes(self) -> None:
        """Iron reserve should decrease over many sols."""
        s = create_scrubber(batch_capacity_kg=200.0)
        initial_iron = s.iron_reserve_kg
        for _ in range(50):
            s, _ = tick_perchlorate(s)
        assert s.iron_reserve_kg < initial_iron

    def test_equipment_degrades_over_time(self) -> None:
        s = create_scrubber()
        for _ in range(100):
            s, _ = tick_perchlorate(s)
        assert s.equipment_health < 1.0 - 100 * EQUIPMENT_DEGRADATION_PER_SOL + 1e-9

    def test_cumulative_regolith_never_decreases(self) -> None:
        s = create_scrubber()
        for _ in range(20):
            s, _ = tick_perchlorate(s)
        assert s.regolith_processed_kg >= 20 * 200.0 - 1e-6

    def test_cumulative_energy_never_decreases(self) -> None:
        s = create_scrubber()
        prev_energy = 0.0
        for _ in range(10):
            s, _ = tick_perchlorate(s)
            assert s.total_energy_kwh >= prev_energy
            prev_energy = s.total_energy_kwh


# =====================================================================
# 12. TestConservationLaws
# =====================================================================

class TestConservationLaws:
    """THE MOST IMPORTANT: mass and stoichiometric conservation."""

    def test_mass_balance_single_tick(self) -> None:
        """regolith_in ≈ clean_soil_out + o2_released."""
        s = create_scrubber()
        _, r = tick_perchlorate(s)
        assert r.regolith_in_kg == pytest.approx(
            r.clean_soil_out_kg + r.o2_released_kg, rel=1e-9,
        )

    def test_mass_balance_multi_tick(self) -> None:
        s = create_scrubber()
        total_in = 0.0
        total_soil = 0.0
        total_o2 = 0.0
        for _ in range(20):
            s, r = tick_perchlorate(s)
            total_in += r.regolith_in_kg
            total_soil += r.clean_soil_out_kg
            total_o2 += r.o2_released_kg
        assert total_in == pytest.approx(total_soil + total_o2, rel=1e-9)

    def test_o2_stoichiometry_thermal_only(self) -> None:
        """O₂ matches stoichiometry for thermally-destroyed perchlorate."""
        s = create_scrubber()
        _, r = tick_perchlorate(s, use_chemical=False)
        thermal_destroyed = r.perchlorate_removed_kg * r.thermal_fraction
        expected_o2 = o2_from_perchlorate_kg(thermal_destroyed)
        assert r.o2_released_kg == pytest.approx(expected_o2, rel=1e-9)

    def test_o2_stoichiometry_mixed(self) -> None:
        """O₂ = stoich(thermal_portion) even when chemical also active."""
        s = create_scrubber()
        _, r = tick_perchlorate(s)
        thermal_destroyed = r.perchlorate_removed_kg * r.thermal_fraction
        expected_o2 = o2_from_perchlorate_kg(thermal_destroyed)
        assert r.o2_released_kg == pytest.approx(expected_o2, rel=1e-9)

    def test_salt_stoichiometry(self) -> None:
        """salt_produced ≈ total_destroyed × (58.44/122.44)."""
        s = create_scrubber()
        _, r = tick_perchlorate(s)
        expected_salt = salt_from_perchlorate_kg(r.perchlorate_removed_kg)
        assert r.salt_produced_kg == pytest.approx(expected_salt, rel=1e-9)

    def test_fractions_sum_to_one(self) -> None:
        """thermal_fraction + chemical_fraction = 1 when removal > 0."""
        s = create_scrubber()
        _, r = tick_perchlorate(s)
        assert r.thermal_fraction + r.chemical_fraction == pytest.approx(1.0)

    def test_regolith_in_equals_clean_soil_plus_o2(self) -> None:
        """Explicit restatement of the core conservation law."""
        s = create_scrubber()
        for _ in range(5):
            s, r = tick_perchlorate(s)
            lhs = r.regolith_in_kg
            rhs = r.clean_soil_out_kg + r.o2_released_kg
            assert lhs == pytest.approx(rhs, rel=1e-9)

    def test_mass_balance_chemical_only(self) -> None:
        """Chemical-only: no O₂ → clean_soil == regolith_in."""
        s = create_scrubber()
        _, r = tick_perchlorate(s, use_thermal=False)
        assert r.o2_released_kg == 0.0
        assert r.clean_soil_out_kg == pytest.approx(r.regolith_in_kg)


# =====================================================================
# 13. TestEdgeCases
# =====================================================================

class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_zero_regolith_input(self) -> None:
        s = create_scrubber()
        _, r = tick_perchlorate(s, regolith_input_kg=0.0)
        assert r.regolith_in_kg == 0.0
        assert r.perchlorate_removed_kg == 0.0
        assert r.energy_used_kwh == 0.0

    def test_zero_perchlorate_fraction(self) -> None:
        s = create_scrubber()
        _, r = tick_perchlorate(s, input_perchlorate_fraction=0.0)
        assert r.perchlorate_removed_kg == 0.0
        assert r.o2_released_kg == 0.0

    def test_negative_inputs_clamped(self) -> None:
        s = create_scrubber()
        _, r = tick_perchlorate(
            s,
            power_available_kwh=-10.0,
            regolith_input_kg=-100.0,
            input_perchlorate_fraction=-0.5,
        )
        assert r.regolith_in_kg == 0.0
        assert r.perchlorate_removed_kg == 0.0
        assert r.energy_used_kwh == 0.0

    def test_max_batch_cap(self) -> None:
        """Input exceeding MAX_BATCH_KG is capped."""
        s = create_scrubber(batch_capacity_kg=1000.0)
        _, r = tick_perchlorate(s, regolith_input_kg=999.0)
        assert r.regolith_in_kg <= MAX_BATCH_KG

    def test_zero_thermal_units(self) -> None:
        """No thermal reactors → thermal path skipped."""
        s = create_scrubber(thermal_units=0)
        _, r = tick_perchlorate(s, use_thermal=True)
        assert r.thermal_fraction == 0.0


# =====================================================================
# 14. TestFactory
# =====================================================================

class TestFactory:
    """Test create_scrubber factory function."""

    def test_default_factory(self) -> None:
        s = create_scrubber()
        assert s.batch_capacity_kg == 200.0
        assert s.thermal_unit_count == 1
        assert s.equipment_health == 1.0
        assert s.sols_running == 0

    def test_custom_capacity(self) -> None:
        s = create_scrubber(batch_capacity_kg=400.0, thermal_units=3)
        assert s.batch_capacity_kg == 400.0
        assert s.thermal_unit_count == 3

    def test_negative_clamped(self) -> None:
        s = create_scrubber(batch_capacity_kg=-10.0, thermal_units=-2)
        assert s.batch_capacity_kg == 0.0
        assert s.thermal_unit_count == 0


# =====================================================================
# Additional conservation and regression tests
# =====================================================================

class TestPerchlorateMassFunction:
    """Test perchlorate_mass_kg helper."""

    def test_default_fraction(self) -> None:
        m = perchlorate_mass_kg(1000.0)
        assert m == pytest.approx(7.0)

    def test_zero_regolith(self) -> None:
        assert perchlorate_mass_kg(0.0) == 0.0

    def test_custom_fraction(self) -> None:
        assert perchlorate_mass_kg(100.0, 0.01) == pytest.approx(1.0)

    def test_negative_regolith_clamped(self) -> None:
        assert perchlorate_mass_kg(-50.0) == 0.0

    def test_fraction_clamped_above_one(self) -> None:
        assert perchlorate_mass_kg(100.0, 1.5) == pytest.approx(100.0)


class TestTickEquipmentDegradation:
    """Detailed equipment degradation behaviour."""

    def test_below_critical_reduces_throughput(self) -> None:
        s = create_scrubber()
        s.equipment_health = 0.05
        _, r = tick_perchlorate(s, regolith_input_kg=200.0)
        assert r.regolith_in_kg < 200.0

    def test_below_warning_degrades_quality(self) -> None:
        """Low health means less decomposition per unit."""
        s_good = create_scrubber()
        _, r_good = tick_perchlorate(s_good)

        s_bad = create_scrubber()
        s_bad.equipment_health = 0.2
        _, r_bad = tick_perchlorate(s_bad)
        assert r_bad.perchlorate_removed_kg < r_good.perchlorate_removed_kg

    def test_health_never_negative(self) -> None:
        s = create_scrubber()
        s.equipment_health = 0.0001
        s, _ = tick_perchlorate(s)
        assert s.equipment_health >= 0.0


class TestTickIronDepletion:
    """Verify iron and water consumption dynamics."""

    def test_iron_decreases_each_tick(self) -> None:
        s = create_scrubber()
        prev_iron = s.iron_reserve_kg
        for _ in range(5):
            s, _ = tick_perchlorate(s)
            assert s.iron_reserve_kg <= prev_iron
            prev_iron = s.iron_reserve_kg

    def test_water_decreases_slowly(self) -> None:
        s = create_scrubber()
        initial_water = s.water_budget_L
        for _ in range(10):
            s, _ = tick_perchlorate(s)
        assert s.water_budget_L < initial_water

    def test_iron_at_zero_chemical_path_inactive(self) -> None:
        s = create_scrubber()
        s.iron_reserve_kg = 0.0
        _, r = tick_perchlorate(s, use_thermal=False)
        # Only UV should work
        assert r.iron_consumed_kg == 0.0


class TestOutputPPMBounds:
    """Verify output concentration is physically reasonable."""

    def test_output_ppm_between_zero_and_input(self) -> None:
        s = create_scrubber()
        _, r = tick_perchlorate(s)
        input_ppm = MARS_REGOLITH_PERCHLORATE_FRACTION * 1_000_000
        assert 0.0 <= r.soil_perchlorate_ppm <= input_ppm

    def test_output_ppm_decreases_with_more_power(self) -> None:
        s1 = create_scrubber()
        _, r_low = tick_perchlorate(s1, power_available_kwh=5.0)
        s2 = create_scrubber()
        _, r_high = tick_perchlorate(s2, power_available_kwh=100.0)
        assert r_high.soil_perchlorate_ppm <= r_low.soil_perchlorate_ppm

    def test_safe_level_achievable(self) -> None:
        """With enough power and low perchlorate, can reach safe level."""
        s = create_scrubber()
        _, r = tick_perchlorate(
            s,
            power_available_kwh=200.0,
            input_perchlorate_fraction=0.00005,
        )
        assert r.soil_perchlorate_ppm < SAFE_SOIL_PERCHLORATE_PPM


class TestUVPathway:
    """Test the UV supplementary pathway."""

    def test_uv_only_removes_some(self) -> None:
        """Even with thermal and chemical off, UV removes 15% of perchlorate."""
        s = create_scrubber()
        _, r = tick_perchlorate(
            s, use_thermal=False, use_chemical=False,
        )
        perchlorate_in = perchlorate_mass_kg(200.0)
        expected = perchlorate_in * UV_EFFICIENCY_FACTOR
        assert r.perchlorate_removed_kg == pytest.approx(expected, rel=1e-6)

    def test_uv_no_energy_cost(self) -> None:
        s = create_scrubber()
        _, r = tick_perchlorate(
            s,
            power_available_kwh=0.0,
            use_thermal=False,
            use_chemical=False,
        )
        assert r.energy_used_kwh == 0.0
        assert r.perchlorate_removed_kg > 0.0

    def test_uv_no_iron_or_water(self) -> None:
        s = create_scrubber()
        _, r = tick_perchlorate(
            s, use_thermal=False, use_chemical=False,
        )
        assert r.iron_consumed_kg == 0.0
        assert r.water_used_L == 0.0
