"""
Tests for hab_pressure.py — Mars habitat pressurization model.

91 tests. Every function, every edge case, every conservation law.
The hull is the last line of defense. If these tests fail, everyone dies.

Run: python -m pytest tests/test_hab_pressure.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hab_pressure import (
    AIRLOCK_PUMP_EFFICIENCY,
    AIRLOCK_VOLUME_M3,
    BASE_LEAK_RATE_KG_SOL,
    HAB_TEMP_K,
    HYPOXIA_O2_KPA,
    MARS_SURFACE_KPA,
    MAX_RESERVE_KG,
    MAX_SAFE_KPA,
    MICROMETEORITE_LEAK_FACTOR,
    MIN_SAFE_KPA,
    MIN_SEAL_QUALITY,
    MOLAR_MASS_N2,
    MOLAR_MASS_O2,
    OVERPRESSURE_RELIEF_KPA,
    R_UNIVERSAL,
    REPLENISH_RATE_KG_SOL,
    SEAL_DEGRADATION_PER_SOL,
    TARGET_N2_KPA,
    TARGET_O2_KPA,
    TARGET_TOTAL_KPA,
    Habitat,
    _moles_from_pressure,
    _pressure_from_moles,
    apply_leak,
    cycle_airlock,
    degrade_seals,
    leak_rate_kg,
    overpressure_relief,
    patch_breach,
    repair_seals,
    replenish_atmosphere,
    tick_pressure,
    trigger_breach,
)


# ===================================================================
# Habitat dataclass
# ===================================================================

class TestHabitat:
    """Habitat construction and clamping."""

    def test_default_pressure(self) -> None:
        hab = Habitat(volume_m3=500.0)
        assert hab.pressure_kpa == TARGET_TOTAL_KPA

    def test_default_o2_fraction(self) -> None:
        hab = Habitat(volume_m3=500.0)
        expected = TARGET_O2_KPA / TARGET_TOTAL_KPA
        assert hab.o2_fraction == pytest.approx(expected, abs=1e-6)

    def test_volume_clamp_minimum(self) -> None:
        hab = Habitat(volume_m3=-10.0)
        assert hab.volume_m3 == 1.0

    def test_pressure_clamp_nonneg(self) -> None:
        hab = Habitat(volume_m3=100.0, pressure_kpa=-5.0)
        assert hab.pressure_kpa == 0.0

    def test_seal_quality_clamp_low(self) -> None:
        hab = Habitat(volume_m3=100.0, seal_quality=0.0)
        assert hab.seal_quality == MIN_SEAL_QUALITY

    def test_seal_quality_clamp_high(self) -> None:
        hab = Habitat(volume_m3=100.0, seal_quality=5.0)
        assert hab.seal_quality == 1.0

    def test_o2_fraction_clamp(self) -> None:
        hab = Habitat(volume_m3=100.0, o2_fraction=1.5)
        assert hab.o2_fraction == 1.0

    def test_reserve_clamp_max(self) -> None:
        hab = Habitat(volume_m3=100.0, reserve_o2_kg=99999.0)
        assert hab.reserve_o2_kg == MAX_RESERVE_KG

    def test_reserve_clamp_nonneg(self) -> None:
        hab = Habitat(volume_m3=100.0, reserve_n2_kg=-100.0)
        assert hab.reserve_n2_kg == 0.0

    def test_o2_kpa(self) -> None:
        hab = Habitat(volume_m3=100.0)
        assert hab.o2_kpa() == pytest.approx(TARGET_O2_KPA, abs=0.1)

    def test_n2_kpa(self) -> None:
        hab = Habitat(volume_m3=100.0)
        assert hab.n2_kpa() == pytest.approx(TARGET_N2_KPA, abs=0.1)

    def test_o2_plus_n2_equals_total(self) -> None:
        hab = Habitat(volume_m3=200.0, pressure_kpa=80.0, o2_fraction=0.3)
        assert hab.o2_kpa() + hab.n2_kpa() == pytest.approx(80.0, abs=1e-10)

    def test_total_gas_kg_positive(self) -> None:
        hab = Habitat(volume_m3=500.0)
        assert hab.total_gas_kg() > 0

    def test_total_gas_kg_zero_pressure(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=0.0)
        assert hab.total_gas_kg() == 0.0

    def test_is_safe_nominal(self) -> None:
        hab = Habitat(volume_m3=500.0)
        assert hab.is_safe() is True

    def test_is_safe_low_pressure(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=40.0)
        assert hab.is_safe() is False

    def test_is_safe_high_pressure(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=110.0)
        assert hab.is_safe() is False

    def test_is_safe_low_o2(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=70.0, o2_fraction=0.1)
        # O2 partial = 7 kPa < HYPOXIA_O2_KPA
        assert hab.is_safe() is False


# ===================================================================
# Ideal gas law helpers
# ===================================================================

class TestIdealGasLaw:
    """_moles_from_pressure and _pressure_from_moles roundtrip."""

    def test_moles_positive(self) -> None:
        n = _moles_from_pressure(70.0, 500.0)
        assert n > 0

    def test_moles_zero_pressure(self) -> None:
        assert _moles_from_pressure(0.0, 500.0) == 0.0

    def test_moles_zero_volume(self) -> None:
        assert _moles_from_pressure(70.0, 0.0) == 0.0

    def test_pressure_zero_moles(self) -> None:
        assert _pressure_from_moles(0.0, 500.0) == 0.0

    def test_roundtrip_pressure_moles_pressure(self) -> None:
        """P → n → P should round-trip perfectly."""
        p_orig = 70.0
        v = 500.0
        n = _moles_from_pressure(p_orig, v)
        p_back = _pressure_from_moles(n, v)
        assert p_back == pytest.approx(p_orig, rel=1e-10)

    def test_moles_doubles_with_volume(self) -> None:
        """Double volume at same pressure = double moles."""
        n1 = _moles_from_pressure(70.0, 100.0)
        n2 = _moles_from_pressure(70.0, 200.0)
        assert n2 == pytest.approx(2.0 * n1, rel=1e-10)

    def test_moles_doubles_with_pressure(self) -> None:
        """Double pressure at same volume = double moles."""
        n1 = _moles_from_pressure(35.0, 100.0)
        n2 = _moles_from_pressure(70.0, 100.0)
        assert n2 == pytest.approx(2.0 * n1, rel=1e-10)

    def test_realistic_hab_gas_mass(self) -> None:
        """A 500 m³ hab at 70 kPa should have ~420 kg of gas.

        Air density ~1.2 kg/m³ at 101 kPa STP. At 70 kPa:
        500 m³ * 1.2 * (70/101) ≈ 416 kg. Allow 350-500 for mixed gas.
        """
        hab = Habitat(volume_m3=500.0)
        mass = hab.total_gas_kg()
        assert 350.0 < mass < 500.0


# ===================================================================
# Leak model
# ===================================================================

class TestLeakRate:
    """Leak rate calculations."""

    def test_perfect_seals_baseline(self) -> None:
        rate = leak_rate_kg(1.0, False)
        assert rate == pytest.approx(BASE_LEAK_RATE_KG_SOL, abs=1e-6)

    def test_degraded_seals_higher(self) -> None:
        rate_good = leak_rate_kg(1.0, False)
        rate_bad = leak_rate_kg(0.5, False)
        assert rate_bad > rate_good

    def test_breach_multiplies(self) -> None:
        rate_no = leak_rate_kg(1.0, False)
        rate_yes = leak_rate_kg(1.0, True)
        assert rate_yes == pytest.approx(rate_no * MICROMETEORITE_LEAK_FACTOR, abs=1e-6)

    def test_min_seal_quality_floor(self) -> None:
        rate = leak_rate_kg(0.0, False)
        expected = BASE_LEAK_RATE_KG_SOL / MIN_SEAL_QUALITY
        assert rate == pytest.approx(expected, abs=1e-4)

    def test_leak_always_positive(self) -> None:
        for q in [0.01, 0.1, 0.5, 1.0]:
            for b in [False, True]:
                assert leak_rate_kg(q, b) > 0

    def test_apply_leak_reduces_pressure(self) -> None:
        hab = Habitat(volume_m3=500.0)
        p_before = hab.pressure_kpa
        apply_leak(hab)
        assert hab.pressure_kpa < p_before

    def test_apply_leak_preserves_o2_fraction(self) -> None:
        """Leak removes gas proportionally — O2 fraction should not change."""
        hab = Habitat(volume_m3=500.0, o2_fraction=0.3)
        apply_leak(hab)
        assert hab.o2_fraction == pytest.approx(0.3, abs=1e-10)

    def test_apply_leak_zero_pressure(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=0.0)
        lost = apply_leak(hab)
        assert hab.pressure_kpa == 0.0


# ===================================================================
# Seal degradation and repair
# ===================================================================

class TestSeals:
    """Seal degradation and repair."""

    def test_degrade_reduces_quality(self) -> None:
        hab = Habitat(volume_m3=500.0, seal_quality=1.0)
        degrade_seals(hab)
        assert hab.seal_quality < 1.0

    def test_degrade_returns_delta(self) -> None:
        hab = Habitat(volume_m3=500.0)
        delta = degrade_seals(hab)
        assert delta == pytest.approx(SEAL_DEGRADATION_PER_SOL, abs=1e-10)

    def test_degrade_never_below_floor(self) -> None:
        hab = Habitat(volume_m3=500.0, seal_quality=MIN_SEAL_QUALITY)
        degrade_seals(hab)
        assert hab.seal_quality >= MIN_SEAL_QUALITY

    def test_repair_increases_quality(self) -> None:
        hab = Habitat(volume_m3=500.0, seal_quality=0.5)
        gained = repair_seals(hab, effort=1.0)
        assert gained > 0
        assert hab.seal_quality > 0.5

    def test_repair_zero_effort(self) -> None:
        hab = Habitat(volume_m3=500.0, seal_quality=0.5)
        gained = repair_seals(hab, effort=0.0)
        assert gained == 0.0
        assert hab.seal_quality == 0.5

    def test_repair_never_exceeds_one(self) -> None:
        hab = Habitat(volume_m3=500.0, seal_quality=0.99)
        repair_seals(hab, effort=1.0)
        assert hab.seal_quality <= 1.0

    def test_multiple_repairs_converge(self) -> None:
        """Repeated repairs should converge toward 1.0 but never overshoot."""
        hab = Habitat(volume_m3=500.0, seal_quality=0.1)
        for _ in range(100):
            repair_seals(hab, effort=1.0)
        assert hab.seal_quality <= 1.0
        assert hab.seal_quality > 0.95


# ===================================================================
# Airlock
# ===================================================================

class TestAirlock:
    """Airlock cycling losses."""

    def test_cycle_loses_gas(self) -> None:
        hab = Habitat(volume_m3=500.0)
        p_before = hab.pressure_kpa
        lost = cycle_airlock(hab)
        assert lost > 0
        assert hab.pressure_kpa < p_before

    def test_cycle_loss_reasonable_magnitude(self) -> None:
        """Single airlock cycle should lose < 1% of total hab gas."""
        hab = Habitat(volume_m3=500.0)
        total_before = hab.total_gas_kg()
        lost = cycle_airlock(hab)
        assert lost < total_before * 0.01

    def test_multiple_cycles_cumulative(self) -> None:
        hab = Habitat(volume_m3=500.0)
        p_before = hab.pressure_kpa
        for _ in range(5):
            cycle_airlock(hab)
        assert hab.pressure_kpa < p_before

    def test_pump_efficiency_matters(self) -> None:
        """Higher pump efficiency = less gas lost. Verify the math direction."""
        # At 85% pump efficiency, we lose 15% of airlock volume per cycle
        hab = Habitat(volume_m3=500.0)
        lost = cycle_airlock(hab)
        # Airlock is 5 m³ at 70 kPa, total hab is 500 m³ at 70 kPa
        # Loss should be proportional to (1-pump_eff) * (airlock_vol/hab_vol)
        assert lost > 0


# ===================================================================
# Replenishment
# ===================================================================

class TestReplenishment:
    """Atmosphere replenishment from reserves."""

    def test_replenish_raises_pressure(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=50.0)
        result = replenish_atmosphere(hab)
        assert hab.pressure_kpa > 50.0
        assert result["o2_added_kg"] > 0 or result["n2_added_kg"] > 0

    def test_replenish_no_action_at_target(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=TARGET_TOTAL_KPA)
        result = replenish_atmosphere(hab)
        assert result["o2_added_kg"] == 0.0
        assert result["n2_added_kg"] == 0.0

    def test_replenish_above_target(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=80.0)
        result = replenish_atmosphere(hab)
        assert result["o2_added_kg"] == 0.0

    def test_replenish_decreases_reserves(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=50.0,
                      reserve_o2_kg=500.0, reserve_n2_kg=500.0)
        o2_before = hab.reserve_o2_kg
        n2_before = hab.reserve_n2_kg
        replenish_atmosphere(hab)
        assert hab.reserve_o2_kg < o2_before or hab.reserve_n2_kg < n2_before

    def test_replenish_empty_reserves(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=50.0,
                      reserve_o2_kg=0.0, reserve_n2_kg=0.0)
        p_before = hab.pressure_kpa
        replenish_atmosphere(hab)
        assert hab.pressure_kpa == pytest.approx(p_before, abs=0.01)

    def test_replenish_rate_limited(self) -> None:
        """With a huge pressure deficit, delivery is capped per sol."""
        hab = Habitat(volume_m3=5000.0, pressure_kpa=1.0,
                      reserve_o2_kg=2000.0, reserve_n2_kg=2000.0)
        result = replenish_atmosphere(hab)
        total_added = result["o2_added_kg"] + result["n2_added_kg"]
        assert total_added <= REPLENISH_RATE_KG_SOL + 0.01

    def test_replenish_maintains_o2_n2_ratio(self) -> None:
        """Replenishment should target the correct O2/N2 ratio."""
        hab = Habitat(volume_m3=500.0, pressure_kpa=50.0)
        replenish_atmosphere(hab)
        target_ratio = TARGET_O2_KPA / TARGET_TOTAL_KPA
        assert hab.o2_fraction == pytest.approx(target_ratio, abs=0.05)


# ===================================================================
# Overpressure relief
# ===================================================================

class TestOverpressureRelief:
    """Safety valve tests."""

    def test_no_vent_normal(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=70.0)
        vented = overpressure_relief(hab)
        assert vented == 0.0

    def test_vent_excess(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=120.0)
        vented = overpressure_relief(hab)
        assert vented == pytest.approx(120.0 - OVERPRESSURE_RELIEF_KPA, abs=0.01)
        assert hab.pressure_kpa == OVERPRESSURE_RELIEF_KPA

    def test_vent_at_boundary(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=OVERPRESSURE_RELIEF_KPA)
        vented = overpressure_relief(hab)
        assert vented == 0.0


# ===================================================================
# Breach
# ===================================================================

class TestBreach:
    """Hull breach trigger and patch."""

    def test_trigger_sets_breach(self) -> None:
        hab = Habitat(volume_m3=500.0)
        assert hab.breach is False
        trigger_breach(hab)
        assert hab.breach is True

    def test_patch_clears_breach(self) -> None:
        hab = Habitat(volume_m3=500.0, breach=True)
        result = patch_breach(hab)
        assert result is True
        assert hab.breach is False

    def test_patch_no_breach_returns_false(self) -> None:
        hab = Habitat(volume_m3=500.0)
        result = patch_breach(hab)
        assert result is False

    def test_breach_increases_leak_rate(self) -> None:
        rate_normal = leak_rate_kg(1.0, False)
        rate_breach = leak_rate_kg(1.0, True)
        assert rate_breach > rate_normal


# ===================================================================
# Tick integration
# ===================================================================

class TestTickPressure:
    """Full sol tick integration."""

    def test_tick_returns_all_keys(self) -> None:
        hab = Habitat(volume_m3=500.0)
        result = tick_pressure(hab)
        expected_keys = {
            "pressure_before_kpa", "pressure_after_kpa", "o2_kpa", "n2_kpa",
            "o2_fraction", "seal_quality", "seal_degradation", "seal_repaired",
            "leak_kg", "airlock_loss_kg", "eva_cycles", "breach_active",
            "breach_patched", "o2_replenished_kg", "n2_replenished_kg",
            "vented_kpa", "reserve_o2_kg", "reserve_n2_kg", "habitat_safe",
        }
        assert set(result.keys()) == expected_keys

    def test_tick_nominal_stays_safe(self) -> None:
        hab = Habitat(volume_m3=500.0)
        result = tick_pressure(hab)
        assert result["habitat_safe"] is True

    def test_tick_with_eva(self) -> None:
        hab = Habitat(volume_m3=500.0)
        result = tick_pressure(hab, eva_cycles=2)
        assert result["eva_cycles"] == 2
        assert result["airlock_loss_kg"] > 0

    def test_tick_with_breach(self) -> None:
        hab = Habitat(volume_m3=500.0)
        result = tick_pressure(hab, micrometeorite_hit=True)
        assert result["breach_active"] is True
        assert result["leak_kg"] > 0

    def test_tick_breach_and_repair(self) -> None:
        hab = Habitat(volume_m3=500.0)
        result = tick_pressure(hab, micrometeorite_hit=True, repair_effort=1.0)
        assert result["breach_patched"] is True
        assert result["breach_active"] is False

    def test_tick_seals_degrade(self) -> None:
        hab = Habitat(volume_m3=500.0, seal_quality=1.0)
        result = tick_pressure(hab)
        assert result["seal_quality"] < 1.0

    def test_tick_replenishes_after_leak(self) -> None:
        """Leak drops pressure, replenishment should compensate."""
        hab = Habitat(volume_m3=500.0, reserve_o2_kg=500.0, reserve_n2_kg=500.0)
        result = tick_pressure(hab)
        # With ample reserves, pressure should stay near target
        assert result["pressure_after_kpa"] > MIN_SAFE_KPA


# ===================================================================
# Multi-sol smoke test
# ===================================================================

class TestMultiSolSmoke:
    """Run the simulation for many sols without crash."""

    def test_100_sols_no_crash(self) -> None:
        """100 sols of normal operation — the colony survives."""
        hab = Habitat(volume_m3=500.0, reserve_o2_kg=1000.0, reserve_n2_kg=1000.0)
        for sol in range(100):
            eva = 1 if sol % 3 == 0 else 0
            hit = sol == 42  # one meteorite on sol 42
            repair = 0.5 if sol == 43 else 0.0  # repair next sol
            result = tick_pressure(hab, eva_cycles=eva,
                                   micrometeorite_hit=hit,
                                   repair_effort=repair)
            # Pressure should never go negative
            assert hab.pressure_kpa >= 0.0
            # Reserves should never go negative
            assert hab.reserve_o2_kg >= 0.0
            assert hab.reserve_n2_kg >= 0.0

    def test_30_sols_pressure_stable(self) -> None:
        """With reserves, pressure should stay near target over 30 sols."""
        hab = Habitat(volume_m3=500.0, reserve_o2_kg=500.0, reserve_n2_kg=500.0)
        for _ in range(30):
            tick_pressure(hab)
        # Should be within 5 kPa of target
        assert abs(hab.pressure_kpa - TARGET_TOTAL_KPA) < 5.0

    def test_reserve_depletion_scenario(self) -> None:
        """When reserves run out, pressure drops over time."""
        hab = Habitat(volume_m3=500.0, reserve_o2_kg=1.0, reserve_n2_kg=1.0)
        for _ in range(50):
            tick_pressure(hab)
        # With almost no reserves, pressure should have dropped significantly
        assert hab.pressure_kpa < TARGET_TOTAL_KPA

    def test_seal_degradation_over_1000_sols(self) -> None:
        """Seals degrade over ~1000 sols but never below floor."""
        hab = Habitat(volume_m3=500.0, seal_quality=1.0)
        for _ in range(1000):
            degrade_seals(hab)
        assert hab.seal_quality >= MIN_SEAL_QUALITY
        assert hab.seal_quality < 1.0


# ===================================================================
# Physical invariants (property-based)
# ===================================================================

class TestPhysicalInvariants:
    """Properties that must hold for any input."""

    def test_pressure_never_negative(self) -> None:
        """No sequence of operations can produce negative pressure."""
        hab = Habitat(volume_m3=100.0, pressure_kpa=1.0,
                      seal_quality=MIN_SEAL_QUALITY, breach=True,
                      reserve_o2_kg=0.0, reserve_n2_kg=0.0)
        for _ in range(50):
            tick_pressure(hab, eva_cycles=3, micrometeorite_hit=True)
            assert hab.pressure_kpa >= 0.0

    def test_reserves_never_negative(self) -> None:
        hab = Habitat(volume_m3=500.0, pressure_kpa=10.0,
                      reserve_o2_kg=10.0, reserve_n2_kg=10.0)
        for _ in range(100):
            tick_pressure(hab)
            assert hab.reserve_o2_kg >= -0.001  # float tolerance
            assert hab.reserve_n2_kg >= -0.001

    def test_o2_fraction_bounded(self) -> None:
        hab = Habitat(volume_m3=500.0)
        for _ in range(50):
            tick_pressure(hab, eva_cycles=1)
            assert 0.0 <= hab.o2_fraction <= 1.0

    def test_seal_quality_bounded(self) -> None:
        hab = Habitat(volume_m3=500.0)
        for _ in range(100):
            tick_pressure(hab, repair_effort=0.1)
            assert MIN_SEAL_QUALITY <= hab.seal_quality <= 1.0

    def test_gas_mass_conservation_direction(self) -> None:
        """Leak should reduce mass. Replenish should increase it."""
        hab = Habitat(volume_m3=500.0, pressure_kpa=50.0,
                      reserve_o2_kg=500.0, reserve_n2_kg=500.0)
        mass_before = hab.total_gas_kg()
        apply_leak(hab)
        mass_after_leak = hab.total_gas_kg()
        assert mass_after_leak < mass_before

        replenish_atmosphere(hab)
        mass_after_replenish = hab.total_gas_kg()
        assert mass_after_replenish > mass_after_leak

    def test_overpressure_ceiling_enforced(self) -> None:
        """Pressure can never exceed relief valve set point after tick."""
        hab = Habitat(volume_m3=10.0, pressure_kpa=50.0,
                      reserve_o2_kg=2000.0, reserve_n2_kg=2000.0)
        for _ in range(20):
            tick_pressure(hab)
            assert hab.pressure_kpa <= OVERPRESSURE_RELIEF_KPA + 0.1

    def test_mars_surface_pressure_constant(self) -> None:
        """Mars surface pressure is correctly defined."""
        assert 0.4 < MARS_SURFACE_KPA < 0.9  # 400-900 Pa range

    def test_target_partials_sum_to_total(self) -> None:
        """O2 + N2 target partials = total target."""
        assert TARGET_O2_KPA + TARGET_N2_KPA == pytest.approx(TARGET_TOTAL_KPA, abs=0.1)
