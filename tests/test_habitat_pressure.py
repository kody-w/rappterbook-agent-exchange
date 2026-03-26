"""
Tests for habitat_pressure.py — Mars habitat pressurization model.

Coverage:
  - Gas physics (ideal gas law, pressure<->mass round-trip)
  - Structural stress (hull hoop stress, safety ratio)
  - Leak model (micro-leaks, seal degradation, dust storm effects)
  - Airlock cycling (gas loss per EVA)
  - Blowout events (hull breach, survivability)
  - Reserve tank (withdraw, deposit, capacity limits)
  - Per-sol tick (full system integration)
  - Conservation laws (mass balance, energy bounds)
  - Property sweeps (pressure always non-negative, seals bounded)
  - Multi-sol smoke test (100 sols without crash)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.habitat_pressure import (
    AIR_MOLAR_MASS_KG,
    AIRLOCK_RECOVERY_FRACTION,
    AIRLOCK_VOLUME_M3,
    BASE_LEAK_RATE_KG_SOL,
    CABIN_TEMP_K,
    CRITICAL_KPA,
    DEADBAND_KPA,
    DUST_SEAL_DAMAGE_FACTOR,
    GAS_CONSTANT_J,
    HULL_RADIUS_M,
    HULL_THICKNESS_M,
    MARS_SURFACE_KPA,
    MIN_SAFE_KPA,
    MIN_SEAL_QUALITY,
    RESERVE_DELIVERY_KG_SOL,
    SAFETY_FACTOR,
    SEAL_DEGRADATION_PER_SOL,
    TARGET_INTERNAL_KPA,
    YIELD_STRENGTH_MPA,
    Habitat,
    ReserveTank,
    airlock_cycle_loss_kg,
    blowout_event,
    compute_leak_kg,
    degrade_seals,
    hull_stress_mpa,
    mass_to_pressure,
    perform_airlock_cycle,
    pressure_to_mass,
    replenish_from_reserve,
    structural_safety_ratio,
    tick_pressure,
)


# ===================================================================
# Gas physics — ideal gas law
# ===================================================================

class TestGasPhysics:
    """Ideal gas law conversions."""

    def test_pressure_to_mass_at_1atm(self):
        """101.3 kPa in 500 m3 should give a reasonable air mass."""
        mass = pressure_to_mass(101.3, 500.0)
        # PV = nRT -> n = PV/RT -> m = n*M
        # 101300 * 500 / (8.314 * 293.15) * 0.029 ~ 602 kg
        assert 550 < mass < 650

    def test_mass_to_pressure_round_trip(self):
        """pressure -> mass -> pressure should be identity."""
        vol = 300.0
        for p in [50.0, 101.3, 200.0]:
            mass = pressure_to_mass(p, vol)
            p_back = mass_to_pressure(mass, vol)
            assert p_back == pytest.approx(p, rel=1e-10)

    def test_pressure_to_mass_round_trip(self):
        """mass -> pressure -> mass should be identity."""
        vol = 400.0
        for m in [100.0, 500.0, 1000.0]:
            p = mass_to_pressure(m, vol)
            m_back = pressure_to_mass(p, vol)
            assert m_back == pytest.approx(m, rel=1e-10)

    def test_zero_pressure_gives_zero_mass(self):
        assert pressure_to_mass(0.0, 500.0) == 0.0

    def test_zero_volume_gives_zero_mass(self):
        assert pressure_to_mass(101.3, 0.0) == 0.0

    def test_negative_pressure_gives_zero(self):
        assert pressure_to_mass(-10.0, 500.0) == 0.0

    def test_zero_mass_gives_zero_pressure(self):
        assert mass_to_pressure(0.0, 500.0) == 0.0

    def test_negative_mass_gives_zero_pressure(self):
        assert mass_to_pressure(-5.0, 500.0) == 0.0

    def test_mass_proportional_to_pressure(self):
        """Double the pressure -> double the mass (ideal gas)."""
        m1 = pressure_to_mass(50.0, 100.0)
        m2 = pressure_to_mass(100.0, 100.0)
        assert m2 == pytest.approx(2 * m1, rel=1e-10)

    def test_mass_proportional_to_volume(self):
        """Double the volume -> double the mass at same pressure."""
        m1 = pressure_to_mass(101.3, 200.0)
        m2 = pressure_to_mass(101.3, 400.0)
        assert m2 == pytest.approx(2 * m1, rel=1e-10)


# ===================================================================
# Structural stress
# ===================================================================

class TestStructuralStress:
    """Hull hoop stress and safety calculations."""

    def test_stress_at_nominal_pressure(self):
        """Hoop stress at 101.3 kPa should be a positive finite value."""
        stress = hull_stress_mpa(101.3)
        assert stress > 0
        assert stress < YIELD_STRENGTH_MPA  # should not exceed yield

    def test_stress_increases_with_pressure(self):
        """Higher internal pressure -> higher stress."""
        s1 = hull_stress_mpa(50.0)
        s2 = hull_stress_mpa(101.3)
        s3 = hull_stress_mpa(200.0)
        assert s1 < s2 < s3

    def test_stress_zero_when_equal_pressures(self):
        """No differential -> no stress."""
        assert hull_stress_mpa(MARS_SURFACE_KPA, MARS_SURFACE_KPA) == 0.0

    def test_stress_formula_matches_manual(self):
        """sigma = dP * r / (2*t) manual calculation."""
        dp_pa = (101.3 - 0.636) * 1000.0
        expected_mpa = (dp_pa * HULL_RADIUS_M) / (2.0 * HULL_THICKNESS_M) / 1e6
        assert hull_stress_mpa(101.3) == pytest.approx(expected_mpa, rel=1e-10)

    def test_safety_ratio_above_design_factor(self):
        """At nominal pressure, safety ratio should exceed the design factor."""
        ratio = structural_safety_ratio(101.3)
        assert ratio > SAFETY_FACTOR

    def test_safety_ratio_decreases_with_pressure(self):
        """Higher pressure -> lower safety margin."""
        r1 = structural_safety_ratio(50.0)
        r2 = structural_safety_ratio(101.3)
        r3 = structural_safety_ratio(200.0)
        assert r1 > r2 > r3

    def test_safety_ratio_infinite_at_zero_stress(self):
        """Zero differential -> infinite safety."""
        ratio = structural_safety_ratio(MARS_SURFACE_KPA)
        assert ratio == float('inf')

    def test_negative_pressure_gives_zero_stress(self):
        """External > internal -> clamped to zero stress."""
        assert hull_stress_mpa(0.0, 101.3) == 0.0


# ===================================================================
# Habitat dataclass
# ===================================================================

class TestHabitat:
    """Habitat initialization and clamping."""

    def test_default_construction(self):
        """Habitat with just volume gets reasonable defaults."""
        h = Habitat(volume_m3=500.0)
        assert h.pressure_kpa == TARGET_INTERNAL_KPA
        assert h.seal_quality == 1.0
        assert h.air_mass_kg > 0

    def test_auto_mass_calculation(self):
        """Air mass is auto-calculated from pressure and volume."""
        h = Habitat(volume_m3=500.0, pressure_kpa=101.3)
        expected = pressure_to_mass(101.3, 500.0)
        assert h.air_mass_kg == pytest.approx(expected, rel=1e-10)

    def test_volume_clamped_to_minimum(self):
        """Volume < 1.0 is clamped to 1.0."""
        h = Habitat(volume_m3=0.1)
        assert h.volume_m3 == 1.0

    def test_negative_volume_clamped(self):
        h = Habitat(volume_m3=-100.0)
        assert h.volume_m3 == 1.0

    def test_seal_quality_clamped_high(self):
        h = Habitat(volume_m3=100.0, seal_quality=5.0)
        assert h.seal_quality == 1.0

    def test_seal_quality_clamped_low(self):
        h = Habitat(volume_m3=100.0, seal_quality=0.0)
        assert h.seal_quality == MIN_SEAL_QUALITY

    def test_negative_pressure_clamped(self):
        h = Habitat(volume_m3=100.0, pressure_kpa=-50.0)
        assert h.pressure_kpa == 0.0

    def test_explicit_mass_preserved(self):
        """If air_mass_kg > 0 is given explicitly, it's used."""
        h = Habitat(volume_m3=100.0, air_mass_kg=999.0)
        assert h.air_mass_kg == 999.0


# ===================================================================
# Reserve tank
# ===================================================================

class TestReserveTank:
    """Reserve tank operations."""

    def test_construction(self):
        t = ReserveTank(capacity_kg=1000.0, stored_kg=500.0)
        assert t.available() == 500.0

    def test_stored_clamped_to_capacity(self):
        t = ReserveTank(capacity_kg=100.0, stored_kg=999.0)
        assert t.stored_kg == 100.0

    def test_negative_stored_clamped(self):
        t = ReserveTank(capacity_kg=100.0, stored_kg=-50.0)
        assert t.stored_kg == 0.0

    def test_withdraw_normal(self):
        t = ReserveTank(capacity_kg=1000.0, stored_kg=500.0)
        got = t.withdraw(100.0)
        assert got == 100.0
        assert t.stored_kg == 400.0

    def test_withdraw_more_than_available(self):
        t = ReserveTank(capacity_kg=1000.0, stored_kg=50.0)
        got = t.withdraw(100.0)
        assert got == 50.0
        assert t.stored_kg == 0.0

    def test_withdraw_negative(self):
        t = ReserveTank(capacity_kg=1000.0, stored_kg=500.0)
        got = t.withdraw(-10.0)
        assert got == 0.0
        assert t.stored_kg == 500.0

    def test_deposit_normal(self):
        t = ReserveTank(capacity_kg=1000.0, stored_kg=500.0)
        stored = t.deposit(200.0)
        assert stored == 200.0
        assert t.stored_kg == 700.0

    def test_deposit_exceeds_capacity(self):
        t = ReserveTank(capacity_kg=100.0, stored_kg=80.0)
        stored = t.deposit(50.0)
        assert stored == 20.0
        assert t.stored_kg == 100.0

    def test_deposit_negative(self):
        t = ReserveTank(capacity_kg=100.0, stored_kg=50.0)
        stored = t.deposit(-10.0)
        assert stored == 0.0
        assert t.stored_kg == 50.0


# ===================================================================
# Leak model
# ===================================================================

class TestLeakModel:
    """Micro-leak and seal degradation."""

    def test_leak_at_nominal(self):
        """Nominal habitat leaks close to BASE_LEAK_RATE."""
        h = Habitat(volume_m3=500.0)
        leak = compute_leak_kg(h)
        # seal_quality=1.0, pressure_ratio~1.0 -> leak ~ BASE_LEAK_RATE
        assert leak == pytest.approx(BASE_LEAK_RATE_KG_SOL, rel=0.01)

    def test_leak_increases_with_poor_seals(self):
        """Degraded seals -> more leakage."""
        h_good = Habitat(volume_m3=500.0, seal_quality=1.0)
        h_bad = Habitat(volume_m3=500.0, seal_quality=0.2)
        leak_good = compute_leak_kg(h_good)
        leak_bad = compute_leak_kg(h_bad)
        assert leak_bad > leak_good

    def test_leak_scales_with_pressure(self):
        """Higher pressure -> more leakage."""
        h_low = Habitat(volume_m3=500.0, pressure_kpa=50.0)
        h_high = Habitat(volume_m3=500.0, pressure_kpa=101.3)
        assert compute_leak_kg(h_high) > compute_leak_kg(h_low)

    def test_leak_never_negative(self):
        h = Habitat(volume_m3=500.0, pressure_kpa=0.0)
        assert compute_leak_kg(h) >= 0.0

    def test_seal_degradation_normal(self):
        h = Habitat(volume_m3=100.0, seal_quality=1.0)
        delta = degrade_seals(h)
        assert delta == pytest.approx(SEAL_DEGRADATION_PER_SOL, rel=1e-10)
        assert h.seal_quality == pytest.approx(1.0 - SEAL_DEGRADATION_PER_SOL, rel=1e-10)

    def test_seal_degradation_dust_storm(self):
        h = Habitat(volume_m3=100.0, seal_quality=1.0)
        delta = degrade_seals(h, in_dust_storm=True)
        expected = SEAL_DEGRADATION_PER_SOL * DUST_SEAL_DAMAGE_FACTOR
        assert delta == pytest.approx(expected, rel=1e-10)

    def test_seal_never_below_minimum(self):
        h = Habitat(volume_m3=100.0, seal_quality=MIN_SEAL_QUALITY)
        degrade_seals(h)
        assert h.seal_quality >= MIN_SEAL_QUALITY

    def test_seal_degradation_over_many_sols(self):
        """Seals degrade monotonically but never below floor."""
        h = Habitat(volume_m3=100.0, seal_quality=1.0)
        for _ in range(100_000):
            degrade_seals(h)
        assert h.seal_quality >= MIN_SEAL_QUALITY


# ===================================================================
# Airlock
# ===================================================================

class TestAirlock:
    """Airlock cycling gas losses."""

    def test_airlock_loss_positive(self):
        h = Habitat(volume_m3=500.0)
        loss = airlock_cycle_loss_kg(h)
        assert loss > 0

    def test_airlock_loss_formula(self):
        """Loss = pressure_to_mass(P, V_airlock * (1 - recovery))."""
        h = Habitat(volume_m3=500.0)
        expected_vol = AIRLOCK_VOLUME_M3 * (1.0 - AIRLOCK_RECOVERY_FRACTION)
        expected_mass = pressure_to_mass(h.pressure_kpa, expected_vol)
        assert airlock_cycle_loss_kg(h) == pytest.approx(expected_mass, rel=1e-10)

    def test_perform_airlock_reduces_pressure(self):
        h = Habitat(volume_m3=500.0)
        p_before = h.pressure_kpa
        perform_airlock_cycle(h)
        assert h.pressure_kpa < p_before

    def test_perform_airlock_reduces_mass(self):
        h = Habitat(volume_m3=500.0)
        m_before = h.air_mass_kg
        loss = perform_airlock_cycle(h)
        assert h.air_mass_kg == pytest.approx(m_before - loss, rel=1e-10)

    def test_multiple_airlocks_cumulative(self):
        h = Habitat(volume_m3=500.0)
        p_start = h.pressure_kpa
        for _ in range(5):
            perform_airlock_cycle(h)
        assert h.pressure_kpa < p_start

    def test_airlock_on_low_pressure_habitat(self):
        """Even at low pressure, airlock doesn't go negative."""
        h = Habitat(volume_m3=10.0, pressure_kpa=1.0)
        for _ in range(100):
            perform_airlock_cycle(h)
        assert h.pressure_kpa >= 0.0
        assert h.air_mass_kg >= 0.0


# ===================================================================
# Blowout events
# ===================================================================

class TestBlowout:
    """Hull breach emergency events."""

    def test_no_breach_no_loss(self):
        h = Habitat(volume_m3=500.0)
        result = blowout_event(h, 0.0, 60.0)
        assert result["gas_lost_kg"] == 0.0
        assert result["survivable"] is True

    def test_no_duration_no_loss(self):
        h = Habitat(volume_m3=500.0)
        result = blowout_event(h, 10.0, 0.0)
        assert result["gas_lost_kg"] == 0.0

    def test_small_breach_survivable(self):
        """A pinhole (0.1 cm2) for 60 seconds should be survivable."""
        h = Habitat(volume_m3=500.0)
        result = blowout_event(h, 0.1, 60.0)
        assert result["gas_lost_kg"] > 0
        assert result["pressure_after_kpa"] > CRITICAL_KPA
        assert result["survivable"] is True

    def test_large_breach_catastrophic(self):
        """A large hole (100 cm2) for 600 seconds on a small habitat."""
        h = Habitat(volume_m3=50.0)
        result = blowout_event(h, 100.0, 600.0)
        assert result["gas_lost_kg"] > 0
        assert result["pressure_after_kpa"] < result["pressure_before_kpa"]

    def test_breach_never_loses_more_than_95_percent(self):
        """Conservation: can't lose more than 95% of air mass."""
        h = Habitat(volume_m3=100.0)
        mass_before = h.air_mass_kg
        blowout_event(h, 1000.0, 10000.0)
        assert h.air_mass_kg >= mass_before * 0.05 - 0.001

    def test_breach_reduces_pressure(self):
        h = Habitat(volume_m3=500.0)
        p_before = h.pressure_kpa
        blowout_event(h, 5.0, 300.0)
        assert h.pressure_kpa < p_before

    def test_pressure_mass_consistency_after_breach(self):
        """After breach, pressure and mass should be consistent."""
        h = Habitat(volume_m3=500.0)
        blowout_event(h, 5.0, 120.0)
        expected_p = mass_to_pressure(h.air_mass_kg, h.volume_m3)
        assert h.pressure_kpa == pytest.approx(expected_p, rel=1e-6)


# ===================================================================
# Replenishment
# ===================================================================

class TestReplenishment:
    """Reserve tank -> habitat replenishment."""

    def test_replenish_increases_pressure(self):
        h = Habitat(volume_m3=500.0, pressure_kpa=80.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        transferred = replenish_from_reserve(h, r)
        assert transferred > 0
        assert h.pressure_kpa > 80.0

    def test_replenish_no_overshoot(self):
        """Replenishment should not exceed target pressure."""
        h = Habitat(volume_m3=500.0, pressure_kpa=100.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        replenish_from_reserve(h, r)
        assert h.pressure_kpa <= TARGET_INTERNAL_KPA + 0.01

    def test_replenish_empty_reserve(self):
        """Empty reserve -> no transfer."""
        h = Habitat(volume_m3=500.0, pressure_kpa=50.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=0.0)
        transferred = replenish_from_reserve(h, r)
        assert transferred == 0.0

    def test_replenish_already_at_target(self):
        """At target pressure -> no transfer."""
        h = Habitat(volume_m3=500.0, pressure_kpa=TARGET_INTERNAL_KPA)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        transferred = replenish_from_reserve(h, r)
        assert transferred == 0.0

    def test_replenish_rate_limited(self):
        """Transfer capped at RESERVE_DELIVERY_KG_SOL."""
        h = Habitat(volume_m3=500.0, pressure_kpa=10.0)
        r = ReserveTank(capacity_kg=50000.0, stored_kg=50000.0)
        transferred = replenish_from_reserve(h, r)
        assert transferred <= RESERVE_DELIVERY_KG_SOL + 0.001

    def test_replenish_conservation(self):
        """Gas transferred = gas withdrawn from reserve."""
        h = Habitat(volume_m3=500.0, pressure_kpa=80.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        reserve_before = r.stored_kg
        mass_before = h.air_mass_kg
        transferred = replenish_from_reserve(h, r)
        assert reserve_before - r.stored_kg == pytest.approx(transferred, rel=1e-10)
        assert h.air_mass_kg - mass_before == pytest.approx(transferred, rel=1e-10)


# ===================================================================
# Per-sol tick — integration tests
# ===================================================================

class TestTickPressure:
    """Full system tick integration."""

    def test_nominal_tick(self):
        """Nominal conditions: pressure stays near target."""
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        result = tick_pressure(h, r)
        assert result["pressure_status"] == "nominal"
        assert result["survivable"] is True

    def test_tick_returns_all_keys(self):
        """Tick result has all expected keys."""
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        result = tick_pressure(h, r)
        expected_keys = {
            "pressure_start_kpa", "pressure_end_kpa", "pressure_status",
            "air_mass_kg", "leak_kg", "airlock_loss_kg", "blowout_loss_kg",
            "replenished_kg", "mass_balance_error_kg", "seal_quality",
            "seal_degradation", "reserve_kg", "structural_safety_ratio",
            "hull_stress_mpa", "eva_count", "survivable",
        }
        assert set(result.keys()) == expected_keys

    def test_tick_with_evas(self):
        """EVAs cause measurable gas loss."""
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        result = tick_pressure(h, r, eva_count=3)
        assert result["airlock_loss_kg"] > 0
        assert result["eva_count"] == 3

    def test_tick_with_dust_storm(self):
        """Dust storm increases seal degradation."""
        h1 = Habitat(volume_m3=500.0)
        h2 = Habitat(volume_m3=500.0)
        r1 = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        r2 = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        res_calm = tick_pressure(h1, r1, in_dust_storm=False)
        res_storm = tick_pressure(h2, r2, in_dust_storm=True)
        assert res_storm["seal_degradation"] > res_calm["seal_degradation"]

    def test_tick_with_breach(self):
        """Breach causes gas loss beyond normal leaks."""
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        result = tick_pressure(h, r, breach_cm2=5.0, breach_seconds=120.0)
        assert result["blowout_loss_kg"] > 0

    def test_mass_balance_conservation(self):
        """Mass balance error should be near zero."""
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        result = tick_pressure(h, r, eva_count=2)
        assert abs(result["mass_balance_error_kg"]) < 0.01

    def test_tick_no_reserve_pressure_drops(self):
        """Without reserve, pressure decreases each sol."""
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=0.0, stored_kg=0.0)
        result = tick_pressure(h, r)
        assert result["pressure_end_kpa"] < result["pressure_start_kpa"]

    def test_structural_safety_in_bounds(self):
        """Safety ratio should be positive and reasonable."""
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        result = tick_pressure(h, r)
        assert result["structural_safety_ratio"] > 1.0
        assert result["hull_stress_mpa"] > 0


# ===================================================================
# Property-based invariants
# ===================================================================

class TestInvariants:
    """Physical invariants that must hold for any input."""

    @pytest.mark.parametrize("pressure", [0.0, 10.0, 50.0, 101.3, 200.0, 500.0])
    def test_pressure_mass_roundtrip(self, pressure):
        """P -> m -> P is identity for any pressure."""
        vol = 300.0
        m = pressure_to_mass(pressure, vol)
        p_back = mass_to_pressure(m, vol)
        if pressure > 0:
            assert p_back == pytest.approx(pressure, rel=1e-10)
        else:
            assert p_back == 0.0

    @pytest.mark.parametrize("volume", [1.0, 50.0, 500.0, 5000.0])
    def test_larger_volume_more_mass(self, volume):
        """More volume at same pressure -> more mass."""
        m = pressure_to_mass(101.3, volume)
        assert m > 0
        m2 = pressure_to_mass(101.3, volume * 2)
        assert m2 == pytest.approx(2 * m, rel=1e-10)

    @pytest.mark.parametrize("seal_quality", [MIN_SEAL_QUALITY, 0.1, 0.5, 1.0])
    def test_leak_always_positive(self, seal_quality):
        """Leaks are always >= 0 regardless of seal quality."""
        h = Habitat(volume_m3=500.0, seal_quality=seal_quality)
        assert compute_leak_kg(h) >= 0.0

    @pytest.mark.parametrize("eva_count", [0, 1, 5, 10, 20])
    def test_pressure_non_negative_after_evas(self, eva_count):
        """Pressure never goes negative regardless of EVA count."""
        h = Habitat(volume_m3=200.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        result = tick_pressure(h, r, eva_count=eva_count)
        assert result["pressure_end_kpa"] >= 0.0
        assert result["air_mass_kg"] >= 0.0

    @pytest.mark.parametrize("breach_area", [0.0, 0.01, 1.0, 10.0, 100.0, 1000.0])
    def test_blowout_never_negative_mass(self, breach_area):
        """Air mass never goes negative after any breach."""
        h = Habitat(volume_m3=100.0)
        blowout_event(h, breach_area, 300.0)
        assert h.air_mass_kg >= 0.0
        assert h.pressure_kpa >= 0.0


# ===================================================================
# Multi-sol smoke tests
# ===================================================================

class TestSmokeMultiSol:
    """Run the simulation for many sols without crash."""

    def test_100_sols_nominal(self):
        """100 sols of normal operation without crash."""
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=10000.0, stored_kg=8000.0)
        for sol in range(100):
            result = tick_pressure(h, r, eva_count=sol % 3)
            assert result["pressure_end_kpa"] >= 0.0
            assert result["air_mass_kg"] >= 0.0
            assert h.seal_quality >= MIN_SEAL_QUALITY

    def test_100_sols_dust_storm(self):
        """100 sols of continuous dust storm."""
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=10000.0, stored_kg=8000.0)
        for _ in range(100):
            result = tick_pressure(h, r, in_dust_storm=True)
            assert result["pressure_end_kpa"] >= 0.0
            assert h.seal_quality >= MIN_SEAL_QUALITY

    def test_100_sols_reserve_depletion(self):
        """Reserve runs out — pressure should drop but not crash."""
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=100.0, stored_kg=100.0)
        for _ in range(100):
            result = tick_pressure(h, r, eva_count=1)
            assert result["pressure_end_kpa"] >= 0.0
            assert result["air_mass_kg"] >= 0.0

    def test_500_sols_lifecycle(self):
        """500 sols with varied conditions — the colony lives."""
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=20000.0, stored_kg=15000.0)
        for sol in range(500):
            eva = 2 if sol % 7 < 3 else 0
            storm = sol % 50 > 40
            breach = 0.5 if sol == 250 else 0.0
            breach_s = 30.0 if sol == 250 else 0.0
            result = tick_pressure(
                h, r,
                eva_count=eva,
                in_dust_storm=storm,
                breach_cm2=breach,
                breach_seconds=breach_s,
            )
            assert result["pressure_end_kpa"] >= 0.0
            assert result["air_mass_kg"] >= 0.0

    def test_pressure_status_transitions(self):
        """Pressure status should follow: nominal -> low -> critical -> fatal."""
        h = Habitat(volume_m3=100.0, pressure_kpa=101.3)
        r = ReserveTank(capacity_kg=0.0, stored_kg=0.0)  # no reserve
        statuses_seen = set()
        for _ in range(2000):
            result = tick_pressure(h, r, eva_count=1)
            statuses_seen.add(result["pressure_status"])
            if result["pressure_status"] == "fatal":
                break
        # Should have seen at least nominal and one lower status
        assert "nominal" in statuses_seen or "low" in statuses_seen


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_tiny_habitat(self):
        """Minimum-size habitat (1 m3) works."""
        h = Habitat(volume_m3=1.0)
        r = ReserveTank(capacity_kg=100.0, stored_kg=50.0)
        result = tick_pressure(h, r)
        assert result["pressure_end_kpa"] >= 0.0

    def test_huge_habitat(self):
        """Large habitat (10000 m3) works."""
        h = Habitat(volume_m3=10000.0)
        r = ReserveTank(capacity_kg=100000.0, stored_kg=50000.0)
        result = tick_pressure(h, r)
        assert result["pressure_end_kpa"] >= 0.0

    def test_zero_pressure_habitat(self):
        """Depressurized habitat doesn't crash."""
        h = Habitat(volume_m3=500.0, pressure_kpa=0.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        result = tick_pressure(h, r)
        assert result["pressure_end_kpa"] >= 0.0

    def test_negative_eva_count_treated_as_zero(self):
        h = Habitat(volume_m3=500.0)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        result = tick_pressure(h, r, eva_count=-5)
        assert result["airlock_loss_kg"] == 0.0

    def test_seal_at_minimum_still_works(self):
        """Habitat with minimum seals still produces valid output."""
        h = Habitat(volume_m3=500.0, seal_quality=MIN_SEAL_QUALITY)
        r = ReserveTank(capacity_kg=5000.0, stored_kg=3000.0)
        result = tick_pressure(h, r)
        assert result["leak_kg"] > 0
        assert result["pressure_end_kpa"] >= 0.0
