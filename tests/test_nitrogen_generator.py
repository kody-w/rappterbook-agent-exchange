"""Tests for nitrogen_generator.py — Mars Habitat Nitrogen Extraction."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nitrogen_generator import (
    NitrogenState, NitrogenTickResult,
    mars_air_density_kg_m3, n2_in_mars_air_kg, compression_energy_kwh,
    distillation_energy_kwh, n2_after_freeze_out, co2_byproduct_kg,
    n2_loss_kg, injection_needed_kg, kpa_from_mass, assess_alert,
    tick_nitrogen, create_nitrogen_system,
    MARS_SURFACE_PRESSURE_PA, MARS_N2_FRACTION, MARS_CO2_FRACTION,
    N2_MOLAR_MASS, CO2_MOLAR_MASS, COMPRESSOR_OUTLET_PA,
    DISTILLATION_N2_RECOVERY, TANK_CAPACITY_KG, HABITAT_VOLUME_M3,
    TARGET_N2_KPA, HABITAT_N2_TOTAL_KG, AIRLOCK_N2_LOSS_KG,
    SEAL_LEAK_FRACTION, EMERGENCY_LEAK_KG, N2_LOW_WARNING_KPA,
    N2_CRITICAL_KPA, MAX_INTAKE_M3_SOL, CRYO_KWH_PER_KG_N2,
)


# ── NitrogenState clamping ─────────────────────────────────────────

class TestNitrogenState:
    def test_defaults(self):
        s = NitrogenState()
        assert s.tank_kg == 200.0
        assert s.hab_n2_kpa == TARGET_N2_KPA
        assert s.alert == "nominal"

    def test_tank_clamped_high(self):
        s = NitrogenState(tank_kg=9999.0)
        assert s.tank_kg == TANK_CAPACITY_KG

    def test_tank_clamped_low(self):
        s = NitrogenState(tank_kg=-50.0)
        assert s.tank_kg == 0.0

    def test_negative_pressure_clamped(self):
        s = NitrogenState(hab_n2_kpa=-10.0)
        assert s.hab_n2_kpa == 0.0

    def test_negative_mass_clamped(self):
        s = NitrogenState(hab_n2_mass_kg=-5.0)
        assert s.hab_n2_mass_kg == 0.0

    def test_intake_rate_clamped(self):
        s = NitrogenState(intake_rate_m3_sol=99999.0)
        assert s.intake_rate_m3_sol == MAX_INTAKE_M3_SOL
        s2 = NitrogenState(intake_rate_m3_sol=-100.0)
        assert s2.intake_rate_m3_sol == 0.0


# ── mars_air_density ───────────────────────────────────────────────

class TestMarsAirDensity:
    def test_standard_conditions(self):
        rho = mars_air_density_kg_m3()
        # Mars air at 636 Pa, 210 K: ~0.016 kg/m³
        assert 0.01 < rho < 0.025

    def test_higher_pressure_denser(self):
        rho_low = mars_air_density_kg_m3(400.0)
        rho_high = mars_air_density_kg_m3(800.0)
        assert rho_high > rho_low

    def test_higher_temp_less_dense(self):
        rho_cold = mars_air_density_kg_m3(temperature_k=180.0)
        rho_warm = mars_air_density_kg_m3(temperature_k=280.0)
        assert rho_cold > rho_warm

    def test_always_positive(self):
        for p in [0.0, 100.0, 636.0, 1000.0]:
            for t in [100.0, 210.0, 300.0]:
                assert mars_air_density_kg_m3(p, t) >= 0.0

    def test_zero_pressure(self):
        assert mars_air_density_kg_m3(0.0) == 0.0


# ── n2_in_mars_air ─────────────────────────────────────────────────

class TestN2InMarsAir:
    def test_zero_intake(self):
        assert n2_in_mars_air_kg(0.0) == 0.0

    def test_positive_intake(self):
        result = n2_in_mars_air_kg(1000.0)
        assert result > 0.0

    def test_proportional(self):
        r1 = n2_in_mars_air_kg(1000.0)
        r2 = n2_in_mars_air_kg(2000.0)
        assert abs(r2 - 2.0 * r1) < 0.001

    def test_negative_clamped(self):
        assert n2_in_mars_air_kg(-500.0) == 0.0

    def test_n2_small_fraction_of_air(self):
        """N₂ should be a tiny fraction of total air mass."""
        rho = mars_air_density_kg_m3()
        total_mass = 1000.0 * rho
        n2_mass = n2_in_mars_air_kg(1000.0)
        assert n2_mass < total_mass * 0.05


# ── compression_energy ─────────────────────────────────────────────

class TestCompressionEnergy:
    def test_zero_intake(self):
        assert compression_energy_kwh(0.0) == 0.0

    def test_positive_for_real_intake(self):
        result = compression_energy_kwh(1000.0)
        assert result > 0.0

    def test_more_intake_more_energy(self):
        e1 = compression_energy_kwh(500.0)
        e2 = compression_energy_kwh(1000.0)
        assert e2 > e1

    def test_higher_ratio_more_energy(self):
        e_low = compression_energy_kwh(1000.0, p_out=100_000.0)
        e_high = compression_energy_kwh(1000.0, p_out=500_000.0)
        assert e_high > e_low

    def test_negative_clamped(self):
        assert compression_energy_kwh(-100.0) == 0.0

    def test_realistic_magnitude(self):
        """2000 m³ Mars air → should be order of 10-100 kWh."""
        result = compression_energy_kwh(2000.0)
        assert 1.0 < result < 200.0


# ── distillation_energy ────────────────────────────────────────────

class TestDistillationEnergy:
    def test_zero_mass(self):
        assert distillation_energy_kwh(0.0) == 0.0

    def test_known_rate(self):
        result = distillation_energy_kwh(10.0)
        assert abs(result - 10.0 * CRYO_KWH_PER_KG_N2) < 0.001

    def test_negative_clamped(self):
        assert distillation_energy_kwh(-5.0) == 0.0


# ── n2_after_freeze_out ───────────────────────────────────────────

class TestN2AfterFreezeOut:
    def test_zero_intake(self):
        assert n2_after_freeze_out(0.0) == 0.0

    def test_recovery_less_than_raw(self):
        raw = n2_in_mars_air_kg(1000.0)
        recovered = n2_after_freeze_out(1000.0)
        assert recovered < raw
        assert abs(recovered - raw * DISTILLATION_N2_RECOVERY) < 0.001

    def test_always_non_negative(self):
        for vol in [-100.0, 0.0, 500.0, 5000.0]:
            assert n2_after_freeze_out(vol) >= 0.0


# ── co2_byproduct ─────────────────────────────────────────────────

class TestCo2Byproduct:
    def test_zero_intake(self):
        assert co2_byproduct_kg(0.0) == 0.0

    def test_co2_dominates_air(self):
        """CO₂ byproduct should be much larger than N₂ extracted."""
        co2 = co2_byproduct_kg(1000.0)
        n2 = n2_after_freeze_out(1000.0)
        assert co2 > n2 * 10

    def test_always_non_negative(self):
        assert co2_byproduct_kg(-100.0) == 0.0


# ── n2_loss ────────────────────────────────────────────────────────

class TestN2Loss:
    def test_zero_cycles_still_leaks(self):
        loss = n2_loss_kg(0, SEAL_LEAK_FRACTION, 300.0)
        assert loss > 0.0

    def test_airlock_cycles_increase_loss(self):
        loss_0 = n2_loss_kg(0, SEAL_LEAK_FRACTION, 300.0)
        loss_4 = n2_loss_kg(4, SEAL_LEAK_FRACTION, 300.0)
        assert loss_4 > loss_0
        assert abs((loss_4 - loss_0) - 4 * AIRLOCK_N2_LOSS_KG) < 0.01

    def test_emergency_adds_loss(self):
        normal = n2_loss_kg(2, SEAL_LEAK_FRACTION, 300.0)
        emergency = n2_loss_kg(2, SEAL_LEAK_FRACTION, 300.0, emergency=True)
        assert abs((emergency - normal) - EMERGENCY_LEAK_KG) < 0.01

    def test_cannot_lose_more_than_exists(self):
        loss = n2_loss_kg(100, SEAL_LEAK_FRACTION, 5.0, emergency=True)
        assert loss <= 5.0

    def test_negative_cycles_clamped(self):
        loss = n2_loss_kg(-5, SEAL_LEAK_FRACTION, 300.0)
        expected = 300.0 * SEAL_LEAK_FRACTION
        assert abs(loss - expected) < 0.01

    def test_zero_hab_mass(self):
        assert n2_loss_kg(4, SEAL_LEAK_FRACTION, 0.0) == 0.0


# ── injection_needed ───────────────────────────────────────────────

class TestInjectionNeeded:
    def test_at_target_no_injection(self):
        assert injection_needed_kg(TARGET_N2_KPA) == 0.0

    def test_above_target_no_injection(self):
        assert injection_needed_kg(TARGET_N2_KPA + 5.0) == 0.0

    def test_below_target_needs_injection(self):
        needed = injection_needed_kg(TARGET_N2_KPA - 5.0)
        assert needed > 0.0

    def test_more_deficit_more_injection(self):
        small = injection_needed_kg(TARGET_N2_KPA - 2.0)
        large = injection_needed_kg(TARGET_N2_KPA - 10.0)
        assert large > small

    def test_physically_reasonable(self):
        """5 kPa deficit in 500 m³ should need ~25-35 kg N₂.
        mass = ΔP × V × M / (R × T) = 5000 × 500 × 0.028 / (8.314 × 293) ≈ 28.7 kg
        """
        needed = injection_needed_kg(TARGET_N2_KPA - 5.0)
        assert 20.0 < needed < 40.0


# ── kpa_from_mass ──────────────────────────────────────────────────

class TestKpaFromMass:
    def test_zero_mass(self):
        assert kpa_from_mass(0.0) == 0.0

    def test_round_trip(self):
        """kpa_from_mass should be close to TARGET_N2_KPA for HABITAT_N2_TOTAL_KG."""
        mass = HABITAT_N2_TOTAL_KG
        computed_kpa = kpa_from_mass(mass)
        assert abs(computed_kpa - TARGET_N2_KPA) < 2.0

    def test_more_mass_higher_pressure(self):
        p1 = kpa_from_mass(100.0)
        p2 = kpa_from_mass(200.0)
        assert p2 > p1

    def test_negative_clamped(self):
        assert kpa_from_mass(-10.0) == 0.0


# ── assess_alert ───────────────────────────────────────────────────

class TestAssessAlert:
    def test_nominal(self):
        assert assess_alert(TARGET_N2_KPA) == "nominal"
        assert assess_alert(50.0) == "nominal"

    def test_warning(self):
        assert assess_alert(N2_LOW_WARNING_KPA - 1.0) == "warning"
        assert assess_alert(40.0) == "warning"

    def test_critical(self):
        assert assess_alert(N2_CRITICAL_KPA - 1.0) == "critical"
        assert assess_alert(20.0) == "critical"
        assert assess_alert(0.0) == "critical"


# ── tick_nitrogen integration ──────────────────────────────────────

class TestTickNitrogen:
    def test_one_tick_nominal(self):
        state = create_nitrogen_system()
        state, result = tick_nitrogen(state)
        assert result.n2_lost_kg > 0.0
        assert result.n2_extracted_kg > 0.0
        assert result.total_energy_kwh > 0.0
        assert state.sols_running == 1

    def test_no_airlock_less_loss(self):
        s1 = create_nitrogen_system()
        s2 = create_nitrogen_system()
        s1, r1 = tick_nitrogen(s1, airlock_cycles=0)
        s2, r2 = tick_nitrogen(s2, airlock_cycles=8)
        assert r2.n2_lost_kg > r1.n2_lost_kg

    def test_emergency_increases_loss(self):
        s1 = create_nitrogen_system()
        s2 = create_nitrogen_system()
        s1, r1 = tick_nitrogen(s1, emergency=False)
        s2, r2 = tick_nitrogen(s2, emergency=True)
        assert r2.n2_lost_kg > r1.n2_lost_kg

    def test_power_limited_extraction(self):
        s1 = create_nitrogen_system()
        s2 = create_nitrogen_system()
        s1, r1 = tick_nitrogen(s1, power_available_kwh=100.0)
        s2, r2 = tick_nitrogen(s2, power_available_kwh=1.0)
        assert r2.n2_extracted_kg <= r1.n2_extracted_kg

    def test_tank_fills(self):
        state = create_nitrogen_system(tank_kg=0.0)
        for _ in range(10):
            state, _ = tick_nitrogen(state, airlock_cycles=0)
        assert state.tank_kg > 0.0

    def test_tank_never_exceeds_capacity(self):
        state = create_nitrogen_system(tank_kg=490.0)
        for _ in range(100):
            state, _ = tick_nitrogen(state, airlock_cycles=0)
        assert state.tank_kg <= TANK_CAPACITY_KG

    def test_hab_pressure_maintained(self):
        """With extraction running, N₂ pressure stays near target."""
        state = create_nitrogen_system()
        for _ in range(50):
            state, result = tick_nitrogen(state, airlock_cycles=4)
        assert abs(result.hab_n2_kpa - TARGET_N2_KPA) < 5.0

    def test_hab_degrades_without_power(self):
        """With no power, N₂ pressure drops over time."""
        state = create_nitrogen_system(tank_kg=0.0)
        for _ in range(100):
            state, result = tick_nitrogen(state, airlock_cycles=4,
                                          power_available_kwh=0.0)
        assert result.hab_n2_kpa < TARGET_N2_KPA

    def test_co2_byproduct_produced(self):
        state = create_nitrogen_system()
        state, result = tick_nitrogen(state)
        assert result.co2_byproduct_kg > 0.0

    def test_co2_much_larger_than_n2(self):
        state = create_nitrogen_system()
        state, result = tick_nitrogen(state)
        assert result.co2_byproduct_kg > result.n2_extracted_kg * 5

    def test_cumulative_tracking(self):
        state = create_nitrogen_system()
        for _ in range(20):
            state, _ = tick_nitrogen(state, airlock_cycles=2)
        assert state.total_extracted_kg > 0.0
        assert state.total_lost_kg > 0.0
        assert state.total_energy_kwh > 0.0
        assert state.sols_running == 20

    def test_alert_escalation_on_depletion(self):
        state = NitrogenState(tank_kg=0.0, hab_n2_mass_kg=50.0,
                              hab_n2_kpa=kpa_from_mass(50.0))
        for _ in range(200):
            state, result = tick_nitrogen(state, airlock_cycles=8,
                                          power_available_kwh=0.0)
        assert result.alert in ("warning", "critical")


# ── Physical invariants ────────────────────────────────────────────

class TestPhysicalInvariants:
    def test_mass_conservation_n2(self):
        """N₂ accounting: tank + hab = initial + extracted - lost."""
        state = create_nitrogen_system(tank_kg=200.0)
        initial_tank = state.tank_kg
        initial_hab = state.hab_n2_mass_kg
        total_extracted = 0.0
        total_lost = 0.0

        for _ in range(50):
            state, result = tick_nitrogen(state, airlock_cycles=3)
            total_extracted += result.n2_extracted_kg
            total_lost += result.n2_lost_kg

        expected = initial_tank + initial_hab + total_extracted - total_lost
        actual = state.tank_kg + state.hab_n2_mass_kg
        assert abs(actual - expected) < 1.0

    def test_extraction_bounded_by_intake(self):
        state = create_nitrogen_system()
        for _ in range(100):
            state, result = tick_nitrogen(state, airlock_cycles=2)
            max_possible = n2_in_mars_air_kg(state.intake_rate_m3_sol)
            assert result.n2_extracted_kg <= max_possible + 0.001

    def test_energy_always_non_negative(self):
        state = create_nitrogen_system()
        for _ in range(50):
            state, result = tick_nitrogen(state)
            assert result.total_energy_kwh >= 0.0
            assert result.compression_kwh >= 0.0
            assert result.cryo_kwh >= 0.0

    def test_tank_never_negative(self):
        state = create_nitrogen_system(tank_kg=1.0)
        for _ in range(200):
            state, _ = tick_nitrogen(state, airlock_cycles=10)
        assert state.tank_kg >= 0.0

    def test_hab_mass_never_negative(self):
        state = NitrogenState(tank_kg=0.0, hab_n2_mass_kg=10.0,
                              hab_n2_kpa=kpa_from_mass(10.0))
        for _ in range(500):
            state, _ = tick_nitrogen(state, airlock_cycles=8,
                                     power_available_kwh=0.0)
        assert state.hab_n2_mass_kg >= 0.0
        assert state.hab_n2_kpa >= 0.0

    def test_pressure_consistent_with_mass(self):
        state = create_nitrogen_system()
        for _ in range(30):
            state, _ = tick_nitrogen(state, airlock_cycles=4)
        computed_kpa = kpa_from_mass(state.hab_n2_mass_kg)
        assert abs(state.hab_n2_kpa - computed_kpa) < 0.1

    def test_long_run_no_crash(self):
        """1000 sols with varied inputs — no crash."""
        state = create_nitrogen_system()
        for sol in range(1000):
            cycles = sol % 8
            emergency = (sol % 200 == 199)
            power = 100.0 if sol % 50 != 49 else 0.0
            state, result = tick_nitrogen(state, airlock_cycles=cycles,
                                          emergency=emergency,
                                          power_available_kwh=power)
        assert state.sols_running == 1000
        assert state.total_extracted_kg > 0.0
        assert state.tank_kg >= 0.0
        assert state.hab_n2_mass_kg >= 0.0

    def test_injection_never_exceeds_tank(self):
        state = NitrogenState(tank_kg=2.0, hab_n2_kpa=30.0,
                              hab_n2_mass_kg=170.0)
        state, result = tick_nitrogen(state, power_available_kwh=0.0,
                                      airlock_cycles=0)
        assert result.n2_injected_kg <= 2.0 + 0.001


# ── create_nitrogen_system ─────────────────────────────────────────

class TestCreateNitrogenSystem:
    def test_default(self):
        state = create_nitrogen_system()
        assert state.tank_kg == 200.0
        assert state.alert == "nominal"

    def test_custom_tank(self):
        state = create_nitrogen_system(tank_kg=100.0)
        assert state.tank_kg == 100.0

    def test_clamped_high(self):
        state = create_nitrogen_system(tank_kg=9999.0)
        assert state.tank_kg == TANK_CAPACITY_KG

    def test_clamped_low(self):
        state = create_nitrogen_system(tank_kg=-50.0)
        assert state.tank_kg == 0.0
