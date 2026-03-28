"""test_magnetic_shield.py -- 130+ tests for the Mars Habitat Magnetic Shield.

Tests cover:
- Pure physics functions (solenoid field, Larmor radius, energy cutoff, etc.)
- Conservation laws and physical bounds
- Simulation tick behavior (ramp-up, steady state, quench, recovery)
- Edge cases (zero field, zero coolant, extreme temperatures)
- Multi-sol simulation with SPE events
- Property-based invariants
"""
from __future__ import annotations

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from magnetic_shield import (
    MU_0, PROTON_MASS_KG, PROTON_CHARGE_C, SPEED_OF_LIGHT,
    PROTON_REST_ENERGY_GEV, MARS_AMBIENT_TEMP_K, SECONDS_PER_SOL,
    LN2_BOILING_POINT_K, LN2_LATENT_HEAT_KJ_KG, YBCO_TC_K,
    GCR_SURFACE_MSV_SOL, MAX_DB_DT_T_PER_SOL, MAX_OPERATING_FIELD_T,
    DEFAULT_COIL_RADIUS_M, DEFAULT_COIL_LENGTH_M, DEFAULT_NUM_TURNS,
    DEFAULT_OPERATING_CURRENT_A, DEFAULT_COOLANT_KG,
    solenoid_field_t, magnetic_dipole_moment, stored_energy_mj,
    coil_volume_m3, larmor_radius_m, proton_larmor_radius_m,
    energy_cutoff_gev, gcr_deflection_fraction, spe_deflection_fraction,
    heat_leak_kw, joint_heating_kw, cryocooler_power_kw,
    coolant_boiloff_kg_per_sol, wire_mass_kg, quench_temperature_rise_k,
    MagneticShieldConfig, ShieldState, TickResult,
    create_shield, tick, run_simulation,
)


# ===========================================================================
# 1. Solenoid field
# ===========================================================================

class TestSolenoidField:
    def test_basic_field(self):
        b = solenoid_field_t(1000, 10.0, 1.0)
        expected = MU_0 * 1000 * 10.0 / 1.0
        assert abs(b - expected) < 1e-10

    def test_zero_length(self):
        assert solenoid_field_t(100, 10.0, 0.0) == 0.0

    def test_zero_turns(self):
        assert solenoid_field_t(0, 10.0, 1.0) == 0.0

    def test_zero_current(self):
        assert solenoid_field_t(100, 0.0, 1.0) == 0.0

    def test_field_proportional_to_current(self):
        b1 = solenoid_field_t(100, 10.0, 1.0)
        b2 = solenoid_field_t(100, 20.0, 1.0)
        assert abs(b2 / b1 - 2.0) < 1e-10

    def test_field_proportional_to_turns(self):
        b1 = solenoid_field_t(100, 10.0, 1.0)
        b2 = solenoid_field_t(200, 10.0, 1.0)
        assert abs(b2 / b1 - 2.0) < 1e-10

    def test_field_inversely_proportional_to_length(self):
        b1 = solenoid_field_t(100, 10.0, 1.0)
        b2 = solenoid_field_t(100, 10.0, 2.0)
        assert abs(b1 / b2 - 2.0) < 1e-10

    def test_negative_turns(self):
        assert solenoid_field_t(-1, 10.0, 1.0) == 0.0

    def test_default_config_field(self):
        b = solenoid_field_t(DEFAULT_NUM_TURNS, DEFAULT_OPERATING_CURRENT_A,
                             DEFAULT_COIL_LENGTH_M)
        assert 0.4 < b < 0.6


# ===========================================================================
# 2. Magnetic dipole moment
# ===========================================================================

class TestDipoleMoment:
    def test_basic_moment(self):
        m = magnetic_dipole_moment(100, 10.0, 1.0)
        assert abs(m - 100 * 10.0 * math.pi) < 1e-10

    def test_proportional_to_radius_squared(self):
        m1 = magnetic_dipole_moment(100, 10.0, 1.0)
        m2 = magnetic_dipole_moment(100, 10.0, 2.0)
        assert abs(m2 / m1 - 4.0) < 1e-10

    def test_zero_current_moment(self):
        assert magnetic_dipole_moment(100, 0.0, 1.0) == 0.0

    def test_zero_radius_moment(self):
        assert magnetic_dipole_moment(100, 10.0, 0.0) == 0.0


# ===========================================================================
# 3. Stored energy
# ===========================================================================

class TestStoredEnergy:
    def test_stored_energy_formula(self):
        b, v = 1.0, 10.0
        expected = (b ** 2 * v) / (2.0 * MU_0) / 1.0e6
        assert abs(stored_energy_mj(b, v) - expected) < 1e-6

    def test_zero_field(self):
        assert stored_energy_mj(0.0, 10.0) == 0.0

    def test_zero_volume(self):
        assert stored_energy_mj(1.0, 0.0) == 0.0

    def test_energy_scales_with_b_squared(self):
        e1 = stored_energy_mj(1.0, 10.0)
        e2 = stored_energy_mj(2.0, 10.0)
        assert abs(e2 / e1 - 4.0) < 1e-10

    def test_energy_positive(self):
        assert stored_energy_mj(0.5, 5.0) > 0.0

    def test_negative_field_zero(self):
        assert stored_energy_mj(-1.0, 10.0) == 0.0


# ===========================================================================
# 4. Coil volume
# ===========================================================================

class TestCoilVolume:
    def test_basic_volume(self):
        v = coil_volume_m3(2.0, 3.0)
        assert abs(v - math.pi * 4.0 * 3.0) < 1e-10

    def test_zero_radius(self):
        assert coil_volume_m3(0.0, 3.0) == 0.0

    def test_zero_length(self):
        assert coil_volume_m3(2.0, 0.0) == 0.0

    def test_negative_radius(self):
        assert coil_volume_m3(-1.0, 3.0) == 0.0


# ===========================================================================
# 5. Larmor radius
# ===========================================================================

class TestLarmorRadius:
    def test_basic_larmor(self):
        r = larmor_radius_m(PROTON_MASS_KG, 1.0e6, PROTON_CHARGE_C, 1.0)
        expected = PROTON_MASS_KG * 1.0e6 / (PROTON_CHARGE_C * 1.0)
        assert abs(r - expected) < 1e-20

    def test_zero_field(self):
        r = larmor_radius_m(PROTON_MASS_KG, 1.0e6, PROTON_CHARGE_C, 0.0)
        assert r == float("inf")

    def test_zero_charge(self):
        r = larmor_radius_m(PROTON_MASS_KG, 1.0e6, 0.0, 1.0)
        assert r == float("inf")

    def test_larmor_inversely_proportional_to_field(self):
        r1 = larmor_radius_m(PROTON_MASS_KG, 1e6, PROTON_CHARGE_C, 0.5)
        r2 = larmor_radius_m(PROTON_MASS_KG, 1e6, PROTON_CHARGE_C, 1.0)
        assert abs(r1 / r2 - 2.0) < 1e-10

    def test_larmor_proportional_to_velocity(self):
        r1 = larmor_radius_m(PROTON_MASS_KG, 1e6, PROTON_CHARGE_C, 1.0)
        r2 = larmor_radius_m(PROTON_MASS_KG, 2e6, PROTON_CHARGE_C, 1.0)
        assert abs(r2 / r1 - 2.0) < 1e-10


# ===========================================================================
# 6. Relativistic proton Larmor radius
# ===========================================================================

class TestProtonLarmorRelativistic:
    def test_low_energy(self):
        r = proton_larmor_radius_m(0.001, 1.0)
        assert 0.0 < r < 1.0

    def test_1gev_proton(self):
        r = proton_larmor_radius_m(1.0, 1.0)
        assert 3.0 < r < 10.0

    def test_zero_energy(self):
        assert proton_larmor_radius_m(0.0, 1.0) == float("inf")

    def test_zero_field(self):
        assert proton_larmor_radius_m(1.0, 0.0) == float("inf")

    def test_higher_energy_larger_radius(self):
        r1 = proton_larmor_radius_m(0.5, 1.0)
        r2 = proton_larmor_radius_m(2.0, 1.0)
        assert r2 > r1

    def test_higher_field_smaller_radius(self):
        r1 = proton_larmor_radius_m(1.0, 0.5)
        r2 = proton_larmor_radius_m(1.0, 2.0)
        assert r2 < r1


# ===========================================================================
# 7. Energy cutoff
# ===========================================================================

class TestEnergyCutoff:
    def test_weak_field_low_cutoff(self):
        e = energy_cutoff_gev(0.1, 5.0)
        assert 0.005 < e < 0.05  # ~12 MeV

    def test_zero_field(self):
        assert energy_cutoff_gev(0.0, 5.0) == 0.0

    def test_zero_radius(self):
        assert energy_cutoff_gev(0.1, 0.0) == 0.0

    def test_cutoff_increases_with_field(self):
        e1 = energy_cutoff_gev(0.05, 5.0)
        e2 = energy_cutoff_gev(0.1, 5.0)
        assert e2 > e1

    def test_cutoff_increases_with_radius(self):
        e1 = energy_cutoff_gev(0.1, 3.0)
        e2 = energy_cutoff_gev(0.1, 5.0)
        assert e2 > e1

    def test_strong_field_high_cutoff(self):
        e = energy_cutoff_gev(1.0, 10.0)
        assert e > 1.0

    def test_consistency_with_larmor(self):
        field = 0.5
        radius = 5.0
        e_cut = energy_cutoff_gev(field, radius)
        r_l = proton_larmor_radius_m(e_cut, field)
        assert abs(r_l - radius) / radius < 0.01


# ===========================================================================
# 8. GCR deflection fraction
# ===========================================================================

class TestGCRDeflection:
    def test_zero_cutoff(self):
        assert gcr_deflection_fraction(0.0) == 0.0

    def test_negative_cutoff(self):
        assert gcr_deflection_fraction(-1.0) == 0.0

    def test_very_high_cutoff(self):
        f = gcr_deflection_fraction(100.0)
        assert f == 0.99

    def test_monotonic_increase(self):
        prev = 0.0
        for e in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            f = gcr_deflection_fraction(e)
            assert f >= prev
            prev = f

    def test_bounded_0_1(self):
        for e in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
            f = gcr_deflection_fraction(e)
            assert 0.0 <= f <= 1.0

    def test_1gev_cutoff(self):
        f = gcr_deflection_fraction(1.0)
        assert 0.65 < f < 0.75

    def test_low_cutoff_low_deflection(self):
        f = gcr_deflection_fraction(0.05)
        assert f < 0.10


# ===========================================================================
# 9. SPE deflection fraction
# ===========================================================================

class TestSPEDeflection:
    def test_zero_field(self):
        assert spe_deflection_fraction(0.0, 5.0) == 0.0

    def test_zero_radius(self):
        assert spe_deflection_fraction(0.1, 0.0) == 0.0

    def test_moderate_field_high_deflection(self):
        f = spe_deflection_fraction(0.5, 5.0)
        assert f > 0.85

    def test_strong_field(self):
        f = spe_deflection_fraction(1.0, 5.0)
        assert f >= 0.99

    def test_bounded_0_1(self):
        for b in [0.0, 0.001, 0.01, 0.1, 1.0]:
            f = spe_deflection_fraction(b, 5.0)
            assert 0.0 <= f <= 1.0

    def test_monotonic_with_field(self):
        prev = 0.0
        for b in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
            f = spe_deflection_fraction(b, 5.0)
            assert f >= prev
            prev = f


# ===========================================================================
# 10. Heat leak
# ===========================================================================

class TestHeatLeak:
    def test_basic_heat_leak(self):
        h = heat_leak_kw(80.0, 210.0, 77.0)
        assert h > 0.0

    def test_no_gradient_no_leak(self):
        h = heat_leak_kw(80.0, 77.0, 77.0)
        assert h == 0.0

    def test_cold_ambient_no_leak(self):
        h = heat_leak_kw(80.0, 50.0, 77.0)
        assert h == 0.0

    def test_larger_area_more_leak(self):
        h1 = heat_leak_kw(40.0, 210.0, 77.0)
        h2 = heat_leak_kw(80.0, 210.0, 77.0)
        assert h2 > h1

    def test_higher_ambient_more_leak(self):
        h1 = heat_leak_kw(80.0, 180.0, 77.0)
        h2 = heat_leak_kw(80.0, 250.0, 77.0)
        assert h2 > h1


# ===========================================================================
# 11. Joint heating
# ===========================================================================

class TestJointHeating:
    def test_zero_current(self):
        assert joint_heating_kw(0.0) == 0.0

    def test_positive_with_current(self):
        assert joint_heating_kw(200.0) > 0.0

    def test_scales_with_current_squared(self):
        h1 = joint_heating_kw(100.0)
        h2 = joint_heating_kw(200.0)
        assert abs(h2 / h1 - 4.0) < 1e-10


# ===========================================================================
# 12. Cryocooler power
# ===========================================================================

class TestCryocoolerPower:
    def test_basic_power(self):
        p = cryocooler_power_kw(1.0, 0.2)
        assert abs(p - 5.0) < 1e-10

    def test_zero_heat(self):
        assert cryocooler_power_kw(0.0) == 0.0

    def test_zero_cop(self):
        assert cryocooler_power_kw(1.0, 0.0) == 0.0

    def test_negative_heat(self):
        assert cryocooler_power_kw(-1.0) == 0.0


# ===========================================================================
# 13. Coolant boiloff
# ===========================================================================

class TestCoolantBoiloff:
    def test_zero_heat(self):
        assert coolant_boiloff_kg_per_sol(0.0) == 0.0

    def test_positive_boiloff(self):
        b = coolant_boiloff_kg_per_sol(0.1)
        assert b > 0.0

    def test_proportional_to_heat(self):
        b1 = coolant_boiloff_kg_per_sol(0.1)
        b2 = coolant_boiloff_kg_per_sol(0.2)
        assert abs(b2 / b1 - 2.0) < 1e-10

    def test_negative_heat(self):
        assert coolant_boiloff_kg_per_sol(-0.1) == 0.0


# ===========================================================================
# 14. Wire mass
# ===========================================================================

class TestWireMass:
    def test_positive_mass(self):
        m = wire_mass_kg(5.0, 500)
        assert m > 0.0

    def test_proportional_to_turns(self):
        m1 = wire_mass_kg(5.0, 100)
        m2 = wire_mass_kg(5.0, 200)
        assert abs(m2 / m1 - 2.0) < 1e-10

    def test_proportional_to_radius(self):
        m1 = wire_mass_kg(5.0, 100)
        m2 = wire_mass_kg(10.0, 100)
        assert abs(m2 / m1 - 2.0) < 1e-10


# ===========================================================================
# 15. Quench temperature rise
# ===========================================================================

class TestQuenchTempRise:
    def test_zero_energy(self):
        assert quench_temperature_rise_k(0.0, 100.0) == 0.0

    def test_zero_mass(self):
        assert quench_temperature_rise_k(10.0, 0.0) == 0.0

    def test_positive_rise(self):
        assert quench_temperature_rise_k(10.0, 100.0) > 0.0

    def test_more_energy_more_rise(self):
        dt1 = quench_temperature_rise_k(5.0, 100.0)
        dt2 = quench_temperature_rise_k(10.0, 100.0)
        assert dt2 > dt1

    def test_more_mass_less_rise(self):
        dt1 = quench_temperature_rise_k(10.0, 100.0)
        dt2 = quench_temperature_rise_k(10.0, 200.0)
        assert dt2 < dt1


# ===========================================================================
# 16. Create shield
# ===========================================================================

class TestCreateShield:
    def test_default_creation(self):
        config, state = create_shield()
        assert config.coil_radius_m == DEFAULT_COIL_RADIUS_M
        assert state.field_t == 0.0
        assert state.coolant_kg == DEFAULT_COOLANT_KG
        assert not state.quenched

    def test_custom_creation(self):
        config, state = create_shield(coil_radius_m=10.0, coolant_kg=5000.0)
        assert config.coil_radius_m == 10.0
        assert state.coolant_kg == 5000.0

    def test_negative_radius_clamped(self):
        config, _ = create_shield(coil_radius_m=-5.0)
        assert config.coil_radius_m == 0.1

    def test_negative_coolant_clamped(self):
        _, state = create_shield(coolant_kg=-100.0)
        assert state.coolant_kg == 0.0


# ===========================================================================
# 17. Single tick behavior
# ===========================================================================

class TestTick:
    def test_first_tick_ramps(self):
        config, state = create_shield()
        result = tick(config, state)
        assert result.sol == 1
        assert result.field_t > 0.0
        assert result.field_t <= MAX_DB_DT_T_PER_SOL + 1e-9

    def test_field_increases_per_tick(self):
        config, state = create_shield()
        prev = 0.0
        for _ in range(5):
            result = tick(config, state)
            assert result.field_t >= prev
            prev = result.field_t

    def test_field_bounded_by_max(self):
        config, state = create_shield()
        for _ in range(200):
            result = tick(config, state)
        assert result.field_t <= config.max_field_t + 1e-9

    def test_coil_temp_stays_cold(self):
        config, state = create_shield()
        for _ in range(50):
            result = tick(config, state)
        assert result.coil_temp_k < YBCO_TC_K

    def test_coolant_decreases(self):
        config, state = create_shield()
        initial = state.coolant_kg
        for _ in range(10):
            tick(config, state)
        assert state.coolant_kg < initial

    def test_power_consumed(self):
        config, state = create_shield()
        for _ in range(10):
            tick(config, state)
        assert state.total_power_consumed_kwh > 0.0

    def test_gcr_deflected_accumulates(self):
        config, state = create_shield()
        for _ in range(20):
            tick(config, state)
        assert state.total_gcr_deflected_msv > 0.0

    def test_spe_deflection_only_when_active(self):
        config, state = create_shield()
        for _ in range(20):
            tick(config, state)
        result = tick(config, state, spe_active=False)
        assert result.spe_dose_reduced_msv == 0.0

    def test_spe_deflection_when_active(self):
        config, state = create_shield()
        for _ in range(30):
            tick(config, state)
        result = tick(config, state, spe_active=True, spe_msv=100.0)
        assert result.spe_dose_reduced_msv > 0.0

    def test_stored_energy_increases_with_field(self):
        config, state = create_shield()
        energies = []
        for _ in range(10):
            result = tick(config, state)
            energies.append(result.stored_energy_mj)
        for i in range(1, len(energies)):
            assert energies[i] >= energies[i - 1] - 1e-9


# ===========================================================================
# 18. Quench behavior
# ===========================================================================

class TestQuench:
    def test_quench_on_overtemp(self):
        config, state = create_shield()
        for _ in range(30):
            tick(config, state)
        assert state.field_t > 0.0
        state.coil_temp_k = YBCO_TC_K + 1.0
        tick(config, state)
        assert state.quenched
        assert state.field_t == 0.0
        assert state.quench_count == 1

    def test_quench_recovery(self):
        config, state = create_shield()
        for _ in range(30):
            tick(config, state)
        state.coil_temp_k = YBCO_TC_K + 1.0
        tick(config, state)
        assert state.quenched
        for _ in range(100):
            tick(config, state)
        if state.coolant_kg > 10.0:
            assert not state.quenched

    def test_quench_temp_spike(self):
        config, state = create_shield()
        for _ in range(40):
            tick(config, state)
        pre_quench_temp = state.coil_temp_k
        state.coil_temp_k = YBCO_TC_K + 0.5
        tick(config, state)
        assert state.coil_temp_k > pre_quench_temp

    def test_no_quench_at_zero_field(self):
        config, state = create_shield()
        state.coil_temp_k = YBCO_TC_K + 10.0
        state.field_t = 0.0
        state.current_a = 0.0
        tick(config, state)
        assert state.quench_count == 0


# ===========================================================================
# 19. No coolant behavior
# ===========================================================================

class TestNoCoolant:
    def test_field_decays_without_coolant(self):
        config, state = create_shield()
        for _ in range(30):
            tick(config, state)
        assert state.field_t > 0.0
        state.coolant_kg = 0.0
        state.cryocooler_on = False
        initial_field = state.field_t
        for _ in range(100):
            tick(config, state)
        assert state.field_t < initial_field

    def test_no_coolant_coil_warms(self):
        config, state = create_shield()
        state.coolant_kg = 0.0
        state.cryocooler_on = False
        initial_temp = state.coil_temp_k
        for _ in range(10):
            tick(config, state)
        assert state.coil_temp_k >= initial_temp


# ===========================================================================
# 20. Cryocooler off behavior
# ===========================================================================

class TestCryocoolerOff:
    def test_no_power_consumed(self):
        config, state = create_shield()
        state.cryocooler_on = False
        power_before = state.total_power_consumed_kwh
        tick(config, state)
        assert state.total_power_consumed_kwh == power_before


# ===========================================================================
# 21. Multi-sol simulation
# ===========================================================================

class TestRunSimulation:
    def test_basic_simulation(self):
        summary = run_simulation(sols=100)
        assert summary["sols_simulated"] == 100
        assert summary["final_field_t"] > 0.0
        assert summary["shield_active"]
        assert summary["total_gcr_deflected_msv"] > 0.0

    def test_short_simulation(self):
        summary = run_simulation(sols=5)
        assert summary["sols_simulated"] == 5

    def test_spe_events(self):
        summary = run_simulation(sols=50, spe_events={30: 200.0, 40: 100.0})
        assert summary["total_spe_deflected_msv"] > 0.0

    def test_coolant_consumption(self):
        summary = run_simulation(sols=200)
        assert summary["coolant_consumed_kg"] > 0.0
        assert summary["coolant_remaining_kg"] < DEFAULT_COOLANT_KG

    def test_quench_count_zero_nominal(self):
        summary = run_simulation(sols=100)
        assert summary["quench_count"] == 0

    def test_energy_cutoff_reported(self):
        summary = run_simulation(sols=50)
        assert summary["energy_cutoff_gev"] > 0.0

    def test_power_consumed_reported(self):
        summary = run_simulation(sols=50)
        assert summary["total_power_consumed_kwh"] > 0.0


# ===========================================================================
# 22. Conservation laws and invariants
# ===========================================================================

class TestConservationLaws:
    def test_field_non_negative(self):
        config, state = create_shield()
        for _ in range(100):
            result = tick(config, state)
            assert result.field_t >= 0.0

    def test_coolant_non_negative(self):
        config, state = create_shield()
        for _ in range(500):
            result = tick(config, state)
            assert result.coolant_kg >= 0.0

    def test_coil_temp_above_coolant(self):
        config, state = create_shield()
        for _ in range(100):
            result = tick(config, state)
            assert result.coil_temp_k >= LN2_BOILING_POINT_K - 0.01

    def test_deflection_bounded_0_1(self):
        config, state = create_shield()
        for _ in range(100):
            result = tick(config, state)
            assert 0.0 <= result.gcr_deflection <= 1.0
            assert 0.0 <= result.spe_deflection <= 1.0

    def test_stored_energy_non_negative(self):
        config, state = create_shield()
        for _ in range(100):
            result = tick(config, state)
            assert result.stored_energy_mj >= 0.0

    def test_sol_counter_increments(self):
        config, state = create_shield()
        for i in range(1, 20):
            result = tick(config, state)
            assert result.sol == i

    def test_cumulative_deflection_monotonic(self):
        config, state = create_shield()
        prev = 0.0
        for _ in range(50):
            tick(config, state)
            assert state.total_gcr_deflected_msv >= prev - 1e-12
            prev = state.total_gcr_deflected_msv

    def test_power_consumption_monotonic(self):
        config, state = create_shield()
        prev = 0.0
        for _ in range(50):
            tick(config, state)
            assert state.total_power_consumed_kwh >= prev - 1e-12
            prev = state.total_power_consumed_kwh

    def test_field_rate_limited(self):
        config, state = create_shield()
        prev_field = 0.0
        for _ in range(50):
            tick(config, state)
            delta = abs(state.field_t - prev_field)
            assert delta <= MAX_DB_DT_T_PER_SOL + 1e-9
            prev_field = state.field_t

    def test_coolant_consumed_matches_decrease(self):
        config, state = create_shield()
        initial = state.coolant_kg
        for _ in range(50):
            tick(config, state)
        consumed = initial - state.coolant_kg
        assert abs(consumed - state.total_coolant_consumed_kg) < 0.1


# ===========================================================================
# 23. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_zero_turns(self):
        config, state = create_shield(num_turns=0)
        result = tick(config, state)
        assert result.field_t == 0.0

    def test_zero_current(self):
        config, state = create_shield(operating_current_a=0.0)
        result = tick(config, state)
        assert result.field_t == 0.0

    def test_tiny_coil(self):
        config, state = create_shield(coil_radius_m=0.01, coil_length_m=0.01)
        for _ in range(10):
            result = tick(config, state)
        assert result.field_t >= 0.0

    def test_massive_coolant(self):
        config, state = create_shield(coolant_kg=100_000.0)
        for _ in range(500):
            tick(config, state)
        assert state.coolant_kg > 90_000.0

    def test_extreme_spe(self):
        config, state = create_shield()
        for _ in range(30):
            tick(config, state)
        result = tick(config, state, spe_active=True, spe_msv=10_000.0)
        assert result.spe_dose_reduced_msv <= 10_000.0
        assert result.spe_dose_reduced_msv > 0.0

    def test_negative_spe_dose(self):
        config, state = create_shield()
        for _ in range(30):
            tick(config, state)
        result = tick(config, state, spe_active=True, spe_msv=-50.0)
        assert result.spe_dose_reduced_msv == 0.0

    def test_extreme_cold_ambient(self):
        config, state = create_shield()
        result = tick(config, state, ambient_temp_k=80.0)
        assert result.heat_leak_kw >= 0.0


# ===========================================================================
# 24. Smoke test
# ===========================================================================

class TestSmokeTest:
    def test_10_sol_no_crash(self):
        config, state = create_shield()
        for _ in range(10):
            result = tick(config, state)
        assert result.sol == 10

    def test_100_sol_no_crash(self):
        summary = run_simulation(sols=100)
        assert summary["sols_simulated"] == 100

    def test_500_sol_with_spe_no_crash(self):
        spe = {50: 100.0, 150: 500.0, 300: 50.0, 450: 200.0}
        summary = run_simulation(sols=500, spe_events=spe)
        assert summary["sols_simulated"] == 500
        assert summary["total_spe_deflected_msv"] > 0.0


# ===========================================================================
# 25. Physical reasonableness
# ===========================================================================

class TestPhysicalReasonableness:
    def test_default_field_reasonable(self):
        config, state = create_shield()
        for _ in range(100):
            tick(config, state)
        assert 0.1 < state.field_t < 1.0

    def test_daily_gcr_deflection_reasonable(self):
        config, state = create_shield()
        for _ in range(100):
            result = tick(config, state)
        assert 0.15 < result.gcr_deflection < 0.50

    def test_cryocooler_power_reasonable(self):
        config, state = create_shield()
        for _ in range(20):
            result = tick(config, state)
        assert 0.1 < result.cryocooler_power_kw < 20.0

    def test_coolant_lifetime_reasonable(self):
        summary = run_simulation(sols=500)
        assert summary["coolant_remaining_kg"] > 500.0

    def test_stored_energy_reasonable(self):
        config, state = create_shield()
        for _ in range(100):
            tick(config, state)
        assert 0.001 < state.stored_energy_mj < 1000.0

    def test_ramp_up_takes_multiple_sols(self):
        summary = run_simulation(sols=100)
        assert summary["ramp_up_sols"] > 1

    def test_wire_mass_reasonable(self):
        m = wire_mass_kg(DEFAULT_COIL_RADIUS_M, DEFAULT_NUM_TURNS)
        assert 100.0 < m < 50_000.0


# ===========================================================================
# 26. Integration with radiation constants
# ===========================================================================

class TestRadiationIntegration:
    def test_gcr_dose_matches_surface(self):
        config, state = create_shield()
        for _ in range(50):
            result = tick(config, state)
        assert result.gcr_dose_reduced_msv <= GCR_SURFACE_MSV_SOL
        assert result.gcr_dose_reduced_msv >= 0.0

    def test_dose_reduction_over_year(self):
        summary = run_simulation(sols=668)
        unshielded = 668 * GCR_SURFACE_MSV_SOL
        assert summary["total_gcr_deflected_msv"] > unshielded * 0.15

    def test_spe_protection(self):
        config, state = create_shield()
        for _ in range(50):
            tick(config, state)
        result = tick(config, state, spe_active=True, spe_msv=500.0)
        assert result.spe_dose_reduced_msv > 350.0


# ===========================================================================
# 27. State initialization invariants
# ===========================================================================

class TestStateInit:
    def test_post_init_clamps_negative_field(self):
        s = ShieldState(field_t=-1.0)
        assert s.field_t == 0.0

    def test_post_init_clamps_negative_coolant(self):
        s = ShieldState(coolant_kg=-100.0)
        assert s.coolant_kg == 0.0

    def test_post_init_clamps_low_temp(self):
        s = ShieldState(coil_temp_k=10.0)
        assert s.coil_temp_k == LN2_BOILING_POINT_K

    def test_post_init_clamps_negative_current(self):
        s = ShieldState(current_a=-50.0)
        assert s.current_a == 0.0

    def test_post_init_clamps_negative_energy(self):
        s = ShieldState(stored_energy_mj=-5.0)
        assert s.stored_energy_mj == 0.0
