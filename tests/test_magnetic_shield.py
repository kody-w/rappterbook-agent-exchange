"""test_magnetic_shield.py -- 141 tests for Mars Habitat Magnetic Shield."""
from __future__ import annotations
import math, sys, os, pytest
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

class TestSolenoidField:
    def test_basic_field(self):
        b = solenoid_field_t(1000, 10.0, 1.0)
        assert abs(b - MU_0 * 10000) < 1e-10
    def test_zero_length(self):
        assert solenoid_field_t(100, 10.0, 0.0) == 0.0
    def test_zero_turns(self):
        assert solenoid_field_t(0, 10.0, 1.0) == 0.0
    def test_zero_current(self):
        assert solenoid_field_t(100, 0.0, 1.0) == 0.0
    def test_proportional_to_current(self):
        assert abs(solenoid_field_t(100, 20.0, 1.0) / solenoid_field_t(100, 10.0, 1.0) - 2.0) < 1e-10
    def test_proportional_to_turns(self):
        assert abs(solenoid_field_t(200, 10.0, 1.0) / solenoid_field_t(100, 10.0, 1.0) - 2.0) < 1e-10
    def test_inversely_proportional_to_length(self):
        assert abs(solenoid_field_t(100, 10.0, 1.0) / solenoid_field_t(100, 10.0, 2.0) - 2.0) < 1e-10
    def test_negative_turns(self):
        assert solenoid_field_t(-1, 10.0, 1.0) == 0.0
    def test_default_config_field(self):
        b = solenoid_field_t(DEFAULT_NUM_TURNS, DEFAULT_OPERATING_CURRENT_A, DEFAULT_COIL_LENGTH_M)
        assert 0.4 < b < 0.6

class TestDipoleMoment:
    def test_basic(self):
        assert abs(magnetic_dipole_moment(100, 10.0, 1.0) - 100 * 10.0 * math.pi) < 1e-10
    def test_radius_squared(self):
        assert abs(magnetic_dipole_moment(100, 10.0, 2.0) / magnetic_dipole_moment(100, 10.0, 1.0) - 4.0) < 1e-10
    def test_zero_current(self):
        assert magnetic_dipole_moment(100, 0.0, 1.0) == 0.0
    def test_zero_radius(self):
        assert magnetic_dipole_moment(100, 10.0, 0.0) == 0.0

class TestStoredEnergy:
    def test_formula(self):
        assert abs(stored_energy_mj(1.0, 10.0) - 10.0 / (2.0 * MU_0) / 1e6) < 1e-6
    def test_zero_field(self):
        assert stored_energy_mj(0.0, 10.0) == 0.0
    def test_zero_volume(self):
        assert stored_energy_mj(1.0, 0.0) == 0.0
    def test_scales_b_squared(self):
        assert abs(stored_energy_mj(2.0, 10.0) / stored_energy_mj(1.0, 10.0) - 4.0) < 1e-10
    def test_positive(self):
        assert stored_energy_mj(0.5, 5.0) > 0.0
    def test_negative_field(self):
        assert stored_energy_mj(-1.0, 10.0) == 0.0

class TestCoilVolume:
    def test_basic(self):
        assert abs(coil_volume_m3(2.0, 3.0) - math.pi * 4.0 * 3.0) < 1e-10
    def test_zero_radius(self):
        assert coil_volume_m3(0.0, 3.0) == 0.0
    def test_zero_length(self):
        assert coil_volume_m3(2.0, 0.0) == 0.0
    def test_negative(self):
        assert coil_volume_m3(-1.0, 3.0) == 0.0

class TestLarmorRadius:
    def test_basic(self):
        r = larmor_radius_m(PROTON_MASS_KG, 1e6, PROTON_CHARGE_C, 1.0)
        assert abs(r - PROTON_MASS_KG * 1e6 / (PROTON_CHARGE_C * 1.0)) < 1e-20
    def test_zero_field(self):
        assert larmor_radius_m(PROTON_MASS_KG, 1e6, PROTON_CHARGE_C, 0.0) == float("inf")
    def test_zero_charge(self):
        assert larmor_radius_m(PROTON_MASS_KG, 1e6, 0.0, 1.0) == float("inf")
    def test_inverse_field(self):
        r1 = larmor_radius_m(PROTON_MASS_KG, 1e6, PROTON_CHARGE_C, 0.5)
        r2 = larmor_radius_m(PROTON_MASS_KG, 1e6, PROTON_CHARGE_C, 1.0)
        assert abs(r1 / r2 - 2.0) < 1e-10
    def test_proportional_velocity(self):
        r1 = larmor_radius_m(PROTON_MASS_KG, 1e6, PROTON_CHARGE_C, 1.0)
        r2 = larmor_radius_m(PROTON_MASS_KG, 2e6, PROTON_CHARGE_C, 1.0)
        assert abs(r2 / r1 - 2.0) < 1e-10

class TestProtonLarmorRelativistic:
    def test_low_energy(self):
        assert 0.0 < proton_larmor_radius_m(0.001, 1.0) < 1.0
    def test_1gev(self):
        assert 3.0 < proton_larmor_radius_m(1.0, 1.0) < 10.0
    def test_zero_energy(self):
        assert proton_larmor_radius_m(0.0, 1.0) == float("inf")
    def test_zero_field(self):
        assert proton_larmor_radius_m(1.0, 0.0) == float("inf")
    def test_higher_energy_larger(self):
        assert proton_larmor_radius_m(2.0, 1.0) > proton_larmor_radius_m(0.5, 1.0)
    def test_higher_field_smaller(self):
        assert proton_larmor_radius_m(1.0, 2.0) < proton_larmor_radius_m(1.0, 0.5)

class TestEnergyCutoff:
    def test_weak_field(self):
        assert 0.005 < energy_cutoff_gev(0.1, 5.0) < 0.05
    def test_zero_field(self):
        assert energy_cutoff_gev(0.0, 5.0) == 0.0
    def test_zero_radius(self):
        assert energy_cutoff_gev(0.1, 0.0) == 0.0
    def test_increases_with_field(self):
        assert energy_cutoff_gev(0.1, 5.0) < energy_cutoff_gev(0.5, 5.0)
    def test_increases_with_radius(self):
        assert energy_cutoff_gev(0.1, 3.0) < energy_cutoff_gev(0.1, 5.0)
    def test_strong_field(self):
        assert energy_cutoff_gev(1.0, 10.0) > 1.0
    def test_larmor_consistency(self):
        e = energy_cutoff_gev(0.5, 5.0)
        assert abs(proton_larmor_radius_m(e, 0.5) - 5.0) / 5.0 < 0.01

class TestGCRDeflection:
    def test_zero(self):
        assert gcr_deflection_fraction(0.0) == 0.0
    def test_negative(self):
        assert gcr_deflection_fraction(-1.0) == 0.0
    def test_high(self):
        assert gcr_deflection_fraction(100.0) == 0.99
    def test_monotonic(self):
        prev = 0.0
        for e in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            f = gcr_deflection_fraction(e); assert f >= prev; prev = f
    def test_bounded(self):
        for e in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
            assert 0.0 <= gcr_deflection_fraction(e) <= 1.0
    def test_1gev(self):
        assert 0.65 < gcr_deflection_fraction(1.0) < 0.75
    def test_low_cutoff(self):
        assert gcr_deflection_fraction(0.05) < 0.10

class TestSPEDeflection:
    def test_zero_field(self):
        assert spe_deflection_fraction(0.0, 5.0) == 0.0
    def test_zero_radius(self):
        assert spe_deflection_fraction(0.1, 0.0) == 0.0
    def test_moderate(self):
        assert spe_deflection_fraction(0.5, 5.0) > 0.85
    def test_strong(self):
        assert spe_deflection_fraction(1.0, 5.0) >= 0.99
    def test_bounded(self):
        for b in [0.0, 0.001, 0.01, 0.1, 1.0]:
            assert 0.0 <= spe_deflection_fraction(b, 5.0) <= 1.0
    def test_monotonic(self):
        prev = 0.0
        for b in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
            f = spe_deflection_fraction(b, 5.0); assert f >= prev; prev = f

class TestHeatLeak:
    def test_basic(self):
        assert heat_leak_kw(80.0, 210.0, 77.0) > 0.0
    def test_no_gradient(self):
        assert heat_leak_kw(80.0, 77.0, 77.0) == 0.0
    def test_cold_ambient(self):
        assert heat_leak_kw(80.0, 50.0, 77.0) == 0.0
    def test_area_scaling(self):
        assert heat_leak_kw(80.0, 210.0, 77.0) > heat_leak_kw(40.0, 210.0, 77.0)
    def test_temp_scaling(self):
        assert heat_leak_kw(80.0, 250.0, 77.0) > heat_leak_kw(80.0, 180.0, 77.0)

class TestJointHeating:
    def test_zero(self):
        assert joint_heating_kw(0.0) == 0.0
    def test_positive(self):
        assert joint_heating_kw(200.0) > 0.0
    def test_i_squared(self):
        assert abs(joint_heating_kw(200.0) / joint_heating_kw(100.0) - 4.0) < 1e-10

class TestCryocoolerPower:
    def test_basic(self):
        assert abs(cryocooler_power_kw(1.0, 0.2) - 5.0) < 1e-10
    def test_zero_heat(self):
        assert cryocooler_power_kw(0.0) == 0.0
    def test_zero_cop(self):
        assert cryocooler_power_kw(1.0, 0.0) == 0.0
    def test_negative(self):
        assert cryocooler_power_kw(-1.0) == 0.0

class TestBoiloff:
    def test_zero(self):
        assert coolant_boiloff_kg_per_sol(0.0) == 0.0
    def test_positive(self):
        assert coolant_boiloff_kg_per_sol(0.1) > 0.0
    def test_proportional(self):
        assert abs(coolant_boiloff_kg_per_sol(0.2) / coolant_boiloff_kg_per_sol(0.1) - 2.0) < 1e-10
    def test_negative(self):
        assert coolant_boiloff_kg_per_sol(-0.1) == 0.0

class TestWireMass:
    def test_positive(self):
        assert wire_mass_kg(5.0, 500) > 0.0
    def test_turns(self):
        assert abs(wire_mass_kg(5.0, 200) / wire_mass_kg(5.0, 100) - 2.0) < 1e-10
    def test_radius(self):
        assert abs(wire_mass_kg(10.0, 100) / wire_mass_kg(5.0, 100) - 2.0) < 1e-10

class TestQuenchTempRise:
    def test_zero_energy(self):
        assert quench_temperature_rise_k(0.0, 100.0) == 0.0
    def test_zero_mass(self):
        assert quench_temperature_rise_k(10.0, 0.0) == 0.0
    def test_positive(self):
        assert quench_temperature_rise_k(10.0, 100.0) > 0.0
    def test_more_energy(self):
        assert quench_temperature_rise_k(10.0, 100.0) > quench_temperature_rise_k(5.0, 100.0)
    def test_more_mass(self):
        assert quench_temperature_rise_k(10.0, 200.0) < quench_temperature_rise_k(10.0, 100.0)

class TestCreateShield:
    def test_default(self):
        c, s = create_shield()
        assert c.coil_radius_m == DEFAULT_COIL_RADIUS_M
        assert s.field_t == 0.0 and s.coolant_kg == DEFAULT_COOLANT_KG
    def test_custom(self):
        c, s = create_shield(coil_radius_m=10.0, coolant_kg=5000.0)
        assert c.coil_radius_m == 10.0 and s.coolant_kg == 5000.0
    def test_clamp_radius(self):
        c, _ = create_shield(coil_radius_m=-5.0)
        assert c.coil_radius_m == 0.1
    def test_clamp_coolant(self):
        _, s = create_shield(coolant_kg=-100.0)
        assert s.coolant_kg == 0.0

class TestTick:
    def test_first_tick_ramps(self):
        c, s = create_shield()
        r = tick(c, s)
        assert r.sol == 1 and 0.0 < r.field_t <= MAX_DB_DT_T_PER_SOL + 1e-9
    def test_field_increases(self):
        c, s = create_shield()
        prev = 0.0
        for _ in range(5):
            r = tick(c, s); assert r.field_t >= prev; prev = r.field_t
    def test_field_bounded(self):
        c, s = create_shield()
        for _ in range(200):
            r = tick(c, s)
        assert r.field_t <= c.max_field_t + 1e-9
    def test_coil_stays_cold(self):
        c, s = create_shield()
        for _ in range(50):
            r = tick(c, s)
        assert r.coil_temp_k < YBCO_TC_K
    def test_coolant_decreases(self):
        c, s = create_shield()
        initial = s.coolant_kg
        for _ in range(10): tick(c, s)
        assert s.coolant_kg < initial
    def test_power_consumed(self):
        c, s = create_shield()
        for _ in range(10): tick(c, s)
        assert s.total_power_consumed_kwh > 0.0
    def test_gcr_accumulates(self):
        c, s = create_shield()
        for _ in range(20): tick(c, s)
        assert s.total_gcr_deflected_msv > 0.0
    def test_no_spe_when_inactive(self):
        c, s = create_shield()
        for _ in range(20): tick(c, s)
        assert tick(c, s, spe_active=False).spe_dose_reduced_msv == 0.0
    def test_spe_when_active(self):
        c, s = create_shield()
        for _ in range(30): tick(c, s)
        assert tick(c, s, spe_active=True, spe_msv=100.0).spe_dose_reduced_msv > 0.0
    def test_energy_increases(self):
        c, s = create_shield()
        e = []
        for _ in range(10):
            e.append(tick(c, s).stored_energy_mj)
        for i in range(1, len(e)): assert e[i] >= e[i-1] - 1e-9

class TestQuench:
    def test_quench_on_overtemp(self):
        c, s = create_shield()
        for _ in range(30): tick(c, s)
        assert s.field_t > 0.0
        s.coil_temp_k = YBCO_TC_K + 1.0
        tick(c, s)
        assert s.quenched and s.field_t == 0.0 and s.quench_count == 1
    def test_quench_recovery(self):
        c, s = create_shield()
        for _ in range(30): tick(c, s)
        s.coil_temp_k = YBCO_TC_K + 1.0
        tick(c, s)
        assert s.quenched
        for _ in range(100): tick(c, s)
        if s.coolant_kg > 10.0: assert not s.quenched
    def test_temp_spike(self):
        c, s = create_shield()
        for _ in range(40): tick(c, s)
        pre = s.coil_temp_k
        s.coil_temp_k = YBCO_TC_K + 0.5
        tick(c, s)
        assert s.coil_temp_k > pre
    def test_no_quench_at_zero_field(self):
        c, s = create_shield()
        s.coil_temp_k = YBCO_TC_K + 10.0
        s.field_t = 0.0; s.current_a = 0.0
        tick(c, s)
        assert s.quench_count == 0

class TestNoCoolant:
    def test_field_decays(self):
        c, s = create_shield()
        for _ in range(30): tick(c, s)
        initial = s.field_t
        s.coolant_kg = 0.0; s.cryocooler_on = False
        for _ in range(100): tick(c, s)
        assert s.field_t < initial
    def test_coil_warms(self):
        c, s = create_shield()
        s.coolant_kg = 0.0; s.cryocooler_on = False
        t0 = s.coil_temp_k
        for _ in range(10): tick(c, s)
        assert s.coil_temp_k >= t0

class TestCryocoolerOff:
    def test_no_power(self):
        c, s = create_shield()
        s.cryocooler_on = False
        p = s.total_power_consumed_kwh
        tick(c, s)
        assert s.total_power_consumed_kwh == p

class TestRunSimulation:
    def test_basic(self):
        r = run_simulation(sols=100)
        assert r["sols_simulated"] == 100 and r["final_field_t"] > 0.0 and r["shield_active"]
    def test_short(self):
        assert run_simulation(sols=5)["sols_simulated"] == 5
    def test_spe(self):
        assert run_simulation(sols=50, spe_events={30: 200.0})["total_spe_deflected_msv"] > 0.0
    def test_coolant(self):
        r = run_simulation(sols=200)
        assert r["coolant_consumed_kg"] > 0.0 and r["coolant_remaining_kg"] < DEFAULT_COOLANT_KG
    def test_no_quench_nominal(self):
        assert run_simulation(sols=100)["quench_count"] == 0
    def test_energy_cutoff(self):
        assert run_simulation(sols=50)["energy_cutoff_gev"] > 0.0
    def test_power(self):
        assert run_simulation(sols=50)["total_power_consumed_kwh"] > 0.0

class TestConservationLaws:
    def test_field_non_negative(self):
        c, s = create_shield()
        for _ in range(100): assert tick(c, s).field_t >= 0.0
    def test_coolant_non_negative(self):
        c, s = create_shield()
        for _ in range(500): assert tick(c, s).coolant_kg >= 0.0
    def test_temp_above_coolant(self):
        c, s = create_shield()
        for _ in range(100): assert tick(c, s).coil_temp_k >= LN2_BOILING_POINT_K - 0.01
    def test_deflection_bounded(self):
        c, s = create_shield()
        for _ in range(100):
            r = tick(c, s)
            assert 0.0 <= r.gcr_deflection <= 1.0 and 0.0 <= r.spe_deflection <= 1.0
    def test_energy_non_negative(self):
        c, s = create_shield()
        for _ in range(100): assert tick(c, s).stored_energy_mj >= 0.0
    def test_sol_increments(self):
        c, s = create_shield()
        for i in range(1, 20): assert tick(c, s).sol == i
    def test_cumulative_monotonic(self):
        c, s = create_shield()
        prev = 0.0
        for _ in range(50):
            tick(c, s)
            assert s.total_gcr_deflected_msv >= prev - 1e-12
            prev = s.total_gcr_deflected_msv
    def test_power_monotonic(self):
        c, s = create_shield()
        prev = 0.0
        for _ in range(50):
            tick(c, s)
            assert s.total_power_consumed_kwh >= prev - 1e-12
            prev = s.total_power_consumed_kwh
    def test_rate_limited(self):
        c, s = create_shield()
        prev = 0.0
        for _ in range(50):
            tick(c, s)
            assert abs(s.field_t - prev) <= MAX_DB_DT_T_PER_SOL + 1e-9
            prev = s.field_t
    def test_coolant_balance(self):
        c, s = create_shield()
        initial = s.coolant_kg
        for _ in range(50): tick(c, s)
        assert abs((initial - s.coolant_kg) - s.total_coolant_consumed_kg) < 0.1

class TestEdgeCases:
    def test_zero_turns(self):
        c, s = create_shield(num_turns=0)
        assert tick(c, s).field_t == 0.0
    def test_zero_current(self):
        c, s = create_shield(operating_current_a=0.0)
        assert tick(c, s).field_t == 0.0
    def test_tiny_coil(self):
        c, s = create_shield(coil_radius_m=0.01, coil_length_m=0.01)
        for _ in range(10): r = tick(c, s)
        assert r.field_t >= 0.0
    def test_massive_coolant(self):
        c, s = create_shield(coolant_kg=100_000.0)
        for _ in range(500): tick(c, s)
        assert s.coolant_kg > 90_000.0
    def test_extreme_spe(self):
        c, s = create_shield()
        for _ in range(30): tick(c, s)
        r = tick(c, s, spe_active=True, spe_msv=10_000.0)
        assert 0.0 < r.spe_dose_reduced_msv <= 10_000.0
    def test_negative_spe(self):
        c, s = create_shield()
        for _ in range(30): tick(c, s)
        assert tick(c, s, spe_active=True, spe_msv=-50.0).spe_dose_reduced_msv == 0.0
    def test_cold_ambient(self):
        c, s = create_shield()
        assert tick(c, s, ambient_temp_k=80.0).heat_leak_kw >= 0.0

class TestSmokeTest:
    def test_10_sol(self):
        c, s = create_shield()
        for _ in range(10): r = tick(c, s)
        assert r.sol == 10
    def test_100_sol(self):
        assert run_simulation(sols=100)["sols_simulated"] == 100
    def test_500_sol_spe(self):
        r = run_simulation(sols=500, spe_events={50: 100, 150: 500, 300: 50, 450: 200})
        assert r["sols_simulated"] == 500 and r["total_spe_deflected_msv"] > 0.0

class TestPhysical:
    def test_field_range(self):
        c, s = create_shield()
        for _ in range(100): tick(c, s)
        assert 0.1 < s.field_t < 1.0
    def test_gcr_deflection_range(self):
        c, s = create_shield()
        for _ in range(100): r = tick(c, s)
        assert 0.15 < r.gcr_deflection < 0.50
    def test_cryo_power_range(self):
        c, s = create_shield()
        for _ in range(20): r = tick(c, s)
        assert 0.1 < r.cryocooler_power_kw < 20.0
    def test_coolant_lifetime(self):
        assert run_simulation(sols=500)["coolant_remaining_kg"] > 500.0
    def test_stored_energy_range(self):
        c, s = create_shield()
        for _ in range(100): tick(c, s)
        assert 0.001 < s.stored_energy_mj < 1000.0
    def test_ramp_up_takes_sols(self):
        assert run_simulation(sols=100)["ramp_up_sols"] > 1
    def test_wire_mass_range(self):
        assert 100 < wire_mass_kg(DEFAULT_COIL_RADIUS_M, DEFAULT_NUM_TURNS) < 50_000

class TestRadiation:
    def test_gcr_bounded(self):
        c, s = create_shield()
        for _ in range(50): r = tick(c, s)
        assert 0.0 <= r.gcr_dose_reduced_msv <= GCR_SURFACE_MSV_SOL
    def test_year_dose(self):
        assert run_simulation(sols=668)["total_gcr_deflected_msv"] > 668 * GCR_SURFACE_MSV_SOL * 0.15
    def test_spe_protection(self):
        c, s = create_shield()
        for _ in range(50): tick(c, s)
        assert tick(c, s, spe_active=True, spe_msv=500.0).spe_dose_reduced_msv > 350.0

class TestStateInit:
    def test_neg_field(self):
        assert ShieldState(field_t=-1.0).field_t == 0.0
    def test_neg_coolant(self):
        assert ShieldState(coolant_kg=-100.0).coolant_kg == 0.0
    def test_low_temp(self):
        assert ShieldState(coil_temp_k=10.0).coil_temp_k == LN2_BOILING_POINT_K
    def test_neg_current(self):
        assert ShieldState(current_a=-50.0).current_a == 0.0
    def test_neg_energy(self):
        assert ShieldState(stored_energy_mj=-5.0).stored_energy_mj == 0.0
