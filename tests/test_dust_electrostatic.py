"""Tests for dust_electrostatic.py -- Mars Electrostatic Dust Mitigation System.

72 tests covering:
  - Van der Waals adhesion physics
  - Gravity force calculations
  - Coulomb force and removal thresholds
  - Cleaning efficiency under varied conditions
  - Power consumption modes
  - Electrode degradation over time
  - Net dust balance (deposition vs cleaning)
  - State clamping and invariants
  - Tick function integration
  - Multi-sol smoke tests
  - Storm survival scenarios
  - Conservation laws (energy, dust mass)
  - Factory scenarios
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dust_electrostatic import (
    DustShieldState, DustShieldResult,
    van_der_waals_force, gravity_force, coulomb_force,
    removal_threshold_field, cleaning_efficiency,
    power_consumption_w, electrode_degradation, net_dust_change,
    tick_dust_shield, create_dust_shield,
    DUST_DENSITY_KG_M3, DUST_MODE_DIAMETER_UM, DUST_CHARGE_C,
    HAMAKER_CONSTANT_J, VDW_CONTACT_SEPARATION_M, MARS_GRAVITY_M_S2,
    THRESHOLD_FIELD_V_M, OPERATING_FIELD_V_M, MAX_FIELD_V_M,
    INTEGRATED_EFFICIENCY, POWER_W_PER_M2, BURST_POWER_MULTIPLIER,
    ITO_DEGRADATION_PER_SOL, ITO_MIN_HEALTH, ITO_REPLACEMENT_THRESHOLD,
    ITO_OPTIMAL_TEMP_LOW_C, ITO_OPTIMAL_TEMP_HIGH_C,
    ITO_MIN_TEMP_C, ITO_MAX_TEMP_C,
    NORMAL_DEPOSITION_RATE, STORM_DEPOSITION_RATE,
    HOURS_PER_SOL, SECONDS_PER_SOL,
)


# =====================================================================
# Van der Waals adhesion
# =====================================================================

class TestVanDerWaals:
    def test_zero_diameter(self):
        assert van_der_waals_force(0.0) == 0.0

    def test_negative_diameter(self):
        assert van_der_waals_force(-1e-6) == 0.0

    def test_positive_force(self):
        f = van_der_waals_force(2.5e-6)
        assert f > 0

    def test_force_proportional_to_diameter(self):
        """VdW force is linear in d: F(2d) = 2*F(d)."""
        f1 = van_der_waals_force(1e-6)
        f2 = van_der_waals_force(2e-6)
        assert abs(f2 / f1 - 2.0) < 1e-10

    def test_known_value(self):
        """Check against hand calculation: A*d/(24*z^2)."""
        d = 2.5e-6
        expected = HAMAKER_CONSTANT_J * d / (24.0 * VDW_CONTACT_SEPARATION_M**2)
        assert abs(van_der_waals_force(d) - expected) < 1e-20

    def test_adhesion_dominates_gravity_for_small_grains(self):
        """For micron-scale grains, VdW >> gravity."""
        d = 2.5e-6
        f_vdw = van_der_waals_force(d)
        f_grav = gravity_force(d)
        assert f_vdw > 100 * f_grav  # orders of magnitude larger


# =====================================================================
# Gravity force
# =====================================================================

class TestGravityForce:
    def test_zero_diameter(self):
        assert gravity_force(0.0) == 0.0

    def test_negative_diameter(self):
        assert gravity_force(-1e-6) == 0.0

    def test_positive(self):
        assert gravity_force(1e-6) > 0

    def test_scales_with_cube(self):
        """Gravity scales as d^3."""
        f1 = gravity_force(1e-6)
        f2 = gravity_force(2e-6)
        assert abs(f2 / f1 - 8.0) < 1e-6

    def test_mars_gravity_used(self):
        """Force should use Mars gravity, not Earth."""
        d = 1e-3  # 1mm grain
        vol = (math.pi / 6.0) * d**3
        expected = vol * DUST_DENSITY_KG_M3 * MARS_GRAVITY_M_S2
        assert abs(gravity_force(d) - expected) < 1e-15


# =====================================================================
# Coulomb force
# =====================================================================

class TestCoulombForce:
    def test_zero_charge(self):
        assert coulomb_force(0.0, 1e5) == 0.0

    def test_zero_field(self):
        assert coulomb_force(1e-14, 0.0) == 0.0

    def test_positive(self):
        f = coulomb_force(DUST_CHARGE_C, OPERATING_FIELD_V_M)
        assert f > 0

    def test_negative_charge_gives_positive_force(self):
        """Force magnitude is always positive."""
        f = coulomb_force(-DUST_CHARGE_C, OPERATING_FIELD_V_M)
        assert f > 0

    def test_linear_in_field(self):
        f1 = coulomb_force(DUST_CHARGE_C, 1e5)
        f2 = coulomb_force(DUST_CHARGE_C, 2e5)
        assert abs(f2 / f1 - 2.0) < 1e-10

    def test_operating_field_overcomes_adhesion(self):
        """At operating field, Coulomb force significant vs gravity."""
        d = DUST_MODE_DIAMETER_UM * 1e-6
        f_coulomb = coulomb_force(DUST_CHARGE_C, OPERATING_FIELD_V_M)
        f_grav = gravity_force(d)
        # Coulomb easily beats gravity; VdW requires traveling wave effect
        assert f_coulomb > f_grav * 1000


# =====================================================================
# Removal threshold field
# =====================================================================

class TestRemovalThreshold:
    def test_zero_diameter(self):
        assert removal_threshold_field(0.0) == float('inf')

    def test_negative_diameter(self):
        assert removal_threshold_field(-5.0) == float('inf')

    def test_positive_finite(self):
        field = removal_threshold_field(DUST_MODE_DIAMETER_UM)
        assert 0 < field < float('inf')

    def test_below_operating_field(self):
        """Large grains (>10 um) removable at operating field."""
        threshold = removal_threshold_field(20.0)  # 20 um grain
        assert threshold < OPERATING_FIELD_V_M

    def test_small_grains_harder(self):
        """Smaller grains need higher fields (adhesion/charge ratio)."""
        large = removal_threshold_field(10.0)
        small = removal_threshold_field(1.0)
        assert small > large

    def test_very_large_grains_easy(self):
        """Large grains (100 um) should have low threshold."""
        field = removal_threshold_field(100.0)
        assert field < THRESHOLD_FIELD_V_M


# =====================================================================
# Cleaning efficiency
# =====================================================================

class TestCleaningEfficiency:
    def test_zero_field(self):
        assert cleaning_efficiency(0.0, 1.0, -40.0) == 0.0

    def test_dead_electrode(self):
        assert cleaning_efficiency(OPERATING_FIELD_V_M, ITO_MIN_HEALTH, -40.0) == 0.0

    def test_below_min_health(self):
        assert cleaning_efficiency(OPERATING_FIELD_V_M, 0.05, -40.0) == 0.0

    def test_nominal_conditions(self):
        """At operating field, full health, optimal temp, get ~85% eff."""
        eff = cleaning_efficiency(OPERATING_FIELD_V_M, 1.0, -40.0)
        assert 0.5 < eff <= 1.0

    def test_bounded_0_1(self):
        """Efficiency always in [0, 1]."""
        for field in [0, 1e4, 1e5, 5e5, 1e6]:
            for health in [0.0, 0.1, 0.5, 1.0]:
                for temp in [-120, -60, 0, 20, 80]:
                    eff = cleaning_efficiency(field, health, temp)
                    assert 0.0 <= eff <= 1.0

    def test_higher_field_better(self):
        eff_low = cleaning_efficiency(1e5, 1.0, -40.0)
        eff_high = cleaning_efficiency(5e5, 1.0, -40.0)
        assert eff_high >= eff_low

    def test_cold_penalty(self):
        """Very cold temp reduces efficiency."""
        eff_normal = cleaning_efficiency(OPERATING_FIELD_V_M, 1.0, -40.0)
        eff_cold = cleaning_efficiency(OPERATING_FIELD_V_M, 1.0, -100.0)
        assert eff_cold < eff_normal

    def test_hot_penalty(self):
        """Very hot temp reduces efficiency."""
        eff_normal = cleaning_efficiency(OPERATING_FIELD_V_M, 1.0, 0.0)
        eff_hot = cleaning_efficiency(OPERATING_FIELD_V_M, 1.0, 70.0)
        assert eff_hot < eff_normal

    def test_degraded_electrode(self):
        """Half health reduces efficiency."""
        eff_full = cleaning_efficiency(OPERATING_FIELD_V_M, 1.0, -40.0)
        eff_half = cleaning_efficiency(OPERATING_FIELD_V_M, 0.5, -40.0)
        assert eff_half < eff_full


# =====================================================================
# Power consumption
# =====================================================================

class TestPowerConsumption:
    def test_zero_area(self):
        assert power_consumption_w(0.0) == 0.0

    def test_negative_area(self):
        assert power_consumption_w(-10.0) == 0.0

    def test_normal_mode(self):
        p = power_consumption_w(100.0, "normal")
        assert abs(p - 100.0 * POWER_W_PER_M2) < 1e-10

    def test_standby_low(self):
        p_normal = power_consumption_w(100.0, "normal")
        p_standby = power_consumption_w(100.0, "standby")
        assert p_standby < p_normal

    def test_burst_high(self):
        p_normal = power_consumption_w(100.0, "normal")
        p_burst = power_consumption_w(100.0, "burst")
        assert p_burst > p_normal
        assert abs(p_burst / p_normal - BURST_POWER_MULTIPLIER) < 1e-10

    def test_proportional_to_area(self):
        p1 = power_consumption_w(50.0)
        p2 = power_consumption_w(100.0)
        assert abs(p2 / p1 - 2.0) < 1e-10

    def test_unknown_mode_defaults_to_normal(self):
        p = power_consumption_w(100.0, "turbo")
        p_normal = power_consumption_w(100.0, "normal")
        assert abs(p - p_normal) < 1e-10


# =====================================================================
# Electrode degradation
# =====================================================================

class TestElectrodeDegradation:
    def test_zero_health_stays_zero(self):
        assert electrode_degradation(0.0) == 0.0

    def test_one_sol_small_change(self):
        h = electrode_degradation(1.0, sol_count=1)
        assert 0.999 < h < 1.0

    def test_monotonic_decrease(self):
        h = 1.0
        for _ in range(100):
            h_new = electrode_degradation(h)
            assert h_new <= h
            h = h_new

    def test_never_negative(self):
        h = electrode_degradation(0.001, sol_count=100)
        assert h >= 0.0

    def test_cold_accelerates(self):
        """Extreme cold increases degradation rate."""
        h_normal = electrode_degradation(1.0, 1, -40.0)
        h_cold = electrode_degradation(1.0, 1, -100.0)
        assert h_cold < h_normal

    def test_hot_accelerates(self):
        """Extreme heat increases degradation rate."""
        h_normal = electrode_degradation(1.0, 1, 0.0)
        h_hot = electrode_degradation(1.0, 1, 60.0)
        assert h_hot < h_normal

    def test_multi_sol(self):
        """More sols = more degradation."""
        h1 = electrode_degradation(1.0, 1)
        h10 = electrode_degradation(1.0, 10)
        assert h10 < h1


# =====================================================================
# Net dust change
# =====================================================================

class TestNetDustChange:
    def test_no_cleaning_positive(self):
        """Without cleaning, dust accumulates."""
        delta = net_dust_change(0.0, NORMAL_DEPOSITION_RATE, 0.0)
        assert delta > 0

    def test_high_cleaning_negative(self):
        """With good cleaning, dust decreases."""
        delta = net_dust_change(0.5, NORMAL_DEPOSITION_RATE, 0.9)
        assert delta < 0

    def test_clean_surface_no_removal(self):
        """Can't remove dust from a clean surface."""
        delta = net_dust_change(0.0, NORMAL_DEPOSITION_RATE, 0.9)
        assert delta > 0  # only deposition

    def test_storm_overwhelms_cleaning(self):
        """During storm with weak shield, deposition wins."""
        delta = net_dust_change(0.01, STORM_DEPOSITION_RATE, 0.1)
        assert delta > 0  # storm deposition beats weak cleaning on low dust


# =====================================================================
# DustShieldState clamping
# =====================================================================

class TestDustShieldState:
    def test_defaults(self):
        s = DustShieldState()
        assert s.area_m2 == 100.0
        assert s.electrode_health == 1.0
        assert s.dust_fraction == 0.0

    def test_negative_area_clamped(self):
        s = DustShieldState(area_m2=-50.0)
        assert s.area_m2 == 0.0

    def test_health_clamped_high(self):
        s = DustShieldState(electrode_health=2.0)
        assert s.electrode_health == 1.0

    def test_health_clamped_low(self):
        s = DustShieldState(electrode_health=-0.5)
        assert s.electrode_health == 0.0

    def test_dust_clamped(self):
        s = DustShieldState(dust_fraction=1.5)
        assert s.dust_fraction == 1.0
        s2 = DustShieldState(dust_fraction=-0.1)
        assert s2.dust_fraction == 0.0

    def test_temp_clamped(self):
        s = DustShieldState(temperature_c=-200.0)
        assert s.temperature_c == ITO_MIN_TEMP_C
        s2 = DustShieldState(temperature_c=200.0)
        assert s2.temperature_c == ITO_MAX_TEMP_C

    def test_field_clamped(self):
        s = DustShieldState(field_v_m=-1000.0)
        assert s.field_v_m == 0.0
        s2 = DustShieldState(field_v_m=1e9)
        assert s2.field_v_m == MAX_FIELD_V_M

    def test_invalid_mode_defaults(self):
        s = DustShieldState(mode="turbo")
        assert s.mode == "normal"

    def test_zones_minimum_one(self):
        s = DustShieldState(coverage_zones=0)
        assert s.coverage_zones == 1


# =====================================================================
# Tick function integration
# =====================================================================

class TestTickDustShield:
    def test_basic_tick(self):
        state = DustShieldState()
        result = tick_dust_shield(state)
        assert result.dust_after >= 0.0
        assert result.power_used_wh >= 0.0
        assert state.sol == 1

    def test_dust_bounded_after_tick(self):
        state = DustShieldState(dust_fraction=0.99)
        result = tick_dust_shield(state, storm_active=True)
        assert 0.0 <= state.dust_fraction <= 1.0

    def test_clean_surface_gains_dust(self):
        """Even with shield, some dust deposits."""
        state = DustShieldState(dust_fraction=0.0)
        result = tick_dust_shield(state)
        assert result.dust_deposited > 0

    def test_storm_increases_deposition(self):
        s1 = DustShieldState(dust_fraction=0.1)
        s2 = DustShieldState(dust_fraction=0.1)
        r_calm = tick_dust_shield(s1, storm_active=False)
        r_storm = tick_dust_shield(s2, storm_active=True)
        assert r_storm.dust_deposited > r_calm.dust_deposited

    def test_storm_triggers_burst_mode(self):
        state = DustShieldState()
        tick_dust_shield(state, storm_active=True)
        assert state.mode == "burst"

    def test_low_dust_triggers_standby(self):
        state = DustShieldState(dust_fraction=0.01)
        tick_dust_shield(state, storm_active=False)
        assert state.mode == "standby"

    def test_medium_dust_normal_mode(self):
        state = DustShieldState(dust_fraction=0.15)
        tick_dust_shield(state, storm_active=False)
        assert state.mode == "normal"

    def test_high_dust_triggers_burst(self):
        state = DustShieldState(dust_fraction=0.5)
        tick_dust_shield(state, storm_active=False)
        assert state.mode == "burst"

    def test_electrode_degrades_each_sol(self):
        state = DustShieldState()
        h_before = state.electrode_health
        tick_dust_shield(state)
        assert state.electrode_health < h_before

    def test_energy_accumulates(self):
        state = DustShieldState()
        tick_dust_shield(state)
        e1 = state.total_energy_wh
        tick_dust_shield(state)
        assert state.total_energy_wh > e1

    def test_temperature_override(self):
        state = DustShieldState(temperature_c=-40.0)
        tick_dust_shield(state, temperature_c=-80.0)
        assert state.temperature_c == -80.0

    def test_inactive_shield_no_cleaning(self):
        state = DustShieldState(dust_fraction=0.5, active=False)
        result = tick_dust_shield(state)
        assert result.cleaning_efficiency == 0.0
        assert result.dust_after > result.dust_before

    def test_dead_electrode_warning(self):
        state = DustShieldState(electrode_health=ITO_MIN_HEALTH + 0.0001)
        result = tick_dust_shield(state)
        assert "electrode" in result.warning

    def test_sol_counter_increments(self):
        state = DustShieldState()
        for i in range(5):
            tick_dust_shield(state)
        assert state.sol == 5


# =====================================================================
# Multi-sol smoke tests
# =====================================================================

class TestSmoke:
    def test_365_sols_no_crash(self):
        """Run the shield for a full Mars year without crashing."""
        state = create_dust_shield("solar_panels")
        for sol in range(365):
            storm = (100 <= sol <= 130)  # 30-sol storm mid-year
            tick_dust_shield(state, storm_active=storm)
        assert state.sol == 365
        assert 0.0 <= state.dust_fraction <= 1.0
        assert state.electrode_health >= 0.0
        assert state.total_energy_wh > 0

    def test_shield_recovers_after_storm(self):
        """Dust drops back down after storm ends."""
        state = create_dust_shield("solar_panels")
        # Pre-storm baseline
        for _ in range(10):
            tick_dust_shield(state, storm_active=False)
        pre_storm_dust = state.dust_fraction

        # Storm hits
        for _ in range(20):
            tick_dust_shield(state, storm_active=True)
        storm_peak = state.dust_fraction
        assert storm_peak > pre_storm_dust

        # Recovery
        for _ in range(30):
            tick_dust_shield(state, storm_active=False)
        post_recovery = state.dust_fraction
        assert post_recovery < storm_peak

    def test_electrode_end_of_life(self):
        """After enough sols, electrode reaches replacement threshold."""
        state = DustShieldState(electrode_health=1.0)
        sols = 0
        while state.electrode_health > ITO_REPLACEMENT_THRESHOLD and sols < 10000:
            tick_dust_shield(state)
            sols += 1
        assert state.electrode_health <= ITO_REPLACEMENT_THRESHOLD
        assert sols > 100  # should take at least hundreds of sols

    def test_10_sol_quick_smoke(self):
        """Minimal smoke: 10 sols, all physical quantities in bounds."""
        state = create_dust_shield("airlock")
        for _ in range(10):
            result = tick_dust_shield(state)
            assert 0.0 <= result.dust_after <= 1.0
            assert result.power_used_wh >= 0.0
            assert 0.0 <= result.electrode_health <= 1.0
            assert result.cleaning_efficiency >= 0.0


# =====================================================================
# Conservation and invariants
# =====================================================================

class TestInvariants:
    def test_energy_non_negative(self):
        """Total energy consumed is always >= 0."""
        state = create_dust_shield("greenhouse")
        for _ in range(50):
            tick_dust_shield(state, storm_active=True)
        assert state.total_energy_wh >= 0

    def test_dust_mass_non_negative(self):
        """Cumulative dust removed is never negative."""
        state = DustShieldState(dust_fraction=0.5)
        for _ in range(100):
            tick_dust_shield(state)
        assert state.dust_removed_kg_m2 >= 0

    def test_dust_fraction_bounded(self):
        """Dust fraction stays in [0, 1] under all conditions."""
        for dust in [0.0, 0.5, 0.99]:
            for storm in [True, False]:
                for health in [0.05, 0.5, 1.0]:
                    state = DustShieldState(
                        dust_fraction=dust,
                        electrode_health=health
                    )
                    tick_dust_shield(state, storm_active=storm)
                    assert 0.0 <= state.dust_fraction <= 1.0

    def test_health_monotonically_decreasing(self):
        """Electrode health never increases (no repair in tick)."""
        state = DustShieldState()
        prev_health = state.electrode_health
        for _ in range(50):
            tick_dust_shield(state)
            assert state.electrode_health <= prev_health
            prev_health = state.electrode_health


# =====================================================================
# Factory scenarios
# =====================================================================

class TestFactory:
    def test_solar_panels(self):
        s = create_dust_shield("solar_panels")
        assert s.area_m2 == 500.0
        assert s.coverage_zones == 10

    def test_airlock(self):
        s = create_dust_shield("airlock")
        assert s.area_m2 == 10.0
        assert s.coverage_zones == 2

    def test_greenhouse(self):
        s = create_dust_shield("greenhouse")
        assert s.area_m2 == 50.0

    def test_visor(self):
        s = create_dust_shield("visor")
        assert s.area_m2 == 0.05

    def test_unknown_defaults_to_solar(self):
        s = create_dust_shield("martian_palace")
        assert s.area_m2 == 500.0

    def test_all_scenarios_valid_state(self):
        """Every factory scenario produces valid state."""
        for scenario in ["solar_panels", "airlock", "greenhouse", "visor"]:
            s = create_dust_shield(scenario)
            assert s.area_m2 > 0
            assert s.electrode_health == 1.0
            assert s.dust_fraction == 0.0
            assert s.active is True
