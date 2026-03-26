"""Tests for wind_turbine.py — Mars wind power generation.

Covers physics bounds, Betz limit, power curve, erosion, maintenance,
multi-turbine farms, and multi-sol simulation smoke tests.
"""
from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from wind_turbine import (
    # Constants
    MARS_AIR_DENSITY_KG_M3, MARS_GAS_CONSTANT_CO2,
    MARS_SURFACE_PRESSURE_PA, MARS_MEAN_TEMP_K,
    BETZ_LIMIT, PRACTICAL_CP, GENERATOR_EFFICIENCY, GEARBOX_EFFICIENCY,
    CUT_IN_SPEED_M_S, RATED_SPEED_M_S, CUT_OUT_SPEED_M_S,
    BLADE_EROSION_RATE_PER_SOL, BLADE_EROSION_RATE_CALM,
    MAX_BLADE_EROSION, SOL_HOURS, MAINTENANCE_INTERVAL_SOLS,
    MAINTENANCE_EFFICIENCY_RESTORE,
    # Functions
    air_density, swept_area, wind_power_available_w,
    turbine_power_output_w, rated_power_w, capacity_factor,
    wind_at_height, tick,
    # Dataclasses
    WindTurbine, WindConditions, TickResult, WindFarm,
    # Factories
    create_turbine, create_storm_conditions, create_calm_conditions,
    create_wind_farm,
)


# ═══════════════════════════════════════════════════════════════════════════
# Physical constants validation
# ═══════════════════════════════════════════════════════════════════════════

class TestConstants:
    """Constants must be in physically meaningful ranges."""

    def test_mars_air_density_range(self):
        """Mars air density ~0.020 kg/m³ (1-2% of Earth's 1.225)."""
        assert 0.005 < MARS_AIR_DENSITY_KG_M3 < 0.05

    def test_betz_limit_value(self):
        """Betz limit is exactly 16/27."""
        assert abs(BETZ_LIMIT - 16.0 / 27.0) < 1e-10

    def test_betz_limit_less_than_one(self):
        assert 0.0 < BETZ_LIMIT < 1.0

    def test_practical_cp_below_betz(self):
        """No real turbine exceeds Betz limit."""
        assert 0.0 < PRACTICAL_CP < BETZ_LIMIT

    def test_cut_in_below_rated(self):
        assert CUT_IN_SPEED_M_S < RATED_SPEED_M_S

    def test_rated_below_cut_out(self):
        assert RATED_SPEED_M_S < CUT_OUT_SPEED_M_S

    def test_generator_efficiency_valid(self):
        assert 0.0 < GENERATOR_EFFICIENCY <= 1.0

    def test_gearbox_efficiency_valid(self):
        assert 0.0 < GEARBOX_EFFICIENCY <= 1.0

    def test_sol_hours_near_24_66(self):
        """Mars sol is 24h 39m 35s ≈ 24.66 hours."""
        assert abs(SOL_HOURS - 24.66) < 0.1

    def test_mars_pressure_range(self):
        """Mars surface pressure ~600 Pa (0.6% of Earth)."""
        assert 400 < MARS_SURFACE_PRESSURE_PA < 900

    def test_mars_temp_range(self):
        """Mars mean temp ~210 K (-63°C)."""
        assert 150 < MARS_MEAN_TEMP_K < 270

    def test_erosion_rates_positive(self):
        assert BLADE_EROSION_RATE_PER_SOL > 0
        assert BLADE_EROSION_RATE_CALM > 0

    def test_storm_erosion_worse_than_calm(self):
        assert BLADE_EROSION_RATE_PER_SOL > BLADE_EROSION_RATE_CALM

    def test_max_erosion_bound(self):
        assert 0.0 < MAX_BLADE_EROSION < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Air density model
# ═══════════════════════════════════════════════════════════════════════════

class TestAirDensity:
    """Ideal gas law for CO₂ atmosphere."""

    def test_standard_conditions(self):
        """Standard Mars conditions give ~0.020 kg/m³."""
        rho = air_density(MARS_SURFACE_PRESSURE_PA, MARS_MEAN_TEMP_K)
        assert abs(rho - 0.0154) < 0.005  # ~0.015-0.020

    def test_zero_pressure(self):
        assert air_density(0.0, 210.0) == 0.0

    def test_zero_temperature(self):
        assert air_density(610.0, 0.0) == 0.0

    def test_negative_pressure(self):
        assert air_density(-100.0, 210.0) == 0.0

    def test_negative_temperature(self):
        assert air_density(610.0, -50.0) == 0.0

    def test_higher_pressure_higher_density(self):
        rho_low = air_density(400.0, 210.0)
        rho_high = air_density(800.0, 210.0)
        assert rho_high > rho_low

    def test_higher_temp_lower_density(self):
        rho_cold = air_density(610.0, 180.0)
        rho_hot = air_density(610.0, 250.0)
        assert rho_cold > rho_hot

    def test_clamped_below_one(self):
        """Even extreme pressure can't exceed 1 kg/m³ (safety clamp)."""
        rho = air_density(1e6, 100.0)
        assert rho <= 1.0

    def test_ideal_gas_formula(self):
        """Direct check: ρ = P / (R·T)."""
        p, t = 610.0, 210.0
        expected = p / (MARS_GAS_CONSTANT_CO2 * t)
        assert abs(air_density(p, t) - expected) < 1e-8


# ═══════════════════════════════════════════════════════════════════════════
# Swept area
# ═══════════════════════════════════════════════════════════════════════════

class TestSweptArea:

    def test_known_radius(self):
        """5m radius → π·25 ≈ 78.54 m²."""
        assert abs(swept_area(5.0) - math.pi * 25.0) < 0.01

    def test_zero_radius(self):
        assert swept_area(0.0) == 0.0

    def test_negative_radius(self):
        assert swept_area(-3.0) == 0.0

    def test_scales_quadratically(self):
        """Doubling radius quadruples area."""
        a1 = swept_area(5.0)
        a2 = swept_area(10.0)
        assert abs(a2 / a1 - 4.0) < 0.01

    def test_always_non_negative(self):
        for r in [0, 0.1, 1, 10, 100]:
            assert swept_area(r) >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Wind power available
# ═══════════════════════════════════════════════════════════════════════════

class TestWindPowerAvailable:

    def test_positive_conditions(self):
        p = wind_power_available_w(0.02, 78.5, 10.0)
        # 0.5 * 0.02 * 78.5 * 1000 = 785 W
        assert abs(p - 785.0) < 1.0

    def test_zero_density(self):
        assert wind_power_available_w(0.0, 78.5, 10.0) == 0.0

    def test_zero_area(self):
        assert wind_power_available_w(0.02, 0.0, 10.0) == 0.0

    def test_zero_wind(self):
        assert wind_power_available_w(0.02, 78.5, 0.0) == 0.0

    def test_negative_density(self):
        assert wind_power_available_w(-0.02, 78.5, 10.0) == 0.0

    def test_cubic_scaling(self):
        """Power scales with v³: doubling wind → 8x power."""
        p1 = wind_power_available_w(0.02, 78.5, 5.0)
        p2 = wind_power_available_w(0.02, 78.5, 10.0)
        assert abs(p2 / p1 - 8.0) < 0.01

    def test_linear_with_density(self):
        p1 = wind_power_available_w(0.01, 78.5, 10.0)
        p2 = wind_power_available_w(0.02, 78.5, 10.0)
        assert abs(p2 / p1 - 2.0) < 0.01

    def test_linear_with_area(self):
        p1 = wind_power_available_w(0.02, 50.0, 10.0)
        p2 = wind_power_available_w(0.02, 100.0, 10.0)
        assert abs(p2 / p1 - 2.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Turbine power output
# ═══════════════════════════════════════════════════════════════════════════

class TestTurbinePowerOutput:

    def test_below_cut_in_is_zero(self):
        assert turbine_power_output_w(CUT_IN_SPEED_M_S - 0.1, 5.0) == 0.0

    def test_at_cut_in_is_positive(self):
        assert turbine_power_output_w(CUT_IN_SPEED_M_S, 5.0) > 0.0

    def test_above_cut_out_is_zero(self):
        assert turbine_power_output_w(CUT_OUT_SPEED_M_S + 0.1, 5.0) == 0.0

    def test_at_cut_out_is_positive(self):
        assert turbine_power_output_w(CUT_OUT_SPEED_M_S, 5.0) > 0.0

    def test_power_increases_with_wind_below_rated(self):
        p1 = turbine_power_output_w(5.0, 5.0)
        p2 = turbine_power_output_w(15.0, 5.0)
        assert p2 > p1

    def test_power_capped_above_rated(self):
        """Between rated and cut-out, power stays at rated level."""
        p_rated = turbine_power_output_w(RATED_SPEED_M_S, 5.0)
        p_above = turbine_power_output_w(RATED_SPEED_M_S + 5.0, 5.0)
        assert abs(p_rated - p_above) < 0.1

    def test_never_exceeds_betz(self):
        """Output can never exceed Betz limit of available wind power."""
        v = 15.0
        area = swept_area(5.0)
        p_available = wind_power_available_w(MARS_AIR_DENSITY_KG_M3, area, v)
        p_output = turbine_power_output_w(v, 5.0)
        assert p_output < p_available * BETZ_LIMIT

    def test_erosion_reduces_output(self):
        p_clean = turbine_power_output_w(15.0, 5.0, blade_erosion=0.0)
        p_eroded = turbine_power_output_w(15.0, 5.0, blade_erosion=0.15)
        assert p_eroded < p_clean

    def test_max_erosion_still_produces_some_power(self):
        p = turbine_power_output_w(15.0, 5.0, blade_erosion=MAX_BLADE_EROSION)
        assert p > 0.0

    def test_larger_rotor_more_power(self):
        p_small = turbine_power_output_w(15.0, 3.0)
        p_large = turbine_power_output_w(15.0, 8.0)
        assert p_large > p_small

    def test_higher_density_more_power(self):
        p_thin = turbine_power_output_w(15.0, 5.0, density_kg_m3=0.01)
        p_thick = turbine_power_output_w(15.0, 5.0, density_kg_m3=0.03)
        assert p_thick > p_thin

    def test_always_non_negative(self):
        for v in [0, 1, 3, 10, 25, 30, 45, 50, 100]:
            assert turbine_power_output_w(v, 5.0) >= 0.0

    def test_cp_capped_at_betz(self):
        """Passing Cp > Betz should be clamped."""
        p = turbine_power_output_w(15.0, 5.0, cp=0.99)
        p_betz = turbine_power_output_w(15.0, 5.0, cp=BETZ_LIMIT)
        assert abs(p - p_betz) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Rated power
# ═══════════════════════════════════════════════════════════════════════════

class TestRatedPower:

    def test_positive(self):
        assert rated_power_w(5.0) > 0.0

    def test_larger_rotor_higher_rated(self):
        assert rated_power_w(10.0) > rated_power_w(5.0)

    def test_matches_output_at_rated_speed(self):
        rp = rated_power_w(5.0)
        op = turbine_power_output_w(RATED_SPEED_M_S, 5.0)
        assert abs(rp - op) < 0.01

    def test_mars_5m_rotor_order_of_magnitude(self):
        """A 5m rotor on Mars at 25 m/s should produce ~order of 1 kW."""
        rp = rated_power_w(5.0)
        assert 10 < rp < 10000  # watts, very broad check


# ═══════════════════════════════════════════════════════════════════════════
# Capacity factor
# ═══════════════════════════════════════════════════════════════════════════

class TestCapacityFactor:

    def test_zero_rated(self):
        assert capacity_factor(100.0, 0.0) == 0.0

    def test_full_rated(self):
        assert abs(capacity_factor(100.0, 100.0) - 1.0) < 1e-10

    def test_half_power(self):
        assert abs(capacity_factor(50.0, 100.0) - 0.5) < 1e-10

    def test_clamped_to_one(self):
        assert capacity_factor(200.0, 100.0) == 1.0

    def test_non_negative(self):
        assert capacity_factor(-10.0, 100.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Wind at height (power law)
# ═══════════════════════════════════════════════════════════════════════════

class TestWindAtHeight:

    def test_same_height_same_speed(self):
        """At reference height, speed unchanged."""
        assert abs(wind_at_height(10.0, 2.0, 2.0) - 10.0) < 0.01

    def test_higher_means_faster(self):
        v_low = wind_at_height(10.0, 2.0, 5.0)
        v_high = wind_at_height(10.0, 2.0, 15.0)
        assert v_high > v_low

    def test_zero_reference_speed(self):
        assert wind_at_height(0.0, 2.0, 15.0) == 0.0

    def test_zero_reference_height(self):
        assert wind_at_height(10.0, 0.0, 15.0) == 0.0

    def test_zero_target_height(self):
        assert wind_at_height(10.0, 2.0, 0.0) == 0.0

    def test_known_value(self):
        """v(15) = 10 * (15/2)^0.20 ≈ 10 * 1.512 ≈ 15.12"""
        v = wind_at_height(10.0, 2.0, 15.0, alpha=0.20)
        expected = 10.0 * (15.0 / 2.0) ** 0.20
        assert abs(v - expected) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# WindTurbine dataclass
# ═══════════════════════════════════════════════════════════════════════════

class TestWindTurbineDataclass:

    def test_defaults(self):
        t = WindTurbine()
        assert t.rotor_radius_m == 5.0
        assert t.hub_height_m == 15.0
        assert t.blade_erosion == 0.0
        assert t.operational is True
        assert t.sol == 0

    def test_clamp_negative_radius(self):
        t = WindTurbine(rotor_radius_m=-10.0)
        assert t.rotor_radius_m == 0.1

    def test_clamp_negative_height(self):
        t = WindTurbine(hub_height_m=-5.0)
        assert t.hub_height_m == 1.0

    def test_clamp_erosion_overflow(self):
        t = WindTurbine(blade_erosion=0.99)
        assert t.blade_erosion == MAX_BLADE_EROSION

    def test_clamp_negative_energy(self):
        t = WindTurbine(total_energy_kwh=-100)
        assert t.total_energy_kwh == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# WindConditions dataclass
# ═══════════════════════════════════════════════════════════════════════════

class TestWindConditionsDataclass:

    def test_defaults(self):
        c = WindConditions()
        assert c.wind_speed_m_s == 5.0
        assert not c.dust_storm_active

    def test_clamp_negative_wind(self):
        c = WindConditions(wind_speed_m_s=-10.0)
        assert c.wind_speed_m_s == 0.0

    def test_clamp_negative_pressure(self):
        c = WindConditions(pressure_pa=-100.0)
        assert c.pressure_pa == 0.0

    def test_clamp_negative_temp(self):
        c = WindConditions(temperature_k=-50.0)
        assert c.temperature_k == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Tick function
# ═══════════════════════════════════════════════════════════════════════════

class TestTick:

    def test_calm_wind_produces_power(self):
        t = create_turbine()
        c = create_calm_conditions(wind_speed_m_s=10.0)
        r = tick(t, c)
        assert r.power_w > 0.0
        assert r.energy_kwh > 0.0

    def test_no_wind_no_power(self):
        t = create_turbine()
        c = WindConditions(wind_speed_m_s=0.0)
        r = tick(t, c)
        assert r.power_w == 0.0
        assert r.energy_kwh == 0.0

    def test_sol_increments(self):
        t = create_turbine()
        c = create_calm_conditions()
        assert t.sol == 0
        tick(t, c)
        assert t.sol == 1
        tick(t, c)
        assert t.sol == 2

    def test_energy_accumulates(self):
        t = create_turbine()
        c = create_calm_conditions(wind_speed_m_s=15.0)
        tick(t, c)
        e1 = t.total_energy_kwh
        tick(t, c)
        e2 = t.total_energy_kwh
        assert e2 > e1

    def test_storm_causes_more_erosion(self):
        t_calm = create_turbine()
        t_storm = create_turbine()
        c_calm = create_calm_conditions(wind_speed_m_s=10.0)
        c_storm = create_storm_conditions(wind_speed_m_s=10.0)
        r_calm = tick(t_calm, c_calm)
        r_storm = tick(t_storm, c_storm)
        assert r_storm.blade_erosion_delta > r_calm.blade_erosion_delta

    def test_feathered_above_cut_out(self):
        """Hub-height wind above cut-out → feathered, no power."""
        t = create_turbine(hub_height_m=2.0)  # same as reference
        c = WindConditions(wind_speed_m_s=CUT_OUT_SPEED_M_S + 5.0)
        r = tick(t, c, reference_wind_height_m=2.0)
        assert r.feathered is True
        assert r.power_w == 0.0

    def test_non_operational_no_power(self):
        t = create_turbine()
        t.operational = False
        c = create_calm_conditions(wind_speed_m_s=15.0)
        r = tick(t, c)
        assert r.power_w == 0.0
        assert t.sol == 1  # still increments

    def test_maintenance_triggers(self):
        t = create_turbine()
        t.sols_since_maintenance = MAINTENANCE_INTERVAL_SOLS - 1
        t.blade_erosion = 0.10
        c = create_calm_conditions(wind_speed_m_s=10.0)
        r = tick(t, c)
        assert r.maintenance_performed is True
        assert t.sols_since_maintenance == 0
        assert t.blade_erosion < 0.10

    def test_maintenance_restores_efficiency(self):
        t = create_turbine()
        t.blade_erosion = 0.15
        t.sols_since_maintenance = MAINTENANCE_INTERVAL_SOLS - 1
        erosion_before = t.blade_erosion
        c = create_calm_conditions(wind_speed_m_s=10.0)
        tick(t, c)
        assert t.blade_erosion < erosion_before

    def test_air_density_in_result(self):
        t = create_turbine()
        c = WindConditions(pressure_pa=700.0, temperature_k=200.0)
        r = tick(t, c)
        expected_rho = air_density(700.0, 200.0)
        assert abs(r.air_density_kg_m3 - expected_rho) < 1e-8

    def test_capacity_factor_in_bounds(self):
        t = create_turbine()
        c = create_calm_conditions(wind_speed_m_s=15.0)
        r = tick(t, c)
        assert 0.0 <= r.capacity_factor <= 1.0

    def test_energy_equals_power_times_sol_hours(self):
        t = create_turbine()
        c = create_calm_conditions(wind_speed_m_s=15.0)
        r = tick(t, c)
        expected = r.power_w * SOL_HOURS / 1000.0
        assert abs(r.energy_kwh - expected) < 0.001


# ═══════════════════════════════════════════════════════════════════════════
# Conservation laws / property-based invariants
# ═══════════════════════════════════════════════════════════════════════════

class TestInvariants:
    """Physical invariants that must hold across any simulation."""

    def test_power_never_negative(self):
        t = create_turbine()
        for v in [0, 2, 5, 10, 20, 30, 45, 50, 100]:
            c = WindConditions(wind_speed_m_s=v)
            r = tick(t, c)
            assert r.power_w >= 0.0
            assert r.energy_kwh >= 0.0

    def test_energy_monotonically_increases(self):
        t = create_turbine()
        c = create_calm_conditions(wind_speed_m_s=10.0)
        prev_energy = 0.0
        for _ in range(50):
            tick(t, c)
            assert t.total_energy_kwh >= prev_energy
            prev_energy = t.total_energy_kwh

    def test_erosion_bounded(self):
        """Erosion never exceeds MAX_BLADE_EROSION."""
        t = create_turbine()
        c = create_storm_conditions(wind_speed_m_s=20.0)
        for _ in range(10000):
            tick(t, c)
        assert t.blade_erosion <= MAX_BLADE_EROSION

    def test_output_below_betz_of_available(self):
        """At any wind speed, output < Betz fraction of available power."""
        t = create_turbine()
        for v in [5.0, 10.0, 15.0, 20.0, 25.0]:
            c = WindConditions(wind_speed_m_s=v)
            r = tick(t, c)
            rho = air_density(c.pressure_pa, c.temperature_k)
            hub_v = wind_at_height(v, 2.0, t.hub_height_m)
            available = wind_power_available_w(
                rho, swept_area(t.rotor_radius_m), hub_v
            )
            if available > 0:
                assert r.power_w <= available * BETZ_LIMIT * 1.01  # 1% tolerance

    def test_sol_always_advances(self):
        t = create_turbine()
        for i in range(20):
            c = WindConditions(wind_speed_m_s=float(i * 3))
            tick(t, c)
            assert t.sol == i + 1


# ═══════════════════════════════════════════════════════════════════════════
# Factory helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestFactories:

    def test_create_turbine_defaults(self):
        t = create_turbine()
        assert t.rotor_radius_m == 5.0
        assert t.hub_height_m == 15.0
        assert t.operational is True

    def test_create_turbine_custom(self):
        t = create_turbine(rotor_radius_m=8.0, hub_height_m=20.0)
        assert t.rotor_radius_m == 8.0
        assert t.hub_height_m == 20.0

    def test_create_storm_conditions(self):
        c = create_storm_conditions()
        assert c.dust_storm_active is True
        assert c.wind_speed_m_s == 25.0

    def test_create_calm_conditions(self):
        c = create_calm_conditions()
        assert c.dust_storm_active is False
        assert c.wind_speed_m_s == 5.0


# ═══════════════════════════════════════════════════════════════════════════
# Wind farm
# ═══════════════════════════════════════════════════════════════════════════

class TestWindFarm:

    def test_create_farm(self):
        farm = create_wind_farm(num_turbines=4)
        assert len(farm.turbines) == 4

    def test_farm_rated_power(self):
        farm = create_wind_farm(num_turbines=4)
        single_rated = rated_power_w(5.0)
        assert abs(farm.total_rated_power_w() - 4 * single_rated) < 0.01

    def test_farm_tick_all(self):
        farm = create_wind_farm(num_turbines=3)
        c = create_calm_conditions(wind_speed_m_s=15.0)
        results = farm.tick_all(c)
        assert len(results) == 3
        for r in results:
            assert r.power_w > 0.0

    def test_wake_loss_reduces_output(self):
        farm = create_wind_farm(num_turbines=2)
        c = create_calm_conditions(wind_speed_m_s=15.0)
        results_wake = farm.tick_all(c)

        farm_no_wake = create_wind_farm(num_turbines=2)
        farm_no_wake.wake_loss_fraction = 0.0
        results_no = farm_no_wake.tick_all(c)

        total_wake = farm.total_power_w(results_wake)
        total_no = farm_no_wake.total_power_w(results_no)
        assert total_wake < total_no

    def test_farm_total_energy(self):
        farm = create_wind_farm(num_turbines=2)
        c = create_calm_conditions(wind_speed_m_s=15.0)
        results = farm.tick_all(c)
        total = farm.total_energy_kwh(results)
        assert total > 0.0

    def test_wake_loss_clamped(self):
        farm = WindFarm(wake_loss_fraction=-0.5)
        assert farm.wake_loss_fraction == 0.0
        farm2 = WindFarm(wake_loss_fraction=0.9)
        assert farm2.wake_loss_fraction == 0.5

    def test_minimum_one_turbine(self):
        farm = create_wind_farm(num_turbines=0)
        assert len(farm.turbines) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Multi-sol smoke test
# ═══════════════════════════════════════════════════════════════════════════

class TestSmoke:
    """Run simulation for extended periods without crash."""

    def test_100_sols_single_turbine(self):
        t = create_turbine()
        for sol in range(100):
            wind = 5.0 + (sol % 20) * 1.5  # varying wind
            storm = sol > 60 and sol < 80    # storm in middle
            c = WindConditions(
                wind_speed_m_s=wind,
                dust_storm_active=storm,
            )
            r = tick(t, c)
            assert r.power_w >= 0.0
            assert r.energy_kwh >= 0.0
        assert t.sol == 100
        assert t.total_energy_kwh > 0.0

    def test_500_sols_farm(self):
        farm = create_wind_farm(num_turbines=4)
        total_energy = 0.0
        for sol in range(500):
            wind = 3.0 + (sol % 30) * 1.0
            storm = (sol % 100) > 70
            c = WindConditions(
                wind_speed_m_s=wind,
                dust_storm_active=storm,
            )
            results = farm.tick_all(c)
            sol_energy = farm.total_energy_kwh(results)
            total_energy += sol_energy
            assert sol_energy >= 0.0
        assert total_energy > 0.0

    def test_extreme_conditions_no_crash(self):
        """Edge cases: near-vacuum, near-zero temp, extreme wind."""
        t = create_turbine()
        extremes = [
            WindConditions(wind_speed_m_s=0.0, pressure_pa=0.0, temperature_k=0.0),
            WindConditions(wind_speed_m_s=200.0, pressure_pa=1000.0, temperature_k=300.0),
            WindConditions(wind_speed_m_s=CUT_OUT_SPEED_M_S, pressure_pa=100.0, temperature_k=100.0),
            WindConditions(wind_speed_m_s=CUT_IN_SPEED_M_S, pressure_pa=1.0, temperature_k=400.0),
        ]
        for c in extremes:
            r = tick(t, c)
            assert r.power_w >= 0.0
            assert r.energy_kwh >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Mars vs Earth comparison (sanity check)
# ═══════════════════════════════════════════════════════════════════════════

class TestMarsVsEarth:
    """Verify Mars turbines produce much less than Earth equivalents."""

    def test_mars_output_fraction_of_earth(self):
        """Same turbine on Mars should produce ~1-2% of Earth output."""
        earth_density = 1.225
        mars_density = MARS_AIR_DENSITY_KG_M3
        wind = 15.0

        p_earth = turbine_power_output_w(wind, 5.0, density_kg_m3=earth_density)
        p_mars = turbine_power_output_w(wind, 5.0, density_kg_m3=mars_density)

        ratio = p_mars / p_earth
        assert ratio < 0.05   # less than 5% of Earth output
        assert ratio > 0.001  # but not zero

    def test_storm_wind_compensates_partially(self):
        """During storms (30 m/s), Mars turbines produce more than calm (5 m/s)."""
        p_calm = turbine_power_output_w(5.0, 5.0)
        p_storm = turbine_power_output_w(RATED_SPEED_M_S, 5.0)
        assert p_storm > p_calm * 10  # cubic scaling helps a lot
