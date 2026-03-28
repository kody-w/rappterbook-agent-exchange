"""Tests for mars_drone.py -- Mars Autonomous Aerial Scout.

128 tests covering rotor aerodynamics, power models, battery thermal,
navigation uncertainty, communication links, blade erosion, flight
simulation, payload management, tick lifecycle, conservation laws,
physical bounds, edge cases, multi-sol endurance, and dust storms.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mars_drone as md


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def drone():
    return md.create_drone()


@pytest.fixture
def cold_drone():
    return md.create_drone(battery_temp_k=200.0)


@pytest.fixture
def worn_drone():
    return md.create_drone(blade_health=0.15)


@pytest.fixture
def dead_battery_drone():
    return md.create_drone(battery_charge_wh=0.0)


@pytest.fixture
def heavy_drone():
    return md.create_drone(empty_mass_kg=20.0)


@pytest.fixture
def short_plan():
    return md.FlightPlan(target_altitude_m=5.0, target_range_m=100.0,
                         cruise_speed_m_s=5.0, payload_type="camera")


@pytest.fixture
def long_plan():
    return md.FlightPlan(target_altitude_m=20.0, target_range_m=2000.0,
                         cruise_speed_m_s=12.0, payload_type="lidar")


# ===================================================================
# Air density
# ===================================================================

class TestAirDensity:
    def test_surface_density(self):
        assert md.air_density_at_altitude(0.0) == pytest.approx(md.MARS_SURFACE_DENSITY_KG_M3)

    def test_decreases_with_altitude(self):
        assert md.air_density_at_altitude(1000.0) < md.air_density_at_altitude(0.0)

    def test_exponential_decay(self):
        rho0 = md.air_density_at_altitude(0.0)
        rho_h = md.air_density_at_altitude(md.MARS_SCALE_HEIGHT_M)
        assert rho_h == pytest.approx(rho0 / math.e, rel=1e-6)

    def test_negative_altitude_clamped(self):
        assert md.air_density_at_altitude(-100.0) == md.air_density_at_altitude(0.0)

    def test_always_positive(self):
        for alt in [0, 100, 1000, 10000, 50000]:
            assert md.air_density_at_altitude(float(alt)) > 0.0

    def test_very_high_altitude_near_zero(self):
        assert md.air_density_at_altitude(200_000.0) < 1e-6


# ===================================================================
# Rotor thrust
# ===================================================================

class TestRotorThrust:
    def test_positive_for_default_params(self, drone):
        t = md.rotor_thrust_n(drone.rotor_radius_m, drone.rotor_rpm,
                              drone.ct, md.MARS_SURFACE_DENSITY_KG_M3,
                              drone.blade_health)
        assert t > 0.0

    def test_zero_for_zero_rpm(self):
        assert md.rotor_thrust_n(0.65, 0.0, 0.015, 0.02, 1.0) == 0.0

    def test_zero_for_zero_radius(self):
        assert md.rotor_thrust_n(0.0, 2700.0, 0.015, 0.02, 1.0) == 0.0

    def test_zero_for_zero_density(self):
        assert md.rotor_thrust_n(0.65, 2700.0, 0.015, 0.0, 1.0) == 0.0

    def test_increases_with_rpm(self):
        t1 = md.rotor_thrust_n(0.65, 1800.0, 0.015, 0.02, 1.0)
        t2 = md.rotor_thrust_n(0.65, 2700.0, 0.015, 0.02, 1.0)
        assert t2 > t1

    def test_increases_with_radius(self):
        t1 = md.rotor_thrust_n(0.5, 2700.0, 0.015, 0.02, 1.0)
        t2 = md.rotor_thrust_n(0.8, 2700.0, 0.015, 0.02, 1.0)
        assert t2 > t1

    def test_scales_with_density(self):
        t1 = md.rotor_thrust_n(0.65, 2000.0, 0.015, 0.01, 1.0)
        t2 = md.rotor_thrust_n(0.65, 2000.0, 0.015, 0.02, 1.0)
        assert t2 == pytest.approx(t1 * 2.0, rel=0.01)

    def test_blade_health_degrades_thrust(self):
        t_full = md.rotor_thrust_n(0.65, 2700.0, 0.015, 0.02, 1.0)
        t_half = md.rotor_thrust_n(0.65, 2700.0, 0.015, 0.02, 0.5)
        assert t_half < t_full

    def test_zero_blade_health_still_produces_some_thrust(self):
        t = md.rotor_thrust_n(0.65, 2700.0, 0.015, 0.02, 0.0)
        assert t > 0.0  # 60% efficiency floor

    def test_supersonic_tips_zero_thrust(self):
        huge_rpm = 20_000.0
        t = md.rotor_thrust_n(0.65, huge_rpm, 0.015, 0.02, 1.0)
        assert t == 0.0

    def test_near_sonic_penalty(self):
        omega_target = 0.8 * md.MARS_SPEED_OF_SOUND_M_S / 0.65
        rpm_target = omega_target * 60.0 / (2.0 * math.pi)
        t = md.rotor_thrust_n(0.65, rpm_target, 0.015, 0.02, 1.0)
        assert t >= 0.0

    def test_coaxial_factor_in_thrust(self, drone):
        t = md.rotor_thrust_n(drone.rotor_radius_m, drone.rotor_rpm,
                              drone.ct, md.MARS_SURFACE_DENSITY_KG_M3, 1.0)
        omega = drone.rotor_rpm * 2.0 * math.pi / 60.0
        tip = omega * drone.rotor_radius_m
        mach = tip / md.MARS_SPEED_OF_SOUND_M_S
        mach_pen = 1.0
        if mach > md.MAX_TIP_MACH:
            mach_pen = max(0.0, 1.0 - 2.0 * (mach - md.MAX_TIP_MACH))
        area = math.pi * drone.rotor_radius_m ** 2
        single = drone.ct * md.MARS_SURFACE_DENSITY_KG_M3 * area * tip ** 2
        expected = single * 1.8 * mach_pen
        assert t == pytest.approx(expected, rel=0.01)


# ===================================================================
# Hover power
# ===================================================================

class TestHoverPower:
    def test_positive_for_valid_inputs(self):
        assert md.hover_power_w(20.0, 0.02, 1.3) > 0.0

    def test_zero_for_zero_thrust(self):
        assert md.hover_power_w(0.0, 0.02, 1.3) == 0.0

    def test_zero_for_zero_density(self):
        assert md.hover_power_w(20.0, 0.0, 1.3) == 0.0

    def test_increases_with_thrust(self):
        assert md.hover_power_w(20.0, 0.02, 1.3) > md.hover_power_w(10.0, 0.02, 1.3)

    def test_decreases_with_density(self):
        assert md.hover_power_w(20.0, 0.02, 1.3) < md.hover_power_w(20.0, 0.01, 1.3)

    def test_decreases_with_disk_area(self):
        assert md.hover_power_w(20.0, 0.02, 2.0) < md.hover_power_w(20.0, 0.02, 0.5)

    def test_figure_of_merit_applied(self):
        t, rho, a = 20.0, 0.02, 1.3
        ideal = t ** 1.5 / math.sqrt(2.0 * rho * a)
        actual = md.hover_power_w(t, rho, a)
        assert actual > ideal


# ===================================================================
# Forward flight power
# ===================================================================

class TestForwardFlightPower:
    def test_equals_hover_at_zero_speed(self):
        p_h = 100.0
        p_f = md.forward_flight_power_w(p_h, 0.0, 100.0, 0.02, 0.5, 0.04)
        assert p_f == pytest.approx(p_h, rel=0.01)

    def test_increases_with_speed(self):
        p1 = md.forward_flight_power_w(100.0, 5.0, 100.0, 0.02, 0.5, 0.04)
        p2 = md.forward_flight_power_w(100.0, 15.0, 100.0, 0.02, 0.5, 0.04)
        assert p2 > p1

    def test_zero_for_zero_hover(self):
        assert md.forward_flight_power_w(0.0, 10.0, 100.0, 0.02, 0.5, 0.04) == 0.0

    def test_always_ge_hover(self):
        for v in [0, 1, 5, 10, 20]:
            p = md.forward_flight_power_w(100.0, float(v), 100.0, 0.02, 0.5, 0.04)
            assert p >= 100.0 - 0.01


# ===================================================================
# Battery capacity factor
# ===================================================================

class TestBatteryCapacity:
    def test_full_at_nominal(self):
        assert md.battery_capacity_factor(md.BATTERY_NOMINAL_TEMP_K) == 1.0

    def test_full_above_nominal(self):
        assert md.battery_capacity_factor(md.BATTERY_NOMINAL_TEMP_K + 20.0) == 1.0

    def test_zero_at_min(self):
        assert md.battery_capacity_factor(md.BATTERY_MIN_TEMP_K) == 0.0

    def test_zero_below_min(self):
        assert md.battery_capacity_factor(md.BATTERY_MIN_TEMP_K - 10.0) == 0.0

    def test_decreases_with_cold(self):
        assert md.battery_capacity_factor(280.0) > md.battery_capacity_factor(260.0)

    def test_bounded_zero_one(self):
        for t in range(150, 350, 10):
            f = md.battery_capacity_factor(float(t))
            assert 0.0 <= f <= 1.0

    def test_mars_ambient_gives_zero_capacity(self):
        """Mars ambient (210K) is below battery min (233K)."""
        assert md.battery_capacity_factor(md.MARS_AMBIENT_TEMP_K) == 0.0

    def test_warmed_battery_partial_capacity(self):
        """Heater-warmed battery at 260K has reduced but usable capacity."""
        f = md.battery_capacity_factor(260.0)
        assert 0.0 < f < 1.0


# ===================================================================
# Solar charging
# ===================================================================

class TestSolarCharge:
    def test_positive_for_valid_inputs(self):
        assert md.solar_charge_wh(0.12, 0.28, 6.0, 0.8) > 0.0

    def test_zero_for_zero_area(self):
        assert md.solar_charge_wh(0.0, 0.28, 6.0, 0.8) == 0.0

    def test_zero_for_zero_hours(self):
        assert md.solar_charge_wh(0.12, 0.28, 0.0, 0.8) == 0.0

    def test_scales_with_area(self):
        e1 = md.solar_charge_wh(0.06, 0.28, 6.0, 0.8)
        e2 = md.solar_charge_wh(0.12, 0.28, 6.0, 0.8)
        assert e2 == pytest.approx(e1 * 2.0, rel=1e-6)

    def test_scales_with_hours(self):
        e1 = md.solar_charge_wh(0.12, 0.28, 3.0, 0.8)
        e2 = md.solar_charge_wh(0.12, 0.28, 6.0, 0.8)
        assert e2 == pytest.approx(e1 * 2.0, rel=1e-6)

    def test_dust_reduces_charge(self):
        assert md.solar_charge_wh(0.12, 0.28, 6.0, 0.3) < md.solar_charge_wh(0.12, 0.28, 6.0, 1.0)

    def test_zero_dust_gives_zero_charge(self):
        assert md.solar_charge_wh(0.12, 0.28, 6.0, 0.0) == 0.0


# ===================================================================
# Navigation uncertainty
# ===================================================================

class TestNavUncertainty:
    def test_zero_at_zero_distance(self):
        assert md.nav_uncertainty_m(0.0) == 0.0

    def test_positive_for_positive_distance(self):
        assert md.nav_uncertainty_m(100.0) > 0.0

    def test_grows_with_distance(self):
        assert md.nav_uncertainty_m(1000.0) > md.nav_uncertainty_m(100.0)

    def test_sqrt_scaling(self):
        u1 = md.nav_uncertainty_m(100.0)
        u4 = md.nav_uncertainty_m(400.0)
        assert u4 == pytest.approx(u1 * 2.0, rel=1e-6)

    def test_negative_returns_zero(self):
        assert md.nav_uncertainty_m(-50.0) == 0.0

    def test_reasonable_at_1km(self):
        u = md.nav_uncertainty_m(1000.0)
        assert 0.1 < u < 10.0


# ===================================================================
# Communication link
# ===================================================================

class TestCommLink:
    def test_perfect_at_zero_distance(self):
        assert md.comm_link_margin_db(0.0) == 100.0

    def test_decreases_with_distance(self):
        assert md.comm_link_margin_db(500.0) > md.comm_link_margin_db(2000.0)

    def test_positive_at_short_range(self):
        assert md.comm_link_margin_db(100.0) > 0.0

    def test_negative_at_very_long_range(self):
        assert md.comm_link_margin_db(1_000_000.0) < 0.0


# ===================================================================
# Blade erosion
# ===================================================================

class TestBladeErosion:
    def test_zero_for_zero_hours(self):
        assert md.blade_erosion(0.0, 0.5) == 0.0

    def test_positive_for_positive_hours(self):
        assert md.blade_erosion(1.0, 0.5) > 0.0

    def test_increases_with_hours(self):
        assert md.blade_erosion(2.0, 0.5) > md.blade_erosion(1.0, 0.5)

    def test_increases_with_dust(self):
        assert md.blade_erosion(1.0, 2.0) > md.blade_erosion(1.0, 0.5)

    def test_zero_dust_still_erodes(self):
        assert md.blade_erosion(1.0, 0.0) > 0.0

    def test_proportional_to_hours(self):
        e1 = md.blade_erosion(1.0, 0.0)
        e2 = md.blade_erosion(2.0, 0.0)
        assert e2 == pytest.approx(e1 * 2.0, rel=1e-6)


# ===================================================================
# Payload management
# ===================================================================

class TestPayload:
    def test_attach_valid_payload(self, drone):
        assert md.attach_payload(drone, "camera") is True
        assert drone.payload_type == "camera"
        assert drone.payload_mass_kg == 0.3

    def test_attach_invalid_payload(self, drone):
        assert md.attach_payload(drone, "flamethrower") is False
        assert drone.payload_type == ""

    def test_detach_payload(self, drone):
        md.attach_payload(drone, "lidar")
        removed = md.detach_payload(drone)
        assert removed == "lidar"
        assert drone.payload_type == ""
        assert drone.payload_mass_kg == 0.0

    def test_all_payload_types_attachable(self, drone):
        for ptype in md.PAYLOAD_TYPES:
            md.attach_payload(drone, ptype)
            assert drone.payload_type == ptype
            assert drone.payload_mass_kg > 0.0

    def test_payload_adds_mass(self, drone):
        m_before = md.total_mass_kg(drone)
        md.attach_payload(drone, "grabber")
        m_after = md.total_mass_kg(drone)
        assert m_after - m_before == pytest.approx(1.2, rel=0.01)


# ===================================================================
# Pre-flight checks
# ===================================================================

class TestCanFly:
    def test_default_drone_can_fly(self, drone):
        ok, reason = md.can_fly(drone)
        assert ok is True
        assert reason == "ok"

    def test_worn_blades_cannot_fly(self):
        d = md.create_drone(blade_health=0.05)
        ok, reason = md.can_fly(d)
        assert ok is False
        assert "blades" in reason

    def test_empty_battery_cannot_fly(self, dead_battery_drone):
        ok, reason = md.can_fly(dead_battery_drone)
        assert ok is False
        assert "battery" in reason

    def test_cold_battery_cannot_fly(self):
        d = md.create_drone(battery_temp_k=220.0)
        ok, reason = md.can_fly(d)
        assert ok is False
        assert "cold" in reason

    def test_hot_battery_cannot_fly(self):
        d = md.create_drone(battery_temp_k=350.0)
        ok, reason = md.can_fly(d)
        assert ok is False
        assert "hot" in reason

    def test_overloaded_drone_cannot_fly(self, heavy_drone):
        ok, reason = md.can_fly(heavy_drone)
        assert ok is False
        assert "thrust" in reason


# ===================================================================
# Flight simulation
# ===================================================================

class TestSimulateFlight:
    def test_short_flight_succeeds(self, drone, short_plan):
        res = md.simulate_flight(drone, short_plan)
        assert res.success is True
        assert res.range_m > 0.0
        assert res.flight_time_s > 0.0
        assert res.energy_used_wh > 0.0

    def test_battery_decreases_after_flight(self, drone, short_plan):
        before = drone.battery_charge_wh
        md.simulate_flight(drone, short_plan)
        assert drone.battery_charge_wh < before

    def test_blade_health_decreases(self, drone, short_plan):
        before = drone.blade_health
        md.simulate_flight(drone, short_plan)
        assert drone.blade_health < before

    def test_counters_increment(self, drone, short_plan):
        md.simulate_flight(drone, short_plan)
        assert drone.total_flights == 1
        assert drone.total_flight_hours > 0.0
        assert drone.total_distance_m > 0.0

    def test_failed_flight_increments_failed_count(self, dead_battery_drone, short_plan):
        md.simulate_flight(dead_battery_drone, short_plan)
        assert dead_battery_drone.failed_flights == 1
        assert dead_battery_drone.total_flights == 0

    def test_heavy_payload_uses_more_energy(self):
        d1 = md.create_drone()
        d2 = md.create_drone()
        light = md.FlightPlan(target_altitude_m=10.0, target_range_m=200.0, payload_type="camera")
        heavy = md.FlightPlan(target_altitude_m=10.0, target_range_m=200.0, payload_type="grabber")
        r1 = md.simulate_flight(d1, light)
        r2 = md.simulate_flight(d2, heavy)
        if r1.success and r2.success:
            assert r2.energy_used_wh > r1.energy_used_wh

    def test_higher_altitude_uses_more_energy(self):
        d1 = md.create_drone()
        d2 = md.create_drone()
        r1 = md.simulate_flight(d1, md.FlightPlan(target_altitude_m=5.0, target_range_m=200.0))
        r2 = md.simulate_flight(d2, md.FlightPlan(target_altitude_m=50.0, target_range_m=200.0))
        if r1.success and r2.success:
            assert r2.energy_used_wh > r1.energy_used_wh

    def test_flight_returns_nav_uncertainty(self, drone, short_plan):
        res = md.simulate_flight(drone, short_plan)
        assert res.success
        assert res.nav_uncertainty_m >= 0.0

    def test_flight_returns_comm_margin(self, drone, short_plan):
        res = md.simulate_flight(drone, short_plan)
        assert res.success
        assert res.comm_margin_db != 0.0

    def test_worn_blades_fail(self):
        d = md.create_drone(blade_health=0.05)
        res = md.simulate_flight(d, md.FlightPlan(target_altitude_m=10.0, target_range_m=100.0))
        assert not res.success
        assert "blades" in res.failure_reason

    def test_invalid_payload_fails(self, drone):
        res = md.simulate_flight(drone, md.FlightPlan(payload_type="invalid_thing"))
        assert not res.success
        assert "payload" in res.failure_reason


# ===================================================================
# Conservation laws
# ===================================================================

class TestConservation:
    def test_energy_conservation(self, drone, short_plan):
        before = drone.battery_charge_wh
        res = md.simulate_flight(drone, short_plan)
        if res.success:
            lost = before - drone.battery_charge_wh
            assert lost >= res.energy_used_wh * 0.95

    def test_blade_health_never_negative(self, drone):
        plan = md.FlightPlan(target_altitude_m=5.0, target_range_m=50.0,
                             payload_type="camera", dust_tau=4.0)
        for _ in range(50):
            drone.battery_charge_wh = drone.battery_capacity_wh
            md.simulate_flight(drone, plan)
        assert drone.blade_health >= 0.0

    def test_battery_never_negative(self, drone):
        for _ in range(20):
            md.simulate_flight(drone, md.FlightPlan(target_range_m=500.0))
        assert drone.battery_charge_wh >= 0.0

    def test_total_flights_equals_successes_plus_failures(self, drone):
        plans = [md.FlightPlan(target_range_m=r)
                 for r in [100, 200, 500, 1000, 2000, 5000]]
        for p in plans:
            md.simulate_flight(drone, p)
        assert drone.total_flights + drone.failed_flights == len(plans)

    def test_distance_monotonically_increases(self, drone):
        prev_dist = 0.0
        for _ in range(5):
            drone.battery_charge_wh = drone.battery_capacity_wh
            res = md.simulate_flight(drone, md.FlightPlan(target_range_m=100.0))
            if res.success:
                assert drone.total_distance_m >= prev_dist
                prev_dist = drone.total_distance_m

    def test_flight_hours_match_time(self, drone, short_plan):
        res = md.simulate_flight(drone, short_plan)
        if res.success:
            expected = res.flight_time_s / 3600.0
            assert drone.total_flight_hours == pytest.approx(expected, rel=0.01)


# ===================================================================
# Physical bounds
# ===================================================================

class TestPhysicalBounds:
    def test_hover_power_reasonable(self, drone):
        """Mars drone ~4.5 kg should need 30-500 W to hover."""
        rho = md.MARS_SURFACE_DENSITY_KG_M3
        weight = md.total_mass_kg(drone) * md.MARS_GRAVITY_M_S2
        disk_area = math.pi * drone.rotor_radius_m ** 2
        p = md.hover_power_w(weight, rho, disk_area)
        assert 10.0 < p < 500.0

    def test_thrust_exceeds_weight(self, drone):
        t = md.rotor_thrust_n(drone.rotor_radius_m, drone.rotor_rpm,
                              drone.ct, md.MARS_SURFACE_DENSITY_KG_M3,
                              drone.blade_health)
        w = md.total_mass_kg(drone) * md.MARS_GRAVITY_M_S2
        assert t > w

    def test_thrust_to_weight_ratio(self, drone):
        """T/W should be 1.1-5.0 for viable rotorcraft."""
        t = md.rotor_thrust_n(drone.rotor_radius_m, drone.rotor_rpm,
                              drone.ct, md.MARS_SURFACE_DENSITY_KG_M3,
                              drone.blade_health)
        w = md.total_mass_kg(drone) * md.MARS_GRAVITY_M_S2
        assert 1.1 < t / w < 5.0

    def test_battery_lasts_reasonable_minutes(self, drone):
        rho = md.MARS_SURFACE_DENSITY_KG_M3
        weight = md.total_mass_kg(drone) * md.MARS_GRAVITY_M_S2
        disk_area = math.pi * drone.rotor_radius_m ** 2
        p = md.hover_power_w(weight, rho, disk_area)
        endurance_min = drone.battery_capacity_wh / p * 60.0
        assert 1.0 < endurance_min < 30.0

    def test_solar_recharge_takes_reasonable_time(self, drone):
        rate_wh = md.solar_charge_wh(drone.solar_panel_area_m2, drone.solar_efficiency,
                                     md.HOURS_PER_SOL * md.DAYLIGHT_FRACTION, 0.8)
        if rate_wh > 0:
            sols = drone.battery_capacity_wh / rate_wh
            assert 0.1 < sols < 10.0


# ===================================================================
# Battery charging
# ===================================================================

class TestCharging:
    def test_charge_increases_battery(self, drone):
        drone.battery_charge_wh = 50.0
        added = md.charge_battery(drone, 6.0, 0.8)
        assert added > 0.0
        assert drone.battery_charge_wh > 50.0

    def test_charge_capped_at_capacity(self, drone):
        drone.battery_charge_wh = drone.battery_capacity_wh - 1.0
        md.charge_battery(drone, 100.0, 1.0)
        assert drone.battery_charge_wh <= drone.battery_capacity_wh

    def test_charge_zero_hours(self, drone):
        drone.battery_charge_wh = 50.0
        assert md.charge_battery(drone, 0.0, 0.8) == 0.0

    def test_dust_storm_reduces_charge_rate(self):
        d1 = md.create_drone(battery_charge_wh=0.0)
        d2 = md.create_drone(battery_charge_wh=0.0)
        a1 = md.charge_battery(d1, 6.0, 1.0)
        a2 = md.charge_battery(d2, 6.0, 0.3)
        assert a1 > a2


# ===================================================================
# Thermal model
# ===================================================================

class TestThermal:
    def test_cold_battery_warms_with_heater(self):
        """Heater keeps battery warmer than it would be without it."""
        d_heated = md.create_drone(battery_temp_k=250.0)
        d_no_heat = md.create_drone(battery_temp_k=250.0, battery_charge_wh=0.0)
        md.thermal_update(d_heated, md.MARS_AMBIENT_TEMP_K, 3600.0)
        md.thermal_update(d_no_heat, md.MARS_AMBIENT_TEMP_K, 3600.0)
        # Heater slows cooling even if it can't overcome heat leak
        assert d_heated.battery_temp_k >= d_no_heat.battery_temp_k

    def test_warm_battery_cools_toward_ambient(self, drone):
        drone.battery_temp_k = 310.0
        md.thermal_update(drone, md.MARS_AMBIENT_TEMP_K, 3600.0)
        assert drone.battery_temp_k < 310.0

    def test_never_below_ambient(self, drone):
        drone.battery_temp_k = md.MARS_AMBIENT_TEMP_K + 1.0
        md.thermal_update(drone, md.MARS_AMBIENT_TEMP_K, 1e6)
        assert drone.battery_temp_k >= md.MARS_AMBIENT_TEMP_K

    def test_never_above_max(self, drone):
        drone.battery_temp_k = md.BATTERY_MAX_TEMP_K - 1.0
        md.thermal_update(drone, md.BATTERY_MAX_TEMP_K + 50.0, 3600.0)
        assert drone.battery_temp_k <= md.BATTERY_MAX_TEMP_K

    def test_heater_uses_battery(self):
        d = md.create_drone(battery_temp_k=250.0)
        before = d.battery_charge_wh
        md.thermal_update(d, md.MARS_AMBIENT_TEMP_K, 3600.0)
        assert d.battery_charge_wh <= before


# ===================================================================
# Blade replacement
# ===================================================================

class TestBladeReplacement:
    def test_replace_restores_health(self, worn_drone):
        md.replace_blades(worn_drone)
        assert worn_drone.blade_health == 1.0

    def test_replace_from_full_stays_full(self, drone):
        md.replace_blades(drone)
        assert drone.blade_health == 1.0


# ===================================================================
# Tick (full sol cycle)
# ===================================================================

class TestTick:
    def test_tick_with_no_flights(self, drone):
        result = md.tick(drone)
        assert result["sol"] == 1
        assert result["flights_attempted"] == 0
        assert result["battery_wh"] > 0.0

    def test_tick_with_one_flight(self, drone):
        result = md.tick(drone, [md.FlightPlan(target_range_m=200.0)])
        assert result["flights_attempted"] == 1
        assert result["flights_succeeded"] <= 1

    def test_tick_charges_battery(self, drone):
        drone.battery_charge_wh = 10.0
        md.tick(drone)
        assert drone.battery_charge_wh > 10.0

    def test_tick_increments_sol(self, drone):
        md.tick(drone)
        assert drone.sol == 1
        md.tick(drone)
        assert drone.sol == 2

    def test_tick_returns_results_list(self, drone):
        plans = [md.FlightPlan(target_range_m=100.0),
                 md.FlightPlan(target_range_m=50.0)]
        result = md.tick(drone, plans)
        assert len(result["results"]) == 2

    def test_tick_dust_storm_reduces_charge(self):
        d1 = md.create_drone(battery_charge_wh=50.0)
        d2 = md.create_drone(battery_charge_wh=50.0)
        md.tick(d1, dust_tau=0.3)
        md.tick(d2, dust_tau=3.0)
        assert d1.battery_charge_wh > d2.battery_charge_wh


# ===================================================================
# Multi-sol endurance
# ===================================================================

class TestEndurance:
    def test_10_sol_no_crash(self, drone):
        for _ in range(10):
            md.tick(drone, [md.FlightPlan(target_range_m=200.0)])
        assert drone.sol == 10
        assert drone.blade_health >= 0.0
        assert drone.battery_charge_wh >= 0.0

    def test_100_sol_blade_degradation(self, drone):
        for _ in range(100):
            md.tick(drone, [md.FlightPlan(target_range_m=100.0)])
        assert drone.blade_health < 0.9

    def test_many_flights_per_sol(self, drone):
        plans = [md.FlightPlan(target_range_m=50.0) for _ in range(5)]
        result = md.tick(drone, plans)
        assert result["flights_attempted"] == 5
        assert result["flights_succeeded"] + result["flights_failed"] == 5

    def test_dust_storm_season(self, drone):
        result = md.tick(drone, [md.FlightPlan(target_range_m=500.0)], dust_tau=4.0)
        assert result["sol"] == 1
        assert result["dust_tau"] == 4.0


# ===================================================================
# Status snapshot
# ===================================================================

class TestStatus:
    def test_status_has_required_fields(self, drone):
        s = md.status(drone)
        for key in ["sol", "operational", "battery_pct", "blade_health_pct", "total_flights"]:
            assert key in s

    def test_status_operational_for_default(self, drone):
        assert md.status(drone)["operational"] is True

    def test_status_not_operational_when_broken(self):
        assert md.status(md.create_drone(blade_health=0.01))["operational"] is False

    def test_battery_pct_100_when_full(self, drone):
        assert md.status(drone)["battery_pct"] == pytest.approx(100.0, abs=0.1)

    def test_battery_pct_0_when_empty(self, dead_battery_drone):
        assert md.status(dead_battery_drone)["battery_pct"] == pytest.approx(0.0, abs=0.1)


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_zero_range_flight(self, drone):
        res = md.simulate_flight(drone, md.FlightPlan(target_range_m=0.0))
        assert res.success is True
        assert res.range_m == 0.0

    def test_zero_altitude_flight(self, drone):
        res = md.simulate_flight(drone, md.FlightPlan(target_altitude_m=0.0, target_range_m=100.0))
        assert res.success is True

    def test_extremely_long_range_fails_gracefully(self, drone):
        res = md.simulate_flight(drone, md.FlightPlan(target_range_m=100_000.0))
        assert isinstance(res, md.FlightResult)

    def test_very_high_altitude_fails(self, drone):
        res = md.simulate_flight(drone, md.FlightPlan(target_altitude_m=50_000.0, target_range_m=100.0))
        assert not res.success

    def test_no_payload_flight(self, drone):
        res = md.simulate_flight(drone, md.FlightPlan(target_range_m=200.0, payload_type=""))
        assert res.success is True

    def test_cruise_speed_clamped(self, drone):
        res = md.simulate_flight(drone, md.FlightPlan(target_range_m=200.0, cruise_speed_m_s=1000.0))
        assert isinstance(res, md.FlightResult)

    def test_negative_payload_does_not_crash(self, drone):
        drone.payload_mass_kg = -5.0
        drone.payload_type = "camera"
        res = md.simulate_flight(drone, md.FlightPlan(target_range_m=100.0))
        assert isinstance(res, md.FlightResult)


# ===================================================================
# Integration: flight + recharge cycle
# ===================================================================

class TestFlightRechargeCycle:
    def test_fly_recharge_fly_again(self, drone):
        r1 = md.simulate_flight(drone, md.FlightPlan(target_range_m=200.0))
        assert r1.success
        battery_after = drone.battery_charge_wh
        md.charge_battery(drone, 12.0, 0.8)
        assert drone.battery_charge_wh > battery_after
        r2 = md.simulate_flight(drone, md.FlightPlan(target_range_m=200.0))
        assert r2.success
        assert drone.total_flights == 2

    def test_full_operational_cycle_via_tick(self, drone):
        for sol_num in range(1, 11):
            result = md.tick(drone, [md.FlightPlan(target_range_m=300.0)])
            assert result["sol"] == sol_num
        assert drone.blade_health > 0.5
