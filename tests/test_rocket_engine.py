"""Tests for rocket_engine.py -- Mars Ascent Vehicle Methalox Engine.

102 tests covering nozzle physics, thrust, Isp, Tsiolkovsky, Mars
environment, engine tick, conservation laws, thermal failure, full
burn simulation, orbit check, physical bounds, smoke tests, edge cases.
"""
from __future__ import annotations

import math
import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rocket_engine as re


@pytest.fixture
def default_engine():
    return re.create_engine()

@pytest.fixture
def small_engine():
    return re.create_engine(thrust_n=10_000.0, lox_kg=350.0,
                            ch4_kg=100.0, dry_mass_kg=200.0)

@pytest.fixture
def empty_engine():
    return re.create_engine(lox_kg=0.0, ch4_kg=0.0)


class TestExhaustVelocity:
    def test_positive_for_valid_inputs(self):
        assert re.exhaust_velocity_m_s(3500.0, 2e6, 100.0) > 0.0

    def test_increases_with_chamber_temp(self):
        assert re.exhaust_velocity_m_s(4000.0, 2e6, 100.0) > re.exhaust_velocity_m_s(2000.0, 2e6, 100.0)

    def test_increases_with_pressure_ratio(self):
        assert re.exhaust_velocity_m_s(3500.0, 2e6, 10.0) > re.exhaust_velocity_m_s(3500.0, 2e6, 1e5)

    def test_zero_for_zero_chamber_pressure(self):
        assert re.exhaust_velocity_m_s(3500.0, 0.0, 100.0) == 0.0

    def test_zero_for_equal_pressures(self):
        assert re.exhaust_velocity_m_s(3500.0, 2e6, 2e6) == 0.0

    def test_zero_for_negative_pressure(self):
        assert re.exhaust_velocity_m_s(3500.0, -1.0, 100.0) == 0.0

    def test_methalox_range(self):
        v = re.exhaust_velocity_m_s(3500.0, 2e6, 100.0)
        assert 2500.0 < v < 4500.0

    def test_exit_pressure_above_chamber_returns_zero(self):
        assert re.exhaust_velocity_m_s(3500.0, 1e5, 2e6) == 0.0


class TestExitPressure:
    def test_decreases_with_area_ratio(self):
        assert re.exit_pressure_pa(2e6, 80.0) < re.exit_pressure_pa(2e6, 20.0)

    def test_identity_at_ratio_one(self):
        assert re.exit_pressure_pa(2e6, 1.0) == 2e6

    def test_below_ratio_one_returns_chamber(self):
        assert re.exit_pressure_pa(2e6, 0.5) == 2e6

    def test_always_positive(self):
        for ar in [5, 20, 60, 100, 200]:
            assert re.exit_pressure_pa(2e6, float(ar)) >= 0.0

    def test_much_less_than_chamber(self):
        assert re.exit_pressure_pa(2e6, 60.0) < 2e6 * 0.01


class TestThrust:
    def test_positive_for_valid_inputs(self):
        assert re.thrust_n(3500.0, 100.0, 610.0, 0.3, 12.0) > 0.0

    def test_zero_for_zero_mass_flow(self):
        assert re.thrust_n(3500.0, 100.0, 610.0, 0.3, 0.0) == 0.0

    def test_scales_with_mass_flow(self):
        f1 = re.thrust_n(3500.0, 100.0, 610.0, 0.3, 10.0)
        f2 = re.thrust_n(3500.0, 100.0, 610.0, 0.3, 20.0)
        assert abs(f2 / f1 - 2.0) < 0.05

    def test_pressure_thrust_contribution(self):
        assert re.thrust_n(3500.0, 50000.0, 610.0, 0.3, 10.0) > re.thrust_n(3500.0, 100.0, 610.0, 0.3, 10.0)

    def test_never_negative(self):
        assert re.thrust_n(100.0, 0.0, 100000.0, 0.3, 0.001) >= 0.0


class TestSpecificImpulse:
    def test_positive_for_valid_inputs(self):
        assert re.specific_impulse_s(3500.0, 100.0, 610.0, 0.3, 12.0) > 0.0

    def test_zero_for_zero_mass_flow(self):
        assert re.specific_impulse_s(3500.0, 100.0, 610.0, 0.3, 0.0) == 0.0

    def test_methalox_range(self):
        isp = re.specific_impulse_s(3500.0, 100.0, 0.0, 0.3, 12.0)
        assert 250.0 < isp < 450.0

    def test_vacuum_isp_higher_than_surface(self):
        assert re.specific_impulse_s(3500.0, 100.0, 0.0, 0.3, 12.0) >= re.specific_impulse_s(3500.0, 100.0, 610.0, 0.3, 12.0)


class TestMassFlowRate:
    def test_positive_for_valid_inputs(self):
        assert re.mass_flow_rate_kg_s(44000.0, 350.0) > 0.0

    def test_zero_for_zero_thrust(self):
        assert re.mass_flow_rate_kg_s(0.0, 350.0) == 0.0

    def test_zero_for_zero_isp(self):
        assert re.mass_flow_rate_kg_s(44000.0, 0.0) == 0.0

    def test_inversely_proportional_to_isp(self):
        assert re.mass_flow_rate_kg_s(44000.0, 300.0) > re.mass_flow_rate_kg_s(44000.0, 400.0)


class TestTsiolkovsky:
    def test_positive_delta_v(self):
        assert re.tsiolkovsky_delta_v(350.0, 5000.0, 1000.0) > 0.0

    def test_zero_for_equal_masses(self):
        assert re.tsiolkovsky_delta_v(350.0, 1000.0, 1000.0) == 0.0

    def test_zero_for_zero_isp(self):
        assert re.tsiolkovsky_delta_v(0.0, 5000.0, 1000.0) == 0.0

    def test_zero_for_final_greater(self):
        assert re.tsiolkovsky_delta_v(350.0, 1000.0, 5000.0) == 0.0

    def test_increases_with_mass_ratio(self):
        assert re.tsiolkovsky_delta_v(350.0, 5000.0, 1000.0) > re.tsiolkovsky_delta_v(350.0, 2000.0, 1000.0)

    def test_increases_with_isp(self):
        assert re.tsiolkovsky_delta_v(400.0, 5000.0, 1000.0) > re.tsiolkovsky_delta_v(300.0, 5000.0, 1000.0)

    def test_known_value(self):
        dv = re.tsiolkovsky_delta_v(350.0, 5000.0, 1000.0)
        expected = 350.0 * re.G0_M_S2 * math.log(5.0)
        assert abs(dv - expected) < 0.1

    def test_zero_masses(self):
        assert re.tsiolkovsky_delta_v(350.0, 0.0, 1000.0) == 0.0
        assert re.tsiolkovsky_delta_v(350.0, 5000.0, 0.0) == 0.0

    def test_negative_isp(self):
        assert re.tsiolkovsky_delta_v(-100.0, 5000.0, 1000.0) == 0.0


class TestGravity:
    def test_surface_gravity(self):
        assert abs(re.gravity_at_altitude_m_s2(0.0) - re.MARS_GRAVITY_M_S2) < 0.001

    def test_decreases_with_altitude(self):
        assert re.gravity_at_altitude_m_s2(250_000.0) < re.gravity_at_altitude_m_s2(0.0)

    def test_always_positive(self):
        for alt in [0, 1000, 100_000, 1_000_000]:
            assert re.gravity_at_altitude_m_s2(float(alt)) > 0.0

    def test_negative_altitude_uses_surface(self):
        assert re.gravity_at_altitude_m_s2(-100.0) == re.gravity_at_altitude_m_s2(0.0)

    def test_inverse_square(self):
        g = re.gravity_at_altitude_m_s2(re.MARS_RADIUS_M)
        assert abs(g - re.MARS_GRAVITY_M_S2 / 4.0) < 0.01


class TestAtmosphere:
    def test_surface_pressure(self):
        assert abs(re.atmospheric_pressure_pa(0.0) - re.MARS_SURFACE_PRESSURE_PA) < 0.1

    def test_decreases_with_altitude(self):
        assert re.atmospheric_pressure_pa(50_000.0) < re.atmospheric_pressure_pa(0.0)

    def test_exponential_decay(self):
        p = re.atmospheric_pressure_pa(re.MARS_SCALE_HEIGHT_M)
        assert abs(p - re.MARS_SURFACE_PRESSURE_PA / math.e) < 0.1

    def test_always_positive(self):
        for alt in [0, 10000, 100_000, 500_000]:
            assert re.atmospheric_pressure_pa(float(alt)) > 0.0

    def test_negative_altitude_returns_surface(self):
        assert re.atmospheric_pressure_pa(-100.0) == re.MARS_SURFACE_PRESSURE_PA

    def test_density_correlates(self):
        assert re.atmospheric_density_kg_m3(50_000.0) < re.atmospheric_density_kg_m3(0.0)

    def test_surface_density(self):
        assert abs(re.atmospheric_density_kg_m3(0.0) - 0.020) < 0.001


class TestDrag:
    def test_zero_at_zero_velocity(self):
        assert re.drag_force_n(0.0, 0.0, 3.0) == 0.0

    def test_increases_with_velocity(self):
        assert re.drag_force_n(200.0, 0.0, 3.0) > re.drag_force_n(100.0, 0.0, 3.0)

    def test_decreases_with_altitude(self):
        assert re.drag_force_n(500.0, 50_000.0, 3.0) < re.drag_force_n(500.0, 0.0, 3.0)

    def test_quadratic_scaling(self):
        d1 = re.drag_force_n(100.0, 0.0, 3.0)
        d2 = re.drag_force_n(200.0, 0.0, 3.0)
        assert abs(d2 / d1 - 4.0) < 0.1

    def test_never_negative(self):
        assert re.drag_force_n(-10.0, 0.0, 3.0) == 0.0


class TestCreateEngine:
    def test_returns_config_and_state(self, default_engine):
        config, state = default_engine
        assert isinstance(config, re.EngineConfig)
        assert isinstance(state, re.VehicleState)

    def test_default_propellant(self, default_engine):
        _, s = default_engine
        assert s.lox_kg == re.DEFAULT_LOX_KG and s.ch4_kg == re.DEFAULT_CH4_KG

    def test_custom_thrust(self):
        c, _ = re.create_engine(thrust_n=50_000.0)
        assert c.thrust_target_n == 50_000.0

    def test_initial_state(self, default_engine):
        _, s = default_engine
        assert s.altitude_m == 0.0 and s.velocity_m_s == 0.0
        assert not s.engine_running and not s.engine_failed

    def test_total_mass(self, default_engine):
        _, s = default_engine
        assert abs(s.total_mass_kg - (s.lox_kg + s.ch4_kg + s.dry_mass_kg)) < 0.01

    def test_mass_ratio(self, default_engine):
        _, s = default_engine
        assert abs(s.mass_ratio - s.total_mass_kg / s.dry_mass_kg) < 0.01

    def test_propellant_property(self, default_engine):
        _, s = default_engine
        assert abs(s.propellant_kg - (s.lox_kg + s.ch4_kg)) < 0.01


class TestTick:
    def test_starts_engine(self, default_engine):
        c, s = default_engine
        re.tick(c, s)
        assert s.engine_running

    def test_produces_thrust(self, default_engine):
        c, s = default_engine
        assert re.tick(c, s).thrust_n > 0.0

    def test_consumes_propellant(self, default_engine):
        c, s = default_engine
        l0, f0 = s.lox_kg, s.ch4_kg
        re.tick(c, s)
        assert s.lox_kg < l0 and s.ch4_kg < f0

    def test_increases_altitude(self, default_engine):
        c, s = default_engine
        for _ in range(5):
            re.tick(c, s)
        assert s.altitude_m > 0.0

    def test_increases_velocity(self, default_engine):
        c, s = default_engine
        for _ in range(5):
            re.tick(c, s)
        assert s.velocity_m_s > 0.0

    def test_accumulates_delta_v(self, default_engine):
        c, s = default_engine
        for _ in range(10):
            re.tick(c, s)
        assert s.delta_v_m_s > 0.0

    def test_wall_temp_increases(self, default_engine):
        c, s = default_engine
        t0 = s.wall_temp_k
        re.tick(c, s)
        assert s.wall_temp_k > t0

    def test_mixture_ratio_maintained(self, default_engine):
        c, s = default_engine
        r = re.tick(c, s)
        if r.ch4_flow_kg_s > 0:
            assert abs(r.lox_flow_kg_s / r.ch4_flow_kg_s - re.MIXTURE_RATIO_OF) < 0.01

    def test_burn_time_increments(self, default_engine):
        c, s = default_engine
        re.tick(c, s); assert s.burn_time_s == 1.0
        re.tick(c, s); assert s.burn_time_s == 2.0

    def test_total_impulse_positive(self, default_engine):
        c, s = default_engine
        re.tick(c, s)
        assert s.total_impulse_ns > 0.0


class TestTickEmptyTank:
    def test_no_burn(self, empty_engine):
        c, s = empty_engine
        assert re.tick(c, s).thrust_n == 0.0 and not s.engine_running

    def test_no_velocity_change(self, empty_engine):
        c, s = empty_engine
        re.tick(c, s)
        assert s.velocity_m_s == 0.0


class TestTickThermalFailure:
    def test_engine_fails_on_overheat(self):
        c, s = re.create_engine()
        c.max_wall_temp_k = 310.0
        for _ in range(200):
            r = re.tick(c, s)
            if s.engine_failed:
                break
        assert s.engine_failed and r.engine_failed

    def test_no_thrust_after_failure(self):
        c, s = re.create_engine()
        c.max_wall_temp_k = 310.0
        for _ in range(200):
            re.tick(c, s)
            if s.engine_failed:
                break
        r = re.tick(c, s)
        assert r.engine_failed and r.thrust_n == 0.0


class TestConservation:
    def test_mass_conservation(self, default_engine):
        c, s = default_engine
        init = s.lox_kg + s.ch4_kg
        for _ in range(50):
            re.tick(c, s)
        consumed = s.total_lox_consumed_kg + s.total_ch4_consumed_kg
        remaining = s.lox_kg + s.ch4_kg
        assert abs((consumed + remaining) - init) < 0.01

    def test_propellant_never_negative(self, default_engine):
        c, s = default_engine
        for _ in range(600):
            re.tick(c, s)
            assert s.lox_kg >= 0.0 and s.ch4_kg >= 0.0
            if s.propellant_kg <= 0:
                break

    def test_mixture_ratio_conservation(self, default_engine):
        c, s = default_engine
        for _ in range(50):
            re.tick(c, s)
        if s.total_ch4_consumed_kg > 0:
            assert abs(s.total_lox_consumed_kg / s.total_ch4_consumed_kg - re.MIXTURE_RATIO_OF) < 0.1

    def test_altitude_non_negative(self, default_engine):
        c, s = default_engine
        for _ in range(100):
            re.tick(c, s)
            assert s.altitude_m >= 0.0

    def test_velocity_non_negative(self, default_engine):
        c, s = default_engine
        for _ in range(100):
            re.tick(c, s)
            assert s.velocity_m_s >= 0.0

    def test_thrust_non_negative(self, default_engine):
        c, s = default_engine
        for _ in range(100):
            assert re.tick(c, s).thrust_n >= 0.0

    def test_isp_bounded(self, default_engine):
        c, s = default_engine
        for _ in range(50):
            r = re.tick(c, s)
            assert 0.0 <= r.isp_s <= 500.0

    def test_delta_v_monotonic(self, default_engine):
        c, s = default_engine
        prev = 0.0
        for _ in range(50):
            re.tick(c, s)
            assert s.delta_v_m_s >= prev
            prev = s.delta_v_m_s

    def test_burn_time_monotonic(self, default_engine):
        c, s = default_engine
        prev = 0.0
        for _ in range(50):
            re.tick(c, s)
            assert s.burn_time_s >= prev
            prev = s.burn_time_s

    def test_gravity_loss_non_negative(self, default_engine):
        c, s = default_engine
        for _ in range(50):
            assert re.tick(c, s).gravity_loss_m_s >= 0.0

    def test_peak_tracking(self, default_engine):
        c, s = default_engine
        for _ in range(100):
            re.tick(c, s)
        assert s.peak_altitude_m >= s.altitude_m and s.peak_acceleration_g > 0.0


class TestRunBurn:
    def test_returns_results(self, default_engine):
        c, s = default_engine
        assert len(re.run_burn(c, s, max_seconds=10)) > 0

    def test_stops_on_exhaustion(self, small_engine):
        c, s = small_engine
        re.run_burn(c, s, max_seconds=1000)
        assert s.propellant_kg < 1.0

    def test_altitude_increases(self, default_engine):
        c, s = default_engine
        results = re.run_burn(c, s, max_seconds=50)
        assert results[-1].altitude_m > results[0].altitude_m

    def test_max_seconds_limit(self, default_engine):
        c, s = default_engine
        assert len(re.run_burn(c, s, max_seconds=5)) <= 5

    def test_default_creation(self):
        assert len(re.run_burn(max_seconds=10)) > 0


class TestCanReachOrbit:
    def test_default_reaches_orbit(self):
        r = re.can_reach_orbit()
        assert r["ideal_delta_v_m_s"] > 0 and r["isp_vacuum_s"] > 300.0

    def test_empty_tank_fails(self):
        c, s = re.create_engine(lox_kg=1.0, ch4_kg=0.3)
        r = re.can_reach_orbit(c, s)
        assert not r["go"] and r["margin_m_s"] < 0

    def test_returns_all_keys(self):
        r = re.can_reach_orbit()
        for k in ["ideal_delta_v_m_s", "isp_vacuum_s", "go", "mass_ratio", "propellant_kg"]:
            assert k in r

    def test_mass_ratio_correct(self, default_engine):
        c, s = default_engine
        assert re.can_reach_orbit(c, s)["mass_ratio"] == round(s.mass_ratio, 2)

    def test_gravity_loss_positive(self):
        assert re.can_reach_orbit()["estimated_gravity_loss_m_s"] > 0


class TestRunSimulation:
    def test_returns_summary(self):
        r = re.run_simulation()
        for k in ["preflight", "burn_ticks", "final_altitude_m"]:
            assert k in r

    def test_produces_delta_v(self):
        assert re.run_simulation()["total_delta_v_m_s"] > 0

    def test_consumes_propellant(self):
        r = re.run_simulation()
        assert r["lox_consumed_kg"] > 0 and r["ch4_consumed_kg"] > 0

    def test_no_engine_failure(self):
        assert not re.run_simulation()["engine_failed"]


class TestPhysicalBounds:
    @pytest.mark.parametrize("thrust", [5000, 10000, 44000, 100000])
    def test_isp_in_range(self, thrust):
        c, s = re.create_engine(thrust_n=float(thrust))
        r = re.tick(c, s)
        assert 200.0 < r.isp_s < 450.0

    @pytest.mark.parametrize("lox,ch4", [(350,100),(1000,286),(3150,900),(7000,2000)])
    def test_propellant_ratio(self, lox, ch4):
        c, s = re.create_engine(lox_kg=float(lox), ch4_kg=float(ch4))
        for _ in range(20):
            re.tick(c, s)
        if s.total_ch4_consumed_kg > 1.0:
            assert abs(s.total_lox_consumed_kg / s.total_ch4_consumed_kg - re.MIXTURE_RATIO_OF) < 0.2

    def test_exhaust_below_light_speed(self):
        assert re.exhaust_velocity_m_s(10000.0, 1e8, 1.0) < 3e8

    def test_gravity_bounded(self):
        for alt in range(0, 1_000_000, 50_000):
            g = re.gravity_at_altitude_m_s2(float(alt))
            assert 0 < g <= re.MARS_GRAVITY_M_S2

    def test_pressure_bounded(self):
        for alt in range(0, 200_000, 10_000):
            assert 0 < re.atmospheric_pressure_pa(float(alt)) <= re.MARS_SURFACE_PRESSURE_PA


class TestSmoke:
    def test_10_ticks_no_crash(self):
        c, s = re.create_engine()
        for i in range(10):
            assert re.tick(c, s).tick == i

    def test_full_burn_no_crash(self):
        assert len(re.run_burn(max_seconds=600)) > 0

    def test_simulation_no_crash(self):
        assert isinstance(re.run_simulation(), dict)

    def test_orbit_check_no_crash(self):
        assert isinstance(re.can_reach_orbit(), dict)

    def test_independent_simulations(self):
        r1 = re.run_simulation()
        r2 = re.run_simulation()
        assert r1["total_delta_v_m_s"] == r2["total_delta_v_m_s"]

    def test_serializable(self):
        assert len(json.dumps(re.run_simulation())) > 0


class TestEdgeCases:
    def test_very_small_propellant(self):
        c, s = re.create_engine(lox_kg=0.1, ch4_kg=0.03)
        assert len(re.run_burn(c, s, max_seconds=10)) > 0

    def test_very_large_thrust(self):
        c, s = re.create_engine(thrust_n=1_000_000.0)
        assert re.tick(c, s).thrust_n > 0

    def test_very_small_thrust(self):
        c, s = re.create_engine(thrust_n=100.0)
        assert re.tick(c, s).thrust_n >= 0

    def test_tiny_dry_mass(self):
        c, s = re.create_engine(dry_mass_kg=0.1)
        assert len(re.run_burn(c, s, max_seconds=10)) > 0

    def test_failed_engine_stays_failed(self):
        c, s = re.create_engine()
        s.engine_failed = True
        r = re.tick(c, s)
        assert r.engine_failed and r.thrust_n == 0.0

    def test_custom_dt(self):
        c, s = re.create_engine()
        re.tick(c, s, dt_s=0.5)
        assert s.burn_time_s == 0.5
