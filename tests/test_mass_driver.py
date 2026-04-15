"""Tests for Mars electromagnetic mass driver simulation.

148 tests across 28 test classes covering physics functions, dataclasses,
simulation logic, conservation laws, thermal limits, orbital targeting,
and edge cases.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import mass_driver as md


# ---------------------------------------------------------------------------
# 1. TestLorentzForce (9 tests)
# ---------------------------------------------------------------------------
class TestLorentzForce:
    def test_default_config_force(self):
        assert md.lorentz_force_n(50_000, 0.5, 2.0, 4) == pytest.approx(200_000)

    def test_single_coil(self):
        assert md.lorentz_force_n(50_000, 0.5, 2.0, 1) == pytest.approx(50_000)

    def test_zero_current(self):
        assert md.lorentz_force_n(0, 0.5, 2.0, 4) == 0.0

    def test_zero_length(self):
        assert md.lorentz_force_n(50_000, 0, 2.0, 4) == 0.0

    def test_zero_field(self):
        assert md.lorentz_force_n(50_000, 0.5, 0, 4) == 0.0

    def test_negative_current(self):
        assert md.lorentz_force_n(-1000, 0.5, 2.0, 4) == 0.0

    def test_linearity_current(self):
        base = md.lorentz_force_n(50_000, 0.5, 2.0, 4)
        doubled = md.lorentz_force_n(100_000, 0.5, 2.0, 4)
        assert doubled == pytest.approx(2 * base)

    def test_linearity_field(self):
        base = md.lorentz_force_n(50_000, 0.5, 2.0, 4)
        doubled = md.lorentz_force_n(50_000, 0.5, 4.0, 4)
        assert doubled == pytest.approx(2 * base)

    def test_proportional_to_coils(self):
        one = md.lorentz_force_n(50_000, 0.5, 2.0, 1)
        four = md.lorentz_force_n(50_000, 0.5, 2.0, 4)
        assert four == pytest.approx(4 * one)


# ---------------------------------------------------------------------------
# 2. TestKineticEnergy (6 tests)
# ---------------------------------------------------------------------------
class TestKineticEnergy:
    def test_known_value(self):
        assert md.kinetic_energy_j(100, 1000) == pytest.approx(50e6)

    def test_zero_mass(self):
        assert md.kinetic_energy_j(0, 1000) == 0.0

    def test_zero_velocity(self):
        assert md.kinetic_energy_j(100, 0) == 0.0

    def test_negative_mass(self):
        assert md.kinetic_energy_j(-10, 1000) == 0.0

    def test_quadratic_in_velocity(self):
        e1 = md.kinetic_energy_j(100, 1000)
        e2 = md.kinetic_energy_j(100, 2000)
        assert e2 == pytest.approx(4 * e1)

    def test_linear_in_mass(self):
        e1 = md.kinetic_energy_j(100, 1000)
        e2 = md.kinetic_energy_j(200, 1000)
        assert e2 == pytest.approx(2 * e1)


# ---------------------------------------------------------------------------
# 3. TestVelocityFromEnergy (4 tests)
# ---------------------------------------------------------------------------
class TestVelocityFromEnergy:
    def test_roundtrip(self):
        ke = md.kinetic_energy_j(100, 1000)
        v = md.velocity_from_energy_m_s(ke, 100)
        assert v == pytest.approx(1000)

    def test_zero_energy(self):
        assert md.velocity_from_energy_m_s(0, 100) == 0.0

    def test_zero_mass(self):
        assert md.velocity_from_energy_m_s(50e6, 0) == 0.0

    def test_known_value(self):
        v = md.velocity_from_energy_m_s(50e6, 100)
        assert v == pytest.approx(1000)


# ---------------------------------------------------------------------------
# 4. TestMarsAirDensity (6 tests)
# ---------------------------------------------------------------------------
class TestMarsAirDensity:
    def test_surface_density(self):
        assert md.mars_air_density_kg_m3(0) == pytest.approx(
            md.MARS_SURFACE_DENSITY_KG_M3
        )

    def test_scale_height(self):
        expected = md.MARS_SURFACE_DENSITY_KG_M3 * math.exp(-1)
        assert md.mars_air_density_kg_m3(md.MARS_SCALE_HEIGHT_M) == pytest.approx(
            expected
        )

    def test_very_high_altitude(self):
        assert md.mars_air_density_kg_m3(200_000) < 1e-6

    def test_negative_altitude(self):
        assert md.mars_air_density_kg_m3(-100) == pytest.approx(
            md.MARS_SURFACE_DENSITY_KG_M3
        )

    def test_monotonic_decrease(self):
        d0 = md.mars_air_density_kg_m3(0)
        d1 = md.mars_air_density_kg_m3(1_000)
        d2 = md.mars_air_density_kg_m3(10_000)
        assert d0 > d1 > d2

    def test_always_positive(self):
        for alt in [0, 100, 1_000, 50_000, 200_000]:
            assert md.mars_air_density_kg_m3(alt) > 0


# ---------------------------------------------------------------------------
# 5. TestDragForce (6 tests)
# ---------------------------------------------------------------------------
class TestDragForce:
    def test_surface_3km_per_s(self):
        f = md.drag_force_n(3000, 0.0)
        assert f == pytest.approx(6750, rel=1e-3)

    def test_zero_velocity(self):
        assert md.drag_force_n(0, 0) == 0.0

    def test_mars_much_less_than_earth(self):
        mars_drag = md.drag_force_n(3000, 0)
        earth_rho = 1.2
        earth_drag = (
            0.5 * earth_rho * 3000**2 * md.DRAG_COEFF * md.SLED_CROSS_SECTION_M2
        )
        assert mars_drag < earth_drag / 50

    def test_high_altitude_reduces_drag(self):
        surface = md.drag_force_n(1000, 0)
        high = md.drag_force_n(1000, 50_000)
        assert high < surface

    def test_quadratic_in_velocity(self):
        f1 = md.drag_force_n(1000, 0)
        f2 = md.drag_force_n(2000, 0)
        assert f2 == pytest.approx(4 * f1, rel=1e-6)

    def test_custom_coefficients(self):
        f = md.drag_force_n(1000, 0, drag_coeff=0.3, cross_section_m2=1.0)
        expected = 0.5 * md.MARS_SURFACE_DENSITY_KG_M3 * 1000**2 * 0.3 * 1.0
        assert f == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 6. TestGravityComponent (5 tests)
# ---------------------------------------------------------------------------
class TestGravityComponent:
    def test_30_degrees(self):
        expected = 150 * md.MARS_GRAVITY_M_S2 * math.sin(math.radians(30))
        assert md.gravity_component_n(150, 30) == pytest.approx(expected)

    def test_zero_elevation(self):
        assert md.gravity_component_n(150, 0) == pytest.approx(0, abs=1e-9)

    def test_90_degrees(self):
        expected = 150 * md.MARS_GRAVITY_M_S2
        assert md.gravity_component_n(150, 90) == pytest.approx(expected)

    def test_zero_mass(self):
        assert md.gravity_component_n(0, 30) == 0.0

    def test_negative_angle_clamped(self):
        assert md.gravity_component_n(150, -10) == pytest.approx(0, abs=1e-9)


# ---------------------------------------------------------------------------
# 7. TestStageAcceleration (4 tests)
# ---------------------------------------------------------------------------
class TestStageAcceleration:
    def test_known_value(self):
        a = md.stage_acceleration_m_s2(200_000, 0, 0, 150)
        assert a == pytest.approx(200_000 / 150)

    def test_drag_reduces_acceleration(self):
        a_clean = md.stage_acceleration_m_s2(200_000, 0, 0, 150)
        a_drag = md.stage_acceleration_m_s2(200_000, 1000, 0, 150)
        assert a_drag < a_clean

    def test_net_negative_clamped_to_zero(self):
        a = md.stage_acceleration_m_s2(100, 200, 0, 150)
        assert a == 0.0

    def test_zero_mass(self):
        assert md.stage_acceleration_m_s2(200_000, 0, 0, 0) == 0.0


# ---------------------------------------------------------------------------
# 8. TestExitVelocity (4 tests)
# ---------------------------------------------------------------------------
class TestExitVelocity:
    def test_from_rest(self):
        v = md.exit_velocity_m_s(0, 1000, 10)
        assert v == pytest.approx(math.sqrt(2 * 1000 * 10))

    def test_preserves_entry_when_no_accel(self):
        v = md.exit_velocity_m_s(100, 0, 10)
        assert v == pytest.approx(100)

    def test_both_zero(self):
        assert md.exit_velocity_m_s(0, 0, 10) == 0.0

    def test_increases_with_acceleration(self):
        v = md.exit_velocity_m_s(100, 100, 10)
        assert v > 100


# ---------------------------------------------------------------------------
# 9. TestStageTransitTime (3 tests)
# ---------------------------------------------------------------------------
class TestStageTransitTime:
    def test_known_value(self):
        t = md.stage_transit_time_s(100, 200, 10)
        assert t == pytest.approx(10.0 / 150.0)

    def test_zero_velocity_returns_inf(self):
        assert md.stage_transit_time_s(0, 0, 10) == float("inf")

    def test_positive_for_moving(self):
        t = md.stage_transit_time_s(100, 100, 10)
        assert 0 < t < float("inf")


# ---------------------------------------------------------------------------
# 10. TestResistiveHeat (5 tests)
# ---------------------------------------------------------------------------
class TestResistiveHeat:
    def test_known_value(self):
        q = md.resistive_heat_j(50_000, 0.001, 0.001)
        assert q == pytest.approx(2500)

    def test_zero_current(self):
        assert md.resistive_heat_j(0, 0.001, 0.001) == 0.0

    def test_zero_resistance(self):
        assert md.resistive_heat_j(50_000, 0, 0.001) == 0.0

    def test_zero_duration(self):
        assert md.resistive_heat_j(50_000, 0.001, 0) == 0.0

    def test_heat_small_fraction_of_ke(self):
        heat = md.resistive_heat_j(50_000, 0.001, 0.001)
        ke = md.kinetic_energy_j(100, 3000)
        assert heat / ke < 0.01


# ---------------------------------------------------------------------------
# 11. TestMagneticFieldEnergy (4 tests)
# ---------------------------------------------------------------------------
class TestMagneticFieldEnergy:
    def test_known_value(self):
        e = md.magnetic_field_energy_j(2.0, 1.0)
        expected = (4.0 / (2.0 * md.VACUUM_PERMEABILITY_H_M)) * 1.0
        assert e == pytest.approx(expected)

    def test_zero_field(self):
        assert md.magnetic_field_energy_j(0, 1.0) == 0.0

    def test_zero_volume(self):
        assert md.magnetic_field_energy_j(2.0, 0) == 0.0

    def test_proportional_to_volume(self):
        e1 = md.magnetic_field_energy_j(2.0, 1.0)
        e2 = md.magnetic_field_energy_j(2.0, 2.0)
        assert e2 == pytest.approx(2 * e1)


# ---------------------------------------------------------------------------
# 12. TestAltitudeGain (3 tests)
# ---------------------------------------------------------------------------
class TestAltitudeGain:
    def test_30_degrees(self):
        h = md.altitude_gain_m(2000, 30)
        assert h == pytest.approx(2000 * math.sin(math.radians(30)))

    def test_zero_elevation(self):
        assert md.altitude_gain_m(2000, 0) == pytest.approx(0, abs=1e-9)

    def test_90_degrees(self):
        assert md.altitude_gain_m(2000, 90) == pytest.approx(2000)


# ---------------------------------------------------------------------------
# 13. TestGLoad (3 tests)
# ---------------------------------------------------------------------------
class TestGLoad:
    def test_1g(self):
        assert md.g_load(md.G0_M_S2) == pytest.approx(1.0)

    def test_zero(self):
        assert md.g_load(0) == 0.0

    def test_10g(self):
        assert md.g_load(10 * md.G0_M_S2) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 14. TestTrackConfig (4 tests)
# ---------------------------------------------------------------------------
class TestTrackConfig:
    def test_stage_length(self):
        c = md.TrackConfig()
        assert c.stage_length_m == pytest.approx(2000.0 / 200)

    def test_max_force(self):
        c = md.TrackConfig()
        assert c.max_force_n == pytest.approx(200_000)

    def test_coil_volume_positive(self):
        c = md.TrackConfig()
        assert c.coil_volume_m3 > 0

    def test_custom_values(self):
        c = md.TrackConfig(track_length_m=5000, num_stages=100)
        assert c.stage_length_m == pytest.approx(50)


# ---------------------------------------------------------------------------
# 15. TestLaunchState (3 tests)
# ---------------------------------------------------------------------------
class TestLaunchState:
    def test_total_mass(self):
        s = md.LaunchState(payload_mass_kg=100, sled_mass_kg=50)
        assert s.total_mass_kg == pytest.approx(150)

    def test_init_stage_temps(self):
        s = md.LaunchState()
        s.init_stage_temps(200)
        assert len(s.stage_temps_k) == 200
        assert all(t == md.AMBIENT_TRACK_TEMP_K for t in s.stage_temps_k)

    def test_init_stage_temps_idempotent(self):
        s = md.LaunchState()
        s.init_stage_temps(200)
        s.stage_temps_k[0] = 300.0
        s.init_stage_temps(200)
        assert s.stage_temps_k[0] == 300.0  # not overwritten


# ---------------------------------------------------------------------------
# 16. TestTick (9 tests)
# ---------------------------------------------------------------------------
class TestTick:
    def test_first_tick_increases_velocity(self):
        config = md.create_track()
        state = md.create_launch()
        result = md.tick(config, state)
        assert state.velocity_m_s > 0
        assert result.exit_velocity_m_s > 0

    def test_advances_stage(self):
        config = md.create_track()
        state = md.create_launch()
        md.tick(config, state)
        assert state.current_stage == 1

    def test_completed_after_all_stages(self):
        config = md.create_track(num_stages=5)
        state = md.create_launch()
        for _ in range(5):
            md.tick(config, state)
        assert state.launch_complete

    def test_already_complete_noop(self):
        config = md.create_track(num_stages=2)
        state = md.create_launch()
        md.tick(config, state)
        md.tick(config, state)
        assert state.launch_complete
        v = state.velocity_m_s
        md.tick(config, state)
        assert state.velocity_m_s == v

    def test_already_failed_noop(self):
        config = md.TrackConfig(coil_resistance_ohm=10.0)
        state = md.create_launch()
        md.tick(config, state)
        assert state.launch_failed
        v = state.velocity_m_s
        md.tick(config, state)
        assert state.velocity_m_s == v

    def test_velocity_monotonic(self):
        config = md.create_track(num_stages=10)
        state = md.create_launch()
        prev_v = 0.0
        for _ in range(10):
            md.tick(config, state)
            assert state.velocity_m_s >= prev_v
            prev_v = state.velocity_m_s

    def test_energy_increases(self):
        config = md.create_track(num_stages=10)
        state = md.create_launch()
        for _ in range(10):
            prev_e = state.total_energy_in_j
            md.tick(config, state)
            assert state.total_energy_in_j >= prev_e

    def test_position_advances(self):
        config = md.create_track()
        state = md.create_launch()
        md.tick(config, state)
        assert state.position_m > 0

    def test_altitude_increases_with_elevation(self):
        config = md.create_track(elevation_deg=30)
        state = md.create_launch()
        md.tick(config, state)
        assert state.altitude_m > 0


# ---------------------------------------------------------------------------
# 17. TestConservation (6 tests)
# ---------------------------------------------------------------------------
class TestConservation:
    def test_energy_accounting(self):
        config = md.create_track(num_stages=20)
        state = md.create_launch()
        md.run_launch(config, state)
        ke_final = md.kinetic_energy_j(state.total_mass_kg, state.velocity_m_s)
        assert state.total_energy_in_j > ke_final

    def test_velocity_never_negative(self):
        config = md.create_track()
        state = md.create_launch()
        results = md.run_launch(config, state)
        for r in results:
            assert r.exit_velocity_m_s >= 0

    def test_mass_constant(self):
        config = md.create_track()
        state = md.create_launch()
        mass_before = state.total_mass_kg
        md.run_launch(config, state)
        assert state.total_mass_kg == pytest.approx(mass_before)

    def test_stage_temps_bounded(self):
        config = md.create_track()
        state = md.create_launch()
        md.run_launch(config, state)
        for t in state.stage_temps_k:
            assert md.AMBIENT_TRACK_TEMP_K <= t <= md.MAX_TRACK_TEMP_K

    def test_position_matches_stages(self):
        config = md.create_track()
        state = md.create_launch()
        md.run_launch(config, state)
        assert state.position_m == pytest.approx(config.track_length_m)

    def test_time_positive(self):
        config = md.create_track()
        state = md.create_launch()
        md.run_launch(config, state)
        assert state.elapsed_time_s > 0


# ---------------------------------------------------------------------------
# 18. TestThermalLimits (3 tests)
# ---------------------------------------------------------------------------
class TestThermalLimits:
    def test_temps_start_ambient(self):
        state = md.LaunchState()
        state.init_stage_temps(10)
        assert all(t == md.AMBIENT_TRACK_TEMP_K for t in state.stage_temps_k)

    def test_temps_rise_during_launch(self):
        config = md.create_track()
        state = md.create_launch()
        md.run_launch(config, state)
        assert any(t > md.AMBIENT_TRACK_TEMP_K for t in state.stage_temps_k)

    def test_overheated_track_fails(self):
        config = md.TrackConfig(coil_resistance_ohm=10.0)
        state = md.create_launch()
        md.tick(config, state)
        assert state.launch_failed
        assert "overheated" in state.failure_reason.lower()


# ---------------------------------------------------------------------------
# 19. TestCooling (3 tests)
# ---------------------------------------------------------------------------
class TestCooling:
    def test_cool_reduces_temp(self):
        config = md.create_track(num_stages=10)
        state = md.create_launch()
        md.run_launch(config, state)
        max_before = max(state.stage_temps_k)
        md.cool_track(state, 100)
        max_after = max(state.stage_temps_k)
        assert max_after < max_before

    def test_cool_approaches_ambient(self):
        config = md.create_track(num_stages=10)
        state = md.create_launch()
        md.run_launch(config, state)
        md.cool_track(state, 1_000_000)
        for t in state.stage_temps_k:
            assert abs(t - md.AMBIENT_TRACK_TEMP_K) < 0.01

    def test_cool_empty_noop(self):
        state = md.LaunchState()
        md.cool_track(state, 100)  # should not error


# ---------------------------------------------------------------------------
# 20. TestGLoadConstraints (4 tests)
# ---------------------------------------------------------------------------
class TestGLoadConstraints:
    def test_g_clamped_within_limit(self):
        config = md.create_track()
        state = md.create_launch()
        md.run_launch(config, state)
        assert state.peak_g_load <= config.g_limit * 1.01

    def test_low_g_limit_reduces_velocity(self):
        config_low = md.create_track(g_limit=3)
        config_high = md.create_track(g_limit=50)
        state_low = md.create_launch()
        state_high = md.create_launch()
        md.run_launch(config_low, state_low)
        md.run_launch(config_high, state_high)
        assert state_low.velocity_m_s < state_high.velocity_m_s

    def test_same_velocity_when_clamped(self):
        """Two different masses get same velocity when g-limit clamps both."""
        config = md.create_track(g_limit=5)
        state_light = md.create_launch(payload_kg=50)
        state_heavy = md.create_launch(payload_kg=200)
        md.run_launch(config, state_light)
        md.run_launch(config, state_heavy)
        assert state_light.velocity_m_s == pytest.approx(
            state_heavy.velocity_m_s, rel=1e-3
        )

    def test_different_velocity_without_clamp(self):
        """With g_limit=500, clamping doesn't apply and lighter goes faster."""
        config = md.create_track(g_limit=500)
        state_light = md.create_launch(payload_kg=50)
        state_heavy = md.create_launch(payload_kg=500)
        md.run_launch(config, state_light)
        md.run_launch(config, state_heavy)
        assert state_light.velocity_m_s > state_heavy.velocity_m_s


# ---------------------------------------------------------------------------
# 21. TestFullLaunch (6 tests)
# ---------------------------------------------------------------------------
class TestFullLaunch:
    def test_default_completes(self):
        config = md.create_track()
        state = md.create_launch()
        md.run_launch(config, state)
        assert state.launch_complete

    def test_returns_stage_results(self):
        results = md.run_launch()
        assert len(results) > 0
        assert isinstance(results[0], md.StageResult)

    def test_exit_velocity_positive(self):
        config = md.create_track()
        state = md.create_launch()
        md.run_launch(config, state)
        assert state.velocity_m_s > 0

    def test_with_none_defaults(self):
        results = md.run_launch(config=None, state=None)
        assert len(results) > 0

    def test_heavy_payload_slower(self):
        """With g_limit=500 (no clamping), heavier payload is slower."""
        config = md.create_track(g_limit=500)
        state_light = md.create_launch(payload_kg=50)
        state_heavy = md.create_launch(payload_kg=500)
        md.run_launch(config, state_light)
        md.run_launch(config, state_heavy)
        assert state_light.velocity_m_s > state_heavy.velocity_m_s

    def test_longer_track_faster(self):
        config_short = md.create_track(track_length_m=1000)
        config_long = md.create_track(track_length_m=5000)
        state_short = md.create_launch()
        state_long = md.create_launch()
        md.run_launch(config_short, state_short)
        md.run_launch(config_long, state_long)
        assert state_long.velocity_m_s > state_short.velocity_m_s


# ---------------------------------------------------------------------------
# 22. TestOrbitalTargeting (4 tests)
# ---------------------------------------------------------------------------
class TestOrbitalTargeting:
    def test_lmo_feasibility(self):
        config = md.create_track(track_length_m=30_000, g_limit=50)
        result = md.can_reach_target(md.LMO_VELOCITY_M_S, config)
        assert result["go"]

    def test_escape_velocity(self):
        config = md.create_track(track_length_m=30_000, g_limit=50)
        result = md.can_reach_target(md.ESCAPE_VELOCITY_M_S, config)
        assert result["go"]

    def test_result_has_expected_keys(self):
        result = md.can_reach_target(md.LMO_VELOCITY_M_S)
        expected_keys = {
            "exit_velocity_m_s",
            "target_velocity_m_s",
            "velocity_margin_m_s",
            "go",
            "peak_g_load",
            "total_energy_mj",
            "kinetic_energy_payload_mj",
            "kinetic_energy_total_mj",
            "drag_loss_mj",
            "thermal_loss_mj",
            "launch_time_s",
            "stages_fired",
            "failed",
            "failure_reason",
            "altitude_at_exit_m",
            "payload_kg",
        }
        assert expected_keys.issubset(result.keys())

    def test_failed_flag_on_insufficient(self):
        config = md.create_track(track_length_m=200, g_limit=10)
        result = md.can_reach_target(md.ESCAPE_VELOCITY_M_S, config)
        assert not result["go"]


# ---------------------------------------------------------------------------
# 23. TestOptimalTrack (2 tests)
# ---------------------------------------------------------------------------
class TestOptimalTrack:
    def test_finds_track_for_lmo(self):
        result = md.optimal_track_for_target(md.LMO_VELOCITY_M_S, max_g=50)
        assert result.get("go", False)

    def test_result_has_track_length(self):
        result = md.optimal_track_for_target(md.LMO_VELOCITY_M_S, max_g=50)
        assert "track_length_m" in result


# ---------------------------------------------------------------------------
# 24. TestRunSimulation (7 tests)
# ---------------------------------------------------------------------------
class TestRunSimulation:
    def test_default_returns_dict(self):
        r = md.run_simulation()
        assert isinstance(r, dict)

    def test_lmo_target(self):
        r = md.run_simulation(target="lmo")
        assert r["target_velocity_m_s"] == md.LMO_VELOCITY_M_S

    def test_escape_target(self):
        r = md.run_simulation(target="escape")
        assert r["target_velocity_m_s"] == md.ESCAPE_VELOCITY_M_S

    def test_earth_target(self):
        r = md.run_simulation(target="earth")
        assert r["target_velocity_m_s"] == md.EARTH_TRANSFER_M_S

    def test_unknown_target_defaults_lmo(self):
        r = md.run_simulation(target="unknown")
        assert r["target_velocity_m_s"] == md.LMO_VELOCITY_M_S

    def test_go_flag_matches_velocity(self):
        r = md.run_simulation(track_length_m=30_000, g_limit=50, target="lmo")
        expected_go = r["exit_velocity_m_s"] >= r["target_velocity_m_s"] and not r[
            "failed"
        ]
        assert r["go"] == expected_go

    def test_efficiency_between_zero_and_one(self):
        r = md.run_simulation(track_length_m=5000)
        assert 0 < r["payload_efficiency"] <= 1.0


# ---------------------------------------------------------------------------
# 25. TestPhysicalBounds (16 parametrized tests)
# ---------------------------------------------------------------------------
class TestPhysicalBounds:
    @pytest.mark.parametrize(
        "current,length,field,coils",
        [
            (10_000, 0.5, 1.0, 1),
            (50_000, 0.5, 2.0, 4),
            (20_000, 1.0, 1.5, 2),
            (100_000, 0.3, 3.0, 1),
        ],
    )
    def test_force_positive(self, current, length, field, coils):
        assert md.lorentz_force_n(current, length, field, coils) > 0

    @pytest.mark.parametrize(
        "mass,velocity",
        [
            (1, 100),
            (10, 500),
            (100, 1000),
            (1000, 3000),
        ],
    )
    def test_ke_nonnegative(self, mass, velocity):
        assert md.kinetic_energy_j(mass, velocity) > 0

    @pytest.mark.parametrize(
        "alt_low,alt_high",
        [
            (0, 1_000),
            (1_000, 5_000),
            (5_000, 20_000),
            (20_000, 100_000),
        ],
    )
    def test_density_decreasing(self, alt_low, alt_high):
        assert md.mars_air_density_kg_m3(alt_low) > md.mars_air_density_kg_m3(
            alt_high
        )

    @pytest.mark.parametrize(
        "velocity",
        [100, 500, 1000, 3000],
    )
    def test_drag_positive_at_speed(self, velocity):
        assert md.drag_force_n(velocity, 0) > 0


# ---------------------------------------------------------------------------
# 26. TestEdgeCases (6 tests)
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_zero_stages(self):
        config = md.create_track(num_stages=0)
        state = md.create_launch()
        results = md.run_launch(config, state)
        assert state.launch_complete
        assert state.velocity_m_s == 0.0

    def test_single_stage(self):
        config = md.create_track(num_stages=1)
        state = md.create_launch()
        results = md.run_launch(config, state)
        assert state.launch_complete
        assert state.velocity_m_s > 0

    def test_very_heavy_payload(self):
        config = md.create_track()
        state = md.create_launch(payload_kg=10_000)
        md.run_launch(config, state)
        assert state.launch_complete
        assert state.velocity_m_s > 0

    def test_zero_elevation(self):
        config = md.create_track(elevation_deg=0)
        state = md.create_launch()
        md.run_launch(config, state)
        assert state.launch_complete
        assert state.altitude_m == pytest.approx(0, abs=1e-9)

    def test_max_elevation(self):
        config = md.create_track(elevation_deg=90)
        state = md.create_launch()
        md.run_launch(config, state)
        assert state.launch_complete
        assert state.velocity_m_s > 0

    def test_very_short_track(self):
        config = md.create_track(track_length_m=1)
        state = md.create_launch()
        md.run_launch(config, state)
        assert state.launch_complete
        assert state.velocity_m_s > 0

    def test_minimal_payload(self):
        config = md.create_track(g_limit=500)
        state = md.create_launch(payload_kg=0.1, sled_kg=0.1)
        md.run_launch(config, state)
        assert state.launch_complete
        assert state.velocity_m_s > 0


# ---------------------------------------------------------------------------
# 27. TestSmoke (8 tests)
# ---------------------------------------------------------------------------
class TestSmoke:
    def test_import_module(self):
        assert hasattr(md, "lorentz_force_n")
        assert hasattr(md, "run_launch")

    def test_constants_defined(self):
        assert md.G0_M_S2 > 0
        assert md.MARS_GRAVITY_M_S2 > 0
        assert md.LMO_VELOCITY_M_S == 3550
        assert md.ESCAPE_VELOCITY_M_S == 5030
        assert md.EARTH_TRANSFER_M_S == 5700

    def test_create_track_returns_config(self):
        c = md.create_track()
        assert isinstance(c, md.TrackConfig)

    def test_create_launch_returns_state(self):
        s = md.create_launch()
        assert isinstance(s, md.LaunchState)

    def test_run_launch_returns_list(self):
        results = md.run_launch()
        assert isinstance(results, list)

    def test_can_reach_target_returns_dict(self):
        result = md.can_reach_target(md.LMO_VELOCITY_M_S)
        assert isinstance(result, dict)

    def test_run_simulation_returns_dict(self):
        result = md.run_simulation()
        assert isinstance(result, dict)

    def test_stage_result_fields(self):
        r = md.StageResult()
        assert hasattr(r, "stage_index")
        assert hasattr(r, "entry_velocity_m_s")
        assert hasattr(r, "exit_velocity_m_s")
        assert hasattr(r, "acceleration_m_s2")
        assert hasattr(r, "g_load")
        assert hasattr(r, "force_n")
        assert hasattr(r, "drag_n")
        assert hasattr(r, "transit_time_s")
        assert hasattr(r, "stage_temp_k")


# ---------------------------------------------------------------------------
# 28. TestPowerRequirements (3 tests)
# ---------------------------------------------------------------------------
class TestPowerRequirements:
    def test_known_power(self):
        assert md.power_required_w(1e6, 1.0) == pytest.approx(1e6)

    def test_zero_duration_returns_inf(self):
        assert md.power_required_w(1e6, 0) == float("inf")

    def test_launch_power_reasonable(self):
        config = md.create_track()
        state = md.create_launch()
        md.run_launch(config, state)
        power = md.power_required_w(state.total_energy_in_j, state.elapsed_time_s)
        assert 1e6 < power < 1e9  # MW range

    def test_power_scales_with_energy(self):
        p1 = md.power_required_w(1e6, 2.0)
        p2 = md.power_required_w(2e6, 2.0)
        assert p2 == pytest.approx(2 * p1)
