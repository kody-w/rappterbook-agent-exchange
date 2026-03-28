"""Tests for mass_driver.py -- Mars Electromagnetic Launch Rail.

148 tests covering Lorentz force, kinetic energy, drag, gravity,
stage acceleration, track tick, conservation laws, thermal limits,
g-load constraints, full launch simulation, orbital targeting,
track optimization, physical bounds, edge cases, and smoke tests.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mass_driver as md


@pytest.fixture
def default_track():
    return md.create_track()

@pytest.fixture
def default_launch():
    return md.create_launch()

@pytest.fixture
def short_track():
    return md.create_track(track_length_m=500.0, num_stages=50)

@pytest.fixture
def long_track():
    return md.create_track(track_length_m=10_000.0, num_stages=500)

@pytest.fixture
def heavy_payload():
    return md.create_launch(payload_kg=500.0, sled_kg=100.0)

@pytest.fixture
def zero_payload():
    return md.create_launch(payload_kg=0.0, sled_kg=0.0)


class TestLorentzForce:
    def test_positive_for_valid_inputs(self):
        f = md.lorentz_force_n(50_000.0, 0.5, 2.0, 4)
        assert f > 0.0

    def test_scales_linearly_with_current(self):
        f1 = md.lorentz_force_n(25_000.0, 0.5, 2.0, 1)
        f2 = md.lorentz_force_n(50_000.0, 0.5, 2.0, 1)
        assert abs(f2 - 2.0 * f1) < 1e-6

    def test_scales_linearly_with_field(self):
        f1 = md.lorentz_force_n(50_000.0, 0.5, 1.0, 1)
        f2 = md.lorentz_force_n(50_000.0, 0.5, 2.0, 1)
        assert abs(f2 - 2.0 * f1) < 1e-6

    def test_scales_with_num_coils(self):
        f1 = md.lorentz_force_n(50_000.0, 0.5, 2.0, 1)
        f4 = md.lorentz_force_n(50_000.0, 0.5, 2.0, 4)
        assert abs(f4 - 4.0 * f1) < 1e-6

    def test_zero_current_gives_zero(self):
        assert md.lorentz_force_n(0.0, 0.5, 2.0, 4) == 0.0

    def test_zero_field_gives_zero(self):
        assert md.lorentz_force_n(50_000.0, 0.5, 0.0, 4) == 0.0

    def test_zero_length_gives_zero(self):
        assert md.lorentz_force_n(50_000.0, 0.0, 2.0, 4) == 0.0

    def test_negative_current_gives_zero(self):
        assert md.lorentz_force_n(-1.0, 0.5, 2.0, 4) == 0.0

    def test_default_track_force_reasonable(self):
        f = md.lorentz_force_n(50_000.0, 0.5, 2.0, 4)
        assert 100_000 < f < 500_000


class TestKineticEnergy:
    def test_positive_for_moving_mass(self):
        assert md.kinetic_energy_j(100.0, 1000.0) > 0.0

    def test_zero_velocity_gives_zero(self):
        assert md.kinetic_energy_j(100.0, 0.0) == 0.0

    def test_zero_mass_gives_zero(self):
        assert md.kinetic_energy_j(0.0, 1000.0) == 0.0

    def test_half_mv_squared(self):
        ke = md.kinetic_energy_j(10.0, 100.0)
        assert abs(ke - 0.5 * 10.0 * 100.0 ** 2) < 1e-6

    def test_scales_quadratically_with_velocity(self):
        ke1 = md.kinetic_energy_j(100.0, 1000.0)
        ke2 = md.kinetic_energy_j(100.0, 2000.0)
        assert abs(ke2 / ke1 - 4.0) < 1e-6

    def test_negative_mass_gives_zero(self):
        assert md.kinetic_energy_j(-1.0, 100.0) == 0.0


class TestVelocityFromEnergy:
    def test_roundtrip_with_kinetic_energy(self):
        v = 3000.0
        m = 150.0
        ke = md.kinetic_energy_j(m, v)
        v_back = md.velocity_from_energy_m_s(ke, m)
        assert abs(v_back - v) < 1e-6

    def test_zero_energy_gives_zero(self):
        assert md.velocity_from_energy_m_s(0.0, 100.0) == 0.0

    def test_zero_mass_gives_zero(self):
        assert md.velocity_from_energy_m_s(1000.0, 0.0) == 0.0

    def test_negative_energy_gives_zero(self):
        assert md.velocity_from_energy_m_s(-1.0, 100.0) == 0.0


class TestMarsAirDensity:
    def test_surface_density(self):
        rho = md.mars_air_density_kg_m3(0.0)
        assert abs(rho - md.MARS_SURFACE_DENSITY_KG_M3) < 1e-9

    def test_decreases_with_altitude(self):
        assert md.mars_air_density_kg_m3(1000.0) < md.mars_air_density_kg_m3(0.0)

    def test_at_scale_height(self):
        rho = md.mars_air_density_kg_m3(md.MARS_SCALE_HEIGHT_M)
        expected = md.MARS_SURFACE_DENSITY_KG_M3 / math.e
        assert abs(rho - expected) < 1e-9

    def test_always_positive(self):
        for alt in [0, 100, 1000, 10_000, 100_000]:
            assert md.mars_air_density_kg_m3(float(alt)) > 0.0

    def test_negative_altitude_uses_surface(self):
        assert md.mars_air_density_kg_m3(-100.0) == md.mars_air_density_kg_m3(0.0)

    def test_very_high_altitude_near_zero(self):
        rho = md.mars_air_density_kg_m3(200_000.0)
        assert rho < 1e-6


class TestDragForce:
    def test_positive_for_moving_sled(self):
        assert md.drag_force_n(1000.0) > 0.0

    def test_zero_velocity_gives_zero(self):
        assert md.drag_force_n(0.0) == 0.0

    def test_scales_quadratically_with_velocity(self):
        d1 = md.drag_force_n(1000.0)
        d2 = md.drag_force_n(2000.0)
        assert abs(d2 / d1 - 4.0) < 0.01

    def test_decreases_with_altitude(self):
        assert md.drag_force_n(1000.0, 10_000.0) < md.drag_force_n(1000.0, 0.0)

    def test_negative_velocity_gives_zero(self):
        assert md.drag_force_n(-1.0) == 0.0

    def test_mars_drag_much_less_than_earth(self):
        f_mars = md.drag_force_n(3000.0, 0.0)
        f_earth = 0.5 * 1.2 * 3000.0**2 * md.DRAG_COEFF * md.SLED_CROSS_SECTION_M2
        assert f_mars < f_earth * 0.05


class TestGravityComponent:
    def test_zero_at_horizontal(self):
        assert md.gravity_component_n(100.0, 0.0) < 1e-9

    def test_max_at_vertical(self):
        g = md.gravity_component_n(100.0, 90.0)
        expected = 100.0 * md.MARS_GRAVITY_M_S2
        assert abs(g - expected) < 1e-6

    def test_at_30_degrees(self):
        g = md.gravity_component_n(100.0, 30.0)
        expected = 100.0 * md.MARS_GRAVITY_M_S2 * math.sin(math.radians(30.0))
        assert abs(g - expected) < 1e-6

    def test_zero_mass_gives_zero(self):
        assert md.gravity_component_n(0.0, 45.0) == 0.0

    def test_always_non_negative(self):
        for deg in [0, 15, 30, 45, 60, 75, 90]:
            assert md.gravity_component_n(100.0, float(deg)) >= 0.0


class TestStageAcceleration:
    def test_basic_case(self):
        a = md.stage_acceleration_m_s2(10_000.0, 100.0, 50.0, 100.0)
        expected = (10_000.0 - 100.0 - 50.0) / 100.0
        assert abs(a - expected) < 1e-6

    def test_zero_mass_gives_zero(self):
        assert md.stage_acceleration_m_s2(10_000.0, 0.0, 0.0, 0.0) == 0.0

    def test_drag_exceeds_force_gives_zero(self):
        a = md.stage_acceleration_m_s2(100.0, 200.0, 0.0, 100.0)
        assert a == 0.0

    def test_never_negative(self):
        for force in [0, 100, 1000]:
            for drag in [0, 500, 2000]:
                for grav in [0, 300, 1000]:
                    a = md.stage_acceleration_m_s2(
                        float(force), float(drag), float(grav), 100.0)
                    assert a >= 0.0


class TestExitVelocity:
    def test_increases_with_positive_accel(self):
        v = md.exit_velocity_m_s(100.0, 500.0, 10.0)
        assert v > 100.0

    def test_unchanged_with_zero_accel(self):
        v = md.exit_velocity_m_s(100.0, 0.0, 10.0)
        assert abs(v - 100.0) < 1e-6

    def test_from_rest(self):
        v = md.exit_velocity_m_s(0.0, 1000.0, 5.0)
        expected = math.sqrt(2.0 * 1000.0 * 5.0)
        assert abs(v - expected) < 1e-6

    def test_zero_length_gives_entry(self):
        v = md.exit_velocity_m_s(500.0, 1000.0, 0.0)
        assert abs(v - 500.0) < 1e-6


class TestStageTransitTime:
    def test_basic(self):
        t = md.stage_transit_time_s(100.0, 200.0, 10.0)
        expected = 10.0 / 150.0
        assert abs(t - expected) < 1e-9

    def test_zero_velocity_gives_infinity(self):
        t = md.stage_transit_time_s(0.0, 0.0, 10.0)
        assert t == float("inf")

    def test_high_velocity_short_time(self):
        t = md.stage_transit_time_s(3000.0, 3100.0, 10.0)
        assert t < 0.01


class TestResistiveHeat:
    def test_positive_for_valid(self):
        assert md.resistive_heat_j(50_000.0, 0.001, 0.001) > 0.0

    def test_zero_current_gives_zero(self):
        assert md.resistive_heat_j(0.0, 0.001, 1.0) == 0.0

    def test_zero_resistance_gives_zero(self):
        assert md.resistive_heat_j(50_000.0, 0.0, 1.0) == 0.0

    def test_i_squared_r_t(self):
        q = md.resistive_heat_j(1000.0, 0.01, 2.0)
        expected = 1000.0 ** 2 * 0.01 * 2.0
        assert abs(q - expected) < 1e-6

    def test_superconducting_negligible_vs_ke(self):
        q = md.resistive_heat_j(50_000.0, 0.001, 0.001)
        ke_payload = md.kinetic_energy_j(150.0, 3000.0)
        assert q / ke_payload < 0.01


class TestMagneticFieldEnergy:
    def test_positive_for_valid(self):
        assert md.magnetic_field_energy_j(2.0, 0.01) > 0.0

    def test_zero_field_gives_zero(self):
        assert md.magnetic_field_energy_j(0.0, 1.0) == 0.0

    def test_zero_volume_gives_zero(self):
        assert md.magnetic_field_energy_j(2.0, 0.0) == 0.0

    def test_scales_with_b_squared(self):
        e1 = md.magnetic_field_energy_j(1.0, 1.0)
        e2 = md.magnetic_field_energy_j(2.0, 1.0)
        assert abs(e2 / e1 - 4.0) < 1e-6


class TestAltitudeGain:
    def test_zero_at_horizontal(self):
        assert md.altitude_gain_m(1000.0, 0.0) < 1e-9

    def test_full_at_vertical(self):
        assert abs(md.altitude_gain_m(1000.0, 90.0) - 1000.0) < 1e-6

    def test_at_30_degrees(self):
        alt = md.altitude_gain_m(2000.0, 30.0)
        expected = 2000.0 * math.sin(math.radians(30.0))
        assert abs(alt - expected) < 1e-6


class TestGLoad:
    def test_1g_at_g0(self):
        assert abs(md.g_load(md.G0_M_S2) - 1.0) < 1e-6

    def test_zero_at_zero(self):
        assert md.g_load(0.0) == 0.0

    def test_mars_gravity_is_subg(self):
        assert md.g_load(md.MARS_GRAVITY_M_S2) < 1.0


class TestTrackConfig:
    def test_stage_length(self, default_track):
        expected = md.DEFAULT_TRACK_LENGTH_M / md.DEFAULT_NUM_STAGES
        assert abs(default_track.stage_length_m - expected) < 1e-6

    def test_max_force_positive(self, default_track):
        assert default_track.max_force_n > 0.0

    def test_coil_volume_positive(self, default_track):
        assert default_track.coil_volume_m3 > 0.0

    def test_zero_stages_gives_zero_length(self):
        cfg = md.TrackConfig(num_stages=0)
        assert cfg.stage_length_m == 0.0


class TestLaunchState:
    def test_total_mass(self, default_launch):
        expected = md.DEFAULT_PAYLOAD_KG + md.DEFAULT_SLED_KG
        assert abs(default_launch.total_mass_kg - expected) < 1e-6

    def test_initial_velocity_zero(self, default_launch):
        assert default_launch.velocity_m_s == 0.0

    def test_init_stage_temps(self, default_launch):
        default_launch.init_stage_temps(100)
        assert len(default_launch.stage_temps_k) == 100
        assert all(t == md.AMBIENT_TRACK_TEMP_K for t in default_launch.stage_temps_k)


class TestTick:
    def test_first_tick_accelerates(self, default_track, default_launch):
        result = md.tick(default_track, default_launch)
        assert result.exit_velocity_m_s > 0.0

    def test_velocity_monotonically_increases(self, default_track, default_launch):
        prev_v = 0.0
        for _ in range(10):
            result = md.tick(default_track, default_launch)
            assert result.exit_velocity_m_s >= prev_v
            prev_v = result.exit_velocity_m_s

    def test_position_advances(self, default_track, default_launch):
        md.tick(default_track, default_launch)
        assert default_launch.position_m > 0.0

    def test_altitude_increases(self, default_track, default_launch):
        md.tick(default_track, default_launch)
        assert default_launch.altitude_m > 0.0

    def test_stage_index_increments(self, default_track, default_launch):
        md.tick(default_track, default_launch)
        assert default_launch.current_stage == 1

    def test_energy_consumed(self, default_track, default_launch):
        md.tick(default_track, default_launch)
        assert default_launch.total_energy_in_j > 0.0

    def test_g_load_within_limit(self, default_track, default_launch):
        for _ in range(default_track.num_stages):
            result = md.tick(default_track, default_launch)
            if default_launch.launch_complete or default_launch.launch_failed:
                break
            assert result.g_load <= default_track.g_limit * 1.01

    def test_completed_launch_is_noop(self, default_track, default_launch):
        default_launch.launch_complete = True
        result = md.tick(default_track, default_launch)
        assert result.exit_velocity_m_s == default_launch.velocity_m_s

    def test_failed_launch_is_noop(self, default_track, default_launch):
        default_launch.launch_failed = True
        v_before = default_launch.velocity_m_s
        result = md.tick(default_track, default_launch)
        assert result.exit_velocity_m_s == v_before


class TestConservation:
    def test_energy_conservation(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        ke_final = md.kinetic_energy_j(state.total_mass_kg, state.velocity_m_s)
        assert state.total_energy_in_j >= ke_final * 0.5

    def test_velocity_monotonic(self, default_track):
        state = md.create_launch()
        prev_v = 0.0
        for _ in range(default_track.num_stages + 1):
            result = md.tick(default_track, state)
            assert result.exit_velocity_m_s >= prev_v - 1e-9
            prev_v = result.exit_velocity_m_s
            if state.launch_complete or state.launch_failed:
                break

    def test_mass_constant(self, default_track):
        state = md.create_launch()
        initial_mass = state.total_mass_kg
        md.run_launch(default_track, state)
        assert abs(state.total_mass_kg - initial_mass) < 1e-9

    def test_position_equals_stages_times_length(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        expected_pos = state.current_stage * default_track.stage_length_m
        assert abs(state.position_m - expected_pos) < 1e-6

    def test_elapsed_time_positive(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        assert state.elapsed_time_s > 0.0

    def test_altitude_matches_geometry(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        expected = md.altitude_gain_m(state.position_m, default_track.elevation_deg)
        assert abs(state.altitude_m - expected) < 1e-3


class TestThermalLimits:
    def test_high_resistance_causes_overheat(self):
        cfg = md.TrackConfig(
            track_length_m=2000.0, num_stages=200,
            current_a=50_000.0, magnetic_field_t=2.0,
            coil_resistance_ohm=10.0, g_limit=50.0,
        )
        state = md.create_launch()
        md.run_launch(cfg, state)
        assert state.launch_failed
        assert "overheat" in state.failure_reason.lower()

    def test_superconducting_stays_cool(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        assert not state.launch_failed or "overheat" not in state.failure_reason.lower()
        if state.stage_temps_k:
            assert max(state.stage_temps_k) < md.MAX_TRACK_TEMP_K

    def test_track_temps_above_ambient(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        for i in range(state.current_stage):
            assert state.stage_temps_k[i] >= md.AMBIENT_TRACK_TEMP_K - 1e-6


class TestCooling:
    def test_cooling_reduces_temperature(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        if state.stage_temps_k:
            max_before = max(state.stage_temps_k)
            md.cool_track(state, 3600.0)
            max_after = max(state.stage_temps_k)
            assert max_after <= max_before

    def test_cooling_approaches_ambient(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        md.cool_track(state, 100_000.0)
        if state.stage_temps_k:
            for temp in state.stage_temps_k:
                assert abs(temp - md.AMBIENT_TRACK_TEMP_K) < 1.0

    def test_cooling_zero_seconds_no_change(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        temps_before = list(state.stage_temps_k)
        md.cool_track(state, 0.0)
        for a, b in zip(temps_before, state.stage_temps_k):
            assert abs(a - b) < 1e-9


class TestGLoadConstraints:
    def test_default_within_30g(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        assert state.peak_g_load <= md.DEFAULT_G_LIMIT + 0.1

    def test_fragile_cargo_within_3g(self):
        cfg = md.create_track(g_limit=md.FRAGILE_CARGO_G_LIMIT,
                              track_length_m=50_000.0, num_stages=1000)
        state = md.create_launch()
        md.run_launch(cfg, state)
        assert state.peak_g_load <= md.FRAGILE_CARGO_G_LIMIT + 0.1

    def test_hardened_cargo_allows_50g(self):
        cfg = md.create_track(g_limit=md.HARDENED_CARGO_G_LIMIT)
        state = md.create_launch()
        md.run_launch(cfg, state)
        assert state.peak_g_load <= md.HARDENED_CARGO_G_LIMIT + 0.1

    def test_higher_g_limit_higher_exit_velocity(self):
        cfg_low = md.create_track(g_limit=5.0)
        cfg_high = md.create_track(g_limit=50.0)
        s_low = md.create_launch()
        s_high = md.create_launch()
        md.run_launch(cfg_low, s_low)
        md.run_launch(cfg_high, s_high)
        assert s_high.velocity_m_s >= s_low.velocity_m_s


class TestFullLaunch:
    def test_default_launch_completes(self, default_track):
        state = md.create_launch()
        results = md.run_launch(default_track, state)
        assert state.launch_complete
        assert len(results) > 0

    def test_exit_velocity_positive(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        assert state.velocity_m_s > 0.0

    def test_all_stages_fired(self, default_track):
        state = md.create_launch()
        md.run_launch(default_track, state)
        if not state.launch_failed:
            assert state.current_stage == default_track.num_stages

    def test_heavy_payload_slower(self):
        cfg = md.create_track(track_length_m=2000.0, num_stages=200, g_limit=500.0)
        light = md.create_launch(payload_kg=50.0)
        heavy = md.create_launch(payload_kg=500.0)
        md.run_launch(cfg, light)
        md.run_launch(cfg, heavy)
        assert light.velocity_m_s > heavy.velocity_m_s

    def test_longer_track_faster(self):
        short = md.create_track(track_length_m=1000.0, num_stages=100)
        long = md.create_track(track_length_m=5000.0, num_stages=500)
        s1 = md.create_launch()
        s2 = md.create_launch()
        md.run_launch(short, s1)
        md.run_launch(long, s2)
        assert s2.velocity_m_s > s1.velocity_m_s

    def test_results_count_matches_stages(self, default_track):
        state = md.create_launch()
        results = md.run_launch(default_track, state)
        assert len(results) <= default_track.num_stages + 1


class TestOrbitalTargeting:
    def test_can_reach_lmo(self):
        result = md.can_reach_target(md.LMO_VELOCITY_M_S,
                                     md.create_track(track_length_m=5000.0, num_stages=500))
        assert "exit_velocity_m_s" in result
        assert "go" in result

    def test_longer_track_more_likely_to_reach(self):
        short = md.can_reach_target(md.LMO_VELOCITY_M_S,
                                    md.create_track(track_length_m=500.0, num_stages=50))
        long = md.can_reach_target(md.LMO_VELOCITY_M_S,
                                   md.create_track(track_length_m=20_000.0, num_stages=2000))
        assert long["exit_velocity_m_s"] > short["exit_velocity_m_s"]

    def test_escape_harder_than_lmo(self):
        cfg = md.create_track(track_length_m=5000.0, num_stages=500)
        lmo = md.can_reach_target(md.LMO_VELOCITY_M_S, cfg)
        esc = md.can_reach_target(md.ESCAPE_VELOCITY_M_S, cfg)
        assert esc["velocity_margin_m_s"] < lmo["velocity_margin_m_s"]

    def test_earth_transfer_hardest(self):
        cfg = md.create_track(track_length_m=5000.0, num_stages=500)
        lmo = md.can_reach_target(md.LMO_VELOCITY_M_S, cfg)
        eth = md.can_reach_target(md.EARTH_TRANSFER_M_S, cfg)
        assert eth["velocity_margin_m_s"] < lmo["velocity_margin_m_s"]


class TestOptimalTrack:
    def test_finds_shorter_track_than_max(self):
        result = md.optimal_track_for_target(
            md.LMO_VELOCITY_M_S, payload_kg=50.0, max_g=50.0)
        if result.get("go"):
            assert result["track_length_m"] <= 20_000.0

    def test_heavier_payload_needs_longer_track(self):
        light = md.optimal_track_for_target(
            md.LMO_VELOCITY_M_S, payload_kg=10.0, max_g=50.0)
        heavy = md.optimal_track_for_target(
            md.LMO_VELOCITY_M_S, payload_kg=500.0, max_g=50.0)
        if light.get("go") and heavy.get("go"):
            assert heavy["track_length_m"] >= light["track_length_m"]


class TestRunSimulation:
    def test_lmo_target(self):
        result = md.run_simulation(target="lmo", track_length_m=5000.0)
        assert result["target"] == "lmo"
        assert result["target_velocity_m_s"] == md.LMO_VELOCITY_M_S

    def test_escape_target(self):
        result = md.run_simulation(target="escape", track_length_m=5000.0)
        assert result["target"] == "escape"

    def test_earth_target(self):
        result = md.run_simulation(target="earth", track_length_m=5000.0)
        assert result["target"] == "earth"

    def test_unknown_target_defaults_lmo(self):
        result = md.run_simulation(target="jupiter")
        assert result["target_velocity_m_s"] == md.LMO_VELOCITY_M_S

    def test_all_fields_present(self):
        result = md.run_simulation()
        required = [
            "target", "exit_velocity_m_s", "go", "payload_kg",
            "track_length_m", "stages_fired", "launch_time_s",
            "peak_g_load", "total_energy_mj", "payload_efficiency",
            "average_power_mw", "failed",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_efficiency_between_0_and_1(self):
        result = md.run_simulation(track_length_m=5000.0)
        assert 0.0 <= result["payload_efficiency"] <= 1.0

    def test_power_positive(self):
        result = md.run_simulation()
        assert result["average_power_mw"] > 0.0


class TestPhysicalBounds:
    @pytest.mark.parametrize("payload", [1.0, 10.0, 100.0, 500.0, 1000.0])
    def test_exit_velocity_bounded(self, payload):
        result = md.run_simulation(payload_kg=payload, track_length_m=5000.0)
        assert 0.0 <= result["exit_velocity_m_s"] <= 20_000.0

    @pytest.mark.parametrize("length", [500.0, 1000.0, 2000.0, 5000.0, 10_000.0])
    def test_launch_time_reasonable(self, length):
        result = md.run_simulation(track_length_m=length)
        assert 0.0 < result["launch_time_s"] < 60.0

    @pytest.mark.parametrize("elevation", [10.0, 20.0, 30.0, 45.0, 60.0])
    def test_altitude_positive_for_elevated_track(self, elevation):
        result = md.run_simulation(elevation_deg=elevation, track_length_m=3000.0)
        assert result["exit_altitude_m"] > 0.0

    def test_drag_loss_positive(self):
        result = md.run_simulation(track_length_m=5000.0)
        assert result["drag_loss_mj"] >= 0.0

    def test_thermal_loss_non_negative(self):
        result = md.run_simulation()
        assert result["thermal_loss_mj"] >= 0.0

    def test_energy_exceeds_kinetic(self):
        result = md.run_simulation(track_length_m=5000.0)
        if result["total_energy_mj"] > 0 and result["kinetic_energy_mj"] > 0:
            assert result["total_energy_mj"] >= result["kinetic_energy_mj"]


class TestEdgeCases:
    def test_zero_payload(self):
        result = md.run_simulation(payload_kg=0.0)
        assert result["exit_velocity_m_s"] >= 0.0

    def test_very_heavy_payload(self):
        result = md.run_simulation(payload_kg=10_000.0)
        assert not result["failed"] or result["exit_velocity_m_s"] >= 0.0

    def test_single_stage(self):
        cfg = md.create_track(track_length_m=100.0, num_stages=1)
        state = md.create_launch()
        results = md.run_launch(cfg, state)
        assert len(results) >= 1

    def test_zero_elevation(self):
        result = md.run_simulation(elevation_deg=0.0)
        assert result["exit_altitude_m"] < 1.0

    def test_steep_elevation(self):
        result = md.run_simulation(elevation_deg=80.0, track_length_m=3000.0)
        assert result["exit_altitude_m"] > 0.0

    def test_very_short_track(self):
        result = md.run_simulation(track_length_m=10.0)
        assert result["exit_velocity_m_s"] >= 0.0
        assert not result["failed"]


class TestSmoke:
    def test_10_tick_no_crash(self):
        cfg = md.create_track()
        state = md.create_launch()
        for _ in range(10):
            md.tick(cfg, state)
            if state.launch_complete or state.launch_failed:
                break

    def test_full_launch_no_crash(self):
        md.run_launch()

    def test_simulation_no_crash(self):
        md.run_simulation()

    def test_can_reach_target_no_crash(self):
        md.can_reach_target(3550.0)

    def test_optimal_track_no_crash(self):
        md.optimal_track_for_target(3550.0, payload_kg=50.0)

    def test_multiple_launches_sequential(self):
        cfg = md.create_track()
        for _ in range(5):
            state = md.create_launch()
            md.run_launch(cfg, state)
            assert state.launch_complete or state.launch_failed

    def test_cool_and_relaunch(self):
        cfg = md.create_track()
        state1 = md.create_launch()
        md.run_launch(cfg, state1)
        md.cool_track(state1, 7200.0)
        state2 = md.create_launch()
        state2.stage_temps_k = list(state1.stage_temps_k)
        md.run_launch(cfg, state2)
        assert state2.launch_complete or state2.launch_failed

    def test_all_target_types(self):
        for target in ["lmo", "escape", "earth"]:
            result = md.run_simulation(target=target, track_length_m=5000.0)
            assert "exit_velocity_m_s" in result


class TestPowerRequirements:
    def test_power_in_megawatt_range(self):
        result = md.run_simulation(track_length_m=2000.0, payload_kg=100.0)
        assert result["average_power_mw"] > 0.01

    def test_energy_scales_with_payload(self):
        light = md.run_simulation(payload_kg=10.0, track_length_m=3000.0)
        heavy = md.run_simulation(payload_kg=500.0, track_length_m=3000.0)
        assert heavy["total_energy_mj"] > light["total_energy_mj"]

    def test_longer_track_spreads_power(self):
        short = md.run_simulation(track_length_m=1000.0, payload_kg=100.0)
        long = md.run_simulation(track_length_m=5000.0, payload_kg=100.0)
        assert not short["failed"]
        assert not long["failed"]
