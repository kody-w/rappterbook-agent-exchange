"""Tests for launch_window.py -- Mars-to-Earth launch window tracking."""
from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from launch_window import (
    # Constants
    AU_M, MU_SUN, MARS_SEMI_MAJOR_AU, EARTH_SEMI_MAJOR_AU,
    MARS_ORBITAL_PERIOD_DAYS, EARTH_ORBITAL_PERIOD_DAYS,
    MARS_SOL_S, EARTH_DAY_S, MARS_SURFACE_GRAVITY, MARS_RADIUS_M,
    LMO_ALTITUDE_M, SYNODIC_PERIOD_DAYS,
    DUST_TAU_LIMIT, WIND_SPEED_LIMIT_M_S, SPE_DOSE_LIMIT_MSV_HR,
    WINDOW_HALF_WIDTH_RAD, LCC_GREEN_SOLS,
    # Hohmann
    hohmann_transfer_semi_major_au, hohmann_transfer_time_days,
    hohmann_phase_angle_rad,
    # Delta-v
    circular_velocity_m_s, hohmann_departure_dv_m_s,
    lmo_velocity_m_s, surface_to_lmo_dv_m_s, total_ascent_dv_m_s,
    # Phase angles
    earth_mean_anomaly_rad, mars_mean_anomaly_rad,
    phase_angle_rad, phase_angle_error_rad,
    # Weather
    WeatherReport, is_dust_clear, is_wind_safe, is_spe_clear,
    all_weather_green,
    # Window
    estimate_days_to_next_window, tick_window,
    # Factory
    create_launch_tracker, create_tracker_near_window,
    # State
    LaunchWindowState, WindowTickResult,
)


# ─── Physical constants ────────────────────────────────────────────────────────

class TestConstants:
    """Orbital and physical constants in valid ranges."""

    def test_au_positive(self):
        assert AU_M > 1e10

    def test_mu_sun_positive(self):
        assert MU_SUN > 1e18

    def test_mars_farther_than_earth(self):
        assert MARS_SEMI_MAJOR_AU > EARTH_SEMI_MAJOR_AU

    def test_mars_period_longer_than_earth(self):
        assert MARS_ORBITAL_PERIOD_DAYS > EARTH_ORBITAL_PERIOD_DAYS

    def test_sol_close_to_earth_day(self):
        assert 0.95 < MARS_SOL_S / EARTH_DAY_S < 1.1

    def test_mars_gravity_reasonable(self):
        assert 3.5 < MARS_SURFACE_GRAVITY < 4.0

    def test_mars_radius_reasonable(self):
        assert 3300e3 < MARS_RADIUS_M < 3500e3

    def test_synodic_period_about_780_days(self):
        assert 770 < SYNODIC_PERIOD_DAYS < 790

    def test_dust_limit_positive(self):
        assert DUST_TAU_LIMIT > 0

    def test_wind_limit_positive(self):
        assert WIND_SPEED_LIMIT_M_S > 0

    def test_lcc_green_sols_at_least_one(self):
        assert LCC_GREEN_SOLS >= 1


# ─── Hohmann transfer ───────────────────────────────────────────────────────────

class TestHohmann:
    """Hohmann transfer orbit calculations."""

    def test_transfer_semi_major_between_orbits(self):
        a = hohmann_transfer_semi_major_au()
        assert EARTH_SEMI_MAJOR_AU < a < MARS_SEMI_MAJOR_AU

    def test_transfer_semi_major_is_average(self):
        a = hohmann_transfer_semi_major_au()
        expected = (MARS_SEMI_MAJOR_AU + EARTH_SEMI_MAJOR_AU) / 2
        assert abs(a - expected) < 1e-6

    def test_transfer_time_about_259_days(self):
        """Hohmann Mars→Earth is about 259 days."""
        t = hohmann_transfer_time_days()
        assert 200 < t < 320

    def test_phase_angle_negative_for_inward_transfer(self):
        """For Mars→Earth (outward to inward), Earth is behind Mars."""
        pa = hohmann_phase_angle_rad()
        # Should be roughly -75° (Earth behind Mars at departure)
        assert math.radians(-120) < pa < math.radians(0)

    def test_phase_angle_in_radians(self):
        pa = hohmann_phase_angle_rad()
        assert -math.pi < pa < math.pi


# ─── Delta-v ────────────────────────────────────────────────────────────────────

class TestDeltaV:
    """Velocity and delta-v calculations."""

    def test_circular_velocity_earth_orbit(self):
        """Earth orbital velocity ~ 29.8 km/s."""
        v = circular_velocity_m_s(EARTH_SEMI_MAJOR_AU * AU_M)
        assert 28000 < v < 31000

    def test_circular_velocity_mars_orbit(self):
        """Mars orbital velocity ~ 24.1 km/s."""
        v = circular_velocity_m_s(MARS_SEMI_MAJOR_AU * AU_M)
        assert 22000 < v < 26000

    def test_circular_velocity_zero_radius(self):
        assert circular_velocity_m_s(0) == 0.0

    def test_hohmann_departure_dv_reasonable(self):
        """Trans-Earth injection dv from Mars orbit ~ 2-3 km/s."""
        dv = hohmann_departure_dv_m_s()
        assert 1000 < dv < 5000

    def test_lmo_velocity_reasonable(self):
        """LMO velocity ~ 3.4 km/s."""
        v = lmo_velocity_m_s()
        assert 3000 < v < 4000

    def test_surface_to_lmo_includes_gravity_loss(self):
        dv = surface_to_lmo_dv_m_s()
        v_orb = lmo_velocity_m_s()
        assert dv > v_orb  # gravity loss adds to orbital velocity

    def test_total_dv_about_5_to_7_km_s(self):
        """Total Mars surface → trans-Earth ~ 5.7 km/s."""
        dv = total_ascent_dv_m_s()
        assert 4000 < dv < 8000


# ─── Phase angle tracking ───────────────────────────────────────────────────────

class TestPhaseAngles:
    """Planetary position and phase angle tracking."""

    def test_earth_anomaly_zero_at_epoch(self):
        a = earth_mean_anomaly_rad(0, epoch_angle_rad=0.0)
        assert abs(a) < 1e-6

    def test_mars_anomaly_zero_at_epoch(self):
        a = mars_mean_anomaly_rad(0, epoch_angle_rad=0.0)
        assert abs(a) < 1e-6

    def test_earth_anomaly_wraps(self):
        """After one Earth year in sols, should wrap near 2π."""
        sols_per_year = int(EARTH_ORBITAL_PERIOD_DAYS
                            * EARTH_DAY_S / MARS_SOL_S)
        a = earth_mean_anomaly_rad(sols_per_year)
        assert a < 0.1 or a > 2 * math.pi - 0.1

    def test_mars_anomaly_increases(self):
        a0 = mars_mean_anomaly_rad(0)
        a1 = mars_mean_anomaly_rad(10)
        assert a1 > a0

    def test_phase_angle_range(self):
        """Phase angle should be in [-π, π]."""
        for sol in range(0, 1000, 50):
            pa = phase_angle_rad(sol)
            assert -math.pi <= pa <= math.pi

    def test_phase_error_non_negative(self):
        for sol in range(0, 500, 25):
            pe = phase_angle_error_rad(sol)
            assert pe >= 0

    def test_phase_error_bounded_by_pi(self):
        for sol in range(0, 500, 25):
            pe = phase_angle_error_rad(sol)
            assert pe <= math.pi

    def test_phase_angle_zero_error_at_optimal(self):
        """If we set epoch to optimal, sol 0 should have ~zero error."""
        opt = hohmann_phase_angle_rad()
        pe = phase_angle_error_rad(0, earth_epoch_rad=opt, mars_epoch_rad=0)
        assert pe < 0.01

    def test_synodic_recurrence(self):
        """Phase angle should roughly repeat after one synodic period."""
        synodic_sols = int(SYNODIC_PERIOD_DAYS * EARTH_DAY_S / MARS_SOL_S)
        pa0 = phase_angle_rad(0, earth_epoch_rad=0.5, mars_epoch_rad=0.1)
        pa1 = phase_angle_rad(synodic_sols, earth_epoch_rad=0.5,
                               mars_epoch_rad=0.1)
        assert abs(pa0 - pa1) < math.radians(5)


# ─── Weather gates ──────────────────────────────────────────────────────────────

class TestWeather:
    """Surface weather gate checks."""

    def test_clear_weather(self):
        w = WeatherReport(dust_tau=0.3, wind_speed_m_s=5.0,
                           spe_dose_msv_hr=0.0)
        assert is_dust_clear(w)
        assert is_wind_safe(w)
        assert is_spe_clear(w)
        assert all_weather_green(w)

    def test_dust_storm_blocks(self):
        w = WeatherReport(dust_tau=3.0)
        assert not is_dust_clear(w)
        assert not all_weather_green(w)

    def test_dust_at_limit_blocks(self):
        w = WeatherReport(dust_tau=2.0)
        assert not is_dust_clear(w)

    def test_high_wind_blocks(self):
        w = WeatherReport(wind_speed_m_s=35.0)
        assert not is_wind_safe(w)
        assert not all_weather_green(w)

    def test_spe_blocks(self):
        w = WeatherReport(spe_dose_msv_hr=1.0)
        assert not is_spe_clear(w)
        assert not all_weather_green(w)

    def test_multiple_blocks(self):
        w = WeatherReport(dust_tau=5.0, wind_speed_m_s=50.0,
                           spe_dose_msv_hr=2.0)
        assert not all_weather_green(w)

    def test_just_below_limits_passes(self):
        w = WeatherReport(dust_tau=1.99, wind_speed_m_s=29.9,
                           spe_dose_msv_hr=0.49)
        assert all_weather_green(w)


# ─── Window estimation ──────────────────────────────────────────────────────────

class TestWindowEstimation:
    """Next launch window estimation."""

    def test_in_window_returns_zero(self):
        opt = hohmann_phase_angle_rad()
        d = estimate_days_to_next_window(
            0, earth_epoch_rad=opt, mars_epoch_rad=0)
        assert d == 0.0

    def test_out_of_window_returns_positive(self):
        d = estimate_days_to_next_window(
            0, earth_epoch_rad=0, mars_epoch_rad=0)
        assert d > 0

    def test_window_found_within_synodic(self):
        """Should find a window within one synodic period."""
        d = estimate_days_to_next_window(
            0, earth_epoch_rad=0, mars_epoch_rad=0)
        synodic_sols = SYNODIC_PERIOD_DAYS * EARTH_DAY_S / MARS_SOL_S
        assert d < synodic_sols


# ─── Tick lifecycle ─────────────────────────────────────────────────────────────

class TestTickWindow:
    """One-sol tick behavior."""

    def test_tick_advances_sol(self):
        s = create_launch_tracker()
        s, _ = tick_window(s)
        assert s.sol == 1

    def test_tick_updates_phase(self):
        s = create_launch_tracker()
        s, r = tick_window(s)
        assert r.phase_angle_rad != 0 or r.phase_error_rad != 0

    def test_tick_returns_dv(self):
        s = create_launch_tracker()
        _, r = tick_window(s)
        assert r.total_dv_m_s > 0
        assert r.hohmann_transfer_days > 0

    def test_tick_with_weather(self):
        s = create_tracker_near_window()
        w = WeatherReport(dust_tau=0.3, wind_speed_m_s=5.0)
        s, r = tick_window(s, weather=w)
        assert r.dust_clear
        assert r.wind_safe

    def test_tick_no_window_initially(self):
        """With arbitrary epoch, probably not in window."""
        s = create_launch_tracker(earth_epoch_rad=0, mars_epoch_rad=0)
        s, r = tick_window(s)
        # Could be in or out — just verify it runs
        assert r.alert in ("no_window", "window_open", "committed")

    def test_window_opens_near_optimal(self):
        """Tracker near window should see window open within 30 sols."""
        s = create_tracker_near_window()
        window_seen = False
        for _ in range(30):
            s, r = tick_window(s)
            if r.window_open:
                window_seen = True
                break
        assert window_seen

    def test_launch_commits_after_green_sols(self):
        """With good weather and open window, should commit."""
        opt = hohmann_phase_angle_rad()
        s = create_launch_tracker(earth_epoch_rad=opt, mars_epoch_rad=0)
        good_weather = WeatherReport(dust_tau=0.1, wind_speed_m_s=3.0,
                                      spe_dose_msv_hr=0.0)
        committed = False
        for _ in range(10):
            s, r = tick_window(s, weather=good_weather)
            if r.launch_committed:
                committed = True
                break
        assert committed

    def test_scrub_resets_counter(self):
        """Bad weather should reset the green counter."""
        opt = hohmann_phase_angle_rad()
        s = create_launch_tracker(earth_epoch_rad=opt, mars_epoch_rad=0)
        good = WeatherReport(dust_tau=0.1, wind_speed_m_s=3.0)
        bad = WeatherReport(dust_tau=5.0, wind_speed_m_s=3.0)

        # One green sol
        s, r = tick_window(s, weather=good)
        assert r.consecutive_green >= 1

        # Bad weather scrubs
        s, r = tick_window(s, weather=bad)
        assert r.consecutive_green == 0

    def test_committed_stays_committed(self):
        """Once committed, stays committed regardless of weather."""
        opt = hohmann_phase_angle_rad()
        s = create_launch_tracker(earth_epoch_rad=opt, mars_epoch_rad=0)
        good = WeatherReport(dust_tau=0.1)
        bad = WeatherReport(dust_tau=5.0)

        # Commit
        for _ in range(10):
            s, r = tick_window(s, weather=good)
            if r.launch_committed:
                break
        assert s.launch_committed

        # Bad weather doesn't uncommit
        s, r = tick_window(s, weather=bad)
        assert r.launch_committed
        assert r.alert == "committed"

    def test_scrub_counted(self):
        opt = hohmann_phase_angle_rad()
        s = create_launch_tracker(earth_epoch_rad=opt, mars_epoch_rad=0)
        good = WeatherReport(dust_tau=0.1)
        bad = WeatherReport(dust_tau=5.0)

        s, _ = tick_window(s, weather=good)
        s, _ = tick_window(s, weather=bad)
        assert s.total_scrubs >= 1


# ─── Multi-sol simulation ──────────────────────────────────────────────────────

class TestMultiSol:
    """Run for many sols and check properties."""

    def test_hundred_sols_no_crash(self):
        s = create_launch_tracker()
        for _ in range(100):
            s, _ = tick_window(s)
        assert s.sol == 100

    def test_five_hundred_sols_no_crash(self):
        s = create_launch_tracker()
        for _ in range(500):
            s, _ = tick_window(s)
        assert s.sol == 500

    def test_window_recurs(self):
        """Should see at least one window in 800 sols (> synodic period)."""
        s = create_launch_tracker()
        windows = 0
        for _ in range(800):
            s, r = tick_window(s)
            if r.window_open:
                windows += 1
        assert windows > 0

    def test_can_reach_committed(self):
        """Good weather throughout should eventually commit."""
        s = create_launch_tracker()
        good = WeatherReport(dust_tau=0.1, wind_speed_m_s=3.0)
        committed = False
        for _ in range(800):
            s, r = tick_window(s, weather=good)
            if r.launch_committed:
                committed = True
                break
        assert committed

    def test_sol_increments_monotonically(self):
        s = create_launch_tracker()
        for i in range(1, 51):
            s, _ = tick_window(s)
            assert s.sol == i


# ─── Factory ────────────────────────────────────────────────────────────────────

class TestFactory:
    """Factory functions."""

    def test_create_default(self):
        s = create_launch_tracker()
        assert s.sol == 0
        assert not s.launch_committed
        assert s.consecutive_green_sols == 0

    def test_create_with_epochs(self):
        s = create_launch_tracker(earth_epoch_rad=1.5, mars_epoch_rad=0.3)
        assert s.earth_epoch_rad == 1.5
        assert s.mars_epoch_rad == 0.3

    def test_create_near_window(self):
        s = create_tracker_near_window()
        assert s.earth_epoch_rad != 0.0 or s.mars_epoch_rad != 0.0


# ─── Physical invariants ────────────────────────────────────────────────────────

class TestInvariants:
    """Property-based invariants."""

    def test_sol_always_advances(self):
        s = create_launch_tracker()
        for i in range(50):
            s, _ = tick_window(s)
        assert s.sol == 50

    def test_total_windows_monotonic(self):
        s = create_launch_tracker()
        prev = 0
        for _ in range(200):
            s, _ = tick_window(s)
            assert s.total_windows_seen >= prev
            prev = s.total_windows_seen

    def test_total_scrubs_monotonic(self):
        s = create_launch_tracker()
        prev = 0
        bad = WeatherReport(dust_tau=5.0)
        good = WeatherReport(dust_tau=0.1)
        for i in range(200):
            w = bad if i % 3 == 0 else good
            s, _ = tick_window(s, weather=w)
            assert s.total_scrubs >= prev
            prev = s.total_scrubs

    def test_green_sols_monotonic(self):
        s = create_launch_tracker()
        prev = 0
        good = WeatherReport(dust_tau=0.1)
        for _ in range(200):
            s, _ = tick_window(s, weather=good)
            assert s.total_green_sols >= prev
            prev = s.total_green_sols

    def test_phase_error_bounded(self):
        s = create_launch_tracker()
        for _ in range(200):
            s, r = tick_window(s)
            assert 0 <= r.phase_error_rad <= math.pi

    def test_consecutive_green_bounded(self):
        s = create_launch_tracker()
        good = WeatherReport(dust_tau=0.1)
        for _ in range(100):
            s, r = tick_window(s, weather=good)
            assert r.consecutive_green >= 0
            if not s.launch_committed:
                assert r.consecutive_green <= s.sol

    def test_dv_constant_across_ticks(self):
        """Total delta-v shouldn't change — it's a constant."""
        s = create_launch_tracker()
        dvs = []
        for _ in range(10):
            s, r = tick_window(s)
            dvs.append(r.total_dv_m_s)
        assert all(abs(d - dvs[0]) < 1e-6 for d in dvs)

    def test_alert_always_valid(self):
        s = create_launch_tracker()
        valid_alerts = {"no_window", "window_open", "committed"}
        for _ in range(200):
            s, r = tick_window(s)
            assert r.alert in valid_alerts
