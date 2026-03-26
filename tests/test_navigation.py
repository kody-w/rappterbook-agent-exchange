"""Tests for navigation.py -- Mars Surface Dead Reckoning & Celestial Fix.

96 tests covering:
  - Physical constants (Mars circumference, sol duration, speed of light)
  - Dead reckoning (position update, slip, heading, cardinal directions)
  - Gyro drift (accumulation, health degradation, sqrt-time model)
  - Odometry error growth (sqrt-distance model)
  - Celestial fixes (star, sun -- accuracy, dust storm blocking)
  - Radio ranging (limits, error model, out-of-range, line-of-sight)
  - Terrain-relative navigation (accuracy, all-weather)
  - Bayesian uncertainty fusion (shrinks, never grows, symmetric)
  - CEP computation (from 2D sigma)
  - Conservation laws (monotonicity, bounds, physical limits)
  - Multi-sol simulation (error growth, fix correction cycles)
  - Edge cases (zero distance, stationary, extreme headings)
  - 10-sol smoke test without crash
"""
from __future__ import annotations

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation import (
    # Constants
    MARS_RADIUS_M,
    MARS_CIRCUMFERENCE_M,
    MARS_SOL_S,
    SPEED_OF_LIGHT_M_S,
    DEG_PER_METRE,
    DEFAULT_SLIP_FRACTION,
    SAND_SLIP_FRACTION,
    SLOPE_SLIP_FRACTION,
    GYRO_DRIFT_DEG_PER_HOUR,
    GYRO_DRIFT_DEG_PER_SOL,
    ODOM_ERROR_COEFF,
    STAR_FIX_ACCURACY_M,
    SUN_FIX_ACCURACY_M,
    SUN_HEADING_ACCURACY_DEG,
    RADIO_CLOCK_ERROR_S,
    RADIO_RANGE_ERROR_M,
    RADIO_MAX_RANGE_M,
    TERRAIN_NAV_ACCURACY_M,
    CEP_FACTOR,
    # Functions
    normalize_heading,
    slip_fraction_for_terrain,
    dead_reckon,
    odometry_error_growth,
    gyro_drift_error,
    bayesian_update_sigma,
    cep_from_sigma,
    radio_range_error,
    max_radio_range_m,
    tick,
    create_navigator,
    # Classes
    NavState,
)


# -- Fixtures -----------------------------------------------------------------

@pytest.fixture
def fresh_nav():
    return create_navigator(x_m=0.0, y_m=0.0, heading_deg=0.0)


@pytest.fixture
def moving_nav():
    n = create_navigator(x_m=100.0, y_m=200.0, heading_deg=90.0)
    tick(n, wheel_distance_m=1000.0)
    return n


# =============================================================================
# S1  CONSTANTS -- PHYSICAL SANITY
# =============================================================================

class TestConstants:

    def test_mars_circumference_range(self):
        """Mars circumference should be ~21 300 km."""
        assert 21_000_000 < MARS_CIRCUMFERENCE_M < 22_000_000

    def test_mars_sol_seconds(self):
        """Sol is ~88 775 seconds."""
        assert 88_000 < MARS_SOL_S < 89_000

    def test_speed_of_light(self):
        assert abs(SPEED_OF_LIGHT_M_S - 299_792_458.0) < 1.0

    def test_radio_range_error_from_clock(self):
        """1 us clock error -> ~150 m range error."""
        expected = SPEED_OF_LIGHT_M_S * 1e-6 / 2.0
        assert abs(RADIO_RANGE_ERROR_M - expected) < 0.1

    def test_slip_ordering(self):
        """Sand > rock, slope > sand."""
        assert DEFAULT_SLIP_FRACTION < SAND_SLIP_FRACTION < SLOPE_SLIP_FRACTION

    def test_cep_factor(self):
        """CEP factor ~ 1.1774."""
        assert abs(CEP_FACTOR - 1.1774) < 0.001

    def test_gyro_drift_per_sol(self):
        """Gyro drift per sol should be positive and reasonable."""
        assert 0.1 < GYRO_DRIFT_DEG_PER_SOL < 5.0


# =============================================================================
# S2  NORMALIZE HEADING
# =============================================================================

class TestNormalizeHeading:

    def test_zero(self):
        assert normalize_heading(0.0) == 0.0

    def test_positive(self):
        assert normalize_heading(90.0) == 90.0

    def test_360_wraps(self):
        assert normalize_heading(360.0) == 0.0

    def test_negative(self):
        result = normalize_heading(-90.0)
        assert abs(result - 270.0) < 0.001

    def test_large_positive(self):
        result = normalize_heading(725.0)
        assert abs(result - 5.0) < 0.001

    def test_large_negative(self):
        result = normalize_heading(-450.0)
        assert abs(result - 270.0) < 0.001


# =============================================================================
# S3  SLIP FRACTION
# =============================================================================

class TestSlipFraction:

    def test_rock(self):
        assert slip_fraction_for_terrain("rock") == DEFAULT_SLIP_FRACTION

    def test_sand(self):
        assert slip_fraction_for_terrain("sand") == SAND_SLIP_FRACTION

    def test_slope(self):
        assert slip_fraction_for_terrain("slope") == SLOPE_SLIP_FRACTION

    def test_unknown_defaults_to_rock(self):
        assert slip_fraction_for_terrain("lava") == DEFAULT_SLIP_FRACTION

    def test_all_fractions_in_bounds(self):
        for t in ["rock", "sand", "slope"]:
            f = slip_fraction_for_terrain(t)
            assert 0.0 < f < 1.0


# =============================================================================
# S4  DEAD RECKONING
# =============================================================================

class TestDeadReckon:

    def test_zero_distance(self):
        x, y, d = dead_reckon(10.0, 20.0, 45.0, 0.0, 0.05)
        assert x == 10.0
        assert y == 20.0
        assert d == 0.0

    def test_negative_distance(self):
        x, y, d = dead_reckon(0.0, 0.0, 0.0, -100.0, 0.0)
        assert d == 0.0

    def test_north_heading(self):
        """Heading 0 = north -> y increases, x unchanged."""
        x, y, d = dead_reckon(0.0, 0.0, 0.0, 100.0, 0.0)
        assert abs(x) < 0.001
        assert abs(y - 100.0) < 0.001
        assert abs(d - 100.0) < 0.001

    def test_east_heading(self):
        """Heading 90 = east -> x increases, y unchanged."""
        x, y, d = dead_reckon(0.0, 0.0, 90.0, 100.0, 0.0)
        assert abs(x - 100.0) < 0.001
        assert abs(y) < 0.001

    def test_south_heading(self):
        """Heading 180 = south -> y decreases."""
        x, y, d = dead_reckon(0.0, 0.0, 180.0, 100.0, 0.0)
        assert abs(x) < 0.001
        assert abs(y - (-100.0)) < 0.001

    def test_west_heading(self):
        """Heading 270 = west -> x decreases."""
        x, y, d = dead_reckon(0.0, 0.0, 270.0, 100.0, 0.0)
        assert abs(x - (-100.0)) < 0.001
        assert abs(y) < 0.001

    def test_slip_reduces_distance(self):
        """5% slip -> effective = 95% of wheel distance."""
        _, _, d = dead_reckon(0.0, 0.0, 0.0, 100.0, 0.05)
        assert abs(d - 95.0) < 0.001

    def test_high_slip(self):
        """25% slip (slope)."""
        _, _, d = dead_reckon(0.0, 0.0, 0.0, 100.0, 0.25)
        assert abs(d - 75.0) < 0.001

    def test_slip_clamped_below_one(self):
        """Slip > 0.99 gets clamped."""
        _, _, d = dead_reckon(0.0, 0.0, 0.0, 100.0, 1.5)
        assert d > 0.0  # clamped to 0.99

    def test_preserves_origin(self):
        """Starting position adds to result."""
        x, y, _ = dead_reckon(500.0, 300.0, 0.0, 100.0, 0.0)
        assert abs(x - 500.0) < 0.001
        assert abs(y - 400.0) < 0.001


# =============================================================================
# S5  ODOMETRY ERROR GROWTH
# =============================================================================

class TestOdometryError:

    def test_zero_distance(self):
        assert odometry_error_growth(0.0) == 0.0

    def test_negative_distance(self):
        assert odometry_error_growth(-10.0) == 0.0

    def test_positive_distance(self):
        e = odometry_error_growth(100.0)
        expected = ODOM_ERROR_COEFF * math.sqrt(100.0)
        assert abs(e - expected) < 0.001

    def test_grows_sublinearly(self):
        """Error grows as sqrt -> doubling distance less than doubles error."""
        e1 = odometry_error_growth(100.0)
        e2 = odometry_error_growth(200.0)
        assert e2 > e1
        assert e2 < 2.0 * e1

    def test_1km_error(self):
        """1 km drive -> 2% * sqrt(1000) ~ 0.63 m error."""
        e = odometry_error_growth(1000.0)
        assert 0.5 < e < 1.0


# =============================================================================
# S6  GYRO DRIFT
# =============================================================================

class TestGyroDrift:

    def test_zero_hours(self):
        assert gyro_drift_error(0.0) == 0.0

    def test_negative_hours(self):
        assert gyro_drift_error(-5.0) == 0.0

    def test_one_hour(self):
        d = gyro_drift_error(1.0)
        assert abs(d - GYRO_DRIFT_DEG_PER_HOUR) < 0.001

    def test_one_sol(self):
        sol_h = MARS_SOL_S / 3600.0
        d = gyro_drift_error(sol_h)
        assert d > 0.0
        assert d < 1.0  # should be moderate

    def test_poor_health_increases_drift(self):
        d_good = gyro_drift_error(10.0, health=1.0)
        d_bad = gyro_drift_error(10.0, health=0.5)
        assert d_bad > d_good

    def test_grows_with_sqrt_time(self):
        d1 = gyro_drift_error(1.0)
        d4 = gyro_drift_error(4.0)
        assert abs(d4 / d1 - 2.0) < 0.01  # sqrt(4) = 2


# =============================================================================
# S7  BAYESIAN UPDATE
# =============================================================================

class TestBayesianUpdate:

    def test_shrinks_uncertainty(self):
        result = bayesian_update_sigma(100.0, 50.0)
        assert result < 100.0
        assert result < 50.0

    def test_zero_prior(self):
        """Zero prior -> returns measurement sigma."""
        assert bayesian_update_sigma(0.0, 50.0) == 50.0

    def test_zero_measurement(self):
        """Perfect measurement -> zero uncertainty."""
        assert bayesian_update_sigma(100.0, 0.0) == 0.0

    def test_equal_sigmas(self):
        """Equal sigmas -> result = sigma / sqrt(2)."""
        result = bayesian_update_sigma(100.0, 100.0)
        expected = 100.0 / math.sqrt(2.0)
        assert abs(result - expected) < 0.01

    def test_very_large_prior(self):
        """Very uncertain prior + good measurement ~ measurement."""
        result = bayesian_update_sigma(1e6, 10.0)
        assert abs(result - 10.0) < 0.1

    def test_symmetric(self):
        """Order of arguments should not matter."""
        r1 = bayesian_update_sigma(100.0, 50.0)
        r2 = bayesian_update_sigma(50.0, 100.0)
        assert abs(r1 - r2) < 0.001


# =============================================================================
# S8  CEP COMPUTATION
# =============================================================================

class TestCEP:

    def test_zero_sigma(self):
        assert cep_from_sigma(0.0, 0.0) == 0.0

    def test_symmetric_sigma(self):
        cep = cep_from_sigma(100.0, 100.0)
        assert abs(cep - CEP_FACTOR * 100.0) < 0.01

    def test_asymmetric_sigma(self):
        cep = cep_from_sigma(100.0, 200.0)
        expected = CEP_FACTOR * 150.0
        assert abs(cep - expected) < 0.01

    def test_always_positive(self):
        for sx in [0, 1, 10, 100, 1000]:
            for sy in [0, 1, 10, 100, 1000]:
                assert cep_from_sigma(float(sx), float(sy)) >= 0.0


# =============================================================================
# S9  RADIO RANGING
# =============================================================================

class TestRadioRange:

    def test_zero_distance(self):
        assert radio_range_error(0.0) == 0.0

    def test_negative_distance(self):
        assert radio_range_error(-10.0) == 0.0

    def test_in_range(self):
        e = radio_range_error(1000.0)
        assert e > RADIO_RANGE_ERROR_M  # base error + multipath
        assert e < RADIO_RANGE_ERROR_M * 2  # not huge

    def test_at_max_range(self):
        e = radio_range_error(RADIO_MAX_RANGE_M)
        assert e < float("inf")
        assert e > 0.0

    def test_beyond_max_range(self):
        e = radio_range_error(RADIO_MAX_RANGE_M + 1.0)
        assert e == float("inf")

    def test_error_increases_with_distance(self):
        e1 = radio_range_error(100.0)
        e2 = radio_range_error(4000.0)
        assert e2 > e1


class TestMaxRadioRange:

    def test_default_height(self):
        r = max_radio_range_m(1.8)
        assert 3000 < r < 4000  # ~3.5 km

    def test_elevated_antenna(self):
        r_low = max_radio_range_m(1.8)
        r_high = max_radio_range_m(10.0)
        assert r_high > r_low

    def test_zero_height(self):
        assert max_radio_range_m(0.0) == 0.0

    def test_negative_height(self):
        assert max_radio_range_m(-1.0) == 0.0


# =============================================================================
# S10  TICK -- STATIONARY
# =============================================================================

class TestTickStationary:

    def test_sol_increments(self, fresh_nav):
        r = tick(fresh_nav)
        assert r["sol"] == 1

    def test_no_movement_no_dead_reckoning(self, fresh_nav):
        r = tick(fresh_nav)
        assert "DEAD_RECKONING" not in r["events"]

    def test_position_unchanged(self, fresh_nav):
        r = tick(fresh_nav)
        assert r["x_m"] == 0.0
        assert r["y_m"] == 0.0

    def test_gyro_drift_accumulates(self, fresh_nav):
        r = tick(fresh_nav)
        assert r["heading_sigma_deg"] > 0.0

    def test_gyro_health_degrades(self, fresh_nav):
        tick(fresh_nav)
        assert fresh_nav.gyro_health < 1.0


# =============================================================================
# S11  TICK -- MOVEMENT
# =============================================================================

class TestTickMovement:

    def test_northward_drive(self, fresh_nav):
        r = tick(fresh_nav, wheel_distance_m=1000.0)
        assert r["y_m"] > 0.0
        assert "DEAD_RECKONING" in r["events"]

    def test_distance_accumulates(self, fresh_nav):
        tick(fresh_nav, wheel_distance_m=1000.0)
        tick(fresh_nav, wheel_distance_m=500.0)
        assert fresh_nav.total_distance_m > 1400.0  # slip reduces

    def test_uncertainty_grows_with_drive(self, fresh_nav):
        tick(fresh_nav, wheel_distance_m=5000.0)
        assert fresh_nav.sigma_x_m > 0.0
        assert fresh_nav.sigma_y_m > 0.0

    def test_heading_change(self, fresh_nav):
        tick(fresh_nav, heading_change_deg=90.0)
        assert abs(fresh_nav.heading_deg - 90.0) < 0.001

    def test_sand_terrain_more_slip(self):
        n_rock = create_navigator()
        n_sand = create_navigator()
        n_sand.terrain = "sand"
        tick(n_rock, wheel_distance_m=1000.0)
        tick(n_sand, wheel_distance_m=1000.0)
        assert n_sand.total_distance_m < n_rock.total_distance_m


# =============================================================================
# S12  TICK -- FIXES
# =============================================================================

class TestTickFixes:

    def test_star_fix_reduces_uncertainty(self):
        n = create_navigator()
        tick(n, wheel_distance_m=10000.0)
        sigma_before = n.sigma_x_m
        tick(n, star_fix=True)
        assert n.sigma_x_m < sigma_before

    def test_star_fix_event(self, fresh_nav):
        r = tick(fresh_nav, star_fix=True)
        assert "STAR_FIX" in r["events"]
        assert r["last_fix_type"] == "star"

    def test_sun_fix_reduces_uncertainty(self):
        n = create_navigator()
        tick(n, wheel_distance_m=10000.0)
        sigma_before = n.sigma_x_m
        tick(n, sun_fix=True)
        assert n.sigma_x_m < sigma_before

    def test_sun_fix_corrects_heading(self):
        n = create_navigator()
        for _ in range(10):
            tick(n)  # accumulate gyro drift
        heading_sigma_before = n.heading_sigma_deg
        tick(n, sun_fix=True)
        assert n.heading_sigma_deg < heading_sigma_before

    def test_radio_fix_in_range(self):
        n = create_navigator()
        tick(n, wheel_distance_m=5000.0)
        sigma_before = n.sigma_x_m
        tick(n, radio_fix=True, radio_distance_m=3000.0)
        assert n.sigma_x_m < sigma_before
        assert n.last_fix_type == "radio"

    def test_radio_fix_out_of_range(self):
        n = create_navigator()
        tick(n, wheel_distance_m=5000.0)
        r = tick(n, radio_fix=True, radio_distance_m=10000.0)
        assert "RADIO_OUT_OF_RANGE" in r["events"]

    def test_terrain_fix_best_accuracy(self):
        n = create_navigator()
        tick(n, wheel_distance_m=10000.0)
        sigma_before = n.sigma_x_m
        tick(n, terrain_fix=True)
        assert n.sigma_x_m < sigma_before
        assert n.last_fix_type == "terrain"

    def test_dust_storm_blocks_celestial(self, fresh_nav):
        r = tick(fresh_nav, star_fix=True, sun_fix=True, dust_storm=True)
        assert "DUST_BLOCKS_CELESTIAL" in r["events"]
        assert "STAR_FIX" not in r["events"]
        assert "SUN_FIX" not in r["events"]

    def test_terrain_fix_works_in_dust_storm(self):
        n = create_navigator()
        tick(n, wheel_distance_m=10000.0)
        sigma_before = n.sigma_x_m
        r = tick(n, terrain_fix=True, dust_storm=True)
        assert "TERRAIN_FIX" in r["events"]
        assert n.sigma_x_m < sigma_before

    def test_fix_resets_distance_since_fix(self):
        n = create_navigator()
        tick(n, wheel_distance_m=5000.0)
        assert n.distance_since_fix_m > 0
        tick(n, star_fix=True)
        assert n.distance_since_fix_m == 0.0


# =============================================================================
# S13  CONSERVATION LAWS / INVARIANTS
# =============================================================================

class TestInvariants:

    def test_uncertainty_never_negative(self, fresh_nav):
        for _ in range(50):
            tick(fresh_nav, wheel_distance_m=100.0, star_fix=True)
            assert fresh_nav.sigma_x_m >= 0.0
            assert fresh_nav.sigma_y_m >= 0.0
            assert fresh_nav.heading_sigma_deg >= 0.0

    def test_distance_monotonically_increasing(self, fresh_nav):
        prev = 0.0
        for _ in range(20):
            tick(fresh_nav, wheel_distance_m=500.0)
            assert fresh_nav.total_distance_m >= prev
            prev = fresh_nav.total_distance_m

    def test_heading_always_normalized(self, fresh_nav):
        for _ in range(100):
            tick(fresh_nav, heading_change_deg=37.3)
            assert 0.0 <= fresh_nav.heading_deg < 360.0
            assert 0.0 <= fresh_nav.true_heading_deg < 360.0

    def test_gyro_health_bounded(self, fresh_nav):
        for _ in range(5000):
            tick(fresh_nav)
        assert fresh_nav.gyro_health >= 0.5

    def test_fix_never_increases_uncertainty(self):
        """Core invariant: any fix can only shrink sigma."""
        n = create_navigator()
        tick(n, wheel_distance_m=10000.0)

        for fix_kwargs in [
            {"star_fix": True},
            {"sun_fix": True},
            {"radio_fix": True, "radio_distance_m": 2000.0},
            {"terrain_fix": True},
        ]:
            before_x = n.sigma_x_m
            before_y = n.sigma_y_m
            tick(n, **fix_kwargs)
            assert n.sigma_x_m <= before_x + 0.001  # small rounding tolerance
            assert n.sigma_y_m <= before_y + 0.001

    def test_cep_matches_sigma(self, fresh_nav):
        """CEP should always match sigma via the formula."""
        for dist in [0, 100, 1000, 5000]:
            tick(fresh_nav, wheel_distance_m=float(dist))
            expected_cep = cep_from_sigma(fresh_nav.sigma_x_m, fresh_nav.sigma_y_m)
            result = tick(fresh_nav)
            assert abs(result["cep_m"] - round(expected_cep, 3)) < 0.01

    def test_sols_count_correctly(self, fresh_nav):
        for i in range(10):
            r = tick(fresh_nav)
            assert r["sol"] == i + 1


# =============================================================================
# S14  MULTI-SOL SIMULATION
# =============================================================================

class TestMultiSol:

    def test_10_sol_smoke(self):
        """Smoke test: 10 sols of driving with periodic fixes -- no crash."""
        n = create_navigator()
        for sol in range(10):
            fix_star = (sol % 3 == 0)
            fix_radio = (sol % 5 == 0) and sol > 0
            r = tick(
                n,
                wheel_distance_m=2000.0,
                heading_change_deg=10.0,
                star_fix=fix_star,
                radio_fix=fix_radio,
                radio_distance_m=3000.0,
            )
            assert r["sol"] == sol + 1
            assert r["total_distance_m"] > 0
            assert r["cep_m"] >= 0.0

    def test_error_grows_without_fixes(self):
        """Without fixes, CEP should grow over time."""
        n = create_navigator()
        cep_values = []
        for _ in range(20):
            r = tick(n, wheel_distance_m=2000.0)
            cep_values.append(r["cep_m"])
        assert cep_values[-1] > cep_values[0]

    def test_fixes_keep_error_bounded(self):
        """With regular fixes, CEP stays bounded."""
        n = create_navigator()
        max_cep = 0.0
        for sol in range(50):
            star = (sol % 2 == 0)
            r = tick(n, wheel_distance_m=1000.0, star_fix=star)
            if r["cep_m"] > max_cep:
                max_cep = r["cep_m"]
        assert max_cep < 1000.0

    def test_terrain_types_affect_accuracy(self):
        """Different terrain types lead to different error accumulation."""
        results = {}
        for terrain in ["rock", "sand", "slope"]:
            n = create_navigator()
            n.terrain = terrain
            for _ in range(10):
                r = tick(n, wheel_distance_m=1000.0)
            results[terrain] = n.total_distance_m
        assert results["rock"] > results["sand"] > results["slope"]

    def test_dust_storm_degrades_navigation(self):
        """Dust storm blocks celestial fixes, so error grows faster."""
        n_clear = create_navigator()
        n_storm = create_navigator()
        for _ in range(10):
            tick(n_clear, wheel_distance_m=2000.0, star_fix=True)
            tick(n_storm, wheel_distance_m=2000.0, star_fix=True, dust_storm=True)
        assert n_storm.sigma_x_m > n_clear.sigma_x_m

    def test_full_mission_profile(self):
        """Simulate a realistic EVA: drive out, take fixes, return."""
        n = create_navigator()
        # Drive out (heading 45 deg, 5 sols)
        tick(n, heading_change_deg=45.0)
        for _ in range(4):
            tick(n, wheel_distance_m=3000.0, sun_fix=True)

        # Turn around (heading 225 deg = 180 more)
        tick(n, heading_change_deg=180.0)
        for _ in range(4):
            tick(n, wheel_distance_m=3000.0, star_fix=True)

        # Should be near origin (within nav uncertainty)
        dist_from_origin = math.sqrt(n.x_m ** 2 + n.y_m ** 2)
        assert dist_from_origin < 5000.0


# =============================================================================
# S15  CREATE NAVIGATOR
# =============================================================================

class TestCreateNavigator:

    def test_default_origin(self):
        n = create_navigator()
        assert n.x_m == 0.0
        assert n.y_m == 0.0
        assert n.heading_deg == 0.0

    def test_custom_position(self):
        n = create_navigator(x_m=100.0, y_m=200.0, heading_deg=45.0)
        assert n.x_m == 100.0
        assert n.true_x_m == 100.0
        assert n.heading_deg == 45.0
        assert n.true_heading_deg == 45.0

    def test_initial_zero_uncertainty(self):
        n = create_navigator()
        assert n.sigma_x_m == 0.0
        assert n.sigma_y_m == 0.0
        assert n.heading_sigma_deg == 0.0

    def test_initial_sol_zero(self):
        n = create_navigator()
        assert n.sol == 0
        assert n.total_distance_m == 0.0
        assert n.fixes_taken == 0
