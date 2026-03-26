"""navigation.py -- Mars Surface Dead Reckoning & Celestial Fix Navigation.

Mars has no GPS constellation.  Every rover traverse, EVA excursion,
and search-and-rescue mission depends on dead reckoning corrected by
periodic celestial fixes and base-station radio ranging.  Without this
module the colony is navigating blind.

Physics modelled
----------------
* **Dead reckoning** -- integrate wheel odometry (or step count) over
  heading.  Position = prev_position + distance * [sin h, cos h].
  Error grows as sqrt(distance) due to wheel slip and heading drift.

* **Heading gyro drift** -- MEMS gyroscopes drift 0.01-0.1 deg/hour.
  Over a sol (24.66 h) that is 0.25-2.5 deg of uncompensated error.
  Drift is modelled as a random walk (sigma grows with sqrt(time)).

* **Wheel slip** -- Martian regolith is loose.  Effective distance
  is actual_wheel_rotation * (1 - slip_fraction).  Slip depends on
  terrain: 2-5% on flat rock, 10-20% on loose sand/dune.

* **Celestial fix (star tracker)** -- At night (or twilight), a star
  tracker can take a fix against Phobos, Deimos, or stellar catalog.
  Accuracy ~0.01 deg -> ~590 m at Mars equator (circumference ~21344 km).
  Fix requires clear sky (fails during dust storms).

* **Sun fix** -- By day, solar elevation + time-of-sol gives latitude.
  Solar azimuth + time gives heading correction.  Accuracy ~0.1 deg in
  position, ~0.5 deg in heading.

* **Radio ranging** -- Base station transmits timing signal.  Round-trip
  time * c / 2 = distance.  With two base stations -> triangulation.
  Accuracy limited by clock sync: 1 us error -> 150 m range error.
  Effective range limited to line-of-sight (~5 km on flat Mars due
  to horizon at 1.8 m height, farther from elevated base antenna).

* **Terrain-relative navigation** -- Compare camera images to orbital
  maps.  Accuracy ~10 m but computationally expensive and requires
  prior mapping.  Works in any weather.

* **Position uncertainty** -- Modelled as 2D Gaussian (sigma_x, sigma_y).
  Dead reckoning grows it; fixes shrink it via Bayesian update.
  Reported as CEP (circular error probable) = median radius of the
  error ellipse ~ 1.1774 * sigma for circular case.

Conservation laws / invariants
------------------------------
- Position uncertainty never negative.
- Dead reckoning error grows monotonically between fixes.
- Fix can only shrink uncertainty (never increase it).
- Heading in [0, 360) degrees.
- Slip fraction in [0, 1).
- Distance travelled monotonically increasing.
- CEP = 0 only at initialisation with perfect fix.

Reference:
  - Mars equatorial circumference: 21 344 km
  - Mars radius: 3 389.5 km
  - Mars sol: 24 h 39 m 35 s ~ 88 775 s
  - Phobos orbital period: 7 h 39 m (visible ~2x per sol)
  - MEMS gyro drift: 0.01-0.1 deg/h (tactical grade)
  - MER Spirit/Opportunity: ~5% wheel slip on flat, 20%+ on slopes
  - Curiosity visual odometry accuracy: ~2% of distance
  - Speed of light: 299 792 458 m/s

One tick = one sol.  Distances in metres, headings in degrees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


# -- Physical constants -------------------------------------------------------

MARS_RADIUS_M = 3_389_500.0
MARS_CIRCUMFERENCE_M = 2.0 * math.pi * MARS_RADIUS_M          # ~21 344 km
MARS_SOL_S = 88_775.0                                          # seconds per sol
SPEED_OF_LIGHT_M_S = 299_792_458.0
DEG_PER_METRE = 360.0 / MARS_CIRCUMFERENCE_M                   # ~1.687e-5 deg/m

# Dead reckoning
DEFAULT_SLIP_FRACTION = 0.05          # 5% on flat rock
SAND_SLIP_FRACTION = 0.18            # loose sand/dunes
SLOPE_SLIP_FRACTION = 0.25           # steep slopes

# Gyro drift (tactical-grade MEMS)
GYRO_DRIFT_DEG_PER_HOUR = 0.05       # nominal drift rate
GYRO_DRIFT_DEG_PER_SOL = GYRO_DRIFT_DEG_PER_HOUR * (MARS_SOL_S / 3600.0)

# Odometry error growth: sigma_position grows as this factor * sqrt(distance_m)
ODOM_ERROR_COEFF = 0.02              # 2% of distance as 1-sigma error (Curiosity-class)

# Celestial fix accuracies (1-sigma, in metres)
STAR_FIX_ACCURACY_M = 590.0          # 0.01 deg at equator
SUN_FIX_ACCURACY_M = 5900.0          # 0.1 deg at equator
SUN_HEADING_ACCURACY_DEG = 0.5       # heading correction from sun

# Radio ranging
RADIO_CLOCK_ERROR_S = 1.0e-6         # 1 us clock sync error
RADIO_RANGE_ERROR_M = SPEED_OF_LIGHT_M_S * RADIO_CLOCK_ERROR_S / 2.0  # ~150 m
RADIO_MAX_RANGE_M = 5_000.0          # line-of-sight on flat Mars (~5 km)

# Terrain-relative navigation
TERRAIN_NAV_ACCURACY_M = 10.0        # from orbital map matching

# CEP conversion: for circular Gaussian, CEP ~ 1.1774 * sigma
CEP_FACTOR = 1.1774


# -- Data structures ----------------------------------------------------------

@dataclass
class NavState:
    """Navigation state for one mobile unit (rover or crew)."""

    # Current estimated position (metres from base station origin)
    x_m: float = 0.0
    y_m: float = 0.0

    # True position (for simulation -- not available to the navigator)
    true_x_m: float = 0.0
    true_y_m: float = 0.0

    # Heading (degrees, 0 = north, clockwise)
    heading_deg: float = 0.0
    true_heading_deg: float = 0.0

    # Uncertainty (1-sigma in metres)
    sigma_x_m: float = 0.0
    sigma_y_m: float = 0.0
    heading_sigma_deg: float = 0.0

    # Odometer
    total_distance_m: float = 0.0
    distance_since_fix_m: float = 0.0

    # Sol counter
    sol: int = 0

    # Terrain type: "rock", "sand", "slope"
    terrain: str = "rock"

    # Fix history
    fixes_taken: int = 0
    last_fix_type: str = "none"

    # Gyro health (1.0 = perfect, degrades over time)
    gyro_health: float = 1.0

    # Events this tick
    events: list = field(default_factory=list)


# -- Pure functions -----------------------------------------------------------

def normalize_heading(deg: float) -> float:
    """Normalize heading to [0, 360)."""
    return deg % 360.0


def slip_fraction_for_terrain(terrain: str) -> float:
    """Return expected wheel slip for terrain type."""
    if terrain == "sand":
        return SAND_SLIP_FRACTION
    if terrain == "slope":
        return SLOPE_SLIP_FRACTION
    return DEFAULT_SLIP_FRACTION


def dead_reckon(
    x_m: float,
    y_m: float,
    heading_deg: float,
    wheel_distance_m: float,
    slip: float,
) -> tuple:
    """Advance position by dead reckoning.

    Returns (new_x, new_y, effective_distance).
    Heading 0 = north, 90 = east.  sin(h) for easting, cos(h) for northing.
    """
    if wheel_distance_m <= 0.0:
        return x_m, y_m, 0.0
    slip = max(0.0, min(slip, 0.99))
    effective = wheel_distance_m * (1.0 - slip)
    rad = math.radians(heading_deg)
    new_x = x_m + effective * math.sin(rad)
    new_y = y_m + effective * math.cos(rad)
    return new_x, new_y, effective


def odometry_error_growth(distance_m: float) -> float:
    """Position uncertainty growth (1-sigma) from dead reckoning.

    Error grows as ODOM_ERROR_COEFF * sqrt(distance) (random-walk model).
    """
    if distance_m <= 0.0:
        return 0.0
    return ODOM_ERROR_COEFF * math.sqrt(distance_m)


def gyro_drift_error(hours: float, health: float = 1.0) -> float:
    """Heading uncertainty growth (degrees) from gyro drift.

    Drift modelled as random walk: sigma = drift_rate * sqrt(hours) / health.
    """
    if hours <= 0.0:
        return 0.0
    rate = GYRO_DRIFT_DEG_PER_HOUR / max(0.1, health)
    return rate * math.sqrt(hours)


def bayesian_update_sigma(prior_sigma: float, measurement_sigma: float) -> float:
    """Bayesian fusion of prior uncertainty with measurement.

    For Gaussian: 1/sigma_post^2 = 1/sigma_prior^2 + 1/sigma_meas^2
    Result is always smaller than both inputs.
    """
    if prior_sigma <= 0.0:
        return measurement_sigma
    if measurement_sigma <= 0.0:
        return 0.0
    inv_var = 1.0 / (prior_sigma ** 2) + 1.0 / (measurement_sigma ** 2)
    return 1.0 / math.sqrt(inv_var)


def cep_from_sigma(sigma_x: float, sigma_y: float) -> float:
    """Circular Error Probable from 2D Gaussian sigmas.

    CEP ~ 1.1774 * mean(sigma_x, sigma_y) for near-circular case.
    """
    mean_sigma = (abs(sigma_x) + abs(sigma_y)) / 2.0
    return CEP_FACTOR * mean_sigma


def radio_range_error(distance_m: float) -> float:
    """Position error from single radio range measurement.

    Error is fixed by clock sync quality, plus distance-proportional
    multipath.  Returns 1-sigma error in metres.
    """
    if distance_m <= 0.0:
        return 0.0
    if distance_m > RADIO_MAX_RANGE_M:
        return float("inf")
    return RADIO_RANGE_ERROR_M + 0.001 * distance_m


def max_radio_range_m(antenna_height_m: float = 1.8) -> float:
    """Line-of-sight distance to horizon on Mars.

    d = sqrt(2 * R * h) where R = Mars radius, h = antenna height.
    """
    if antenna_height_m <= 0.0:
        return 0.0
    return math.sqrt(2.0 * MARS_RADIUS_M * antenna_height_m)


# -- Tick function ------------------------------------------------------------

def tick(
    state: NavState,
    wheel_distance_m: float = 0.0,
    heading_change_deg: float = 0.0,
    star_fix: bool = False,
    sun_fix: bool = False,
    radio_fix: bool = False,
    terrain_fix: bool = False,
    radio_distance_m: float = 0.0,
    dust_storm: bool = False,
) -> Dict[str, Any]:
    """Advance navigation state by one sol.

    Parameters
    ----------
    state : NavState
        Mutable navigation state.
    wheel_distance_m : float
        Wheel odometry distance driven this sol (metres).
    heading_change_deg : float
        Commanded heading change (degrees, positive = clockwise).
    star_fix : bool
        Attempt celestial star fix (needs clear sky).
    sun_fix : bool
        Attempt solar fix (daytime, needs clear sky).
    radio_fix : bool
        Attempt radio ranging fix from base station.
    terrain_fix : bool
        Attempt terrain-relative visual navigation.
    radio_distance_m : float
        True distance from base station (for radio fix simulation).
    dust_storm : bool
        Is a dust storm active?  Blocks star and sun fixes.

    Returns
    -------
    Dict with navigation state summary and events.
    """
    state.sol += 1
    state.events = []
    wheel_distance_m = max(0.0, wheel_distance_m)

    # -- Heading update ---------------------------------------------------
    state.true_heading_deg = normalize_heading(
        state.true_heading_deg + heading_change_deg
    )
    state.heading_deg = normalize_heading(
        state.heading_deg + heading_change_deg
    )

    # Gyro drift over one sol
    sol_hours = MARS_SOL_S / 3600.0
    drift = gyro_drift_error(sol_hours, state.gyro_health)
    state.heading_sigma_deg += drift

    # Gyro degrades slowly (dust, thermal cycling)
    state.gyro_health = max(0.5, state.gyro_health - 0.0005)

    # -- Dead reckoning ---------------------------------------------------
    slip = slip_fraction_for_terrain(state.terrain)

    # True position (perfect knowledge for simulation)
    state.true_x_m, state.true_y_m, effective_true = dead_reckon(
        state.true_x_m, state.true_y_m,
        state.true_heading_deg, wheel_distance_m, slip,
    )

    # Navigator's estimate (uses estimated heading, slightly different slip)
    nav_slip = slip * 0.95  # navigator underestimates slip slightly
    state.x_m, state.y_m, effective_nav = dead_reckon(
        state.x_m, state.y_m,
        state.heading_deg, wheel_distance_m, nav_slip,
    )

    state.total_distance_m += effective_true
    state.distance_since_fix_m += effective_true

    # Uncertainty growth from odometry
    if effective_true > 0:
        odom_err = odometry_error_growth(effective_true)
        state.sigma_x_m = math.sqrt(state.sigma_x_m ** 2 + odom_err ** 2)
        state.sigma_y_m = math.sqrt(state.sigma_y_m ** 2 + odom_err ** 2)
        state.events.append("DEAD_RECKONING")

    # -- Fixes (shrink uncertainty) ----------------------------------------
    if dust_storm:
        star_fix = False
        sun_fix = False
        state.events.append("DUST_BLOCKS_CELESTIAL")

    if star_fix:
        state.sigma_x_m = bayesian_update_sigma(state.sigma_x_m, STAR_FIX_ACCURACY_M)
        state.sigma_y_m = bayesian_update_sigma(state.sigma_y_m, STAR_FIX_ACCURACY_M)
        state.fixes_taken += 1
        state.last_fix_type = "star"
        state.distance_since_fix_m = 0.0
        state.events.append("STAR_FIX")

    if sun_fix:
        state.sigma_x_m = bayesian_update_sigma(state.sigma_x_m, SUN_FIX_ACCURACY_M)
        state.sigma_y_m = bayesian_update_sigma(state.sigma_y_m, SUN_FIX_ACCURACY_M)
        state.heading_sigma_deg = bayesian_update_sigma(
            state.heading_sigma_deg, SUN_HEADING_ACCURACY_DEG
        )
        state.fixes_taken += 1
        state.last_fix_type = "sun"
        state.distance_since_fix_m = 0.0
        state.events.append("SUN_FIX")

    if radio_fix and radio_distance_m > 0:
        if radio_distance_m <= RADIO_MAX_RANGE_M:
            r_err = radio_range_error(radio_distance_m)
            state.sigma_x_m = bayesian_update_sigma(state.sigma_x_m, r_err)
            state.sigma_y_m = bayesian_update_sigma(state.sigma_y_m, r_err)
            state.fixes_taken += 1
            state.last_fix_type = "radio"
            state.distance_since_fix_m = 0.0
            state.events.append("RADIO_FIX")
        else:
            state.events.append("RADIO_OUT_OF_RANGE")

    if terrain_fix:
        state.sigma_x_m = bayesian_update_sigma(state.sigma_x_m, TERRAIN_NAV_ACCURACY_M)
        state.sigma_y_m = bayesian_update_sigma(state.sigma_y_m, TERRAIN_NAV_ACCURACY_M)
        state.fixes_taken += 1
        state.last_fix_type = "terrain"
        state.distance_since_fix_m = 0.0
        state.events.append("TERRAIN_FIX")

    # -- Build result -----------------------------------------------------
    actual_error = math.sqrt(
        (state.x_m - state.true_x_m) ** 2 +
        (state.y_m - state.true_y_m) ** 2
    )

    return {
        "sol": state.sol,
        "x_m": round(state.x_m, 3),
        "y_m": round(state.y_m, 3),
        "heading_deg": round(state.heading_deg, 3),
        "sigma_x_m": round(state.sigma_x_m, 3),
        "sigma_y_m": round(state.sigma_y_m, 3),
        "cep_m": round(cep_from_sigma(state.sigma_x_m, state.sigma_y_m), 3),
        "heading_sigma_deg": round(state.heading_sigma_deg, 4),
        "total_distance_m": round(state.total_distance_m, 3),
        "distance_since_fix_m": round(state.distance_since_fix_m, 3),
        "fixes_taken": state.fixes_taken,
        "last_fix_type": state.last_fix_type,
        "terrain": state.terrain,
        "gyro_health": round(state.gyro_health, 4),
        "actual_error_m": round(actual_error, 3),
        "events": list(state.events),
    }


def create_navigator(
    x_m: float = 0.0,
    y_m: float = 0.0,
    heading_deg: float = 0.0,
) -> NavState:
    """Create a new navigator at a known position with perfect fix."""
    return NavState(
        x_m=x_m, y_m=y_m,
        true_x_m=x_m, true_y_m=y_m,
        heading_deg=heading_deg,
        true_heading_deg=heading_deg,
    )
