"""launch_window.py -- Mars-to-Earth Launch Window & Ascent Weather Gate

The colony stores propellant and tracks mission readiness.  But *when*
can it actually launch?  This module answers that question by computing
Hohmann transfer windows, Mars-Earth synodic alignment, and surface
weather constraints that must all be satisfied simultaneously.

Physics
-------
* **Hohmann transfer orbit**: Minimum-energy trajectory between Mars
  (1.524 AU) and Earth (1.0 AU).  Transfer time ≈ 259 days.
  Delta-v from Mars surface (via low Mars orbit) ≈ 5.7 km/s.
* **Synodic period**: Earth-Mars synodic period ≈ 780 days (≈ 2.135 yr).
  Launch windows open every ~26 months for ~30-60 days.
* **Planetary alignment**: Optimal when Mars leads Earth by the transfer
  angle θ = π(1 - (1/2·((r1+r2)/(2·r2)))^1.5) ≈ 44.3° for Earth-Mars.
  Wait: we launch Mars→Earth so departure planet is Mars.
  θ_depart = π(1 - (T_transfer/(T_earth))·(1/2π)) — simplified:
  the phase angle at departure ≈ 75° (Earth ahead of Mars).
* **Weather gates**: Dust storms (tau > 2.0 blocks launch), wind speed
  > 30 m/s blocks pad ops, solar flare (SPE) events block EVA and launch.
* **Launch commit criteria** (LCC): All gates green for 2 consecutive
  sols before commit.  Scrub resets the counter.

References:
  Bate, Mueller & White — Fundamentals of Astrodynamics (1971)
  Mars DRA 5.0 — NASA/SP-2009-566
  Zubrin — The Case for Mars (1996)

One tick = one sol.  Angles in radians, distances in AU, delta-v in m/s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# -- Orbital constants --------------------------------------------------------

AU_M = 1.496e11                     # 1 AU in metres
MU_SUN = 1.327e20                   # Sun gravitational parameter (m³/s²)

MARS_SEMI_MAJOR_AU = 1.524          # Mars orbital semi-major axis (AU)
EARTH_SEMI_MAJOR_AU = 1.000         # Earth orbital semi-major axis (AU)

MARS_ORBITAL_PERIOD_DAYS = 687.0    # Mars sidereal period (days)
EARTH_ORBITAL_PERIOD_DAYS = 365.25  # Earth sidereal period (days)

MARS_SOL_S = 88775.0               # one Mars sol (seconds)
EARTH_DAY_S = 86400.0              # one Earth day (seconds)

MARS_SURFACE_GRAVITY = 3.72        # m/s²
MARS_RADIUS_M = 3389.5e3          # Mars mean radius (m)
LMO_ALTITUDE_M = 250e3            # Low Mars Orbit altitude (m)

# -- Hohmann transfer constants -----------------------------------------------

def hohmann_transfer_semi_major_au() -> float:
    """Semi-major axis of Hohmann transfer ellipse (AU)."""
    return (MARS_SEMI_MAJOR_AU + EARTH_SEMI_MAJOR_AU) / 2.0


def hohmann_transfer_time_days() -> float:
    """Transfer time for Hohmann orbit, Mars → Earth (days).

    T = π * sqrt(a³ / μ), converted to days.
    """
    a_m = hohmann_transfer_semi_major_au() * AU_M
    t_seconds = math.pi * math.sqrt(a_m ** 3 / MU_SUN)
    return t_seconds / EARTH_DAY_S


def hohmann_phase_angle_rad() -> float:
    """Required Earth-ahead-of-Mars phase angle at departure (rad).

    For Mars→Earth transfer, Earth must be ahead by:
      θ = π - (2π · T_transfer) / (2 · T_earth)
      (simplified: Earth travels during transfer, needs to arrive at
       the right point when spacecraft arrives)
    """
    t_transfer = hohmann_transfer_time_days()
    earth_angular_rate = 2 * math.pi / EARTH_ORBITAL_PERIOD_DAYS
    angle_earth_travels = earth_angular_rate * t_transfer
    return math.pi - angle_earth_travels


SYNODIC_PERIOD_DAYS = 1.0 / abs(
    1.0 / EARTH_ORBITAL_PERIOD_DAYS - 1.0 / MARS_ORBITAL_PERIOD_DAYS
)

# -- Delta-v calculations ----------------------------------------------------

def circular_velocity_m_s(radius_m: float, mu: float = MU_SUN) -> float:
    """Circular orbital velocity at given radius (m/s)."""
    if radius_m <= 0:
        return 0.0
    return math.sqrt(mu / radius_m)


def hohmann_departure_dv_m_s() -> float:
    """Delta-v for trans-Earth injection from Mars orbit (m/s).

    Departure burn at Mars orbit radius to enter transfer ellipse.
    """
    r_mars = MARS_SEMI_MAJOR_AU * AU_M
    a_transfer = hohmann_transfer_semi_major_au() * AU_M
    v_mars = circular_velocity_m_s(r_mars)
    v_departure = math.sqrt(MU_SUN * (2.0 / r_mars - 1.0 / a_transfer))
    return abs(v_departure - v_mars)


def lmo_velocity_m_s() -> float:
    """Velocity in Low Mars Orbit (m/s)."""
    r = MARS_RADIUS_M + LMO_ALTITUDE_M
    mu_mars = MARS_SURFACE_GRAVITY * MARS_RADIUS_M ** 2
    return math.sqrt(mu_mars / r)


def surface_to_lmo_dv_m_s() -> float:
    """Approximate delta-v from Mars surface to LMO (m/s).

    Includes gravity losses (~1.4 km/s above orbital velocity).
    """
    v_orb = lmo_velocity_m_s()
    gravity_loss = 1400.0  # typical gravity loss estimate (m/s)
    return v_orb + gravity_loss


def total_ascent_dv_m_s() -> float:
    """Total delta-v: Mars surface → LMO → trans-Earth injection (m/s)."""
    return surface_to_lmo_dv_m_s() + hohmann_departure_dv_m_s()


# -- Phase angle tracking ----------------------------------------------------

def earth_mean_anomaly_rad(sol: int, epoch_angle_rad: float = 0.0) -> float:
    """Earth's mean anomaly at given sol (radians from epoch)."""
    days = sol * MARS_SOL_S / EARTH_DAY_S
    return (epoch_angle_rad + 2 * math.pi * days
            / EARTH_ORBITAL_PERIOD_DAYS) % (2 * math.pi)


def mars_mean_anomaly_rad(sol: int, epoch_angle_rad: float = 0.0) -> float:
    """Mars's mean anomaly at given sol (radians from epoch)."""
    days = sol * MARS_SOL_S / EARTH_DAY_S
    return (epoch_angle_rad + 2 * math.pi * days
            / MARS_ORBITAL_PERIOD_DAYS) % (2 * math.pi)


def phase_angle_rad(sol: int,
                     earth_epoch_rad: float = 0.0,
                     mars_epoch_rad: float = 0.0) -> float:
    """Phase angle: Earth longitude minus Mars longitude (rad).

    Positive = Earth ahead of Mars.  Range [-π, π].
    """
    e = earth_mean_anomaly_rad(sol, earth_epoch_rad)
    m = mars_mean_anomaly_rad(sol, mars_epoch_rad)
    diff = (e - m) % (2 * math.pi)
    if diff > math.pi:
        diff -= 2 * math.pi
    return diff


def phase_angle_error_rad(sol: int,
                           earth_epoch_rad: float = 0.0,
                           mars_epoch_rad: float = 0.0) -> float:
    """Absolute error from optimal phase angle (rad)."""
    current = phase_angle_rad(sol, earth_epoch_rad, mars_epoch_rad)
    optimal = hohmann_phase_angle_rad()
    error = abs(current - optimal)
    if error > math.pi:
        error = 2 * math.pi - error
    return error


# -- Weather gates ------------------------------------------------------------

DUST_TAU_LIMIT = 2.0               # optical depth limit (blocks launch)
WIND_SPEED_LIMIT_M_S = 30.0        # surface wind limit (m/s)
SPE_DOSE_LIMIT_MSV_HR = 0.5        # solar particle event dose limit

WINDOW_HALF_WIDTH_RAD = math.radians(15.0)  # ±15° from optimal
LCC_GREEN_SOLS = 2                  # consecutive green sols to commit


@dataclass
class WeatherReport:
    """Surface weather for one sol."""
    dust_tau: float = 0.3           # atmospheric optical depth
    wind_speed_m_s: float = 8.0     # surface wind (m/s)
    spe_dose_msv_hr: float = 0.0    # SPE radiation dose rate


def is_dust_clear(weather: WeatherReport) -> bool:
    """True if dust levels are below launch limit."""
    return weather.dust_tau < DUST_TAU_LIMIT


def is_wind_safe(weather: WeatherReport) -> bool:
    """True if wind speed is below pad operations limit."""
    return weather.wind_speed_m_s < WIND_SPEED_LIMIT_M_S


def is_spe_clear(weather: WeatherReport) -> bool:
    """True if no solar particle event threatens crew."""
    return weather.spe_dose_msv_hr < SPE_DOSE_LIMIT_MSV_HR


def all_weather_green(weather: WeatherReport) -> bool:
    """True if all weather gates are green."""
    return (is_dust_clear(weather)
            and is_wind_safe(weather)
            and is_spe_clear(weather))


# -- Launch window state ------------------------------------------------------

@dataclass
class LaunchWindowState:
    """Tracks planetary alignment and launch commit status."""
    sol: int = 0

    # Epoch orbital positions (radians)
    earth_epoch_rad: float = 0.0
    mars_epoch_rad: float = 0.0

    # Window tracking
    window_open: bool = False
    consecutive_green_sols: int = 0
    launch_committed: bool = False
    launch_sol: int = -1

    # Cumulative stats
    total_windows_seen: int = 0
    total_scrubs: int = 0
    total_green_sols: int = 0

    # Current computed values (updated each tick)
    current_phase_angle_rad: float = 0.0
    current_phase_error_rad: float = 0.0

    alert: str = "no_window"


@dataclass
class WindowTickResult:
    """Result of one sol of launch window evaluation."""
    phase_angle_rad: float = 0.0
    phase_error_rad: float = 0.0
    window_open: bool = False
    dust_clear: bool = True
    wind_safe: bool = True
    spe_clear: bool = True
    all_green: bool = True
    consecutive_green: int = 0
    launch_committed: bool = False
    days_to_next_window: float = 0.0
    hohmann_transfer_days: float = 0.0
    total_dv_m_s: float = 0.0
    alert: str = "no_window"


# -- Next window estimation ---------------------------------------------------

def estimate_days_to_next_window(sol: int,
                                  earth_epoch_rad: float = 0.0,
                                  mars_epoch_rad: float = 0.0,
                                  max_search_sols: int = 1000
                                  ) -> float:
    """Estimate sols until next launch window opens.

    Scans forward by 1-sol increments until phase error < window width.
    Returns sols to wait (0 if currently in window).
    """
    optimal = hohmann_phase_angle_rad()
    for ds in range(max_search_sols):
        future_sol = sol + ds
        err = phase_angle_error_rad(
            future_sol, earth_epoch_rad, mars_epoch_rad)
        if err < WINDOW_HALF_WIDTH_RAD:
            return float(ds)
    return float(max_search_sols)


# -- Main tick ----------------------------------------------------------------

def tick_window(state: LaunchWindowState,
                 weather: WeatherReport | None = None,
                 ) -> tuple:
    """Advance launch window tracking by one sol.

    Returns (LaunchWindowState, WindowTickResult).
    """
    if weather is None:
        weather = WeatherReport()

    result = WindowTickResult()
    state.sol += 1

    # Compute orbital geometry
    pa = phase_angle_rad(state.sol, state.earth_epoch_rad,
                          state.mars_epoch_rad)
    pe = phase_angle_error_rad(state.sol, state.earth_epoch_rad,
                                state.mars_epoch_rad)
    state.current_phase_angle_rad = pa
    state.current_phase_error_rad = pe

    result.phase_angle_rad = pa
    result.phase_error_rad = pe
    result.hohmann_transfer_days = hohmann_transfer_time_days()
    result.total_dv_m_s = total_ascent_dv_m_s()

    # Is window open?
    was_open = state.window_open
    state.window_open = pe < WINDOW_HALF_WIDTH_RAD
    result.window_open = state.window_open

    if state.window_open and not was_open:
        state.total_windows_seen += 1

    # Weather gates
    result.dust_clear = is_dust_clear(weather)
    result.wind_safe = is_wind_safe(weather)
    result.spe_clear = is_spe_clear(weather)
    result.all_green = all_weather_green(weather)

    # Launch commit logic
    if state.launch_committed:
        # Already committed — keep state
        result.launch_committed = True
        result.consecutive_green = state.consecutive_green_sols
        result.alert = "committed"
        state.alert = "committed"
        return state, result

    if state.window_open and result.all_green:
        state.consecutive_green_sols += 1
        state.total_green_sols += 1
    elif state.window_open and not result.all_green:
        if state.consecutive_green_sols > 0:
            state.total_scrubs += 1
        state.consecutive_green_sols = 0
    else:
        state.consecutive_green_sols = 0

    result.consecutive_green = state.consecutive_green_sols

    # Check commit
    if state.consecutive_green_sols >= LCC_GREEN_SOLS:
        state.launch_committed = True
        state.launch_sol = state.sol
        result.launch_committed = True
        result.alert = "committed"
        state.alert = "committed"
    elif state.window_open:
        result.alert = "window_open"
        state.alert = "window_open"
    else:
        # Estimate next window
        result.days_to_next_window = estimate_days_to_next_window(
            state.sol, state.earth_epoch_rad, state.mars_epoch_rad)
        result.alert = "no_window"
        state.alert = "no_window"

    return state, result


# -- Factory ------------------------------------------------------------------

def create_launch_tracker(earth_epoch_rad: float = 0.0,
                           mars_epoch_rad: float = 0.0
                           ) -> LaunchWindowState:
    """Create a launch window tracker.

    Epoch angles define the initial orbital positions at sol 0.
    To start near a window, set earth_epoch_rad ≈ mars_epoch_rad +
    hohmann_phase_angle_rad().
    """
    return LaunchWindowState(
        earth_epoch_rad=earth_epoch_rad,
        mars_epoch_rad=mars_epoch_rad,
    )


def create_tracker_near_window() -> LaunchWindowState:
    """Create a tracker with planets positioned near a launch window.

    Useful for testing — window should open within ~20 sols.
    """
    optimal = hohmann_phase_angle_rad()
    # Earth ahead of Mars by slightly more than optimal → window approaching
    return create_launch_tracker(
        earth_epoch_rad=optimal + math.radians(10),
        mars_epoch_rad=0.0,
    )
