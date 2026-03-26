"""meteorology.py -- Mars Weather Station & Forecast System.

Models a surface weather station that measures atmospheric conditions,
builds a rolling forecast, and predicts dust storms. The colony's eyes
on the sky. Every EVA decision, solar power estimate, and rover route
depends on what this station reports.

Each tick = 1 sol of weather observations and forecasting.

Physics modelled
----------------
* **Temperature** -- diurnal range from Viking/MSL data: daytime highs
  near -20°C (equatorial summer) to -80°C (polar winter), overnight lows
  -73°C to -120°C. Seasonal sinusoidal with orbital eccentricity.
* **Pressure** -- annual cycle driven by CO₂ sublimation/deposition at
  poles. Measured by Phoenix/MSL: 600-1000 Pa range. Viking saw ±25%
  seasonal swing.
* **Wind** -- Martian winds: typical 2-7 m/s surface, gusts to 30 m/s
  in dust storms. Dust devils common (>1 per sol in warm season).
* **Dust opacity (tau)** -- optical depth from solar extinction. Clear
  sky tau ≈ 0.3, regional storm tau ≈ 1-3, global storm tau ≈ 5-9.
  Tau directly affects solar panel output.
* **Forecast model** -- simple persistence + seasonal trend + storm
  probability. 3-sol lookahead. Accuracy degrades with forecast horizon.
* **Storm prediction** -- probability model based on season (Ls),
  current tau trend, and historical frequency. Mars dust storm season
  peaks at Ls 200-330 (southern spring/summer).

Reference data:
  - MSL/Curiosity REMS: continuous weather since sol 1 (2012)
  - Viking 1/2 meteorology: 1976-1982
  - Mars year Ls: 0=vernal equinox, 90=summer solstice,
    180=autumnal equinox, 270=winter solstice
  - Global dust storms: ~1 per 3 Mars years (every ~2000 sols)
  - Regional storms: ~10 per Mars year
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# -- Physical constants -------------------------------------------------------

MARS_YEAR_SOLS = 668            # one Mars year in sols
SOL_HOURS = 24.66               # one sol in hours

# Temperature (°C) -- equatorial MSL-like site
TEMP_ANNUAL_MEAN_C = -55.0      # annual mean surface temperature
TEMP_ANNUAL_AMP_C = 20.0        # seasonal amplitude (summer-winter)
TEMP_DIURNAL_AMP_C = 40.0       # day-night swing at equator
TEMP_NOISE_STD_C = 3.0          # random daily variation

# Pressure (Pa)
PRESSURE_ANNUAL_MEAN_PA = 730.0  # annual mean (Gale Crater ~730 Pa)
PRESSURE_ANNUAL_AMP_PA = 120.0   # seasonal CO₂ cycle amplitude
PRESSURE_NOISE_STD_PA = 15.0     # daily weather variation

# Wind (m/s)
WIND_MEAN_MS = 4.5               # typical surface wind
WIND_STD_MS = 2.0                # standard deviation
WIND_GUST_MULTIPLIER = 2.5       # max gust = mean * this
WIND_STORM_BOOST_MS = 15.0       # added wind during dust storms

# Dust opacity (tau -- dimensionless optical depth)
TAU_CLEAR = 0.3                  # clear sky baseline
TAU_NOISE_STD = 0.05             # daily variation in clear conditions
TAU_REGIONAL_STORM = 2.5         # typical regional dust storm tau
TAU_GLOBAL_STORM = 6.0           # typical global dust storm tau
TAU_DECAY_RATE = 0.15            # tau drops 15% per sol after storm peak

# Storm probabilities (per sol)
REGIONAL_STORM_DAILY_PROB = 0.015   # ~10 per year (10/668)
GLOBAL_STORM_DAILY_PROB = 0.0005    # ~1 per 3 years (1/2000)
STORM_SEASON_LS_START = 200.0       # dust storm season start (Ls degrees)
STORM_SEASON_LS_END = 330.0         # dust storm season end
STORM_SEASON_MULTIPLIER = 3.0       # probability boost during season

# Forecast
FORECAST_HORIZON_SOLS = 3           # how many sols ahead we forecast
FORECAST_DECAY_FACTOR = 0.6         # accuracy multiplier per sol ahead

# Sensor degradation
SENSOR_DUST_RATE = 0.0005           # sensor accuracy loss per sol from dust
SENSOR_CLEAN_RESTORE = 0.95         # cleaning restores to this fraction


# -- Data structures ----------------------------------------------------------

@dataclass
class WeatherReading:
    """A single sol's weather measurements."""
    sol: int = 0
    ls: float = 0.0                # solar longitude (0-360°)
    temp_high_c: float = -20.0
    temp_low_c: float = -80.0
    pressure_pa: float = 730.0
    wind_mean_ms: float = 4.5
    wind_gust_ms: float = 11.0
    tau: float = 0.3               # dust opacity
    dust_storm_active: bool = False
    storm_type: str = "none"       # "none", "regional", "global"


@dataclass
class Forecast:
    """Multi-sol weather forecast."""
    issued_sol: int = 0
    horizon_sols: int = FORECAST_HORIZON_SOLS
    temp_highs_c: list[float] = field(default_factory=list)
    temp_lows_c: list[float] = field(default_factory=list)
    pressures_pa: list[float] = field(default_factory=list)
    wind_means_ms: list[float] = field(default_factory=list)
    taus: list[float] = field(default_factory=list)
    storm_probabilities: list[float] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)


@dataclass
class StationState:
    """Complete state of the weather station."""
    sol: int = 0
    sensor_health: float = 1.0     # 0-1, degrades with dust
    readings_history: list[WeatherReading] = field(default_factory=list)
    history_max: int = 30          # keep last N sols of readings

    # Storm tracking
    active_storm: bool = False
    storm_type: str = "none"
    storm_sols_remaining: int = 0
    current_tau: float = TAU_CLEAR

    # Cumulative stats
    total_storms_observed: int = 0
    total_regional_storms: int = 0
    total_global_storms: int = 0
    max_tau_observed: float = TAU_CLEAR
    max_wind_observed: float = 0.0
    min_temp_observed: float = 0.0

    operational: bool = True

    def __post_init__(self) -> None:
        self.sensor_health = max(0.0, min(1.0, self.sensor_health))
        self.current_tau = max(0.0, self.current_tau)
        self.storm_sols_remaining = max(0, self.storm_sols_remaining)


# -- Pure physics functions ---------------------------------------------------

def sol_to_ls(sol: int) -> float:
    """Convert sol number to solar longitude Ls (0-360°).

    Simplified model: linear mapping over one Mars year.
    Real Ls is non-linear due to orbital eccentricity, but this
    captures the seasonal pattern for storm probability.
    """
    return (sol % MARS_YEAR_SOLS) / MARS_YEAR_SOLS * 360.0


def seasonal_temperature(ls: float) -> tuple[float, float]:
    """Expected temperature high/low for a given Ls.

    Returns (temp_high_c, temp_low_c).
    Uses sinusoidal seasonal model centered on Ls=90 (summer solstice).
    """
    # Seasonal component: warmest at Ls ~270 (southern summer = global dust season)
    # Simplification: warmest at Ls=90 for northern hemisphere site
    seasonal = TEMP_ANNUAL_AMP_C * math.sin(math.radians(ls))
    mean = TEMP_ANNUAL_MEAN_C + seasonal
    high = mean + TEMP_DIURNAL_AMP_C / 2
    low = mean - TEMP_DIURNAL_AMP_C / 2
    return (round(high, 2), round(low, 2))


def seasonal_pressure(ls: float) -> float:
    """Expected surface pressure for a given Ls.

    CO₂ cycle: pressure minimum at Ls~148 (CO₂ frozen at south pole),
    maximum at Ls~250 (CO₂ sublimated). Modelled as sinusoidal.
    """
    # Phase shifted so minimum at ~Ls 148, maximum at ~Ls 250
    phase = math.radians(ls - 250.0)
    pressure = PRESSURE_ANNUAL_MEAN_PA + PRESSURE_ANNUAL_AMP_PA * math.cos(phase)
    return round(max(0.0, pressure), 2)


def storm_probability(ls: float, current_tau: float) -> float:
    """Probability of a new dust storm starting on this sol.

    Higher during dust storm season (Ls 200-330).
    Higher when tau is already elevated (cascading storms).
    """
    base_regional = REGIONAL_STORM_DAILY_PROB
    base_global = GLOBAL_STORM_DAILY_PROB

    # Season multiplier
    in_season = False
    if STORM_SEASON_LS_START <= ls <= STORM_SEASON_LS_END:
        in_season = True
    # Handle wrap-around (not needed for 200-330 but future-proof)
    if STORM_SEASON_LS_START > STORM_SEASON_LS_END:
        in_season = ls >= STORM_SEASON_LS_START or ls <= STORM_SEASON_LS_END

    season_mult = STORM_SEASON_MULTIPLIER if in_season else 1.0

    # Elevated tau increases probability (positive feedback)
    tau_mult = 1.0 + max(0.0, (current_tau - TAU_CLEAR) / TAU_CLEAR) * 0.5

    p_regional = min(0.3, base_regional * season_mult * tau_mult)
    p_global = min(0.01, base_global * season_mult * tau_mult)

    # Combined: probability of ANY storm starting
    p_any = 1.0 - (1.0 - p_regional) * (1.0 - p_global)
    return round(min(1.0, max(0.0, p_any)), 6)


def tau_evolution(
    current_tau: float,
    storm_active: bool,
    storm_type: str,
) -> float:
    """Evolve dust opacity for one sol.

    During storms, tau rises toward storm-type peak.
    After storms, tau decays exponentially toward clear sky.
    """
    if storm_active:
        target = TAU_REGIONAL_STORM if storm_type == "regional" else TAU_GLOBAL_STORM
        # Rapid rise: move 40% toward target per sol
        new_tau = current_tau + 0.4 * (target - current_tau)
    else:
        # Exponential decay toward clear
        new_tau = TAU_CLEAR + (current_tau - TAU_CLEAR) * (1.0 - TAU_DECAY_RATE)

    return round(max(TAU_CLEAR * 0.8, new_tau), 4)


def solar_efficiency_from_tau(tau: float) -> float:
    """Solar panel efficiency modifier from dust opacity.

    Beer-Lambert law: transmission = exp(-tau).
    This directly scales solar array output.
    """
    return round(max(0.0, min(1.0, math.exp(-max(0.0, tau)))), 4)


def wind_speed(
    ls: float,
    storm_active: bool,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Generate wind mean and gust for a sol.

    Returns (wind_mean_ms, wind_gust_ms).
    """
    r = rng or random.Random()

    # Base wind with seasonal variation (windier in storm season)
    seasonal_boost = 0.0
    if STORM_SEASON_LS_START <= ls <= STORM_SEASON_LS_END:
        seasonal_boost = 2.0

    mean = max(0.5, r.gauss(WIND_MEAN_MS + seasonal_boost, WIND_STD_MS))
    if storm_active:
        mean += WIND_STORM_BOOST_MS

    gust = mean * WIND_GUST_MULTIPLIER
    return (round(mean, 2), round(gust, 2))


# -- Forecast engine ----------------------------------------------------------

def generate_forecast(
    state: StationState,
    rng: random.Random | None = None,
) -> Forecast:
    """Generate a multi-sol weather forecast from current state.

    Uses persistence (current conditions) + seasonal model + storm
    probability. Confidence degrades with forecast horizon.
    """
    r = rng or random.Random()
    forecast = Forecast(issued_sol=state.sol, horizon_sols=FORECAST_HORIZON_SOLS)

    for i in range(FORECAST_HORIZON_SOLS):
        future_sol = state.sol + i + 1
        ls = sol_to_ls(future_sol)

        # Temperature forecast: seasonal + noise scaled by distance
        noise_scale = 1.0 + i * 0.5  # more uncertainty further out
        base_high, base_low = seasonal_temperature(ls)
        forecast.temp_highs_c.append(round(base_high + r.gauss(0, TEMP_NOISE_STD_C * noise_scale), 2))
        forecast.temp_lows_c.append(round(base_low + r.gauss(0, TEMP_NOISE_STD_C * noise_scale), 2))

        # Pressure forecast
        base_p = seasonal_pressure(ls)
        forecast.pressures_pa.append(round(base_p + r.gauss(0, PRESSURE_NOISE_STD_PA * noise_scale), 2))

        # Tau forecast: decay or maintain
        projected_tau = tau_evolution(
            state.current_tau if i == 0 else forecast.taus[-1],
            state.active_storm and state.storm_sols_remaining > i,
            state.storm_type,
        )
        forecast.taus.append(projected_tau)

        # Wind forecast
        storm_future = state.active_storm and state.storm_sols_remaining > i
        w_mean, w_gust = wind_speed(ls, storm_future, r)
        forecast.wind_means_ms.append(w_mean)

        # Storm probability
        forecast.storm_probabilities.append(storm_probability(ls, projected_tau))

        # Confidence decays with horizon
        confidence = max(0.1, FORECAST_DECAY_FACTOR ** (i + 1)) * state.sensor_health
        forecast.confidence.append(round(confidence, 4))

    return forecast


# -- Tick function ------------------------------------------------------------

def tick_station(
    state: StationState,
    rng: random.Random | None = None,
    maintenance: bool = False,
) -> tuple[WeatherReading, Forecast]:
    """Advance the weather station by one sol.

    Args:
        state: current station state (mutated in place)
        rng: optional random number generator for reproducibility
        maintenance: if True, clean sensors this sol

    Returns:
        (reading, forecast) — this sol's observation and future forecast.
    """
    r = rng or random.Random()
    state.sol += 1
    ls = sol_to_ls(state.sol)

    # Maintenance: clean sensors
    if maintenance:
        state.sensor_health = min(1.0, max(state.sensor_health, SENSOR_CLEAN_RESTORE))

    # Sensor degradation
    state.sensor_health = max(0.0, state.sensor_health - SENSOR_DUST_RATE)

    # Storm lifecycle
    if state.active_storm:
        state.storm_sols_remaining -= 1
        if state.storm_sols_remaining <= 0:
            state.active_storm = False
            state.storm_type = "none"
            state.storm_sols_remaining = 0

    # Check for new storm (only if none active)
    if not state.active_storm:
        p = storm_probability(ls, state.current_tau)
        roll = r.random()
        if roll < GLOBAL_STORM_DAILY_PROB * (STORM_SEASON_MULTIPLIER if STORM_SEASON_LS_START <= ls <= STORM_SEASON_LS_END else 1.0):
            state.active_storm = True
            state.storm_type = "global"
            state.storm_sols_remaining = r.randint(30, 120)
            state.total_storms_observed += 1
            state.total_global_storms += 1
        elif roll < p:
            state.active_storm = True
            state.storm_type = "regional"
            state.storm_sols_remaining = r.randint(3, 15)
            state.total_storms_observed += 1
            state.total_regional_storms += 1

    # Evolve tau
    state.current_tau = tau_evolution(state.current_tau, state.active_storm, state.storm_type)
    state.max_tau_observed = max(state.max_tau_observed, state.current_tau)

    # Temperature
    base_high, base_low = seasonal_temperature(ls)
    noise = r.gauss(0, TEMP_NOISE_STD_C)
    sensor_noise = (1.0 - state.sensor_health) * r.gauss(0, 2.0)
    temp_high = round(base_high + noise + sensor_noise, 2)
    temp_low = round(base_low + noise - abs(r.gauss(0, 2.0)) + sensor_noise, 2)
    # Ensure high > low
    if temp_high <= temp_low:
        temp_high = temp_low + 1.0
    state.min_temp_observed = min(state.min_temp_observed, temp_low)

    # Pressure
    base_p = seasonal_pressure(ls)
    pressure = round(max(0.0, base_p + r.gauss(0, PRESSURE_NOISE_STD_PA) + sensor_noise * 5), 2)

    # Wind
    w_mean, w_gust = wind_speed(ls, state.active_storm, r)
    state.max_wind_observed = max(state.max_wind_observed, w_gust)

    # Build reading
    reading = WeatherReading(
        sol=state.sol,
        ls=round(ls, 2),
        temp_high_c=temp_high,
        temp_low_c=temp_low,
        pressure_pa=pressure,
        wind_mean_ms=w_mean,
        wind_gust_ms=w_gust,
        tau=state.current_tau,
        dust_storm_active=state.active_storm,
        storm_type=state.storm_type,
    )

    # Store in history (ring buffer)
    state.readings_history.append(reading)
    if len(state.readings_history) > state.history_max:
        state.readings_history = state.readings_history[-state.history_max:]

    # Generate forecast
    forecast = generate_forecast(state, r)

    return (reading, forecast)


def create_station() -> StationState:
    """Create a fresh weather station with factory-new sensors."""
    return StationState()
