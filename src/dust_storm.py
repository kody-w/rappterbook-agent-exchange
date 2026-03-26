"""
dust_storm.py — Mars dust storm simulation model.

Models the lifecycle, intensity, and colony impact of Martian dust
storms from local devils to global planet-encircling events.

Physical references:
  - Mars global dust storms: optical depth τ > 6, last 1-3 months
  - Regional storms: τ 1-4, last 5-20 sols, occur ~10x per Mars year
  - Local dust devils: τ < 0.5, last minutes to hours, clean panels
  - Dust particle diameter: 1-3 μm (suspension), up to 100 μm (saltation)
  - Solar irradiance reduction: I = I₀ · exp(-τ) (Beer-Lambert law)
  - Opportunity rover: survived τ = 5.5 (2018 global storm, but died)
  - Curiosity RTG: unaffected by dust (nuclear), but sensors impacted
  - Dust deposition rate: 2-10 mg/m²/sol during storms (Pathfinder)
  - Wind speed in storms: 60-100 km/h (gusts to 170 km/h in dust devils)
  - MER dust accumulation: 0.28%/sol normal, up to 1.5%/sol in storms

One tick = one sol. Optical depth is dimensionless.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Physical constants (NASA / ESA mission data)
# ---------------------------------------------------------------------------

# Optical depth thresholds
TAU_CLEAR = 0.3              # nominal clear-sky τ on Mars
TAU_LOCAL_MAX = 0.5          # dust devil / local event ceiling
TAU_REGIONAL_MAX = 4.0       # regional storm ceiling
TAU_GLOBAL_MAX = 8.0         # planet-encircling event ceiling
TAU_LETHAL = 10.0            # equipment failure threshold (hypothetical)

# Storm lifecycle (sols)
LOCAL_DURATION_RANGE = (1, 3)       # local dust devils
REGIONAL_DURATION_RANGE = (5, 20)   # regional storms
GLOBAL_DURATION_RANGE = (30, 90)    # global dust storms

# Probability per sol (based on Mars climate data)
P_LOCAL_PER_SOL = 0.15        # ~55 local events per Mars year
P_REGIONAL_PER_SOL = 0.015    # ~10 regional per Mars year
P_GLOBAL_PER_SOL = 0.001      # ~1 every 3 Mars years

# Seasonal modifiers (Mars Ls — solar longitude)
# Dust storm season peaks at Ls 200-330 (southern spring/summer)
STORM_SEASON_PEAK_LS = 270.0  # perihelion passage
STORM_SEASON_WIDTH = 60.0     # half-width in degrees Ls

# Solar irradiance (Beer-Lambert attenuation)
MARS_SOLAR_IRRADIANCE_W_M2 = 590.0  # average Mars orbit

# Dust deposition on surfaces
DEPOSITION_RATE_CLEAR = 0.001    # fraction of surface covered per sol (clear)
DEPOSITION_RATE_STORM = 0.015    # fraction per sol during storm (Pathfinder)

# Wind effects
WIND_SPEED_CLEAR_KMH = 20.0     # average Mars surface wind
WIND_SPEED_STORM_KMH = 80.0     # typical storm wind
WIND_GUST_MAX_KMH = 170.0       # extreme dust devil gust

# Abrasion damage per sol (fraction of equipment health lost)
ABRASION_RATE_CLEAR = 0.0       # negligible in calm weather
ABRASION_RATE_STORM = 0.0005    # 0.05%/sol during storms

# Temperature effect: dust storms WARM the atmosphere (greenhouse)
# but COOL the surface (blocking sunlight). Net surface cooling.
SURFACE_COOLING_PER_TAU = -2.5   # °C per unit optical depth


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DustEvent:
    """A single dust event (devil, regional, or global storm).

    Attributes:
        kind: 'local', 'regional', or 'global'
        tau_peak: peak optical depth of the event
        duration_sols: total lifespan of the event
        sols_elapsed: how many sols since event began
        active: whether event is still ongoing
    """
    kind: str
    tau_peak: float
    duration_sols: int
    sols_elapsed: int = 0
    active: bool = True

    def __post_init__(self) -> None:
        """Validate fields."""
        valid_kinds = ('local', 'regional', 'global')
        if self.kind not in valid_kinds:
            raise ValueError(f"kind must be one of {valid_kinds}, got {self.kind!r}")
        self.tau_peak = max(0.0, float(self.tau_peak))
        self.duration_sols = max(1, int(self.duration_sols))
        self.sols_elapsed = max(0, int(self.sols_elapsed))

    def current_tau(self) -> float:
        """Optical depth at current point in event lifecycle.

        Ramps up in first 20%, sustains 60%, decays in last 20%.
        Returns 0.0 if event is no longer active.
        """
        if not self.active:
            return 0.0
        frac = self.sols_elapsed / max(1, self.duration_sols)
        if frac >= 1.0:
            return 0.0
        ramp_end = 0.2
        sustain_end = 0.8
        if frac < ramp_end:
            # Linear ramp up
            return self.tau_peak * (frac / ramp_end)
        elif frac < sustain_end:
            # Full intensity
            return self.tau_peak
        else:
            # Linear decay
            return self.tau_peak * (1.0 - (frac - sustain_end) / (1.0 - sustain_end))


@dataclass
class DustState:
    """Colony dust storm tracking state.

    Attributes:
        sol: current simulation sol
        current_tau: combined optical depth from all active events
        solar_factor: fraction of sunlight reaching surface [0, 1]
        dust_coverage: fraction of exposed surfaces covered by dust [0, 1]
        equipment_health: equipment condition [0, 1] (abrasion damage)
        surface_temp_delta_c: temperature change from dust effects
        wind_speed_kmh: current effective wind speed
        events: list of active and recent dust events
        total_storm_sols: cumulative sols under storm conditions
        worst_tau_ever: highest optical depth recorded
    """
    sol: int = 0
    current_tau: float = TAU_CLEAR
    solar_factor: float = 1.0
    dust_coverage: float = 0.0
    equipment_health: float = 1.0
    surface_temp_delta_c: float = 0.0
    wind_speed_kmh: float = WIND_SPEED_CLEAR_KMH
    events: List[DustEvent] = field(default_factory=list)
    total_storm_sols: int = 0
    worst_tau_ever: float = 0.0


# ---------------------------------------------------------------------------
# Pure physics functions
# ---------------------------------------------------------------------------

def seasonal_storm_modifier(sol: int, sols_per_year: float = 668.6) -> float:
    """Seasonal modifier for storm probability based on Mars Ls.

    Dust storm season peaks near perihelion (Ls ~270°).
    Returns a multiplier [0.2, 3.0] applied to base storm probability.

    Args:
        sol: current simulation sol
        sols_per_year: sols in one Mars year (668.6)

    Returns:
        Probability multiplier for dust storm generation.
    """
    ls = (sol / sols_per_year) * 360.0 % 360.0
    delta = min(abs(ls - STORM_SEASON_PEAK_LS),
                360.0 - abs(ls - STORM_SEASON_PEAK_LS))
    # Gaussian-ish seasonal curve
    modifier = 0.2 + 2.8 * math.exp(-(delta ** 2) / (2 * STORM_SEASON_WIDTH ** 2))
    return modifier


def beer_lambert(tau: float) -> float:
    """Solar irradiance fraction reaching surface through dust.

    Applies Beer-Lambert law: I/I₀ = exp(-τ).

    Args:
        tau: optical depth (dimensionless, >= 0)

    Returns:
        Fraction of sunlight reaching surface [0, 1].
    """
    tau = max(0.0, float(tau))
    return math.exp(-tau)


def dust_deposition_rate(tau: float) -> float:
    """Rate of dust accumulation on surfaces given current optical depth.

    Interpolates between clear-sky and storm deposition rates.

    Args:
        tau: current optical depth

    Returns:
        Fraction of surface newly covered per sol.
    """
    tau = max(0.0, float(tau))
    if tau <= TAU_CLEAR:
        return DEPOSITION_RATE_CLEAR
    storm_frac = min(1.0, (tau - TAU_CLEAR) / (TAU_REGIONAL_MAX - TAU_CLEAR))
    return DEPOSITION_RATE_CLEAR + storm_frac * (DEPOSITION_RATE_STORM - DEPOSITION_RATE_CLEAR)


def wind_speed(tau: float) -> float:
    """Estimate wind speed from dust activity level.

    Higher optical depth generally correlates with stronger winds.

    Args:
        tau: current optical depth

    Returns:
        Estimated wind speed in km/h.
    """
    tau = max(0.0, float(tau))
    if tau <= TAU_CLEAR:
        return WIND_SPEED_CLEAR_KMH
    storm_frac = min(1.0, (tau - TAU_CLEAR) / TAU_GLOBAL_MAX)
    return WIND_SPEED_CLEAR_KMH + storm_frac * (WIND_SPEED_STORM_KMH - WIND_SPEED_CLEAR_KMH)


def abrasion_damage(tau: float) -> float:
    """Equipment abrasion damage per sol from wind-driven dust.

    Args:
        tau: current optical depth

    Returns:
        Fraction of equipment health lost this sol.
    """
    tau = max(0.0, float(tau))
    if tau <= TAU_CLEAR:
        return ABRASION_RATE_CLEAR
    storm_frac = min(1.0, (tau - TAU_CLEAR) / TAU_GLOBAL_MAX)
    return storm_frac * ABRASION_RATE_STORM


def surface_temp_effect(tau: float) -> float:
    """Surface temperature change from dust optical depth.

    Dust in atmosphere blocks sunlight -> surface cooling.
    Dust also provides greenhouse warming, but net effect is cooling.

    Args:
        tau: current optical depth

    Returns:
        Temperature delta in °C (negative = cooling).
    """
    tau = max(0.0, float(tau))
    excess = tau - TAU_CLEAR
    if excess <= 0:
        return 0.0
    return SURFACE_COOLING_PER_TAU * excess


def visibility_km(tau: float) -> float:
    """Horizontal visibility estimate from optical depth.

    Uses empirical Mars relation: V ≈ 3.0 / τ (km).
    Clear sky (~τ 0.3) gives ~10 km visibility.

    Args:
        tau: optical depth

    Returns:
        Estimated visibility in km (clamped to [0.01, 100]).
    """
    tau = max(0.01, float(tau))
    return max(0.01, min(100.0, 3.0 / tau))


def generate_event(kind: str, rng: random.Random | None = None) -> DustEvent:
    """Create a new dust event of the given kind.

    Args:
        kind: 'local', 'regional', or 'global'
        rng: optional Random instance for reproducibility

    Returns:
        A new DustEvent with randomized parameters.
    """
    r = rng or random.Random()
    if kind == 'local':
        tau = r.uniform(0.1, TAU_LOCAL_MAX)
        dur = r.randint(*LOCAL_DURATION_RANGE)
    elif kind == 'regional':
        tau = r.uniform(TAU_LOCAL_MAX, TAU_REGIONAL_MAX)
        dur = r.randint(*REGIONAL_DURATION_RANGE)
    elif kind == 'global':
        tau = r.uniform(TAU_REGIONAL_MAX, TAU_GLOBAL_MAX)
        dur = r.randint(*GLOBAL_DURATION_RANGE)
    else:
        raise ValueError(f"Unknown event kind: {kind!r}")
    return DustEvent(kind=kind, tau_peak=tau, duration_sols=dur)


# ---------------------------------------------------------------------------
# Tick function — advance one sol
# ---------------------------------------------------------------------------

def tick_dust(
    state: DustState,
    rng: random.Random | None = None,
    force_event: str | None = None,
) -> DustState:
    """Advance dust storm simulation by one sol.

    Generates new events stochastically, advances existing events,
    computes combined optical depth and all derived effects.

    Args:
        state: current DustState (mutated in place)
        rng: optional Random for reproducibility
        force_event: if set, force-spawn an event of this kind

    Returns:
        The same state object (mutated).
    """
    r = rng or random.Random()
    state.sol += 1

    # --- Generate new events ---
    season = seasonal_storm_modifier(state.sol)

    if force_event:
        state.events.append(generate_event(force_event, r))
    else:
        if r.random() < P_LOCAL_PER_SOL * season:
            state.events.append(generate_event('local', r))
        if r.random() < P_REGIONAL_PER_SOL * season:
            state.events.append(generate_event('regional', r))
        if r.random() < P_GLOBAL_PER_SOL * season:
            state.events.append(generate_event('global', r))

    # --- Advance existing events ---
    combined_tau = TAU_CLEAR
    for event in state.events:
        if event.active:
            event.sols_elapsed += 1
            if event.sols_elapsed >= event.duration_sols:
                event.active = False
            combined_tau += event.current_tau()

    state.current_tau = min(combined_tau, TAU_LETHAL)

    # --- Derive effects ---
    state.solar_factor = beer_lambert(state.current_tau)
    state.dust_coverage = min(1.0, state.dust_coverage +
                              dust_deposition_rate(state.current_tau))
    state.equipment_health = max(0.0, state.equipment_health -
                                  abrasion_damage(state.current_tau))
    state.surface_temp_delta_c = surface_temp_effect(state.current_tau)
    state.wind_speed_kmh = wind_speed(state.current_tau)

    # Track storm sols
    if state.current_tau > TAU_CLEAR + 0.1:
        state.total_storm_sols += 1

    # Track worst ever
    if state.current_tau > state.worst_tau_ever:
        state.worst_tau_ever = state.current_tau

    # Prune dead events (keep last 10 for history)
    dead = [e for e in state.events if not e.active]
    alive = [e for e in state.events if e.active]
    state.events = alive + dead[-10:]

    return state


def clean_surfaces(state: DustState, fraction: float = 0.9) -> None:
    """Simulate crew cleaning dust from surfaces.

    Args:
        state: current DustState (mutated)
        fraction: fraction of dust removed [0, 1]
    """
    fraction = max(0.0, min(1.0, fraction))
    state.dust_coverage = max(0.0, state.dust_coverage * (1.0 - fraction))


# ---------------------------------------------------------------------------
# Colony-scale factory
# ---------------------------------------------------------------------------

def create_colony_dust_state(scale: str = "outpost") -> DustState:
    """Create initial dust state for a colony scale.

    Args:
        scale: 'outpost', 'base', or 'settlement'

    Returns:
        A fresh DustState ready for simulation.
    """
    return DustState()
