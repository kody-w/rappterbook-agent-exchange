"""
seismology.py — Marsquake simulation and structural impact model.

Models seismic activity on Mars based on NASA InSight mission data:
  - Stochastic quake generation (magnitude-frequency from InSight catalog)
  - Ground acceleration from magnitude and distance (attenuation model)
  - Structural damage assessment (peak ground acceleration vs building codes)
  - Cumulative fatigue on habitat structures
  - Aftershock sequences (modified Omori law)
  - Seasonal variation (InSight observed tidal modulation)

Physical references:
  - InSight detected 1,319 marsquakes over ~4 Earth years (2018-2022)
  - Largest: S1222a, magnitude 4.7 (May 2022)
  - Most quakes: M 1.0-3.0, shallow crustal events
  - Mars lacks plate tectonics — quakes from thermal contraction + impacts
  - Cerberus Fossae: most active seismic region
  - Seismic velocity: ~3.5 km/s (crust), slower than Earth (~6 km/s)
  - Mars mantle Q (quality factor): ~300 (less attenuating than Moon)
  - Frequency: ~1 quake/sol on average (InSight catalog)
  - Colony concern threshold: PGA > 0.01 m/s² (~0.001g)

One tick = one sol. Magnitude in Richter-like scale. Acceleration in m/s².
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Physical constants (InSight mission data)
# ---------------------------------------------------------------------------

# Quake frequency
QUAKE_RATE_PER_SOL = 0.9             # average events/sol (InSight: ~1/sol)
MAGNITUDE_DISTRIBUTION = [
    # (mag_min, mag_max, relative_weight)
    (0.5, 1.0, 0.40),     # very weak — most common
    (1.0, 2.0, 0.35),     # weak
    (2.0, 3.0, 0.18),     # moderate
    (3.0, 4.0, 0.05),     # strong (rare)
    (4.0, 5.0, 0.015),    # very strong (very rare)
    (5.0, 6.0, 0.005),    # extreme (once per mission)
]

# Attenuation: PGA = a × 10^(b×M) / r^c
ATTEN_A = 2.0e-6           # base amplitude (m/s²)
ATTEN_B = 1.0              # magnitude exponent
ATTEN_C = 1.5              # geometric spreading exponent
DEFAULT_DISTANCE_KM = 500.0

# Structural thresholds (PGA in m/s²)
PGA_IMPERCEPTIBLE = 0.001   # instruments only
PGA_PERCEPTIBLE = 0.01      # colonists feel it
PGA_MINOR_DAMAGE = 0.1      # equipment shifts
PGA_MODERATE_DAMAGE = 0.5   # seal stress
PGA_SEVERE_DAMAGE = 2.0     # habitat compromise

# Fatigue
FATIGUE_RATE_PER_PGA = 0.001
FATIGUE_THRESHOLD = 1.0      # mandatory inspection
FATIGUE_CRITICAL = 2.0       # structural risk

# Aftershocks (modified Omori law)
AFTERSHOCK_K = 3.0
AFTERSHOCK_C = 0.5
AFTERSHOCK_P = 1.1
AFTERSHOCK_MAG_DROP = 1.2    # aftershocks ~1.2 mag below mainshock
AFTERSHOCK_MIN_MAINSHOCK = 3.0  # only M >= 3.0 triggers aftershocks

# Seasonal modulation
SEASONAL_AMPLITUDE = 0.3     # ±30% rate variation
SEASONAL_PERIOD_SOLS = 669.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Quake:
    """A single marsquake event."""
    magnitude: float
    distance_km: float
    pga_m_s2: float
    sol: int
    is_aftershock: bool = False


@dataclass
class SeismicState:
    """Colony seismic monitoring state."""
    sol: int = 0
    structural_fatigue: float = 0.0
    quake_log: list = field(default_factory=list)
    total_quakes: int = 0
    max_magnitude: float = 0.0
    max_pga: float = 0.0
    aftershock_queue: list = field(default_factory=list)
    colony_distance_km: float = DEFAULT_DISTANCE_KM


# ---------------------------------------------------------------------------
# Physics functions
# ---------------------------------------------------------------------------

def peak_ground_acceleration(magnitude: float, distance_km: float) -> float:
    """PGA from magnitude and distance: PGA = a × 10^(b×M) / r^c."""
    if distance_km <= 0 or magnitude < 0:
        return 0.0
    amplitude = ATTEN_A * (10.0 ** (ATTEN_B * magnitude))
    return amplitude / (distance_km ** ATTEN_C)


def seasonal_rate_modifier(sol: int) -> float:
    """Seasonal quake rate modifier. Returns multiplier [0.7, 1.3]."""
    phase = 2.0 * math.pi * (sol % SEASONAL_PERIOD_SOLS) / SEASONAL_PERIOD_SOLS
    return 1.0 + SEASONAL_AMPLITUDE * math.sin(phase)


def damage_category(pga: float) -> str:
    """Classify structural damage from PGA."""
    if pga < PGA_IMPERCEPTIBLE:
        return "none"
    if pga < PGA_PERCEPTIBLE:
        return "imperceptible"
    if pga < PGA_MINOR_DAMAGE:
        return "perceptible"
    if pga < PGA_MODERATE_DAMAGE:
        return "minor"
    if pga < PGA_SEVERE_DAMAGE:
        return "moderate"
    return "severe"


def generate_quakes(
    sol: int,
    colony_distance_km: float = DEFAULT_DISTANCE_KM,
    rng: random.Random | None = None,
) -> list[Quake]:
    """Generate stochastic marsquakes for one sol."""
    r = rng if rng else random
    rate = QUAKE_RATE_PER_SOL * seasonal_rate_modifier(sol)

    # Poisson process for event count
    n_events = 0
    prob = math.exp(-rate)
    cumulative = prob
    roll = r.random()
    while cumulative < roll and n_events < 20:
        n_events += 1
        prob *= rate / n_events
        cumulative += prob

    quakes: list[Quake] = []
    for _ in range(n_events):
        # Magnitude from weighted distribution
        mag_roll = r.random()
        cum_weight = 0.0
        mag = 1.0
        for mag_min, mag_max, weight in MAGNITUDE_DISTRIBUTION:
            cum_weight += weight
            if mag_roll <= cum_weight:
                mag = r.uniform(mag_min, mag_max)
                break

        dist = colony_distance_km * r.uniform(0.5, 1.5)
        pga = peak_ground_acceleration(mag, dist)
        quakes.append(Quake(magnitude=mag, distance_km=dist, pga_m_s2=pga, sol=sol))

    return quakes


def generate_aftershocks(
    mainshock: Quake,
    current_sol: int,
    rng: random.Random | None = None,
) -> list[tuple[int, float]]:
    """Aftershock sequence from a significant mainshock (M >= 3.0).

    Uses modified Omori law: n(t) = K / (t + c)^p.
    Returns list of (sol, magnitude).
    """
    if mainshock.magnitude < AFTERSHOCK_MIN_MAINSHOCK:
        return []

    r = rng if rng else random
    aftershocks: list[tuple[int, float]] = []
    max_duration = int(mainshock.magnitude * 5)

    for dt in range(1, max_duration + 1):
        expected = AFTERSHOCK_K / (dt + AFTERSHOCK_C) ** AFTERSHOCK_P
        if r.random() < expected:
            mag = mainshock.magnitude - AFTERSHOCK_MAG_DROP + r.gauss(0, 0.3)
            mag = max(0.5, min(mag, mainshock.magnitude - 0.1))
            aftershocks.append((current_sol + dt, mag))

    return aftershocks


def fatigue_status(structural_fatigue: float) -> str:
    """Classify structural fatigue level."""
    if structural_fatigue < FATIGUE_THRESHOLD:
        return "nominal"
    if structural_fatigue < FATIGUE_CRITICAL:
        return "inspection_needed"
    return "critical"


# ---------------------------------------------------------------------------
# Tick integrator
# ---------------------------------------------------------------------------

def tick_seismology(
    state: SeismicState,
    rng: random.Random | None = None,
) -> dict:
    """Advance seismic state by one sol."""
    state.sol += 1
    events: list[str] = []
    sol_quakes: list[Quake] = []

    # --- New quakes ---
    sol_quakes.extend(generate_quakes(state.sol, state.colony_distance_km, rng))

    # --- Process aftershock queue ---
    remaining: list[tuple[int, float]] = []
    for after_sol, after_mag in state.aftershock_queue:
        if after_sol <= state.sol:
            r = rng if rng else random
            dist = state.colony_distance_km * r.uniform(0.5, 1.5)
            pga = peak_ground_acceleration(after_mag, dist)
            sol_quakes.append(Quake(
                magnitude=after_mag, distance_km=dist,
                pga_m_s2=pga, sol=state.sol, is_aftershock=True,
            ))
        else:
            remaining.append((after_sol, after_mag))
    state.aftershock_queue = remaining

    # --- Update stats ---
    sol_max_pga = 0.0
    sol_max_mag = 0.0
    for q in sol_quakes:
        sol_max_pga = max(sol_max_pga, q.pga_m_s2)
        sol_max_mag = max(sol_max_mag, q.magnitude)
        state.structural_fatigue += q.pga_m_s2 * FATIGUE_RATE_PER_PGA
        state.total_quakes += 1
        if q.magnitude > state.max_magnitude:
            state.max_magnitude = q.magnitude
        if q.pga_m_s2 > state.max_pga:
            state.max_pga = q.pga_m_s2

    # --- Queue aftershocks ---
    for q in sol_quakes:
        if q.magnitude >= AFTERSHOCK_MIN_MAINSHOCK and not q.is_aftershock:
            afters = generate_aftershocks(q, state.sol, rng)
            state.aftershock_queue.extend(afters)
            if afters:
                events.append(f"aftershock_sequence:M{q.magnitude:.1f}:{len(afters)}")

    # --- Prune quake log (last 30 sols) ---
    state.quake_log.extend(sol_quakes)
    cutoff = state.sol - 30
    state.quake_log = [q for q in state.quake_log if q.sol > cutoff]

    # --- Events ---
    if sol_max_mag >= 3.0:
        events.append(f"significant_quake:M{sol_max_mag:.1f}")
    if sol_max_pga >= PGA_MINOR_DAMAGE:
        events.append(f"structural_concern:PGA={sol_max_pga:.6f}")

    return {
        "sol": state.sol,
        "quake_count": len(sol_quakes),
        "max_magnitude": sol_max_mag,
        "max_pga_m_s2": sol_max_pga,
        "damage_category": damage_category(sol_max_pga),
        "structural_fatigue": state.structural_fatigue,
        "fatigue_status": fatigue_status(state.structural_fatigue),
        "total_quakes": state.total_quakes,
        "pending_aftershocks": len(state.aftershock_queue),
        "events": events,
    }
