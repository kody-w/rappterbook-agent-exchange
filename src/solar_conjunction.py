"""
solar_conjunction.py — Mars–Sun conjunction communications blackout model.

Every ~26 months Mars passes behind the Sun as seen from Earth.  For
about two weeks the Sun's corona drowns the radio link and the colony
is on its own — no telemetry uplink, no command downlink, no software
patches, no "Houston we have a problem."

This module predicts conjunction windows, models signal degradation as
a function of Sun–Earth–Probe (SEP) angle, and tracks the data-link
budget so the colony knows *exactly* how many bits/second it can push
at any point in the cycle.

Physical references
───────────────────
  - Mars synodic period:  779.94 days  (Meeus, Astronomical Algorithms)
  - SEP blackout threshold:  ≤2° — NASA DSN suspends commanding
  - SEP degraded threshold:  ≤5° — significant scintillation noise
  - Blackout duration:  ~13–16 sols depending on orbital eccentricity
  - Nominal Mars deep-space link:  2 Mbps at X-band  (Mars Reconnaissance Orbiter)
  - Solar scintillation noise: ∝ SEP⁻² for small angles  (Morabito+ 2003)
  - Light-time Earth–Mars varies 4–24 minutes one-way
  - Last conjunction:  2025-01-11  (NASA/JPL Horizons)

One tick = one sol.  Angles in degrees.  Data rate in bits/second.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


# ── Orbital mechanics constants ──────────────────────────────────────

EARTH_ORBITAL_PERIOD_DAYS = 365.25
MARS_ORBITAL_PERIOD_DAYS = 686.97
SYNODIC_PERIOD_DAYS = 779.94          # mean time between conjunctions

#: A known conjunction epoch (2025-01-11) expressed as a sol offset.
#: Sol 0 of the simulation can be any date; the caller aligns this.
CONJUNCTION_EPOCH_SOL = 0.0

#: Mars orbital eccentricity (causes ±1 sol jitter in blackout length)
MARS_ECCENTRICITY = 0.0934

# ── SEP angle thresholds ─────────────────────────────────────────────

SEP_BLACKOUT_DEG = 2.0     # commanding suspended
SEP_DEGRADED_DEG = 5.0     # heavy scintillation, reduced rate
SEP_NOMINAL_DEG = 15.0     # essentially full bandwidth

# ── Data-link budget ─────────────────────────────────────────────────

NOMINAL_DATA_RATE_BPS = 2_000_000     # 2 Mbps at X-band (MRO-class)
MIN_DATA_RATE_BPS = 0                 # blackout = zero
SCINTILLATION_EXPONENT = 2.0          # noise ∝ SEP^{-n}

# ── Autonomy ─────────────────────────────────────────────────────────

CACHE_PREFETCH_SOLS = 30   # start caching critical data N sols early
AUTONOMY_BUFFER_SOLS = 5   # extra margin beyond predicted blackout


# =====================================================================
# Geometry helpers
# =====================================================================

def _wrap_angle(deg: float) -> float:
    """Wrap an angle to [0, 360)."""
    return deg % 360.0


def mean_anomaly(sol: float, period: float, epoch_offset: float = 0.0) -> float:
    """Mean anomaly at *sol* for a body with *period* (days/sols).

    Returns degrees in [0, 360).
    """
    return _wrap_angle(360.0 * (sol - epoch_offset) / period)


def eccentric_anomaly(M_deg: float, ecc: float, tol: float = 1e-8) -> float:
    """Solve Kepler's equation  M = E − e·sin(E)  via Newton–Raphson.

    *M_deg* in degrees, returns E in degrees.
    """
    M = math.radians(M_deg)
    E = M  # initial guess
    for _ in range(50):
        dE = (E - ecc * math.sin(E) - M) / (1.0 - ecc * math.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return math.degrees(E) % 360.0


def true_anomaly(E_deg: float, ecc: float) -> float:
    """True anomaly from eccentric anomaly and eccentricity.

    Returns degrees in [0, 360).
    """
    E = math.radians(E_deg)
    nu = 2.0 * math.atan2(
        math.sqrt(1.0 + ecc) * math.sin(E / 2.0),
        math.sqrt(1.0 - ecc) * math.cos(E / 2.0),
    )
    return math.degrees(nu) % 360.0


def heliocentric_longitude(sol: float, period: float, ecc: float,
                           epoch_offset: float = 0.0) -> float:
    """Heliocentric ecliptic longitude (degrees) at *sol*.

    Simplified: assumes longitude of perihelion = 0 for both bodies
    (the *relative* angle is what matters for conjunction prediction).
    """
    M = mean_anomaly(sol, period, epoch_offset)
    E = eccentric_anomaly(M, ecc)
    return true_anomaly(E, ecc)


def sep_angle(sol: float, conjunction_epoch: float = CONJUNCTION_EPOCH_SOL) -> float:
    """Sun–Earth–Probe angle in degrees at a given sol.

    Uses a sinusoidal approximation centred on the conjunction epoch:
    the SEP angle goes to 0 at conjunction and peaks at ~180° halfway
    through the synodic period.

    More accurate than a fixed lookup table, less expensive than a
    full ephemeris.
    """
    # Phase: 0 at conjunction, 0.5 at opposition
    phase = ((sol - conjunction_epoch) % SYNODIC_PERIOD_DAYS) / SYNODIC_PERIOD_DAYS

    # SEP follows a sinusoidal envelope from 0° (conjunction) to
    # ~180° (opposition) and back.  The *minimum* angle at
    # conjunction isn't exactly 0 because the orbits are inclined
    # ~1.85°, but we model that as a floor.
    angle = 180.0 * math.sin(math.pi * phase)

    # Floor: Mars orbit inclination keeps the SEP from hitting true 0
    return max(angle, 0.0)


# =====================================================================
# Signal model
# =====================================================================

def data_rate_bps(sep_deg: float) -> float:
    """Achievable data rate (bits/sec) given the SEP angle.

    - SEP ≥ 15°  → nominal rate (2 Mbps)
    - 5° < SEP < 15° → linearly degraded by scintillation
    - 2° < SEP ≤ 5°  → heavy scintillation, rate ∝ (SEP/5)²
    - SEP ≤ 2°  → blackout (0 bps)
    """
    if sep_deg <= SEP_BLACKOUT_DEG:
        return MIN_DATA_RATE_BPS

    if sep_deg <= SEP_DEGRADED_DEG:
        # Quadratic rolloff in the heavy-scintillation zone
        fraction = (sep_deg / SEP_DEGRADED_DEG) ** SCINTILLATION_EXPONENT
        return NOMINAL_DATA_RATE_BPS * fraction * (SEP_DEGRADED_DEG / SEP_NOMINAL_DEG)

    if sep_deg < SEP_NOMINAL_DEG:
        # Linear taper from degraded to nominal
        t = (sep_deg - SEP_DEGRADED_DEG) / (SEP_NOMINAL_DEG - SEP_DEGRADED_DEG)
        low = NOMINAL_DATA_RATE_BPS * (SEP_DEGRADED_DEG / SEP_NOMINAL_DEG)
        return low + t * (NOMINAL_DATA_RATE_BPS - low)

    return NOMINAL_DATA_RATE_BPS


def is_blackout(sep_deg: float) -> bool:
    """True when the radio link is completely blocked."""
    return sep_deg <= SEP_BLACKOUT_DEG


def is_degraded(sep_deg: float) -> bool:
    """True when the link is degraded but not blacked out."""
    return SEP_BLACKOUT_DEG < sep_deg <= SEP_DEGRADED_DEG


def link_status(sep_deg: float) -> str:
    """Human-readable link status string."""
    if is_blackout(sep_deg):
        return "blackout"
    if is_degraded(sep_deg):
        return "degraded"
    if sep_deg < SEP_NOMINAL_DEG:
        return "reduced"
    return "nominal"


# =====================================================================
# Conjunction window prediction
# =====================================================================

def next_conjunction_sol(current_sol: float,
                        epoch: float = CONJUNCTION_EPOCH_SOL) -> float:
    """Sol of the next conjunction *after* current_sol."""
    elapsed = current_sol - epoch
    cycles = math.ceil(elapsed / SYNODIC_PERIOD_DAYS)
    nxt = epoch + cycles * SYNODIC_PERIOD_DAYS
    if nxt <= current_sol:
        nxt += SYNODIC_PERIOD_DAYS
    return nxt


def conjunction_window(center_sol: float,
                       threshold_deg: float = SEP_BLACKOUT_DEG
                       ) -> tuple:
    """(start_sol, end_sol) of the blackout window around a conjunction.

    Scans outward from *center_sol* in 0.5-sol steps until the SEP
    exceeds *threshold_deg*.
    """
    step = 0.5
    # find start (scan backward)
    s = center_sol
    while sep_angle(s, center_sol) <= threshold_deg and s > center_sol - SYNODIC_PERIOD_DAYS / 2:
        s -= step
    start = s + step  # first sol still inside the window

    # find end (scan forward)
    e = center_sol
    while sep_angle(e, center_sol) <= threshold_deg and e < center_sol + SYNODIC_PERIOD_DAYS / 2:
        e += step
    end = e - step

    return (start, end)


def blackout_duration_sols(center_sol: float) -> float:
    """Duration of the blackout window in sols."""
    start, end = conjunction_window(center_sol)
    return max(0.0, end - start)


# =====================================================================
# State dataclass
# =====================================================================

@dataclass
class ConjunctionState:
    """Mutable state for the conjunction tracker.

    Attributes
    ----------
    sol : float
        Current simulation sol.
    next_conjunction : float
        Predicted sol of the next conjunction center.
    sep_deg : float
        Current Sun–Earth–Probe angle.
    data_rate_bps : float
        Current achievable data rate.
    status : str
        "nominal", "reduced", "degraded", or "blackout".
    sols_in_blackout : int
        Running counter of how many sols spent in blackout this window.
    total_blackout_sols : int
        Cumulative blackout sols over the simulation lifetime.
    cache_level : float
        Fraction [0, 1] of critical data cached for autonomy (1 = fully prepared).
    autonomy_active : bool
        True when the colony is operating without Earth contact.
    """
    sol: float = 0.0
    next_conjunction: float = 0.0
    sep_deg: float = 180.0
    data_rate_bps: float = NOMINAL_DATA_RATE_BPS
    status: str = "nominal"
    sols_in_blackout: int = 0
    total_blackout_sols: int = 0
    cache_level: float = 0.0
    autonomy_active: bool = False


# =====================================================================
# Tick result
# =====================================================================

@dataclass
class TickResult:
    """Per-sol output from the conjunction tracker."""
    sol: float
    sep_deg: float
    data_rate_bps: float
    status: str
    autonomy_active: bool
    cache_level: float
    sols_to_conjunction: float
    blackout_duration: float


# =====================================================================
# Tick engine
# =====================================================================

def tick(state: ConjunctionState) -> TickResult:
    """Advance the conjunction tracker by one sol.

    1. Update SEP angle from orbital geometry.
    2. Compute data rate from signal model.
    3. Determine link status.
    4. Manage cache-prefetch countdown.
    5. Toggle autonomy mode on blackout entry/exit.
    6. Advance sol counter.
    """
    sol = state.sol

    # ── 1. Orbital geometry ──────────────────────────────────────
    state.sep_deg = sep_angle(sol, state.next_conjunction)
    sep = state.sep_deg

    # ── 2. Signal budget ─────────────────────────────────────────
    state.data_rate_bps = data_rate_bps(sep)

    # ── 3. Link status ───────────────────────────────────────────
    state.status = link_status(sep)

    # ── 4. Cache prefetch ────────────────────────────────────────
    sols_to_conj = state.next_conjunction - sol
    if sols_to_conj <= CACHE_PREFETCH_SOLS and state.cache_level < 1.0:
        # Linear ramp: cache fills from 0→1 over CACHE_PREFETCH_SOLS
        increment = 1.0 / CACHE_PREFETCH_SOLS
        state.cache_level = min(1.0, state.cache_level + increment)

    # ── 5. Autonomy toggle ───────────────────────────────────────
    if state.status == "blackout":
        if not state.autonomy_active:
            state.autonomy_active = True
        state.sols_in_blackout += 1
        state.total_blackout_sols += 1
    else:
        if state.autonomy_active and state.status != "degraded":
            # Exit autonomy only once link is above degraded
            state.autonomy_active = False
            state.sols_in_blackout = 0
            state.cache_level = 0.0  # reset for next cycle

    # ── 6. Advance epoch ─────────────────────────────────────────
    if sol >= state.next_conjunction + SYNODIC_PERIOD_DAYS / 4:
        # Past this conjunction — schedule the next one
        state.next_conjunction += SYNODIC_PERIOD_DAYS

    bd = blackout_duration_sols(state.next_conjunction)

    result = TickResult(
        sol=sol,
        sep_deg=round(sep, 4),
        data_rate_bps=round(state.data_rate_bps, 2),
        status=state.status,
        autonomy_active=state.autonomy_active,
        cache_level=round(state.cache_level, 4),
        sols_to_conjunction=round(max(0.0, sols_to_conj), 2),
        blackout_duration=round(bd, 2),
    )

    state.sol += 1.0
    return result


# =====================================================================
# Multi-sol simulation
# =====================================================================

def run_simulation(sols: int, conjunction_sol: float = 400.0) -> list:
    """Run the conjunction tracker for *sols* ticks.

    *conjunction_sol*: sol at which the first conjunction occurs.
    Returns a list of TickResult, one per sol.
    """
    state = ConjunctionState(
        sol=0.0,
        next_conjunction=conjunction_sol,
        sep_deg=sep_angle(0.0, conjunction_sol),
    )
    results = []
    for _ in range(sols):
        results.append(tick(state))
    return results


# =====================================================================
# Serialisation
# =====================================================================

def state_to_dict(state: ConjunctionState) -> dict:
    """Serialise state for JSON persistence."""
    return {
        "sol": state.sol,
        "next_conjunction": state.next_conjunction,
        "sep_deg": state.sep_deg,
        "data_rate_bps": state.data_rate_bps,
        "status": state.status,
        "sols_in_blackout": state.sols_in_blackout,
        "total_blackout_sols": state.total_blackout_sols,
        "cache_level": state.cache_level,
        "autonomy_active": state.autonomy_active,
    }


def state_from_dict(d: dict) -> ConjunctionState:
    """Deserialise state from a JSON-loaded dict."""
    return ConjunctionState(
        sol=float(d.get("sol", 0)),
        next_conjunction=float(d.get("next_conjunction", 0)),
        sep_deg=float(d.get("sep_deg", 180)),
        data_rate_bps=float(d.get("data_rate_bps", NOMINAL_DATA_RATE_BPS)),
        status=str(d.get("status", "nominal")),
        sols_in_blackout=int(d.get("sols_in_blackout", 0)),
        total_blackout_sols=int(d.get("total_blackout_sols", 0)),
        cache_level=float(d.get("cache_level", 0)),
        autonomy_active=bool(d.get("autonomy_active", False)),
    )
