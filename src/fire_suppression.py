"""fire_suppression.py -- Mars Colony Fire Detection and Suppression System.

Fire in a Mars habitat is existential.  Apollo 1 killed three astronauts
in 17 seconds.  In an enriched-O2 habitat at 70 kPa, an uncontrolled fire
consumes oxygen, fills the sealed volume with toxic smoke, and can breach
the hull.  There is nowhere to run on Mars.

Each tick = 1 minute.

Physics modelled:
  - Ignition probability (O2, humidity, dust, electrical faults)
  - t-squared fire growth limited by available O2
  - O2 consumption, CO2 and CO production
  - Smoke opacity and detector triggering
  - CO2 flooding suppression
  - Crew CO exposure tracking
  - Post-fire atmospheric cleanup

Physical references:
  - Apollo 1: 16.7 psi pure O2, death in 17 seconds
  - ISS fire protocol: alarm, isolate, suppress, ventilate
  - CO lethal: 1200 ppm for 1-3 min
  - CO2 flooding: 34 pct smothers most fires
  - Mars hab: 70 kPa, 30 pct O2 (NASA DRA 5.0)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

NOMINAL_O2_FRACTION = 0.30
NOMINAL_O2_KPA = 21.0
MIN_O2_FOR_COMBUSTION_KPA = 14.0
CRITICAL_O2_KPA = 30.0

ALPHA_SLOW = 0.003
ALPHA_MEDIUM = 0.012
ALPHA_FAST = 0.047

O2_CONSUMED_PER_KW_MIN = 0.08
CO2_PRODUCED_PER_KW_MIN = 0.06
CO_PRODUCED_PER_KW_MIN = 0.002

SMOKE_PER_KW_MIN = 0.05
SMOKE_DETECTOR_THRESHOLD = 0.15
SMOKE_DISSIPATION_RATE = 0.01

CO_LETHAL_KPA = 0.12
CO_DANGEROUS_KPA = 0.04
CO_SCRUB_RATE_KPA_PER_MIN = 0.003

CO2_FLOOD_RATE_KPA_PER_MIN = 5.0
CO2_SMOTHER_THRESHOLD_FRACTION = 0.34
EXTINGUISHER_CHARGES = 5
CHARGE_DURATION_MIN = 3

BASE_FAULT_PROBABILITY = 0.001
DUST_FAULT_MULTIPLIER = 2.0
HUMIDITY_SAFETY_FACTOR = 0.5

DETECTION_DELAY_MIN = 1

DETECTION_POWER_KW = 0.05
SUPPRESSION_POWER_KW = 1.5
VENTILATION_POWER_KW = 0.8


@dataclass
class FireSuppressionState:
    """Complete state of the fire detection and suppression system."""

    o2_kpa: float = NOMINAL_O2_KPA
    co2_kpa: float = 0.04
    co_kpa: float = 0.0
    humidity_fraction: float = 0.40
    dust_loading: float = 0.0

    fire_active: bool = False
    fire_intensity_kw: float = 0.0
    fire_growth_alpha: float = ALPHA_MEDIUM
    fire_duration_min: int = 0
    fires_total: int = 0

    smoke_opacity: float = 0.0

    alarm_active: bool = False
    alarm_triggered_tick: int = 0

    suppression_active: bool = False
    suppression_remaining_min: int = 0
    extinguisher_charges: int = EXTINGUISHER_CHARGES
    co2_flood_kpa: float = 0.0

    crew_co_exposure: float = 0.0
    crew_smoke_exposure: float = 0.0

    risk_score: float = 0.0
    power_draw_kw: float = DETECTION_POWER_KW

    tick: int = 0
    alerts: list = field(default_factory=list)
    rng_seed: int = 42


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def compute_risk_score(state: FireSuppressionState) -> float:
    """Compute composite fire risk score [0-1]."""
    o2_factor = 0.0
    if state.o2_kpa > MIN_O2_FOR_COMBUSTION_KPA:
        o2_normalized = (state.o2_kpa - MIN_O2_FOR_COMBUSTION_KPA) / (
            CRITICAL_O2_KPA - MIN_O2_FOR_COMBUSTION_KPA)
        o2_factor = clamp(o2_normalized ** 1.5, 0.0, 1.0)
    humidity_factor = clamp(1.0 - state.humidity_fraction, 0.0, 1.0)
    dust_factor = clamp(state.dust_loading, 0.0, 1.0)
    risk = 0.4 * o2_factor + 0.3 * humidity_factor + 0.3 * dust_factor
    return clamp(risk, 0.0, 1.0)


def check_ignition(state: FireSuppressionState, rng: random.Random) -> bool:
    """Determine if a fire ignites this tick."""
    if state.fire_active or state.o2_kpa < MIN_O2_FOR_COMBUSTION_KPA:
        return False
    fault_prob = BASE_FAULT_PROBABILITY
    fault_prob *= (1.0 + state.dust_loading * DUST_FAULT_MULTIPLIER)
    fault_prob *= (1.0 - state.humidity_fraction * HUMIDITY_SAFETY_FACTOR)
    if state.o2_kpa > NOMINAL_O2_KPA:
        o2_mult = 1.0 + ((state.o2_kpa - NOMINAL_O2_KPA) / 10.0) ** 2
        fault_prob *= o2_mult
    return rng.random() < fault_prob


def activate_suppression(state: FireSuppressionState) -> FireSuppressionState:
    """Manually activate CO2 flood suppression."""
    state.alerts = []
    if state.extinguisher_charges <= 0:
        state.alerts.append("CRITICAL: no suppression charges remaining")
        return state
    if state.suppression_active:
        state.alerts.append("Suppression already active")
        return state
    state.suppression_active = True
    state.suppression_remaining_min = CHARGE_DURATION_MIN
    state.extinguisher_charges -= 1
    state.alerts.append(
        "SUPPRESSION ACTIVATED -- %d charges remaining" % state.extinguisher_charges)
    return state


def recharge_extinguishers(state: FireSuppressionState) -> FireSuppressionState:
    """Recharge extinguisher system (maintenance)."""
    state.alerts = []
    if state.fire_active or state.suppression_active:
        state.alerts.append("Cannot recharge during active fire/suppression")
        return state
    state.extinguisher_charges = EXTINGUISHER_CHARGES
    state.alerts.append("Extinguishers recharged to full")
    return state


def tick(state: FireSuppressionState) -> FireSuppressionState:
    """Advance fire suppression system by one minute."""
    state.alerts = []
    state.tick += 1
    power = DETECTION_POWER_KW

    seed_val = state.rng_seed + state.tick if state.rng_seed is not None else None
    rng = random.Random(seed_val)

    state.risk_score = compute_risk_score(state)
    if state.risk_score > 0.7 and not state.fire_active:
        state.alerts.append("WARNING: fire risk elevated")

    if not state.fire_active and check_ignition(state, rng):
        state.fire_active = True
        state.fire_duration_min = 0
        state.fires_total += 1
        state.fire_growth_alpha = rng.choice([ALPHA_SLOW, ALPHA_MEDIUM, ALPHA_FAST])
        state.alerts.append("FIRE DETECTED -- automatic response initiating")

    if state.fire_active:
        state.fire_duration_min += 1
        t_sec = state.fire_duration_min * 60.0
        state.fire_intensity_kw = state.fire_growth_alpha * (t_sec ** 2) / 1000.0
        max_intensity = (state.o2_kpa / O2_CONSUMED_PER_KW_MIN) * 0.5
        state.fire_intensity_kw = min(state.fire_intensity_kw, max(0.0, max_intensity))

        o2_consumed = O2_CONSUMED_PER_KW_MIN * state.fire_intensity_kw
        state.o2_kpa = max(0.0, state.o2_kpa - o2_consumed)
        state.co2_kpa += CO2_PRODUCED_PER_KW_MIN * state.fire_intensity_kw
        state.co_kpa += CO_PRODUCED_PER_KW_MIN * state.fire_intensity_kw
        state.smoke_opacity += SMOKE_PER_KW_MIN * state.fire_intensity_kw
        state.smoke_opacity = clamp(state.smoke_opacity, 0.0, 1.0)

        if state.o2_kpa < MIN_O2_FOR_COMBUSTION_KPA:
            state.fire_active = False
            state.fire_intensity_kw = 0.0
            state.alerts.append(
                "Fire self-extinguished -- O2 below combustion threshold")

    if state.smoke_opacity >= SMOKE_DETECTOR_THRESHOLD and not state.alarm_active:
        state.alarm_active = True
        state.alarm_triggered_tick = state.tick
        state.alerts.append("SMOKE ALARM -- detection triggered")

    if (state.alarm_active and state.fire_active
            and not state.suppression_active
            and state.tick >= state.alarm_triggered_tick + DETECTION_DELAY_MIN
            and state.extinguisher_charges > 0):
        state = activate_suppression(state)

    if state.suppression_active:
        state.suppression_remaining_min -= 1
        state.co2_flood_kpa += CO2_FLOOD_RATE_KPA_PER_MIN
        state.co2_kpa += CO2_FLOOD_RATE_KPA_PER_MIN
        power += SUPPRESSION_POWER_KW
        total_p = state.o2_kpa + state.co2_kpa + state.co_kpa
        if total_p > 0 and (state.co2_kpa / total_p) >= CO2_SMOTHER_THRESHOLD_FRACTION:
            state.fire_active = False
            state.fire_intensity_kw = 0.0
            state.alerts.append("Fire suppressed -- CO2 flooding effective")
        if state.suppression_remaining_min <= 0:
            state.suppression_active = False
            state.alerts.append("Suppression charge depleted")

    if not state.fire_active:
        if state.co_kpa > 0:
            scrubbed = min(CO_SCRUB_RATE_KPA_PER_MIN, state.co_kpa)
            state.co_kpa = max(0.0, state.co_kpa - scrubbed)
            power += VENTILATION_POWER_KW
        if state.smoke_opacity > 0:
            state.smoke_opacity = max(0.0,
                                      state.smoke_opacity - SMOKE_DISSIPATION_RATE)
            power += VENTILATION_POWER_KW
        if state.smoke_opacity < SMOKE_DETECTOR_THRESHOLD * 0.5 and state.alarm_active:
            state.alarm_active = False

    if state.co_kpa > 0:
        state.crew_co_exposure += state.co_kpa * 10.0
    if state.smoke_opacity > 0.1:
        state.crew_smoke_exposure += state.smoke_opacity

    if state.co_kpa >= CO_LETHAL_KPA:
        state.alerts.append("CRITICAL: CO at lethal levels -- evacuate immediately")
    elif state.co_kpa >= CO_DANGEROUS_KPA:
        state.alerts.append("DANGER: CO levels dangerous -- don breathing apparatus")

    state.o2_kpa = clamp(state.o2_kpa, 0.0, 50.0)
    state.co2_kpa = clamp(state.co2_kpa, 0.0, 100.0)
    state.co_kpa = clamp(state.co_kpa, 0.0, 10.0)
    state.smoke_opacity = clamp(state.smoke_opacity, 0.0, 1.0)
    state.fire_intensity_kw = max(0.0, state.fire_intensity_kw)
    state.power_draw_kw = power
    state.risk_score = clamp(state.risk_score, 0.0, 1.0)
    return state


def run_simulation(num_ticks, rng_seed=42, force_ignition_at=-1):
    """Run simulation for num_ticks minutes. Returns list of state snapshots."""
    from dataclasses import asdict
    state = FireSuppressionState(rng_seed=rng_seed)
    history = []
    for t in range(num_ticks):
        if force_ignition_at >= 0 and t == force_ignition_at:
            state.fire_active = True
            state.fire_duration_min = 0
            state.fires_total += 1
            state.fire_growth_alpha = ALPHA_MEDIUM
        state = tick(state)
        history.append(FireSuppressionState(**asdict(state)))
    return history
