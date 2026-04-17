"""
Behavioral psychology organ for Mars-100 (engine v9.0).

Maps psychological state to action-weight modifiers.  Stressed colonists
gravitate toward rest/prayer, lonely colonists seek cooperation, and
purpose-driven colonists lean into research/terraform.

Phase 1 (this module):
  - compute_behavior_weights(): pure function PsychState -> action deltas
  - Crisis-forced-rest via persisted ``forced_rest_until`` field
  - Per-action cap on total psych influence to prevent rest-attractor spirals
  - Critical-resource floors preserved (farm/terraform/code stay viable)

Design notes:
  - No new RNG stream -- all outputs are deterministic functions of PsychState
  - Deltas are additive, matching cultural_pressure / econ_pressure pattern
  - Called AFTER base weights, cultural, and economic pressure
"""
from __future__ import annotations

from src.mars100.psychology import PsychState

# Maximum absolute delta any single psych axis can contribute per action.
AXIS_CAP = 0.6

# Total cap across all psych axes per action.
TOTAL_CAP = 1.0

# Critical-work actions that keep minimum weight even under max psych pressure.
CRITICAL_ACTIONS = frozenset({"farm", "terraform", "code"})
CRITICAL_FLOOR_DELTA = -0.3


def _clamp_delta(delta: float, cap: float) -> float:
    return max(-cap, min(cap, delta))


def _add(d: dict[str, float], key: str, val: float) -> None:
    """Accumulate into dict."""
    d[key] = d.get(key, 0.0) + val


def compute_behavior_weights(psych: PsychState) -> dict[str, float]:
    """Compute action-weight deltas from a colonist's psychological state.

    Returns a dict mapping action names to additive weight adjustments.
    Positive = more likely, negative = less likely.  Only includes
    actions whose weights would change (sparse).
    """
    deltas: dict[str, float] = {}

    # --- stress axis ---
    if psych.stress > 0.6:
        excess = psych.stress - 0.6
        _add(deltas, "rest", _clamp_delta(excess * 2.0, AXIS_CAP))
        _add(deltas, "pray", _clamp_delta(excess * 1.2, AXIS_CAP))
        _add(deltas, "explore", _clamp_delta(-excess * 1.5, AXIS_CAP))
        _add(deltas, "research", _clamp_delta(-excess * 1.5, AXIS_CAP))
        _add(deltas, "terraform", _clamp_delta(-excess * 1.0, AXIS_CAP))
    elif psych.stress < 0.2:
        calm = 0.2 - psych.stress
        _add(deltas, "explore", _clamp_delta(calm * 1.0, AXIS_CAP))
        _add(deltas, "research", _clamp_delta(calm * 0.8, AXIS_CAP))

    # --- loneliness axis ---
    if psych.loneliness > 0.5:
        excess = psych.loneliness - 0.5
        _add(deltas, "cooperate", _clamp_delta(excess * 2.0, AXIS_CAP))
        _add(deltas, "mediate", _clamp_delta(excess * 1.5, AXIS_CAP))
        _add(deltas, "hoard", _clamp_delta(-excess * 1.0, AXIS_CAP))
        _add(deltas, "sabotage", _clamp_delta(-excess * 1.2, AXIS_CAP))

    # --- purpose axis ---
    if psych.purpose > 0.7:
        excess = psych.purpose - 0.7
        _add(deltas, "research", _clamp_delta(excess * 2.5, AXIS_CAP))
        _add(deltas, "terraform", _clamp_delta(excess * 1.5, AXIS_CAP))
        _add(deltas, "code", _clamp_delta(excess * 1.5, AXIS_CAP))
        _add(deltas, "rest", _clamp_delta(-excess * 1.0, AXIS_CAP))
    elif psych.purpose < 0.3:
        aimless = 0.3 - psych.purpose
        _add(deltas, "rest", _clamp_delta(aimless * 1.5, AXIS_CAP))
        _add(deltas, "hoard", _clamp_delta(aimless * 0.8, AXIS_CAP))

    # --- morale composite (low morale emergency) ---
    if psych.morale < 0.3:
        deficit = 0.3 - psych.morale
        _add(deltas, "rest", _clamp_delta(deficit * 1.5, AXIS_CAP))
        _add(deltas, "pray", _clamp_delta(deficit * 1.0, AXIS_CAP))
        _add(deltas, "explore", _clamp_delta(-deficit * 1.0, AXIS_CAP))

    # --- apply total cap and critical floors ---
    for action in list(deltas):
        deltas[action] = _clamp_delta(deltas[action], TOTAL_CAP)
        if action in CRITICAL_ACTIONS and deltas[action] < CRITICAL_FLOOR_DELTA:
            deltas[action] = CRITICAL_FLOOR_DELTA

    return deltas


def is_forced_rest(psych: PsychState, year: int) -> bool:
    """Check if a colonist is in forced rest from a prior-year crisis.

    A crisis sets ``forced_rest_until`` to ``crisis_year + 1``.
    On that next year, the colonist MUST rest regardless of weights.
    """
    return (hasattr(psych, 'forced_rest_until')
            and psych.forced_rest_until is not None
            and year <= psych.forced_rest_until)
