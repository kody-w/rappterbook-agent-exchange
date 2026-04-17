"""
Pressure system for Mars-100.

Psychological pressure accumulates from environmental, social, and existential
sources. It modifies action selection, death risk, birth probability, and
governance behaviour — closing the feedback loop between colony events and
colonist psychology.

Pressure is always clamped to [0.0, 1.0].
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.mars100.colonist import Colonist, STAT_NAMES
from src.mars100.colony import Resources, SocialGraph, RESOURCE_NAMES
from src.mars100.events import Event


# --- tuning knobs -----------------------------------------------------------

ENV_WEIGHT = 0.4
SOCIAL_WEIGHT = 0.35
EXISTENTIAL_WEIGHT = 0.25

PRESSURE_INERTIA = 0.6
PRESSURE_NEW = 0.4

RELEASE_RATES: dict[str, float] = {
    "mediate": 0.12,
    "pray": 0.10,
    "rest": 0.08,
    "cooperate": 0.06,
    "farm": 0.03,
    "terraform": 0.02,
    "code": 0.02,
    "explore": 0.04,
    "hoard": -0.02,
    "sabotage": -0.05,
}

CRITICAL_PRESSURE = 0.85
STRESS_DEATH_RATE = 0.008


# --- pressure source computation --------------------------------------------

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def compute_environmental_pressure(
    resources: Resources,
    events: list[Event],
    deaths_this_year: int,
    active_count: int,
) -> float:
    """Pressure from resource scarcity, severe events, and witnessed deaths."""
    scarcity = 0.0
    for name in RESOURCE_NAMES:
        val = getattr(resources, name)
        if val < 0.3:
            scarcity += (0.3 - val) / 0.3
    scarcity /= len(RESOURCE_NAMES)

    event_severity = 0.0
    for ev in events:
        event_severity += ev.severity
    event_severity = min(1.0, event_severity)

    death_shock = min(1.0, deaths_this_year * 0.25) if active_count > 0 else 0.0

    return _clamp(scarcity * 0.5 + event_severity * 0.3 + death_shock * 0.2)


def compute_social_pressure(
    colonist: Colonist,
    social: SocialGraph,
    active_ids: list[str],
    gov_type: str,
) -> float:
    """Pressure from low peer trust, governance dissatisfaction, and isolation."""
    if not active_ids or colonist.id not in active_ids:
        return 0.0

    peers = [cid for cid in active_ids if cid != colonist.id]
    if not peers:
        return 0.5

    avg_trust = 0.0
    for peer in peers:
        rel = social.get(peer, colonist.id)
        avg_trust += rel.trust
    avg_trust /= len(peers)
    distrust_pressure = _clamp(1.0 - avg_trust)

    gov_pref = _governance_satisfaction(colonist, gov_type)
    gov_pressure = _clamp(1.0 - gov_pref)

    isolation = _clamp(1.0 - len(peers) / 10.0)

    return _clamp(distrust_pressure * 0.5 + gov_pressure * 0.3 + isolation * 0.2)


def _governance_satisfaction(colonist: Colonist, gov_type: str) -> float:
    """How satisfied a colonist is with the current governance (0-1, 1=happy)."""
    if gov_type == "anarchy":
        return colonist.stats.improvisation * 0.5 + colonist.stats.paranoia * 0.3
    if gov_type == "council":
        return colonist.stats.empathy * 0.5 + colonist.stats.resolve * 0.3
    if gov_type == "dictator":
        return colonist.stats.resolve * 0.5 - colonist.stats.empathy * 0.2
    if gov_type == "consensus":
        return colonist.stats.empathy * 0.4 + colonist.stats.faith * 0.4
    if gov_type == "lottery":
        return colonist.stats.faith * 0.5 + colonist.stats.improvisation * 0.3
    if gov_type == "ai_governor":
        return colonist.skills.coding * 0.4 + colonist.stats.improvisation * 0.3
    if gov_type == "direct_democracy":
        return colonist.stats.empathy * 0.4 + colonist.stats.resolve * 0.3 + 0.2
    return 0.5


def compute_existential_pressure(
    colonist: Colonist,
    year: int,
    meta_events_this_year: int,
) -> float:
    """Pressure from duration of isolation, meta-awareness, and philosophical dread."""
    time_weight = min(1.0, year / 100.0) * 0.3
    meta_weight = min(1.0, meta_events_this_year * 0.3) * 0.4
    faith_buffer = colonist.stats.faith * 0.3
    return _clamp(time_weight + meta_weight - faith_buffer)


# --- pressure update ---------------------------------------------------------

def update_pressure(
    colonist: Colonist,
    env_p: float,
    social_p: float,
    exist_p: float,
) -> float:
    """Blend pressure sources into the colonist's running pressure value.

    Uses inertia so pressure changes are gradual, not jumpy.
    Returns the new pressure value (also written to colonist.pressure).
    """
    raw = (
        ENV_WEIGHT * env_p
        + SOCIAL_WEIGHT * social_p
        + EXISTENTIAL_WEIGHT * exist_p
    )
    new_pressure = PRESSURE_INERTIA * colonist.pressure + PRESSURE_NEW * raw
    new_pressure = _clamp(new_pressure)
    colonist.pressure = new_pressure
    colonist.pressure_history.append(new_pressure)
    if len(colonist.pressure_history) > 10:
        colonist.pressure_history = colonist.pressure_history[-10:]
    return new_pressure


# --- pressure release --------------------------------------------------------

def apply_pressure_release(colonist: Colonist, action: str) -> float:
    """Reduce (or increase) pressure based on the action taken.

    Returns the delta applied.
    """
    delta = RELEASE_RATES.get(action, 0.0)
    resolve_factor = 0.5 + colonist.stats.resolve * 0.5
    delta *= resolve_factor
    colonist.pressure = _clamp(colonist.pressure - delta)
    return -delta


# --- pressure modifiers for engine hooks ------------------------------------

def pressure_action_modifier(
    colonist: Colonist,
    weights: dict[str, float],
) -> dict[str, float]:
    """Modify action-selection weights based on colonist's pressure.

    High pressure biases toward extreme / self-soothing actions;
    low pressure biases toward constructive actions.
    """
    p = colonist.pressure
    modified = dict(weights)
    if p > 0.5:
        excess = p - 0.5
        modified["sabotage"] = modified.get("sabotage", 1.0) * (1.0 + excess * 2.0)
        modified["pray"] = modified.get("pray", 1.0) * (1.0 + excess * 1.5)
        modified["hoard"] = modified.get("hoard", 1.0) * (1.0 + excess * 1.5)
        modified["rest"] = modified.get("rest", 1.0) * (1.0 + excess * 1.0)
        modified["cooperate"] = modified.get("cooperate", 1.0) * max(0.3, 1.0 - excess)
        modified["mediate"] = modified.get("mediate", 1.0) * max(0.3, 1.0 - excess * 0.5)
    elif p < 0.2:
        calm = 0.2 - p
        modified["cooperate"] = modified.get("cooperate", 1.0) * (1.0 + calm * 2.0)
        modified["explore"] = modified.get("explore", 1.0) * (1.0 + calm * 1.5)
        modified["terraform"] = modified.get("terraform", 1.0) * (1.0 + calm * 1.0)
    return modified


def pressure_death_modifier(colonist: Colonist) -> float:
    """Additional death-rate contribution from extreme stress.

    Only kicks in above CRITICAL_PRESSURE.
    """
    if colonist.pressure > CRITICAL_PRESSURE:
        return STRESS_DEATH_RATE * (colonist.pressure - CRITICAL_PRESSURE) / (1.0 - CRITICAL_PRESSURE)
    return 0.0


def pressure_birth_modifier(colonists: list[Colonist]) -> float:
    """Multiplier on birth probability based on collective pressure.

    Returns 1.0 at zero pressure, down to 0.2 at max pressure.
    """
    active = [c for c in colonists if c.is_active()]
    if not active:
        return 1.0
    avg = sum(c.pressure for c in active) / len(active)
    return max(0.2, 1.0 - avg * 0.8)


def collective_pressure(colonists: list[Colonist]) -> float:
    """Colony-wide average pressure among active colonists."""
    active = [c for c in colonists if c.is_active()]
    if not active:
        return 0.0
    return sum(c.pressure for c in active) / len(active)


# --- summary for year results -----------------------------------------------

@dataclass
class PressureSnapshot:
    """Year-end pressure state for logging."""
    collective: float
    environmental: float
    social_avg: float
    existential_avg: float
    individual: dict[str, float] = field(default_factory=dict)
    high_pressure_colonists: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "collective": round(self.collective, 4),
            "environmental": round(self.environmental, 4),
            "social_avg": round(self.social_avg, 4),
            "existential_avg": round(self.existential_avg, 4),
            "individual": {k: round(v, 4) for k, v in self.individual.items()},
            "high_pressure_colonists": self.high_pressure_colonists,
        }
