"""
Mars-100 behavior organ — psychology-driven action perturbation.

Translates psychological state (stress, morale, purpose) into action-weight
modifiers, propagates stress/morale through social trust networks, and tracks
learned action preferences from resource outcomes.

Engine v9.0.  Deterministic — no separate RNG stream needed.

Deferred from psychology v8.0 (line 14: "Defer: action-selection perturbation
(v9+)").  This organ completes that design intent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- constants ---

# Perturbation caps
STRESS_WEIGHT_CAP: float = 0.3
MORALE_WEIGHT_CAP: float = 0.2
PURPOSE_WEIGHT_CAP: float = 0.15
LEARNED_WEIGHT_CAP: float = 0.25

# Social contagion
STRESS_CONTAGION_CAP: float = 0.05
LONELINESS_CONTAGION_CAP: float = 0.03
PURPOSE_CONTAGION_CAP: float = 0.04
TRUST_THRESHOLD: float = 0.3

# Learning
LEARNING_RATE: float = 0.05
DECAY_RATE: float = 0.01
PREF_CAP: float = 1.0
SMALL_PREF_THRESHOLD: float = 0.005

# Risk tolerance
RISK_BASE: float = 0.5
RISK_STRESS_FACTOR: float = -0.3
RISK_PURPOSE_FACTOR: float = 0.2
RISK_MORALE_FACTOR: float = 0.1

# Action categories for perturbation
STRESS_BOOST_ACTIONS = frozenset({"pray", "rest", "hoard"})
STRESS_REDUCE_ACTIONS = frozenset({"terraform", "explore", "research"})
MORALE_BOOST_ACTIONS = frozenset({"cooperate", "mediate", "explore"})
PURPOSE_BOOST_ACTIONS = frozenset({"terraform", "research", "code"})

# Action → resource mapping for learned preferences
ACTION_RESOURCE_MAP: dict[str, list[str]] = {
    "farm": ["food"],
    "terraform": ["water", "air"],
    "code": ["power"],
    "research": ["power"],
    "cooperate": ["food", "water"],
    "explore": ["water"],
}


# --- data classes ---

@dataclass
class BehaviorProfile:
    """Per-colonist learned behavior profile."""
    action_preferences: dict[str, float] = field(default_factory=dict)
    total_actions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_preferences": dict(self.action_preferences),
            "total_actions": self.total_actions,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BehaviorProfile:
        return cls(
            action_preferences=dict(d.get("action_preferences", {})),
            total_actions=d.get("total_actions", 0),
        )


@dataclass
class ContagionDelta:
    """Social contagion result for one colonist."""
    colonist_id: str
    stress_delta: float = 0.0
    loneliness_delta: float = 0.0
    purpose_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "colonist_id": self.colonist_id,
            "stress_delta": round(self.stress_delta, 6),
            "loneliness_delta": round(self.loneliness_delta, 6),
            "purpose_delta": round(self.purpose_delta, 6),
        }


@dataclass
class BehaviorTickResult:
    """Result of one year's behavior processing."""
    contagion: list[dict] = field(default_factory=list)
    perturbations: dict[str, dict[str, float]] = field(default_factory=dict)
    learned_updates: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contagion": self.contagion,
            "perturbations": self.perturbations,
            "learned_updates": self.learned_updates,
        }


# --- pure functions ---

def compute_action_perturbation(
    stress: float,
    morale: float,
    purpose: float,
    profile: BehaviorProfile,
    actions: list[str],
) -> dict[str, float]:
    """Compute action-weight deltas from psychology + learned preferences.

    Returns dict mapping action name → additive weight delta.
    Positive = more likely, negative = less likely.
    """
    deltas: dict[str, float] = {}
    for action in actions:
        d = 0.0
        # Stress: high stress boosts comfort actions, penalizes ambitious ones
        if action in STRESS_BOOST_ACTIONS:
            d += stress * STRESS_WEIGHT_CAP
        elif action in STRESS_REDUCE_ACTIONS:
            d -= stress * STRESS_WEIGHT_CAP * 0.5
        # Morale: high morale boosts cooperative/exploration actions
        if action in MORALE_BOOST_ACTIONS:
            d += (morale - 0.5) * MORALE_WEIGHT_CAP
        # Purpose: high purpose boosts goal-oriented actions
        if action in PURPOSE_BOOST_ACTIONS:
            d += (purpose - 0.5) * PURPOSE_WEIGHT_CAP
        # Learned preferences (capped)
        pref = profile.action_preferences.get(action, 0.0)
        d += max(-LEARNED_WEIGHT_CAP, min(LEARNED_WEIGHT_CAP, pref))
        deltas[action] = d
    return deltas


def compute_social_contagion(
    colonist_id: str,
    psych_snapshot: dict[str, dict[str, Any]],
    trust_pairs: list[tuple[str, float]],
) -> ContagionDelta:
    """Compute stress/loneliness/purpose contagion from trusted neighbors.

    Uses frozen psych snapshot for simultaneous update (no order bias).
    """
    if not trust_pairs or colonist_id not in psych_snapshot:
        return ContagionDelta(colonist_id=colonist_id)

    my_psych = psych_snapshot[colonist_id]
    my_stress = my_psych.get("stress", 0.0)
    my_loneliness = my_psych.get("loneliness", 0.0)
    my_purpose = my_psych.get("purpose", 0.5)

    stress_sum = 0.0
    loneliness_sum = 0.0
    purpose_sum = 0.0
    weight_sum = 0.0

    for other_id, trust in trust_pairs:
        if trust < TRUST_THRESHOLD:
            continue
        other = psych_snapshot.get(other_id)
        if other is None:
            continue
        w = trust - TRUST_THRESHOLD
        stress_sum += w * (other.get("stress", 0.0) - my_stress)
        loneliness_sum += w * (other.get("loneliness", 0.0) - my_loneliness)
        purpose_sum += w * (other.get("purpose", 0.5) - my_purpose)
        weight_sum += w

    if weight_sum < 1e-9:
        return ContagionDelta(colonist_id=colonist_id)

    stress_d = max(-STRESS_CONTAGION_CAP,
                   min(STRESS_CONTAGION_CAP, stress_sum / weight_sum * 0.3))
    loneliness_d = max(-LONELINESS_CONTAGION_CAP,
                       min(LONELINESS_CONTAGION_CAP,
                           loneliness_sum / weight_sum * 0.3))
    purpose_d = max(-PURPOSE_CONTAGION_CAP,
                    min(PURPOSE_CONTAGION_CAP,
                        purpose_sum / weight_sum * 0.3))

    return ContagionDelta(
        colonist_id=colonist_id,
        stress_delta=stress_d,
        loneliness_delta=loneliness_d,
        purpose_delta=purpose_d,
    )


def update_learned_preferences(
    profile: BehaviorProfile,
    action_taken: str,
    resource_delta: dict[str, float],
) -> dict[str, float]:
    """Update learned action preferences based on action-linked resource outcomes.

    Returns the updated preference dict.
    """
    profile.total_actions += 1

    linked_resources = ACTION_RESOURCE_MAP.get(action_taken, [])
    reward = sum(resource_delta.get(r, 0.0) for r in linked_resources)

    old_pref = profile.action_preferences.get(action_taken, 0.0)
    new_pref = max(-PREF_CAP, min(PREF_CAP, old_pref + LEARNING_RATE * reward))
    profile.action_preferences[action_taken] = new_pref

    # Decay all other preferences toward zero
    to_remove = []
    for act, pref in profile.action_preferences.items():
        if act == action_taken:
            continue
        decayed = pref * (1.0 - DECAY_RATE)
        if abs(decayed) < SMALL_PREF_THRESHOLD:
            to_remove.append(act)
        else:
            profile.action_preferences[act] = decayed
    for act in to_remove:
        del profile.action_preferences[act]

    return dict(profile.action_preferences)


def compute_risk_tolerance(
    stress: float, morale: float, purpose: float,
) -> float:
    """Compute colonist risk tolerance from psychological state. Returns [0, 1]."""
    raw = (RISK_BASE
           + stress * RISK_STRESS_FACTOR
           + purpose * RISK_PURPOSE_FACTOR
           + morale * RISK_MORALE_FACTOR)
    return max(0.0, min(1.0, raw))
