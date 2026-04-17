"""
Psychology engine for Mars-100.

Emotional contagion, trauma, resilience, and collective mood.
Each colonist carries a MoodState with 6 emotions that evolve per year
based on events, deaths, births, resource crises, and social contagion.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

EMOTION_NAMES = ("joy", "grief", "fear", "anger", "hope", "despair")

BASELINE: dict[str, float] = {
    "joy": 0.4, "grief": 0.1, "fear": 0.2,
    "anger": 0.1, "hope": 0.5, "despair": 0.1,
}

# How strongly each emotion biases specific actions.
# Positive means "more likely", negative means "less likely".
ACTION_MOOD_BIASES: dict[str, dict[str, float]] = {
    "joy":     {"cooperate": 0.8, "explore": 0.5, "mediate": 0.3},
    "grief":   {"pray": 1.0, "rest": 0.6, "cooperate": -0.4},
    "fear":    {"hoard": 1.0, "rest": 0.5, "explore": -0.6},
    "anger":   {"sabotage": 0.8, "mediate": -0.5, "terraform": 0.3},
    "hope":    {"terraform": 0.7, "cooperate": 0.5, "farm": 0.4},
    "despair": {"rest": 1.0, "terraform": -0.3, "cooperate": -0.3},
}

# Contagion rates: how fast each emotion spreads through trust edges.
CONTAGION_RATE: dict[str, float] = {
    "joy": 0.08, "grief": 0.15, "fear": 0.12,
    "anger": 0.06, "hope": 0.10, "despair": 0.10,
}

MAX_CONTAGION_DELTA = 0.15


@dataclass
class MoodState:
    """Six emotions, each 0.0-1.0."""
    joy: float = 0.4
    grief: float = 0.1
    fear: float = 0.2
    anger: float = 0.1
    hope: float = 0.5
    despair: float = 0.1

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in EMOTION_NAMES}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> MoodState:
        return cls(**{k: d.get(k, BASELINE.get(k, 0.3)) for k in EMOTION_NAMES})

    def clamp(self) -> None:
        """Ensure all emotions are in [0, 1]."""
        for name in EMOTION_NAMES:
            setattr(self, name, max(0.0, min(1.0, getattr(self, name))))

    def dominant(self) -> str:
        """Return the strongest emotion."""
        return max(EMOTION_NAMES, key=lambda n: getattr(self, n))

    def valence(self) -> float:
        """Net emotional valence: positive emotions minus negative."""
        positive = self.joy + self.hope
        negative = self.grief + self.fear + self.anger + self.despair
        return (positive - negative) / 3.0


def compute_resilience(resolve: float, faith: float) -> float:
    """Compute resilience from resolve and faith stats. Always derived, never stored."""
    return (resolve + faith) / 2.0


def compute_mood_shift(
    mood: MoodState,
    events: list[dict[str, Any]],
    deaths: list[dict[str, Any]],
    births: list[dict[str, Any]],
    resource_avg: float,
    relationship_to_dead: dict[str, float],
    rng: random.Random,
) -> None:
    """Shift mood based on this year's events, deaths, births, and resources.

    Mutates *mood* in place.  ``relationship_to_dead`` maps dead-colonist
    IDs to relationship-score (0-1) so grief scales with closeness.
    """
    # Deaths of close ones
    for dead_id, closeness in relationship_to_dead.items():
        grief_impact = 0.15 * closeness
        mood.grief = min(1.0, mood.grief + grief_impact + rng.gauss(0, 0.02))
        mood.despair = min(1.0, mood.despair + grief_impact * 0.5)
        mood.hope = max(0.0, mood.hope - grief_impact * 0.3)

    # Births bring joy
    for _ in births:
        mood.joy = min(1.0, mood.joy + 0.08 + rng.gauss(0, 0.02))
        mood.hope = min(1.0, mood.hope + 0.05)

    # Event-driven shifts
    for ev in events:
        severity = ev.get("severity", 0.3)
        category = ev.get("category", "")
        effects = ev.get("effects", {})
        morale_effect = effects.get("morale", 0.0)
        if morale_effect < 0:
            mood.fear = min(1.0, mood.fear + abs(morale_effect) * 0.3)
            mood.anger = min(1.0, mood.anger + abs(morale_effect) * 0.15)
        elif morale_effect > 0:
            mood.joy = min(1.0, mood.joy + morale_effect * 0.3)
            mood.hope = min(1.0, mood.hope + morale_effect * 0.2)
        if category == "cosmic" and severity > 0.3:
            mood.fear = min(1.0, mood.fear + severity * 0.1)
        if ev.get("name") == "colonist_conflict":
            mood.anger = min(1.0, mood.anger + 0.1)

    # Resource pressure
    if resource_avg < 0.3:
        deficit = 0.3 - resource_avg
        mood.fear = min(1.0, mood.fear + deficit * 0.4)
        mood.despair = min(1.0, mood.despair + deficit * 0.3)
        mood.hope = max(0.0, mood.hope - deficit * 0.2)
    elif resource_avg > 0.6:
        surplus = resource_avg - 0.6
        mood.joy = min(1.0, mood.joy + surplus * 0.2)
        mood.fear = max(0.0, mood.fear - surplus * 0.15)

    mood.clamp()


def decay_toward_baseline(
    mood: MoodState, resilience: float, rng: random.Random,
) -> None:
    """Pull emotions toward baseline. Stronger decay at extremes (>0.7).

    Higher resilience means faster recovery.
    """
    base_rate = 0.10 + resilience * 0.05
    for name in EMOTION_NAMES:
        current = getattr(mood, name)
        target = BASELINE[name]
        diff = target - current
        rate = base_rate
        if abs(current - target) > 0.4:
            rate *= 1.8  # faster pull at extremes
        delta = diff * rate + rng.gauss(0, 0.01)
        setattr(mood, name, max(0.0, min(1.0, current + delta)))


def contagion_spread(
    mood_map: dict[str, MoodState],
    social_edges: dict[str, dict[str, Any]],
    empathy_map: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Spread emotions through the social graph. Snapshot-based.

    Uses pre-contagion mood snapshots so order doesn't matter.
    Returns per-colonist delta dict for testing/logging.
    """
    snapshot: dict[str, dict[str, float]] = {
        cid: mood.to_dict() for cid, mood in mood_map.items()
    }
    deltas: dict[str, dict[str, float]] = {
        cid: {e: 0.0 for e in EMOTION_NAMES} for cid in mood_map
    }

    for receiver_id, mood in mood_map.items():
        receiver_empathy = empathy_map.get(receiver_id, 0.5)
        for sender_id, sender_snap in snapshot.items():
            if sender_id == receiver_id:
                continue
            rel = social_edges.get(sender_id, {}).get(receiver_id)
            if rel is None:
                continue
            trust = rel.get("trust", 0.5) if isinstance(rel, dict) else getattr(rel, "trust", 0.5)
            if trust < 0.3:
                continue
            for emotion in EMOTION_NAMES:
                rate = CONTAGION_RATE[emotion]
                sender_val = sender_snap[emotion]
                receiver_val = getattr(mood, emotion)
                spread = (sender_val - receiver_val) * rate * trust * receiver_empathy
                deltas[receiver_id][emotion] += spread

    # Apply deltas with per-colonist cap
    for cid, delta in deltas.items():
        total_abs = sum(abs(v) for v in delta.values())
        if total_abs > MAX_CONTAGION_DELTA:
            scale = MAX_CONTAGION_DELTA / total_abs
            delta = {k: v * scale for k, v in delta.items()}
            deltas[cid] = delta
        mood = mood_map[cid]
        for emotion, d in delta.items():
            current = getattr(mood, emotion)
            setattr(mood, emotion, max(0.0, min(1.0, current + d)))

    return deltas


def compute_action_bias(mood: MoodState) -> dict[str, float]:
    """Compute action weight biases from current mood.

    Returns a dict mapping action names to additive weight adjustments.
    Only emotions above 0.3 contribute, preventing noise from low emotions.
    """
    biases: dict[str, float] = {}
    for emotion in EMOTION_NAMES:
        level = getattr(mood, emotion)
        if level < 0.3:
            continue
        intensity = level - 0.3  # 0.0 to 0.7
        action_map = ACTION_MOOD_BIASES.get(emotion, {})
        for action, weight in action_map.items():
            biases[action] = biases.get(action, 0.0) + weight * intensity
    return biases


def collective_mood(mood_map: dict[str, MoodState]) -> dict[str, float]:
    """Compute colony-wide average of each emotion.

    Returns dict with emotion averages plus derived metrics:
    - morale: (joy + hope - grief - despair) / 2
    - stability: 1.0 - max(fear, anger)
    """
    if not mood_map:
        return {e: BASELINE[e] for e in EMOTION_NAMES} | {"morale": 0.3, "stability": 0.5}
    averages: dict[str, float] = {}
    n = len(mood_map)
    for emotion in EMOTION_NAMES:
        averages[emotion] = sum(getattr(m, emotion) for m in mood_map.values()) / n
    averages["morale"] = (averages["joy"] + averages["hope"] - averages["grief"] - averages["despair"]) / 2
    averages["stability"] = 1.0 - max(averages["fear"], averages["anger"])
    return averages
