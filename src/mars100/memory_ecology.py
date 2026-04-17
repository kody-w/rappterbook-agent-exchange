"""Cultural memory ecology for Mars-100 colony simulation.

Implements colony-wide cultural memory that records, decays, and biases
colonist behaviour across generations. Memories are theme-keyed entries
with salience and valence that decay over time and influence both
individual action choices and collective governance votes.

Constitutional basis: Amendment XIII (Turtles All the Way Down) —
sub-simulations inherit parent constitution, and cultural memory is the
mechanism by which the parent's *experience* transmits, not just its rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ── Constants ────────────────────────────────────────────────────────
MAX_POOL_SIZE: int = 50
SALIENCE_DECAY: float = 0.92
FIDELITY_REINFORCE_PENALTY: float = 0.05
MAX_ACTION_BIAS: float = 0.8
MAX_VOTE_BIAS: float = 0.15
INHERIT_SAMPLE: int = 3
MIN_SALIENCE: float = 0.01


# ── Data ─────────────────────────────────────────────────────────────
@dataclass
class CulturalMemory:
    """A single cultural memory entry in the colony pool."""
    theme: str
    first_year: int
    last_year: int
    salience: float = 1.0
    valence: float = 0.0
    fidelity: float = 1.0
    reinforcements: int = 0

    def to_dict(self) -> dict:
        return {
            "theme": self.theme,
            "first_year": self.first_year,
            "last_year": self.last_year,
            "salience": round(self.salience, 4),
            "valence": round(self.valence, 4),
            "fidelity": round(self.fidelity, 4),
            "reinforcements": self.reinforcements,
        }


class MemoryPool:
    """Colony-wide cultural memory pool.

    Stores theme-keyed memories that decay over time. Provides query
    methods for action/vote biasing and newborn inheritance.
    """

    def __init__(self) -> None:
        self.entries: dict[str, CulturalMemory] = {}

    def record(self, theme: str, year: int, valence: float = 0.0) -> None:
        """Record or reinforce a cultural memory theme."""
        if theme in self.entries:
            mem = self.entries[theme]
            mem.last_year = year
            mem.salience = min(2.0, mem.salience + 0.3)
            mem.valence = (mem.valence * mem.reinforcements + valence) / (mem.reinforcements + 1)
            mem.fidelity = max(0.0, mem.fidelity - FIDELITY_REINFORCE_PENALTY)
            mem.reinforcements += 1
        else:
            if len(self.entries) >= MAX_POOL_SIZE:
                weakest = min(self.entries, key=lambda k: self.entries[k].salience)
                del self.entries[weakest]
            self.entries[theme] = CulturalMemory(
                theme=theme,
                first_year=year,
                last_year=year,
                salience=1.0,
                valence=valence,
                fidelity=1.0,
            )

    def decay(self, current_year: int) -> None:
        """Apply yearly salience decay and prune dead memories."""
        dead: list[str] = []
        for key, mem in self.entries.items():
            mem.salience *= SALIENCE_DECAY
            if mem.salience < MIN_SALIENCE:
                dead.append(key)
        for key in dead:
            del self.entries[key]

    def top(self, n: int = 5) -> list[CulturalMemory]:
        """Return the n most salient memories."""
        return sorted(self.entries.values(), key=lambda m: m.salience, reverse=True)[:n]

    def summary(self) -> dict:
        """Compact summary for year-level results."""
        top = self.top(5)
        return {
            "size": len(self.entries),
            "top_themes": [{"theme": m.theme, "salience": round(m.salience, 3),
                            "valence": round(m.valence, 3)} for m in top],
        }

    def to_dict(self) -> dict:
        """Full serialisation for simulation-level results."""
        return {
            "size": len(self.entries),
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }


# ── Theme mapping ────────────────────────────────────────────────────
_EVENT_THEME_MAP: dict[str, str] = {
    "dust_storm": "natural_disaster",
    "solar_flare": "natural_disaster",
    "meteor_shower": "natural_disaster",
    "equipment_failure": "scarcity",
    "crop_failure": "scarcity",
    "resource_shortage": "scarcity",
    "resource_discovery": "abundance",
    "tech_breakthrough": "abundance",
    "earth_contact": "connection",
    "earth_signal": "connection",
    "alien_signal": "wonder",
    "anomaly": "wonder",
    "disease_outbreak": "plague",
    "radiation_spike": "plague",
    "conflict": "conflict",
    "rebellion": "conflict",
    "birth": "renewal",
    "founding": "renewal",
}


def event_to_theme(event_name: str) -> str | None:
    """Map an event name to a canonical cultural theme."""
    lower = event_name.lower().replace(" ", "_")
    if lower in _EVENT_THEME_MAP:
        return _EVENT_THEME_MAP[lower]
    for fragment, theme in _EVENT_THEME_MAP.items():
        if fragment in lower:
            return theme
    return None


# ── Bias functions ───────────────────────────────────────────────────
_THEME_ACTION_MAP: dict[str, dict[str, float]] = {
    "natural_disaster": {"build": 0.3, "hoard": 0.2, "pray": 0.1},
    "scarcity": {"hoard": 0.4, "build": 0.2, "terraform": -0.1},
    "abundance": {"terraform": 0.3, "explore": 0.2, "hoard": -0.2},
    "connection": {"mediate": 0.2, "explore": 0.1},
    "wonder": {"explore": 0.3, "pray": 0.2},
    "plague": {"build": 0.2, "hoard": 0.3, "explore": -0.2},
    "conflict": {"mediate": 0.3, "sabotage": -0.3, "hoard": 0.1},
    "renewal": {"terraform": 0.2, "build": 0.1, "explore": 0.1},
    "loss": {"pray": 0.2, "mediate": 0.1, "hoard": 0.1},
    "governance_change": {"mediate": 0.2, "build": 0.1},
    "tyranny": {"sabotage": 0.2, "mediate": 0.1, "hoard": 0.1},
}


def memory_action_bias(pool: MemoryPool) -> dict[str, float]:
    """Compute per-action weight biases from cultural memory."""
    biases: dict[str, float] = {}
    for mem in pool.top(5):
        theme_biases = _THEME_ACTION_MAP.get(mem.theme, {})
        for action, raw_bias in theme_biases.items():
            scaled = raw_bias * mem.salience * mem.fidelity
            biases[action] = biases.get(action, 0.0) + scaled
    return {a: max(-MAX_ACTION_BIAS, min(MAX_ACTION_BIAS, v)) for a, v in biases.items()}


_THEME_GOV_MAP: dict[str, dict[str, float]] = {
    "conflict": {"democracy": 0.1, "dictator": -0.1},
    "scarcity": {"council": 0.1, "dictator": 0.05},
    "abundance": {"democracy": 0.05},
    "tyranny": {"dictator": -0.15, "democracy": 0.1},
    "governance_change": {"council": 0.05},
}


def memory_vote_bias(pool: MemoryPool, gov_type: str) -> float:
    """Compute vote bias for a governance proposal type."""
    bias = 0.0
    for mem in pool.top(5):
        theme_gov = _THEME_GOV_MAP.get(mem.theme, {})
        if gov_type in theme_gov:
            bias += theme_gov[gov_type] * mem.salience * mem.fidelity
    return max(-MAX_VOTE_BIAS, min(MAX_VOTE_BIAS, bias))


def inherit_cultural_memory(
    pool: MemoryPool,
    year: int,
    rng: object,
) -> list[dict]:
    """Select cultural themes for a newborn colonist to inherit."""
    candidates = pool.top(10)
    if not candidates:
        return []

    weights = [m.salience for m in candidates]
    total = sum(weights)
    if total <= 0:
        return []

    probs = [w / total for w in weights]
    n = min(INHERIT_SAMPLE, len(candidates))

    selected: list[CulturalMemory] = []
    available = list(zip(candidates, probs))
    for _ in range(n):
        if not available:
            break
        r = rng.random()
        cumulative = 0.0
        for i, (mem, p) in enumerate(available):
            cumulative += p
            if r <= cumulative:
                selected.append(mem)
                available.pop(i)
                remaining_total = sum(p2 for _, p2 in available)
                if remaining_total > 0:
                    available = [(m2, p2 / remaining_total) for m2, p2 in available]
                break

    return [
        {
            "theme": mem.theme,
            "inherited_year": year,
            "original_valence": round(mem.valence, 3),
            "inherited_fidelity": round(mem.fidelity * 0.7, 3),
        }
        for mem in selected
    ]
