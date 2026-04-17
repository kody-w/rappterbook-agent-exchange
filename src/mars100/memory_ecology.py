"""
Cultural memory ecology for Mars-100.

A bounded colony-wide memory pool that colonists write to and read from.
Memories decay, aggregate by theme, and produce small biases on action
choice and governance voting.  No new actions, no stat mutations.

Pool entries are keyed by **theme** (derived from event names) so duplicate
retellings aggregate rather than append.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

MAX_POOL_SIZE = 50
SALIENCE_DECAY = 0.92
FIDELITY_LOSS_PER_RETELL = 0.05
MYTH_FIDELITY_THRESHOLD = 0.3
MAX_ACTION_BIAS = 0.8
MAX_VOTE_BIAS = 0.15

THEME_ACTION_MAP: dict[str, dict[str, float]] = {
    "famine": {"farm": 0.6, "hoard": 0.3},
    "drought": {"terraform": 0.5, "farm": 0.2},
    "power_crisis": {"code": 0.4, "research": 0.3},
    "conflict": {"mediate": 0.5, "cooperate": 0.3},
    "discovery": {"explore": 0.4, "research": 0.3},
    "plague": {"cooperate": 0.3, "rest": 0.2},
    "cosmic_event": {"pray": 0.3, "explore": 0.2},
    "tyranny": {"mediate": 0.3, "cooperate": 0.2},
    "prosperity": {"explore": 0.2, "terraform": 0.2},
    "loss": {"cooperate": 0.3, "pray": 0.2},
}

THEME_GOV_MAP: dict[str, dict[str, float]] = {
    "tyranny": {"dictator": -0.3, "council": 0.1, "consensus": 0.1},
    "conflict": {"consensus": 0.1, "dictator": -0.1},
    "prosperity": {"council": 0.05},
    "discovery": {"ai_governor": 0.1},
}

EVENT_THEME_MAP: dict[str, str] = {
    "dust_storm": "drought",
    "resource_strike": "discovery",
    "equipment_failure": "power_crisis",
    "earth_contact": "prosperity",
    "alien_signal": "cosmic_event",
    "solar_flare": "cosmic_event",
    "ice_volcano": "discovery",
    "colonist_conflict": "conflict",
    "epidemic": "plague",
    "meteor_strike": "cosmic_event",
    "cave_discovery": "discovery",
    "aurora": "cosmic_event",
}


def event_to_theme(event_name: str) -> str:
    """Map an event name to a canonical cultural theme."""
    return EVENT_THEME_MAP.get(event_name, "general")


@dataclass
class CulturalMemory:
    """A single aggregated cultural memory in the colony pool."""
    theme: str
    salience: float
    fidelity: float
    origin_year: int
    last_reinforced: int
    retell_count: int = 0
    valence: float = 0.0

    @property
    def is_myth(self) -> bool:
        return self.fidelity < MYTH_FIDELITY_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme": self.theme,
            "salience": round(self.salience, 4),
            "fidelity": round(self.fidelity, 4),
            "origin_year": self.origin_year,
            "last_reinforced": self.last_reinforced,
            "retell_count": self.retell_count,
            "valence": round(self.valence, 4),
            "is_myth": self.is_myth,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CulturalMemory:
        return cls(
            theme=d["theme"],
            salience=d.get("salience", 0.5),
            fidelity=d.get("fidelity", 1.0),
            origin_year=d.get("origin_year", 0),
            last_reinforced=d.get("last_reinforced", 0),
            retell_count=d.get("retell_count", 0),
            valence=d.get("valence", 0.0),
        )


@dataclass
class MemoryPool:
    """Colony-wide cultural memory pool.  Bounded, theme-keyed."""
    entries: dict[str, CulturalMemory] = field(default_factory=dict)

    def record(self, theme: str, year: int, valence: float) -> None:
        """Record or reinforce a cultural memory for a theme."""
        if theme in self.entries:
            mem = self.entries[theme]
            mem.salience = min(1.0, mem.salience + 0.1)
            mem.last_reinforced = year
            mem.retell_count += 1
            mem.fidelity = max(0.0, mem.fidelity - FIDELITY_LOSS_PER_RETELL)
            mem.valence = mem.valence * 0.7 + valence * 0.3
        else:
            self.entries[theme] = CulturalMemory(
                theme=theme, salience=0.5, fidelity=1.0,
                origin_year=year, last_reinforced=year,
                retell_count=0, valence=valence,
            )
        self._enforce_cap()

    def decay(self, current_year: int) -> None:
        """Apply yearly salience decay.  Prune entries that fade away."""
        to_remove: list[str] = []
        for theme, mem in self.entries.items():
            mem.salience *= SALIENCE_DECAY
            if mem.salience < 0.01:
                to_remove.append(theme)
        for theme in to_remove:
            del self.entries[theme]

    def _enforce_cap(self) -> None:
        """Keep pool within MAX_POOL_SIZE by dropping lowest-salience entries."""
        while len(self.entries) > MAX_POOL_SIZE:
            weakest = min(self.entries, key=lambda t: self.entries[t].salience)
            del self.entries[weakest]

    def top_memories(self, n: int = 5) -> list[CulturalMemory]:
        """Return the n most salient memories."""
        return sorted(
            self.entries.values(), key=lambda m: m.salience, reverse=True
        )[:n]

    def myths(self) -> list[CulturalMemory]:
        """Return all mythologized memories (fidelity < threshold)."""
        return [m for m in self.entries.values() if m.is_myth]

    def summary(self) -> dict[str, Any]:
        """Return a compact per-year summary for YearResult."""
        top = self.top_memories(5)
        return {
            "pool_size": len(self.entries),
            "myth_count": len(self.myths()),
            "top_themes": [
                {"theme": m.theme, "salience": round(m.salience, 3),
                 "fidelity": round(m.fidelity, 3), "is_myth": m.is_myth}
                for m in top
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {theme: mem.to_dict() for theme, mem in self.entries.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryPool:
        pool = cls()
        for theme, mem_data in d.items():
            pool.entries[theme] = CulturalMemory.from_dict(mem_data)
        return pool


def memory_action_bias(pool: MemoryPool) -> dict[str, float]:
    """Compute action-weight biases from cultural memory.

    Returns a dict of action -> bias.  Capped at MAX_ACTION_BIAS per action.
    Negative-valence memories amplify the bias (trauma drives behaviour).
    """
    biases: dict[str, float] = {}
    for mem in pool.entries.values():
        mapping = THEME_ACTION_MAP.get(mem.theme, {})
        for action, weight in mapping.items():
            nudge = weight * mem.salience * (1.0 if mem.valence < 0 else 0.5)
            biases[action] = biases.get(action, 0.0) + nudge
    return {a: max(-MAX_ACTION_BIAS, min(MAX_ACTION_BIAS, v))
            for a, v in biases.items()}


def memory_vote_bias(pool: MemoryPool, gov_type: str) -> float:
    """Compute governance-vote bias for a given governance type.

    Positive = memory encourages this type, negative = discourages.
    Capped at +/-MAX_VOTE_BIAS.
    """
    bias = 0.0
    for mem in pool.entries.values():
        mapping = THEME_GOV_MAP.get(mem.theme, {})
        gov_nudge = mapping.get(gov_type, 0.0)
        bias += gov_nudge * mem.salience
    return max(-MAX_VOTE_BIAS, min(MAX_VOTE_BIAS, bias))


def inherit_cultural_memory(pool: MemoryPool, year: int,
                            rng: random.Random) -> list[str]:
    """Select themes a newborn inherits from the cultural pool.

    Returns list of theme names (max 3).  Probability proportional to salience.
    """
    if not pool.entries:
        return []
    candidates = pool.top_memories(10)
    inherited: list[str] = []
    for mem in candidates:
        if rng.random() < mem.salience * 0.6:
            inherited.append(mem.theme)
            if len(inherited) >= 3:
                break
    return inherited
