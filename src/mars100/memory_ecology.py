"""
Cultural memory ecology for Mars-100.

A bounded colony-wide memory pool that influences colonist action choices
and governance voting.  Memories decay, lose fidelity, and eventually
mythologize — becoming powerful but inaccurate cultural narratives.

Key design decisions (from rubber-duck critique):
  - Aggregate by THEME, not per-retelling — prevents duplicate explosion
  - Read PREVIOUS year's pool to avoid double-counting current events
  - Two narrow hooks: memory_action_bias() and memory_vote_bias()
  - Cap salience growth per theme to prevent runaway feedback loops
  - Structured theme tags, not free-text matching
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Constants ────────────────────────────────────────────────────────
MAX_POOL_SIZE = 50
SALIENCE_DECAY = 0.92          # per-year multiplier
FIDELITY_DECAY = 0.97          # per-year multiplier
MYTH_FIDELITY_THRESHOLD = 0.3  # below this → mythologized
SALIENCE_FLOOR = 0.05          # pruned below this
MAX_SALIENCE = 3.0             # cap to prevent runaway
INHERIT_FIDELITY_PENALTY = 0.85  # children inherit at reduced fidelity


# ── Theme mapping ────────────────────────────────────────────────────

THEME_MAP: dict[str, str] = {
    # Event categories → themes
    "dust_storm": "survival:environment",
    "solar_flare": "survival:environment",
    "equipment_failure": "survival:infrastructure",
    "meteor_strike": "survival:environment",
    "resource_strike": "prosperity:discovery",
    "earth_contact": "connection:earth",
    "alien_signal": "mystery:cosmic",
    "colonist_conflict": "social:conflict",
    "birth": "growth:birth",
    "aurora": "mystery:cosmic",
    "calm": "stability:peace",
    # Governance outcomes → themes
    "governance_passed": "governance:change",
    "governance_failed": "governance:stability",
    # Death/exile → themes
    "death": "loss:death",
    "exile": "social:exile",
    # Actions → themes
    "sabotage": "social:conflict",
    "cooperate": "social:cooperation",
    "pray": "spiritual:faith",
    "explore": "discovery:exploration",
}

DEFAULT_THEME = "misc:unknown"


def event_to_theme(event_name: str) -> str:
    """Map an event/action name to a structured theme tag."""
    return THEME_MAP.get(event_name, DEFAULT_THEME)


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class CulturalMemory:
    """A single cultural memory in the colony pool."""
    theme: str
    year_formed: int
    salience: float         # how prominent (0–MAX_SALIENCE)
    fidelity: float         # how accurate (0–1, decays toward myth)
    event_count: int = 1    # how many events aggregated into this theme
    mythologized: bool = False

    def decay(self) -> None:
        """Apply one year of decay."""
        self.salience *= SALIENCE_DECAY
        self.fidelity *= FIDELITY_DECAY
        if self.fidelity < MYTH_FIDELITY_THRESHOLD and not self.mythologized:
            self.mythologized = True

    def reinforce(self, amount: float = 0.3) -> None:
        """Reinforce this memory (same theme recurred)."""
        self.salience = min(MAX_SALIENCE, self.salience + amount)
        self.event_count += 1

    def is_prunable(self) -> bool:
        """Below salience floor → ready for removal."""
        return self.salience < SALIENCE_FLOOR

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "theme": self.theme,
            "year_formed": self.year_formed,
            "salience": round(self.salience, 4),
            "fidelity": round(self.fidelity, 4),
            "event_count": self.event_count,
            "mythologized": self.mythologized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CulturalMemory:
        """Deserialize from JSON."""
        return cls(
            theme=data["theme"],
            year_formed=data["year_formed"],
            salience=data["salience"],
            fidelity=data["fidelity"],
            event_count=data.get("event_count", 1),
            mythologized=data.get("mythologized", False),
        )


class MemoryPool:
    """Bounded colony-wide cultural memory pool.

    Memories are keyed by theme.  New events reinforce existing themes
    or create new entries.  Each year: decay → prune → cap.
    """

    def __init__(self) -> None:
        self.memories: dict[str, CulturalMemory] = {}

    def record(self, event_name: str, year: int, salience: float = 1.0) -> None:
        """Record an event into the memory pool."""
        theme = event_to_theme(event_name)
        if theme in self.memories:
            self.memories[theme].reinforce(salience * 0.3)
        else:
            self.memories[theme] = CulturalMemory(
                theme=theme,
                year_formed=year,
                salience=min(MAX_SALIENCE, salience),
                fidelity=1.0,
            )

    def tick(self) -> None:
        """Advance one year: decay all, prune dead, enforce cap."""
        for mem in self.memories.values():
            mem.decay()
        # Prune
        self.memories = {
            k: v for k, v in self.memories.items()
            if not v.is_prunable()
        }
        # Cap pool size — keep highest salience
        if len(self.memories) > MAX_POOL_SIZE:
            ranked = sorted(self.memories.items(),
                            key=lambda kv: kv[1].salience, reverse=True)
            self.memories = dict(ranked[:MAX_POOL_SIZE])

    def top_themes(self, n: int = 5) -> list[CulturalMemory]:
        """Return the N most salient memories."""
        ranked = sorted(self.memories.values(),
                        key=lambda m: m.salience, reverse=True)
        return ranked[:n]

    def get_theme(self, theme: str) -> CulturalMemory | None:
        """Look up a specific theme."""
        return self.memories.get(theme)

    def summary(self) -> dict:
        """Compact summary for year output."""
        return {
            "pool_size": len(self.memories),
            "top_themes": [m.to_dict() for m in self.top_themes(5)],
            "myth_count": sum(1 for m in self.memories.values() if m.mythologized),
        }

    def full_state(self) -> list[dict]:
        """Full serialization for final output."""
        return [m.to_dict() for m in self.memories.values()]

    def load_state(self, data: list[dict]) -> None:
        """Restore from serialized state."""
        self.memories = {}
        for entry in data:
            mem = CulturalMemory.from_dict(entry)
            self.memories[mem.theme] = mem


# ── Bias helpers (the two narrow hooks) ──────────────────────────────

# Theme → action affinity.  Positive = boost, negative = suppress.
THEME_ACTION_AFFINITY: dict[str, dict[str, float]] = {
    "survival:environment": {"terraform": 0.4, "pray": 0.2, "farm": 0.2},
    "survival:infrastructure": {"code": 0.4, "research": 0.3},
    "prosperity:discovery": {"explore": 0.3, "research": 0.2},
    "connection:earth": {"code": 0.2, "pray": 0.2},
    "mystery:cosmic": {"pray": 0.3, "explore": 0.2},
    "social:conflict": {"mediate": 0.4, "sabotage": -0.2},
    "growth:birth": {"farm": 0.2, "cooperate": 0.2},
    "stability:peace": {"cooperate": 0.1},
    "governance:change": {"mediate": 0.2},
    "governance:stability": {"cooperate": 0.1},
    "loss:death": {"farm": 0.2, "pray": 0.2, "rest": 0.1},
    "social:exile": {"mediate": 0.2, "cooperate": 0.1},
    "social:cooperation": {"cooperate": 0.2},
    "spiritual:faith": {"pray": 0.3},
    "discovery:exploration": {"explore": 0.3, "research": 0.2},
}

# Theme → governance affinity (positive = bias toward YES).
THEME_GOV_AFFINITY: dict[str, dict[str, float]] = {
    "social:conflict": {"council": 0.2, "dictator": 0.15},
    "survival:environment": {"council": 0.1, "consensus": 0.15},
    "governance:change": {"lottery": 0.1, "ai_governor": 0.1},
    "governance:stability": {"consensus": 0.15},
    "loss:death": {"council": 0.1, "consensus": 0.1},
    "mystery:cosmic": {"ai_governor": 0.15, "lottery": 0.1},
    "spiritual:faith": {"consensus": 0.15, "lottery": 0.1},
}


def memory_action_bias(pool: MemoryPool, action: str) -> float:
    """Compute a bias weight for an action based on cultural memory.

    Returns a float (can be negative) to ADD to the action's weight
    in the engine's _choose_action weighted selection.  Capped to ±0.5.
    """
    bias = 0.0
    for mem in pool.top_themes(5):
        affinities = THEME_ACTION_AFFINITY.get(mem.theme, {})
        raw = affinities.get(action, 0.0)
        # Mythologized memories have amplified but less precise influence
        mult = 1.5 if mem.mythologized else 1.0
        bias += raw * mem.salience * mult
    return max(-0.5, min(0.5, bias))


def memory_vote_bias(pool: MemoryPool, gov_type: str) -> float:
    """Compute a bias for a governance vote based on cultural memory.

    Returns a float to ADD to the vote score.  Capped to ±0.3.
    """
    bias = 0.0
    for mem in pool.top_themes(5):
        affinities = THEME_GOV_AFFINITY.get(mem.theme, {})
        raw = affinities.get(gov_type, 0.0)
        mult = 1.3 if mem.mythologized else 1.0
        bias += raw * mem.salience * mult
    return max(-0.3, min(0.3, bias))


def inherit_cultural_memory(parent_pool: MemoryPool) -> MemoryPool:
    """Create a new pool for a child colonist's generation.

    Children inherit the top themes at reduced fidelity.
    """
    child_pool = MemoryPool()
    for mem in parent_pool.top_themes(10):
        child_mem = CulturalMemory(
            theme=mem.theme,
            year_formed=mem.year_formed,
            salience=mem.salience * 0.5,
            fidelity=mem.fidelity * INHERIT_FIDELITY_PENALTY,
            event_count=mem.event_count,
            mythologized=mem.mythologized or mem.fidelity * INHERIT_FIDELITY_PENALTY < MYTH_FIDELITY_THRESHOLD,
        )
        child_pool.memories[mem.theme] = child_mem
    return child_pool
