"""
Emergent culture for Mars-100.

Traditions emerge from high-cohesion factions, drift colonist stats slightly
toward faction centroids, and decay without reinforcement.  Effects are
intentionally tiny to avoid contaminating value convergence metrics.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist, STAT_NAMES

TRADITION_EFFECT_MAGNITUDE = 0.005  # tiny per-year stat drift
DECAY_RATE = 0.1  # traditions lose 10% strength per year without reinforcement
MIN_COHESION_FOR_TRADITION = 0.65
MAX_TRADITIONS = 10
MIN_AGE_TO_PARTICIPATE = 5  # children need 5 years before joining traditions

TRADITION_TEMPLATES = [
    ("The Rite of {stat}", "Annual ceremony celebrating {stat} as a colony virtue."),
    ("The {stat} Accord", "A shared commitment to uphold {stat} in all decisions."),
    ("{stat} Vigil", "A silent watch held each year in honor of {stat}."),
    ("The {stat} Council", "Elders gather to discuss the meaning of {stat}."),
    ("Dust Song of {stat}", "A chant passed down through years, invoking {stat}."),
    ("The {stat} Trial", "Young colonists prove their {stat} in a yearly challenge."),
    ("Red Dawn {stat}", "At first light each Mars year, the colony reflects on {stat}."),
]


@dataclass
class Tradition:
    """A cultural tradition that emerged from a faction."""
    id: str
    name: str
    description: str
    faction_origin: str
    founding_year: int
    target_stat: str
    strength: float = 1.0  # 0.0 = dead, 1.0 = vibrant
    participants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "faction_origin": self.faction_origin,
            "founding_year": self.founding_year,
            "target_stat": self.target_stat,
            "strength": round(self.strength, 4),
            "participants": self.participants,
        }

    @property
    def alive(self) -> bool:
        return self.strength > 0.05


def generate_tradition(
    faction_id: str,
    faction_name: str,
    dominant_stat: str,
    year: int,
    rng: random.Random,
) -> Tradition:
    """Generate a new tradition from a faction's dominant stat."""
    template = rng.choice(TRADITION_TEMPLATES)
    pretty_stat = dominant_stat.replace("_", " ").title()
    name = template[0].format(stat=pretty_stat)
    description = template[1].format(stat=pretty_stat)
    tid = f"trad-{faction_id[:12]}-y{year}"
    return Tradition(
        id=tid, name=name, description=description,
        faction_origin=faction_id, founding_year=year,
        target_stat=dominant_stat,
    )


def maybe_create_tradition(
    faction_id: str,
    faction_name: str,
    dominant_stat: str,
    cohesion: float,
    existing_traditions: list[Tradition],
    year: int,
    rng: random.Random,
) -> Tradition | None:
    """Potentially create a new tradition for a high-cohesion faction.

    Probability increases with cohesion. Capped at MAX_TRADITIONS total.
    """
    if len(existing_traditions) >= MAX_TRADITIONS:
        return None
    if cohesion < MIN_COHESION_FOR_TRADITION:
        return None
    # One tradition per faction per stat max
    for t in existing_traditions:
        if t.faction_origin == faction_id and t.target_stat == dominant_stat and t.alive:
            return None
    prob = (cohesion - MIN_COHESION_FOR_TRADITION) * 0.3
    if rng.random() > prob:
        return None
    return generate_tradition(faction_id, faction_name, dominant_stat, year, rng)


def apply_traditions(
    traditions: list[Tradition],
    colonists: list[Colonist],
    faction_members: dict[str, list[str]],
    year: int,
) -> list[dict]:
    """Apply tradition effects: tiny stat drift toward the tradition's stat.

    Returns a list of application records for logging.
    """
    applications: list[dict] = []
    for trad in traditions:
        if not trad.alive:
            continue
        eligible = faction_members.get(trad.faction_origin, [])
        trad.participants = []
        for c in colonists:
            if not c.is_active():
                continue
            if c.id not in eligible:
                continue
            age = year - getattr(c, 'birth_year', 0)
            if age < MIN_AGE_TO_PARTICIPATE:
                continue
            trad.participants.append(c.id)
            current_val = getattr(c.stats, trad.target_stat, 0.5)
            drift = TRADITION_EFFECT_MAGNITUDE * trad.strength
            new_val = min(1.0, max(0.0, current_val + drift))
            setattr(c.stats, trad.target_stat, new_val)
        if trad.participants:
            applications.append({
                "tradition": trad.id, "stat": trad.target_stat,
                "participants": len(trad.participants),
                "strength": round(trad.strength, 4),
            })
    return applications


def decay_traditions(traditions: list[Tradition], year: int) -> list[str]:
    """Decay all traditions by DECAY_RATE. Returns ids of traditions that died."""
    died: list[str] = []
    for trad in traditions:
        if not trad.alive:
            continue
        if not trad.participants:
            trad.strength -= DECAY_RATE * 2  # faster decay without participants
        else:
            trad.strength -= DECAY_RATE * 0.3  # slow decay with participation
        if not trad.alive:
            died.append(trad.id)
    return died


def reinforce_traditions(
    traditions: list[Tradition],
    faction_members: dict[str, list[str]],
) -> None:
    """Reinforce traditions whose factions still exist and have members."""
    for trad in traditions:
        if not trad.alive:
            continue
        members = faction_members.get(trad.faction_origin, [])
        if len(members) >= 2:
            trad.strength = min(1.0, trad.strength + 0.05)
