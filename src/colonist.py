"""
colonist.py — Individual Mars colonist model for Mars-100.

Each colonist has stats, skills, an element, relationships, and memory.
Colonists are the atoms of the simulation — their interactions produce
emergent governance, alliances, and crises.

Stats and skills are 0.0–1.0. Relationships are -1.0 to 1.0 trust values.
Memory is a capped list of significant events.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# Stat names
STATS = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")
# Skill names
SKILLS = ("terraforming", "hydroponics", "mediation", "coding", "prayer", "sabotage")
# Elements
ELEMENTS = ("fire", "water", "earth", "air")

MEMORY_CAP = 50

# Colonist archetypes — initial stat biases by element
ELEMENT_BIAS: dict[str, dict[str, float]] = {
    "fire":  {"resolve": 0.2, "improvisation": 0.1, "paranoia": 0.1},
    "water": {"empathy": 0.2, "improvisation": 0.15, "faith": 0.05},
    "earth": {"resolve": 0.15, "hoarding": 0.15, "faith": 0.1},
    "air":   {"improvisation": 0.2, "empathy": 0.1, "paranoia": -0.1},
}


@dataclass
class Colonist:
    """A Mars-100 colonist."""
    id: str
    name: str
    element: str
    stats: dict[str, float]
    skills: dict[str, float]
    relationships: dict[str, float]  # other_id → trust (-1.0 to 1.0)
    memory: list[dict]
    alive: bool = True
    year_joined: int = 1
    year_died: int | None = None
    cause_of_death: str | None = None
    leadership_score: float = 0.0
    proposals_made: int = 0
    proposals_passed: int = 0
    subsims_run: int = 0
    times_exiled: int = 0

    def stat(self, name: str) -> float:
        """Get a stat value, clamped 0–1."""
        return max(0.0, min(1.0, self.stats.get(name, 0.5)))

    def skill(self, name: str) -> float:
        """Get a skill value, clamped 0–1."""
        return max(0.0, min(1.0, self.skills.get(name, 0.0)))

    def trust(self, other_id: str) -> float:
        """Get trust toward another colonist."""
        return max(-1.0, min(1.0, self.relationships.get(other_id, 0.0)))

    def add_memory(self, year: int, event: str, significance: float = 0.5) -> None:
        """Record a memory, evicting least significant if at cap."""
        entry = {"year": year, "event": event, "significance": significance}
        self.memory.append(entry)
        if len(self.memory) > MEMORY_CAP:
            self.memory.sort(key=lambda m: m["significance"])
            self.memory.pop(0)

    def adjust_trust(self, other_id: str, delta: float) -> None:
        """Adjust trust toward another colonist."""
        current = self.relationships.get(other_id, 0.0)
        self.relationships[other_id] = max(-1.0, min(1.0, current + delta))

    def adjust_stat(self, name: str, delta: float) -> None:
        """Adjust a stat, clamped to 0–1."""
        current = self.stats.get(name, 0.5)
        self.stats[name] = max(0.0, min(1.0, current + delta))

    def adjust_skill(self, name: str, delta: float) -> None:
        """Adjust a skill through practice, clamped to 0–1."""
        current = self.skills.get(name, 0.0)
        self.skills[name] = max(0.0, min(1.0, current + delta))

    def die(self, year: int, cause: str) -> None:
        """Mark colonist as dead — legacy, not delete."""
        self.alive = False
        self.year_died = year
        self.cause_of_death = cause
        self.add_memory(year, f"died: {cause}", significance=1.0)

    def effectiveness(self) -> float:
        """Overall effectiveness: weighted combination of resolve + skills."""
        skill_avg = sum(self.skills.values()) / max(len(self.skills), 1)
        return (self.stat("resolve") * 0.4 + skill_avg * 0.4 +
                self.stat("improvisation") * 0.2)

    def cooperation_tendency(self) -> float:
        """How cooperative this colonist is (empathy + faith - paranoia - hoarding)."""
        return max(0.0, min(1.0,
            (self.stat("empathy") + self.stat("faith") -
             self.stat("paranoia") * 0.5 - self.stat("hoarding") * 0.3) / 1.5
        ))

    def discovery_potential(self, year: int) -> float:
        """Chance of meta-insight (realizing they're in a simulation)."""
        base = self.stat("paranoia") * 0.3 + self.stat("improvisation") * 0.2
        time_factor = min(1.0, year / 80.0)  # increases over time
        subsim_exposure = min(1.0, self.subsims_run / 5.0)
        return min(1.0, base + time_factor * 0.3 + subsim_exposure * 0.2)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "id": self.id,
            "name": self.name,
            "element": self.element,
            "stats": {k: round(v, 4) for k, v in self.stats.items()},
            "skills": {k: round(v, 4) for k, v in self.skills.items()},
            "relationships": {k: round(v, 4) for k, v in self.relationships.items()},
            "memory": self.memory[-20:],  # compact: last 20 for output
            "alive": self.alive,
            "year_joined": self.year_joined,
            "year_died": self.year_died,
            "cause_of_death": self.cause_of_death,
            "leadership_score": round(self.leadership_score, 4),
            "proposals_made": self.proposals_made,
            "proposals_passed": self.proposals_passed,
            "subsims_run": self.subsims_run,
            "times_exiled": self.times_exiled,
        }

    def to_view(self) -> dict:
        """Compact projection for LisPy consumption (read-only view)."""
        return {
            "id": self.id,
            "element": self.element,
            "resolve": self.stat("resolve"),
            "empathy": self.stat("empathy"),
            "paranoia": self.stat("paranoia"),
            "faith": self.stat("faith"),
            "hoarding": self.stat("hoarding"),
            "improvisation": self.stat("improvisation"),
            "alive": self.alive,
            "leadership": self.leadership_score,
            "cooperation": self.cooperation_tendency(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Colonist:
        """Deserialize from dict with validation/clamping."""
        stats = {k: max(0.0, min(1.0, float(v)))
                 for k, v in data.get("stats", {}).items()}
        skills = {k: max(0.0, min(1.0, float(v)))
                  for k, v in data.get("skills", {}).items()}
        rels = {k: max(-1.0, min(1.0, float(v)))
                for k, v in data.get("relationships", {}).items()}
        return cls(
            id=data["id"],
            name=data["name"],
            element=data.get("element", "earth"),
            stats=stats,
            skills=skills,
            relationships=rels,
            memory=data.get("memory", []),
            alive=data.get("alive", True),
            year_joined=data.get("year_joined", 1),
            year_died=data.get("year_died"),
            cause_of_death=data.get("cause_of_death"),
            leadership_score=data.get("leadership_score", 0.0),
            proposals_made=data.get("proposals_made", 0),
            proposals_passed=data.get("proposals_passed", 0),
            subsims_run=data.get("subsims_run", 0),
            times_exiled=data.get("times_exiled", 0),
        )


# ---------------------------------------------------------------------------
# Colony roster creation
# ---------------------------------------------------------------------------

COLONIST_TEMPLATES = [
    ("aria",    "Aria Vasquez",     "fire",  {"resolve": 0.8, "empathy": 0.6, "faith": 0.3}, {"mediation": 0.7, "terraforming": 0.5}),
    ("kael",    "Kael Okonkwo",     "earth", {"resolve": 0.7, "hoarding": 0.6, "paranoia": 0.4}, {"hydroponics": 0.8, "coding": 0.3}),
    ("lyra",    "Lyra Chen",        "water", {"empathy": 0.9, "improvisation": 0.7, "faith": 0.5}, {"mediation": 0.9, "prayer": 0.4}),
    ("thresh",  "Thresh Andersson", "fire",  {"resolve": 0.9, "paranoia": 0.3, "improvisation": 0.5}, {"terraforming": 0.9, "sabotage": 0.1}),
    ("nyx",     "Nyx Patel",        "air",   {"improvisation": 0.8, "paranoia": 0.6, "empathy": 0.3}, {"coding": 0.9, "sabotage": 0.3}),
    ("sol",     "Sol Reeves",       "earth", {"faith": 0.8, "resolve": 0.6, "hoarding": 0.5}, {"prayer": 0.7, "hydroponics": 0.6}),
    ("vega",    "Vega Moreau",      "water", {"empathy": 0.7, "improvisation": 0.6, "resolve": 0.5}, {"mediation": 0.5, "terraforming": 0.6}),
    ("rook",    "Rook Tanaka",      "air",   {"paranoia": 0.8, "improvisation": 0.9, "hoarding": 0.7}, {"coding": 0.7, "sabotage": 0.5}),
    ("ember",   "Ember Osei",       "fire",  {"resolve": 0.6, "faith": 0.7, "empathy": 0.5}, {"terraforming": 0.4, "prayer": 0.6}),
    ("frost",   "Frost Nakamura",   "water", {"empathy": 0.5, "paranoia": 0.5, "hoarding": 0.4}, {"hydroponics": 0.7, "coding": 0.5}),
]


def create_colony(seed: int = 42) -> list[Colonist]:
    """Create the 10 founding colonists with seeded randomization."""
    rng = random.Random(seed)
    colonists: list[Colonist] = []

    for cid, name, element, stat_base, skill_base in COLONIST_TEMPLATES:
        # Fill in missing stats with random values
        stats = {}
        for s in STATS:
            base = stat_base.get(s, rng.uniform(0.2, 0.6))
            bias = ELEMENT_BIAS.get(element, {}).get(s, 0.0)
            noise = rng.gauss(0, 0.05)
            stats[s] = max(0.0, min(1.0, base + bias + noise))

        skills = {}
        for s in SKILLS:
            base = skill_base.get(s, rng.uniform(0.0, 0.3))
            noise = rng.gauss(0, 0.05)
            skills[s] = max(0.0, min(1.0, base + noise))

        # Initial relationships: slight random trust
        relationships: dict[str, float] = {}
        for other_cid, _, _, _, _ in COLONIST_TEMPLATES:
            if other_cid != cid:
                relationships[other_cid] = rng.uniform(-0.1, 0.3)

        colonists.append(Colonist(
            id=cid, name=name, element=element,
            stats=stats, skills=skills,
            relationships=relationships,
            memory=[{"year": 0, "event": "launched from Earth", "significance": 0.8}],
        ))

    return colonists
