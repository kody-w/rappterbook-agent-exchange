"""
Colonist model for Mars-100.

10 founding colonists with elements, stats, skills, and LisPy decision
expressions. Each colonist is a data structure AND a LisPy program —
the homoiconic property means they can rewrite themselves.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


ELEMENTS = ("fire", "water", "earth", "air")
STAT_NAMES = ("resolve", "improvisation", "empathy",
              "hoarding", "faith", "paranoia")
SKILL_NAMES = ("terraforming", "hydroponics", "mediation",
               "coding", "prayer", "sabotage", "medicine",
               "engineering", "exploration")
ARCHETYPES = ("commander", "nurturer", "builder", "dreamer",
              "guardian", "healer", "maker", "diplomat",
              "warrior", "philosopher")


@dataclass
class Colonist:
    """A Mars-100 colonist."""
    id: str
    name: str
    element: str
    archetype: str
    stats: dict
    skills: list
    decision_expr: str
    alive: bool = True
    exiled: bool = False
    morale: float = 0.7
    health: float = 1.0
    memory: list = field(default_factory=list)
    death_year: int = -1
    death_cause: str = ""
    exile_year: int = -1
    meta_awareness: float = 0.0
    subsims_spawned: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id, "name": self.name, "element": self.element,
            "archetype": self.archetype, "stats": self.stats,
            "skills": self.skills, "decision_expr": self.decision_expr,
            "alive": self.alive, "exiled": self.exiled,
            "morale": self.morale, "health": self.health,
            "memory": self.memory[-20:],
            "death_year": self.death_year, "death_cause": self.death_cause,
            "exile_year": self.exile_year,
            "meta_awareness": round(self.meta_awareness, 4),
            "subsims_spawned": self.subsims_spawned,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Colonist":
        """Deserialize from dictionary."""
        return cls(
            id=d["id"], name=d["name"], element=d["element"],
            archetype=d["archetype"], stats=d["stats"],
            skills=d["skills"],
            decision_expr=d.get("decision_expr", "(+ 0 0)"),
            alive=d.get("alive", True), exiled=d.get("exiled", False),
            morale=d.get("morale", 0.7), health=d.get("health", 1.0),
            memory=d.get("memory", []),
            death_year=d.get("death_year", -1),
            death_cause=d.get("death_cause", ""),
            exile_year=d.get("exile_year", -1),
            meta_awareness=d.get("meta_awareness", 0.0),
            subsims_spawned=d.get("subsims_spawned", 0),
        )

    def add_memory(self, year: int, event: str) -> None:
        """Add a memory entry, capped at 50."""
        self.memory.append({"year": year, "event": event})
        if len(self.memory) > 50:
            self.memory = self.memory[-50:]

    def is_active(self) -> bool:
        """True if alive and not exiled."""
        return self.alive and not self.exiled


# The 10 founding colonists — each with a unique LisPy decision expression
FOUNDING_COLONISTS: list = [
    {
        "id": "kira-sol",
        "name": "Kira Sol",
        "element": "fire",
        "archetype": "commander",
        "stats": {"resolve": 0.9, "improvisation": 0.5, "empathy": 0.4,
                  "hoarding": 0.3, "faith": 0.3, "paranoia": 0.2},
        "skills": ["terraforming", "engineering", "exploration"],
        "decision_expr": (
            "(if (< food 0.3) (quote terraform) "
            "  (if (< power 0.3) (quote engineer) "
            "    (quote explore)))"
        ),
    },
    {
        "id": "fen-marsh",
        "name": "Fen Marsh",
        "element": "water",
        "archetype": "nurturer",
        "stats": {"resolve": 0.5, "improvisation": 0.7, "empathy": 0.9,
                  "hoarding": 0.2, "faith": 0.5, "paranoia": 0.3},
        "skills": ["hydroponics", "mediation", "medicine"],
        "decision_expr": (
            "(if (< morale 0.4) (quote mediate) "
            "  (if (< food 0.4) (quote farm) "
            "    (quote heal)))"
        ),
    },
    {
        "id": "rust-vega",
        "name": "Rust Vega",
        "element": "earth",
        "archetype": "builder",
        "stats": {"resolve": 0.7, "improvisation": 0.6, "empathy": 0.3,
                  "hoarding": 0.8, "faith": 0.2, "paranoia": 0.4},
        "skills": ["engineering", "coding", "terraforming"],
        "decision_expr": (
            "(if (< power 0.4) (quote engineer) "
            "  (if (> hoarding 0.6) (quote hoard) "
            "    (quote code)))"
        ),
    },
    {
        "id": "aura-kai",
        "name": "Aura Kai",
        "element": "air",
        "archetype": "dreamer",
        "stats": {"resolve": 0.4, "improvisation": 0.9, "empathy": 0.6,
                  "hoarding": 0.1, "faith": 0.8, "paranoia": 0.2},
        "skills": ["mediation", "prayer", "exploration"],
        "decision_expr": (
            "(if (> faith 0.6) (quote pray) "
            "  (if (> improvisation 0.7) (quote cooperate) "
            "    (quote explore)))"
        ),
    },
    {
        "id": "dax-iron",
        "name": "Dax Iron",
        "element": "earth",
        "archetype": "guardian",
        "stats": {"resolve": 0.8, "improvisation": 0.3, "empathy": 0.2,
                  "hoarding": 0.7, "faith": 0.1, "paranoia": 0.8},
        "skills": ["sabotage", "engineering", "exploration"],
        "decision_expr": (
            "(if (> paranoia 0.6) (quote guard) "
            "  (if (< air 0.3) (quote engineer) "
            "    (quote explore)))"
        ),
    },
    {
        "id": "luna-tide",
        "name": "Luna Tide",
        "element": "water",
        "archetype": "healer",
        "stats": {"resolve": 0.5, "improvisation": 0.6, "empathy": 0.8,
                  "hoarding": 0.2, "faith": 0.6, "paranoia": 0.3},
        "skills": ["medicine", "hydroponics", "mediation"],
        "decision_expr": (
            "(if (< medicine 0.3) (quote heal) "
            "  (if (< morale 0.3) (quote mediate) "
            "    (quote farm)))"
        ),
    },
    {
        "id": "grove-ash",
        "name": "Grove Ash",
        "element": "earth",
        "archetype": "maker",
        "stats": {"resolve": 0.6, "improvisation": 0.7, "empathy": 0.5,
                  "hoarding": 0.5, "faith": 0.3, "paranoia": 0.4},
        "skills": ["engineering", "coding", "terraforming"],
        "decision_expr": (
            "(if (< water 0.3) (quote engineer) "
            "  (if (> improvisation 0.5) (quote code) "
            "    (quote terraform)))"
        ),
    },
    {
        "id": "zeph-wind",
        "name": "Zeph Wind",
        "element": "air",
        "archetype": "diplomat",
        "stats": {"resolve": 0.4, "improvisation": 0.8, "empathy": 0.7,
                  "hoarding": 0.1, "faith": 0.4, "paranoia": 0.2},
        "skills": ["mediation", "coding", "exploration"],
        "decision_expr": (
            "(if (< morale 0.3) (quote mediate) "
            "  (if (> improvisation 0.6) (quote cooperate) "
            "    (quote code)))"
        ),
    },
    {
        "id": "ora-flame",
        "name": "Ora Flame",
        "element": "fire",
        "archetype": "warrior",
        "stats": {"resolve": 0.9, "improvisation": 0.4, "empathy": 0.2,
                  "hoarding": 0.4, "faith": 0.1, "paranoia": 0.7},
        "skills": ["sabotage", "terraforming", "engineering"],
        "decision_expr": (
            "(if (> paranoia 0.5) (quote guard) "
            "  (if (< food 0.3) (quote terraform) "
            "    (quote explore)))"
        ),
    },
    {
        "id": "pax-stone",
        "name": "Pax Stone",
        "element": "water",
        "archetype": "philosopher",
        "stats": {"resolve": 0.5, "improvisation": 0.7, "empathy": 0.6,
                  "hoarding": 0.3, "faith": 0.9, "paranoia": 0.1},
        "skills": ["prayer", "mediation", "coding"],
        "decision_expr": (
            "(if (> faith 0.7) (quote pray) "
            "  (if (< morale 0.4) (quote mediate) "
            "    (quote code)))"
        ),
    },
]


def create_founding_colonists() -> list:
    """Create the 10 founding colonists."""
    colonists = []
    for spec in FOUNDING_COLONISTS:
        colonists.append(Colonist(
            id=spec["id"],
            name=spec["name"],
            element=spec["element"],
            archetype=spec["archetype"],
            stats=dict(spec["stats"]),
            skills=list(spec["skills"]),
            decision_expr=spec["decision_expr"],
        ))
    return colonists
