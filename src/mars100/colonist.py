"""
Colonist model for Mars-100.

Each colonist has identity, stats, skills, memory, and a LisPy decision expression.
Colonists are data structures AND LisPy programs (homoiconic).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

ELEMENTS = ("fire", "water", "earth", "air")
STAT_NAMES = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")
SKILL_NAMES = ("terraforming", "hydroponics", "mediation", "coding", "prayer", "sabotage")

COLONIST_NAMES = [
    "Nova Sol", "Cinder Ash", "Ripple Tide", "Gale Frost", "Ember Clay",
    "Moss Stone", "Spark Dune", "Echo Vale", "Drift Sand", "Bloom Peak",
    "Fern Ridge", "Haze Cliff", "Reed Shore", "Slate Gorge", "Coral Mist",
    "Jade Hollow", "Flint Mesa", "Willow Rift", "Cedar Bluff", "Opal Creek",
]


@dataclass
class ColonistStats:
    """Six personality stats, each 0.0-1.0."""
    resolve: float = 0.5
    improvisation: float = 0.5
    empathy: float = 0.5
    hoarding: float = 0.5
    faith: float = 0.5
    paranoia: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in STAT_NAMES}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> ColonistStats:
        return cls(**{k: d.get(k, 0.5) for k in STAT_NAMES})

    def dominant(self) -> str:
        return max(STAT_NAMES, key=lambda n: getattr(self, n))


@dataclass
class ColonistSkills:
    """Six skills, each 0.0-1.0."""
    terraforming: float = 0.0
    hydroponics: float = 0.0
    mediation: float = 0.0
    coding: float = 0.0
    prayer: float = 0.0
    sabotage: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in SKILL_NAMES}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> ColonistSkills:
        return cls(**{k: d.get(k, 0.0) for k in SKILL_NAMES})

    def best_skill(self) -> str:
        return max(SKILL_NAMES, key=lambda n: getattr(self, n))


@dataclass
class MemoryEntry:
    """A single memory from a colonist's experience."""
    year: int
    event: str
    emotional_valence: float

    def to_dict(self) -> dict:
        return {"year": self.year, "event": self.event, "valence": self.emotional_valence}


@dataclass
class Wallet:
    """Personal resource stockpile for a colonist (economics organ).

    Holdings are unbounded nonneg floats in abstract resource-token units.
    Separate from colony-level resources.
    """
    holdings: dict[str, float] = field(default_factory=lambda: {
        "food": 0.0, "water": 0.0, "power": 0.0, "air": 0.0, "medicine": 0.0,
    })
    total_earned: float = 0.0
    total_traded: float = 0.0
    total_taxed: float = 0.0

    def total_wealth(self) -> float:
        return sum(self.holdings.values())

    def deposit(self, resource: str, amount: float) -> None:
        if amount < 0:
            return
        self.holdings[resource] = self.holdings.get(resource, 0.0) + amount

    def withdraw(self, resource: str, amount: float) -> float:
        if amount <= 0:
            return 0.0
        available = self.holdings.get(resource, 0.0)
        actual = min(available, amount)
        self.holdings[resource] = available - actual
        return actual

    def to_dict(self) -> dict:
        return {
            "holdings": dict(self.holdings),
            "total_earned": round(self.total_earned, 6),
            "total_traded": round(self.total_traded, 6),
            "total_taxed": round(self.total_taxed, 6),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Wallet:
        if not d:
            return cls()
        res_names = ("food", "water", "power", "air", "medicine")
        holdings = {n: d.get("holdings", {}).get(n, 0.0) for n in res_names}
        return cls(
            holdings=holdings,
            total_earned=d.get("total_earned", 0.0),
            total_traded=d.get("total_traded", 0.0),
            total_taxed=d.get("total_taxed", 0.0),
        )


@dataclass
class Colonist:
    """A Mars-100 colonist."""
    id: str
    name: str
    element: str
    archetype: str
    stats: ColonistStats
    skills: ColonistSkills
    decision_expr: str
    alive: bool = True
    exiled: bool = False
    birth_year: int = 0
    death_year: int | None = None
    death_cause: str | None = None
    exile_year: int | None = None
    memories: list[MemoryEntry] = field(default_factory=list)
    subsim_count: int = 0
    governance_votes: int = 0
    wallet: Wallet = field(default_factory=Wallet)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id, "name": self.name, "element": self.element,
            "archetype": self.archetype, "stats": self.stats.to_dict(),
            "skills": self.skills.to_dict(), "decision_expr": self.decision_expr,
            "alive": self.alive, "exiled": self.exiled, "birth_year": self.birth_year,
            "subsim_count": self.subsim_count, "governance_votes": self.governance_votes,
            "memories": [m.to_dict() for m in self.memories],
            "wallet": self.wallet.to_dict(),
        }
        if self.death_year is not None:
            d["death_year"] = self.death_year
            d["death_cause"] = self.death_cause
        if self.exile_year is not None:
            d["exile_year"] = self.exile_year
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Colonist:
        memories = [MemoryEntry(m["year"], m["event"], m["valence"]) for m in d.get("memories", [])]
        wallet = Wallet.from_dict(d.get("wallet", {}))
        return cls(
            id=d["id"], name=d["name"], element=d["element"], archetype=d["archetype"],
            stats=ColonistStats.from_dict(d["stats"]), skills=ColonistSkills.from_dict(d["skills"]),
            decision_expr=d.get("decision_expr", "(+ resolve empathy)"),
            alive=d.get("alive", True), exiled=d.get("exiled", False),
            birth_year=d.get("birth_year", 0),
            death_year=d.get("death_year"), death_cause=d.get("death_cause"),
            exile_year=d.get("exile_year"), memories=memories,
            subsim_count=d.get("subsim_count", 0), governance_votes=d.get("governance_votes", 0),
            wallet=wallet,
        )

    def is_active(self) -> bool:
        return self.alive and not self.exiled

    def add_memory(self, year: int, event: str, valence: float) -> None:
        self.memories.append(MemoryEntry(year, event, valence))
        if len(self.memories) > 50:
            self.memories = self.memories[-50:]

    def die(self, year: int, cause: str) -> None:
        self.alive = False
        self.death_year = year
        self.death_cause = cause

    def exile(self, year: int) -> None:
        self.exiled = True
        self.exile_year = year

    def evolve_stats(self, event_type: str, rng: random.Random) -> None:
        drift = 0.03
        for name in STAT_NAMES:
            current = getattr(self.stats, name)
            delta = rng.gauss(0, drift)
            if self.element == "fire" and name == "resolve": delta += 0.005
            elif self.element == "water" and name == "empathy": delta += 0.005
            elif self.element == "earth" and name == "hoarding": delta += 0.005
            elif self.element == "air" and name == "improvisation": delta += 0.005
            setattr(self.stats, name, max(0.0, min(1.0, current + delta)))

    def evolve_skills(self, action: str, rng: random.Random) -> None:
        skill_map = {"terraform": "terraforming", "farm": "hydroponics",
                     "mediate": "mediation", "code": "coding",
                     "pray": "prayer", "sabotage": "sabotage",
                     "research": "coding"}
        target = skill_map.get(action)
        if target:
            current = getattr(self.skills, target)
            gain = rng.uniform(0.01, 0.03) * (1.0 - current)
            setattr(self.skills, target, min(1.0, current + gain))

    def lispy_bindings(self) -> dict[str, Any]:
        bindings: dict[str, Any] = {}
        for name in STAT_NAMES:
            bindings[name] = getattr(self.stats, name)
        for name in SKILL_NAMES:
            bindings[name] = getattr(self.skills, name)
        bindings["element"] = self.element
        bindings["alive"] = self.alive
        bindings["memory-count"] = len(self.memories)
        bindings["wealth"] = self.wallet.total_wealth()
        return bindings


def create_founding_ten(seed: int = 42) -> list[Colonist]:
    """Create the 10 founding Mars-100 colonists with distinct personalities."""
    rng = random.Random(seed)
    specs = [
        {"id": "kira-sol", "name": "Kira Sol", "element": "fire", "archetype": "commander",
         "stats": {"resolve": 0.9, "improvisation": 0.4, "empathy": 0.5, "hoarding": 0.3, "faith": 0.2, "paranoia": 0.4},
         "skills": {"terraforming": 0.7, "hydroponics": 0.2, "mediation": 0.5, "coding": 0.3, "prayer": 0.0, "sabotage": 0.1},
         "decision_expr": "(if (> resolve 0.7) (+ resolve improvisation) (* empathy 2))"},
        {"id": "fen-marsh", "name": "Fen Marsh", "element": "water", "archetype": "healer",
         "stats": {"resolve": 0.4, "improvisation": 0.6, "empathy": 0.9, "hoarding": 0.2, "faith": 0.7, "paranoia": 0.1},
         "skills": {"terraforming": 0.1, "hydroponics": 0.8, "mediation": 0.7, "coding": 0.1, "prayer": 0.5, "sabotage": 0.0},
         "decision_expr": "(* empathy (+ faith 0.5))"},
        {"id": "rust-vega", "name": "Rust Vega", "element": "earth", "archetype": "engineer",
         "stats": {"resolve": 0.6, "improvisation": 0.7, "empathy": 0.3, "hoarding": 0.6, "faith": 0.1, "paranoia": 0.5},
         "skills": {"terraforming": 0.9, "hydroponics": 0.3, "mediation": 0.1, "coding": 0.8, "prayer": 0.0, "sabotage": 0.3},
         "decision_expr": "(let ((risk (* paranoia 0.5))) (- improvisation risk))"},
        {"id": "aura-kai", "name": "Aura Kai", "element": "air", "archetype": "philosopher",
         "stats": {"resolve": 0.3, "improvisation": 0.8, "empathy": 0.7, "hoarding": 0.1, "faith": 0.9, "paranoia": 0.2},
         "skills": {"terraforming": 0.2, "hydroponics": 0.4, "mediation": 0.9, "coding": 0.5, "prayer": 0.8, "sabotage": 0.0},
         "decision_expr": "(if (> faith 0.5) (* faith empathy) improvisation)"},
        {"id": "dax-iron", "name": "Dax Iron", "element": "fire", "archetype": "soldier",
         "stats": {"resolve": 0.8, "improvisation": 0.3, "empathy": 0.2, "hoarding": 0.7, "faith": 0.1, "paranoia": 0.8},
         "skills": {"terraforming": 0.5, "hydroponics": 0.1, "mediation": 0.0, "coding": 0.2, "prayer": 0.0, "sabotage": 0.7},
         "decision_expr": "(if (> paranoia 0.6) (* hoarding resolve) (- resolve empathy))"},
        {"id": "luna-tide", "name": "Luna Tide", "element": "water", "archetype": "scientist",
         "stats": {"resolve": 0.5, "improvisation": 0.9, "empathy": 0.4, "hoarding": 0.3, "faith": 0.3, "paranoia": 0.3},
         "skills": {"terraforming": 0.6, "hydroponics": 0.6, "mediation": 0.3, "coding": 0.9, "prayer": 0.1, "sabotage": 0.1},
         "decision_expr": "(+ (* improvisation coding) (/ terraforming 2))"},
        {"id": "grove-ash", "name": "Grove Ash", "element": "earth", "archetype": "farmer",
         "stats": {"resolve": 0.7, "improvisation": 0.5, "empathy": 0.6, "hoarding": 0.8, "faith": 0.4, "paranoia": 0.3},
         "skills": {"terraforming": 0.4, "hydroponics": 0.9, "mediation": 0.4, "coding": 0.1, "prayer": 0.2, "sabotage": 0.0},
         "decision_expr": "(+ hoarding (* hydroponics resolve))"},
        {"id": "zeph-wind", "name": "Zeph Wind", "element": "air", "archetype": "trickster",
         "stats": {"resolve": 0.4, "improvisation": 0.9, "empathy": 0.3, "hoarding": 0.5, "faith": 0.2, "paranoia": 0.7},
         "skills": {"terraforming": 0.3, "hydroponics": 0.2, "mediation": 0.2, "coding": 0.6, "prayer": 0.1, "sabotage": 0.8},
         "decision_expr": "(if (> paranoia 0.5) sabotage (* improvisation coding))"},
        {"id": "ora-flame", "name": "Ora Flame", "element": "fire", "archetype": "prophet",
         "stats": {"resolve": 0.6, "improvisation": 0.5, "empathy": 0.8, "hoarding": 0.1, "faith": 0.95, "paranoia": 0.6},
         "skills": {"terraforming": 0.2, "hydroponics": 0.3, "mediation": 0.6, "coding": 0.0, "prayer": 0.9, "sabotage": 0.0},
         "decision_expr": "(* faith (if (> empathy 0.5) (+ empathy resolve) paranoia))"},
        {"id": "pax-stone", "name": "Pax Stone", "element": "earth", "archetype": "judge",
         "stats": {"resolve": 0.7, "improvisation": 0.3, "empathy": 0.6, "hoarding": 0.4, "faith": 0.5, "paranoia": 0.4},
         "skills": {"terraforming": 0.3, "hydroponics": 0.3, "mediation": 0.8, "coding": 0.4, "prayer": 0.3, "sabotage": 0.0},
         "decision_expr": "(+ (* empathy mediation) (* resolve 0.5))"},
    ]
    colonists: list[Colonist] = []
    for spec in specs:
        c = Colonist(id=spec["id"], name=spec["name"], element=spec["element"],
                     archetype=spec["archetype"],
                     stats=ColonistStats.from_dict(spec["stats"]),
                     skills=ColonistSkills.from_dict(spec["skills"]),
                     decision_expr=spec["decision_expr"])
        for name in STAT_NAMES:
            current = getattr(c.stats, name)
            setattr(c.stats, name, max(0.0, min(1.0, current + rng.gauss(0, 0.02))))
        colonists.append(c)
    return colonists


def create_child(parent_a: Colonist, parent_b: Colonist, child_id: str,
                 birth_year: int, rng: random.Random) -> Colonist:
    """Create a child colonist by blending two parents' stats/skills with mutation.

    Stats are averaged from parents with gaussian noise.  Skills start near zero
    (children must learn).  Element is randomly inherited from one parent.
    """
    name_pool = [n for n in COLONIST_NAMES]
    name = rng.choice(name_pool) if name_pool else f"Child-{child_id}"

    element = rng.choice([parent_a.element, parent_b.element])

    stats_dict: dict[str, float] = {}
    for sn in STAT_NAMES:
        avg = (getattr(parent_a.stats, sn) + getattr(parent_b.stats, sn)) / 2
        stats_dict[sn] = max(0.0, min(1.0, avg + rng.gauss(0, 0.08)))

    skills_dict: dict[str, float] = {}
    for sk in SKILL_NAMES:
        skills_dict[sk] = max(0.0, rng.gauss(0.05, 0.03))

    expr_fragments = [parent_a.decision_expr, parent_b.decision_expr]
    decision_expr = rng.choice(expr_fragments)

    return Colonist(
        id=child_id, name=name, element=element, archetype="child",
        stats=ColonistStats.from_dict(stats_dict),
        skills=ColonistSkills.from_dict(skills_dict),
        decision_expr=decision_expr, birth_year=birth_year,
    )


# --- Immigrant archetypes (adults arriving from Earth) ---

IMMIGRANT_ARCHETYPES = [
    {"archetype": "engineer", "bias": {"coding": 0.6, "terraforming": 0.4},
     "stat_bias": {"resolve": 0.6, "improvisation": 0.5}},
    {"archetype": "medic", "bias": {"mediation": 0.5, "hydroponics": 0.3},
     "stat_bias": {"empathy": 0.7, "faith": 0.4}},
    {"archetype": "scientist", "bias": {"coding": 0.7, "terraforming": 0.3},
     "stat_bias": {"improvisation": 0.7, "paranoia": 0.3}},
    {"archetype": "diplomat", "bias": {"mediation": 0.7, "prayer": 0.2},
     "stat_bias": {"empathy": 0.6, "resolve": 0.5}},
    {"archetype": "pioneer", "bias": {"terraforming": 0.6, "hydroponics": 0.4},
     "stat_bias": {"resolve": 0.7, "hoarding": 0.4}},
]


def create_immigrant(immigrant_id: str, arrival_year: int,
                     rng: random.Random) -> Colonist:
    """Create an adult immigrant colonist arriving from Earth.

    Unlike children (blended from parents), immigrants arrive with
    adult-level skills and a randomly chosen archetype.
    """
    template = rng.choice(IMMIGRANT_ARCHETYPES)
    name = rng.choice([n for n in COLONIST_NAMES])
    element = rng.choice(list(ELEMENTS))

    stats_dict: dict[str, float] = {}
    for sn in STAT_NAMES:
        base = template["stat_bias"].get(sn, 0.5)
        stats_dict[sn] = max(0.0, min(1.0, base + rng.gauss(0, 0.08)))

    skills_dict: dict[str, float] = {}
    for sk in SKILL_NAMES:
        base = template["bias"].get(sk, 0.1)
        skills_dict[sk] = max(0.0, min(1.0, base + rng.gauss(0, 0.06)))

    decision_expr = "(+ (* resolve improvisation) (* empathy 0.3))"

    return Colonist(
        id=immigrant_id, name=name, element=element,
        archetype=template["archetype"],
        stats=ColonistStats.from_dict(stats_dict),
        skills=ColonistSkills.from_dict(skills_dict),
        decision_expr=decision_expr, birth_year=arrival_year,
    )

