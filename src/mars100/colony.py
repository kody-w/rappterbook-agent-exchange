"""
Colony resource model and social graph for Mars-100.

Manages shared resources, relationships between colonists, and
resource consumption/production per year.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist

RESOURCE_NAMES = ("food", "water", "power", "air", "medicine")

BASE_PRODUCTION = {"food": 0.08, "water": 0.07, "power": 0.06, "air": 0.05, "medicine": 0.03}
BASE_CONSUMPTION = {"food": 0.06, "water": 0.05, "power": 0.04, "air": 0.04, "medicine": 0.01}
MAINTENANCE_COST = {"food": 0.02, "water": 0.02, "power": 0.03, "air": 0.01, "medicine": 0.01}
SPOILAGE_RATE = {"food": 0.03, "water": 0.01, "power": 0.0, "air": 0.02, "medicine": 0.02}


@dataclass
class Resources:
    """Colony resource levels, normalized 0.0-1.0."""
    food: float = 0.7
    water: float = 0.7
    power: float = 0.8
    air: float = 0.9
    medicine: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in RESOURCE_NAMES}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Resources:
        return cls(**{k: d.get(k, 0.5) for k in RESOURCE_NAMES})

    def clamp(self) -> None:
        for name in RESOURCE_NAMES:
            val = getattr(self, name)
            setattr(self, name, max(0.0, min(1.0, val)))

    def total(self) -> float:
        return sum(getattr(self, name) for name in RESOURCE_NAMES)

    def critical(self) -> list[str]:
        return [n for n in RESOURCE_NAMES if getattr(self, n) < 0.15]

    def average(self) -> float:
        return self.total() / len(RESOURCE_NAMES)


@dataclass
class Relationship:
    """Directed relationship from one colonist to another."""
    trust: float = 0.5
    affection: float = 0.5
    respect: float = 0.5

    def score(self) -> float:
        return self.trust * 0.4 + self.affection * 0.3 + self.respect * 0.3

    def to_dict(self) -> dict[str, float]:
        return {"trust": self.trust, "affection": self.affection, "respect": self.respect}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Relationship:
        return cls(trust=d.get("trust", 0.5), affection=d.get("affection", 0.5),
                   respect=d.get("respect", 0.5))


@dataclass
class SocialGraph:
    """Directed relationship graph between colonists."""
    edges: dict[str, dict[str, Relationship]] = field(default_factory=dict)

    def initialize(self, colonist_ids: list[str], rng: random.Random) -> None:
        for a in colonist_ids:
            self.edges[a] = {}
            for b in colonist_ids:
                if a != b:
                    self.edges[a][b] = Relationship(
                        trust=max(0.0, min(1.0, 0.5 + rng.gauss(0, 0.1))),
                        affection=max(0.0, min(1.0, 0.5 + rng.gauss(0, 0.1))),
                        respect=max(0.0, min(1.0, 0.5 + rng.gauss(0, 0.1))))

    def add_colonist(self, new_id: str, existing_ids: list[str],
                     rng: random.Random) -> None:
        """Integrate a new colonist (e.g. newborn) into the social graph."""
        self.edges[new_id] = {}
        for other in existing_ids:
            if other != new_id:
                self.edges[new_id][other] = Relationship(
                    trust=max(0.0, min(1.0, 0.4 + rng.gauss(0, 0.1))),
                    affection=max(0.0, min(1.0, 0.4 + rng.gauss(0, 0.1))),
                    respect=max(0.0, min(1.0, 0.3 + rng.gauss(0, 0.1))))
                if other in self.edges:
                    self.edges[other][new_id] = Relationship(
                        trust=max(0.0, min(1.0, 0.4 + rng.gauss(0, 0.1))),
                        affection=max(0.0, min(1.0, 0.5 + rng.gauss(0, 0.1))),
                        respect=max(0.0, min(1.0, 0.3 + rng.gauss(0, 0.1))))

    def get(self, from_id: str, to_id: str) -> Relationship:
        return self.edges.get(from_id, {}).get(to_id, Relationship())

    def update_from_event(self, participants: list[str], valence: float,
                          rng: random.Random) -> None:
        drift = 0.05 * valence
        for a in participants:
            for b in participants:
                if a != b and a in self.edges and b in self.edges.get(a, {}):
                    rel = self.edges[a][b]
                    rel.trust = max(0.0, min(1.0, rel.trust + drift + rng.gauss(0, 0.02)))
                    rel.affection = max(0.0, min(1.0, rel.affection + drift * 0.5 + rng.gauss(0, 0.02)))
                    rel.respect = max(0.0, min(1.0, rel.respect + drift * 0.3 + rng.gauss(0, 0.02)))

    def update_from_cooperation(self, a_id: str, b_id: str, rng: random.Random) -> None:
        for pair in [(a_id, b_id), (b_id, a_id)]:
            if pair[0] in self.edges and pair[1] in self.edges.get(pair[0], {}):
                rel = self.edges[pair[0]][pair[1]]
                rel.trust = min(1.0, rel.trust + 0.03 + rng.gauss(0, 0.01))
                rel.respect = min(1.0, rel.respect + 0.02 + rng.gauss(0, 0.01))

    def update_from_conflict(self, a_id: str, b_id: str, rng: random.Random) -> None:
        for pair in [(a_id, b_id), (b_id, a_id)]:
            if pair[0] in self.edges and pair[1] in self.edges.get(pair[0], {}):
                rel = self.edges[pair[0]][pair[1]]
                rel.trust = max(0.0, rel.trust - 0.05 - abs(rng.gauss(0, 0.02)))
                rel.affection = max(0.0, rel.affection - 0.04 - abs(rng.gauss(0, 0.02)))

    def most_trusted_by(self, colonist_id: str, active_ids: list[str]) -> str | None:
        if colonist_id not in self.edges:
            return None
        candidates = [(cid, self.edges[colonist_id][cid].trust)
                      for cid in active_ids
                      if cid != colonist_id and cid in self.edges.get(colonist_id, {})]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    def colony_cohesion(self, active_ids: list[str]) -> float:
        total = 0.0
        count = 0
        for a in active_ids:
            for b in active_ids:
                if a != b and a in self.edges and b in self.edges.get(a, {}):
                    total += self.edges[a][b].trust
                    count += 1
        return total / max(1, count)

    def to_dict(self) -> dict:
        return {a: {b: r.to_dict() for b, r in rels.items()} for a, rels in self.edges.items()}


def tick_resources(resources: Resources, active_count: int,
                   skill_bonuses: dict[str, float],
                   event_effects: dict[str, float],
                   infra_modifiers: dict[str, float] | None = None) -> dict[str, float]:
    """Advance resources by one year. Returns delta dict for conservation tracking."""
    before = resources.to_dict()
    mods = infra_modifiers or {}
    for name in RESOURCE_NAMES:
        current = getattr(resources, name)
        production = BASE_PRODUCTION[name] * active_count * (1.0 + skill_bonuses.get(name, 0.0))
        consumption = BASE_CONSUMPTION[name] * active_count
        maintenance = MAINTENANCE_COST[name] * mods.get(f"{name}_maintenance_mult", 1.0)
        spoilage = current * SPOILAGE_RATE[name] * mods.get(f"{name}_spoilage_mult", 1.0)
        event_delta = event_effects.get(name, 0.0)
        new_val = current + production - consumption - maintenance - spoilage + event_delta
        setattr(resources, name, new_val)
    resources.clamp()
    after = resources.to_dict()
    return {name: after[name] - before[name] for name in RESOURCE_NAMES}


def compute_value_convergence(colonists: list) -> dict:
    """Measure stat standard deviation across living colonists.

    Returns a dict mapping stat names to their std-dev.  Lower values mean
    the population's values are converging; higher values mean diverging.
    A convergence_score (mean of all std-devs) summarises the trend.
    """
    from src.mars100.colonist import STAT_NAMES
    active = [c for c in colonists if getattr(c, "is_active", lambda: True)()]
    if len(active) < 2:
        return {name: 0.0 for name in STAT_NAMES} | {"convergence_score": 0.0}
    result: dict[str, float] = {}
    for name in STAT_NAMES:
        values = [getattr(c.stats, name) for c in active]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        result[name] = variance ** 0.5
    result["convergence_score"] = sum(v for k, v in result.items() if k != "convergence_score") / len(STAT_NAMES)
    return result
