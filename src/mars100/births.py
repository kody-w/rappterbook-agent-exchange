"""
Birth system for Mars-100.

Colonists can reproduce under strict conditions. Children inherit averaged
parent traits with mutation noise, start in a juvenile phase (reduced
productivity), and impose a real resource cost on the colony.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from src.mars100.colonist import (
    Colonist, ColonistSkills, ColonistStats, MemoryEntry,
    ELEMENTS, STAT_NAMES, SKILL_NAMES,
)
from src.mars100.colony import Resources, SocialGraph, RESOURCE_NAMES

MAX_POPULATION = 20
MIN_FOOD_FOR_BIRTH = 0.55
MIN_MEDICINE_FOR_BIRTH = 0.3
MIN_TRUST_FOR_PAIR = 0.6
BIRTH_PROBABILITY = 0.15
BIRTH_FOOD_COST = 0.08
BIRTH_MEDICINE_COST = 0.06
JUVENILE_YEARS = 5
PARENT_COOLDOWN_YEARS = 8

NAME_SUFFIXES = [
    "II", "III", "Jr", "Nova", "Sol", "Kai", "Rex", "Lux", "Zen", "Pax",
    "Ash", "Sky", "Ember", "Dune", "Crest", "Vale", "Drift", "Peak",
]


@dataclass
class BirthEvent:
    """Record of a birth in the colony."""
    year: int
    child_id: str
    child_name: str
    parent_a_id: str
    parent_b_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year, "child_id": self.child_id,
            "child_name": self.child_name,
            "parent_a_id": self.parent_a_id,
            "parent_b_id": self.parent_b_id,
        }


def _generate_child_name(parent_a: Colonist, parent_b: Colonist,
                         rng: random.Random) -> str:
    """Generate a name from parent names + suffix."""
    base = rng.choice([parent_a.name.split()[0], parent_b.name.split()[0]])
    suffix = rng.choice(NAME_SUFFIXES)
    return f"{base}-{suffix}"


def _generate_child_id(parent_a: Colonist, parent_b: Colonist,
                       year: int) -> str:
    """Generate a unique child ID."""
    return f"child-{parent_a.id[:4]}-{parent_b.id[:4]}-y{year}"


def _inherit_stats(parent_a: ColonistStats, parent_b: ColonistStats,
                   rng: random.Random) -> ColonistStats:
    """Average parent stats with mutation noise."""
    d: dict[str, float] = {}
    for name in STAT_NAMES:
        avg = (getattr(parent_a, name) + getattr(parent_b, name)) / 2.0
        mutated = avg + rng.gauss(0, 0.08)
        d[name] = max(0.0, min(1.0, mutated))
    return ColonistStats.from_dict(d)


def _initial_child_skills(rng: random.Random) -> ColonistSkills:
    """Children start with very low skills."""
    d: dict[str, float] = {}
    for name in SKILL_NAMES:
        d[name] = rng.uniform(0.0, 0.15)
    return ColonistSkills.from_dict(d)


def find_eligible_pair(
    colonists: list[Colonist],
    social: SocialGraph,
    year: int,
    parent_cooldowns: dict[str, int],
) -> tuple[Colonist, Colonist] | None:
    """Find a pair of active colonists eligible to reproduce."""
    active = [c for c in colonists if c.is_active()]
    if len(active) < 2:
        return None
    candidates: list[tuple[Colonist, Colonist, float]] = []
    for i, a in enumerate(active):
        if parent_cooldowns.get(a.id, 0) > year:
            continue
        for b in active[i + 1:]:
            if parent_cooldowns.get(b.id, 0) > year:
                continue
            rel_ab = social.get(a.id, b.id)
            rel_ba = social.get(b.id, a.id)
            avg_trust = (rel_ab.trust + rel_ba.trust) / 2.0
            if avg_trust >= MIN_TRUST_FOR_PAIR:
                candidates.append((a, b, avg_trust))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[2])
    return candidates[0][0], candidates[0][1]


def can_birth(
    resources: Resources,
    active_count: int,
) -> bool:
    """Check colony-level birth preconditions."""
    if active_count >= MAX_POPULATION:
        return False
    if resources.food < MIN_FOOD_FOR_BIRTH:
        return False
    if resources.medicine < MIN_MEDICINE_FOR_BIRTH:
        return False
    return True


def attempt_birth(
    colonists: list[Colonist],
    resources: Resources,
    social: SocialGraph,
    year: int,
    rng: random.Random,
    parent_cooldowns: dict[str, int],
) -> BirthEvent | None:
    """Attempt to produce a birth this year. Returns BirthEvent or None."""
    active = [c for c in colonists if c.is_active()]
    if not can_birth(resources, len(active)):
        return None
    if rng.random() > BIRTH_PROBABILITY:
        return None
    pair = find_eligible_pair(colonists, social, year, parent_cooldowns)
    if pair is None:
        return None
    parent_a, parent_b = pair
    child_id = _generate_child_id(parent_a, parent_b, year)
    child_name = _generate_child_name(parent_a, parent_b, rng)
    child_element = rng.choice(ELEMENTS)
    archetypes = ["settler", "pioneer", "native", "dreamer", "rebel"]
    child_archetype = rng.choice(archetypes)
    child_stats = _inherit_stats(parent_a.stats, parent_b.stats, rng)
    child_skills = _initial_child_skills(rng)
    parent_dominant = parent_a.stats.dominant()
    child_expr = f"(+ {parent_dominant} (* improvisation 0.5))"
    child = Colonist(
        id=child_id, name=child_name, element=child_element,
        archetype=child_archetype, stats=child_stats, skills=child_skills,
        decision_expr=child_expr,
    )
    child.add_memory(year, f"Born to {parent_a.name} and {parent_b.name}", 0.8)
    # Resource cost
    resources.food -= BIRTH_FOOD_COST
    resources.medicine -= BIRTH_MEDICINE_COST
    resources.clamp()
    # Cooldowns
    parent_cooldowns[parent_a.id] = year + PARENT_COOLDOWN_YEARS
    parent_cooldowns[parent_b.id] = year + PARENT_COOLDOWN_YEARS
    # Add to colony
    colonists.append(child)
    # Initialize social edges
    active_ids = [c.id for c in colonists if c.is_active()]
    social.edges[child_id] = {}
    for cid in active_ids:
        if cid == child_id:
            continue
        base_trust = 0.5
        if cid == parent_a.id or cid == parent_b.id:
            base_trust = 0.8
        social.edges[child_id][cid] = type(social.get(cid, child_id))(
            trust=max(0.0, min(1.0, base_trust + rng.gauss(0, 0.05))),
            affection=max(0.0, min(1.0, base_trust + rng.gauss(0, 0.05))),
            respect=max(0.0, min(1.0, 0.4 + rng.gauss(0, 0.05))),
        )
        if cid not in social.edges:
            social.edges[cid] = {}
        social.edges[cid][child_id] = type(social.get(cid, child_id))(
            trust=max(0.0, min(1.0, base_trust + rng.gauss(0, 0.05))),
            affection=max(0.0, min(1.0, base_trust + rng.gauss(0, 0.05))),
            respect=max(0.0, min(1.0, 0.3 + rng.gauss(0, 0.05))),
        )
    return BirthEvent(
        year=year, child_id=child_id, child_name=child_name,
        parent_a_id=parent_a.id, parent_b_id=parent_b.id,
    )


def is_juvenile(colonist: Colonist, current_year: int) -> bool:
    """Check if a colonist is still a juvenile (reduced productivity)."""
    if not colonist.memories:
        return False
    birth_mem = next((m for m in colonist.memories if "Born to" in m.event), None)
    if birth_mem is None:
        return False
    return (current_year - birth_mem.year) < JUVENILE_YEARS
