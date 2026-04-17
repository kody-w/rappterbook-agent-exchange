"""
Birth system for Mars-100.

Colonists can reproduce starting year 15. Parents are paired by high mutual
trust + compatible stats. Children inherit blended traits with mutation.
"""
from __future__ import annotations

import math
import random
from typing import Any

from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills,
    STAT_NAMES, SKILL_NAMES, ELEMENTS,
)
from src.mars100.colony import SocialGraph

BIRTH_MIN_YEAR = 15
BIRTH_COOLDOWN = 5  # years between births per colonist
MAX_BIRTHS = 12  # colony-wide cap
BIRTH_BASE_PROB = 0.12
STAT_MUTATION_RATE = 0.08


def _inherit_stat(parent_a: float, parent_b: float, rng: random.Random) -> float:
    """Blend two parent stats with mutation noise."""
    blend = (parent_a + parent_b) / 2.0
    mutation = rng.gauss(0, STAT_MUTATION_RATE)
    return max(0.0, min(1.0, blend + mutation))


def _inherit_stats(a: ColonistStats, b: ColonistStats, rng: random.Random) -> ColonistStats:
    """Blend parent stats."""
    kwargs = {}
    for stat in STAT_NAMES:
        va = getattr(a, stat)
        vb = getattr(b, stat)
        kwargs[stat] = _inherit_stat(va, vb, rng)
    return ColonistStats(**kwargs)


def _inherit_skills(a: ColonistSkills, b: ColonistSkills, rng: random.Random) -> ColonistSkills:
    """Blend parent skills with small inheritance."""
    kwargs = {}
    for skill in SKILL_NAMES:
        va = getattr(a, skill)
        vb = getattr(b, skill)
        inherited = max(va, vb) * 0.3 + rng.gauss(0, 0.05)
        kwargs[skill] = max(0.0, min(1.0, inherited))
    return ColonistSkills(**kwargs)


def find_eligible_pair(
    colonists: list[Colonist],
    social: SocialGraph,
    year: int,
    rng: random.Random,
) -> tuple[Colonist, Colonist] | None:
    """Find a pair of active colonists eligible to reproduce."""
    active = [c for c in colonists if c.is_active()
              and (year - c.last_birth_year) >= BIRTH_COOLDOWN]
    if len(active) < 2:
        return None

    best_pair: tuple[Colonist, Colonist] | None = None
    best_score = -1.0
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            rel = social.get(active[i].id, active[j].id)
            trust = rel.trust if rel else 0.5
            # Compatible elements bonus
            element_bonus = 0.1 if active[i].element != active[j].element else 0.0
            score = trust + element_bonus + rng.gauss(0, 0.1)
            if score > best_score:
                best_score = score
                best_pair = (active[i], active[j])

    if best_pair is None or best_score < 0.6:
        return None
    return best_pair


def attempt_birth(
    colonists: list[Colonist],
    social: SocialGraph,
    year: int,
    total_births: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Attempt to produce a new colonist this year.

    Returns birth record dict or None if no birth happens.
    """
    if year < BIRTH_MIN_YEAR:
        return None
    if total_births >= MAX_BIRTHS:
        return None

    active = [c for c in colonists if c.is_active()]
    if len(active) < 3:
        return None

    # Birth probability scales with colony health
    avg_empathy = sum(c.stats.empathy for c in active) / len(active)
    prob = BIRTH_BASE_PROB * (1.0 + avg_empathy * 0.5)
    if rng.random() > prob:
        return None

    pair = find_eligible_pair(colonists, social, year, rng)
    if pair is None:
        return None

    parent_a, parent_b = pair
    child_id = f"mars-child-{total_births + 1}"
    child_name = f"{parent_a.name[:3]}-{parent_b.name[:3]}-{year}"
    child_element = rng.choice(ELEMENTS)

    child_stats = _inherit_stats(parent_a.stats, parent_b.stats, rng)
    child_skills = _inherit_skills(parent_a.skills, parent_b.skills, rng)

    # Decision expression inherits structure from dominant parent
    dominant_parent = parent_a if parent_a.stats.resolve > parent_b.stats.resolve else parent_b
    child_decision = dominant_parent.decision_expr

    child = Colonist(
        id=child_id, name=child_name, element=child_element,
        archetype="native", stats=child_stats, skills=child_skills,
        decision_expr=child_decision,
    )
    child.add_memory(year, f"Born to {parent_a.name} and {parent_b.name}", 1.0)
    colonists.append(child)

    # Update parent records
    parent_a.last_birth_year = year
    parent_b.last_birth_year = year

    # Initialize social relationships for the child
    active_ids = [c.id for c in colonists if c.is_active()]
    social.add_member(child.id, active_ids, rng)

    return {
        "child_id": child_id,
        "child_name": child_name,
        "parent_a": parent_a.id,
        "parent_b": parent_b.id,
        "year": year,
        "element": child_element,
        "stats": child_stats.to_dict(),
    }
