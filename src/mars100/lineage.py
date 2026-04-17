"""
Birth mechanics and lineage tracking for Mars-100.

Mars-born colonists emerge from year 15+ with blended parent stats,
gaussian mutation, and inherited elemental affinity. Maximum population
capped at 30 to prevent resource model instability.
"""
from __future__ import annotations

import random
from typing import Any

from src.mars100.colonist import (
    ELEMENTS,
    SKILL_NAMES,
    STAT_NAMES,
    Colonist,
    ColonistSkills,
    ColonistStats,
)
from src.mars100.colony import RESOURCE_NAMES, Resources

MAX_POPULATION = 30
MIN_BIRTH_YEAR = 15
MIN_PARENT_AGE = 15
BIRTH_STAT_MUTATION = 0.08
ARCHETYPES = ("pioneer", "scholar", "artisan", "sentinel", "mystic", "mediator")


def _blend(a: float, b: float, rng: random.Random) -> float:
    """Blend two parent values with gaussian mutation, clamped to [0, 1]."""
    weight = rng.uniform(0.3, 0.7)
    blended = a * weight + b * (1.0 - weight) + rng.gauss(0, BIRTH_STAT_MUTATION)
    return max(0.0, min(1.0, blended))


def _generate_child_id(parent_a: Colonist, parent_b: Colonist,
                       year: int, existing_ids: set[str]) -> str:
    """Generate a deterministic unique ID for a newborn."""
    prefix = parent_a.name.split("-")[0].split(" ")[0][:3].lower()
    suffix = parent_b.name.split("-")[0].split(" ")[0][:3].lower()
    base = f"{prefix}{suffix}-y{year}"
    if base not in existing_ids:
        return base
    for i in range(2, 100):
        candidate = f"{base}-{i}"
        if candidate not in existing_ids:
            return candidate
    return f"{base}-{year}"


def _generate_child_name(parent_a: Colonist, parent_b: Colonist,
                         generation: int, rng: random.Random) -> str:
    """Generate a display name for a Mars-born colonist."""
    prefix_names = ["Sol", "Dust", "Rust", "Sky", "Red", "Storm",
                    "Dawn", "Frost", "Iron", "Clay", "Ember", "Dusk"]
    suffix_names = parent_a.name.split(" ")[-1][:3]
    chosen = rng.choice(prefix_names)
    gen_tag = f"-{'I' * min(generation, 5)}" if generation > 1 else ""
    return f"{chosen} {suffix_names}{gen_tag}"


def _blend_skills(a: ColonistSkills, b: ColonistSkills,
                  rng: random.Random) -> ColonistSkills:
    """Blend parent skills with mutation. Children start weaker."""
    vals: dict[str, float] = {}
    for name in SKILL_NAMES:
        parent_avg = (getattr(a, name) + getattr(b, name)) / 2.0
        # Children inherit ~40% of parent skill, plus noise
        vals[name] = max(0.0, min(1.0, parent_avg * 0.4 + rng.gauss(0, 0.05)))
    return ColonistSkills.from_dict(vals)


def _blend_stats(a: ColonistStats, b: ColonistStats,
                 rng: random.Random) -> ColonistStats:
    """Blend parent stats with mutation."""
    vals: dict[str, float] = {}
    for name in STAT_NAMES:
        vals[name] = _blend(getattr(a, name), getattr(b, name), rng)
    return ColonistStats.from_dict(vals)


def _generate_decision_expr(stats: ColonistStats, element: str,
                            rng: random.Random) -> str:
    """Generate a LisPy decision expression for a newborn."""
    dominant = stats.dominant()
    templates = [
        f"(if (> {dominant} 0.5) (+ {dominant} empathy) (* resolve 0.8))",
        f"(let ((drive (* {dominant} resolve))) (if (> drive 0.4) drive faith))",
        f"(+ (* {dominant} 0.6) (* improvisation 0.4))",
        f"(if (> faith 0.5) (* faith {dominant}) (- improvisation paranoia))",
    ]
    return rng.choice(templates)


def can_birth(year: int, active_count: int, resources: Resources) -> bool:
    """Check if conditions allow a birth this year."""
    if year < MIN_BIRTH_YEAR:
        return False
    if active_count >= MAX_POPULATION:
        return False
    avg_resources = resources.average()
    if avg_resources < 0.35:
        return False
    return True


def select_parents(colonists: list[Colonist], year: int,
                   rng: random.Random) -> tuple[Colonist, Colonist] | None:
    """Select two compatible parents from active colonists."""
    eligible = [c for c in colonists
                if c.is_active() and c.age(year) >= MIN_PARENT_AGE]
    if len(eligible) < 2:
        return None
    # Weight by empathy + resolve (stable parents)
    weights = [(c.stats.empathy * 0.5 + c.stats.resolve * 0.3 + 0.2)
               for c in eligible]
    total = sum(weights)
    if total <= 0:
        return None
    # Select first parent
    r1 = rng.random() * total
    cumulative = 0.0
    parent_a = eligible[0]
    for c, w in zip(eligible, weights):
        cumulative += w
        if r1 <= cumulative:
            parent_a = c
            break
    # Select second parent (different from first)
    remaining = [(c, w) for c, w in zip(eligible, weights) if c.id != parent_a.id]
    if not remaining:
        return None
    total2 = sum(w for _, w in remaining)
    if total2 <= 0:
        return None
    r2 = rng.random() * total2
    cumulative = 0.0
    parent_b = remaining[0][0]
    for c, w in remaining:
        cumulative += w
        if r2 <= cumulative:
            parent_b = c
            break
    return (parent_a, parent_b)


def maybe_birth(colonists: list[Colonist], year: int,
                resources: Resources, rng: random.Random) -> Colonist | None:
    """Attempt to produce a Mars-born colonist. Returns None if conditions aren't met."""
    active = [c for c in colonists if c.is_active()]
    if not can_birth(year, len(active), resources):
        return None
    # Birth probability: higher with better resources, lower with more population
    avg_res = resources.average()
    birth_prob = 0.15 * avg_res - 0.02 * (len(active) / MAX_POPULATION)
    if rng.random() > birth_prob:
        return None
    parents = select_parents(colonists, year, rng)
    if parents is None:
        return None
    parent_a, parent_b = parents
    existing_ids = {c.id for c in colonists}
    child_id = _generate_child_id(parent_a, parent_b, year, existing_ids)
    generation = max(parent_a.generation, parent_b.generation) + 1
    child_name = _generate_child_name(parent_a, parent_b, generation, rng)
    element = rng.choice([parent_a.element, parent_b.element])
    archetype = rng.choice(ARCHETYPES)
    stats = _blend_stats(parent_a.stats, parent_b.stats, rng)
    skills = _blend_skills(parent_a.skills, parent_b.skills, rng)
    decision_expr = _generate_decision_expr(stats, element, rng)
    child = Colonist(
        id=child_id, name=child_name, element=element, archetype=archetype,
        stats=stats, skills=skills, decision_expr=decision_expr,
        birth_year=year, generation=generation,
        parent_ids=[parent_a.id, parent_b.id],
    )
    return child
