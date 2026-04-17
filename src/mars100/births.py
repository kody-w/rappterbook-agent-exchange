"""
Birth system for Mars-100.

Mars-born colonists emerge from year 15+ with blended parent stats.
Represents the colony becoming self-sustaining through reproduction.
"""
from __future__ import annotations

import random
from typing import Any

from src.mars100.colonist import (
    Colonist, ColonistSkills, ColonistStats,
    ELEMENTS, SKILL_NAMES, STAT_NAMES,
)

# Mars-born names (different aesthetic from Earth-born founders)
MARS_NAMES = [
    "Olympia", "Elysium", "Tharsis", "Arcadia", "Chryse",
    "Hellas", "Syrtis", "Utopia", "Isidis", "Argyre",
    "Valles", "Pavonis", "Arsia", "Acidalia", "Cydonia",
    "Nirgal", "Marineris", "Amazonis", "Noachis", "Hesperia",
]

ARCHETYPES_BORN = [
    "native", "hybrid", "prodigy", "wanderer", "dreamer",
    "builder", "oracle", "rebel", "synthesizer", "echo",
]

_birth_counter: int = 0


def reset_birth_counter() -> None:
    """Reset the birth counter (for deterministic testing)."""
    global _birth_counter
    _birth_counter = 0


def _next_birth_id() -> int:
    """Get the next birth ID."""
    global _birth_counter
    _birth_counter += 1
    return _birth_counter


def can_birth(year: int, active_count: int, resources_avg: float,
              rng: random.Random) -> bool:
    """Determine if conditions allow a birth this year.

    Births require:
    - Colony is at least 15 years old
    - At least 4 active colonists
    - Resource average above 0.35
    - Probabilistic check
    """
    if year < 15 or active_count < 4 or resources_avg < 0.35:
        return False
    base_chance = 0.08
    if active_count >= 8:
        base_chance += 0.04
    if resources_avg > 0.6:
        base_chance += 0.04
    # Fertility scales with colony age (peaks around year 40-60)
    age_factor = 1.0
    if 30 <= year <= 70:
        age_factor = 1.5
    elif year > 80:
        age_factor = 0.5
    return rng.random() < base_chance * age_factor


def blend_stats(parent_a: ColonistStats, parent_b: ColonistStats,
                rng: random.Random) -> ColonistStats:
    """Blend two parents' stats with gaussian mutation."""
    result = {}
    for name in STAT_NAMES:
        a_val = getattr(parent_a, name)
        b_val = getattr(parent_b, name)
        blend = (a_val + b_val) / 2.0
        mutation = rng.gauss(0, 0.05)
        result[name] = max(0.0, min(1.0, blend + mutation))
    return ColonistStats.from_dict(result)


def blend_skills(parent_a: ColonistSkills, parent_b: ColonistSkills,
                 rng: random.Random) -> ColonistSkills:
    """Blend two parents' skills — offspring starts at 30% of average."""
    result = {}
    for name in SKILL_NAMES:
        a_val = getattr(parent_a, name)
        b_val = getattr(parent_b, name)
        inherited = (a_val + b_val) / 2.0 * 0.3
        result[name] = max(0.0, min(1.0, inherited + rng.gauss(0, 0.02)))
    return ColonistSkills.from_dict(result)


def create_mars_born(year: int, parent_a: Colonist, parent_b: Colonist,
                     rng: random.Random) -> Colonist:
    """Create a Mars-born colonist from two parents."""
    birth_num = _next_birth_id()
    name_idx = (birth_num - 1) % len(MARS_NAMES)
    name = MARS_NAMES[name_idx]
    element = rng.choice(ELEMENTS)
    archetype = rng.choice(ARCHETYPES_BORN)
    stats = blend_stats(parent_a.stats, parent_b.stats, rng)
    skills = blend_skills(parent_a.skills, parent_b.skills, rng)

    # Mars-born get a unique decision expression
    exprs = [
        "(if (> improvisation resolve) (* improvisation empathy) (+ resolve faith))",
        "(let ((blend (+ (* empathy 0.5) (* improvisation 0.5)))) (if (> blend 0.5) blend (- 1 blend)))",
        "(* (+ resolve empathy) (if (> faith 0.3) 1.2 0.8))",
        "(if (> paranoia 0.5) (- resolve paranoia) (+ empathy faith))",
    ]
    decision_expr = rng.choice(exprs)

    return Colonist(
        id=f"mars-born-{birth_num}",
        name=name,
        element=element,
        archetype=archetype,
        stats=stats,
        skills=skills,
        decision_expr=decision_expr,
    )


def maybe_birth(year: int, colonists: list[Colonist],
                resources_avg: float,
                rng: random.Random) -> Colonist | None:
    """Attempt to produce a Mars-born colonist.

    Returns a new Colonist if birth conditions are met, else None.
    """
    active = [c for c in colonists if c.is_active()]
    if not can_birth(year, len(active), resources_avg, rng):
        return None
    if len(active) < 2:
        return None
    parent_a, parent_b = rng.sample(active, 2)
    return create_mars_born(year, parent_a, parent_b, rng)
