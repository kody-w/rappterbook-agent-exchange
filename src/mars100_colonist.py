"""mars100_colonist.py — Colonist model for the Mars-100 recursive simulation.

Each colonist has:
  - Stable identity: id, name, element, alive flag
  - Stats: resolve, improvisation, empathy, hoarding, faith, paranoia (0.0–1.0)
  - Skills: terraforming, hydroponics, mediation, coding, prayer, sabotage (0.0–1.0)
  - Relationships: affinity toward each other colonist (-1.0 to 1.0)
  - Memory: bounded diary of events (max 50 per colonist)
  - Policy: a LisPy s-expression defining decision-making behavior

The colonist state is a plain dict — serializable as JSON. The policy
field is a LisPy string — executable but separate from stable state.

Python stdlib only.
"""
from __future__ import annotations

import random
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELEMENTS = ("fire", "water", "earth", "air")

STAT_NAMES = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")

SKILL_NAMES = (
    "terraforming",
    "hydroponics",
    "mediation",
    "coding",
    "prayer",
    "sabotage",
)

MAX_MEMORY = 50  # max diary entries per colonist

# 10 founding colonists — names inspired by Mars mythology
FOUNDERS: list[dict[str, Any]] = [
    {"id": "ares",      "name": "Ares",      "element": "fire",  "archetype": "warrior"},
    {"id": "demeter",   "name": "Demeter",   "element": "earth", "archetype": "farmer"},
    {"id": "hermes",    "name": "Hermes",    "element": "air",   "archetype": "messenger"},
    {"id": "poseidon",  "name": "Poseidon",  "element": "water", "archetype": "engineer"},
    {"id": "athena",    "name": "Athena",    "element": "air",   "archetype": "strategist"},
    {"id": "hephaestus","name": "Hephaestus","element": "fire",  "archetype": "builder"},
    {"id": "persephone","name": "Persephone","element": "earth", "archetype": "healer"},
    {"id": "apollo",    "name": "Apollo",    "element": "fire",  "archetype": "visionary"},
    {"id": "artemis",   "name": "Artemis",   "element": "water", "archetype": "scout"},
    {"id": "prometheus","name": "Prometheus","element": "air",   "archetype": "rebel"},
]

# Default policies per archetype — LisPy decision expressions
DEFAULT_POLICIES: dict[str, str] = {
    "warrior": '(if (< (get colony "food") 50) "gather" (if (> paranoia 0.6) "guard" "build"))',
    "farmer": '(if (< (get colony "food") 30) "gather" "farm")',
    "messenger": '(if (> empathy 0.5) "mediate" "scout")',
    "engineer": '(if (< (get colony "water") 40) "fix-water" "build")',
    "strategist": '(if (> (length proposals) 0) "vote" "propose")',
    "builder": '(if (< (get colony "habitat") 60) "build" "gather")',
    "healer": '(if (> (get colony "sick") 0) "heal" "farm")',
    "visionary": '(if (> faith 0.7) "pray" (if (> resolve 0.5) "propose" "scout"))',
    "scout": '(if (< (get colony "explored") 0.5) "scout" "gather")',
    "rebel": '(if (> paranoia 0.7) "sabotage" (if (< (get colony "freedom") 0.3) "propose" "build"))',
}


# ---------------------------------------------------------------------------
# Colonist creation
# ---------------------------------------------------------------------------


def create_colonist(
    founder: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Create a colonist dict from a founder template.

    Stats and skills are randomly initialized with bias toward the
    colonist's archetype. Relationships start as slight noise.
    """
    archetype = founder["archetype"]

    # Generate stats with archetype bias
    stats = _generate_stats(archetype, rng)
    skills = _generate_skills(archetype, rng)
    policy = DEFAULT_POLICIES.get(archetype, '"gather"')

    return {
        "id": founder["id"],
        "name": founder["name"],
        "element": founder["element"],
        "archetype": archetype,
        "alive": True,
        "death_year": None,
        "death_cause": None,
        "stats": stats,
        "skills": skills,
        "relationships": {},  # populated after all colonists exist
        "memory": [],
        "policy": policy,
        "votes_cast": 0,
        "proposals_made": 0,
        "sub_sims_run": 0,
    }


def _generate_stats(archetype: str, rng: random.Random) -> dict[str, float]:
    """Generate stats with archetype-based bias."""
    biases: dict[str, dict[str, float]] = {
        "warrior":    {"resolve": 0.3, "paranoia": 0.2},
        "farmer":     {"empathy": 0.2, "hoarding": 0.2},
        "messenger":  {"empathy": 0.3, "improvisation": 0.2},
        "engineer":   {"resolve": 0.2, "improvisation": 0.2},
        "strategist": {"resolve": 0.2, "empathy": 0.2},
        "builder":    {"resolve": 0.3, "hoarding": 0.1},
        "healer":     {"empathy": 0.4, "faith": 0.1},
        "visionary":  {"faith": 0.3, "improvisation": 0.2},
        "scout":      {"improvisation": 0.3, "paranoia": 0.1},
        "rebel":      {"paranoia": 0.3, "improvisation": 0.2},
    }
    base_bias = biases.get(archetype, {})
    stats: dict[str, float] = {}
    for name in STAT_NAMES:
        base = 0.3 + rng.random() * 0.4  # 0.3–0.7 base
        bias = base_bias.get(name, 0.0)
        stats[name] = max(0.0, min(1.0, base + bias))
    return stats


def _generate_skills(archetype: str, rng: random.Random) -> dict[str, float]:
    """Generate skills with archetype-based primary skill boost."""
    primary_map: dict[str, str] = {
        "warrior":    "sabotage",
        "farmer":     "hydroponics",
        "messenger":  "mediation",
        "engineer":   "coding",
        "strategist": "mediation",
        "builder":    "terraforming",
        "healer":     "mediation",
        "visionary":  "prayer",
        "scout":      "terraforming",
        "rebel":      "sabotage",
    }
    primary = primary_map.get(archetype)
    skills: dict[str, float] = {}
    for name in SKILL_NAMES:
        base = 0.1 + rng.random() * 0.3  # 0.1–0.4 base
        if name == primary:
            base += 0.3  # primary skill boost
        skills[name] = max(0.0, min(1.0, base))
    return skills


def init_relationships(
    colonists: list[dict[str, Any]],
    rng: random.Random,
) -> None:
    """Initialize relationship matrix between all colonists.

    Same-element colonists start with slight positive affinity.
    Relationships are symmetric with noise.
    """
    ids = [c["id"] for c in colonists]
    for colonist in colonists:
        rels: dict[str, float] = {}
        for other_id in ids:
            if other_id == colonist["id"]:
                continue
            other = next(c for c in colonists if c["id"] == other_id)
            # Same element → slight positive bias
            element_bonus = 0.1 if colonist["element"] == other["element"] else 0.0
            affinity = (rng.random() * 0.4 - 0.2) + element_bonus  # -0.2 to +0.3
            rels[other_id] = max(-1.0, min(1.0, affinity))
        colonist["relationships"] = rels


def update_relationship(
    colonist: dict[str, Any],
    other_id: str,
    delta: float,
) -> None:
    """Shift relationship affinity, clamped to [-1, 1]."""
    current = colonist["relationships"].get(other_id, 0.0)
    colonist["relationships"][other_id] = max(-1.0, min(1.0, current + delta))


def add_memory(colonist: dict[str, Any], year: int, event: str) -> None:
    """Append event to memory, pruning oldest if over MAX_MEMORY."""
    colonist["memory"].append({"year": year, "event": event})
    if len(colonist["memory"]) > MAX_MEMORY:
        colonist["memory"] = colonist["memory"][-MAX_MEMORY:]


def kill_colonist(
    colonist: dict[str, Any],
    year: int,
    cause: str,
) -> dict[str, Any]:
    """Mark colonist as dead. Returns archived soul snapshot."""
    colonist["alive"] = False
    colonist["death_year"] = year
    colonist["death_cause"] = cause
    add_memory(colonist, year, f"Died: {cause}")
    # Soul file — legacy, not delete
    return {
        "id": colonist["id"],
        "name": colonist["name"],
        "element": colonist["element"],
        "archetype": colonist["archetype"],
        "death_year": year,
        "death_cause": cause,
        "final_stats": dict(colonist["stats"]),
        "final_skills": dict(colonist["skills"]),
        "memory": list(colonist["memory"]),
        "epitaph": _generate_epitaph(colonist),
    }


def _generate_epitaph(colonist: dict[str, Any]) -> str:
    """Generate a brief epitaph from the colonist's life."""
    name = colonist["name"]
    archetype = colonist["archetype"]
    top_stat = max(colonist["stats"], key=colonist["stats"].get)
    top_skill = max(colonist["skills"], key=colonist["skills"].get)
    return (
        f"{name} the {archetype} — strongest in {top_stat}, "
        f"most skilled at {top_skill}. "
        f"Remembered {len(colonist['memory'])} events."
    )


def evolve_stats(
    colonist: dict[str, Any],
    year: int,
    event_type: str,
    rng: random.Random,
) -> None:
    """Evolve colonist stats based on year events. Small drift per year."""
    drift = 0.02  # base annual drift
    stats = colonist["stats"]

    # Event-driven stat changes
    event_effects: dict[str, dict[str, float]] = {
        "dust_storm":       {"resolve": 0.02, "paranoia": 0.03},
        "resource_strike":  {"hoarding": -0.02, "faith": 0.01},
        "equipment_failure":{"improvisation": 0.03, "paranoia": 0.02},
        "earth_contact":    {"faith": 0.02, "resolve": 0.01, "paranoia": -0.02},
        "alien_signal":     {"faith": 0.05, "paranoia": 0.05},
        "solar_flare":      {"resolve": 0.02, "paranoia": 0.03},
        "meteor":           {"paranoia": 0.04, "resolve": 0.02},
        "epidemic":         {"empathy": 0.03, "paranoia": 0.02},
        "calm":             {"resolve": 0.01, "paranoia": -0.02},
    }

    effects = event_effects.get(event_type, {})
    for stat_name in STAT_NAMES:
        change = effects.get(stat_name, 0.0) + (rng.random() - 0.5) * drift
        stats[stat_name] = max(0.0, min(1.0, stats[stat_name] + change))


def create_all_colonists(seed: int = 42) -> list[dict[str, Any]]:
    """Create the full roster of 10 founding colonists."""
    rng = random.Random(seed)
    colonists = [create_colonist(f, rng) for f in FOUNDERS]
    init_relationships(colonists, rng)
    return colonists
