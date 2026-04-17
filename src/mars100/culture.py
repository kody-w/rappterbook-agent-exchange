"""
Culture engine for Mars-100.

Events become stories.  Stories become myths.  Myths become rituals.
Rituals become constitutional norms.  This is how a colony makes meaning.

Cultural memories decay without retelling.  Children inherit their
parents' stories.  Dead carriers take unshared memories to the grave.
Factions drift apart as the same event is retold with different spins.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# Lifecycle stages
STAGE_STORY = "story"
STAGE_MYTH = "myth"
STAGE_RITUAL = "ritual"
STAGE_NORM = "norm"
STAGES = (STAGE_STORY, STAGE_MYTH, STAGE_RITUAL, STAGE_NORM)

# Promotion thresholds
MYTH_RETELL_THRESHOLD = 5
MYTH_MIN_AGE = 8
RITUAL_CARRIER_FRACTION = 0.5
NORM_MIN_RITUAL_AGE = 15

# Decay & limits
ANNUAL_DECAY = 0.08
RETELL_RECOVERY = 0.15
INITIAL_STRENGTH = 0.85
DEATH_STRENGTH_PENALTY = 0.1
MAX_MEMORIES = 50


@dataclass
class CulturalMemory:
    """A single cultural memory — story, myth, ritual, or norm."""

    id: str
    origin_year: int
    origin_event: str
    narrative: str
    stage: str = STAGE_STORY
    strength: float = INITIAL_STRENGTH
    carriers: list[str] = field(default_factory=list)
    retellings: int = 0
    mutations: int = 0
    codified_year: int | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id, "origin_year": self.origin_year,
            "origin_event": self.origin_event, "narrative": self.narrative,
            "stage": self.stage, "strength": round(self.strength, 4),
            "carriers": list(self.carriers), "retellings": self.retellings,
            "mutations": self.mutations,
        }
        if self.codified_year is not None:
            d["codified_year"] = self.codified_year
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CulturalMemory:
        return cls(
            id=d["id"], origin_year=d["origin_year"],
            origin_event=d["origin_event"], narrative=d["narrative"],
            stage=d.get("stage", STAGE_STORY),
            strength=d.get("strength", INITIAL_STRENGTH),
            carriers=d.get("carriers", []),
            retellings=d.get("retellings", 0),
            mutations=d.get("mutations", 0),
            codified_year=d.get("codified_year"),
        )

    @property
    def alive(self) -> bool:
        """A memory is alive if it has carriers and nonzero strength."""
        return len(self.carriers) > 0 and self.strength > 0.0

    def age(self, current_year: int) -> int:
        return current_year - self.origin_year


@dataclass
class CultureState:
    """Colony-level cultural state."""
    memories: list[CulturalMemory] = field(default_factory=list)
    next_id: int = 0
    total_created: int = 0
    total_promoted: int = 0
    total_died: int = 0

    def to_dict(self) -> dict:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "next_id": self.next_id,
            "total_created": self.total_created,
            "total_promoted": self.total_promoted,
            "total_died": self.total_died,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CultureState:
        return cls(
            memories=[CulturalMemory.from_dict(m) for m in d.get("memories", [])],
            next_id=d.get("next_id", 0),
            total_created=d.get("total_created", 0),
            total_promoted=d.get("total_promoted", 0),
            total_died=d.get("total_died", 0),
        )

    def living_memories(self) -> list[CulturalMemory]:
        return [m for m in self.memories if m.alive]

    def by_stage(self, stage: str) -> list[CulturalMemory]:
        return [m for m in self.living_memories() if m.stage == stage]


def create_memory_from_event(
    culture: CultureState,
    year: int,
    event_desc: str,
    narrative: str,
    witness_ids: list[str],
) -> CulturalMemory:
    """Create a new story from a significant event."""
    mem = CulturalMemory(
        id=f"mem-{culture.next_id}",
        origin_year=year,
        origin_event=event_desc,
        narrative=narrative,
        carriers=list(witness_ids),
    )
    culture.next_id += 1
    culture.total_created += 1
    culture.memories.append(mem)
    return mem


def tick_culture(
    culture: CultureState,
    year: int,
    active_ids: list[str],
    dead_ids: list[str],
    cooperation_pairs: list[tuple[str, str]],
    subsim_count: int,
    rng: random.Random,
) -> dict:
    """Advance cultural state by one year.

    Returns a summary dict of what happened culturally this year.
    """
    promoted: list[dict] = []
    retold: list[str] = []
    died: list[str] = []

    # --- Decay & carrier pruning ---
    for mem in culture.memories:
        # Remove dead carriers
        before_count = len(mem.carriers)
        mem.carriers = [c for c in mem.carriers if c in active_ids]
        lost = before_count - len(mem.carriers)
        if not mem.carriers:
            if mem.strength > 0.0:
                died.append(mem.id)
                culture.total_died += 1
            mem.strength = 0.0
            continue
        mem.strength = max(0.0, mem.strength - ANNUAL_DECAY)
        # Extra penalty per dead carrier
        mem.strength = max(0.0, mem.strength - DEATH_STRENGTH_PENALTY * lost)

    # --- Retelling via cooperation ---
    for a, b in cooperation_pairs:
        for mem in culture.living_memories():
            if a in mem.carriers and b not in mem.carriers:
                mem.carriers.append(b)
                mem.retellings += 1
                mem.strength = min(1.0, mem.strength + RETELL_RECOVERY)
                retold.append(mem.id)
            elif b in mem.carriers and a not in mem.carriers:
                mem.carriers.append(a)
                mem.retellings += 1
                mem.strength = min(1.0, mem.strength + RETELL_RECOVERY)
                retold.append(mem.id)
            elif a in mem.carriers and b in mem.carriers:
                if rng.random() < 0.2:
                    mem.mutations += 1
                    mem.retellings += 1
                    mem.strength = min(1.0, mem.strength + RETELL_RECOVERY * 0.5)
                    retold.append(mem.id)

    # --- Subsim boost: high subsim activity = cultural ferment ---
    if subsim_count > 3:
        for mem in culture.living_memories():
            if mem.stage in (STAGE_STORY, STAGE_MYTH):
                mem.strength = min(1.0, mem.strength + 0.02)

    # --- Promotions ---
    for mem in culture.living_memories():
        new_stage = _check_promotion(mem, year, len(active_ids))
        if new_stage and new_stage != mem.stage:
            old = mem.stage
            mem.stage = new_stage
            culture.total_promoted += 1
            promoted.append({"id": mem.id, "from": old, "to": new_stage})
            if new_stage == STAGE_NORM:
                mem.codified_year = year

    # --- Prune dead & cap ---
    culture.memories = [m for m in culture.memories if m.alive or m.stage == STAGE_NORM]
    if len(culture.memories) > MAX_MEMORIES:
        # Norms are always kept; cap applies only to non-norms
        norms = [m for m in culture.memories if m.stage == STAGE_NORM]
        others = [m for m in culture.memories if m.stage != STAGE_NORM]
        others.sort(key=lambda m: m.strength, reverse=True)
        cap = MAX_MEMORIES - len(norms)
        for m in others[cap:]:
            died.append(m.id)
            culture.total_died += 1
        culture.memories = norms + others[:cap]

    return {
        "year": year,
        "living_count": len(culture.living_memories()),
        "by_stage": {s: len(culture.by_stage(s)) for s in STAGES},
        "promoted": promoted,
        "retold_count": len(retold),
        "died_count": len(died),
    }


def _check_promotion(mem: CulturalMemory, year: int, population: int) -> str | None:
    """Check if a memory should be promoted to the next lifecycle stage."""
    age = mem.age(year)
    if mem.stage == STAGE_STORY:
        if age >= MYTH_MIN_AGE and mem.retellings >= MYTH_RETELL_THRESHOLD:
            return STAGE_MYTH
    elif mem.stage == STAGE_MYTH:
        carrier_fraction = len(mem.carriers) / max(1, population)
        if carrier_fraction >= RITUAL_CARRIER_FRACTION and mem.strength > 0.4:
            return STAGE_RITUAL
    elif mem.stage == STAGE_RITUAL:
        if age >= NORM_MIN_RITUAL_AGE and mem.strength > 0.5:
            return STAGE_NORM
    return None


def transmit_to_child(
    culture: CultureState,
    child_id: str,
    parent_ids: list[str],
    rng: random.Random,
) -> int:
    """Transmit cultural memories from parents to a newborn child.

    Each parent's strong memories have a chance of being passed on.
    Returns count of memories transmitted.
    """
    transmitted = 0
    for mem in culture.living_memories():
        parent_carriers = [p for p in parent_ids if p in mem.carriers]
        if not parent_carriers:
            continue
        chance = mem.strength * (0.5 + 0.25 * len(parent_carriers))
        if rng.random() < chance:
            if child_id not in mem.carriers:
                mem.carriers.append(child_id)
                transmitted += 1
    return transmitted


def generate_narratives(
    events: list[dict],
    deaths: list[dict],
    governance: dict | None,
    year: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Generate (event_desc, narrative) pairs for culturally significant events.

    Only events with severity > 0.5, deaths, or governance changes
    become stories.  Returns at most 2 per year.
    """
    candidates: list[tuple[str, str]] = []

    for ev in events:
        severity = ev.get("severity", 0.0)
        if severity > 0.5:
            desc = ev.get("description", ev.get("type", "unknown event"))
            templates = [
                f"The year the {desc} tested us",
                f"When {desc} struck and we endured",
                f"The {desc} of Year {year}",
            ]
            candidates.append((desc, rng.choice(templates)))

    for d in deaths:
        name = d.get("name", d.get("id", "unknown"))
        cause = d.get("cause", "unknown causes")
        templates = [
            f"We lost {name} to {cause} — never forget",
            f"The ballad of {name}, taken by {cause}",
            f"{name}'s last year — {cause} claimed them",
        ]
        candidates.append((f"death of {name}", rng.choice(templates)))

    if governance and governance.get("passed"):
        gov_type = governance.get("gov_type", "change")
        templates = [
            f"The day we chose {gov_type}",
            f"How {gov_type} came to govern us",
        ]
        candidates.append((f"governance shift to {gov_type}", rng.choice(templates)))

    rng.shuffle(candidates)
    return candidates[:2]


def cultural_summary_for_emergence(culture: CultureState, total_years: int) -> dict:
    """Produce a summary suitable for emergence analysis."""
    norms = culture.by_stage(STAGE_NORM)
    rituals = culture.by_stage(STAGE_RITUAL)
    myths = culture.by_stage(STAGE_MYTH)
    stories = culture.by_stage(STAGE_STORY)
    oldest = min(
        (m for m in culture.living_memories()),
        key=lambda m: m.origin_year,
        default=None,
    )
    most_retold = max(
        (m for m in culture.memories if m.retellings > 0),
        key=lambda m: m.retellings,
        default=None,
    )
    most_mutated = max(
        (m for m in culture.memories if m.mutations > 0),
        key=lambda m: m.mutations,
        default=None,
    )
    return {
        "total_created": culture.total_created,
        "total_promoted": culture.total_promoted,
        "total_died": culture.total_died,
        "living": len(culture.living_memories()),
        "by_stage": {
            STAGE_STORY: len(stories),
            STAGE_MYTH: len(myths),
            STAGE_RITUAL: len(rituals),
            STAGE_NORM: len(norms),
        },
        "norms": [m.to_dict() for m in norms],
        "oldest_memory": oldest.to_dict() if oldest else None,
        "most_retold": most_retold.to_dict() if most_retold else None,
        "most_mutated": most_mutated.to_dict() if most_mutated else None,
        "survival_rate": (
            round(len(culture.living_memories()) / max(1, culture.total_created), 3)
            if culture.total_created > 0 else 0.0
        ),
    }
