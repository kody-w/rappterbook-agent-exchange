"""
Memory Ecology — intergenerational cultural transmission for Mars-100.

When colonists die, their most intense memories become ancestral memories.
Living colonists can inherit, teach, and dream ancestral memories.
Shared themes crystallize into colony-wide knowledge bonuses.
Amendment X made concrete: legacy, not delete.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AncestralMemory:
    """A memory preserved from a dead colonist."""
    colonist_id: str
    colonist_name: str
    year_recorded: int
    year_archived: int
    event: str
    emotional_valence: float
    theme: str
    legacy_strength: float

    def to_dict(self) -> dict:
        return {
            "colonist_id": self.colonist_id,
            "colonist_name": self.colonist_name,
            "year_recorded": self.year_recorded,
            "year_archived": self.year_archived,
            "event": self.event,
            "valence": self.emotional_valence,
            "theme": self.theme,
            "legacy_strength": round(self.legacy_strength, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AncestralMemory":
        return cls(
            colonist_id=d["colonist_id"],
            colonist_name=d["colonist_name"],
            year_recorded=d["year_recorded"],
            year_archived=d["year_archived"],
            event=d["event"],
            emotional_valence=d.get("valence", d.get("emotional_valence", 0.0)),
            theme=d["theme"],
            legacy_strength=d["legacy_strength"],
        )


@dataclass
class CrystalKnowledge:
    """Colony-wide knowledge crystallized from shared ancestral themes."""
    theme: str
    strength: float
    contributors: list
    year_formed: int
    stat_bonuses: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "theme": self.theme,
            "strength": round(self.strength, 4),
            "contributors": self.contributors,
            "year_formed": self.year_formed,
            "stat_bonuses": {k: round(v, 4) for k, v in self.stat_bonuses.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CrystalKnowledge":
        return cls(
            theme=d["theme"], strength=d["strength"],
            contributors=d["contributors"], year_formed=d["year_formed"],
            stat_bonuses=d.get("stat_bonuses", {}),
        )


@dataclass
class ColonyMemoryBank:
    """The colony collective memory — ancestors, crystals, dreams."""
    ancestral_memories: list = field(default_factory=list)
    crystals: list = field(default_factory=list)
    total_dreams: int = 0
    total_teachings: int = 0

    def to_dict(self) -> dict:
        return {
            "ancestral_memories": [m.to_dict() for m in self.ancestral_memories],
            "crystals": [c.to_dict() for c in self.crystals],
            "stats": {
                "total_ancestors": len(self.ancestral_memories),
                "total_crystals": len(self.crystals),
                "total_dreams": self.total_dreams,
                "total_teachings": self.total_teachings,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ColonyMemoryBank":
        bank = cls()
        bank.ancestral_memories = [
            AncestralMemory.from_dict(m) for m in d.get("ancestral_memories", [])
        ]
        bank.crystals = [
            CrystalKnowledge.from_dict(c) for c in d.get("crystals", [])
        ]
        stats = d.get("stats", {})
        bank.total_dreams = stats.get("total_dreams", 0)
        bank.total_teachings = stats.get("total_teachings", 0)
        return bank


THEME_KEYWORDS = {
    "survival": ["storm", "resource", "food", "water", "power", "air",
                  "failure", "breach", "radiation", "deprivation", "equipment"],
    "social": ["conflict", "cooperation", "trust", "betray", "exile",
               "alliance", "mediation", "vote", "council"],
    "cosmic": ["signal", "alien", "earth", "simulation", "meta", "dream",
               "interpreter", "pattern", "authored"],
    "growth": ["birth", "child", "teach", "learn", "evolve", "discover",
               "terraform", "build", "farm"],
}


def classify_memory_theme(event):
    """Classify a memory event into a theme based on keywords."""
    lower = event.lower()
    scores = {}
    for theme, keywords in THEME_KEYWORDS.items():
        scores[theme] = sum(1 for kw in keywords if kw in lower)
    best = max(scores, key=lambda t: scores[t])
    return best if scores[best] > 0 else "mundane"


def archive_dead_colonist(colonist, year, bank, max_memories=5):
    """Archive a dead colonist most intense memories into the colony bank."""
    if not hasattr(colonist, "memories") or not colonist.memories:
        return 0
    sorted_mems = sorted(
        colonist.memories,
        key=lambda m: abs(m.emotional_valence),
        reverse=True,
    )
    archived = 0
    legacy = min(1.0, len(colonist.memories) / 50.0)
    for mem in sorted_mems[:max_memories]:
        theme = classify_memory_theme(mem.event)
        am = AncestralMemory(
            colonist_id=colonist.id,
            colonist_name=colonist.name,
            year_recorded=mem.year,
            year_archived=year,
            event=mem.event,
            emotional_valence=mem.emotional_valence,
            theme=theme,
            legacy_strength=legacy,
        )
        bank.ancestral_memories.append(am)
        archived += 1
    return archived


def inherit_parent_memories(child, parents, bank, rng, max_inherit=3):
    """A newborn inherits dampened memories from parents and ancestors."""
    parent_ids = {p.id for p in parents}
    relevant = [m for m in bank.ancestral_memories if m.colonist_id in parent_ids]
    others = [m for m in bank.ancestral_memories if m.colonist_id not in parent_ids]
    candidates = relevant + (rng.sample(others, min(2, len(others))) if others else [])
    if not candidates:
        return 0
    rng.shuffle(candidates)
    inherited = 0
    for am in candidates[:max_inherit]:
        dampened_valence = am.emotional_valence * 0.3 * am.legacy_strength
        child.add_memory(am.year_recorded, "[inherited] " + am.event, dampened_valence)
        inherited += 1
    return inherited


def teach_memory(teacher, student, bank, rng):
    """Teacher shares their most vivid memory with student."""
    if not hasattr(teacher, "memories") or not teacher.memories:
        return False
    sorted_mems = sorted(
        teacher.memories,
        key=lambda m: abs(m.emotional_valence),
        reverse=True,
    )
    mem = sorted_mems[0]
    dampened = mem.emotional_valence * 0.8
    student.add_memory(mem.year, "[taught by " + teacher.name + "] " + mem.event, dampened)
    bank.total_teachings += 1
    return True


def dream_ancestral(colonist, bank, rng):
    """Colonist dreams of an ancestor, gaining a faint memory echo."""
    if not bank.ancestral_memories:
        return None
    faith = getattr(colonist.stats, "faith", 0.5)
    dream_prob = 0.1 + faith * 0.2
    if rng.random() > dream_prob:
        return None
    weights = [m.legacy_strength + 0.01 for m in bank.ancestral_memories]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    chosen = bank.ancestral_memories[0]
    for m, w in zip(bank.ancestral_memories, weights):
        cumulative += w
        if r <= cumulative:
            chosen = m
            break
    dampened = chosen.emotional_valence * 0.2 * chosen.legacy_strength
    colonist.add_memory(
        chosen.year_recorded,
        "[dream of " + chosen.colonist_name + "] " + chosen.event,
        dampened,
    )
    bank.total_dreams += 1
    return (colonist.name + " dreams of " + chosen.colonist_name + ": "
            + "'" + chosen.event + "' (year " + str(chosen.year_recorded) + ")")


def attempt_crystallization(bank, year, min_contributors=3):
    """If enough ancestors share a theme, crystallize into colony knowledge."""
    existing_themes = {c.theme for c in bank.crystals}
    theme_groups = {}
    for mem in bank.ancestral_memories:
        if mem.theme != "mundane" and mem.theme not in existing_themes:
            theme_groups.setdefault(mem.theme, []).append(mem)
    for theme, mems in theme_groups.items():
        unique_colonists = {m.colonist_id for m in mems}
        if len(unique_colonists) >= min_contributors:
            avg_strength = sum(m.legacy_strength for m in mems) / len(mems)
            bonuses = _theme_to_bonuses(theme, avg_strength)
            crystal = CrystalKnowledge(
                theme=theme, strength=avg_strength,
                contributors=list(unique_colonists),
                year_formed=year, stat_bonuses=bonuses,
            )
            bank.crystals.append(crystal)
            return crystal
    return None


def _theme_to_bonuses(theme, strength):
    """Map a theme to stat bonuses scaled by strength."""
    mapping = {
        "survival": {"resolve": 0.02, "improvisation": 0.01},
        "social": {"empathy": 0.02, "resolve": 0.01},
        "cosmic": {"faith": 0.02, "improvisation": 0.01},
        "growth": {"empathy": 0.01, "resolve": 0.01, "improvisation": 0.01},
    }
    base = mapping.get(theme, {"resolve": 0.01})
    return {stat: val * strength for stat, val in base.items()}


def apply_crystal_bonuses(colonists, bank):
    """Apply crystallized knowledge bonuses to all active colonists."""
    for crystal in bank.crystals:
        for colonist in colonists:
            if not colonist.is_active():
                continue
            for stat_name, bonus in crystal.stat_bonuses.items():
                current = getattr(colonist.stats, stat_name, None)
                if current is not None:
                    new_val = min(1.0, current + bonus)
                    setattr(colonist.stats, stat_name, new_val)


def decay_memories(bank, rate=0.05):
    """Decay ancestral memories and crystal strengths. Returns memories removed."""
    removed = 0
    surviving = []
    for mem in bank.ancestral_memories:
        mem.legacy_strength *= (1.0 - rate)
        if mem.legacy_strength > 0.01:
            surviving.append(mem)
        else:
            removed += 1
    bank.ancestral_memories = surviving
    for crystal in bank.crystals:
        crystal.strength *= (1.0 - rate * 0.5)
        crystal.stat_bonuses = {
            k: v * (1.0 - rate * 0.5) for k, v in crystal.stat_bonuses.items()
        }
    bank.crystals = [c for c in bank.crystals if c.strength > 0.005]
    return removed


def colony_instinct(bank):
    """Compute the colony collective instinct from ancestral memory themes."""
    theme_weight = {}
    for mem in bank.ancestral_memories:
        theme_weight[mem.theme] = theme_weight.get(mem.theme, 0.0) + mem.legacy_strength
    total = sum(theme_weight.values()) or 1.0
    return {t: w / total for t, w in theme_weight.items()}


def memory_ecology_summary(bank):
    """Return a summary dict of the memory ecology state."""
    return {
        "total_ancestors": len(bank.ancestral_memories),
        "total_crystals": len(bank.crystals),
        "total_dreams": bank.total_dreams,
        "total_teachings": bank.total_teachings,
        "themes": colony_instinct(bank),
        "crystals": [c.to_dict() for c in bank.crystals],
        "strongest_ancestor": (
            max(bank.ancestral_memories,
                key=lambda m: m.legacy_strength).to_dict()
            if bank.ancestral_memories else None
        ),
    }
