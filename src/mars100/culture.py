"""Cultural Memory organ for Mars-100 colony simulation."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

MAX_TRADITIONS = 20
MAX_ORAL_HISTORY = 15
MAX_MARTYRS = 10
MAX_TABOOS = 10
PRESSURE_BOUND = 0.5


@dataclass
class Tradition:
    name: str
    source_year: int
    source_type: str
    importance: float = 1.0
    description: str = ""


@dataclass
class OralHistory:
    year: int
    event_type: str
    severity: float
    narrative: str


@dataclass
class Martyr:
    colonist_id: str
    year: int
    dominant_trait: str
    description: str = ""


@dataclass
class Taboo:
    action: str
    source_year: int
    strength: float = 1.0
    reason: str = ""


@dataclass
class YearContext:
    year: int
    event_type: str
    event_severity: float
    deaths: list
    exiles: list
    governance_proposals: list
    subsim_count: int
    action_counts: dict
    resources: dict
    colonists: list


class CulturalMemory:
    def __init__(self) -> None:
        self.traditions: list[Tradition] = []
        self.oral_history: list[OralHistory] = []
        self.martyrs: list[Martyr] = []
        self.taboos: list[Taboo] = []

    def summary(self) -> dict:
        return {
            "traditions": len(self.traditions),
            "oral_history": len(self.oral_history),
            "martyrs": len(self.martyrs),
            "taboos": len(self.taboos),
            "tradition_names": [t.name for t in self.traditions],
        }

    def to_dict(self) -> dict:
        return {
            "traditions": [
                {"name": t.name, "source_year": t.source_year,
                 "source_type": t.source_type, "importance": t.importance,
                 "description": t.description}
                for t in self.traditions
            ],
            "oral_history": [
                {"year": o.year, "event_type": o.event_type,
                 "severity": o.severity, "narrative": o.narrative}
                for o in self.oral_history
            ],
            "martyrs": [
                {"colonist_id": m.colonist_id, "year": m.year,
                 "dominant_trait": m.dominant_trait, "description": m.description}
                for m in self.martyrs
            ],
            "taboos": [
                {"action": t.action, "source_year": t.source_year,
                 "strength": t.strength, "reason": t.reason}
                for t in self.taboos
            ],
        }


def evolve_culture(culture, ctx, rng):
    """Evolve cultural memory for one year."""
    if ctx.event_severity > 0.5:
        culture.oral_history.append(OralHistory(
            year=ctx.year, event_type=ctx.event_type,
            severity=ctx.event_severity,
            narrative="Year {}: the great {} (severity {:.2f})".format(
                ctx.year, ctx.event_type, ctx.event_severity),
        ))
    for death in ctx.deaths:
        cid = death.get("colonist_id", "unknown")
        trait = death.get("dominant_trait", "resolve")
        culture.martyrs.append(Martyr(
            colonist_id=cid, year=ctx.year, dominant_trait=trait,
            description="{} fell in year {}".format(cid, ctx.year),
        ))
    for exile in ctx.exiles:
        eid = exile.get("colonist_id", "unknown")
        culture.taboos.append(Taboo(
            action="sabotage", source_year=ctx.year, strength=1.0,
            reason="Exile of {} in year {}".format(eid, ctx.year),
        ))
    for prop in ctx.governance_proposals:
        if prop.get("passed"):
            name = prop.get("title", "Edict of Year {}".format(ctx.year))
            culture.traditions.append(Tradition(
                name=name, source_year=ctx.year, source_type="governance",
                importance=1.0,
                description="Passed in year {}".format(ctx.year),
            ))
    existing = {t.name for t in culture.traditions}
    if ctx.subsim_count >= 2 and "Consulting the Oracle" not in existing:
        culture.traditions.append(Tradition(
            name="Consulting the Oracle", source_year=ctx.year,
            source_type="subsim", importance=0.8,
            description="Colonists run simulations before big decisions",
        ))
    cooperate = ctx.action_counts.get("cooperate", 0)
    total_acts = max(sum(ctx.action_counts.values()), 1)
    if (cooperate / total_acts) > 0.5 and "The Common Table" not in existing:
        culture.traditions.append(Tradition(
            name="The Common Table", source_year=ctx.year,
            source_type="cooperation", importance=0.9,
            description="A tradition of sharing and cooperation",
        ))
    for resource, level in ctx.resources.items():
        if level < 0.15:
            tname = "Conserve {}".format(resource)
            if tname not in existing:
                culture.traditions.append(Tradition(
                    name=tname, source_year=ctx.year,
                    source_type="scarcity", importance=0.7,
                    description="Born from {} scarcity in year {}".format(resource, ctx.year),
                ))
    for t in culture.traditions:
        age = ctx.year - t.source_year
        if age > 20:
            t.importance *= 0.98
    _enforce_bounds(culture)


def compute_cultural_pressure(culture):
    """Compute action-weight modifiers from cultural state."""
    pressure = {}
    for taboo in culture.taboos:
        key = taboo.action
        pressure[key] = pressure.get(key, 0.0) - 0.15 * taboo.strength
    trait_to_action = {
        "resolve": "cooperate", "improvisation": "innovate",
        "empathy": "cooperate", "hoarding": "hoard",
        "faith": "pray", "paranoia": "sabotage",
    }
    for martyr in culture.martyrs:
        action = trait_to_action.get(martyr.dominant_trait, "cooperate")
        pressure[action] = pressure.get(action, 0.0) + 0.05
    for key in pressure:
        pressure[key] = max(-PRESSURE_BOUND, min(PRESSURE_BOUND, pressure[key]))
    return pressure


def transmit_to_child(culture, rng):
    """Create a child culture for a new generation."""
    child = CulturalMemory()
    for t in culture.traditions:
        if rng.random() < t.importance:
            child.traditions.append(Tradition(
                name=t.name, source_year=t.source_year,
                source_type=t.source_type, importance=t.importance * 0.9,
                description=t.description,
            ))
    for tab in culture.taboos:
        if rng.random() < tab.strength:
            child.taboos.append(Taboo(
                action=tab.action, source_year=tab.source_year,
                strength=tab.strength * 0.85, reason=tab.reason,
            ))
    return child


def _enforce_bounds(culture):
    """Trim collections to maximum sizes."""
    if len(culture.traditions) > MAX_TRADITIONS:
        culture.traditions.sort(key=lambda t: t.importance, reverse=True)
        culture.traditions = culture.traditions[:MAX_TRADITIONS]
    if len(culture.oral_history) > MAX_ORAL_HISTORY:
        culture.oral_history = culture.oral_history[-MAX_ORAL_HISTORY:]
    if len(culture.martyrs) > MAX_MARTYRS:
        culture.martyrs = culture.martyrs[-MAX_MARTYRS:]
    if len(culture.taboos) > MAX_TABOOS:
        culture.taboos.sort(key=lambda t: t.strength, reverse=True)
        culture.taboos = culture.taboos[:MAX_TABOOS]
