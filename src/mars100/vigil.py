"""
Vigil organ for Mars-100 (engine v12.0).

The Vigil watches for *channels of life* going silent.  A channel is any
recurrent practice the colony depends on — a skill that's atrophying, an
action no one performs anymore, a faction that's lost all members, a
tradition no one keeps.  When a channel flatlines for ``DORMANCY_THRESHOLD``
years, the Vigil flags it as "dormant" and starts pushing revival pressure
into next year's action selection.  If it stays silent past
``EXTINCTION_THRESHOLD`` years, it's marked extinct and the pressure
decays — the colony has moved on.

This is the colony-scale analog of the channel_health monitor in the parent
seed: instead of dead subrappters with zero posts in N frames, we track
dead practices with zero adoption in N Martian years.  The output is
identical in shape — per-channel vitals, plus revival prompts that feed
the next tick.

The organ never deletes anything.  Dormant entities are remembered;
extinct entities become part of the cultural archive.  Death is data.

RNG offset: ``seed + 13591``

Wire-up
-------
- Ticked LAST in the year so all other organs' activity counts as a
  spark.  See ``Mars100Engine.tick``.
- ``compute_revival_pressure`` is summed with cultural / economic /
  behavioral / diplomatic pressures during ``_choose_action`` next year.
- Per-channel vitals are written to ``YearResult.vigil`` for the
  ``state/mars100.json`` snapshot — that's what the archivist reads.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DORMANCY_THRESHOLD = 10
EXTINCTION_THRESHOLD = 25

SKILL_SPARK_THRESHOLD = 0.10
TRADITION_SPARK_IMPORTANCE = 0.20
FACTION_SPARK_SIZE = 1

REVIVAL_NUDGE_BASE = 0.04
REVIVAL_NUDGE_MAX = 0.20
EXTINCTION_NUDGE_DECAY = 0.5

SKILL_TO_ACTION = {
    "terraforming": "terraform",
    "hydroponics":  "farm",
    "mediation":    "mediate",
    "coding":       "code",
    "prayer":       "pray",
}

DEFAULT_ACTIONS = (
    "terraform", "farm", "mediate", "code", "pray",
    "cooperate", "explore", "rest", "research",
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ChannelVitals:
    kind: str
    name: str
    last_spark_year: int
    silent_for: int = 0
    status: str = "alive"
    revivals: int = 0
    flatlines: int = 0
    born_year: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "name": self.name,
            "last_spark_year": self.last_spark_year,
            "silent_for": self.silent_for, "status": self.status,
            "revivals": self.revivals, "flatlines": self.flatlines,
            "born_year": self.born_year,
            "notes": list(self.notes[-3:]),
        }


@dataclass
class VigilState:
    channels: dict[str, ChannelVitals] = field(default_factory=dict)
    total_flatlines: int = 0
    total_revivals: int = 0
    total_extinctions: int = 0

    def to_dict(self) -> dict:
        return {
            "channels": {k: v.to_dict() for k, v in self.channels.items()},
            "totals": {
                "flatlines":   self.total_flatlines,
                "revivals":    self.total_revivals,
                "extinctions": self.total_extinctions,
                "alive":       sum(1 for v in self.channels.values()
                                   if v.status == "alive"),
                "dormant":     sum(1 for v in self.channels.values()
                                   if v.status == "dormant"),
                "extinct":     sum(1 for v in self.channels.values()
                                   if v.status == "extinct"),
            },
        }


@dataclass
class VigilTickResult:
    year: int
    newly_dormant: list[dict] = field(default_factory=list)
    newly_extinct: list[dict] = field(default_factory=list)
    newly_revived: list[dict] = field(default_factory=list)
    revival_prompts: list[dict] = field(default_factory=list)
    channel_snapshot: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "newly_dormant":  list(self.newly_dormant),
            "newly_extinct":  list(self.newly_extinct),
            "newly_revived":  list(self.newly_revived),
            "revival_prompts": list(self.revival_prompts),
            "channel_snapshot": dict(self.channel_snapshot),
        }


# ---------------------------------------------------------------------------
# Spark detection
# ---------------------------------------------------------------------------

def _skill_sparks(active_colonists: list[dict]) -> set[str]:
    sparked: set[str] = set()
    for c in active_colonists:
        skills = c.get("skills", {}) or {}
        for name, val in skills.items():
            if isinstance(val, (int, float)) and val >= SKILL_SPARK_THRESHOLD:
                sparked.add(name)
    return sparked


def _action_sparks(actions: dict[str, str]) -> set[str]:
    return {a for a in actions.values() if a}


def _faction_sparks(diplomacy: dict | None) -> set[str]:
    if not diplomacy:
        return set()
    sparked: set[str] = set()
    for fac in diplomacy.get("factions", []) or []:
        size = fac.get("size") or len(fac.get("members", []) or [])
        if size >= FACTION_SPARK_SIZE:
            fid = fac.get("id") or fac.get("name")
            if fid:
                sparked.add(str(fid))
    return sparked


def _tradition_sparks(culture: dict | None) -> set[str]:
    if not culture:
        return set()
    sparked: set[str] = set()
    for t in culture.get("traditions", []) or []:
        name = t.get("name")
        imp = t.get("importance", 0.0)
        if name and isinstance(imp, (int, float)) and imp >= TRADITION_SPARK_IMPORTANCE:
            sparked.add(str(name))
    return sparked


# ---------------------------------------------------------------------------
# Core tick
# ---------------------------------------------------------------------------

def tick_vigil(
    state: VigilState,
    year: int,
    active_colonists: list[dict],
    actions: dict[str, str],
    action_channel_list: Iterable[str],
    diplomacy: dict | None = None,
    culture: dict | None = None,
    rng: random.Random | None = None,
) -> VigilTickResult:
    """Advance the Vigil one year."""
    rng = rng or random.Random(year)
    result = VigilTickResult(year=year)

    skill_sparked = _skill_sparks(active_colonists)
    action_sparked = _action_sparks(actions)
    faction_sparked = _faction_sparks(diplomacy)
    tradition_sparked = _tradition_sparks(culture)

    discovered_skills: set[str] = set()
    for c in active_colonists:
        for name in (c.get("skills", {}) or {}).keys():
            discovered_skills.add(name)
    for skill in discovered_skills:
        _ensure_channel(state, kind="skill", name=skill, year=year)

    for act in action_channel_list:
        _ensure_channel(state, kind="action", name=act, year=year)

    for fid in faction_sparked:
        _ensure_channel(state, kind="faction", name=fid, year=year)

    if culture:
        for t in culture.get("traditions", []) or []:
            name = t.get("name")
            if name:
                _ensure_channel(state, kind="tradition", name=str(name),
                                year=t.get("source_year", year))

    sparks = {
        "skill":     skill_sparked,
        "action":    action_sparked,
        "faction":   faction_sparked,
        "tradition": tradition_sparked,
    }

    for vitals in state.channels.values():
        prior_status = vitals.status
        if vitals.name in sparks.get(vitals.kind, set()):
            vitals.last_spark_year = year
            vitals.silent_for = 0
            if prior_status in ("dormant", "extinct"):
                vitals.status = "revived"
                vitals.revivals += 1
                vitals.notes.append(f"revived in year {year}")
                state.total_revivals += 1
                result.newly_revived.append(_event(vitals))
            else:
                vitals.status = "alive"
        else:
            vitals.silent_for += 1
            if (vitals.silent_for >= EXTINCTION_THRESHOLD
                    and prior_status != "extinct"):
                vitals.status = "extinct"
                vitals.notes.append(f"extinct in year {year}")
                state.total_extinctions += 1
                result.newly_extinct.append(_event(vitals))
            elif (vitals.silent_for >= DORMANCY_THRESHOLD
                  and prior_status == "alive"):
                vitals.status = "dormant"
                vitals.flatlines += 1
                vitals.notes.append(f"flatlined in year {year}")
                state.total_flatlines += 1
                result.newly_dormant.append(_event(vitals))

    for vitals in state.channels.values():
        if vitals.status not in ("dormant", "extinct"):
            continue
        action = _channel_to_action(vitals)
        if action is None:
            continue
        decay = (EXTINCTION_NUDGE_DECAY if vitals.status == "extinct" else 1.0)
        strength = min(REVIVAL_NUDGE_MAX,
                       REVIVAL_NUDGE_BASE * (1 + vitals.silent_for / 10)) * decay
        result.revival_prompts.append({
            "channel_kind": vitals.kind,
            "channel_name": vitals.name,
            "action":       action,
            "strength":     round(strength, 4),
            "status":       vitals.status,
            "silent_for":   vitals.silent_for,
            "prompt": _prompt_text(vitals, action),
        })

    result.channel_snapshot = {k: v.to_dict() for k, v in state.channels.items()}
    return result


def compute_revival_pressure(
    state: VigilState,
    actions_in_play: Iterable[str],
) -> dict[str, float]:
    """Convert dormant-channel prompts into action weight perturbations."""
    in_play = set(actions_in_play)
    pressure: dict[str, float] = {}
    for vitals in state.channels.values():
        if vitals.status not in ("dormant", "extinct"):
            continue
        action = _channel_to_action(vitals)
        if action is None or action not in in_play:
            continue
        decay = (EXTINCTION_NUDGE_DECAY if vitals.status == "extinct" else 1.0)
        nudge = min(REVIVAL_NUDGE_MAX,
                    REVIVAL_NUDGE_BASE * (1 + vitals.silent_for / 10)) * decay
        pressure[action] = pressure.get(action, 0.0) + nudge
    for a in list(pressure.keys()):
        pressure[a] = max(-REVIVAL_NUDGE_MAX,
                          min(REVIVAL_NUDGE_MAX, pressure[a]))
    return pressure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _channel_key(kind: str, name: str) -> str:
    return f"{kind}:{name}"


def _ensure_channel(state: VigilState, *, kind: str, name: str, year: int) -> ChannelVitals:
    key = _channel_key(kind, name)
    v = state.channels.get(key)
    if v is None:
        v = ChannelVitals(kind=kind, name=name, last_spark_year=year,
                          born_year=year)
        state.channels[key] = v
    return v


def _channel_to_action(vitals: ChannelVitals) -> str | None:
    if vitals.kind == "action":
        if vitals.name in ("sabotage", "hoard"):
            return None
        return vitals.name
    if vitals.kind == "skill":
        return SKILL_TO_ACTION.get(vitals.name)
    if vitals.kind == "tradition":
        return "cooperate"
    if vitals.kind == "faction":
        return "mediate"
    return None


def _event(v: ChannelVitals) -> dict:
    return {
        "kind": v.kind, "name": v.name, "status": v.status,
        "silent_for": v.silent_for, "last_spark_year": v.last_spark_year,
    }


def _prompt_text(v: ChannelVitals, action: str) -> str:
    if v.status == "extinct":
        return (f"The {v.kind} '{v.name}' has been silent for "
                f"{v.silent_for} years.  Considered lost — but {action} "
                f"could still echo it back.")
    return (f"The {v.kind} '{v.name}' has gone dormant "
            f"({v.silent_for} years silent).  Revive via {action}.")
