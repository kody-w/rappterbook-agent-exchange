"""
Diplomacy engine for Mars-100.

Detects factions from colonist stats, manages alliances and rivalries,
and modifies action weights so colonists coordinate (or compete) as groups.
Crisis protocol suspends hostilities when any resource drops critically low.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist, STAT_NAMES
from src.mars100.colony import Resources, RESOURCE_NAMES

# --- Constants -----------------------------------------------------------

MIN_FACTION_SIZE = 2
FORMATION_YEARS = 2       # proto-faction must be stable this long
CORE_RETENTION = 0.5      # existing faction survives if >= 50% core remains

PACT_DURATION = 10        # years an alliance/embargo lasts
EMBARGO_DURATION = 5

ALLIANCE_COOP_BOOST = 1.5
RIVALRY_SABOTAGE_BOOST = 1.3
CRISIS_THRESHOLD = 0.15   # resource level that triggers crisis protocol
CRISIS_COOP_BOOST = 2.0

STAT_TO_FACTION = {
    "resolve":        "The Resolute",
    "improvisation":  "The Inventors",
    "empathy":        "The Empaths",
    "hoarding":       "The Hoarders",
    "faith":          "The Faithful",
    "paranoia":       "The Watchful",
}


# --- Data classes --------------------------------------------------------

@dataclass
class Faction:
    """A detected colonist bloc with a shared dominant stat."""
    name: str
    dominant_stat: str
    member_ids: list[str] = field(default_factory=list)
    formed_year: int = 0
    stable_years: int = 0
    solidified: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dominant_stat": self.dominant_stat,
            "member_ids": list(self.member_ids),
            "formed_year": self.formed_year,
            "stable_years": self.stable_years,
            "solidified": self.solidified,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Faction:
        return cls(
            name=d["name"], dominant_stat=d["dominant_stat"],
            member_ids=list(d.get("member_ids", [])),
            formed_year=d.get("formed_year", 0),
            stable_years=d.get("stable_years", 0),
            solidified=d.get("solidified", False),
        )


@dataclass
class Pact:
    """An alliance or rivalry (embargo) between two factions."""
    faction_a: str
    faction_b: str
    kind: str           # "alliance" | "embargo"
    start_year: int
    duration: int       # years remaining
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "faction_a": self.faction_a, "faction_b": self.faction_b,
            "kind": self.kind, "start_year": self.start_year,
            "duration": self.duration, "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Pact:
        return cls(**d)


@dataclass
class DiplomacyState:
    """Persistent diplomacy state across simulation years."""
    factions: list[Faction] = field(default_factory=list)
    pacts: list[Pact] = field(default_factory=list)
    crisis_active: bool = False
    crisis_years: int = 0
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factions": [f.to_dict() for f in self.factions],
            "pacts": [p.to_dict() for p in self.pacts],
            "crisis_active": self.crisis_active,
            "crisis_years": self.crisis_years,
            "history_len": len(self.history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiplomacyState:
        return cls(
            factions=[Faction.from_dict(f) for f in d.get("factions", [])],
            pacts=[Pact.from_dict(p) for p in d.get("pacts", [])],
            crisis_active=d.get("crisis_active", False),
            crisis_years=d.get("crisis_years", 0),
        )

    def faction_of(self, colonist_id: str) -> Faction | None:
        """Return the faction a colonist belongs to, if any."""
        for f in self.factions:
            if colonist_id in f.member_ids:
                return f
        return None

    def pact_between(self, fa: str, fb: str) -> Pact | None:
        """Return active pact between two faction names, if any."""
        for p in self.pacts:
            if {p.faction_a, p.faction_b} == {fa, fb}:
                return p
        return None

    def summary(self, year: int) -> dict:
        """Compact summary for serialization into YearResult."""
        return {
            "year": year,
            "factions": [f.to_dict() for f in self.factions],
            "pacts": [p.to_dict() for p in self.pacts],
            "crisis_active": self.crisis_active,
            "crisis_years": self.crisis_years,
            "num_factions": len(self.factions),
            "num_solidified": sum(1 for f in self.factions if f.solidified),
            "num_pacts": len(self.pacts),
        }


@dataclass
class DiplomacyYearResult:
    """What happened diplomatically in one year."""
    factions_detected: list[dict]
    pacts_formed: list[dict]
    pacts_expired: list[dict]
    crisis_active: bool
    action_modifiers: dict[str, dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "factions_detected": self.factions_detected,
            "pacts_formed": self.pacts_formed,
            "pacts_expired": self.pacts_expired,
            "crisis_active": self.crisis_active,
            "modifier_count": len(self.action_modifiers),
        }


# --- Core functions ------------------------------------------------------

def detect_factions(
    colonists: list[Colonist],
    state: DiplomacyState,
    year: int,
) -> list[dict]:
    """Detect factions from colonist dominant stats.

    Groups active colonists by dominant stat. Proto-factions need
    FORMATION_YEARS of stability before solidifying. Existing solidified
    factions persist if CORE_RETENTION of members remain.

    Returns a list of faction-event dicts for logging.
    """
    groups: dict[str, list[str]] = {}
    for c in colonists:
        if not c.is_active():
            continue
        dominant = c.stats.dominant()
        groups.setdefault(dominant, []).append(c.id)

    events: list[dict] = []
    seen_names: set[str] = set()

    for stat, members in groups.items():
        if len(members) < MIN_FACTION_SIZE:
            continue
        name = STAT_TO_FACTION.get(stat, f"The {stat.title()}")
        seen_names.add(name)

        existing = next((f for f in state.factions if f.name == name), None)
        if existing:
            old_set = set(existing.member_ids)
            new_set = set(members)
            overlap = old_set & new_set
            if len(overlap) >= max(1, len(old_set) * CORE_RETENTION):
                existing.member_ids = members
                existing.stable_years += 1
                if not existing.solidified and existing.stable_years >= FORMATION_YEARS:
                    existing.solidified = True
                    events.append({
                        "type": "faction_solidified", "name": name,
                        "year": year, "members": members,
                    })
            else:
                existing.member_ids = members
                existing.stable_years = 0
                existing.solidified = False
                events.append({
                    "type": "faction_reformed", "name": name,
                    "year": year, "members": members,
                })
        else:
            faction = Faction(
                name=name, dominant_stat=stat, member_ids=members,
                formed_year=year, stable_years=1,
            )
            state.factions.append(faction)
            events.append({
                "type": "faction_detected", "name": name,
                "year": year, "members": members,
            })

    # Prune factions that no longer meet minimum size
    state.factions = [f for f in state.factions if f.name in seen_names]

    return events


def check_crisis(resources: Resources) -> list[str]:
    """Return list of resources below crisis threshold."""
    critical: list[str] = []
    for name in RESOURCE_NAMES:
        if getattr(resources, name) < CRISIS_THRESHOLD:
            critical.append(name)
    return critical


def manage_pacts(
    state: DiplomacyState,
    year: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Form new pacts between solidified factions and expire old ones.

    Returns (formed, expired) event lists.
    """
    formed: list[dict] = []
    expired: list[dict] = []

    # Expire old pacts
    surviving: list[Pact] = []
    for pact in state.pacts:
        pact.duration -= 1
        if pact.duration <= 0:
            expired.append(pact.to_dict())
        else:
            surviving.append(pact)
    state.pacts = surviving

    # Form new pacts between solidified factions without existing pacts
    solidified = [f for f in state.factions if f.solidified]
    for i, fa in enumerate(solidified):
        for fb in solidified[i + 1:]:
            if state.pact_between(fa.name, fb.name) is not None:
                continue
            if _factions_compatible(fa, fb):
                kind = "alliance"
                dur = PACT_DURATION
                reason = f"Shared values: {fa.dominant_stat} ↔ {fb.dominant_stat}"
            elif rng.random() < 0.4:
                kind = "embargo"
                dur = EMBARGO_DURATION
                reason = f"Value clash: {fa.dominant_stat} vs {fb.dominant_stat}"
            else:
                continue
            pact = Pact(
                faction_a=fa.name, faction_b=fb.name,
                kind=kind, start_year=year, duration=dur, reason=reason,
            )
            state.pacts.append(pact)
            formed.append(pact.to_dict())

    return formed, expired


def _factions_compatible(fa: Faction, fb: Faction) -> bool:
    """Check if two factions have compatible dominant stats."""
    compatible_pairs = {
        frozenset({"resolve", "faith"}),
        frozenset({"empathy", "improvisation"}),
        frozenset({"resolve", "improvisation"}),
        frozenset({"empathy", "faith"}),
    }
    return frozenset({fa.dominant_stat, fb.dominant_stat}) in compatible_pairs


def compute_action_modifiers(
    state: DiplomacyState,
    colonists: list[Colonist],
    resources: Resources,
) -> dict[str, dict[str, float]]:
    """Compute per-colonist action weight multipliers from diplomacy state.

    Returns {colonist_id: {action: multiplier}} where multiplier > 1.0
    means boosted, < 1.0 means suppressed, 1.0 means neutral.
    """
    crisis_resources = check_crisis(resources)
    modifiers: dict[str, dict[str, float]] = {}

    for c in colonists:
        if not c.is_active():
            continue
        mods: dict[str, float] = {}
        faction = state.faction_of(c.id)

        # Crisis protocol: cooperation boosted, sabotage suppressed
        if crisis_resources:
            mods["cooperate"] = CRISIS_COOP_BOOST
            mods["sabotage"] = 0.2
            if "food" in crisis_resources:
                mods["farm"] = 1.8
            if "power" in crisis_resources:
                mods["code"] = 1.6
            if "water" in crisis_resources:
                mods["terraform"] = 1.6
            if "air" in crisis_resources:
                mods["terraform"] = mods.get("terraform", 1.0) + 0.5
        elif faction:
            for pact in state.pacts:
                other_name = None
                if pact.faction_a == faction.name:
                    other_name = pact.faction_b
                elif pact.faction_b == faction.name:
                    other_name = pact.faction_a
                if other_name is None:
                    continue

                if pact.kind == "alliance":
                    mods["cooperate"] = mods.get("cooperate", 1.0) * ALLIANCE_COOP_BOOST
                    mods["sabotage"] = mods.get("sabotage", 1.0) * 0.5
                elif pact.kind == "embargo":
                    mods["sabotage"] = mods.get("sabotage", 1.0) * RIVALRY_SABOTAGE_BOOST
                    mods["hoard"] = mods.get("hoard", 1.0) * 1.3

            # Faction identity bonus
            stat_action = {
                "resolve": "terraform",
                "improvisation": "explore",
                "empathy": "mediate",
                "hoarding": "hoard",
                "faith": "pray",
                "paranoia": "sabotage",
            }
            identity_action = stat_action.get(faction.dominant_stat)
            if identity_action:
                mods[identity_action] = mods.get(identity_action, 1.0) * 1.2

        modifiers[c.id] = mods

    return modifiers


def tick_diplomacy(
    state: DiplomacyState,
    colonists: list[Colonist],
    resources: Resources,
    year: int,
    rng: random.Random,
) -> DiplomacyYearResult:
    """Run one year of diplomacy: detect factions, manage pacts, compute modifiers.

    This is the single entry point called by the engine each tick.
    """
    faction_events = detect_factions(colonists, state, year)
    pacts_formed, pacts_expired = manage_pacts(state, year, rng)

    crisis_resources = check_crisis(resources)
    if crisis_resources:
        if not state.crisis_active:
            state.history.append({
                "type": "crisis_started", "year": year,
                "resources": crisis_resources,
            })
        state.crisis_active = True
        state.crisis_years += 1
    else:
        if state.crisis_active:
            state.history.append({
                "type": "crisis_ended", "year": year,
                "duration": state.crisis_years,
            })
        state.crisis_active = False
        state.crisis_years = 0

    modifiers = compute_action_modifiers(state, colonists, resources)

    return DiplomacyYearResult(
        factions_detected=faction_events,
        pacts_formed=pacts_formed,
        pacts_expired=pacts_expired,
        crisis_active=state.crisis_active,
        action_modifiers=modifiers,
    )
