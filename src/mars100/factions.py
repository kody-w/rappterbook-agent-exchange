"""
Emergent political factions for Mars-100.

Factions form organically when colonists with similar dominant stats and
high mutual trust cluster together.  They influence governance votes
(bloc voting) and action weights (shared priorities).

Canonical membership lives on each Colonist (faction_id field).
FactionState is a derived index rebuilt each tick from colonist data.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

MIN_FACTION_SIZE = 2
MAX_FACTIONS = 5
FORMATION_TRUST_THRESHOLD = 0.6
STAT_SIMILARITY_THRESHOLD = 0.25
SPLIT_THRESHOLD = 0.35
DISSOLUTION_SIZE = 1

FACTION_NAMES = [
    "Iron Covenant", "Dust Collective", "Ember Circle",
    "Tide Assembly", "Stone Accord", "Wind Pact",
    "Flame Senate", "Root Council", "Rift Union",
    "Sky Commune",
]


@dataclass
class Faction:
    """A political faction within the colony."""
    id: str
    name: str
    founded_year: int
    ideology: str
    member_ids: list[str] = field(default_factory=list)
    influence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name,
            "founded_year": self.founded_year,
            "ideology": self.ideology,
            "member_ids": list(self.member_ids),
            "influence": round(self.influence, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Faction:
        return cls(
            id=d["id"], name=d["name"],
            founded_year=d.get("founded_year", 0),
            ideology=d.get("ideology", "pragmatist"),
            member_ids=list(d.get("member_ids", [])),
            influence=d.get("influence", 0.0),
        )


@dataclass
class FactionState:
    """Derived faction index — rebuilt each tick from colonist faction_ids."""
    factions: dict[str, Faction] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    next_faction_num: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "factions": {fid: f.to_dict() for fid, f in self.factions.items()},
            "history": self.history,
            "next_faction_num": self.next_faction_num,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FactionState:
        factions = {fid: Faction.from_dict(fd)
                    for fid, fd in d.get("factions", {}).items()}
        return cls(
            factions=factions,
            history=d.get("history", []),
            next_faction_num=d.get("next_faction_num", 0),
        )

    def faction_for(self, colonist_id: str) -> Faction | None:
        """Look up which faction a colonist belongs to."""
        for f in self.factions.values():
            if colonist_id in f.member_ids:
                return f
        return None

    def active_factions(self) -> list[Faction]:
        """Return factions with at least MIN_FACTION_SIZE members."""
        return [f for f in self.factions.values()
                if len(f.member_ids) >= MIN_FACTION_SIZE]


def _stat_distance(stats_a: dict[str, float], stats_b: dict[str, float]) -> float:
    """Euclidean distance between two stat dictionaries (normalized 0-1 per axis)."""
    keys = set(stats_a) & set(stats_b)
    if not keys:
        return 1.0
    return (sum((stats_a[k] - stats_b[k]) ** 2 for k in keys) / len(keys)) ** 0.5


def _ideology_from_dominant(dominant: str) -> str:
    """Map a dominant stat to a faction ideology label."""
    mapping = {
        "resolve": "militarist", "improvisation": "innovator",
        "empathy": "communalist", "hoarding": "survivalist",
        "faith": "spiritualist", "paranoia": "isolationist",
    }
    return mapping.get(dominant, "pragmatist")


def attempt_formation(
    colonists: list, social: Any, faction_state: FactionState,
    year: int, rng: random.Random,
) -> list[dict]:
    """Try to form new factions from unaffiliated colonists.

    Returns a list of event dicts describing any formations.
    """
    if len(faction_state.factions) >= MAX_FACTIONS:
        return []

    unaffiliated = [c for c in colonists
                    if c.is_active() and c.faction_id is None]
    if len(unaffiliated) < MIN_FACTION_SIZE:
        return []

    events: list[dict] = []
    used: set[str] = set()

    for anchor in unaffiliated:
        if anchor.id in used:
            continue
        candidates = []
        anchor_stats = anchor.stats.to_dict()
        for other in unaffiliated:
            if other.id == anchor.id or other.id in used:
                continue
            dist = _stat_distance(anchor_stats, other.stats.to_dict())
            trust = social.get(anchor.id, other.id).trust
            if dist < STAT_SIMILARITY_THRESHOLD and trust > FORMATION_TRUST_THRESHOLD:
                candidates.append(other)
        if len(candidates) >= MIN_FACTION_SIZE - 1:
            members = [anchor] + candidates[:4]
            fid = f"faction-{faction_state.next_faction_num}"
            faction_state.next_faction_num += 1
            name = FACTION_NAMES[faction_state.next_faction_num - 1 % len(FACTION_NAMES)] \
                if faction_state.next_faction_num <= len(FACTION_NAMES) \
                else f"Coalition {faction_state.next_faction_num}"
            ideology = _ideology_from_dominant(anchor.stats.dominant())
            faction = Faction(
                id=fid, name=name, founded_year=year,
                ideology=ideology,
                member_ids=[m.id for m in members],
                influence=len(members) / max(1, len(colonists)),
            )
            faction_state.factions[fid] = faction
            for m in members:
                m.faction_id = fid
                used.add(m.id)
            faction_state.history.append({
                "year": year, "event": "formation",
                "faction_id": fid, "name": name,
                "members": [m.id for m in members],
            })
            events.append({
                "type": "faction_formed", "faction_id": fid,
                "name": name, "ideology": ideology,
                "members": [m.id for m in members], "year": year,
            })
            if len(faction_state.factions) >= MAX_FACTIONS:
                break
    return events


def attempt_recruitment(
    colonists: list, social: Any, faction_state: FactionState,
    year: int, rng: random.Random,
) -> list[dict]:
    """Existing factions try to recruit unaffiliated colonists."""
    events: list[dict] = []
    unaffiliated = [c for c in colonists
                    if c.is_active() and c.faction_id is None]
    for newcomer in unaffiliated:
        best_faction: Faction | None = None
        best_trust = 0.0
        for faction in faction_state.active_factions():
            avg_trust = sum(
                social.get(newcomer.id, mid).trust
                for mid in faction.member_ids
                if mid != newcomer.id
            ) / max(1, len(faction.member_ids))
            if avg_trust > FORMATION_TRUST_THRESHOLD and avg_trust > best_trust:
                best_faction = faction
                best_trust = avg_trust
        if best_faction and rng.random() < 0.4:
            newcomer.faction_id = best_faction.id
            best_faction.member_ids.append(newcomer.id)
            events.append({
                "type": "faction_recruit", "faction_id": best_faction.id,
                "colonist_id": newcomer.id, "year": year,
            })
    return events


def check_splits(
    colonists: list, social: Any, faction_state: FactionState,
    year: int, rng: random.Random,
) -> list[dict]:
    """Check if internal distrust causes a faction to split."""
    events: list[dict] = []
    for fid in list(faction_state.factions):
        faction = faction_state.factions[fid]
        if len(faction.member_ids) < 4:
            continue
        members = [c for c in colonists
                   if c.id in faction.member_ids and c.is_active()]
        if len(members) < 4:
            continue
        avg_internal_trust = 0.0
        pairs = 0
        for a in members:
            for b in members:
                if a.id != b.id:
                    avg_internal_trust += social.get(a.id, b.id).trust
                    pairs += 1
        avg_internal_trust /= max(1, pairs)
        if avg_internal_trust < SPLIT_THRESHOLD and rng.random() < 0.3:
            rng.shuffle(members)
            half = len(members) // 2
            splitters = members[half:]
            if len(splitters) >= MIN_FACTION_SIZE:
                new_fid = f"faction-{faction_state.next_faction_num}"
                faction_state.next_faction_num += 1
                name_idx = (faction_state.next_faction_num - 1) % len(FACTION_NAMES)
                new_name = FACTION_NAMES[name_idx]
                leader = splitters[0]
                new_ideology = _ideology_from_dominant(leader.stats.dominant())
                new_faction = Faction(
                    id=new_fid, name=new_name, founded_year=year,
                    ideology=new_ideology,
                    member_ids=[s.id for s in splitters],
                    influence=len(splitters) / max(1, len(colonists)),
                )
                faction_state.factions[new_fid] = new_faction
                for s in splitters:
                    s.faction_id = new_fid
                    faction.member_ids.remove(s.id)
                faction_state.history.append({
                    "year": year, "event": "split",
                    "parent_faction": fid, "new_faction": new_fid,
                    "splitters": [s.id for s in splitters],
                })
                events.append({
                    "type": "faction_split", "parent_id": fid,
                    "new_id": new_fid, "new_name": new_name,
                    "year": year,
                })
    return events


def cleanup_dead(
    colonists: list, faction_state: FactionState, year: int,
) -> list[dict]:
    """Remove dead/exiled colonists from factions.  Dissolve empty factions."""
    events: list[dict] = []
    active_ids = {c.id for c in colonists if c.is_active()}
    for fid in list(faction_state.factions):
        faction = faction_state.factions[fid]
        before = len(faction.member_ids)
        faction.member_ids = [mid for mid in faction.member_ids
                              if mid in active_ids]
        if len(faction.member_ids) <= DISSOLUTION_SIZE:
            for c in colonists:
                if c.faction_id == fid:
                    c.faction_id = None
            faction_state.history.append({
                "year": year, "event": "dissolution",
                "faction_id": fid,
            })
            events.append({
                "type": "faction_dissolved", "faction_id": fid,
                "name": faction.name, "year": year,
            })
            del faction_state.factions[fid]
    return events


def update_influence(
    colonists: list, faction_state: FactionState,
) -> None:
    """Recompute faction influence scores."""
    active_count = sum(1 for c in colonists if c.is_active())
    for faction in faction_state.factions.values():
        faction.influence = len(faction.member_ids) / max(1, active_count)


def faction_tick(
    colonists: list, social: Any, faction_state: FactionState,
    year: int, rng: random.Random,
) -> list[dict]:
    """Run one year of faction dynamics.  Returns events."""
    events: list[dict] = []
    events.extend(cleanup_dead(colonists, faction_state, year))
    events.extend(attempt_formation(colonists, social, faction_state, year, rng))
    events.extend(attempt_recruitment(colonists, social, faction_state, year, rng))
    events.extend(check_splits(colonists, social, faction_state, year, rng))
    update_influence(colonists, faction_state)
    return events


def faction_bloc_vote(
    voter_id: str, proposal_gov_type: str,
    faction_state: FactionState, rng: random.Random,
) -> bool | None:
    """If voter is in a faction, return faction's bloc vote preference.

    Returns True (for), False (against), or None (no faction opinion).
    """
    faction = faction_state.faction_for(voter_id)
    if faction is None:
        return None
    affinity = {
        "militarist": {"dictator": 0.7, "council": 0.3, "anarchy": 0.1},
        "communalist": {"consensus": 0.8, "council": 0.5, "anarchy": 0.3},
        "innovator": {"ai_governor": 0.7, "lottery": 0.5, "anarchy": 0.4},
        "spiritualist": {"consensus": 0.6, "council": 0.5, "dictator": 0.3},
        "survivalist": {"dictator": 0.6, "council": 0.4, "anarchy": 0.2},
        "isolationist": {"anarchy": 0.7, "lottery": 0.4, "dictator": 0.2},
        "pragmatist": {"council": 0.5, "consensus": 0.4, "lottery": 0.3},
    }
    prefs = affinity.get(faction.ideology, {})
    score = prefs.get(proposal_gov_type, 0.3)
    return rng.random() < score


def action_weight_modifier(
    colonist_id: str, action: str, faction_state: FactionState,
) -> float:
    """Return an additive weight modifier based on faction ideology."""
    faction = faction_state.faction_for(colonist_id)
    if faction is None:
        return 0.0
    boosts = {
        "militarist": {"terraform": 0.3, "sabotage": 0.2},
        "communalist": {"cooperate": 0.5, "mediate": 0.3, "farm": 0.2},
        "innovator": {"code": 0.4, "explore": 0.3},
        "spiritualist": {"pray": 0.4, "mediate": 0.2},
        "survivalist": {"hoard": 0.4, "farm": 0.3},
        "isolationist": {"hoard": 0.3, "rest": 0.2},
        "pragmatist": {"cooperate": 0.2, "farm": 0.2},
    }
    return boosts.get(faction.ideology, {}).get(action, 0.0)
