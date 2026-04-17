"""Diplomacy engine -- faction-based politics for Mars-100.

Formalises emergent factions into persistent political entities with
treaties, schisms, and faction-weighted governance voting.

Constitutional basis:
  - Amendment XIII  (Turtles All the Way Down): sub-sims may model diplomacy
  - Amendment XVI   (Dream Catcher):  deltas keyed by (frame, utc)
  - Amendment X     (Legacy Not Delete): dead factions archived, never removed
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

MIN_FACTION_SIZE = 3
MAX_FACTIONS = 5
HYSTERESIS_YEARS = 2
SCHISM_THRESHOLD = 0.25
TREATY_DURATION_YEARS = 10
VOTE_MODIFIER_CAP = 0.25
EMERGENCY_LABOUR_BONUS = 0.04

GOV_FACTION_BIAS: dict[str, str] = {
    "council": "empathy", "dictator": "resolve", "lottery": "faith",
    "consensus": "empathy", "ai_governor": "improvisation", "anarchy": "paranoia",
}
STAT_FACTION_MAP = {
    "resolve": "The Resolute", "faith": "The Faithful", "empathy": "The Empaths",
    "improvisation": "The Innovators", "paranoia": "The Paranoids",
    "hoarding": "The Pragmatists",
}
TREATY_TYPES = ["research_pact", "air_mutual_aid", "labour_share"]


@dataclass
class Faction:
    """A political grouping of colonists."""
    id: str
    name: str
    formed_year: int
    member_ids: list[str] = field(default_factory=list)
    dominant_stat: str = "resolve"
    cohesion: float = 0.5
    archived: bool = False
    archived_year: int | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id, "name": self.name, "formed_year": self.formed_year,
            "member_ids": list(self.member_ids), "dominant_stat": self.dominant_stat,
            "cohesion": round(self.cohesion, 4), "archived": self.archived,
        }
        if self.archived_year is not None:
            d["archived_year"] = self.archived_year
        return d


@dataclass
class Treaty:
    """A bilateral agreement between two factions."""
    id: str
    faction_a: str
    faction_b: str
    treaty_type: str
    signed_year: int
    expires_year: int
    active: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id, "faction_a": self.faction_a,
            "faction_b": self.faction_b, "treaty_type": self.treaty_type,
            "signed_year": self.signed_year, "expires_year": self.expires_year,
            "active": self.active,
        }


@dataclass
class DiplomacyState:
    """Full diplomatic state of the colony."""
    factions: dict[str, Faction] = field(default_factory=dict)
    treaties: list[Treaty] = field(default_factory=list)
    archived_factions: list[dict] = field(default_factory=list)
    next_faction_id: int = 0
    next_treaty_id: int = 0

    def to_dict(self) -> dict:
        return {
            "factions": {k: v.to_dict() for k, v in self.factions.items()},
            "treaties": [t.to_dict() for t in self.treaties],
            "archived_factions": list(self.archived_factions),
            "active_faction_count": len([f for f in self.factions.values() if not f.archived]),
            "active_treaty_count": len([t for t in self.treaties if t.active]),
        }


@dataclass
class DiplomacyTickResult:
    """What happened in one year of diplomacy."""
    factions_formed: list[str] = field(default_factory=list)
    factions_dissolved: list[str] = field(default_factory=list)
    schisms: list[dict] = field(default_factory=list)
    treaties_proposed: list[str] = field(default_factory=list)
    treaties_expired: list[str] = field(default_factory=list)
    treaty_effects: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "factions_formed": self.factions_formed,
            "factions_dissolved": self.factions_dissolved,
            "schisms": self.schisms,
            "treaties_proposed": self.treaties_proposed,
            "treaties_expired": self.treaties_expired,
            "treaty_effects": {k: round(v, 4) for k, v in self.treaty_effects.items()},
        }


def _compute_cohesion(member_ids: list[str], social_graph: Any,
                      rng: random.Random) -> float:
    """Mean pairwise trust among faction members."""
    if len(member_ids) < 2:
        return 0.5
    total = count = 0.0
    for i, a in enumerate(member_ids):
        for b in member_ids[i + 1:]:
            rel = social_graph.get(a, b)
            total += rel.trust
            count += 1
    return total / count if count else 0.5


def detect_factions(colonists: list[Any], social_graph: Any,
                    existing: dict[str, Faction], year: int,
                    rng: random.Random) -> dict[str, Faction]:
    """Detect factions from colonist stats with hysteresis."""
    from src.mars100.colonist import STAT_NAMES
    active = [c for c in colonists if c.is_active()]
    if len(active) < MIN_FACTION_SIZE:
        return existing

    clusters: dict[str, list[str]] = {}
    for c in active:
        stats = {s: getattr(c.stats, s) for s in STAT_NAMES}
        dominant = max(stats, key=stats.get)
        clusters.setdefault(dominant, []).append(c.id)

    large = {s: ids for s, ids in clusters.items() if len(ids) >= MIN_FACTION_SIZE}
    for stat, ids in clusters.items():
        if stat not in large and large:
            biggest = max(large, key=lambda s: len(large[s]))
            large[biggest].extend(ids)

    new_factions: dict[str, Faction] = {}
    used_names: set[str] = set()
    for stat, member_ids in large.items():
        if len(member_ids) < MIN_FACTION_SIZE:
            continue
        fname = STAT_FACTION_MAP.get(stat, f"The {stat.title()}s")
        matched = None
        for fid, f in existing.items():
            if f.dominant_stat == stat and not f.archived:
                matched = f
                break
        if matched:
            matched.member_ids = member_ids
            matched.cohesion = _compute_cohesion(member_ids, social_graph, rng)
            new_factions[matched.id] = matched
            used_names.add(matched.name)
        elif year >= HYSTERESIS_YEARS:
            if len(new_factions) < MAX_FACTIONS and fname not in used_names:
                fid = f"faction-{stat}"
                new_factions[fid] = Faction(
                    id=fid, name=fname, formed_year=year,
                    member_ids=member_ids, dominant_stat=stat,
                    cohesion=_compute_cohesion(member_ids, social_graph, rng),
                )
                used_names.add(fname)
    return new_factions


def check_schisms(factions: dict[str, Faction],
                  social_graph: Any, year: int) -> list[dict]:
    """Check for faction splits due to low cohesion."""
    schisms: list[dict] = []
    to_archive: list[str] = []
    for fid, faction in factions.items():
        if faction.archived or len(faction.member_ids) < MIN_FACTION_SIZE:
            continue
        if faction.cohesion < SCHISM_THRESHOLD:
            schisms.append({
                "faction_id": fid, "faction_name": faction.name,
                "year": year, "cohesion": round(faction.cohesion, 4),
                "reason": "low_cohesion",
            })
            to_archive.append(fid)
    for fid in to_archive:
        factions[fid].archived = True
        factions[fid].archived_year = year
    return schisms


def propose_treaty(state: DiplomacyState, faction_a: str,
                   faction_b: str, year: int,
                   rng: random.Random) -> Treaty | None:
    """Propose a treaty between two factions if conditions are met."""
    fa = state.factions.get(faction_a)
    fb = state.factions.get(faction_b)
    if not fa or not fb or fa.archived or fb.archived:
        return None
    for t in state.treaties:
        if t.active and {t.faction_a, t.faction_b} == {faction_a, faction_b}:
            return None
    prob = (fa.cohesion + fb.cohesion) / 2.0 * 0.3
    if rng.random() > prob:
        return None
    treaty_type = rng.choice(TREATY_TYPES)
    tid = f"treaty-{state.next_treaty_id}"
    state.next_treaty_id += 1
    return Treaty(id=tid, faction_a=faction_a, faction_b=faction_b,
                  treaty_type=treaty_type, signed_year=year,
                  expires_year=year + TREATY_DURATION_YEARS)


def sign_treaty(state: DiplomacyState, treaty: Treaty) -> None:
    """Add a treaty to the state."""
    state.treaties.append(treaty)


def expire_treaties(state: DiplomacyState, year: int) -> list[str]:
    """Expire treaties past their duration."""
    expired: list[str] = []
    for t in state.treaties:
        if t.active and year >= t.expires_year:
            t.active = False
            expired.append(t.id)
    return expired


def compute_treaty_effects(state: DiplomacyState) -> dict[str, float]:
    """Compute colony-wide effects from all active treaties."""
    effects = {"research_bonus": 0.0, "air_crisis_bonus": 0.0, "build_speed_bonus": 0.0}
    for t in state.treaties:
        if not t.active:
            continue
        if t.treaty_type == "research_pact":
            effects["research_bonus"] += 0.02
        elif t.treaty_type == "air_mutual_aid":
            effects["air_crisis_bonus"] += EMERGENCY_LABOUR_BONUS
        elif t.treaty_type == "labour_share":
            effects["build_speed_bonus"] += 0.015
    return effects


def faction_vote_modifier(colonist_id: str, gov_type: str,
                          factions: dict[str, Faction]) -> float:
    """Compute vote bias from faction membership.

    Returns a value in [-VOTE_MODIFIER_CAP, +VOTE_MODIFIER_CAP].
    """
    for faction in factions.values():
        if faction.archived:
            continue
        if colonist_id in faction.member_ids:
            preferred_stat = GOV_FACTION_BIAS.get(gov_type, "")
            if faction.dominant_stat == preferred_stat:
                return min(VOTE_MODIFIER_CAP, faction.cohesion * 0.4)
            else:
                return max(-VOTE_MODIFIER_CAP, -0.1)
    return 0.0


def tick_diplomacy(state: DiplomacyState, colonists: list[Any],
                   social_graph: Any, year: int,
                   rng: random.Random) -> DiplomacyTickResult:
    """Run one year of diplomacy. Mutates state in place."""
    result = DiplomacyTickResult()
    old_fids = set(state.factions.keys())
    state.factions = detect_factions(colonists, social_graph, state.factions, year, rng)
    result.factions_formed = list(set(state.factions.keys()) - old_fids)

    schisms = check_schisms(state.factions, social_graph, year)
    result.schisms = schisms
    for s in schisms:
        result.factions_dissolved.append(s["faction_id"])
        state.archived_factions.append(s)

    result.treaties_expired = expire_treaties(state, year)

    active_fids = [fid for fid, f in state.factions.items() if not f.archived]
    for i, fa in enumerate(active_fids):
        for fb in active_fids[i + 1:]:
            treaty = propose_treaty(state, fa, fb, year, rng)
            if treaty:
                sign_treaty(state, treaty)
                result.treaties_proposed.append(treaty.id)

    result.treaty_effects = compute_treaty_effects(state)
    return result
