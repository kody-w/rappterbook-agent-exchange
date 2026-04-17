"""
Diplomacy organ for Mars-100 colony simulation (engine v9.0).

Tracks political factions that emerge organically as colony population grows.
Factions form around shared ideology (stat vectors), gated by social trust.
Members join, defect, elect leaders, and bias governance votes.

Phase 1 scope:
  - Faction formation via hybrid stat-similarity + trust clustering
  - Membership drift with hysteresis (min tenure, defect cooldown)
  - Leader election per faction per year
  - Governance voting bias from faction loyalty
  - Serialization round-trip
  - Defer: alliances, tensions, action-weight pressure (v10+)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

# -- constants ---------------------------------------------------------------

FORMATION_MIN_YEAR = 15
FORMATION_MIN_POP = 12
MAX_FACTIONS = 4
MIN_FACTION_SIZE = 3
COHESION_THRESHOLD = 0.55
TRUST_GATE = 0.25

JOIN_AFFINITY_THRESHOLD = 0.60
DEFECT_AFFINITY_THRESHOLD = 0.35
MIN_TENURE = 3

LEADER_RESOLVE_W = 0.40
LEADER_EMPATHY_W = 0.35
LEADER_FAITH_W = 0.25

VOTE_LOYALTY_BIAS = 0.25
PROPOSAL_PRIORITY_BONUS = 0.20

FACTION_NAMES = [
    "The Resolute",     "The Pioneers",     "The Harmonists",
    "The Stewards",     "The Faithful",     "The Vigilants",
    "The Builders",     "The Seekers",
]

IDEOLOGY_AXES = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")


# -- data structures ---------------------------------------------------------

@dataclass
class Faction:
    """A political faction within the colony."""
    id: str
    name: str
    ideology: dict[str, float]
    founder_id: str
    members: list[str]
    formed_year: int
    leader_id: str | None = None
    dissolved: bool = False
    dissolved_year: int | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "ideology": {k: round(v, 4) for k, v in self.ideology.items()},
            "founder_id": self.founder_id, "members": list(self.members),
            "formed_year": self.formed_year,
            "leader_id": self.leader_id,
            "dissolved": self.dissolved,
            "dissolved_year": self.dissolved_year,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Faction:
        return cls(
            id=d.get("id", "faction-0"),
            name=d.get("name", "Unknown"),
            ideology=d.get("ideology", {}),
            founder_id=d.get("founder_id", ""),
            members=list(d.get("members", [])),
            formed_year=d.get("formed_year", 0),
            leader_id=d.get("leader_id"),
            dissolved=d.get("dissolved", False),
            dissolved_year=d.get("dissolved_year"),
        )


@dataclass
class DiplomacyState:
    """Colony-wide faction tracking."""
    factions: list[Faction] = field(default_factory=list)
    next_faction_id: int = 0
    join_year: dict[str, int] = field(default_factory=dict)

    def active_factions(self) -> list[Faction]:
        return [f for f in self.factions if not f.dissolved]

    def faction_of(self, colonist_id: str) -> Faction | None:
        for f in self.active_factions():
            if colonist_id in f.members:
                return f
        return None

    def to_dict(self) -> dict:
        return {
            "factions": [f.to_dict() for f in self.factions],
            "next_faction_id": self.next_faction_id,
            "join_year": dict(self.join_year),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiplomacyState:
        return cls(
            factions=[Faction.from_dict(f) for f in d.get("factions", [])],
            next_faction_id=d.get("next_faction_id", 0),
            join_year=dict(d.get("join_year", {})),
        )


@dataclass
class FactionEvent:
    """A single faction event during a year tick."""
    kind: str       # "formed", "joined", "defected", "leader_elected", "dissolved"
    faction_id: str
    colonist_id: str | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"kind": self.kind, "faction_id": self.faction_id}
        if self.colonist_id:
            d["colonist_id"] = self.colonist_id
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass
class DiplomacyTickResult:
    """Result of one year of faction dynamics."""
    events: list[FactionEvent] = field(default_factory=list)
    factions_formed: int = 0
    joins: int = 0
    defections: int = 0
    dissolutions: int = 0

    def to_dict(self) -> dict:
        return {
            "events": [e.to_dict() for e in self.events],
            "factions_formed": self.factions_formed,
            "joins": self.joins,
            "defections": self.defections,
            "dissolutions": self.dissolutions,
        }


# -- helper functions --------------------------------------------------------

def _stat_vector(colonist_stats: dict[str, float]) -> list[float]:
    """Extract ordered stat vector from a colonist stats dict."""
    return [colonist_stats.get(ax, 0.5) for ax in IDEOLOGY_AXES]


def _ideology_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Euclidean distance between two ideology vectors."""
    total = 0.0
    for ax in IDEOLOGY_AXES:
        diff = a.get(ax, 0.5) - b.get(ax, 0.5)
        total += diff * diff
    return math.sqrt(total)


def _ideology_affinity(colonist_stats: dict[str, float],
                       ideology: dict[str, float]) -> float:
    """Affinity score (0-1) between a colonist and a faction ideology.

    1.0 = perfect match, 0.0 = maximally different.
    Max possible distance across 6 axes each [0,1] = sqrt(6) ≈ 2.449.
    """
    dist = _ideology_distance(colonist_stats, ideology)
    max_dist = math.sqrt(len(IDEOLOGY_AXES))
    return max(0.0, 1.0 - dist / max_dist)


def _average_ideology(stats_list: list[dict[str, float]]) -> dict[str, float]:
    """Compute average ideology from a list of stat dicts."""
    if not stats_list:
        return {ax: 0.5 for ax in IDEOLOGY_AXES}
    result: dict[str, float] = {}
    for ax in IDEOLOGY_AXES:
        result[ax] = sum(s.get(ax, 0.5) for s in stats_list) / len(stats_list)
    return result


def _cluster_cohesion(stats_list: list[dict[str, float]]) -> float:
    """Compute cohesion of a group (1.0 = identical, 0.0 = maximally spread)."""
    if len(stats_list) < 2:
        return 1.0
    centroid = _average_ideology(stats_list)
    distances = [_ideology_distance(s, centroid) for s in stats_list]
    avg_dist = sum(distances) / len(distances)
    max_dist = math.sqrt(len(IDEOLOGY_AXES))
    return max(0.0, 1.0 - avg_dist / max_dist)


def _avg_trust_to_group(colonist_id: str, group_ids: list[str],
                        social_get: Any) -> float:
    """Average trust from colonist to a group via social graph getter."""
    if not group_ids:
        return 0.0
    trusts = [social_get(colonist_id, oid).trust
              for oid in group_ids if oid != colonist_id]
    return sum(trusts) / max(1, len(trusts))


# -- core functions ----------------------------------------------------------

def try_form_factions(
    state: DiplomacyState,
    year: int,
    colonist_data: list[dict],
    social_get: Any,
    rng: random.Random,
) -> list[FactionEvent]:
    """Attempt to form new factions via stat-similarity + trust gating.

    Uses 2-means clustering on stat vectors. Only creates a faction if
    the cluster has cohesion > threshold and average internal trust > gate.
    """
    events: list[FactionEvent] = []
    active = [c for c in colonist_data if c.get("alive") and not c.get("exiled")]

    if year < FORMATION_MIN_YEAR or len(active) < FORMATION_MIN_POP:
        return events

    if len(state.active_factions()) >= MAX_FACTIONS:
        return events

    # Find unaffiliated colonists
    affiliated = set()
    for f in state.active_factions():
        affiliated.update(f.members)
    unaffiliated = [c for c in active if c["id"] not in affiliated]

    if len(unaffiliated) < MIN_FACTION_SIZE:
        return events

    # 2-means clustering on stat vectors
    stats_map = {c["id"]: c.get("stats", {}) for c in unaffiliated}
    ids = list(stats_map.keys())
    rng.shuffle(ids)

    # Pick two most dissimilar colonists as seeds
    best_dist = -1.0
    seed_a, seed_b = ids[0], ids[1] if len(ids) > 1 else ids[0]
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            d = _ideology_distance(stats_map[a_id], stats_map[b_id])
            if d > best_dist:
                best_dist = d
                seed_a, seed_b = a_id, b_id

    cluster_a: list[str] = []
    cluster_b: list[str] = []
    for cid in ids:
        dist_a = _ideology_distance(stats_map[cid], stats_map[seed_a])
        dist_b = _ideology_distance(stats_map[cid], stats_map[seed_b])
        if dist_a <= dist_b:
            cluster_a.append(cid)
        else:
            cluster_b.append(cid)

    # Try to form a faction from each cluster
    for cluster in [cluster_a, cluster_b]:
        if len(cluster) < MIN_FACTION_SIZE:
            continue
        if len(state.active_factions()) >= MAX_FACTIONS:
            break

        cluster_stats = [stats_map[cid] for cid in cluster]
        cohesion = _cluster_cohesion(cluster_stats)
        if cohesion < COHESION_THRESHOLD:
            continue

        # Trust gate: average pairwise trust within cluster must exceed threshold
        trust_sum = 0.0
        trust_count = 0
        for i, a_id in enumerate(cluster):
            for b_id in cluster[i + 1:]:
                trust_sum += social_get(a_id, b_id).trust
                trust_count += 1
        avg_trust = trust_sum / max(1, trust_count)
        if avg_trust < TRUST_GATE:
            continue

        ideology = _average_ideology(cluster_stats)
        fid = f"faction-{state.next_faction_id}"
        state.next_faction_id += 1

        name = _pick_faction_name(ideology, state, rng)
        founder = cluster[0]
        faction = Faction(
            id=fid, name=name, ideology=ideology,
            founder_id=founder, members=list(cluster),
            formed_year=year,
        )
        state.factions.append(faction)
        for cid in cluster:
            state.join_year[cid] = year

        events.append(FactionEvent(
            kind="formed", faction_id=fid,
            colonist_id=founder,
            detail=f"{name} formed with {len(cluster)} members",
        ))

    return events


def _pick_faction_name(ideology: dict[str, float],
                       state: DiplomacyState,
                       rng: random.Random) -> str:
    """Pick a faction name not already in use."""
    used = {f.name for f in state.factions}
    # Prefer name matching dominant ideology axis
    dominant = max(IDEOLOGY_AXES, key=lambda ax: ideology.get(ax, 0.0))
    axis_names = {
        "resolve": "The Resolute", "improvisation": "The Pioneers",
        "empathy": "The Harmonists", "hoarding": "The Stewards",
        "faith": "The Faithful", "paranoia": "The Vigilants",
    }
    preferred = axis_names.get(dominant, "The Seekers")
    if preferred not in used:
        return preferred
    available = [n for n in FACTION_NAMES if n not in used]
    if available:
        return rng.choice(available)
    return f"Faction {state.next_faction_id}"


def tick_membership(
    state: DiplomacyState,
    year: int,
    colonist_data: list[dict],
    social_get: Any,
    rng: random.Random,
) -> list[FactionEvent]:
    """Update faction membership: joins, defections, cleanup of dead members."""
    events: list[FactionEvent] = []
    active_ids = {c["id"] for c in colonist_data
                  if c.get("alive") and not c.get("exiled")}
    stats_map = {c["id"]: c.get("stats", {}) for c in colonist_data
                 if c["id"] in active_ids}

    # Remove dead/exiled members
    for faction in state.active_factions():
        removed = [m for m in faction.members if m not in active_ids]
        for cid in removed:
            faction.members.remove(cid)
            state.join_year.pop(cid, None)

    # Unaffiliated colonists may join
    affiliated = set()
    for f in state.active_factions():
        affiliated.update(f.members)

    for cid in active_ids - affiliated:
        stats = stats_map.get(cid)
        if not stats:
            continue
        best_faction = None
        best_affinity = JOIN_AFFINITY_THRESHOLD
        for faction in state.active_factions():
            aff = _ideology_affinity(stats, faction.ideology)
            trust = _avg_trust_to_group(cid, faction.members, social_get)
            combined = aff * 0.7 + trust * 0.3
            if combined > best_affinity:
                best_affinity = combined
                best_faction = faction
        if best_faction and rng.random() < 0.4:
            best_faction.members.append(cid)
            state.join_year[cid] = year
            events.append(FactionEvent(
                kind="joined", faction_id=best_faction.id,
                colonist_id=cid,
            ))

    # Existing members may defect (with hysteresis)
    for faction in state.active_factions():
        for cid in list(faction.members):
            join_yr = state.join_year.get(cid, 0)
            if year - join_yr < MIN_TENURE:
                continue
            stats = stats_map.get(cid)
            if not stats:
                continue
            aff = _ideology_affinity(stats, faction.ideology)
            if aff < DEFECT_AFFINITY_THRESHOLD and rng.random() < 0.3:
                faction.members.remove(cid)
                state.join_year.pop(cid, None)
                events.append(FactionEvent(
                    kind="defected", faction_id=faction.id,
                    colonist_id=cid,
                ))

    # Dissolve factions below minimum size
    for faction in state.active_factions():
        if len(faction.members) < MIN_FACTION_SIZE:
            faction.dissolved = True
            faction.dissolved_year = year
            for cid in faction.members:
                state.join_year.pop(cid, None)
            faction.members.clear()
            events.append(FactionEvent(
                kind="dissolved", faction_id=faction.id,
                detail=f"{faction.name} dissolved — too few members",
            ))

    return events


def elect_leaders(
    state: DiplomacyState,
    colonist_data: list[dict],
) -> list[FactionEvent]:
    """Elect a leader for each active faction based on stats."""
    events: list[FactionEvent] = []
    stats_map = {c["id"]: c.get("stats", {}) for c in colonist_data}

    for faction in state.active_factions():
        if not faction.members:
            continue
        best_id = None
        best_score = -1.0
        for cid in faction.members:
            s = stats_map.get(cid, {})
            score = (s.get("resolve", 0.5) * LEADER_RESOLVE_W
                     + s.get("empathy", 0.5) * LEADER_EMPATHY_W
                     + s.get("faith", 0.5) * LEADER_FAITH_W)
            if score > best_score:
                best_score = score
                best_id = cid
        old_leader = faction.leader_id
        faction.leader_id = best_id
        if best_id and best_id != old_leader:
            events.append(FactionEvent(
                kind="leader_elected", faction_id=faction.id,
                colonist_id=best_id,
            ))

    return events


def compute_vote_bias(
    colonist_id: str,
    proposer_id: str,
    state: DiplomacyState,
) -> float:
    """Compute faction-based voting bias.

    Returns a value in [-VOTE_LOYALTY_BIAS, +VOTE_LOYALTY_BIAS]:
      - Positive if voter and proposer share a faction
      - Negative if they're in rival factions
      - Zero if either is unaffiliated
    """
    voter_faction = state.faction_of(colonist_id)
    proposer_faction = state.faction_of(proposer_id)

    if voter_faction is None or proposer_faction is None:
        return 0.0
    if voter_faction.id == proposer_faction.id:
        return VOTE_LOYALTY_BIAS
    return -VOTE_LOYALTY_BIAS * 0.5


def is_faction_leader(colonist_id: str, state: DiplomacyState) -> bool:
    """Check if a colonist is a faction leader (gets proposal priority)."""
    for f in state.active_factions():
        if f.leader_id == colonist_id:
            return True
    return False


@dataclass
class FactionContext:
    """Context needed to run the diplomacy tick."""
    year: int
    colonist_data: list[dict]
    social_get: Any
    rng: random.Random


def tick_diplomacy(
    state: DiplomacyState,
    ctx: FactionContext,
) -> DiplomacyTickResult:
    """Run one year of faction dynamics.

    Order: formation → membership → leaders.
    """
    result = DiplomacyTickResult()

    # Formation
    formed = try_form_factions(
        state, ctx.year, ctx.colonist_data, ctx.social_get, ctx.rng)
    result.events.extend(formed)
    result.factions_formed = sum(1 for e in formed if e.kind == "formed")

    # Membership drift
    membership = tick_membership(
        state, ctx.year, ctx.colonist_data, ctx.social_get, ctx.rng)
    result.events.extend(membership)
    result.joins = sum(1 for e in membership if e.kind == "joined")
    result.defections = sum(1 for e in membership if e.kind == "defected")
    result.dissolutions = sum(1 for e in membership if e.kind == "dissolved")

    # Leader election
    leaders = elect_leaders(state, ctx.colonist_data)
    result.events.extend(leaders)

    return result
