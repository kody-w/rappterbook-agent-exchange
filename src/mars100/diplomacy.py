"""Diplomacy organ — faction detection, alliances, schisms, and vote bias.

Factions emerge from the social graph via density-based clustering. Alliances
form between compatible factions. Schisms split oversized factions. Factions
influence governance votes through `faction_vote_bias()`.

Constitutional basis: Amendment XVI (Dream Catcher) — diplomacy events are
appended as deltas, never overwriting colonist state.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist
from src.mars100.colony import SocialGraph


# ── Configuration ──────────────────────────────────────────────────────────

MIN_FACTION_SIZE = 2
TRUST_THRESHOLD = 0.55       # minimum trust to be "connected" for clustering
MAX_ALLIANCE_PER_FACTION = 2
SCHISM_SIZE_THRESHOLD = 5    # factions above this may split
VOTE_BIAS_CAP = 0.12         # maximum influence a faction can exert on a vote


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class Faction:
    """A detected faction within the colony."""
    id: str
    member_ids: list[str] = field(default_factory=list)
    formed_year: int = 0
    platform: str = ""  # governance preference
    cohesion: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "member_ids": list(self.member_ids),
            "formed_year": self.formed_year, "platform": self.platform,
            "cohesion": self.cohesion,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Faction:
        return cls(
            id=d["id"], member_ids=list(d.get("member_ids", [])),
            formed_year=d.get("formed_year", 0),
            platform=d.get("platform", ""),
            cohesion=d.get("cohesion", 0.0),
        )


@dataclass
class Alliance:
    """An agreement between two factions."""
    faction_a: str
    faction_b: str
    formed_year: int = 0
    strength: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "faction_a": self.faction_a, "faction_b": self.faction_b,
            "formed_year": self.formed_year, "strength": self.strength,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Alliance:
        return cls(
            faction_a=d["faction_a"], faction_b=d["faction_b"],
            formed_year=d.get("formed_year", 0), strength=d.get("strength", 0.5),
        )


@dataclass
class DiplomacyState:
    """Persistent diplomacy state across simulation years."""
    factions: list[Faction] = field(default_factory=list)
    alliances: list[Alliance] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def active_faction_count(self) -> int:
        return len(self.factions)

    def faction_of(self, colonist_id: str) -> Faction | None:
        """Return the faction a colonist belongs to, or None."""
        for f in self.factions:
            if colonist_id in f.member_ids:
                return f
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_faction_count": self.active_faction_count,
            "factions": [f.to_dict() for f in self.factions],
            "alliances": [a.to_dict() for a in self.alliances],
            "history": self.history[-50:],  # keep last 50 events
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiplomacyState:
        return cls(
            factions=[Faction.from_dict(f) for f in d.get("factions", [])],
            alliances=[Alliance.from_dict(a) for a in d.get("alliances", [])],
            history=list(d.get("history", [])),
        )


# ── Faction detection (density-based) ─────────────────────────────────────

def _build_adjacency(social: SocialGraph, active_ids: list[str]) -> dict[str, set[str]]:
    """Build trust-based adjacency from social graph."""
    adj: dict[str, set[str]] = {cid: set() for cid in active_ids}
    for i, a in enumerate(active_ids):
        for b in active_ids[i + 1:]:
            rel = social.get(a, b)
            if rel.trust >= TRUST_THRESHOLD:
                adj[a].add(b)
                adj[b].add(a)
    return adj


def _connected_components(adj: dict[str, set[str]]) -> list[set[str]]:
    """Find connected components via BFS."""
    visited: set[str] = set()
    components: list[set[str]] = []
    for node in adj:
        if node in visited:
            continue
        component: set[str] = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(component) >= MIN_FACTION_SIZE:
            components.append(component)
    return components


def detect_factions(social: SocialGraph, active_ids: list[str]) -> list[set[str]]:
    """Detect factions as connected components of the high-trust subgraph."""
    adj = _build_adjacency(social, active_ids)
    return _connected_components(adj)


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def reconcile_factions(
    old_factions: list[Faction],
    new_clusters: list[set[str]],
    year: int,
) -> list[Faction]:
    """Match new clusters to existing factions for stable IDs.

    Uses Jaccard similarity for greedy matching. Unmatched clusters get new IDs.
    """
    used_old: set[str] = set()
    result: list[Faction] = []
    next_id_num = max(
        (int(f.id.split("-")[1]) for f in old_factions if "-" in f.id),
        default=0,
    ) + 1

    for cluster in sorted(new_clusters, key=len, reverse=True):
        best_match: Faction | None = None
        best_score = 0.0
        for old_f in old_factions:
            if old_f.id in used_old:
                continue
            score = _jaccard(set(old_f.member_ids), cluster)
            if score > best_score:
                best_score = score
                best_match = old_f
        if best_match and best_score >= 0.3:
            used_old.add(best_match.id)
            result.append(Faction(
                id=best_match.id,
                member_ids=sorted(cluster),
                formed_year=best_match.formed_year,
                platform=best_match.platform,
            ))
        else:
            result.append(Faction(
                id=f"faction-{next_id_num}",
                member_ids=sorted(cluster),
                formed_year=year,
            ))
            next_id_num += 1
    return result


# ── Governance platform assignment ─────────────────────────────────────────

GOV_PLATFORMS = ["direct_democracy", "council", "technocracy", "meritocracy", "anarchy"]


def _avg_stat(colonists: list[Colonist], member_ids: list[str], stat: str) -> float:
    """Average a stat across faction members."""
    members = [c for c in colonists if c.id in member_ids and c.alive]
    if not members:
        return 0.5
    return sum(getattr(c.stats, stat, 0.5) for c in members) / len(members)


def assign_platform(faction: Faction, colonists: list[Colonist]) -> str:
    """Assign a governance platform based on faction member stats."""
    avg_resolve = _avg_stat(colonists, faction.member_ids, "resolve")
    avg_empathy = _avg_stat(colonists, faction.member_ids, "empathy")
    avg_improvisation = _avg_stat(colonists, faction.member_ids, "improvisation")

    if avg_resolve > 0.7:
        return "technocracy"
    if avg_empathy > 0.7:
        return "direct_democracy"
    if avg_improvisation > 0.7:
        return "anarchy"
    if avg_resolve > 0.5 and avg_empathy > 0.5:
        return "council"
    return "meritocracy"


# ── Faction cohesion ───────────────────────────────────────────────────────

def compute_faction_cohesion(
    faction: Faction,
    social: SocialGraph,
) -> float:
    """Mean intra-faction trust. Returns 0.0 if faction has <2 members."""
    members = faction.member_ids
    if len(members) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            total += social.get(a, b).trust
            count += 1
    return total / count if count > 0 else 0.0


# ── Alliances ──────────────────────────────────────────────────────────────

def try_form_alliance(
    state: DiplomacyState,
    social: SocialGraph,
    year: int,
    rng: random.Random,
) -> Alliance | None:
    """Attempt to form an alliance between two factions.

    Requirements: compatible platforms, some inter-faction trust, not already allied.
    """
    factions = state.factions
    if len(factions) < 2:
        return None

    existing_pairs = {
        (a.faction_a, a.faction_b) for a in state.alliances
    } | {
        (a.faction_b, a.faction_a) for a in state.alliances
    }

    candidates: list[tuple[Faction, Faction, float]] = []
    for i, fa in enumerate(factions):
        if sum(1 for a in state.alliances
               if a.faction_a == fa.id or a.faction_b == fa.id) >= MAX_ALLIANCE_PER_FACTION:
            continue
        for fb in factions[i + 1:]:
            if (fa.id, fb.id) in existing_pairs:
                continue
            if sum(1 for a in state.alliances
                   if a.faction_a == fb.id or a.faction_b == fb.id) >= MAX_ALLIANCE_PER_FACTION:
                continue
            # Inter-faction trust
            trust_sum = 0.0
            count = 0
            for mid_a in fa.member_ids:
                for mid_b in fb.member_ids:
                    trust_sum += social.get(mid_a, mid_b).trust
                    count += 1
            avg_trust = trust_sum / count if count > 0 else 0.0
            if avg_trust >= 0.4:
                candidates.append((fa, fb, avg_trust))

    if not candidates:
        return None

    # Probabilistic selection weighted by inter-faction trust
    fa, fb, trust = rng.choices(
        candidates,
        weights=[t for _, _, t in candidates],
        k=1,
    )[0]

    alliance = Alliance(
        faction_a=fa.id, faction_b=fb.id,
        formed_year=year, strength=trust,
    )
    return alliance


def check_alliance_breakups(
    state: DiplomacyState,
    social: SocialGraph,
) -> list[Alliance]:
    """Remove alliances where inter-faction trust has dropped below threshold."""
    broken: list[Alliance] = []
    faction_map = {f.id: f for f in state.factions}

    surviving: list[Alliance] = []
    for alliance in state.alliances:
        fa = faction_map.get(alliance.faction_a)
        fb = faction_map.get(alliance.faction_b)
        if not fa or not fb:
            broken.append(alliance)
            continue
        trust_sum = 0.0
        count = 0
        for mid_a in fa.member_ids:
            for mid_b in fb.member_ids:
                trust_sum += social.get(mid_a, mid_b).trust
                count += 1
        avg_trust = trust_sum / count if count > 0 else 0.0
        if avg_trust < 0.3:
            broken.append(alliance)
        else:
            alliance.strength = avg_trust
            surviving.append(alliance)

    state.alliances = surviving
    return broken


# ── Schisms ────────────────────────────────────────────────────────────────

def check_schism(
    faction: Faction,
    social: SocialGraph,
    year: int,
    rng: random.Random,
) -> list[Faction] | None:
    """Check if a faction should split. Returns two child factions or None."""
    if len(faction.member_ids) < 2 * MIN_FACTION_SIZE:
        return None

    # Find the lowest-trust pair to use as schism seed
    members = faction.member_ids
    min_trust = 1.0
    split_a, split_b = members[0], members[-1]
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            t = social.get(a, b).trust
            if t < min_trust:
                min_trust = t
                split_a, split_b = a, b

    if min_trust > 0.35:  # not enough tension for schism
        return None

    # Probabilistic schism — lower trust = more likely
    if rng.random() > (0.5 - min_trust):
        return None

    # Split: assign each member to closer seed by trust
    group_a: list[str] = []
    group_b: list[str] = []
    for m in members:
        if m == split_a:
            group_a.append(m)
        elif m == split_b:
            group_b.append(m)
        elif social.get(m, split_a).trust >= social.get(m, split_b).trust:
            group_a.append(m)
        else:
            group_b.append(m)

    if len(group_a) < MIN_FACTION_SIZE or len(group_b) < MIN_FACTION_SIZE:
        return None

    child_a = Faction(
        id=faction.id, member_ids=sorted(group_a), formed_year=year,
        platform=faction.platform,
    )
    child_b = Faction(
        id=f"{faction.id}-split",
        member_ids=sorted(group_b), formed_year=year,
    )
    return [child_a, child_b]


# ── Vote bias ──────────────────────────────────────────────────────────────

def faction_vote_bias(
    state: DiplomacyState,
    colonist: Colonist,
    proposal_gov_type: str,
) -> float:
    """Compute vote bias from faction alignment. Capped at ±VOTE_BIAS_CAP."""
    faction = state.faction_of(colonist.id)
    if not faction or not faction.platform:
        return 0.0

    # Favor proposals matching faction platform
    if proposal_gov_type == faction.platform:
        return VOTE_BIAS_CAP
    # Slight penalty for opposing platform
    return -VOTE_BIAS_CAP * 0.5


# ── Tick function (main entry point) ──────────────────────────────────────

def tick_diplomacy(
    state: DiplomacyState,
    social: SocialGraph,
    colonists: list[Colonist],
    active_ids: list[str],
    year: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Run one year of diplomacy. Returns list of diplomacy events."""
    events: list[dict[str, Any]] = []

    # 1. Detect current factions
    clusters = detect_factions(social, active_ids)
    new_factions = reconcile_factions(state.factions, clusters, year)

    # Assign platforms to new factions
    for f in new_factions:
        if not f.platform:
            f.platform = assign_platform(f, colonists)

    # Compute cohesion
    for f in new_factions:
        f.cohesion = compute_faction_cohesion(f, social)

    # Log faction changes
    old_ids = {f.id for f in state.factions}
    new_ids = {f.id for f in new_factions}
    for fid in new_ids - old_ids:
        f = next(nf for nf in new_factions if nf.id == fid)
        events.append({
            "type": "faction_formed", "year": year,
            "faction_id": fid, "members": f.member_ids,
            "platform": f.platform,
        })
    for fid in old_ids - new_ids:
        events.append({
            "type": "faction_dissolved", "year": year, "faction_id": fid,
        })

    state.factions = new_factions

    # 2. Check for schisms in large factions
    schism_results: list[Faction] = []
    factions_to_remove: list[str] = []
    for f in list(state.factions):
        result = check_schism(f, social, year, rng)
        if result:
            factions_to_remove.append(f.id)
            schism_results.extend(result)
            events.append({
                "type": "schism", "year": year,
                "parent_faction": f.id,
                "child_factions": [r.id for r in result],
            })
    if factions_to_remove:
        state.factions = [f for f in state.factions if f.id not in factions_to_remove]
        for sf in schism_results:
            sf.platform = assign_platform(sf, colonists)
            sf.cohesion = compute_faction_cohesion(sf, social)
        state.factions.extend(schism_results)

    # 3. Try forming alliances
    alliance = try_form_alliance(state, social, year, rng)
    if alliance:
        state.alliances.append(alliance)
        events.append({
            "type": "alliance_formed", "year": year,
            "factions": [alliance.faction_a, alliance.faction_b],
            "strength": round(alliance.strength, 3),
        })

    # 4. Check for alliance breakups
    broken = check_alliance_breakups(state, social)
    for b in broken:
        events.append({
            "type": "alliance_broken", "year": year,
            "factions": [b.faction_a, b.faction_b],
        })

    # Record history
    state.history.extend(events)
    return events
