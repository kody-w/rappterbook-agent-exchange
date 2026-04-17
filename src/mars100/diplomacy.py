"""
Mars-100 diplomacy organ — faction emergence, treaties, bloc voting (engine v11.0).

As the colony grows beyond 10 colonists (births + immigration), ideological
factions emerge from stat-vector similarity and social trust density.  Factions
engage in alliances, trade pacts, and non-aggression agreements.  Bloc voting
gives factions political weight in governance.

Deterministic — uses dedicated RNG stream (seed + 12553).

Design principles:
  - Factions are *detected*, not assigned.  They emerge from clustering.
  - Faction identity is *stable*: tracked by member overlap, not re-invented.
  - Treaties influence, never hard-block (non-aggression reduces exile
    willingness, doesn't prevent it).
  - Bloc voting is probabilistic influence, capped at ±0.25 on vote score.
  - One colonist belongs to at most one active faction.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist, STAT_NAMES

# --- constants ---

MIN_FACTION_SIZE: int = 3
MAX_FACTIONS: int = 4
SIMILARITY_THRESHOLD: float = 0.30
TRUST_DENSITY_THRESHOLD: float = 0.35
HYSTERESIS_OVERLAP: float = 0.5
COHESION_DECAY: float = 0.02
TREATY_PROPOSAL_THRESHOLD: float = 0.45
BLOC_VOTE_CAP: float = 0.25
NON_AGGRESSION_EXILE_REDUCTION: float = 0.4
TRADE_PACT_RESOURCE_BONUS: float = 0.01
INCIDENT_TRUST_PENALTY: float = 0.08
MIN_POPULATION_FOR_FACTIONS: int = 8

FACTION_NAMES: dict[str, list[str]] = {
    "resolve": ["Iron Compact", "Steel Pact", "Resolve Front"],
    "improvisation": ["Free Thinkers", "Spark Collective", "Mavericks"],
    "empathy": ["Harmony Circle", "Empaths Guild", "Unity Bond"],
    "hoarding": ["Resource League", "Vault Alliance", "Stockpile Union"],
    "faith": ["Covenant Assembly", "Faithful Order", "Believers Synod"],
    "paranoia": ["Shadow Watch", "Vigilance Corps", "Sentinel Ring"],
}

TREATY_TYPES: tuple[str, ...] = ("alliance", "trade_pact", "non_aggression")


# --- data classes ---

@dataclass
class Faction:
    """An ideological faction within the colony."""
    id: str
    name: str
    ideology: str
    member_ids: list[str] = field(default_factory=list)
    leader_id: str | None = None
    cohesion: float = 0.5
    founded_year: int = 0
    dissolved_year: int | None = None

    def is_active(self) -> bool:
        return self.dissolved_year is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "ideology": self.ideology,
            "member_ids": list(self.member_ids),
            "leader_id": self.leader_id,
            "cohesion": round(self.cohesion, 4),
            "founded_year": self.founded_year,
            "dissolved_year": self.dissolved_year,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Faction:
        return cls(
            id=d["id"], name=d["name"], ideology=d.get("ideology", "resolve"),
            member_ids=list(d.get("member_ids", [])),
            leader_id=d.get("leader_id"),
            cohesion=d.get("cohesion", 0.5),
            founded_year=d.get("founded_year", 0),
            dissolved_year=d.get("dissolved_year"),
        )


@dataclass
class Treaty:
    """A diplomatic agreement between two factions."""
    id: str
    treaty_type: str
    faction_a: str
    faction_b: str
    start_year: int
    duration: int
    active: bool = True

    def pair_key(self) -> tuple[str, str]:
        return tuple(sorted([self.faction_a, self.faction_b]))  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "treaty_type": self.treaty_type,
            "faction_a": self.faction_a, "faction_b": self.faction_b,
            "start_year": self.start_year, "duration": self.duration,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Treaty:
        return cls(
            id=d["id"], treaty_type=d.get("treaty_type", "alliance"),
            faction_a=d["faction_a"], faction_b=d["faction_b"],
            start_year=d.get("start_year", 0),
            duration=d.get("duration", 10),
            active=d.get("active", True),
        )


@dataclass
class DiplomacyState:
    """Colony-wide diplomatic state."""
    factions: list[Faction] = field(default_factory=list)
    treaties: list[Treaty] = field(default_factory=list)
    incidents: list[dict] = field(default_factory=list)
    next_faction_id: int = 0

    def active_factions(self) -> list[Faction]:
        return [f for f in self.factions if f.is_active()]

    def active_treaties(self) -> list[Treaty]:
        return [t for t in self.treaties if t.active]

    def faction_of(self, colonist_id: str) -> Faction | None:
        """Return the active faction a colonist belongs to, or None."""
        for f in self.active_factions():
            if colonist_id in f.member_ids:
                return f
        return None

    def treaty_between(self, fid_a: str, fid_b: str,
                       treaty_type: str | None = None) -> Treaty | None:
        """Find active treaty between two factions."""
        pair = tuple(sorted([fid_a, fid_b]))
        for t in self.active_treaties():
            if t.pair_key() == pair:
                if treaty_type is None or t.treaty_type == treaty_type:
                    return t
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "factions": [f.to_dict() for f in self.factions],
            "treaties": [t.to_dict() for t in self.treaties],
            "incidents": list(self.incidents[-50:]),
            "next_faction_id": self.next_faction_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiplomacyState:
        return cls(
            factions=[Faction.from_dict(f) for f in d.get("factions", [])],
            treaties=[Treaty.from_dict(t) for t in d.get("treaties", [])],
            incidents=list(d.get("incidents", [])),
            next_faction_id=d.get("next_faction_id", 0),
        )


@dataclass
class DiplomacyTickResult:
    """Result of one year's diplomacy processing."""
    factions_formed: list[dict] = field(default_factory=list)
    factions_dissolved: list[dict] = field(default_factory=list)
    treaties_proposed: list[dict] = field(default_factory=list)
    treaties_expired: list[dict] = field(default_factory=list)
    incidents: list[dict] = field(default_factory=list)
    faction_snapshots: list[dict] = field(default_factory=list)
    membership_changes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factions_formed": self.factions_formed,
            "factions_dissolved": self.factions_dissolved,
            "treaties_proposed": self.treaties_proposed,
            "treaties_expired": self.treaties_expired,
            "incidents": self.incidents,
            "faction_snapshots": self.faction_snapshots,
            "membership_changes": self.membership_changes,
        }


# --- pure functions ---

def stat_vector(colonist: Colonist) -> list[float]:
    """Extract the 6-dimensional stat vector for a colonist."""
    return [getattr(colonist.stats, name) for name in STAT_NAMES]


def stat_similarity(a: Colonist, b: Colonist) -> float:
    """Cosine similarity between two colonists' stat vectors.

    Returns 0.0-1.0.  Higher = more ideologically aligned.
    """
    va = stat_vector(a)
    vb = stat_vector(b)
    dot = sum(x * y for x, y in zip(va, vb))
    mag_a = math.sqrt(sum(x * x for x in va))
    mag_b = math.sqrt(sum(x * x for x in vb))
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


def trust_density(member_ids: list[str], social_edges: dict) -> float:
    """Compute average internal trust within a group.

    social_edges: dict[str, dict[str, Relationship-like]] with .trust
    Returns average trust across all directed internal pairs.
    """
    if len(member_ids) < 2:
        return 0.0
    total = 0.0
    count = 0
    for a in member_ids:
        for b in member_ids:
            if a != b and a in social_edges:
                rel = social_edges[a].get(b)
                if rel is not None:
                    total += rel.trust
                    count += 1
    return total / max(1, count)


def detect_clusters(colonists: list[Colonist],
                    social_edges: dict,
                    rng: random.Random) -> list[list[str]]:
    """Detect ideological clusters using stat similarity + trust density.

    Greedy agglomerative: start from most similar pairs, grow clusters
    that maintain internal trust density above threshold.
    """
    if len(colonists) < MIN_FACTION_SIZE:
        return []

    ids = [c.id for c in colonists]
    by_id = {c.id: c for c in colonists}

    # Compute all pairwise similarities
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(colonists):
        for b in colonists[i + 1:]:
            sim = stat_similarity(a, b)
            pairs.append((a.id, b.id, sim))
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Greedy cluster formation
    assigned: set[str] = set()
    clusters: list[list[str]] = []

    for a_id, b_id, sim in pairs:
        if sim < SIMILARITY_THRESHOLD:
            break
        if a_id in assigned or b_id in assigned:
            continue
        if len(clusters) >= MAX_FACTIONS:
            break

        # Seed a cluster with this pair
        cluster = [a_id, b_id]
        assigned.add(a_id)
        assigned.add(b_id)

        # Try to grow by adding unassigned colonists
        candidates = [cid for cid in ids if cid not in assigned]
        rng.shuffle(candidates)
        for cid in candidates:
            c = by_id[cid]
            avg_sim = sum(stat_similarity(c, by_id[m]) for m in cluster) / len(cluster)
            if avg_sim >= SIMILARITY_THRESHOLD:
                trial = cluster + [cid]
                td = trust_density(trial, social_edges)
                if td >= TRUST_DENSITY_THRESHOLD:
                    cluster.append(cid)
                    assigned.add(cid)

        if len(cluster) >= MIN_FACTION_SIZE:
            clusters.append(cluster)

    return clusters


def match_faction(cluster_ids: list[str],
                  existing_factions: list[Faction]) -> Faction | None:
    """Find an existing faction whose membership overlaps enough with a new cluster.

    Returns the best-matching faction if overlap >= HYSTERESIS_OVERLAP, else None.
    """
    cluster_set = set(cluster_ids)
    best: Faction | None = None
    best_overlap = 0.0
    for faction in existing_factions:
        if not faction.is_active():
            continue
        faction_set = set(faction.member_ids)
        if not faction_set:
            continue
        overlap = len(cluster_set & faction_set) / len(cluster_set | faction_set)
        if overlap >= HYSTERESIS_OVERLAP and overlap > best_overlap:
            best = faction
            best_overlap = overlap
    return best


def determine_ideology(colonists: list[Colonist], member_ids: list[str]) -> str:
    """Determine faction ideology from the dominant stat of its members."""
    by_id = {c.id: c for c in colonists}
    stat_sums: dict[str, float] = {name: 0.0 for name in STAT_NAMES}
    for mid in member_ids:
        c = by_id.get(mid)
        if c:
            for name in STAT_NAMES:
                stat_sums[name] += getattr(c.stats, name)
    return max(stat_sums, key=lambda k: stat_sums[k])


def elect_leader(member_ids: list[str], social_edges: dict,
                 rng: random.Random) -> str | None:
    """Elect a faction leader based on highest average respect from members."""
    if not member_ids:
        return None
    scores: dict[str, float] = {}
    for candidate in member_ids:
        total_respect = 0.0
        count = 0
        for voter in member_ids:
            if voter != candidate and voter in social_edges:
                rel = social_edges[voter].get(candidate)
                if rel is not None:
                    total_respect += rel.respect
                    count += 1
        scores[candidate] = total_respect / max(1, count)
    # Break ties deterministically
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[0][0] if ranked else None


def compute_faction_cohesion(member_ids: list[str], social_edges: dict,
                             colonists_by_id: dict[str, Colonist]) -> float:
    """Compute faction cohesion from internal trust density + stat similarity.

    Returns 0.0-1.0.
    """
    if len(member_ids) < 2:
        return 0.0
    td = trust_density(member_ids, social_edges)
    members = [colonists_by_id[mid] for mid in member_ids if mid in colonists_by_id]
    if len(members) < 2:
        return td
    # Average pairwise stat similarity
    total_sim = 0.0
    count = 0
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            total_sim += stat_similarity(a, b)
            count += 1
    avg_sim = total_sim / max(1, count)
    return max(0.0, min(1.0, td * 0.6 + avg_sim * 0.4))


def propose_treaty(faction_a: Faction, faction_b: Faction,
                   state: DiplomacyState, year: int,
                   rng: random.Random) -> Treaty | None:
    """Attempt to propose a treaty between two factions.

    Requires both leaders to have cohesion > threshold.
    Enforces (sorted_pair, treaty_type) uniqueness.
    """
    if faction_a.cohesion < TREATY_PROPOSAL_THRESHOLD:
        return None
    if faction_b.cohesion < TREATY_PROPOSAL_THRESHOLD:
        return None
    if not faction_a.leader_id or not faction_b.leader_id:
        return None

    # Pick treaty type not already active between this pair
    pair = tuple(sorted([faction_a.id, faction_b.id]))
    active_types = set()
    for t in state.active_treaties():
        if t.pair_key() == pair:
            active_types.add(t.treaty_type)

    available = [tt for tt in TREATY_TYPES if tt not in active_types]
    if not available:
        return None

    treaty_type = rng.choice(available)
    duration = rng.choice([5, 10, 15])
    tid = f"treaty-y{year}-{faction_a.id}-{faction_b.id}-{treaty_type}"

    return Treaty(
        id=tid, treaty_type=treaty_type,
        faction_a=faction_a.id, faction_b=faction_b.id,
        start_year=year, duration=duration,
    )


def resolve_incidents(factions: list[Faction],
                      events: list[dict],
                      actions: dict[str, str],
                      year: int,
                      rng: random.Random) -> list[dict]:
    """Generate diplomatic incidents from inter-faction conflicts.

    Sabotage actions between members of different factions create incidents.
    Severe events affecting faction members create friction.
    """
    incidents: list[dict] = []
    faction_of: dict[str, str] = {}
    for f in factions:
        for mid in f.member_ids:
            faction_of[mid] = f.id

    # Sabotage-driven incidents
    for cid, action in actions.items():
        if action != "sabotage":
            continue
        aggressor_faction = faction_of.get(cid)
        if not aggressor_faction:
            continue
        for fid, f_id in faction_of.items():
            if fid != cid and f_id != aggressor_faction:
                if rng.random() < 0.3:
                    incidents.append({
                        "year": year, "type": "sabotage_friction",
                        "aggressor": cid, "aggressor_faction": aggressor_faction,
                        "victim_faction": f_id,
                    })
                break

    # Severe-event driven incidents
    for ev in events:
        severity = ev.get("severity", 0.0)
        if severity > 0.6 and len(factions) >= 2 and rng.random() < 0.2:
            f1, f2 = rng.sample(factions, 2)
            incidents.append({
                "year": year, "type": "crisis_blame",
                "faction_a": f1.id, "faction_b": f2.id,
                "event": ev.get("name", "unknown"),
            })

    return incidents


def compute_bloc_vote_influence(colonist_id: str, proposal_gov_type: str,
                                faction: Faction | None,
                                leader_preference: float) -> float:
    """Compute faction influence on a colonist's governance vote.

    Returns a bounded modifier (capped at ±BLOC_VOTE_CAP) to add to the
    colonist's independent vote score.  If colonist is not in a faction,
    returns 0.
    """
    if faction is None:
        return 0.0

    # Leader preference is already computed; scale by faction cohesion
    influence = leader_preference * faction.cohesion * 0.5

    # Ideology alignment bonus
    ideology_alignment: dict[str, dict[str, float]] = {
        "resolve": {"dictator": 0.1, "council": 0.05},
        "empathy": {"consensus": 0.1, "council": 0.08},
        "faith": {"lottery": 0.08, "ai_governor": -0.05},
        "improvisation": {"anarchy": 0.08, "ai_governor": 0.05},
        "hoarding": {"dictator": 0.05, "anarchy": 0.08},
        "paranoia": {"anarchy": 0.1, "consensus": -0.05},
    }
    ideology_bonus = ideology_alignment.get(
        faction.ideology, {}).get(proposal_gov_type, 0.0)
    influence += ideology_bonus

    return max(-BLOC_VOTE_CAP, min(BLOC_VOTE_CAP, influence))


def compute_diplomacy_pressure(state: DiplomacyState,
                               colonist_id: str,
                               actions: list[str]) -> dict[str, float]:
    """Compute action-weight modifiers from faction membership and treaties.

    Faction members avoid sabotaging allies, cooperate more within faction.
    """
    deltas: dict[str, float] = {a: 0.0 for a in actions}
    faction = state.faction_of(colonist_id)
    if faction is None:
        return deltas

    # Faction members cooperate more
    deltas["cooperate"] = deltas.get("cooperate", 0.0) + 0.15 * faction.cohesion
    deltas["mediate"] = deltas.get("mediate", 0.0) + 0.05 * faction.cohesion

    # Faction members sabotage less (unless paranoia-faction)
    if faction.ideology != "paranoia":
        deltas["sabotage"] = deltas.get("sabotage", 0.0) - 0.1 * faction.cohesion

    # Alliance treaties boost cooperation further
    for treaty in state.active_treaties():
        if treaty.treaty_type == "alliance":
            if faction.id in (treaty.faction_a, treaty.faction_b):
                deltas["cooperate"] = deltas.get("cooperate", 0.0) + 0.05

    return deltas


def compute_exile_modifier(state: DiplomacyState,
                           target_id: str,
                           voter_id: str) -> float:
    """Compute exile willingness modifier from faction treaties.

    Non-aggression pacts between factions reduce exile willingness.
    Returns a multiplier (0.0-1.0) on exile probability.
    """
    target_faction = state.faction_of(target_id)
    voter_faction = state.faction_of(voter_id)
    if target_faction is None or voter_faction is None:
        return 1.0
    if target_faction.id == voter_faction.id:
        return 0.3  # Faction members resist exiling each other

    treaty = state.treaty_between(
        target_faction.id, voter_faction.id, "non_aggression")
    if treaty is not None:
        return 1.0 - NON_AGGRESSION_EXILE_REDUCTION
    return 1.0


def compute_loneliness_reduction(state: DiplomacyState,
                                 colonist_id: str) -> float:
    """Faction membership reduces loneliness.

    Returns a non-negative reduction amount (to subtract from loneliness).
    """
    faction = state.faction_of(colonist_id)
    if faction is None:
        return 0.0
    member_count = len(faction.member_ids)
    return min(0.06, 0.02 * (member_count - 1) * faction.cohesion)


def compute_trade_pact_bonus(state: DiplomacyState) -> dict[str, float]:
    """Compute resource bonuses from active trade pact treaties.

    Each trade pact adds a small bonus to food and water.
    """
    bonuses: dict[str, float] = {}
    trade_count = sum(1 for t in state.active_treaties()
                      if t.treaty_type == "trade_pact")
    if trade_count > 0:
        bonuses["food"] = TRADE_PACT_RESOURCE_BONUS * trade_count
        bonuses["water"] = TRADE_PACT_RESOURCE_BONUS * trade_count * 0.5
    return bonuses


def compute_fragmentation(state: DiplomacyState,
                          active_count: int) -> float:
    """Compute colony fragmentation from faction structure.

    High fragmentation = many factions with low inter-faction trust.
    Returns 0.0-1.0 where 0 = unified, 1 = deeply fragmented.
    """
    active = state.active_factions()
    if len(active) <= 1:
        return 0.0
    factioned = sum(len(f.member_ids) for f in active)
    faction_ratio = factioned / max(1, active_count)

    # More factions relative to population = more fragmentation
    faction_pressure = len(active) / MAX_FACTIONS

    # Treaties reduce fragmentation
    treaty_reduction = len(state.active_treaties()) * 0.05

    raw = faction_ratio * faction_pressure - treaty_reduction
    return max(0.0, min(1.0, raw))


def tick_diplomacy(state: DiplomacyState,
                   colonists: list[Colonist],
                   social_edges: dict,
                   events: list[dict],
                   actions: dict[str, str],
                   year: int,
                   rng: random.Random) -> DiplomacyTickResult:
    """Advance diplomacy by one year.

    Detects/evolves factions, proposes treaties, resolves incidents,
    expires old treaties.  Pure-ish function (mutates state in place).
    """
    result = DiplomacyTickResult()
    active = [c for c in colonists if c.is_active()]
    active_ids = {c.id for c in active}
    by_id = {c.id: c for c in active}

    # --- Prune dead/exiled from existing factions ---
    for faction in state.active_factions():
        before = set(faction.member_ids)
        faction.member_ids = [mid for mid in faction.member_ids
                              if mid in active_ids]
        removed = before - set(faction.member_ids)
        for rid in removed:
            result.membership_changes.append({
                "year": year, "colonist_id": rid,
                "faction_id": faction.id, "change": "removed_inactive",
            })
        # Re-elect leader if removed
        if faction.leader_id not in faction.member_ids:
            faction.leader_id = elect_leader(
                faction.member_ids, social_edges, rng)

    # --- Dissolve undersized factions ---
    for faction in state.active_factions():
        if len(faction.member_ids) < MIN_FACTION_SIZE:
            faction.dissolved_year = year
            result.factions_dissolved.append({
                "id": faction.id, "name": faction.name, "year": year,
                "reason": "undersized",
            })
            # Expire treaties referencing dissolved faction
            for treaty in state.active_treaties():
                if faction.id in (treaty.faction_a, treaty.faction_b):
                    treaty.active = False
                    result.treaties_expired.append({
                        "id": treaty.id, "year": year,
                        "reason": "faction_dissolved",
                    })

    # --- Detect new clusters (only if population is large enough) ---
    if len(active) >= MIN_POPULATION_FOR_FACTIONS:
        # Only consider colonists not in active factions
        factioned_ids = set()
        for f in state.active_factions():
            factioned_ids.update(f.member_ids)
        unaffiliated = [c for c in active if c.id not in factioned_ids]

        if len(unaffiliated) >= MIN_FACTION_SIZE:
            new_clusters = detect_clusters(unaffiliated, social_edges, rng)
            for cluster_ids in new_clusters:
                if len(state.active_factions()) >= MAX_FACTIONS:
                    break
                # Check for overlap with dissolved factions (possible revival)
                matched = match_faction(cluster_ids, state.factions)
                if matched and not matched.is_active():
                    # Revive faction with new membership
                    matched.dissolved_year = None
                    matched.member_ids = list(cluster_ids)
                    matched.ideology = determine_ideology(active, cluster_ids)
                    matched.leader_id = elect_leader(
                        cluster_ids, social_edges, rng)
                    matched.cohesion = compute_faction_cohesion(
                        cluster_ids, social_edges, by_id)
                    result.factions_formed.append({
                        "id": matched.id, "name": matched.name,
                        "year": year, "revival": True,
                        "members": list(cluster_ids),
                    })
                else:
                    # Create new faction
                    ideology = determine_ideology(active, cluster_ids)
                    name_pool = FACTION_NAMES.get(ideology, ["Unknown Bloc"])
                    used_names = {f.name for f in state.factions}
                    available_names = [n for n in name_pool if n not in used_names]
                    fname = (available_names[0] if available_names
                             else f"{ideology.title()} Bloc {state.next_faction_id}")
                    fid = f"faction-{state.next_faction_id}"
                    state.next_faction_id += 1
                    leader = elect_leader(cluster_ids, social_edges, rng)
                    cohesion = compute_faction_cohesion(
                        cluster_ids, social_edges, by_id)
                    new_faction = Faction(
                        id=fid, name=fname, ideology=ideology,
                        member_ids=list(cluster_ids), leader_id=leader,
                        cohesion=cohesion, founded_year=year,
                    )
                    state.factions.append(new_faction)
                    result.factions_formed.append({
                        "id": fid, "name": fname, "year": year,
                        "revival": False, "members": list(cluster_ids),
                    })

    # --- Evolve existing factions: recruit unaffiliated colonists ---
    factioned_ids = set()
    for f in state.active_factions():
        factioned_ids.update(f.member_ids)
    unaffiliated = [c for c in active if c.id not in factioned_ids]

    for faction in state.active_factions():
        for candidate in unaffiliated:
            if candidate.id in factioned_ids:
                continue
            avg_sim = sum(
                stat_similarity(candidate, by_id[mid])
                for mid in faction.member_ids if mid in by_id
            ) / max(1, len(faction.member_ids))
            if avg_sim < SIMILARITY_THRESHOLD:
                continue
            trial = faction.member_ids + [candidate.id]
            td = trust_density(trial, social_edges)
            if td >= TRUST_DENSITY_THRESHOLD * 0.9:  # Slightly lower bar for joining
                faction.member_ids.append(candidate.id)
                factioned_ids.add(candidate.id)
                result.membership_changes.append({
                    "year": year, "colonist_id": candidate.id,
                    "faction_id": faction.id, "change": "joined",
                })

    # --- Update faction properties ---
    for faction in state.active_factions():
        faction.ideology = determine_ideology(active, faction.member_ids)
        faction.leader_id = elect_leader(
            faction.member_ids, social_edges, rng)
        faction.cohesion = compute_faction_cohesion(
            faction.member_ids, social_edges, by_id)
        # Cohesion decays slightly each year
        faction.cohesion = max(0.0, faction.cohesion - COHESION_DECAY)

    # --- Expire old treaties ---
    for treaty in state.active_treaties():
        if year >= treaty.start_year + treaty.duration:
            treaty.active = False
            result.treaties_expired.append({
                "id": treaty.id, "year": year, "reason": "expired",
            })

    # --- Propose new treaties ---
    active_factions = state.active_factions()
    if len(active_factions) >= 2:
        for i, fa in enumerate(active_factions):
            for fb in active_factions[i + 1:]:
                if rng.random() < 0.15:
                    treaty = propose_treaty(fa, fb, state, year, rng)
                    if treaty is not None:
                        state.treaties.append(treaty)
                        result.treaties_proposed.append(treaty.to_dict())

    # --- Resolve diplomatic incidents ---
    event_dicts = events if isinstance(events, list) else []
    year_incidents = resolve_incidents(
        active_factions, event_dicts, actions, year, rng)
    state.incidents.extend(year_incidents)
    result.incidents = year_incidents

    # Trim incident log
    if len(state.incidents) > 100:
        state.incidents = state.incidents[-100:]

    # --- Snapshot ---
    result.faction_snapshots = [f.to_dict() for f in state.active_factions()]

    return result
