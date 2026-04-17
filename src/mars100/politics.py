"""
Politics organ for Mars-100 colony simulation (engine v9.0).

Models emergent political behaviour: per-colonist opinions on three
ideological axes, faction formation via deterministic agglomerative
clustering, government legitimacy tracking, and action-selection
perturbation from psychological + political state.

Phase 1 scope:
  - PoliticalState per colonist (three opinion axes + engagement)
  - Opinion evolution from personality, events, social influence, gov performance
  - Deterministic faction detection (agglomerative, max 4)
  - Legitimacy tracking (observable signals only)
  - Action-weight modifiers from psych + politics (fulfils v8 deferral)
  - Downstream hook: low legitimacy triggers governance proposal

Deferred to v10:
  - Enacted laws / policies with resource consumers
  - Revolution mechanic (beyond triggering proposals)
  - Faction platforms influencing governance proposals
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# -- constants ---------------------------------------------------------------

OPINION_AXES = ("liberty_vs_security", "growth_vs_sustainability",
                "individual_vs_collective")

# Personality → opinion axis mapping (stat_name → (axis, sign, weight))
PERSONALITY_PRESSURE: list[tuple[str, str, float]] = [
    ("resolve",        "liberty_vs_security",      0.02),
    ("paranoia",       "liberty_vs_security",      0.03),
    ("empathy",        "individual_vs_collective",  0.03),
    ("hoarding",       "individual_vs_collective", -0.03),
    ("improvisation",  "growth_vs_sustainability", -0.02),
    ("faith",          "growth_vs_sustainability",  0.02),
]

SOCIAL_INFLUENCE_WEIGHT = 0.04
EVENT_PRESSURE_SCALE = 0.06
RESOURCE_PRESSURE_SCALE = 0.05
GOV_PERFORMANCE_SCALE = 0.03

OPINION_CAP_DELTA = 0.15
ENGAGEMENT_DECAY = 0.02
ENGAGEMENT_CAP_DELTA = 0.10

FACTION_DISTANCE_THRESHOLD = 0.50
MIN_FACTION_SIZE = 3
MAX_FACTIONS = 4

LEGITIMACY_INITIAL = 0.60
LEGITIMACY_CAP_DELTA = 0.15
LEGITIMACY_PROPOSAL_THRESHOLD = 0.20

FACTION_NAMES = ["Vanguard", "Stewards", "Sovereigns", "Commons",
                 "Outriders", "Foundry", "Shepherds", "Sentinels"]


# -- data classes ------------------------------------------------------------

@dataclass
class PoliticalState:
    """Per-colonist political state.  Evolves each year."""
    liberty_vs_security: float = 0.0
    growth_vs_sustainability: float = 0.0
    individual_vs_collective: float = 0.0
    engagement: float = 0.30
    faction_id: str | None = None

    def opinion_vector(self) -> tuple[float, float, float]:
        return (self.liberty_vs_security,
                self.growth_vs_sustainability,
                self.individual_vs_collective)

    def to_dict(self) -> dict:
        return {
            "liberty_vs_security": round(self.liberty_vs_security, 4),
            "growth_vs_sustainability": round(self.growth_vs_sustainability, 4),
            "individual_vs_collective": round(self.individual_vs_collective, 4),
            "engagement": round(self.engagement, 4),
            "faction_id": self.faction_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PoliticalState:
        return cls(
            liberty_vs_security=d.get("liberty_vs_security", 0.0),
            growth_vs_sustainability=d.get("growth_vs_sustainability", 0.0),
            individual_vs_collective=d.get("individual_vs_collective", 0.0),
            engagement=d.get("engagement", 0.30),
            faction_id=d.get("faction_id"),
        )


@dataclass
class Faction:
    """An emergent political faction."""
    id: str
    name: str
    formation_year: int
    member_ids: list[str] = field(default_factory=list)
    centroid: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dominant_axis: str = "liberty_vs_security"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "formation_year": self.formation_year,
            "member_ids": list(self.member_ids),
            "centroid": [round(c, 4) for c in self.centroid],
            "dominant_axis": self.dominant_axis,
        }


@dataclass
class PoliticsTickResult:
    """Result of one year's politics tick."""
    snapshots: dict[str, dict] = field(default_factory=dict)
    factions: list[Faction] = field(default_factory=list)
    legitimacy: float = 0.60
    legitimacy_delta: float = 0.0
    proposal_triggered: bool = False
    faction_changes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "snapshots": self.snapshots,
            "factions": [f.to_dict() for f in self.factions],
            "legitimacy": round(self.legitimacy, 4),
            "legitimacy_delta": round(self.legitimacy_delta, 4),
            "proposal_triggered": self.proposal_triggered,
            "faction_changes": self.faction_changes,
        }


@dataclass
class PoliticsContext:
    """Year context for the politics tick."""
    colonist_id: str
    stats: dict[str, float]
    trusted_ids: list[str]
    trust_weights: list[float]
    event_severity: float
    resource_avg: float
    resource_delta: float
    gov_type: str
    had_crisis: bool
    action: str


# -- pure helpers ------------------------------------------------------------

def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _cap_delta(delta: float, cap: float) -> float:
    return max(-cap, min(cap, delta))


def _opinion_distance(a: tuple[float, float, float],
                      b: tuple[float, float, float]) -> float:
    """Euclidean distance in 3D opinion space."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# -- opinion evolution -------------------------------------------------------

def compute_personality_pressure(stats: dict[str, float]) -> dict[str, float]:
    """Compute opinion deltas from personality traits."""
    deltas: dict[str, float] = {ax: 0.0 for ax in OPINION_AXES}
    for stat_name, axis, weight in PERSONALITY_PRESSURE:
        val = stats.get(stat_name, 0.5)
        deltas[axis] += (val - 0.5) * weight
    return deltas


def compute_social_influence(
    own_opinions: tuple[float, float, float],
    trusted_opinions: list[tuple[float, float, float]],
    trust_weights: list[float],
) -> dict[str, float]:
    """Trust-weighted pull toward opinions of trusted colonists."""
    deltas: dict[str, float] = {ax: 0.0 for ax in OPINION_AXES}
    if not trusted_opinions:
        return deltas
    total_weight = sum(trust_weights)
    if total_weight < 0.001:
        return deltas
    for i, (ax_name, own_v) in enumerate(zip(OPINION_AXES, own_opinions)):
        weighted_sum = sum(
            tw * top[i] for tw, top in zip(trust_weights, trusted_opinions))
        avg_trusted = weighted_sum / total_weight
        deltas[ax_name] = (avg_trusted - own_v) * SOCIAL_INFLUENCE_WEIGHT
    return deltas


def compute_event_pressure(event_severity: float) -> dict[str, float]:
    """Crises push toward security and collective action."""
    pressure = event_severity * EVENT_PRESSURE_SCALE
    return {
        "liberty_vs_security": pressure,
        "growth_vs_sustainability": pressure * 0.5,
        "individual_vs_collective": pressure * 0.7,
    }


def compute_resource_pressure(
    resource_avg: float,
    resource_delta: float,
) -> dict[str, float]:
    """Scarcity → collective/security; abundance → individual/growth."""
    scarcity = max(0.0, 0.5 - resource_avg) * RESOURCE_PRESSURE_SCALE
    trend = resource_delta * RESOURCE_PRESSURE_SCALE * 0.5
    return {
        "liberty_vs_security": scarcity - trend,
        "growth_vs_sustainability": -trend,
        "individual_vs_collective": scarcity,
    }


def compute_engagement_delta(
    action: str,
    had_crisis: bool,
    event_severity: float,
) -> float:
    """Compute engagement change.  Political actions boost it."""
    delta = -ENGAGEMENT_DECAY
    if action in ("mediate", "cooperate"):
        delta += 0.04
    if action == "sabotage":
        delta += 0.06
    if had_crisis:
        delta += 0.05
    if event_severity > 0.5:
        delta += 0.03
    return _cap_delta(delta, ENGAGEMENT_CAP_DELTA)


# -- faction detection -------------------------------------------------------

def detect_factions(
    politics_map: dict[str, PoliticalState],
    active_ids: list[str],
    year: int,
    existing_factions: list[Faction],
    rng: random.Random,
) -> list[Faction]:
    """Deterministic agglomerative faction detection.

    1. Sort active colonists by ID (deterministic order).
    2. Compute pairwise opinion distances.
    3. Greedily merge closest pairs below threshold.
    4. Label clusters of size >= MIN_FACTION_SIZE as factions.
    5. Cap at MAX_FACTIONS (keep largest).
    """
    if len(active_ids) < MIN_FACTION_SIZE:
        for cid in active_ids:
            if cid in politics_map:
                politics_map[cid].faction_id = None
        return []

    sorted_ids = sorted(active_ids)
    opinions = {cid: politics_map[cid].opinion_vector()
                for cid in sorted_ids if cid in politics_map}
    if len(opinions) < MIN_FACTION_SIZE:
        return []

    # Union-Find for deterministic clustering
    parent: dict[str, str] = {cid: cid for cid in opinions}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Compute all pairwise distances, sorted for determinism
    pairs: list[tuple[float, str, str]] = []
    ids_list = list(opinions.keys())
    for i in range(len(ids_list)):
        for j in range(i + 1, len(ids_list)):
            d = _opinion_distance(opinions[ids_list[i]], opinions[ids_list[j]])
            pairs.append((d, ids_list[i], ids_list[j]))
    pairs.sort()

    for dist, a, b in pairs:
        if dist > FACTION_DISTANCE_THRESHOLD:
            break
        union(a, b)

    # Collect clusters
    clusters: dict[str, list[str]] = {}
    for cid in ids_list:
        root = find(cid)
        clusters.setdefault(root, []).append(cid)

    # Filter by size, sort by size descending, cap
    viable = [(root, members) for root, members in clusters.items()
              if len(members) >= MIN_FACTION_SIZE]
    viable.sort(key=lambda x: (-len(x[1]), x[0]))
    viable = viable[:MAX_FACTIONS]

    # Reuse existing faction IDs where possible (by centroid proximity)
    existing_by_id = {f.id: f for f in existing_factions}
    name_pool = list(FACTION_NAMES)
    used_names: set[str] = set()

    new_factions: list[Faction] = []
    for idx, (root, members) in enumerate(viable):
        centroid = _compute_centroid(opinions, members)
        dominant = _dominant_axis(centroid)

        # Try to match an existing faction
        matched_id = None
        best_dist = float("inf")
        for ef in existing_factions:
            d = _opinion_distance(centroid, ef.centroid)
            if d < best_dist and d < FACTION_DISTANCE_THRESHOLD * 0.6:
                best_dist = d
                matched_id = ef.id

        if matched_id and matched_id not in [f.id for f in new_factions]:
            fid = matched_id
            fname = existing_by_id[matched_id].name
        else:
            fid = f"faction-{year}-{idx}"
            available = [n for n in name_pool if n not in used_names]
            fname = available[idx % len(available)] if available else f"Bloc-{idx}"

        used_names.add(fname)
        faction = Faction(
            id=fid, name=fname, formation_year=year,
            member_ids=sorted(members), centroid=centroid,
            dominant_axis=dominant,
        )
        new_factions.append(faction)

    # Update colonist faction assignments
    member_to_faction: dict[str, str] = {}
    for f in new_factions:
        for cid in f.member_ids:
            member_to_faction[cid] = f.id

    for cid in active_ids:
        if cid in politics_map:
            politics_map[cid].faction_id = member_to_faction.get(cid)

    return new_factions


def _compute_centroid(
    opinions: dict[str, tuple[float, float, float]],
    member_ids: list[str],
) -> tuple[float, float, float]:
    """Compute average opinion of a group."""
    n = len(member_ids)
    if n == 0:
        return (0.0, 0.0, 0.0)
    sums = [0.0, 0.0, 0.0]
    for cid in member_ids:
        op = opinions.get(cid, (0.0, 0.0, 0.0))
        for i in range(3):
            sums[i] += op[i]
    return (sums[0] / n, sums[1] / n, sums[2] / n)


def _dominant_axis(centroid: tuple[float, float, float]) -> str:
    """Which axis has the strongest opinion in the centroid."""
    abs_vals = [(abs(centroid[i]), OPINION_AXES[i]) for i in range(3)]
    abs_vals.sort(key=lambda x: (-x[0], x[1]))
    return abs_vals[0][1]


# -- legitimacy --------------------------------------------------------------

def compute_legitimacy_delta(
    resource_delta_avg: float,
    gov_type: str,
    factions: list[Faction],
    active_count: int,
    year: int,
    event_severity: float,
) -> float:
    """Compute legitimacy change from observable signals.

    Rises: resources improving, large faction aligned with gov.
    Falls: resources declining, crises, fragmented factions.
    """
    delta = 0.0
    # Resource trend
    delta += resource_delta_avg * 3.0

    # Crisis penalty
    if event_severity > 0.5:
        delta -= event_severity * 0.08

    # Faction alignment bonus: largest faction supporting gov
    if factions:
        largest = max(factions, key=lambda f: len(f.member_ids))
        coverage = len(largest.member_ids) / max(1, active_count)
        delta += (coverage - 0.3) * 0.05
    else:
        delta -= 0.02

    # Stability bonus for long-running government
    if gov_type not in ("anarchy",):
        delta += 0.01
    else:
        delta -= 0.01

    return _cap_delta(delta, LEGITIMACY_CAP_DELTA)


# -- action modifiers (fulfils v8 deferral) ----------------------------------

def compute_action_modifiers(
    psych_stress: float,
    psych_purpose: float,
    psych_morale: float,
    political_engagement: float,
    legitimacy: float,
    faction_id: str | None,
) -> dict[str, float]:
    """Compute action-weight modifiers from psychological + political state.

    Uses *previous year's* psych/politics to influence current action choice.
    Returns additive weight modifiers for each action.
    """
    mods: dict[str, float] = {}

    # High stress → bias rest/pray
    if psych_stress > 0.6:
        excess = psych_stress - 0.6
        mods["rest"] = mods.get("rest", 0.0) + excess * 2.0
        mods["pray"] = mods.get("pray", 0.0) + excess * 1.0

    # Low purpose → bias sabotage
    if psych_purpose < 0.3:
        deficit = 0.3 - psych_purpose
        mods["sabotage"] = mods.get("sabotage", 0.0) + deficit * 1.5

    # High morale → bias productive actions
    if psych_morale > 0.7:
        excess = psych_morale - 0.7
        mods["terraform"] = mods.get("terraform", 0.0) + excess * 1.0
        mods["research"] = mods.get("research", 0.0) + excess * 1.0
        mods["cooperate"] = mods.get("cooperate", 0.0) + excess * 0.8

    # Low legitimacy + high engagement → unrest actions
    if legitimacy < 0.35 and political_engagement > 0.5:
        unrest = (0.35 - legitimacy) * political_engagement
        mods["sabotage"] = mods.get("sabotage", 0.0) + unrest * 2.0
        mods["mediate"] = mods.get("mediate", 0.0) + unrest * 1.0

    # Faction membership → cooperate bias
    if faction_id is not None:
        mods["cooperate"] = mods.get("cooperate", 0.0) + 0.3
        mods["mediate"] = mods.get("mediate", 0.0) + 0.2

    return mods


# -- main tick ---------------------------------------------------------------

def tick_politics(
    politics_map: dict[str, PoliticalState],
    contexts: list[PoliticsContext],
    factions: list[Faction],
    legitimacy: float,
    year: int,
    rng: random.Random,
) -> PoliticsTickResult:
    """Run one year of political evolution.  Mutates politics_map in place.

    Returns updated factions and legitimacy.
    """
    snapshots: dict[str, dict] = {}
    faction_changes: list[dict] = []
    active_ids = [ctx.colonist_id for ctx in contexts]

    # Build opinion lookup for social influence
    opinion_lookup: dict[str, tuple[float, float, float]] = {}
    for cid in active_ids:
        ps = politics_map.get(cid)
        if ps:
            opinion_lookup[cid] = ps.opinion_vector()

    for ctx in contexts:
        cid = ctx.colonist_id
        ps = politics_map.get(cid)
        if ps is None:
            ps = _init_political_state(ctx.stats, rng)
            politics_map[cid] = ps

        # Compute opinion deltas from multiple sources
        personality_d = compute_personality_pressure(ctx.stats)
        event_d = compute_event_pressure(ctx.event_severity)
        resource_d = compute_resource_pressure(ctx.resource_avg, ctx.resource_delta)

        # Social influence from trusted colonists
        trusted_ops = [opinion_lookup[tid] for tid in ctx.trusted_ids
                       if tid in opinion_lookup]
        social_d = compute_social_influence(
            ps.opinion_vector(), trusted_ops, ctx.trust_weights[:len(trusted_ops)])

        # Apply deltas
        for axis in OPINION_AXES:
            total = (personality_d.get(axis, 0.0)
                     + event_d.get(axis, 0.0)
                     + resource_d.get(axis, 0.0)
                     + social_d.get(axis, 0.0))
            total = _cap_delta(total, OPINION_CAP_DELTA)
            current = getattr(ps, axis)
            setattr(ps, axis, _clamp(current + total))

        # Engagement
        eng_d = compute_engagement_delta(
            ctx.action, ctx.had_crisis, ctx.event_severity)
        ps.engagement = _clamp01(ps.engagement + eng_d)

        snapshots[cid] = ps.to_dict()

    # Detect factions
    old_assignments = {cid: politics_map[cid].faction_id
                       for cid in active_ids if cid in politics_map}
    new_factions = detect_factions(politics_map, active_ids, year, factions, rng)
    new_assignments = {cid: politics_map[cid].faction_id
                       for cid in active_ids if cid in politics_map}

    # Track faction changes
    for cid in active_ids:
        old_f = old_assignments.get(cid)
        new_f = new_assignments.get(cid)
        if old_f != new_f:
            faction_changes.append({"colonist_id": cid, "from": old_f, "to": new_f})

    # Legitimacy
    avg_resource_delta = 0.0
    if contexts:
        avg_resource_delta = sum(c.resource_delta for c in contexts) / len(contexts)
    avg_event_severity = 0.0
    if contexts:
        avg_event_severity = sum(c.event_severity for c in contexts) / len(contexts)

    leg_delta = compute_legitimacy_delta(
        avg_resource_delta,
        contexts[0].gov_type if contexts else "anarchy",
        new_factions, len(active_ids), year,
        avg_event_severity)
    new_legitimacy = _clamp01(legitimacy + leg_delta)

    proposal_triggered = new_legitimacy < LEGITIMACY_PROPOSAL_THRESHOLD

    return PoliticsTickResult(
        snapshots=snapshots,
        factions=new_factions,
        legitimacy=new_legitimacy,
        legitimacy_delta=leg_delta,
        proposal_triggered=proposal_triggered,
        faction_changes=faction_changes,
    )


def _init_political_state(stats: dict[str, float],
                          rng: random.Random) -> PoliticalState:
    """Initialize political state from personality stats + noise."""
    ps = PoliticalState()
    pressure = compute_personality_pressure(stats)
    for axis in OPINION_AXES:
        base = pressure.get(axis, 0.0) * 5.0
        setattr(ps, axis, _clamp(base + rng.gauss(0, 0.15)))
    ps.engagement = _clamp01(0.3 + rng.gauss(0, 0.1))
    return ps
