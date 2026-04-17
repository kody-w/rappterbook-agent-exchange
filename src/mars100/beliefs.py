"""
Belief Systems organ for Mars-100 colony simulation (engine v9.0).

Models emergent ideological diversity along four axes:
  collectivism  (-1 individualist … +1 collectivist)
  authority     (-1 libertarian  … +1 authoritarian)
  spiritualism  (-1 materialist  … +1 spiritual)
  risk_appetite (-1 cautious     … +1 bold)

Beliefs propagate through trust-weighted social influence, shift in
response to experiences (events, deaths, governance changes), cluster
into emergent factions, and bias action-selection and governance voting.

Key design choices:
  - Updates are SYNCHRONOUS: all next-year beliefs computed from a
    frozen snapshot, then committed.  Order-independent.
  - Social propagation is approximately mean-preserving.
  - Per-axis max yearly delta is capped to prevent runaway drift.
  - Martyrdom effects are bounded and decay over a fixed horizon.
  - Children inherit blended parental beliefs + random drift.
  - Immigrants receive beliefs loosely initialized from their stats.

Phase 1 scope (v9.0):
  - BeliefState per colonist (4 axes)
  - tick_beliefs(): social propagation + experience shifts
  - Faction detection (thresholded distance, min size 2)
  - Martyrdom effect (bounded, decaying)
  - Belief-based action weight modifiers
  - Belief-based governance vote bias
  - Defer: belief-driven sub-sim topics (v10+)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# -- constants ---------------------------------------------------------------

BELIEF_AXES = ("collectivism", "authority", "spiritualism", "risk_appetite")
BELIEF_CAP_DELTA = 0.15          # max per-year shift on any axis
PROPAGATION_RATE = 0.04          # how fast trusted peers pull beliefs
EXPERIENCE_SCALE = 0.08          # base magnitude of event-driven shifts
MARTYRDOM_INITIAL = 0.10         # initial belief shift toward martyr's beliefs
MARTYRDOM_DECAY = 0.5            # multiplicative decay per year
MARTYRDOM_HORIZON = 3            # years after which effect is zeroed
FACTION_DISTANCE_THRESHOLD = 0.6 # max distance to be in same faction
MIN_FACTION_SIZE = 2
CHILD_DRIFT = 0.15               # std dev of random drift for child beliefs
ACTION_BELIEF_WEIGHT = 0.4       # max total action weight shift from beliefs
GOV_BELIEF_BIAS = 0.2            # max governance vote bias from beliefs


# -- data classes ------------------------------------------------------------

@dataclass
class BeliefState:
    """Per-colonist ideological position on four axes, each [-1, +1]."""
    collectivism: float = 0.0
    authority: float = 0.0
    spiritualism: float = 0.0
    risk_appetite: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {axis: round(getattr(self, axis), 4) for axis in BELIEF_AXES}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> BeliefState:
        return cls(**{k: max(-1.0, min(1.0, d.get(k, 0.0))) for k in BELIEF_AXES})

    def distance(self, other: BeliefState) -> float:
        """Euclidean distance in 4D belief space, normalized to [0, 1]."""
        raw = math.sqrt(sum(
            (getattr(self, a) - getattr(other, a)) ** 2
            for a in BELIEF_AXES
        ))
        # max possible distance = sqrt(4 * 4) = 4.0 (corners of [-1,1]^4)
        return raw / 4.0

    def copy(self) -> BeliefState:
        return BeliefState(**{a: getattr(self, a) for a in BELIEF_AXES})


@dataclass
class MartyrdomEffect:
    """A lingering ideological shift from a dead colonist."""
    source_id: str
    belief_snapshot: dict[str, float]
    year_of_death: int
    affected_ids: list[str] = field(default_factory=list)

    def current_strength(self, current_year: int) -> float:
        """Decaying strength; zero after MARTYRDOM_HORIZON years."""
        years_elapsed = current_year - self.year_of_death
        if years_elapsed > MARTYRDOM_HORIZON or years_elapsed < 0:
            return 0.0
        return MARTYRDOM_INITIAL * (MARTYRDOM_DECAY ** years_elapsed)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "belief_snapshot": self.belief_snapshot,
            "year_of_death": self.year_of_death,
            "affected_ids": self.affected_ids,
        }


@dataclass
class Faction:
    """An emergent ideological cluster."""
    name: str
    centroid: dict[str, float]
    member_ids: list[str]
    cohesion: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "centroid": {k: round(v, 4) for k, v in self.centroid.items()},
            "member_ids": self.member_ids,
            "cohesion": round(self.cohesion, 4),
        }


@dataclass
class BeliefTickResult:
    """Result of one year's belief-system tick."""
    snapshots: dict[str, dict] = field(default_factory=dict)
    factions: list[Faction] = field(default_factory=list)
    faction_count: int = 0
    notable_shifts: list[dict] = field(default_factory=list)
    martyrdom_active: int = 0
    colony_polarization: float = 0.0

    def to_dict(self) -> dict:
        return {
            "snapshots": self.snapshots,
            "factions": [f.to_dict() for f in self.factions],
            "faction_count": self.faction_count,
            "notable_shifts": self.notable_shifts,
            "martyrdom_active": self.martyrdom_active,
            "colony_polarization": round(self.colony_polarization, 4),
        }


# -- pure helpers ------------------------------------------------------------

def _clamp_belief(v: float) -> float:
    return max(-1.0, min(1.0, v))


def _cap_delta(delta: float) -> float:
    return max(-BELIEF_CAP_DELTA, min(BELIEF_CAP_DELTA, delta))


# -- initialization ----------------------------------------------------------

def init_beliefs_from_stats(
    resolve: float, empathy: float, faith: float,
    paranoia: float, improvisation: float, hoarding: float,
    rng: random.Random,
) -> BeliefState:
    """Initialize a colonist's beliefs loosely from personality stats.

    The mapping is loose — beliefs and stats are correlated but not
    redundant.  Random noise prevents perfect prediction from stats.
    """
    noise = lambda: rng.gauss(0, 0.15)
    return BeliefState(
        collectivism=_clamp_belief((empathy - hoarding) * 0.4 + noise()),
        authority=_clamp_belief((resolve - improvisation) * 0.3 + noise()),
        spiritualism=_clamp_belief((faith - paranoia) * 0.5 + noise()),
        risk_appetite=_clamp_belief((improvisation - paranoia) * 0.4 + noise()),
    )


def inherit_beliefs(
    parent_a: BeliefState,
    parent_b: BeliefState,
    rng: random.Random,
) -> BeliefState:
    """Create child beliefs by averaging parents + random drift."""
    child = BeliefState()
    for axis in BELIEF_AXES:
        midpoint = (getattr(parent_a, axis) + getattr(parent_b, axis)) / 2.0
        drift = rng.gauss(0, CHILD_DRIFT)
        setattr(child, axis, _clamp_belief(midpoint + drift))
    return child


# -- social propagation (synchronous, mean-preserving) -----------------------

def compute_social_influence(
    colonist_id: str,
    belief_snapshot: dict[str, BeliefState],
    trust_func,
    active_ids: list[str],
) -> dict[str, float]:
    """Compute per-axis belief shift from social influence.

    trust_func(a, b) -> float: returns trust from a toward b.
    Uses frozen snapshot to ensure order-independence.
    """
    my_beliefs = belief_snapshot.get(colonist_id)
    if my_beliefs is None:
        return {a: 0.0 for a in BELIEF_AXES}

    deltas: dict[str, float] = {a: 0.0 for a in BELIEF_AXES}
    total_trust = 0.0

    for other_id in active_ids:
        if other_id == colonist_id:
            continue
        other_beliefs = belief_snapshot.get(other_id)
        if other_beliefs is None:
            continue
        trust = trust_func(colonist_id, other_id)
        if trust <= 0.0:
            continue
        total_trust += trust
        for axis in BELIEF_AXES:
            gap = getattr(other_beliefs, axis) - getattr(my_beliefs, axis)
            deltas[axis] += trust * gap

    if total_trust > 0:
        for axis in BELIEF_AXES:
            deltas[axis] = (deltas[axis] / total_trust) * PROPAGATION_RATE

    return {a: _cap_delta(d) for a, d in deltas.items()}


# -- experience-driven shifts ------------------------------------------------

def compute_experience_shift(
    event_type: str,
    event_severity: float,
    resource_avg: float,
    action: str,
    death_count: int,
    gov_changed: bool,
) -> dict[str, float]:
    """Compute per-axis belief shift from the year's experiences.

    Events push beliefs: crises increase collectivism, deaths increase
    spiritualism, governance changes affect authority beliefs.
    """
    deltas: dict[str, float] = {a: 0.0 for a in BELIEF_AXES}

    # Resource scarcity → collectivism
    if resource_avg < 0.3:
        deltas["collectivism"] += (0.3 - resource_avg) * EXPERIENCE_SCALE * 2
    elif resource_avg > 0.7:
        deltas["collectivism"] -= (resource_avg - 0.7) * EXPERIENCE_SCALE

    # Severe events → caution, spiritualism
    if event_severity > 0.5:
        deltas["risk_appetite"] -= event_severity * EXPERIENCE_SCALE
        deltas["spiritualism"] += event_severity * EXPERIENCE_SCALE * 0.5

    # Deaths nearby → spiritualism increase
    if death_count > 0:
        deltas["spiritualism"] += min(death_count, 3) * EXPERIENCE_SCALE * 0.5

    # Governance change → authority axis shifts toward new system
    if gov_changed:
        deltas["authority"] += EXPERIENCE_SCALE * 0.5

    # Action-based drift
    action_shifts: dict[str, dict[str, float]] = {
        "cooperate": {"collectivism": 0.02},
        "hoard": {"collectivism": -0.02},
        "pray": {"spiritualism": 0.02},
        "research": {"spiritualism": -0.01, "risk_appetite": 0.01},
        "explore": {"risk_appetite": 0.02},
        "rest": {"risk_appetite": -0.01},
        "sabotage": {"authority": -0.02, "collectivism": -0.03},
        "mediate": {"collectivism": 0.01, "authority": 0.01},
    }
    for axis, delta in action_shifts.get(action, {}).items():
        deltas[axis] += delta

    return {a: _cap_delta(d) for a, d in deltas.items()}


# -- martyrdom ---------------------------------------------------------------

def create_martyrdom_effect(
    dead_colonist_id: str,
    belief_map: dict[str, BeliefState],
    connected_ids: list[str],
) -> MartyrdomEffect | None:
    """Create a martyrdom effect when a colonist dies."""
    beliefs = belief_map.get(dead_colonist_id)
    if beliefs is None:
        return None
    return MartyrdomEffect(
        source_id=dead_colonist_id,
        belief_snapshot=beliefs.to_dict(),
        year_of_death=0,  # will be set by caller
        affected_ids=connected_ids,
    )


def apply_martyrdom(
    belief_map: dict[str, BeliefState],
    effects: list[MartyrdomEffect],
    current_year: int,
) -> list[dict]:
    """Apply all active martyrdom effects.  Returns log of shifts."""
    log: list[dict] = []
    for effect in effects:
        strength = effect.current_strength(current_year)
        if strength <= 0:
            continue
        for cid in effect.affected_ids:
            beliefs = belief_map.get(cid)
            if beliefs is None:
                continue
            for axis in BELIEF_AXES:
                martyr_val = effect.belief_snapshot.get(axis, 0.0)
                current_val = getattr(beliefs, axis)
                shift = (martyr_val - current_val) * strength
                shift = _cap_delta(shift)
                setattr(beliefs, axis, _clamp_belief(current_val + shift))
            log.append({
                "colonist_id": cid, "martyr_id": effect.source_id,
                "strength": round(strength, 4),
            })
    return log


# -- faction detection -------------------------------------------------------

def detect_factions(
    belief_map: dict[str, BeliefState],
    active_ids: list[str],
) -> list[Faction]:
    """Detect ideological factions via thresholded single-linkage clustering.

    Simple and stable: two colonists are in the same faction if their
    belief distance < FACTION_DISTANCE_THRESHOLD.  Factions with fewer
    than MIN_FACTION_SIZE members are dissolved.
    """
    if len(active_ids) < MIN_FACTION_SIZE:
        return []

    # Build adjacency via distance threshold
    adjacency: dict[str, set[str]] = {cid: set() for cid in active_ids}
    for i, a in enumerate(active_ids):
        ba = belief_map.get(a)
        if ba is None:
            continue
        for b in active_ids[i + 1:]:
            bb = belief_map.get(b)
            if bb is None:
                continue
            if ba.distance(bb) < FACTION_DISTANCE_THRESHOLD:
                adjacency[a].add(b)
                adjacency[b].add(a)

    # Connected components (BFS)
    visited: set[str] = set()
    clusters: list[list[str]] = []
    for cid in active_ids:
        if cid in visited:
            continue
        cluster: list[str] = []
        queue = [cid]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(cluster) >= MIN_FACTION_SIZE:
            clusters.append(cluster)

    # Name and compute centroids
    faction_names = [
        "Collective", "Vanguard", "Seekers", "Pragmatists",
        "Frontier", "Shepherds", "Architects", "Wildlings",
    ]
    factions: list[Faction] = []
    for i, members in enumerate(clusters):
        name = faction_names[i % len(faction_names)]
        centroid: dict[str, float] = {}
        for axis in BELIEF_AXES:
            vals = [getattr(belief_map[m], axis) for m in members
                    if m in belief_map]
            centroid[axis] = sum(vals) / len(vals) if vals else 0.0
        # Cohesion = 1 - mean internal distance
        dists: list[float] = []
        for j, a in enumerate(members):
            ba = belief_map.get(a)
            if ba is None:
                continue
            for b in members[j + 1:]:
                bb = belief_map.get(b)
                if bb is None:
                    continue
                dists.append(ba.distance(bb))
        cohesion = 1.0 - (sum(dists) / len(dists)) if dists else 1.0
        factions.append(Faction(name=name, centroid=centroid,
                                member_ids=members, cohesion=cohesion))
    return factions


# -- colony-level metrics ----------------------------------------------------

def compute_polarization(
    belief_map: dict[str, BeliefState],
    active_ids: list[str],
) -> float:
    """Colony polarization: mean pairwise distance, normalized [0, 1]."""
    if len(active_ids) < 2:
        return 0.0
    total_dist = 0.0
    count = 0
    for i, a in enumerate(active_ids):
        ba = belief_map.get(a)
        if ba is None:
            continue
        for b in active_ids[i + 1:]:
            bb = belief_map.get(b)
            if bb is None:
                continue
            total_dist += ba.distance(bb)
            count += 1
    return total_dist / count if count > 0 else 0.0


# -- action weight modifiers -------------------------------------------------

def compute_belief_action_weights(beliefs: BeliefState) -> dict[str, float]:
    """Compute action-weight modifiers from colonist beliefs.

    Returns a dict of action -> weight delta (positive = more likely).
    Total contribution is bounded by ACTION_BELIEF_WEIGHT.
    """
    w: dict[str, float] = {}

    # Collectivism axis
    w["cooperate"] = beliefs.collectivism * 0.15
    w["hoard"] = -beliefs.collectivism * 0.12
    w["mediate"] = beliefs.collectivism * 0.08

    # Authority axis
    w["sabotage"] = -beliefs.authority * 0.10
    w["code"] = beliefs.authority * 0.05

    # Spiritualism axis
    w["pray"] = beliefs.spiritualism * 0.15
    w["research"] = -beliefs.spiritualism * 0.10

    # Risk appetite axis
    w["explore"] = beliefs.risk_appetite * 0.12
    w["terraform"] = beliefs.risk_appetite * 0.08
    w["rest"] = -beliefs.risk_appetite * 0.10

    # Clamp total contribution
    total_abs = sum(abs(v) for v in w.values())
    if total_abs > ACTION_BELIEF_WEIGHT:
        scale = ACTION_BELIEF_WEIGHT / total_abs
        w = {k: v * scale for k, v in w.items()}

    return w


def compute_governance_vote_bias(
    beliefs: BeliefState,
    proposal_gov_type: str,
) -> float:
    """Compute vote bias from beliefs toward a governance proposal.

    Returns a float in [-GOV_BELIEF_BIAS, +GOV_BELIEF_BIAS] to be
    added to the colonist's vote score.
    """
    bias = 0.0
    if proposal_gov_type == "council":
        bias += beliefs.collectivism * 0.08 + beliefs.authority * 0.05
    elif proposal_gov_type == "dictator":
        bias += beliefs.authority * 0.12 - beliefs.collectivism * 0.05
    elif proposal_gov_type == "lottery":
        bias -= beliefs.authority * 0.08
        bias += beliefs.risk_appetite * 0.06
    elif proposal_gov_type == "consensus":
        bias += beliefs.collectivism * 0.10 - beliefs.authority * 0.06
    elif proposal_gov_type == "ai_governor":
        bias -= beliefs.spiritualism * 0.08
        bias += beliefs.authority * 0.05
    elif proposal_gov_type == "anarchy":
        bias -= beliefs.authority * 0.12
        bias += beliefs.risk_appetite * 0.05

    return max(-GOV_BELIEF_BIAS, min(GOV_BELIEF_BIAS, bias))


# -- main tick ---------------------------------------------------------------

@dataclass
class BeliefYearContext:
    """Year context needed for belief updates."""
    colonist_id: str
    action: str
    event_type: str
    event_severity: float
    resource_avg: float
    death_count: int
    gov_changed: bool


def tick_beliefs(
    belief_map: dict[str, BeliefState],
    contexts: list[BeliefYearContext],
    trust_func,
    active_ids: list[str],
    year: int,
    martyrdom_effects: list[MartyrdomEffect],
    rng: random.Random,
) -> BeliefTickResult:
    """Run one year of belief updates.  Mutates belief_map in place.

    Updates are synchronous: all shifts are computed from a frozen snapshot
    of the current belief state, then applied in one batch.
    """
    # Ensure all active colonists have a belief state
    for ctx in contexts:
        if ctx.colonist_id not in belief_map:
            belief_map[ctx.colonist_id] = BeliefState()

    # Freeze snapshot for synchronous updates
    snapshot: dict[str, BeliefState] = {
        cid: beliefs.copy() for cid, beliefs in belief_map.items()
        if cid in active_ids
    }

    # Compute all deltas from frozen snapshot
    all_deltas: dict[str, dict[str, float]] = {}
    notable_shifts: list[dict] = []

    for ctx in contexts:
        cid = ctx.colonist_id
        # Social influence (from frozen snapshot)
        social_deltas = compute_social_influence(
            cid, snapshot, trust_func, active_ids)
        # Experience-driven shifts
        experience_deltas = compute_experience_shift(
            ctx.event_type, ctx.event_severity, ctx.resource_avg,
            ctx.action, ctx.death_count, ctx.gov_changed)
        # Combine
        combined: dict[str, float] = {}
        for axis in BELIEF_AXES:
            raw = social_deltas.get(axis, 0.0) + experience_deltas.get(axis, 0.0)
            combined[axis] = _cap_delta(raw)
        all_deltas[cid] = combined

        # Track notable shifts (magnitude > 0.05 on any axis)
        max_shift = max(abs(v) for v in combined.values())
        if max_shift > 0.05:
            notable_shifts.append({
                "colonist_id": cid, "year": year,
                "axis": max(combined, key=lambda a: abs(combined[a])),
                "magnitude": round(max_shift, 4),
            })

    # Apply all deltas in one batch
    for cid, deltas in all_deltas.items():
        beliefs = belief_map.get(cid)
        if beliefs is None:
            continue
        for axis in BELIEF_AXES:
            current = getattr(beliefs, axis)
            setattr(beliefs, axis, _clamp_belief(current + deltas.get(axis, 0.0)))

    # Apply martyrdom effects
    martyrdom_log = apply_martyrdom(belief_map, martyrdom_effects, year)
    active_martyrdoms = sum(
        1 for e in martyrdom_effects if e.current_strength(year) > 0)

    # Prune expired martyrdom effects
    martyrdom_effects[:] = [
        e for e in martyrdom_effects
        if e.current_strength(year) > 0
    ]

    # Detect factions
    factions = detect_factions(belief_map, active_ids)

    # Colony-level metrics
    polarization = compute_polarization(belief_map, active_ids)

    # Build snapshots
    snapshots = {cid: belief_map[cid].to_dict()
                 for cid in active_ids if cid in belief_map}

    return BeliefTickResult(
        snapshots=snapshots,
        factions=factions,
        faction_count=len(factions),
        notable_shifts=notable_shifts,
        martyrdom_active=active_martyrdoms,
        colony_polarization=polarization,
    )
