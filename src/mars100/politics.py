"""
Mars-100 politics organ — factions, grievances, crisis-driven governance.

Translates social structure, economic inequality, and shared trauma into
emergent political factions.  Factions form alliances, accumulate grievances,
and drive governance proposals.  Deep sub-sim insights that recur across
factions get promoted to constitutional amendments.

Engine v10.0.  Pure computation — deterministic given RNG stream.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Faction detection
MIN_FACTION_SIZE: int = 2
AFFINITY_THRESHOLD: float = 0.55
MAX_FACTIONS: int = 5

# Grievance system
GRIEVANCE_DECAY: float = 0.15
GRIEVANCE_CAP: float = 5.0
REVOLT_THRESHOLD: float = 3.5

# Alliance
ALLIANCE_THRESHOLD: float = 0.4
ALLIANCE_DECAY: float = 0.05

# Crisis proposal
CRISIS_PROPOSAL_THRESHOLD: float = 2.0

# Amendment promotion
AMENDMENT_THRESHOLD: int = 3  # recurring insights needed


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Faction:
    """A cluster of ideologically aligned colonists."""
    id: str
    member_ids: list[str]
    ideology: dict[str, float]  # stat name → avg value
    formed_year: int
    cohesion: float = 0.0
    grievance: float = 0.0
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "member_ids": list(self.member_ids),
            "ideology": {k: round(v, 4) for k, v in self.ideology.items()},
            "formed_year": self.formed_year,
            "cohesion": round(self.cohesion, 4),
            "grievance": round(self.grievance, 4),
            "name": self.name,
        }


@dataclass
class Alliance:
    """A coalition between two factions."""
    faction_a: str
    faction_b: str
    strength: float = 0.5
    formed_year: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "factions": [self.faction_a, self.faction_b],
            "strength": round(self.strength, 4),
            "formed_year": self.formed_year,
        }


@dataclass
class Grievance:
    """A recorded political grievance."""
    source: str  # faction id or "colony"
    cause: str
    intensity: float
    year: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "cause": self.cause,
            "intensity": round(self.intensity, 4), "year": self.year,
        }


@dataclass
class PoliticalState:
    """Current political landscape of the colony."""
    factions: list[Faction] = field(default_factory=list)
    alliances: list[Alliance] = field(default_factory=list)
    grievances: list[Grievance] = field(default_factory=list)
    amendments: list[dict] = field(default_factory=list)
    faction_history: list[dict] = field(default_factory=list)
    revolt_cooldown: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "factions": [f.to_dict() for f in self.factions],
            "alliances": [a.to_dict() for a in self.alliances],
            "grievances": [g.to_dict() for g in self.grievances],
            "amendments": list(self.amendments),
            "faction_history": list(self.faction_history),
            "revolt_cooldown": self.revolt_cooldown,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PoliticalState:
        factions = [
            Faction(id=f["id"], member_ids=f["member_ids"],
                    ideology=f["ideology"], formed_year=f["formed_year"],
                    cohesion=f.get("cohesion", 0.0),
                    grievance=f.get("grievance", 0.0),
                    name=f.get("name", ""))
            for f in d.get("factions", [])
        ]
        alliances = [
            Alliance(faction_a=a["factions"][0], faction_b=a["factions"][1],
                     strength=a.get("strength", 0.5),
                     formed_year=a.get("formed_year", 0))
            for a in d.get("alliances", [])
        ]
        grievances = [
            Grievance(source=g["source"], cause=g["cause"],
                      intensity=g["intensity"], year=g["year"])
            for g in d.get("grievances", [])
        ]
        return cls(
            factions=factions, alliances=alliances, grievances=grievances,
            amendments=d.get("amendments", []),
            faction_history=d.get("faction_history", []),
            revolt_cooldown=d.get("revolt_cooldown", 0),
        )


@dataclass
class PoliticalTickResult:
    """Result of one year's politics processing."""
    factions_formed: int = 0
    factions_dissolved: int = 0
    alliances_formed: int = 0
    alliances_broken: int = 0
    grievances_added: int = 0
    crisis_proposal: bool = False
    amendment_proposed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "factions_formed": self.factions_formed,
            "factions_dissolved": self.factions_dissolved,
            "alliances_formed": self.alliances_formed,
            "alliances_broken": self.alliances_broken,
            "grievances_added": self.grievances_added,
            "crisis_proposal": self.crisis_proposal,
            "amendment_proposed": self.amendment_proposed,
        }


# ---------------------------------------------------------------------------
# Faction names
# ---------------------------------------------------------------------------

FACTION_NAMES = [
    "Iron Covenant", "Dust Commune", "Flame Circle", "Water Pact",
    "Sky Assembly", "Stone Guild", "Wind Council", "Ember Syndicate",
    "Root Collective", "Star Chamber", "Sand Parliament", "Deep Accord",
]


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def _stat_vector(colonist: Any) -> dict[str, float]:
    """Extract a colonist's stat vector for affinity computation."""
    from src.mars100.colonist import STAT_NAMES
    return {name: getattr(colonist.stats, name) for name in STAT_NAMES}


def compute_affinity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute ideological affinity between two stat vectors.

    Returns 0.0 (opposite) to 1.0 (identical).  Uses cosine similarity
    on the stat vectors.
    """
    keys = sorted(set(vec_a) & set(vec_b))
    if not keys:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in keys)
    mag_a = math.sqrt(sum(vec_a[k] ** 2 for k in keys))
    mag_b = math.sqrt(sum(vec_b[k] ** 2 for k in keys))
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


def detect_factions(
    colonists: list[Any],
    social_graph: Any,
    year: int,
    rng: random.Random,
    existing: list[Faction] | None = None,
) -> list[Faction]:
    """Detect emergent factions from value alignment and social trust.

    Uses greedy clustering: pick highest-affinity unassigned pair, grow
    the cluster by adding colonists with affinity > threshold to any member.
    """
    active = [c for c in colonists if c.is_active()]
    if len(active) < MIN_FACTION_SIZE * 2:
        return existing or []

    vectors = {c.id: _stat_vector(c) for c in active}
    assigned: set[str] = set()
    clusters: list[list[str]] = []

    # Greedy clustering
    pairs: list[tuple[str, str, float]] = []
    ids = list(vectors.keys())
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            stat_aff = compute_affinity(vectors[a_id], vectors[b_id])
            trust = social_graph.get(a_id, b_id).trust
            combined = stat_aff * 0.6 + trust * 0.4
            pairs.append((a_id, b_id, combined))
    pairs.sort(key=lambda x: x[2], reverse=True)

    for a_id, b_id, score in pairs:
        if score < AFFINITY_THRESHOLD:
            break
        if a_id in assigned and b_id in assigned:
            continue
        if len(clusters) >= MAX_FACTIONS:
            break

        # Find or create cluster
        target_cluster: list[str] | None = None
        for cluster in clusters:
            if a_id in cluster or b_id in cluster:
                target_cluster = cluster
                break

        if target_cluster is None:
            if a_id not in assigned and b_id not in assigned:
                clusters.append([a_id, b_id])
                assigned.add(a_id)
                assigned.add(b_id)
        else:
            for cid in (a_id, b_id):
                if cid not in assigned:
                    target_cluster.append(cid)
                    assigned.add(cid)

    # Build faction objects
    existing_map = {frozenset(f.member_ids): f for f in (existing or [])}
    name_idx = 0
    factions: list[Faction] = []
    for cluster in clusters:
        if len(cluster) < MIN_FACTION_SIZE:
            continue
        member_set = frozenset(cluster)
        # Reuse existing faction if membership overlaps significantly
        reused = None
        for old_set, old_f in existing_map.items():
            overlap = len(member_set & old_set) / max(1, len(member_set | old_set))
            if overlap > 0.5:
                reused = old_f
                break

        ideology = {}
        from src.mars100.colonist import STAT_NAMES
        for stat in STAT_NAMES:
            vals = [vectors[cid][stat] for cid in cluster if cid in vectors]
            ideology[stat] = sum(vals) / max(1, len(vals))

        # Cohesion: average internal trust
        trust_sum = 0.0
        trust_count = 0
        for a in cluster:
            for b in cluster:
                if a != b:
                    trust_sum += social_graph.get(a, b).trust
                    trust_count += 1
        cohesion = trust_sum / max(1, trust_count)

        if reused:
            reused.member_ids = list(cluster)
            reused.ideology = ideology
            reused.cohesion = cohesion
            factions.append(reused)
        else:
            name = FACTION_NAMES[name_idx % len(FACTION_NAMES)]
            name_idx += 1
            factions.append(Faction(
                id=f"faction-y{year}-{len(factions)}",
                member_ids=list(cluster),
                ideology=ideology,
                formed_year=year,
                cohesion=cohesion,
                name=name,
            ))

    return factions


def form_alliances(
    factions: list[Faction],
    year: int,
    existing: list[Alliance] | None = None,
) -> list[Alliance]:
    """Form or maintain alliances between factions with compatible ideologies."""
    if len(factions) < 2:
        return []

    existing_pairs: dict[tuple[str, str], Alliance] = {}
    for a in (existing or []):
        key = tuple(sorted([a.faction_a, a.faction_b]))
        existing_pairs[key] = a

    alliances: list[Alliance] = []
    faction_ids = [f.id for f in factions]
    ideology_map = {f.id: f.ideology for f in factions}

    for i, fa_id in enumerate(faction_ids):
        for fb_id in faction_ids[i + 1:]:
            affinity = compute_affinity(ideology_map[fa_id], ideology_map[fb_id])
            key = tuple(sorted([fa_id, fb_id]))

            if key in existing_pairs:
                old = existing_pairs[key]
                new_strength = old.strength * (1.0 - ALLIANCE_DECAY)
                if affinity > ALLIANCE_THRESHOLD:
                    new_strength += 0.1
                new_strength = max(0.0, min(1.0, new_strength))
                if new_strength > 0.1:
                    old.strength = new_strength
                    alliances.append(old)
            elif affinity > ALLIANCE_THRESHOLD:
                alliances.append(Alliance(
                    faction_a=fa_id, faction_b=fb_id,
                    strength=affinity * 0.5, formed_year=year,
                ))

    return alliances


def accumulate_grievances(
    factions: list[Faction],
    resource_avg: float,
    gini: float,
    recent_deaths: int,
    gov_type: str,
    year: int,
) -> list[Grievance]:
    """Generate new grievances from colony conditions."""
    new_grievances: list[Grievance] = []

    # Colony-wide: resource scarcity
    if resource_avg < 0.3:
        intensity = (0.3 - resource_avg) * 3.0
        new_grievances.append(Grievance(
            source="colony", cause="resource_scarcity",
            intensity=min(GRIEVANCE_CAP, intensity), year=year,
        ))

    # Colony-wide: inequality
    if gini > 0.4:
        intensity = (gini - 0.4) * 4.0
        new_grievances.append(Grievance(
            source="colony", cause="inequality",
            intensity=min(GRIEVANCE_CAP, intensity), year=year,
        ))

    # Colony-wide: deaths
    if recent_deaths > 0:
        new_grievances.append(Grievance(
            source="colony", cause="deaths",
            intensity=min(GRIEVANCE_CAP, recent_deaths * 0.8), year=year,
        ))

    # Per-faction: governance mismatch
    for faction in factions:
        dominant_stat = max(faction.ideology, key=faction.ideology.get)
        # High-empathy factions dislike dictators
        if dominant_stat == "empathy" and gov_type == "dictator":
            new_grievances.append(Grievance(
                source=faction.id, cause="governance_mismatch",
                intensity=1.2, year=year,
            ))
        # High-paranoia factions distrust consensus
        if dominant_stat == "paranoia" and gov_type == "consensus":
            new_grievances.append(Grievance(
                source=faction.id, cause="governance_mismatch",
                intensity=0.8, year=year,
            ))
        # High-faith factions dislike AI governor
        if dominant_stat == "faith" and gov_type == "ai_governor":
            new_grievances.append(Grievance(
                source=faction.id, cause="governance_mismatch",
                intensity=1.0, year=year,
            ))

    return new_grievances


def decay_grievances(state: PoliticalState) -> None:
    """Decay and prune old grievances."""
    surviving: list[Grievance] = []
    for g in state.grievances:
        g.intensity *= (1.0 - GRIEVANCE_DECAY)
        if g.intensity > 0.05:
            surviving.append(g)
    state.grievances = surviving


def total_grievance(state: PoliticalState) -> float:
    """Sum of all active grievance intensities."""
    return sum(g.intensity for g in state.grievances)


def should_crisis_propose(state: PoliticalState) -> bool:
    """Check if grievance pressure warrants a crisis governance proposal."""
    if state.revolt_cooldown > 0:
        return False
    return total_grievance(state) > CRISIS_PROPOSAL_THRESHOLD


def compute_faction_pressure(
    factions: list[Faction],
    colonist_id: str,
) -> dict[str, float]:
    """Compute action-weight pressure from faction membership.

    Returns additive deltas for action weights.
    """
    deltas: dict[str, float] = {}
    for faction in factions:
        if colonist_id not in faction.member_ids:
            continue
        dominant = max(faction.ideology, key=faction.ideology.get)
        # Faction ideology biases action selection
        if dominant == "empathy":
            deltas["mediate"] = deltas.get("mediate", 0.0) + 0.15
            deltas["cooperate"] = deltas.get("cooperate", 0.0) + 0.1
        elif dominant == "resolve":
            deltas["terraform"] = deltas.get("terraform", 0.0) + 0.15
            deltas["research"] = deltas.get("research", 0.0) + 0.1
        elif dominant == "paranoia":
            deltas["hoard"] = deltas.get("hoard", 0.0) + 0.15
            deltas["sabotage"] = deltas.get("sabotage", 0.0) + 0.1
        elif dominant == "faith":
            deltas["pray"] = deltas.get("pray", 0.0) + 0.15
            deltas["mediate"] = deltas.get("mediate", 0.0) + 0.1
        elif dominant == "improvisation":
            deltas["explore"] = deltas.get("explore", 0.0) + 0.15
            deltas["code"] = deltas.get("code", 0.0) + 0.1
        elif dominant == "hoarding":
            deltas["hoard"] = deltas.get("hoard", 0.0) + 0.2
            deltas["farm"] = deltas.get("farm", 0.0) + 0.1

        # Grievance increases sabotage/hoard tendency
        if faction.grievance > 1.5:
            scale = min(0.3, (faction.grievance - 1.5) * 0.15)
            deltas["sabotage"] = deltas.get("sabotage", 0.0) + scale
            deltas["hoard"] = deltas.get("hoard", 0.0) + scale * 0.5

    return deltas


def compute_voting_bloc(
    factions: list[Faction],
    alliances: list[Alliance],
    voter_id: str,
    proposer_id: str,
) -> float:
    """Compute voting bias from faction/alliance membership.

    Returns a float bias: positive = more likely to vote yes,
    negative = more likely to vote no.
    """
    voter_faction: Faction | None = None
    proposer_faction: Faction | None = None
    for f in factions:
        if voter_id in f.member_ids:
            voter_faction = f
        if proposer_id in f.member_ids:
            proposer_faction = f

    if voter_faction is None:
        return 0.0

    # Same faction: strong positive bias
    if proposer_faction and voter_faction.id == proposer_faction.id:
        return 0.3 * voter_faction.cohesion

    # Allied factions: mild positive bias
    if proposer_faction:
        for alliance in alliances:
            pair = {alliance.faction_a, alliance.faction_b}
            if voter_faction.id in pair and proposer_faction.id in pair:
                return 0.15 * alliance.strength

    # Unaffiliated proposer: no bias
    if proposer_faction is None:
        return 0.0

    # Different non-allied faction: mild negative
    return -0.1


def check_amendment_promotion(
    state: PoliticalState,
    insight_queue: list[dict],
    year: int,
) -> dict | None:
    """Check if recurring sub-sim insights should be promoted to amendment.

    Looks for themes (result patterns) that appear across multiple factions
    or colonists. If a theme recurs enough, propose an amendment.
    """
    if len(insight_queue) < AMENDMENT_THRESHOLD:
        return None

    # Group insights by result theme
    themes: dict[str, list[dict]] = {}
    for ins in insight_queue:
        key = str(ins.get("result", ""))[:50]
        if not key:
            continue
        themes.setdefault(key, []).append(ins)

    for theme_key, instances in themes.items():
        if len(instances) < AMENDMENT_THRESHOLD:
            continue

        # Check if insight spans multiple colonists (not just one obsessed thinker)
        colonist_set = {i.get("colonist_id", "") for i in instances}
        if len(colonist_set) < 2:
            continue

        years = [i.get("year", year) for i in instances]
        depths = [i.get("depth", 1) for i in instances]

        amendment = {
            "theme": theme_key,
            "first_seen_year": min(years),
            "last_seen_year": max(years),
            "occurrences": len(instances),
            "max_depth": max(depths),
            "colonists_involved": list(colonist_set),
            "proposed_year": year,
            "proposed_text": _draft_amendment_text(theme_key, instances),
            "status": "proposed",
        }

        # Remove consumed insights
        for ins in instances:
            if ins in insight_queue:
                insight_queue.remove(ins)

        return amendment

    return None


def _draft_amendment_text(theme: str, instances: list[dict]) -> str:
    """Draft amendment text from a recurring insight theme."""
    colonists = list({i.get("colonist_id", "unknown") for i in instances})
    max_depth = max(i.get("depth", 1) for i in instances)
    count = len(instances)

    if "trust" in theme.lower() or "empathy" in theme.lower():
        return (f"Amendment: Governance decisions shall weight trust-based consensus, "
                f"as validated by {count} sub-simulations reaching depth {max_depth}.")
    if "survival" in theme.lower() or "resource" in theme.lower():
        return (f"Amendment: Resource allocation shall follow collective need over individual "
                f"hoarding, confirmed by {count} recursive models.")
    if "faith" in theme.lower() or "purpose" in theme.lower():
        return (f"Amendment: Spiritual and purpose-driven governance input shall be formally "
                f"recognized, per {count} sub-sim validations.")
    return (f"Amendment: Colony governance shall incorporate the principle '{theme[:80]}', "
            f"validated by {count} sub-simulations across {len(colonists)} colonists.")


def tick_politics(
    state: PoliticalState,
    colonists: list[Any],
    social_graph: Any,
    resources_avg: float,
    gini: float,
    recent_deaths: int,
    gov_type: str,
    year: int,
    rng: random.Random,
) -> PoliticalTickResult:
    """Advance the political landscape by one year.

    Detects factions, manages alliances, accumulates grievances,
    checks for crisis proposals.
    """
    result = PoliticalTickResult()

    # Decay cooldown
    if state.revolt_cooldown > 0:
        state.revolt_cooldown -= 1

    # Detect factions (re-cluster every 5 years or if no factions exist)
    if not state.factions or year % 5 == 0:
        old_count = len(state.factions)
        state.factions = detect_factions(
            colonists, social_graph, year, rng, existing=state.factions)
        new_count = len(state.factions)
        result.factions_formed = max(0, new_count - old_count)
        result.factions_dissolved = max(0, old_count - new_count)

        if result.factions_formed > 0 or result.factions_dissolved > 0:
            state.faction_history.append({
                "year": year,
                "factions": [f.to_dict() for f in state.factions],
            })

    # Remove dead/exiled members from factions
    active_ids = {c.id for c in colonists if c.is_active()}
    for faction in state.factions:
        faction.member_ids = [m for m in faction.member_ids if m in active_ids]
    state.factions = [f for f in state.factions if len(f.member_ids) >= MIN_FACTION_SIZE]

    # Form/maintain alliances
    old_alliances = len(state.alliances)
    state.alliances = form_alliances(state.factions, year, existing=state.alliances)
    new_alliances = len(state.alliances)
    result.alliances_formed = max(0, new_alliances - old_alliances)
    result.alliances_broken = max(0, old_alliances - new_alliances)

    # Accumulate grievances
    new_grievances = accumulate_grievances(
        state.factions, resources_avg, gini, recent_deaths, gov_type, year)
    state.grievances.extend(new_grievances)
    result.grievances_added = len(new_grievances)

    # Distribute colony grievances to factions
    colony_grief = sum(g.intensity for g in state.grievances if g.source == "colony")
    for faction in state.factions:
        faction_grief = sum(
            g.intensity for g in state.grievances if g.source == faction.id)
        faction.grievance = min(GRIEVANCE_CAP, faction_grief + colony_grief * 0.3)

    # Decay old grievances
    decay_grievances(state)

    # Crisis proposal check
    if should_crisis_propose(state):
        result.crisis_proposal = True
        state.revolt_cooldown = 5

    return result


def run_simulation(years: int = 100, seed: int = 42) -> dict:
    """Backward-compatible entry point for the simulation."""
    from src.mars100.engine import Mars100Engine
    engine = Mars100Engine(seed=seed, total_years=years)
    return engine.run()
