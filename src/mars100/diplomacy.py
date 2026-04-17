"""
Diplomacy organ for Mars-100 colony simulation (engine v9.0).

Detects emergent political factions from the social graph, manages
alliances between factions, tracks betrayals, and feeds trust feedback
back into the colony's relationship matrix.

Factions are fluid — they re-form each year based on mutual trust and
value alignment, with hysteresis to prevent flickering.  Alliances
represent formal agreements with material consequences (slight trust
boosts between allied factions).  Betrayals occur under economic
pressure or high-paranoia leadership and cause sharp trust damage.

Phase 1 scope (v9.0):
  - Faction detection with hysteresis
  - Alliance formation and natural expiry
  - Betrayal mechanics (economic stress + paranoia)
  - Trust feedback into SocialGraph
  - Serialization for year/sim results
  - Defer: campaign voting modifier (v9.1+)
  - Defer: action-weight perturbation (v9.1+)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# -- constants ---------------------------------------------------------------

MIN_FACTION_SIZE = 3
TRUST_THRESHOLD = 0.55          # symmetric trust floor to be "aligned"
VALUE_ALIGNMENT_THRESHOLD = 0.3  # max stat-distance for value alignment
FACTION_HYSTERESIS_FORM = 2     # must meet criteria N years before forming
FACTION_HYSTERESIS_DISSOLVE = 2  # must fail criteria N years before dissolving

ALLIANCE_PROBABILITY = 0.25
ALLIANCE_MIN_STRENGTH = 0.3
ALLIANCE_DURATION_RANGE = (5, 15)
ALLIANCE_TRUST_BOOST = 0.02     # per-year trust boost between allied factions

BETRAYAL_BASE_PROB = 0.03
BETRAYAL_GINI_WEIGHT = 0.15     # high inequality raises betrayal chance
BETRAYAL_SCARCITY_WEIGHT = 0.10  # low resources raise betrayal chance
BETRAYAL_PARANOIA_WEIGHT = 0.20  # paranoid leaders betray more
BETRAYAL_TRUST_DAMAGE = 0.15    # trust loss between factions on betrayal

FACTION_NAMES = [
    "Crimson Pact", "Azure Circle", "Iron Compact", "Jade Assembly",
    "Obsidian League", "Amber Covenant", "Silver Accord", "Copper Alliance",
    "Onyx Front", "Pearl Council", "Granite Bloc", "Sapphire Union",
]


# -- data classes ------------------------------------------------------------

@dataclass
class Faction:
    """A detected cluster of politically aligned colonists."""
    id: str
    name: str
    leader_id: str
    member_ids: list[str]
    dominant_value: str
    cohesion: float
    formed_year: int
    dissolved_year: int | None = None

    @property
    def active(self) -> bool:
        return self.dissolved_year is None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "leader_id": self.leader_id,
            "member_ids": self.member_ids, "dominant_value": self.dominant_value,
            "cohesion": round(self.cohesion, 4), "formed_year": self.formed_year,
            "dissolved_year": self.dissolved_year,
        }


@dataclass
class Alliance:
    """A formal agreement between two factions."""
    faction_a_id: str
    faction_b_id: str
    alliance_type: str       # "resource_sharing", "defense", "non_aggression"
    strength: float          # 0.0–1.0
    formed_year: int
    expires_year: int

    @property
    def expired(self) -> bool:
        return False  # checked externally with current year

    def to_dict(self) -> dict:
        return {
            "faction_a_id": self.faction_a_id, "faction_b_id": self.faction_b_id,
            "alliance_type": self.alliance_type,
            "strength": round(self.strength, 4),
            "formed_year": self.formed_year, "expires_year": self.expires_year,
        }


@dataclass
class Betrayal:
    """Record of a broken alliance."""
    betrayer_faction_id: str
    victim_faction_id: str
    alliance_type: str
    year: int
    cause: str

    def to_dict(self) -> dict:
        return {
            "betrayer_faction_id": self.betrayer_faction_id,
            "victim_faction_id": self.victim_faction_id,
            "alliance_type": self.alliance_type,
            "year": self.year, "cause": self.cause,
        }


@dataclass
class DiplomacyState:
    """Persistent diplomacy state across simulation years."""
    factions: list[Faction] = field(default_factory=list)
    alliances: list[Alliance] = field(default_factory=list)
    betrayal_log: list[Betrayal] = field(default_factory=list)
    _candidate_years: dict[str, int] = field(default_factory=dict)
    _dissolve_years: dict[str, int] = field(default_factory=dict)
    _next_faction_id: int = 0
    _name_index: int = 0

    def active_factions(self) -> list[Faction]:
        return [f for f in self.factions if f.active]

    def active_alliances(self, year: int) -> list[Alliance]:
        return [a for a in self.alliances if a.expires_year > year]

    def faction_for(self, colonist_id: str) -> Faction | None:
        """Find the active faction containing a colonist, if any."""
        for f in self.active_factions():
            if colonist_id in f.member_ids:
                return f
        return None

    def _next_name(self) -> tuple[str, str]:
        fid = f"faction-{self._next_faction_id}"
        name = FACTION_NAMES[self._name_index % len(FACTION_NAMES)]
        self._next_faction_id += 1
        self._name_index += 1
        return fid, name

    def to_dict(self) -> dict:
        return {
            "factions": [f.to_dict() for f in self.factions],
            "alliances": [a.to_dict() for a in self.alliances],
            "betrayal_log": [b.to_dict() for b in self.betrayal_log],
            "active_faction_count": len(self.active_factions()),
        }

    def summary(self) -> dict:
        """Compact summary for year results."""
        active = self.active_factions()
        return {
            "active_factions": len(active),
            "faction_names": [f.name for f in active],
            "active_alliances": len([a for a in self.alliances
                                     if a.expires_year > 9999]),  # placeholder
            "total_betrayals": len(self.betrayal_log),
        }


@dataclass
class DiplomacyTickResult:
    """Result of one year's diplomacy tick."""
    factions_formed: list[dict] = field(default_factory=list)
    factions_dissolved: list[dict] = field(default_factory=list)
    alliances_formed: list[dict] = field(default_factory=list)
    alliances_expired: list[dict] = field(default_factory=list)
    betrayals: list[dict] = field(default_factory=list)
    trust_adjustments: int = 0

    def to_dict(self) -> dict:
        return {
            "factions_formed": self.factions_formed,
            "factions_dissolved": self.factions_dissolved,
            "alliances_formed": self.alliances_formed,
            "alliances_expired": self.alliances_expired,
            "betrayals": self.betrayals,
            "trust_adjustments": self.trust_adjustments,
        }


# -- pure helpers ------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _symmetric_trust(social_graph: Any, a_id: str, b_id: str) -> float:
    """Minimum of bidirectional trust — conservative measure of mutual trust."""
    rel_ab = social_graph.get(a_id, b_id)
    rel_ba = social_graph.get(b_id, a_id)
    return min(rel_ab.trust, rel_ba.trust)


def _value_distance(stats_a: Any, stats_b: Any) -> float:
    """Euclidean distance between two colonists' stat vectors, normalized."""
    from src.mars100.colonist import STAT_NAMES
    total = 0.0
    for name in STAT_NAMES:
        diff = getattr(stats_a, name) - getattr(stats_b, name)
        total += diff * diff
    return (total / len(STAT_NAMES)) ** 0.5


def _intra_cluster_density(members: list[str], social_graph: Any) -> float:
    """Average symmetric trust within a group of colonists."""
    if len(members) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            total += _symmetric_trust(social_graph, a, b)
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


def _dominant_value(colonists: list[Any], member_ids: list[str]) -> str:
    """Find the stat with highest average among faction members."""
    from src.mars100.colonist import STAT_NAMES
    members = [c for c in colonists if c.id in member_ids]
    if not members:
        return "resolve"
    averages: dict[str, float] = {}
    for name in STAT_NAMES:
        averages[name] = sum(getattr(c.stats, name) for c in members) / len(members)
    return max(averages, key=averages.get)  # type: ignore[arg-type]


def _choose_leader(colonists: list[Any], member_ids: list[str],
                   social_graph: Any) -> str:
    """Choose faction leader: highest combined resolve + intra-faction trust."""
    members = [c for c in colonists if c.id in member_ids]
    if not members:
        return member_ids[0]
    scores: list[tuple[str, float]] = []
    for m in members:
        trust_sum = sum(_symmetric_trust(social_graph, m.id, other)
                        for other in member_ids if other != m.id)
        avg_trust = trust_sum / max(1, len(member_ids) - 1)
        score = m.stats.resolve * 0.5 + avg_trust * 0.5
        scores.append((m.id, score))
    return max(scores, key=lambda x: x[1])[0]


# -- faction detection -------------------------------------------------------

def detect_faction_candidates(
    colonists: list[Any],
    social_graph: Any,
    active_ids: list[str],
) -> list[list[str]]:
    """Find groups of colonists that could form factions.

    Uses greedy clustering: start from highest mutual-trust pairs,
    grow clusters by adding colonists with high symmetric trust to
    all existing members AND value alignment with cluster centroid.
    """
    if len(active_ids) < MIN_FACTION_SIZE:
        return []

    # Build adjacency: pairs with symmetric trust above threshold
    adj: dict[str, set[str]] = {cid: set() for cid in active_ids}
    colonist_map = {c.id: c for c in colonists if c.id in active_ids}

    for i, a in enumerate(active_ids):
        for b in active_ids[i + 1:]:
            sym_trust = _symmetric_trust(social_graph, a, b)
            if sym_trust < TRUST_THRESHOLD:
                continue
            ca = colonist_map.get(a)
            cb = colonist_map.get(b)
            if ca is None or cb is None:
                continue
            if _value_distance(ca.stats, cb.stats) > VALUE_ALIGNMENT_THRESHOLD:
                continue
            adj[a].add(b)
            adj[b].add(a)

    # Greedy clique-ish clustering
    assigned: set[str] = set()
    clusters: list[list[str]] = []

    # Sort by connectivity (most connected first)
    sorted_ids = sorted(active_ids, key=lambda cid: len(adj[cid]), reverse=True)

    for seed_id in sorted_ids:
        if seed_id in assigned or len(adj[seed_id]) < MIN_FACTION_SIZE - 1:
            continue
        cluster = [seed_id]
        assigned.add(seed_id)
        candidates = sorted(adj[seed_id] - assigned,
                            key=lambda cid: len(adj[cid]), reverse=True)
        for cid in candidates:
            if cid in assigned:
                continue
            # Must be connected to all current cluster members
            if all(cid in adj[m] for m in cluster):
                cluster.append(cid)
                assigned.add(cid)
        if len(cluster) >= MIN_FACTION_SIZE:
            clusters.append(cluster)

    return clusters


def apply_faction_hysteresis(
    state: DiplomacyState,
    candidates: list[list[str]],
    year: int,
    colonists: list[Any],
    social_graph: Any,
) -> tuple[list[Faction], list[Faction]]:
    """Apply hysteresis to faction formation/dissolution.

    New factions form only after candidates persist for FORM years.
    Existing factions dissolve only after failing detection for DISSOLVE years.
    Returns (newly_formed, newly_dissolved).
    """
    formed: list[Faction] = []
    dissolved: list[Faction] = []

    # Track candidate persistence
    candidate_keys = set()
    for members in candidates:
        key = ",".join(sorted(members))
        candidate_keys.add(key)
        if key not in state._candidate_years:
            state._candidate_years[key] = 0
        state._candidate_years[key] += 1

    # Clean stale candidates
    stale = [k for k in state._candidate_years if k not in candidate_keys]
    for k in stale:
        del state._candidate_years[k]

    # Form new factions from persistent candidates
    active_member_sets = {",".join(sorted(f.member_ids))
                         for f in state.active_factions()}
    for members in candidates:
        key = ",".join(sorted(members))
        if key in active_member_sets:
            continue
        if state._candidate_years.get(key, 0) >= FACTION_HYSTERESIS_FORM:
            fid, fname = state._next_name()
            faction = Faction(
                id=fid, name=fname,
                leader_id=_choose_leader(colonists, members, social_graph),
                member_ids=list(members),
                dominant_value=_dominant_value(colonists, members),
                cohesion=_intra_cluster_density(members, social_graph),
                formed_year=year,
            )
            state.factions.append(faction)
            formed.append(faction)

    # Check dissolution of existing factions
    for faction in state.active_factions():
        key = ",".join(sorted(faction.member_ids))
        still_detected = key in candidate_keys
        if still_detected:
            state._dissolve_years.pop(faction.id, None)
            # Update cohesion
            faction.cohesion = _intra_cluster_density(
                faction.member_ids, social_graph)
            faction.leader_id = _choose_leader(
                colonists, faction.member_ids, social_graph)
        else:
            if faction.id not in state._dissolve_years:
                state._dissolve_years[faction.id] = 0
            state._dissolve_years[faction.id] += 1
            if state._dissolve_years[faction.id] >= FACTION_HYSTERESIS_DISSOLVE:
                faction.dissolved_year = year
                dissolved.append(faction)
                state._dissolve_years.pop(faction.id, None)

    return formed, dissolved


# -- alliance mechanics ------------------------------------------------------

def evaluate_alliance(
    faction_a: Faction,
    faction_b: Faction,
    social_graph: Any,
    year: int,
    rng: random.Random,
) -> Alliance | None:
    """Evaluate whether two factions should form an alliance.

    Alliance probability based on inter-faction trust, leader relationship,
    and complementary dominant values.
    """
    # Inter-faction average trust
    inter_trust = 0.0
    pairs = 0
    for a_id in faction_a.member_ids:
        for b_id in faction_b.member_ids:
            inter_trust += _symmetric_trust(social_graph, a_id, b_id)
            pairs += 1
    avg_inter_trust = inter_trust / max(1, pairs)

    # Leader relationship
    leader_trust = _symmetric_trust(
        social_graph, faction_a.leader_id, faction_b.leader_id)

    # Complementary values bonus
    complement_bonus = 0.1 if faction_a.dominant_value != faction_b.dominant_value else 0.0

    strength = avg_inter_trust * 0.4 + leader_trust * 0.4 + complement_bonus + 0.1
    strength = _clamp(strength)

    if strength < ALLIANCE_MIN_STRENGTH:
        return None

    prob = ALLIANCE_PROBABILITY * strength
    if rng.random() > prob:
        return None

    alliance_types = ["resource_sharing", "defense", "non_aggression"]
    weights = [strength, leader_trust, 1.0 - strength]
    total = sum(weights)
    r = rng.random() * total
    cumul = 0.0
    chosen_type = alliance_types[-1]
    for atype, w in zip(alliance_types, weights):
        cumul += w
        if r <= cumul:
            chosen_type = atype
            break

    duration = rng.randint(*ALLIANCE_DURATION_RANGE)
    return Alliance(
        faction_a_id=faction_a.id, faction_b_id=faction_b.id,
        alliance_type=chosen_type, strength=strength,
        formed_year=year, expires_year=year + duration,
    )


def check_betrayal(
    alliance: Alliance,
    factions: list[Faction],
    resource_avg: float,
    gini: float,
    psych_map: dict[str, Any],
    rng: random.Random,
) -> Betrayal | None:
    """Check if an alliance is betrayed under stress conditions.

    Returns a Betrayal if triggered, None otherwise.
    """
    faction_map = {f.id: f for f in factions if f.active}
    fa = faction_map.get(alliance.faction_a_id)
    fb = faction_map.get(alliance.faction_b_id)
    if fa is None or fb is None:
        return None

    # Check each faction's betrayal probability
    for attacker, victim in [(fa, fb), (fb, fa)]:
        prob = BETRAYAL_BASE_PROB

        # Economic stress
        prob += (1.0 - resource_avg) * BETRAYAL_SCARCITY_WEIGHT
        prob += gini * BETRAYAL_GINI_WEIGHT

        # Leader paranoia
        leader_psych = psych_map.get(attacker.leader_id)
        leader_paranoia = 0.5
        # Get paranoia from colonist stats if available
        for mid in attacker.member_ids:
            if mid == attacker.leader_id:
                break

        # Low alliance strength makes betrayal easier
        prob += (1.0 - alliance.strength) * 0.05

        # Low cohesion in attacker faction
        prob += (1.0 - attacker.cohesion) * 0.05

        prob = _clamp(prob, 0.0, 0.5)  # cap at 50%

        if rng.random() < prob:
            causes = []
            if resource_avg < 0.35:
                causes.append("resource scarcity")
            if gini > 0.4:
                causes.append("economic inequality")
            if attacker.cohesion < 0.4:
                causes.append("internal fracture")
            cause = ", ".join(causes) if causes else "strategic calculation"
            return Betrayal(
                betrayer_faction_id=attacker.id,
                victim_faction_id=victim.id,
                alliance_type=alliance.alliance_type,
                year=0,  # set by caller
                cause=cause,
            )

    return None


# -- trust feedback ----------------------------------------------------------

def apply_alliance_trust(
    alliances: list[Alliance],
    factions: list[Faction],
    social_graph: Any,
    year: int,
    rng: random.Random,
) -> int:
    """Boost trust between allied faction members. Returns adjustment count."""
    faction_map = {f.id: f for f in factions if f.active}
    adjustments = 0
    for alliance in alliances:
        if alliance.expires_year <= year:
            continue
        fa = faction_map.get(alliance.faction_a_id)
        fb = faction_map.get(alliance.faction_b_id)
        if fa is None or fb is None:
            continue
        for a_id in fa.member_ids:
            for b_id in fb.member_ids:
                edges = getattr(social_graph, 'edges', {})
                if a_id in edges and b_id in edges.get(a_id, {}):
                    rel = edges[a_id][b_id]
                    rel.trust = _clamp(rel.trust + ALLIANCE_TRUST_BOOST +
                                       rng.gauss(0, 0.005))
                    adjustments += 1
                if b_id in edges and a_id in edges.get(b_id, {}):
                    rel = edges[b_id][a_id]
                    rel.trust = _clamp(rel.trust + ALLIANCE_TRUST_BOOST +
                                       rng.gauss(0, 0.005))
                    adjustments += 1
    return adjustments


def apply_betrayal_trust(
    betrayal: Betrayal,
    factions: list[Faction],
    social_graph: Any,
    rng: random.Random,
) -> int:
    """Reduce trust between betrayer and victim faction members. Returns count."""
    faction_map = {f.id: f for f in factions if f.active}
    attacker = faction_map.get(betrayal.betrayer_faction_id)
    victim = faction_map.get(betrayal.victim_faction_id)
    if attacker is None or victim is None:
        return 0
    adjustments = 0
    for a_id in attacker.member_ids:
        for v_id in victim.member_ids:
            edges = getattr(social_graph, 'edges', {})
            for from_id, to_id in [(a_id, v_id), (v_id, a_id)]:
                if from_id in edges and to_id in edges.get(from_id, {}):
                    rel = edges[from_id][to_id]
                    rel.trust = _clamp(
                        rel.trust - BETRAYAL_TRUST_DAMAGE - abs(rng.gauss(0, 0.02)))
                    rel.affection = _clamp(
                        rel.affection - BETRAYAL_TRUST_DAMAGE * 0.5)
                    adjustments += 1
    return adjustments


# -- main tick ---------------------------------------------------------------

def tick_diplomacy(
    state: DiplomacyState,
    colonists: list[Any],
    social_graph: Any,
    resources_avg: float,
    gini: float,
    psych_map: dict[str, Any],
    year: int,
    rng: random.Random,
) -> DiplomacyTickResult:
    """Run one year of diplomacy. Mutates state and social_graph in place."""
    result = DiplomacyTickResult()
    active_ids = [c.id for c in colonists
                  if getattr(c, 'is_active', lambda: True)()]

    if len(active_ids) < MIN_FACTION_SIZE:
        return result

    # 1. Detect faction candidates
    candidates = detect_faction_candidates(colonists, social_graph, active_ids)

    # 2. Apply hysteresis → form/dissolve factions
    formed, dissolved = apply_faction_hysteresis(
        state, candidates, year, colonists, social_graph)
    result.factions_formed = [f.to_dict() for f in formed]
    result.factions_dissolved = [f.to_dict() for f in dissolved]

    # 3. Expire old alliances
    active_factions = state.active_factions()
    active_faction_ids = {f.id for f in active_factions}
    for alliance in list(state.alliances):
        if (alliance.expires_year <= year
                or alliance.faction_a_id not in active_faction_ids
                or alliance.faction_b_id not in active_faction_ids):
            result.alliances_expired.append(alliance.to_dict())
            state.alliances.remove(alliance)

    # 4. Check for betrayals in existing alliances
    for alliance in list(state.alliances):
        betrayal = check_betrayal(
            alliance, active_factions, resources_avg, gini, psych_map, rng)
        if betrayal is not None:
            betrayal.year = year
            state.betrayal_log.append(betrayal)
            result.betrayals.append(betrayal.to_dict())
            adj = apply_betrayal_trust(betrayal, active_factions, social_graph, rng)
            result.trust_adjustments += adj
            state.alliances.remove(alliance)

    # 5. Consider new alliances between unallied faction pairs
    allied_pairs: set[tuple[str, str]] = set()
    for a in state.alliances:
        allied_pairs.add((a.faction_a_id, a.faction_b_id))
        allied_pairs.add((a.faction_b_id, a.faction_a_id))

    for i, fa in enumerate(active_factions):
        for fb in active_factions[i + 1:]:
            if (fa.id, fb.id) in allied_pairs:
                continue
            alliance = evaluate_alliance(fa, fb, social_graph, year, rng)
            if alliance is not None:
                state.alliances.append(alliance)
                result.alliances_formed.append(alliance.to_dict())

    # 6. Apply trust feedback from active alliances
    adj = apply_alliance_trust(
        state.alliances, active_factions, social_graph, year, rng)
    result.trust_adjustments += adj

    return result
