"""
Mars-100 diplomacy engine — emergent factions, alliances, and political blocs.

Factions form organically from trust clusters in the social graph.
Alliances are formal pacts between faction leaders. Schisms split
factions when internal dissent reaches critical mass. Vote bias
nudges colonist votes toward faction consensus without being
deterministic.

Engine version: 5.1 (diplomacy organ)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Faction:
    """A political bloc that emerged from trust clustering."""
    id: str
    name: str
    members: list[str]
    leader_id: str
    formed_year: int
    cohesion: float = 1.0
    ideology: str = "pragmatist"

    IDEOLOGIES = [
        "expansionist", "conservationist", "technocrat",
        "spiritualist", "libertarian", "pragmatist",
    ]

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "members": list(self.members),
            "leader_id": self.leader_id, "formed_year": self.formed_year,
            "cohesion": round(self.cohesion, 4), "ideology": self.ideology,
        }


@dataclass
class Alliance:
    """A formal pact between two factions."""
    faction_a: str
    faction_b: str
    formed_year: int
    strength: float = 0.5

    def to_dict(self) -> dict:
        return {
            "faction_a": self.faction_a, "faction_b": self.faction_b,
            "formed_year": self.formed_year,
            "strength": round(self.strength, 4),
        }


@dataclass
class DiplomacyState:
    """Container for all diplomacy state across the simulation."""
    factions: list[Faction] = field(default_factory=list)
    alliances: list[Alliance] = field(default_factory=list)
    dissolved: list[dict] = field(default_factory=list)
    schism_log: list[dict] = field(default_factory=list)
    next_faction_id: int = 0

    def to_dict(self) -> dict:
        return {
            "factions": [f.to_dict() for f in self.factions],
            "alliances": [a.to_dict() for a in self.alliances],
            "dissolved": self.dissolved,
            "schism_log": self.schism_log,
        }

    def faction_of(self, colonist_id: str) -> Faction | None:
        """Return the faction a colonist belongs to, or None."""
        for f in self.factions:
            if colonist_id in f.members:
                return f
        return None


# ---------------------------------------------------------------------------
# Faction detection — density-based clustering
# ---------------------------------------------------------------------------

DENSITY_GAP_THRESHOLD = 0.12
MIN_FACTION_SIZE = 3

FACTION_NAMES = [
    "The Dustborn", "Olympus Compact", "Red Horizon", "Cydonia Circle",
    "Ares Vanguard", "Mariner's Watch", "Valles Concord", "Phobos Front",
    "Hellas Collective", "Tharsis Union", "Elysium Pact", "Noachian Guard",
]


def _compute_density(members: list[str], social: Any) -> float:
    """Average trust among members of a group."""
    if len(members) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            total += social.get(a, b).trust
            total += social.get(b, a).trust
            pairs += 2
    return total / max(pairs, 1)


def _external_density(members: list[str], all_ids: list[str],
                      social: Any) -> float:
    """Average trust between members and non-members."""
    outside = [cid for cid in all_ids if cid not in members]
    if not outside or not members:
        return 0.0
    total = 0.0
    count = 0
    for m in members:
        for o in outside:
            total += social.get(m, o).trust
            count += 1
    return total / max(count, 1)


def detect_factions(active_ids: list[str], social: Any,
                    rng: random.Random) -> list[Faction]:
    """Detect emergent factions via greedy density-based seed expansion.

    1. Rank colonists by average trust to others (most connected first).
    2. For each unassigned colonist, try to grow a faction by adding
       neighbours that keep internal density high.
    3. Accept if internal-external density > threshold and size >= MIN_FACTION_SIZE.
    """
    if len(active_ids) < MIN_FACTION_SIZE * 2:
        return []

    avg_trust: dict[str, float] = {}
    for cid in active_ids:
        others = [oid for oid in active_ids if oid != cid]
        if not others:
            avg_trust[cid] = 0.0
            continue
        avg_trust[cid] = sum(social.get(cid, o).trust for o in others) / len(others)

    seeds = sorted(active_ids, key=lambda c: avg_trust[c], reverse=True)
    assigned: set[str] = set()
    factions: list[Faction] = []

    for seed_id in seeds:
        if seed_id in assigned:
            continue
        cluster = [seed_id]
        candidates = [c for c in active_ids
                      if c != seed_id and c not in assigned]
        candidates.sort(
            key=lambda c: social.get(seed_id, c).trust, reverse=True,
        )
        for cand in candidates:
            trial = cluster + [cand]
            d_in = _compute_density(trial, social)
            d_out = _external_density(trial, active_ids, social)
            if d_in - d_out > DENSITY_GAP_THRESHOLD:
                cluster.append(cand)

        if len(cluster) < MIN_FACTION_SIZE:
            continue

        d_in = _compute_density(cluster, social)
        d_out = _external_density(cluster, active_ids, social)
        if d_in - d_out <= DENSITY_GAP_THRESHOLD:
            continue

        assigned.update(cluster)
        factions.append(Faction(
            id="", name="", members=cluster,
            leader_id=max(cluster, key=lambda c: avg_trust.get(c, 0.0)),
            formed_year=0, cohesion=d_in,
        ))

    return factions


# ---------------------------------------------------------------------------
# Reconciliation — stable faction IDs across ticks
# ---------------------------------------------------------------------------

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


JACCARD_THRESHOLD = 0.3


def reconcile_factions(new_factions: list[Faction],
                       old_state: DiplomacyState,
                       year: int, rng: random.Random) -> list[Faction]:
    """Match new factions to old ones by Jaccard overlap, assign IDs."""
    used_old: set[str] = set()
    result: list[Faction] = []

    for nf in new_factions:
        best_id = ""
        best_score = 0.0
        for of in old_state.factions:
            if of.id in used_old:
                continue
            score = _jaccard(set(nf.members), set(of.members))
            if score > best_score:
                best_score = score
                best_id = of.id
        if best_score >= JACCARD_THRESHOLD and best_id:
            nf.id = best_id
            old_match = next(f for f in old_state.factions if f.id == best_id)
            nf.name = old_match.name
            nf.ideology = old_match.ideology
            nf.formed_year = old_match.formed_year
            used_old.add(best_id)
        else:
            nf.id = f"faction-{old_state.next_faction_id}"
            old_state.next_faction_id += 1
            nf.name = rng.choice(FACTION_NAMES)
            nf.ideology = rng.choice(Faction.IDEOLOGIES)
            nf.formed_year = year
        result.append(nf)

    for of in old_state.factions:
        if of.id not in used_old:
            old_state.dissolved.append({
                "id": of.id, "name": of.name, "dissolved_year": year,
                "members_at_dissolution": of.members,
            })

    return result


# ---------------------------------------------------------------------------
# Schism — faction splits
# ---------------------------------------------------------------------------

SCHISM_PROBABILITY = 0.40
DISSENTER_DENSITY_THRESHOLD = 0.50


def check_schism(faction: Faction, social: Any,
                 rng: random.Random, year: int) -> tuple[Faction, Faction] | None:
    """Check if a faction should split.

    Requires: size >= 6, dissident cluster with density > threshold,
    probability gate passes, both daughters >= 3 members.
    """
    if len(faction.members) < 6:
        return None

    leader = faction.leader_id
    dissidents = [
        m for m in faction.members
        if m != leader and social.get(m, leader).trust < 0.4
    ]
    if len(dissidents) < MIN_FACTION_SIZE:
        return None
    loyalists = [m for m in faction.members if m not in dissidents]
    if len(loyalists) < MIN_FACTION_SIZE:
        return None

    d_density = _compute_density(dissidents, social)
    if d_density < DISSENTER_DENSITY_THRESHOLD:
        return None

    if rng.random() > SCHISM_PROBABILITY:
        return None

    avg_trust_d: dict[str, float] = {}
    for cid in dissidents:
        others = [o for o in dissidents if o != cid]
        avg_trust_d[cid] = (sum(social.get(cid, o).trust for o in others) / max(1, len(others))
                            if others else 0.0)
    dissident_leader = max(dissidents, key=lambda c: avg_trust_d.get(c, 0.0))

    daughter_a = Faction(
        id=faction.id, name=faction.name, members=loyalists,
        leader_id=leader, formed_year=faction.formed_year,
        cohesion=_compute_density(loyalists, social),
        ideology=faction.ideology,
    )
    daughter_b = Faction(
        id="", name="", members=dissidents,
        leader_id=dissident_leader, formed_year=year,
        cohesion=d_density,
        ideology=rng.choice(Faction.IDEOLOGIES),
    )
    return daughter_a, daughter_b


# ---------------------------------------------------------------------------
# Alliances
# ---------------------------------------------------------------------------

ALLIANCE_TRUST_THRESHOLD = 0.55
ALLIANCE_BREAK_THRESHOLD = 0.30


def update_alliances(factions: list[Faction], alliances: list[Alliance],
                     social: Any, year: int) -> list[Alliance]:
    """Form new alliances and break weak ones based on leader trust."""
    faction_ids = {f.id for f in factions}
    alive = [a for a in alliances
             if a.faction_a in faction_ids and a.faction_b in faction_ids]
    existing_pairs = {(a.faction_a, a.faction_b) for a in alive}

    faction_map = {f.id: f for f in factions}

    for i, fa in enumerate(factions):
        for fb in factions[i + 1:]:
            pair = (fa.id, fb.id)
            rpair = (fb.id, fa.id)
            if pair in existing_pairs or rpair in existing_pairs:
                continue
            leader_trust = social.get(fa.leader_id, fb.leader_id).trust
            if leader_trust > ALLIANCE_TRUST_THRESHOLD:
                alive.append(Alliance(
                    faction_a=fa.id, faction_b=fb.id,
                    formed_year=year, strength=leader_trust,
                ))
                existing_pairs.add(pair)

    result: list[Alliance] = []
    for a in alive:
        fa = faction_map.get(a.faction_a)
        fb = faction_map.get(a.faction_b)
        if fa and fb:
            a.strength = social.get(fa.leader_id, fb.leader_id).trust
            if a.strength >= ALLIANCE_BREAK_THRESHOLD:
                result.append(a)
    return result


# ---------------------------------------------------------------------------
# Vote bias — factions influence governance votes
# ---------------------------------------------------------------------------

MAX_VOTE_BIAS = 0.12


def faction_vote_bias(colonist_id: str, proposal_gov_type: str,
                      diplomacy: DiplomacyState) -> float:
    """Compute vote bias from faction membership.

    Bias = ideology affinity * cohesion, capped at +/-MAX_VOTE_BIAS.
    """
    faction = diplomacy.faction_of(colonist_id)
    if faction is None:
        return 0.0

    affinity: dict[str, dict[str, float]] = {
        "expansionist":    {"dictator": 0.3, "council": 0.1, "anarchy": -0.2},
        "conservationist": {"consensus": 0.3, "council": 0.2, "dictator": -0.3},
        "technocrat":      {"ai_governor": 0.4, "council": 0.1, "anarchy": -0.2},
        "spiritualist":    {"lottery": 0.3, "consensus": 0.2, "dictator": -0.2},
        "libertarian":     {"anarchy": 0.3, "lottery": 0.2, "dictator": -0.4},
        "pragmatist":      {"council": 0.2, "consensus": 0.1},
    }
    ideo_map = affinity.get(faction.ideology, {})
    raw = ideo_map.get(proposal_gov_type, 0.0)
    bias = raw * faction.cohesion
    return max(-MAX_VOTE_BIAS, min(MAX_VOTE_BIAS, bias))


# ---------------------------------------------------------------------------
# Ideology assignment helper
# ---------------------------------------------------------------------------

def _assign_ideology(members: list[str], colonists_map: dict[str, Any],
                     rng: random.Random) -> str:
    """Assign ideology based on aggregate stats of members."""
    if not members:
        return rng.choice(Faction.IDEOLOGIES)

    total_empathy = total_paranoia = total_coding = total_faith = 0.0
    count = 0
    for mid in members:
        c = colonists_map.get(mid)
        if c is None:
            continue
        total_empathy += c.stats.empathy
        total_paranoia += c.stats.paranoia
        total_coding += getattr(c.skills, "coding", 0.0)
        total_faith += c.stats.faith
        count += 1

    if count == 0:
        return rng.choice(Faction.IDEOLOGIES)

    scores = {
        "conservationist": total_empathy / count,
        "libertarian": total_paranoia / count,
        "technocrat": total_coding / count,
        "spiritualist": total_faith / count,
        "expansionist": (total_empathy + total_paranoia) / (2 * count),
        "pragmatist": 0.45,
    }
    return max(scores, key=scores.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Main tick function
# ---------------------------------------------------------------------------

def tick_diplomacy(diplomacy: DiplomacyState, active_ids: list[str],
                   social: Any, year: int, rng: random.Random,
                   colonists_map: dict[str, Any] | None = None) -> dict:
    """Run one year of diplomacy.  Returns a summary dict for YearResult."""
    raw_factions = detect_factions(active_ids, social, rng)

    if colonists_map:
        for f in raw_factions:
            f.ideology = _assign_ideology(f.members, colonists_map, rng)

    factions = reconcile_factions(raw_factions, diplomacy, year, rng)

    new_factions: list[Faction] = []
    for f in factions:
        schism = check_schism(f, social, rng, year)
        if schism:
            daughter_a, daughter_b = schism
            daughter_b.id = f"faction-{diplomacy.next_faction_id}"
            diplomacy.next_faction_id += 1
            daughter_b.name = rng.choice(FACTION_NAMES)
            new_factions.extend([daughter_a, daughter_b])
            diplomacy.schism_log.append({
                "parent_id": f.id, "year": year,
                "loyalist_id": daughter_a.id,
                "dissident_id": daughter_b.id,
                "dissident_members": daughter_b.members,
            })
        else:
            new_factions.append(f)

    diplomacy.factions = new_factions
    diplomacy.alliances = update_alliances(
        diplomacy.factions, diplomacy.alliances, social, year,
    )

    return {
        "year": year,
        "factions": [f.to_dict() for f in diplomacy.factions],
        "alliances": [a.to_dict() for a in diplomacy.alliances],
        "schisms_this_year": [s for s in diplomacy.schism_log if s["year"] == year],
        "dissolved_this_year": [d for d in diplomacy.dissolved
                                if d.get("dissolved_year") == year],
    }
