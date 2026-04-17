"""
Diplomacy organ for Mars-100 (engine v11.0).

Models emergent factions, alliances, and political coalitions.
Factions form organically from colonist value clustering and social trust.
Bloc voting nudges governance proposals; faction pressure nudges actions.

One-year lag: diplomacy ticks AFTER deaths/exiles/immigration so faction
state matches end-of-year survivors.  State feeds into NEXT year's
_choose_action() and _vote_on_proposal().

RNG offset: seed + 12553
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IDEOLOGY_NAMES = (
    "cooperative", "survivalist", "spiritual", "technocratic", "isolationist",
)

FACTION_MIN_SIZE = 3
FACTION_TRUST_THRESHOLD = 0.35
IDEOLOGY_HYSTERESIS_MARGIN = 0.08

# Pressure caps — diplomacy nudges, doesn't dominate
MAX_BLOC_PRESSURE = 0.25
MAX_VOTE_BIAS = 0.15

# Alliance / tension
ALLIANCE_THRESHOLD = 0.30
ALLIANCE_BREAK_THRESHOLD = 0.60
TENSION_DECAY = 0.05
TENSION_FROM_COMPETITION = 0.02
TENSION_FROM_SCHISM = 0.15

# Schism
SCHISM_COHESION_THRESHOLD = 0.20
SCHISM_MIN_SIZE = 5

# Faction naming pools
_FACTION_ADJECTIVES = [
    "Red", "Iron", "Dust", "Solar", "Free", "New", "Old", "Deep",
    "High", "First", "Last", "True", "Wild", "Calm", "Bright",
]
_FACTION_NOUNS = [
    "Compact", "Circle", "Front", "Union", "Pact", "Lodge", "Guild",
    "Senate", "Commune", "Order", "Assembly", "Caucus", "Alliance", "Band",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Faction:
    """An emergent political faction."""
    id: str
    name: str
    ideology: str
    members: list[str]
    leader_id: str | None
    founding_year: int
    cohesion: float = 0.5
    influence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "ideology": self.ideology,
            "members": list(self.members), "leader_id": self.leader_id,
            "founding_year": self.founding_year,
            "cohesion": round(self.cohesion, 4),
            "influence": round(self.influence, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Faction:
        return cls(
            id=d["id"], name=d["name"], ideology=d["ideology"],
            members=list(d.get("members", [])),
            leader_id=d.get("leader_id"),
            founding_year=d.get("founding_year", 0),
            cohesion=d.get("cohesion", 0.5),
            influence=d.get("influence", 0.0),
        )


@dataclass
class Alliance:
    """Formal cooperation between two factions."""
    faction_a: str
    faction_b: str
    strength: float
    formed_year: int

    def pair_key(self) -> tuple[str, str]:
        return (min(self.faction_a, self.faction_b),
                max(self.faction_a, self.faction_b))

    def to_dict(self) -> dict:
        return {
            "faction_a": self.faction_a, "faction_b": self.faction_b,
            "strength": round(self.strength, 4),
            "formed_year": self.formed_year,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Alliance:
        return cls(
            faction_a=d["faction_a"], faction_b=d["faction_b"],
            strength=d.get("strength", 0.5),
            formed_year=d.get("formed_year", 0),
        )


@dataclass
class DiplomacyState:
    """Colony-wide diplomacy state."""
    factions: dict[str, Faction] = field(default_factory=dict)
    alliances: list[Alliance] = field(default_factory=list)
    tensions: dict[str, float] = field(default_factory=dict)
    next_faction_id: int = 0
    ideology_cache: dict[str, str] = field(default_factory=dict)
    ideology_age: dict[str, int] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factions": {fid: f.to_dict() for fid, f in self.factions.items()},
            "alliances": [a.to_dict() for a in self.alliances],
            "tensions": dict(self.tensions),
            "next_faction_id": self.next_faction_id,
            "faction_count": len(self.factions),
            "alliance_count": len(self.alliances),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiplomacyState:
        state = cls()
        for fid, fd in d.get("factions", {}).items():
            state.factions[fid] = Faction.from_dict(fd)
        for ad in d.get("alliances", []):
            state.alliances.append(Alliance.from_dict(ad))
        state.tensions = dict(d.get("tensions", {}))
        state.next_faction_id = d.get("next_faction_id", 0)
        return state


@dataclass
class DiplomacyTickResult:
    """Result of one year of diplomacy."""
    factions_formed: list[dict] = field(default_factory=list)
    factions_dissolved: list[dict] = field(default_factory=list)
    alliances_formed: list[dict] = field(default_factory=list)
    alliances_broken: list[dict] = field(default_factory=list)
    schisms: list[dict] = field(default_factory=list)
    leader_changes: list[dict] = field(default_factory=list)
    faction_count: int = 0
    alliance_count: int = 0

    def to_dict(self) -> dict:
        return {
            "factions_formed": self.factions_formed,
            "factions_dissolved": self.factions_dissolved,
            "alliances_formed": self.alliances_formed,
            "alliances_broken": self.alliances_broken,
            "schisms": self.schisms,
            "leader_changes": self.leader_changes,
            "faction_count": self.faction_count,
            "alliance_count": self.alliance_count,
        }


# ---------------------------------------------------------------------------
# Ideology classification
# ---------------------------------------------------------------------------

def _ideology_scores(stats: dict[str, float],
                     skills: dict[str, float]) -> dict[str, float]:
    """Compute ideology affinity scores from colonist stats/skills."""
    return {
        "cooperative": (stats.get("empathy", 0.5) * 0.4
                        + skills.get("mediation", 0.0) * 0.3
                        + (1.0 - stats.get("paranoia", 0.5)) * 0.3),
        "survivalist": (stats.get("paranoia", 0.5) * 0.3
                        + stats.get("hoarding", 0.5) * 0.3
                        + stats.get("resolve", 0.5) * 0.2
                        + skills.get("sabotage", 0.0) * 0.2),
        "spiritual":   (stats.get("faith", 0.5) * 0.5
                        + skills.get("prayer", 0.0) * 0.3
                        + stats.get("empathy", 0.5) * 0.2),
        "technocratic": (skills.get("coding", 0.0) * 0.4
                         + stats.get("improvisation", 0.5) * 0.3
                         + skills.get("terraforming", 0.0) * 0.3),
        "isolationist": (stats.get("hoarding", 0.5) * 0.3
                         + stats.get("paranoia", 0.5) * 0.2
                         + (1.0 - stats.get("empathy", 0.5)) * 0.3
                         + stats.get("faith", 0.5) * 0.2),
    }


def classify_ideology(stats: dict[str, float],
                       skills: dict[str, float],
                       prior: str | None = None) -> str:
    """Classify a colonist's ideology with hysteresis.

    If the colonist had a prior ideology, only switch if the new winner
    exceeds the prior score by IDEOLOGY_HYSTERESIS_MARGIN.
    """
    scores = _ideology_scores(stats, skills)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if prior is not None and prior in scores:
        if scores[best] - scores[prior] < IDEOLOGY_HYSTERESIS_MARGIN:
            return prior
    return best


# ---------------------------------------------------------------------------
# Faction formation
# ---------------------------------------------------------------------------

def _tension_key(fa: str, fb: str) -> str:
    """Normalized tension key for a faction pair."""
    a, b = min(fa, fb), max(fa, fb)
    return f"{a}:{b}"


def _generate_faction_name(ideology: str, faction_id: int,
                            rng: Any) -> str:
    """Generate a procedural faction name."""
    adj = rng.choice(_FACTION_ADJECTIVES)
    noun = rng.choice(_FACTION_NOUNS)
    return f"{adj} {noun}"


def _avg_pair_trust(members: list[str],
                     social_get: Any) -> float:
    """Average pairwise trust among faction members."""
    if len(members) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, a in enumerate(sorted(members)):
        for b in sorted(members)[i + 1:]:
            rel = social_get(a, b)
            total += getattr(rel, "trust", 0.5)
            count += 1
    return total / max(1, count)


def _update_cohesion(faction: Faction, social_get: Any) -> None:
    """Recompute faction cohesion from member trust."""
    faction.cohesion = _avg_pair_trust(faction.members, social_get)


def _update_influence(faction: Faction, total_active: int) -> None:
    """Recompute faction influence from member proportion."""
    if total_active <= 0:
        faction.influence = 0.0
    else:
        faction.influence = min(1.0, len(faction.members) / total_active)


def _elect_leader(faction: Faction, stats_lookup: dict[str, dict],
                   rng: Any) -> str | None:
    """Elect faction leader — highest resolve among members."""
    if not faction.members:
        return None
    candidates = sorted(faction.members)
    best_id = candidates[0]
    best_resolve = 0.0
    for mid in candidates:
        s = stats_lookup.get(mid, {})
        r = s.get("resolve", 0.5) + rng.gauss(0, 0.05)
        if r > best_resolve:
            best_resolve = r
            best_id = mid
    return best_id


# ---------------------------------------------------------------------------
# Main tick
# ---------------------------------------------------------------------------

def tick_diplomacy(
    state: DiplomacyState,
    active_colonists: list[dict],
    social_get: Any,
    actions: dict[str, str],
    year: int,
    rng: Any,
) -> DiplomacyTickResult:
    """Advance diplomacy by one year.

    Called AFTER deaths/exiles/immigration so faction membership
    reflects the true end-of-year population.

    Args:
        active_colonists: list of colonist.to_dict() for active colonists.
        social_get: callable(a_id, b_id) -> Relationship.
        actions: year's action map {colonist_id: action}.
        year: current simulation year.
        rng: dedicated diplomacy RNG.

    Returns:
        DiplomacyTickResult with all changes.
    """
    result = DiplomacyTickResult()
    active_ids = sorted(c["id"] for c in active_colonists)
    if len(active_ids) < FACTION_MIN_SIZE:
        # Too few colonists for factions
        _cleanup_all(state, result)
        result.faction_count = 0
        result.alliance_count = 0
        return result

    stats_lookup = {c["id"]: c.get("stats", {}) for c in active_colonists}
    skills_lookup = {c["id"]: c.get("skills", {}) for c in active_colonists}

    # 1. Update ideology for every active colonist (with hysteresis)
    for cid in active_ids:
        prior = state.ideology_cache.get(cid)
        ideology = classify_ideology(
            stats_lookup.get(cid, {}), skills_lookup.get(cid, {}), prior)
        if ideology != prior:
            state.ideology_cache[cid] = ideology
            state.ideology_age[cid] = 0
        else:
            state.ideology_age[cid] = state.ideology_age.get(cid, 0) + 1

    # 2. Prune dead/exiled members from existing factions
    _prune_factions(state, active_ids, stats_lookup, rng, result)

    # 3. Try to form new factions from unaffiliated colonists
    affiliated = set()
    for f in state.factions.values():
        affiliated.update(f.members)
    unaffiliated = [cid for cid in active_ids if cid not in affiliated]

    by_ideology: dict[str, list[str]] = {}
    for cid in sorted(unaffiliated):
        ideo = state.ideology_cache.get(cid, "cooperative")
        # Only form factions with stable ideology (2+ years)
        if state.ideology_age.get(cid, 0) >= 2:
            by_ideology.setdefault(ideo, []).append(cid)

    for ideo, candidates in sorted(by_ideology.items()):
        if len(candidates) >= FACTION_MIN_SIZE:
            avg_trust = _avg_pair_trust(candidates, social_get)
            if avg_trust >= FACTION_TRUST_THRESHOLD:
                fid = f"faction-{state.next_faction_id}"
                state.next_faction_id += 1
                name = _generate_faction_name(ideo, state.next_faction_id, rng)
                faction = Faction(
                    id=fid, name=name, ideology=ideo,
                    members=list(candidates),
                    leader_id=None, founding_year=year,
                    cohesion=avg_trust,
                )
                faction.leader_id = _elect_leader(
                    faction, stats_lookup, rng)
                _update_influence(faction, len(active_ids))
                state.factions[fid] = faction
                result.factions_formed.append(faction.to_dict())

    # 4. Check for schisms in large factions
    for fid in sorted(list(state.factions.keys())):
        faction = state.factions.get(fid)
        if faction is None or len(faction.members) < SCHISM_MIN_SIZE:
            continue
        if faction.cohesion < SCHISM_COHESION_THRESHOLD:
            _handle_schism(state, fid, stats_lookup, skills_lookup,
                           social_get, active_ids, year, rng, result)

    # 5. Update cohesion and influence for all factions
    for faction in state.factions.values():
        _update_cohesion(faction, social_get)
        _update_influence(faction, len(active_ids))

    # 6. Alliance dynamics
    _tick_alliances(state, year, rng, result)

    # 7. Tension dynamics
    _tick_tensions(state, actions, rng)

    result.faction_count = len(state.factions)
    result.alliance_count = len(state.alliances)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prune_factions(state: DiplomacyState, active_ids: list[str],
                     stats_lookup: dict, rng: Any,
                     result: DiplomacyTickResult) -> None:
    """Remove dead/exiled members; dissolve tiny factions."""
    active_set = set(active_ids)
    to_dissolve: list[str] = []
    for fid in sorted(state.factions.keys()):
        faction = state.factions[fid]
        faction.members = [m for m in faction.members if m in active_set]
        if len(faction.members) < 2:
            to_dissolve.append(fid)
            continue
        if faction.leader_id not in active_set:
            faction.leader_id = _elect_leader(faction, stats_lookup, rng)
            result.leader_changes.append({
                "faction_id": fid, "new_leader": faction.leader_id,
            })
    for fid in to_dissolve:
        result.factions_dissolved.append(state.factions[fid].to_dict())
        del state.factions[fid]
    # Clean alliances referencing dissolved factions
    alive = set(state.factions.keys())
    state.alliances = [a for a in state.alliances
                        if a.faction_a in alive and a.faction_b in alive]
    # Clean tensions
    state.tensions = {
        k: v for k, v in state.tensions.items()
        if all(part in alive for part in k.split(":"))
    }


def _handle_schism(state: DiplomacyState, fid: str,
                    stats_lookup: dict, skills_lookup: dict,
                    social_get: Any, active_ids: list[str],
                    year: int, rng: Any,
                    result: DiplomacyTickResult) -> None:
    """Split a low-cohesion faction into two."""
    faction = state.factions[fid]
    members = sorted(faction.members)
    mid = len(members) // 2
    group_a = members[:mid]
    group_b = members[mid:]
    if len(group_a) < 2 or len(group_b) < 2:
        return

    # Splinter group becomes new faction
    new_fid = f"faction-{state.next_faction_id}"
    state.next_faction_id += 1
    # Re-classify ideology for splinter group
    ideo_counts: dict[str, int] = {}
    for cid in group_b:
        ideo = state.ideology_cache.get(cid, "cooperative")
        ideo_counts[ideo] = ideo_counts.get(ideo, 0) + 1
    new_ideo = max(ideo_counts, key=ideo_counts.get) if ideo_counts else faction.ideology  # type: ignore[arg-type]

    new_faction = Faction(
        id=new_fid,
        name=_generate_faction_name(new_ideo, state.next_faction_id, rng),
        ideology=new_ideo, members=group_b,
        leader_id=None, founding_year=year,
    )
    new_faction.leader_id = _elect_leader(new_faction, stats_lookup, rng)
    _update_cohesion(new_faction, social_get)
    _update_influence(new_faction, len(active_ids))

    # Shrink original
    faction.members = group_a
    faction.leader_id = _elect_leader(faction, stats_lookup, rng)
    _update_cohesion(faction, social_get)

    state.factions[new_fid] = new_faction

    # Add tension between parent and splinter
    tk = _tension_key(fid, new_fid)
    state.tensions[tk] = min(1.0, state.tensions.get(tk, 0.0)
                              + TENSION_FROM_SCHISM)

    result.schisms.append({
        "parent_faction": fid, "splinter_faction": new_fid,
        "year": year, "splinter_ideology": new_ideo,
    })


def _tick_alliances(state: DiplomacyState, year: int,
                     rng: Any,
                     result: DiplomacyTickResult) -> None:
    """Form and break alliances based on ideology compatibility."""
    fids = sorted(state.factions.keys())
    existing_pairs = {a.pair_key() for a in state.alliances}

    # Try forming new alliances
    for i, fa in enumerate(fids):
        for fb in fids[i + 1:]:
            pair = (min(fa, fb), max(fa, fb))
            if pair in existing_pairs:
                continue
            f_a = state.factions[fa]
            f_b = state.factions[fb]
            # Compatible ideologies form alliances more easily
            compat = _ideology_compatibility(f_a.ideology, f_b.ideology)
            tk = _tension_key(fa, fb)
            tension = state.tensions.get(tk, 0.0)
            if compat > 0.3 and tension < ALLIANCE_THRESHOLD and rng.random() < 0.1:
                alliance = Alliance(faction_a=fa, faction_b=fb,
                                     strength=compat * 0.5, formed_year=year)
                state.alliances.append(alliance)
                result.alliances_formed.append(alliance.to_dict())

    # Break alliances with high tension
    to_remove: list[int] = []
    for idx, alliance in enumerate(state.alliances):
        tk = _tension_key(alliance.faction_a, alliance.faction_b)
        tension = state.tensions.get(tk, 0.0)
        if tension > ALLIANCE_BREAK_THRESHOLD:
            to_remove.append(idx)
            result.alliances_broken.append(alliance.to_dict())
        else:
            # Strengthen over time
            alliance.strength = min(1.0, alliance.strength + 0.02)
    for idx in reversed(to_remove):
        state.alliances.pop(idx)


def _tick_tensions(state: DiplomacyState,
                    actions: dict[str, str],
                    rng: Any) -> None:
    """Evolve inter-faction tensions."""
    fids = sorted(state.factions.keys())

    # Natural decay
    for key in list(state.tensions.keys()):
        state.tensions[key] = max(0.0, state.tensions[key] - TENSION_DECAY)

    # Competition: factions competing for same resources increase tension
    for i, fa in enumerate(fids):
        f_a = state.factions[fa]
        for fb in fids[i + 1:]:
            f_b = state.factions[fb]
            tk = _tension_key(fa, fb)
            # Same ideology = more competition for same niche
            if f_a.ideology == f_b.ideology:
                state.tensions[tk] = min(
                    1.0, state.tensions.get(tk, 0.0) + TENSION_FROM_COMPETITION)
            # Opposing ideologies = philosophical tension
            compat = _ideology_compatibility(f_a.ideology, f_b.ideology)
            if compat < 0.2:
                state.tensions[tk] = min(
                    1.0, state.tensions.get(tk, 0.0)
                    + TENSION_FROM_COMPETITION * 0.5)


def _ideology_compatibility(a: str, b: str) -> float:
    """How compatible two ideologies are (0=opposed, 1=aligned)."""
    if a == b:
        return 0.8
    compat_matrix = {
        ("cooperative", "spiritual"): 0.6,
        ("cooperative", "technocratic"): 0.4,
        ("cooperative", "survivalist"): 0.2,
        ("cooperative", "isolationist"): 0.1,
        ("survivalist", "isolationist"): 0.5,
        ("survivalist", "technocratic"): 0.3,
        ("survivalist", "spiritual"): 0.2,
        ("spiritual", "isolationist"): 0.4,
        ("spiritual", "technocratic"): 0.2,
        ("technocratic", "isolationist"): 0.3,
    }
    key = (min(a, b), max(a, b))
    return compat_matrix.get(key, 0.3)


def _cleanup_all(state: DiplomacyState,
                  result: DiplomacyTickResult) -> None:
    """Dissolve everything when population is too small."""
    for fid, faction in list(state.factions.items()):
        result.factions_dissolved.append(faction.to_dict())
    state.factions.clear()
    state.alliances.clear()
    state.tensions.clear()


# ---------------------------------------------------------------------------
# External pressure functions (called by engine)
# ---------------------------------------------------------------------------

def compute_bloc_pressure(state: DiplomacyState,
                           colonist_id: str,
                           action_names: list[str]) -> dict[str, float]:
    """Compute action weight nudges from faction membership.

    Returns a dict of action -> pressure (positive = encourage).
    Pressure is capped at MAX_BLOC_PRESSURE.
    """
    pressure: dict[str, float] = {}
    faction = _find_faction(state, colonist_id)
    if faction is None or faction.cohesion < 0.3:
        return pressure

    scale = faction.cohesion * faction.influence
    ideology_bias = _ideology_action_bias(faction.ideology)
    for action in action_names:
        raw = ideology_bias.get(action, 0.0) * scale
        pressure[action] = max(-MAX_BLOC_PRESSURE,
                                min(MAX_BLOC_PRESSURE, raw))
    return pressure


def compute_faction_vote_bias(state: DiplomacyState,
                                colonist_id: str,
                                proposer_id: str) -> float:
    """Compute voting bias from faction loyalty.

    Same faction as proposer → positive bias.
    Rival faction (high tension) → negative bias.
    """
    my_faction = _find_faction(state, colonist_id)
    their_faction = _find_faction(state, proposer_id)
    if my_faction is None or their_faction is None:
        return 0.0
    if my_faction.id == their_faction.id:
        return MAX_VOTE_BIAS * my_faction.cohesion
    tk = _tension_key(my_faction.id, their_faction.id)
    tension = state.tensions.get(tk, 0.0)
    allied = any(
        a.pair_key() == (min(my_faction.id, their_faction.id),
                          max(my_faction.id, their_faction.id))
        for a in state.alliances
    )
    if allied:
        return MAX_VOTE_BIAS * 0.5
    return -MAX_VOTE_BIAS * tension


def _find_faction(state: DiplomacyState,
                   colonist_id: str) -> Faction | None:
    """Find which faction a colonist belongs to."""
    for faction in state.factions.values():
        if colonist_id in faction.members:
            return faction
    return None


def _ideology_action_bias(ideology: str) -> dict[str, float]:
    """What actions does this ideology encourage/discourage?"""
    biases: dict[str, dict[str, float]] = {
        "cooperative": {"cooperate": 0.4, "mediate": 0.3, "sabotage": -0.3},
        "survivalist": {"hoard": 0.3, "farm": 0.2, "sabotage": 0.1,
                         "cooperate": -0.2},
        "spiritual": {"pray": 0.4, "mediate": 0.2, "cooperate": 0.1,
                        "sabotage": -0.3},
        "technocratic": {"code": 0.3, "research": 0.3, "terraform": 0.2,
                          "pray": -0.2},
        "isolationist": {"hoard": 0.3, "rest": 0.2, "explore": 0.2,
                          "cooperate": -0.3, "mediate": -0.2},
    }
    return biases.get(ideology, {})
