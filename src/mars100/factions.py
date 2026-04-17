"""
Factions organ for Mars-100 colony simulation (engine v9.0).

Colonists naturally cluster into ideological factions based on shared
stat profiles.  Factions emerge bottom-up (never predefined), influence
governance voting, and reduce member loneliness.

Key dynamics:
  - Formation: after year 8, unaffiliated colonists with similar stat
    profiles (mean absolute distance < threshold) cluster into factions
  - Recruitment: existing factions attract nearby unaffiliated colonists
  - Dissolution: factions with < 2 active members dissolve
  - Faction drift: ideology tracks the average of current members' stats
  - Downstream hooks:
    * Governance voting: same-faction bias (+0.15)
    * Psychology: faction membership reduces loneliness (via context flag)
  - Deferred to v10+: faction influence on trade trust thresholds

Determinism: uses a dedicated RNG stream (seed + 10007).
All colonist/faction IDs are sorted before iteration to ensure order
independence.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist, STAT_NAMES

# -- constants ---------------------------------------------------------------

FORMATION_MIN_YEAR = 8
FORMATION_DISTANCE_THRESHOLD = 0.18  # max mean-absolute-stat-distance to cluster
RECRUITMENT_DISTANCE_THRESHOLD = 0.22  # slightly looser for joining an existing faction
MIN_FACTION_SIZE = 2
MAX_FACTIONS = 6
FACTION_IDEOLOGY_DRIFT_RATE = 0.3  # how fast ideology tracks members

VOTING_SAME_FACTION_BIAS = 0.15

# Faction naming: map dominant-stat pairs to thematic names
_FACTION_NAMES: list[tuple[tuple[str, ...], list[str]]] = [
    (("faith", "resolve"), ["The Covenant", "Iron Faith", "The Steadfast"]),
    (("faith", "empathy"), ["The Mercy Circle", "Faithful Hearts", "The Compassionate"]),
    (("improvisation", "coding"), ["The Forge", "Circuit Minds", "Innovation Front"]),
    (("paranoia", "hoarding"), ["The Watchers", "Survival First", "The Cautious"]),
    (("empathy", "improvisation"), ["The Dreamers", "Open Horizon", "Free Spirits"]),
    (("resolve", "hoarding"), ["The Founders", "Bedrock Pact", "The Pillars"]),
    (("paranoia", "resolve"), ["The Sentinels", "Iron Guard", "Vigilance"]),
    (("empathy", "faith"), ["The Healers", "Soul Keepers", "The Gentle"]),
]

_GENERIC_NAMES = [
    "The Alliance", "The Collective", "The Assembly",
    "The Circle", "The Union", "The Accord",
    "The Syndicate", "The Order", "The Guild",
]


# -- data classes ------------------------------------------------------------

@dataclass
class Faction:
    """An emergent ideological faction."""
    id: str
    name: str
    formed_year: int
    ideology: dict[str, float]  # stat-name -> preferred value
    dissolved: bool = False
    dissolved_year: int | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id, "name": self.name,
            "formed_year": self.formed_year,
            "ideology": {k: round(v, 4) for k, v in self.ideology.items()},
            "dissolved": self.dissolved,
        }
        if self.dissolved_year is not None:
            d["dissolved_year"] = self.dissolved_year
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Faction:
        return cls(
            id=d["id"], name=d["name"],
            formed_year=d["formed_year"],
            ideology=d.get("ideology", {}),
            dissolved=d.get("dissolved", False),
            dissolved_year=d.get("dissolved_year"),
        )


@dataclass
class FactionState:
    """Colony-wide faction state.

    Membership dict is the single source of truth: colonist_id -> faction_id.
    Faction member lists are derived, never stored independently.
    """
    factions: dict[str, Faction] = field(default_factory=dict)
    membership: dict[str, str | None] = field(default_factory=dict)
    next_id: int = 0
    used_names: set[str] = field(default_factory=set)

    def active_factions(self) -> list[Faction]:
        """Return non-dissolved factions sorted by ID for determinism."""
        return sorted(
            [f for f in self.factions.values() if not f.dissolved],
            key=lambda f: f.id,
        )

    def members_of(self, faction_id: str) -> list[str]:
        """Derive member list from membership dict, sorted for determinism."""
        return sorted(
            cid for cid, fid in self.membership.items() if fid == faction_id
        )

    def faction_of(self, colonist_id: str) -> str | None:
        """Return faction ID for a colonist, or None."""
        return self.membership.get(colonist_id)

    def same_faction(self, cid_a: str, cid_b: str) -> bool:
        """Check if two colonists belong to the same active faction."""
        fa = self.membership.get(cid_a)
        fb = self.membership.get(cid_b)
        if fa is None or fb is None:
            return False
        if fa != fb:
            return False
        faction = self.factions.get(fa)
        return faction is not None and not faction.dissolved

    def to_dict(self) -> dict:
        return {
            "factions": {fid: f.to_dict() for fid, f in self.factions.items()},
            "membership": {k: v for k, v in self.membership.items()},
            "next_id": self.next_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FactionState:
        factions = {
            fid: Faction.from_dict(fd)
            for fid, fd in d.get("factions", {}).items()
        }
        used = {f.name for f in factions.values()}
        return cls(
            factions=factions,
            membership=d.get("membership", {}),
            next_id=d.get("next_id", 0),
            used_names=used,
        )


# -- tick result -------------------------------------------------------------

@dataclass
class FactionTickResult:
    """Result of one year's faction dynamics."""
    formed: list[dict] = field(default_factory=list)
    dissolved: list[dict] = field(default_factory=list)
    recruited: list[dict] = field(default_factory=list)
    active_count: int = 0
    total_affiliated: int = 0

    def to_dict(self) -> dict:
        return {
            "formed": self.formed,
            "dissolved": self.dissolved,
            "recruited": self.recruited,
            "active_count": self.active_count,
            "total_affiliated": self.total_affiliated,
        }


# -- pure helpers ------------------------------------------------------------

def stat_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Mean absolute distance between two stat profiles.

    Lower = more similar. Range [0.0, 1.0] when stats are in [0, 1].
    """
    diffs = [abs(a.get(s, 0.5) - b.get(s, 0.5)) for s in STAT_NAMES]
    return sum(diffs) / len(diffs)


def colonist_stat_profile(colonist: Colonist) -> dict[str, float]:
    """Extract stat profile as dict for distance computation."""
    return {s: getattr(colonist.stats, s) for s in STAT_NAMES}


def _pick_faction_name(ideology: dict[str, float],
                       used_names: set[str],
                       rng: random.Random) -> str:
    """Generate a faction name from ideology profile."""
    ranked = sorted(STAT_NAMES, key=lambda s: ideology.get(s, 0.0), reverse=True)
    top_pair = tuple(ranked[:2])

    for stat_pair, names in _FACTION_NAMES:
        if set(top_pair) == set(stat_pair):
            available = [n for n in names if n not in used_names]
            if available:
                return rng.choice(available)

    available_generic = [n for n in _GENERIC_NAMES if n not in used_names]
    if available_generic:
        return rng.choice(available_generic)
    return f"Faction-{rng.randint(100, 999)}"


def compute_ideology(members: list[Colonist]) -> dict[str, float]:
    """Compute average stat profile of a group of colonists."""
    if not members:
        return {s: 0.5 for s in STAT_NAMES}
    result: dict[str, float] = {}
    for s in STAT_NAMES:
        vals = [getattr(c.stats, s) for c in members]
        result[s] = sum(vals) / len(vals)
    return result


# -- formation ---------------------------------------------------------------

def _find_clusters(unaffiliated: list[Colonist],
                   threshold: float) -> list[list[Colonist]]:
    """Find clusters of similar colonists using deterministic greedy grouping.

    Sorts colonists by ID, then greedily forms groups: each unaffiliated
    colonist joins the first cluster whose centroid is within threshold,
    or starts a new cluster.
    """
    sorted_colonists = sorted(unaffiliated, key=lambda c: c.id)
    clusters: list[list[Colonist]] = []
    centroids: list[dict[str, float]] = []

    for colonist in sorted_colonists:
        profile = colonist_stat_profile(colonist)
        placed = False
        for i, centroid in enumerate(centroids):
            if stat_distance(profile, centroid) < threshold:
                clusters[i].append(colonist)
                centroids[i] = compute_ideology(clusters[i])
                placed = True
                break
        if not placed:
            clusters.append([colonist])
            centroids.append(profile)

    return [c for c in clusters if len(c) >= MIN_FACTION_SIZE]


def try_form_factions(state: FactionState, colonists: list[Colonist],
                      year: int, rng: random.Random) -> list[Faction]:
    """Attempt to form new factions from unaffiliated colonists."""
    if year < FORMATION_MIN_YEAR:
        return []

    active_count = len(state.active_factions())
    if active_count >= MAX_FACTIONS:
        return []

    active_colonists = sorted(
        [c for c in colonists if c.is_active()], key=lambda c: c.id)
    unaffiliated = [
        c for c in active_colonists
        if state.membership.get(c.id) is None
    ]
    if len(unaffiliated) < MIN_FACTION_SIZE:
        return []

    clusters = _find_clusters(unaffiliated, FORMATION_DISTANCE_THRESHOLD)
    formed: list[Faction] = []
    for cluster in clusters:
        if active_count + len(formed) >= MAX_FACTIONS:
            break
        ideology = compute_ideology(cluster)
        name = _pick_faction_name(ideology, state.used_names, rng)
        fid = f"faction-{state.next_id}"
        state.next_id += 1
        faction = Faction(id=fid, name=name, formed_year=year,
                          ideology=ideology)
        state.factions[fid] = faction
        state.used_names.add(name)
        for c in cluster:
            state.membership[c.id] = fid
        formed.append(faction)
    return formed


# -- recruitment -------------------------------------------------------------

def recruit_unaffiliated(state: FactionState, colonists: list[Colonist],
                         rng: random.Random) -> list[dict]:
    """Recruit unaffiliated colonists into nearby existing factions."""
    active_factions = state.active_factions()
    if not active_factions:
        return []

    active_colonists = sorted(
        [c for c in colonists if c.is_active()], key=lambda c: c.id)
    unaffiliated = [
        c for c in active_colonists
        if state.membership.get(c.id) is None
    ]

    recruited: list[dict] = []
    for colonist in unaffiliated:
        profile = colonist_stat_profile(colonist)
        best_faction: Faction | None = None
        best_dist = RECRUITMENT_DISTANCE_THRESHOLD

        for faction in active_factions:
            dist = stat_distance(profile, faction.ideology)
            if dist < best_dist:
                best_dist = dist
                best_faction = faction

        if best_faction is not None and rng.random() < 0.4:
            state.membership[colonist.id] = best_faction.id
            recruited.append({
                "colonist_id": colonist.id,
                "faction_id": best_faction.id,
                "distance": round(best_dist, 4),
            })

    return recruited


# -- dissolution & cleanup ---------------------------------------------------

def dissolve_empty_factions(state: FactionState,
                            active_ids: set[str],
                            year: int) -> list[dict]:
    """Dissolve factions with fewer than MIN_FACTION_SIZE active members."""
    dissolved: list[dict] = []
    for faction in state.active_factions():
        members = state.members_of(faction.id)
        active_members = [m for m in members if m in active_ids]
        if len(active_members) < MIN_FACTION_SIZE:
            faction.dissolved = True
            faction.dissolved_year = year
            for cid in members:
                state.membership[cid] = None
            dissolved.append({
                "faction_id": faction.id,
                "name": faction.name,
                "year": year,
                "reason": "insufficient_members",
            })
    return dissolved


def remove_inactive_members(state: FactionState,
                            active_ids: set[str]) -> None:
    """Remove dead/exiled colonists from membership."""
    for cid in list(state.membership.keys()):
        if cid not in active_ids:
            state.membership[cid] = None


# -- ideology drift ----------------------------------------------------------

def drift_ideologies(state: FactionState,
                     colonists: list[Colonist]) -> None:
    """Drift each faction's ideology toward its current members' average."""
    colonist_map = {c.id: c for c in colonists}
    for faction in state.active_factions():
        members = state.members_of(faction.id)
        member_colonists = [
            colonist_map[cid] for cid in members if cid in colonist_map
        ]
        if not member_colonists:
            continue
        current_avg = compute_ideology(member_colonists)
        for stat in STAT_NAMES:
            old = faction.ideology.get(stat, 0.5)
            target = current_avg.get(stat, 0.5)
            faction.ideology[stat] = old + (target - old) * FACTION_IDEOLOGY_DRIFT_RATE


# -- main tick ---------------------------------------------------------------

def tick_factions(state: FactionState, colonists: list[Colonist],
                  year: int, rng: random.Random) -> FactionTickResult:
    """Run one year of faction dynamics.  Mutates state in place."""
    result = FactionTickResult()

    active_ids = {c.id for c in colonists if c.is_active()}

    # 1. Remove dead/exiled members
    remove_inactive_members(state, active_ids)

    # 2. Dissolve factions that lost too many members
    result.dissolved = dissolve_empty_factions(state, active_ids, year)

    # 3. Try forming new factions from unaffiliated colonists
    formed = try_form_factions(state, colonists, year, rng)
    result.formed = [f.to_dict() for f in formed]

    # 4. Recruit unaffiliated colonists into existing factions
    result.recruited = recruit_unaffiliated(state, colonists, rng)

    # 5. Drift ideologies toward current member averages
    drift_ideologies(state, colonists)

    # 6. Summary stats
    result.active_count = len(state.active_factions())
    result.total_affiliated = sum(
        1 for cid in active_ids if state.membership.get(cid) is not None
    )

    return result
