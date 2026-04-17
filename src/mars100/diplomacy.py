"""
Diplomacy organ for Mars-100 (engine v11.0).

Models factional politics: formation, ideology, alliances, rivalries,
schisms, power balance, and governance pressure.  Factions emerge
organically from the social graph's trust clusters combined with
stat similarity.  Pure computation — no I/O.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_FACTION_SIZE = 3
TRUST_THRESHOLD = 0.55          # avg mutual trust to seed a cluster
STAT_SIMILARITY_THRESHOLD = 0.4 # max L2 distance on ideology axes
SCHISM_VARIANCE_THRESHOLD = 0.55
SCHISM_MIN_SIZE = 5
ALLIANCE_IDEOLOGY_DIST = 0.5
RIVALRY_IDEOLOGY_DIST = 1.0
ALLIANCE_MIN_TENURE = 3         # years before an alliance can break
FACTION_DISSOLUTION_GRACE = 2   # years below threshold before dissolving
COOLDOWN_YEARS = 2              # years between schisms for same faction
MAX_HISTORY = 40                # pruned per tick
OVERLAP_MATCH_THRESHOLD = 0.4   # Jaccard overlap to match prior faction

FACTION_NAMES = [
    "Ares Compact", "Olympus League", "Valles Union", "Hellas Pact",
    "Tharsis Circle", "Elysium Accord", "Syrtis Front", "Chryse Band",
    "Isidis Collective", "Utopia Bloc", "Arcadia Assembly", "Amazonis Caucus",
    "Cydonia Order", "Noachis Alliance", "Hesperia Compact", "Marineris Vow",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Faction:
    """A named political group of colonists."""
    id: str
    name: str
    leader_id: str
    member_ids: list[str]
    ideology: dict[str, float]  # collectivism, expansionism, spiritualism
    formed_year: int
    dissolved_year: int | None = None
    power: float = 0.0
    last_schism_year: int = 0

    def is_active(self) -> bool:
        return self.dissolved_year is None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "leader_id": self.leader_id,
            "member_ids": list(self.member_ids),
            "ideology": dict(self.ideology), "formed_year": self.formed_year,
            "dissolved_year": self.dissolved_year, "power": self.power,
            "last_schism_year": self.last_schism_year,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Faction:
        return cls(
            id=d["id"], name=d["name"], leader_id=d["leader_id"],
            member_ids=list(d["member_ids"]),
            ideology=dict(d.get("ideology", {})),
            formed_year=d["formed_year"],
            dissolved_year=d.get("dissolved_year"),
            power=d.get("power", 0.0),
            last_schism_year=d.get("last_schism_year", 0),
        )


@dataclass
class Alliance:
    """Formal pact between two factions (canonical pair ordering)."""
    faction_a: str  # always < faction_b lexicographically
    faction_b: str
    strength: float  # 0.0-1.0
    formed_year: int

    def pair(self) -> tuple[str, str]:
        return (self.faction_a, self.faction_b)

    def to_dict(self) -> dict:
        return {"faction_a": self.faction_a, "faction_b": self.faction_b,
                "strength": self.strength, "formed_year": self.formed_year}

    @classmethod
    def from_dict(cls, d: dict) -> Alliance:
        a, b = sorted([d["faction_a"], d["faction_b"]])
        return cls(faction_a=a, faction_b=b,
                   strength=d.get("strength", 0.5),
                   formed_year=d["formed_year"])


@dataclass
class Rivalry:
    """Hostile relationship between two factions."""
    faction_a: str
    faction_b: str
    intensity: float  # 0.0-1.0
    formed_year: int

    def pair(self) -> tuple[str, str]:
        return (self.faction_a, self.faction_b)

    def to_dict(self) -> dict:
        return {"faction_a": self.faction_a, "faction_b": self.faction_b,
                "intensity": self.intensity, "formed_year": self.formed_year}

    @classmethod
    def from_dict(cls, d: dict) -> Rivalry:
        a, b = sorted([d["faction_a"], d["faction_b"]])
        return cls(faction_a=a, faction_b=b,
                   intensity=d.get("intensity", 0.5),
                   formed_year=d["formed_year"])


@dataclass
class DiplomacyState:
    """Persistent diplomacy state across years."""
    factions: dict[str, Faction] = field(default_factory=dict)
    alliances: list[Alliance] = field(default_factory=list)
    rivalries: list[Rivalry] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    next_faction_id: int = 0
    name_cursor: int = 0
    dissolution_countdown: dict[str, int] = field(default_factory=dict)

    def active_factions(self) -> dict[str, Faction]:
        return {fid: f for fid, f in self.factions.items() if f.is_active()}

    def faction_of(self, colonist_id: str) -> str | None:
        """Return faction id for a colonist, or None."""
        for fid, f in self.factions.items():
            if f.is_active() and colonist_id in f.member_ids:
                return fid
        return None

    def to_dict(self) -> dict:
        return {
            "factions": {fid: f.to_dict() for fid, f in self.factions.items()},
            "alliances": [a.to_dict() for a in self.alliances],
            "rivalries": [r.to_dict() for r in self.rivalries],
            "history": list(self.history[-MAX_HISTORY:]),
            "next_faction_id": self.next_faction_id,
            "name_cursor": self.name_cursor,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiplomacyState:
        factions = {fid: Faction.from_dict(fd) for fid, fd in d.get("factions", {}).items()}
        alliances = [Alliance.from_dict(ad) for ad in d.get("alliances", [])]
        rivalries = [Rivalry.from_dict(rd) for rd in d.get("rivalries", [])]
        return cls(factions=factions, alliances=alliances, rivalries=rivalries,
                   history=list(d.get("history", [])),
                   next_faction_id=d.get("next_faction_id", 0),
                   name_cursor=d.get("name_cursor", 0))


# ---------------------------------------------------------------------------
# Context and result
# ---------------------------------------------------------------------------
@dataclass
class DiplomacyYearContext:
    """Inputs for one year of diplomacy."""
    year: int
    active_colonist_ids: list[str]
    colonist_stats: dict[str, dict[str, float]]
    social_trusts: dict[str, dict[str, float]]
    governance_type: str
    resource_avg: float

    def trust_between(self, a: str, b: str) -> float:
        return self.social_trusts.get(a, {}).get(b, 0.5)


@dataclass
class DiplomacyTickResult:
    """Output of one diplomacy tick."""
    factions_formed: list[dict] = field(default_factory=list)
    factions_dissolved: list[dict] = field(default_factory=list)
    alliances_formed: list[dict] = field(default_factory=list)
    alliances_broken: list[dict] = field(default_factory=list)
    rivalries_formed: list[dict] = field(default_factory=list)
    rivalries_ended: list[dict] = field(default_factory=list)
    schisms: list[dict] = field(default_factory=list)
    power_balance: dict[str, float] = field(default_factory=dict)
    governance_pressure: dict[str, float] = field(default_factory=dict)
    loneliness_modifiers: dict[str, float] = field(default_factory=dict)
    purpose_modifiers: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "factions_formed": self.factions_formed,
            "factions_dissolved": self.factions_dissolved,
            "alliances_formed": self.alliances_formed,
            "alliances_broken": self.alliances_broken,
            "rivalries_formed": self.rivalries_formed,
            "rivalries_ended": self.rivalries_ended,
            "schisms": self.schisms,
            "power_balance": self.power_balance,
            "governance_pressure": self.governance_pressure,
        }


# ---------------------------------------------------------------------------
# Ideology computation
# ---------------------------------------------------------------------------
def compute_ideology(member_ids: list[str],
                     stats: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute faction ideology from member stats."""
    if not member_ids:
        return {"collectivism": 0.0, "expansionism": 0.0, "spiritualism": 0.0}
    n = len(member_ids)
    c = sum(stats.get(m, {}).get("empathy", 0.5) - stats.get(m, {}).get("hoarding", 0.5)
            for m in member_ids) / n
    e = sum(stats.get(m, {}).get("resolve", 0.5) - stats.get(m, {}).get("paranoia", 0.5)
            for m in member_ids) / n
    s = sum(stats.get(m, {}).get("faith", 0.5) for m in member_ids) / n
    return {
        "collectivism": max(-1.0, min(1.0, c)),
        "expansionism": max(-1.0, min(1.0, e)),
        "spiritualism": max(0.0, min(1.0, s)),
    }


def ideology_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Euclidean distance between two ideology vectors."""
    dc = (a.get("collectivism", 0.0) - b.get("collectivism", 0.0)) ** 2
    de = (a.get("expansionism", 0.0) - b.get("expansionism", 0.0)) ** 2
    ds = (a.get("spiritualism", 0.0) - b.get("spiritualism", 0.0)) ** 2
    return math.sqrt(dc + de + ds)


def _colonist_ideology(stats: dict[str, float]) -> dict[str, float]:
    """Individual ideology from stats."""
    return {
        "collectivism": max(-1.0, min(1.0, stats.get("empathy", 0.5) - stats.get("hoarding", 0.5))),
        "expansionism": max(-1.0, min(1.0, stats.get("resolve", 0.5) - stats.get("paranoia", 0.5))),
        "spiritualism": max(0.0, min(1.0, stats.get("faith", 0.5))),
    }


# ---------------------------------------------------------------------------
# Cluster detection (deterministic, sorted)
# ---------------------------------------------------------------------------
def _avg_mutual_trust(group: list[str],
                      ctx: DiplomacyYearContext) -> float:
    """Average pairwise trust in a group."""
    if len(group) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, a in enumerate(group):
        for b in group[i + 1:]:
            total += ctx.trust_between(a, b)
            total += ctx.trust_between(b, a)
            count += 2
    return total / max(1, count)


def _ideology_variance(member_ids: list[str],
                       stats: dict[str, dict[str, float]],
                       centroid: dict[str, float]) -> float:
    """Max ideology distance from centroid among members."""
    if not member_ids:
        return 0.0
    return max(ideology_distance(_colonist_ideology(stats.get(m, {})), centroid)
               for m in member_ids)


def detect_clusters(ctx: DiplomacyYearContext,
                    already_assigned: set[str]) -> list[list[str]]:
    """Find trust clusters among unassigned colonists.

    Deterministic: sorted IDs, greedy seed-then-grow.
    """
    available = sorted(set(ctx.active_colonist_ids) - already_assigned)
    if len(available) < MIN_FACTION_SIZE:
        return []

    clusters: list[list[str]] = []
    used: set[str] = set()

    for seed_id in available:
        if seed_id in used:
            continue
        # Find all available colonists with high trust to seed
        candidates = sorted([
            cid for cid in available
            if cid != seed_id and cid not in used
            and (ctx.trust_between(seed_id, cid) + ctx.trust_between(cid, seed_id)) / 2 > TRUST_THRESHOLD
        ])
        group = [seed_id]
        seed_ideo = _colonist_ideology(ctx.colonist_stats.get(seed_id, {}))
        for cid in candidates:
            cid_ideo = _colonist_ideology(ctx.colonist_stats.get(cid, {}))
            if ideology_distance(seed_ideo, cid_ideo) < STAT_SIMILARITY_THRESHOLD:
                # Check avg trust with existing group
                trusts = [(ctx.trust_between(cid, m) + ctx.trust_between(m, cid)) / 2
                          for m in group]
                if sum(trusts) / len(trusts) > TRUST_THRESHOLD:
                    group.append(cid)

        if len(group) >= MIN_FACTION_SIZE:
            clusters.append(group)
            used.update(group)

    return clusters


# ---------------------------------------------------------------------------
# Reconciliation (handle deaths, exits, leader succession)
# ---------------------------------------------------------------------------
def reconcile_factions(state: DiplomacyState,
                       active_ids: set[str],
                       stats: dict[str, dict[str, float]],
                       year: int,
                       result: DiplomacyTickResult) -> None:
    """Remove inactive members, handle succession, dissolve empty factions."""
    for fid in list(state.active_factions()):
        faction = state.factions[fid]
        # Remove inactive members
        before = len(faction.member_ids)
        faction.member_ids = [m for m in faction.member_ids if m in active_ids]

        if len(faction.member_ids) < MIN_FACTION_SIZE:
            countdown = state.dissolution_countdown.get(fid, 0) + 1
            state.dissolution_countdown[fid] = countdown
            if countdown >= FACTION_DISSOLUTION_GRACE:
                faction.dissolved_year = year
                state.dissolution_countdown.pop(fid, None)
                result.factions_dissolved.append({
                    "id": fid, "name": faction.name, "year": year,
                    "reason": "too_few_members"})
                state.history.append({
                    "type": "dissolution", "faction": fid,
                    "year": year, "reason": "too_few_members"})
                continue
        else:
            state.dissolution_countdown.pop(fid, None)

        # Leader succession
        if faction.leader_id not in faction.member_ids and faction.member_ids:
            sorted_members = sorted(
                faction.member_ids,
                key=lambda m: stats.get(m, {}).get("resolve", 0.0),
                reverse=True)
            faction.leader_id = sorted_members[0]
            state.history.append({
                "type": "succession", "faction": fid,
                "new_leader": faction.leader_id, "year": year})

        # Update ideology
        faction.ideology = compute_ideology(faction.member_ids, stats)


# ---------------------------------------------------------------------------
# Faction formation (match existing or create new)
# ---------------------------------------------------------------------------
def _match_existing(cluster: list[str],
                    state: DiplomacyState) -> str | None:
    """Find an active faction with high member overlap (Jaccard)."""
    cluster_set = set(cluster)
    best_fid: str | None = None
    best_jaccard = 0.0
    for fid, faction in state.active_factions().items():
        member_set = set(faction.member_ids)
        intersection = len(cluster_set & member_set)
        union = len(cluster_set | member_set)
        if union == 0:
            continue
        jaccard = intersection / union
        if jaccard > best_jaccard:
            best_jaccard = jaccard
            best_fid = fid
    if best_jaccard >= OVERLAP_MATCH_THRESHOLD:
        return best_fid
    return None


def _next_faction_name(state: DiplomacyState) -> str:
    """Get next available faction name."""
    name = FACTION_NAMES[state.name_cursor % len(FACTION_NAMES)]
    state.name_cursor += 1
    return name


def form_factions(clusters: list[list[str]],
                  ctx: DiplomacyYearContext,
                  state: DiplomacyState,
                  result: DiplomacyTickResult) -> None:
    """Create new factions from detected clusters."""
    _ensure_next_id(state)
    for cluster in clusters:
        existing = _match_existing(cluster, state)
        if existing is not None:
            # Update existing faction membership
            faction = state.factions[existing]
            faction.member_ids = list(cluster)
            faction.ideology = compute_ideology(cluster, ctx.colonist_stats)
            continue

        fid = f"faction-{state.next_faction_id}"
        state.next_faction_id += 1
        ideology = compute_ideology(cluster, ctx.colonist_stats)
        leader = max(cluster,
                     key=lambda m: ctx.colonist_stats.get(m, {}).get("resolve", 0.0))
        name = _next_faction_name(state)
        faction = Faction(
            id=fid, name=name, leader_id=leader,
            member_ids=list(cluster), ideology=ideology,
            formed_year=ctx.year)
        state.factions[fid] = faction
        result.factions_formed.append(faction.to_dict())
        state.history.append({
            "type": "formation", "faction": fid,
            "name": name, "year": ctx.year,
            "members": list(cluster)})


# ---------------------------------------------------------------------------
# Schism detection
# ---------------------------------------------------------------------------
def _ensure_next_id(state: DiplomacyState) -> None:
    """Ensure next_faction_id won't collide with existing factions."""
    for fid in state.factions:
        if fid.startswith("faction-"):
            try:
                num = int(fid.split("-", 1)[1])
                if num >= state.next_faction_id:
                    state.next_faction_id = num + 1
            except ValueError:
                pass


def check_schisms(state: DiplomacyState,
                  ctx: DiplomacyYearContext,
                  rng: random.Random,
                  result: DiplomacyTickResult) -> None:
    """Split factions with high internal ideology variance."""
    _ensure_next_id(state)
    for fid in list(state.active_factions()):
        faction = state.factions[fid]
        if len(faction.member_ids) < SCHISM_MIN_SIZE:
            continue
        if ctx.year - faction.last_schism_year < COOLDOWN_YEARS:
            continue

        centroid = faction.ideology
        variance = _ideology_variance(
            faction.member_ids, ctx.colonist_stats, centroid)

        if variance <= SCHISM_VARIANCE_THRESHOLD:
            continue

        # Split: partition members by distance from centroid
        members_with_dist = sorted(
            [(m, ideology_distance(
                _colonist_ideology(ctx.colonist_stats.get(m, {})), centroid))
             for m in faction.member_ids],
            key=lambda x: x[1], reverse=True)

        # Farthest members form the splinter
        split_point = len(members_with_dist) // 2
        splinter = [m for m, _ in members_with_dist[:split_point]]
        remaining = [m for m, _ in members_with_dist[split_point:]]

        if len(splinter) < MIN_FACTION_SIZE or len(remaining) < MIN_FACTION_SIZE:
            continue  # record dissent but don't split

        # Update original faction
        faction.member_ids = remaining
        faction.ideology = compute_ideology(remaining, ctx.colonist_stats)
        faction.last_schism_year = ctx.year

        # Create splinter faction
        sfid = f"faction-{state.next_faction_id}"
        state.next_faction_id += 1
        s_ideology = compute_ideology(splinter, ctx.colonist_stats)
        s_leader = max(splinter,
                       key=lambda m: ctx.colonist_stats.get(m, {}).get("resolve", 0.0))
        s_name = _next_faction_name(state)
        splinter_faction = Faction(
            id=sfid, name=s_name, leader_id=s_leader,
            member_ids=splinter, ideology=s_ideology,
            formed_year=ctx.year, last_schism_year=ctx.year)
        state.factions[sfid] = splinter_faction

        result.schisms.append({
            "original": fid, "splinter": sfid,
            "year": ctx.year, "remaining": remaining, "departed": splinter})
        state.history.append({
            "type": "schism", "original": fid, "splinter": sfid,
            "year": ctx.year})


# ---------------------------------------------------------------------------
# Alliances and rivalries
# ---------------------------------------------------------------------------
def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Return sorted pair."""
    return (min(a, b), max(a, b))


def update_alliances(state: DiplomacyState,
                     ctx: DiplomacyYearContext,
                     rng: random.Random,
                     result: DiplomacyTickResult) -> None:
    """Form or break alliances between active factions."""
    active = state.active_factions()
    active_ids = sorted(active.keys())

    # Check existing alliances for breakage
    surviving: list[Alliance] = []
    for alliance in state.alliances:
        fa, fb = alliance.faction_a, alliance.faction_b
        if fa not in active or fb not in active:
            result.alliances_broken.append(alliance.to_dict())
            state.history.append({
                "type": "alliance_broken", "pair": [fa, fb],
                "year": ctx.year, "reason": "faction_dissolved"})
            continue
        tenure = ctx.year - alliance.formed_year
        if tenure < ALLIANCE_MIN_TENURE:
            surviving.append(alliance)
            continue
        dist = ideology_distance(active[fa].ideology, active[fb].ideology)
        if dist > ALLIANCE_IDEOLOGY_DIST * 1.5:
            # Ideology drifted too far
            result.alliances_broken.append(alliance.to_dict())
            state.history.append({
                "type": "alliance_broken", "pair": [fa, fb],
                "year": ctx.year, "reason": "ideology_drift"})
        else:
            # Decay strength slightly
            alliance.strength = max(0.1, alliance.strength - 0.02)
            surviving.append(alliance)
    state.alliances = surviving

    # Check for new alliances
    existing_pairs = {a.pair() for a in state.alliances}
    for i, fa_id in enumerate(active_ids):
        for fb_id in active_ids[i + 1:]:
            pair = _canonical_pair(fa_id, fb_id)
            if pair in existing_pairs:
                continue
            dist = ideology_distance(active[fa_id].ideology,
                                     active[fb_id].ideology)
            if dist < ALLIANCE_IDEOLOGY_DIST:
                prob = (ALLIANCE_IDEOLOGY_DIST - dist) / ALLIANCE_IDEOLOGY_DIST * 0.3
                if rng.random() < prob:
                    alliance = Alliance(
                        faction_a=pair[0], faction_b=pair[1],
                        strength=1.0 - dist / ALLIANCE_IDEOLOGY_DIST,
                        formed_year=ctx.year)
                    state.alliances.append(alliance)
                    result.alliances_formed.append(alliance.to_dict())
                    state.history.append({
                        "type": "alliance_formed", "pair": list(pair),
                        "year": ctx.year})

    # Update rivalries
    surviving_rivalries: list[Rivalry] = []
    for rivalry in state.rivalries:
        fa, fb = rivalry.faction_a, rivalry.faction_b
        if fa not in active or fb not in active:
            result.rivalries_ended.append(rivalry.to_dict())
            continue
        dist = ideology_distance(active[fa].ideology, active[fb].ideology)
        if dist < RIVALRY_IDEOLOGY_DIST * 0.7:
            result.rivalries_ended.append(rivalry.to_dict())
            state.history.append({
                "type": "rivalry_ended", "pair": [fa, fb],
                "year": ctx.year, "reason": "ideology_convergence"})
        else:
            rivalry.intensity = min(1.0, rivalry.intensity + 0.02)
            surviving_rivalries.append(rivalry)
    state.rivalries = surviving_rivalries

    # New rivalries
    rivalry_pairs = {r.pair() for r in state.rivalries}
    for i, fa_id in enumerate(active_ids):
        for fb_id in active_ids[i + 1:]:
            pair = _canonical_pair(fa_id, fb_id)
            if pair in rivalry_pairs or pair in existing_pairs:
                continue
            dist = ideology_distance(active[fa_id].ideology,
                                     active[fb_id].ideology)
            if dist > RIVALRY_IDEOLOGY_DIST:
                prob = (dist - RIVALRY_IDEOLOGY_DIST) * 0.2
                if rng.random() < prob:
                    rivalry = Rivalry(
                        faction_a=pair[0], faction_b=pair[1],
                        intensity=min(1.0, (dist - RIVALRY_IDEOLOGY_DIST) * 0.5),
                        formed_year=ctx.year)
                    state.rivalries.append(rivalry)
                    result.rivalries_formed.append(rivalry.to_dict())
                    state.history.append({
                        "type": "rivalry_formed", "pair": list(pair),
                        "year": ctx.year})


# ---------------------------------------------------------------------------
# Power and governance pressure
# ---------------------------------------------------------------------------
def compute_power_balance(state: DiplomacyState,
                          ctx: DiplomacyYearContext) -> dict[str, float]:
    """Compute power for each active faction."""
    active = state.active_factions()
    total_pop = max(1, len(ctx.active_colonist_ids))
    balance: dict[str, float] = {}
    for fid, faction in active.items():
        raw_power = sum(
            ctx.colonist_stats.get(m, {}).get("resolve", 0.0) * 0.4
            + ctx.colonist_stats.get(m, {}).get("empathy", 0.0) * 0.3
            + ctx.colonist_stats.get(m, {}).get("improvisation", 0.0) * 0.3
            for m in faction.member_ids)
        balance[fid] = raw_power / total_pop
        faction.power = balance[fid]
    return balance


def compute_governance_pressure(state: DiplomacyState) -> dict[str, float]:
    """Compute governance-type pressure from faction ideologies.

    Returns normalized pressure values for each governance type.
    Collectivist factions push for direct_democracy.
    Individualist factions push for council.
    Spiritual factions push for theocracy.
    """
    active = state.active_factions()
    if not active:
        return {}
    raw: dict[str, float] = {}
    for faction in active.values():
        c = faction.ideology.get("collectivism", 0.0)
        s = faction.ideology.get("spiritualism", 0.0)
        p = faction.power
        if c > 0.2:
            raw["direct_democracy"] = raw.get("direct_democracy", 0.0) + c * p
        elif c < -0.2:
            raw["council"] = raw.get("council", 0.0) + abs(c) * p
        if s > 0.5:
            raw["theocracy"] = raw.get("theocracy", 0.0) + s * p

    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: min(1.0, v / total) for k, v in raw.items()}


def compute_psych_modifiers(state: DiplomacyState,
                            active_ids: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """Faction membership reduces loneliness, boosts purpose.

    Returns (loneliness_mods, purpose_mods) — transient yearly adjustments.
    """
    loneliness: dict[str, float] = {}
    purpose: dict[str, float] = {}
    active = state.active_factions()
    for faction in active.values():
        size_bonus = min(0.08, len(faction.member_ids) * 0.01)
        for mid in faction.member_ids:
            loneliness[mid] = -size_bonus
            purpose[mid] = size_bonus * 0.8
    return loneliness, purpose


# ---------------------------------------------------------------------------
# Main tick
# ---------------------------------------------------------------------------
def tick_diplomacy(state: DiplomacyState,
                   ctx: DiplomacyYearContext,
                   rng: random.Random) -> DiplomacyTickResult:
    """Run one year of diplomacy. Pure function on state + context."""
    result = DiplomacyTickResult()
    active_set = set(ctx.active_colonist_ids)

    # Phase 1: Reconcile (deaths, exits, succession)
    reconcile_factions(state, active_set, ctx.colonist_stats, ctx.year, result)

    # Phase 2: Detect new clusters among unaffiliated colonists
    assigned = set()
    for f in state.active_factions().values():
        assigned.update(f.member_ids)
    clusters = detect_clusters(ctx, assigned)
    form_factions(clusters, ctx, state, result)

    # Phase 3: Check for schisms
    check_schisms(state, ctx, rng, result)

    # Phase 4: Alliances and rivalries
    update_alliances(state, ctx, rng, result)

    # Phase 5: Power balance and governance pressure
    result.power_balance = compute_power_balance(state, ctx)
    result.governance_pressure = compute_governance_pressure(state)

    # Phase 6: Psych modifiers
    loneliness_mods, purpose_mods = compute_psych_modifiers(
        state, ctx.active_colonist_ids)
    result.loneliness_modifiers = loneliness_mods
    result.purpose_modifiers = purpose_mods

    # Prune history
    state.history = state.history[-MAX_HISTORY:]

    return result
