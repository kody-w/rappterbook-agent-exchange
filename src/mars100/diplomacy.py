"""
Diplomacy organ for Mars-100 (engine v11.0).

Models inter-faction negotiations, treaties, and diplomatic crises.
Factions emerge from social-graph clustering.  Treaties modulate economics,
governance, and conflict.  One-year lag: THIS year's diplomacy drives NEXT
year's action-weight modifiers.

RNG offset: seed + 12553
"""
from __future__ import annotations

import random as _random_module
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FACTION_SIZE = 3
FACTION_RECOMPUTE_INTERVAL = 10
TRUST_CLUSTER_THRESHOLD = 0.45
TREATY_PROPOSE_TRUST = 0.40
TREATY_VIOLATION_LIMIT = 3
TREATY_DEFAULT_DURATION = 20

# Action-weight pressure caps (combined cap with other pressure sources)
DIPLO_PRESSURE_CAP = 0.20

# Resource-sharing treaty bonus per active treaty (lagged one year)
SHARING_EFFICIENCY_BONUS = 0.005
KNOWLEDGE_RESEARCH_BONUS = 0.010
DEFENSE_SABOTAGE_PENALTY = 0.15

FACTION_NAMES = [
    "The Architects", "The Stewards", "The Seekers", "The Wardens",
    "The Pioneers", "The Mediators", "The Watchmen", "The Dreamers",
    "The Builders", "The Sentinels",
]

TREATY_TYPES = ("resource_sharing", "knowledge_exchange",
                "mutual_defense", "non_aggression")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Faction:
    """A cluster of colonists with shared values and high mutual trust."""
    id: str
    name: str
    formed_year: int
    member_ids: list[str] = field(default_factory=list)
    dominant_value: str = "resolve"
    cohesion: float = 0.5
    ambassador_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "formed_year": self.formed_year,
            "member_ids": self.member_ids, "dominant_value": self.dominant_value,
            "cohesion": self.cohesion, "ambassador_id": self.ambassador_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Faction:
        return cls(
            id=d["id"], name=d["name"], formed_year=d.get("formed_year", 0),
            member_ids=d.get("member_ids", []),
            dominant_value=d.get("dominant_value", "resolve"),
            cohesion=d.get("cohesion", 0.5),
            ambassador_id=d.get("ambassador_id"),
        )


@dataclass
class Treaty:
    """A bilateral agreement between two factions."""
    id: str
    treaty_type: str
    party_a: str
    party_b: str
    formed_year: int
    expires_year: int | None = None
    violations: int = 0
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "treaty_type": self.treaty_type,
            "party_a": self.party_a, "party_b": self.party_b,
            "formed_year": self.formed_year, "expires_year": self.expires_year,
            "violations": self.violations, "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Treaty:
        return cls(
            id=d["id"], treaty_type=d["treaty_type"],
            party_a=d["party_a"], party_b=d["party_b"],
            formed_year=d.get("formed_year", 0),
            expires_year=d.get("expires_year"),
            violations=d.get("violations", 0),
            status=d.get("status", "active"),
        )


@dataclass
class DiplomacyState:
    """Full diplomatic state of the colony."""
    factions: dict[str, Faction] = field(default_factory=dict)
    treaties: dict[str, Treaty] = field(default_factory=dict)
    faction_membership: dict[str, str] = field(default_factory=dict)
    last_recompute_year: int = 0
    next_faction_id: int = 0
    crises_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factions": {k: v.to_dict() for k, v in self.factions.items()},
            "treaties": {k: v.to_dict() for k, v in self.treaties.items()},
            "faction_membership": dict(self.faction_membership),
            "last_recompute_year": self.last_recompute_year,
            "next_faction_id": self.next_faction_id,
            "crises_log": list(self.crises_log),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiplomacyState:
        state = cls()
        for k, v in d.get("factions", {}).items():
            state.factions[k] = Faction.from_dict(v)
        for k, v in d.get("treaties", {}).items():
            state.treaties[k] = Treaty.from_dict(v)
        state.faction_membership = dict(d.get("faction_membership", {}))
        state.last_recompute_year = d.get("last_recompute_year", 0)
        state.next_faction_id = d.get("next_faction_id", 0)
        state.crises_log = list(d.get("crises_log", []))
        return state

    def active_treaties(self) -> list[Treaty]:
        """Return all treaties with status 'active'."""
        return [t for t in self.treaties.values() if t.status == "active"]

    def faction_of(self, colonist_id: str) -> Faction | None:
        """Look up which faction a colonist belongs to."""
        fid = self.faction_membership.get(colonist_id)
        return self.factions.get(fid) if fid else None


@dataclass
class DiplomacyTickResult:
    """Output of one year's diplomacy tick."""
    factions_formed: list[dict] = field(default_factory=list)
    factions_dissolved: list[dict] = field(default_factory=list)
    treaties_proposed: list[dict] = field(default_factory=list)
    treaties_dissolved: list[dict] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    crises: list[dict] = field(default_factory=list)
    pressure: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factions_formed": self.factions_formed,
            "factions_dissolved": self.factions_dissolved,
            "treaties_proposed": self.treaties_proposed,
            "treaties_dissolved": self.treaties_dissolved,
            "violations": self.violations,
            "crises": self.crises,
            "pressure": self.pressure,
        }


# ---------------------------------------------------------------------------
# Stat names (imported inline to avoid circular deps)
# ---------------------------------------------------------------------------

_STAT_NAMES = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")


# ---------------------------------------------------------------------------
# Faction detection
# ---------------------------------------------------------------------------


def _dominant_stat(colonist: Any) -> str:
    """Return the name of a colonist's highest stat."""
    return max(_STAT_NAMES, key=lambda s: getattr(colonist.stats, s, 0.0))


def detect_factions(
    colonists: list[Any],
    social: Any,
    state: DiplomacyState,
    year: int,
    rng: _random_module.Random,
) -> list[dict]:
    """Detect factions from social-graph clustering.

    Groups colonists by dominant stat, then validates internal trust.
    Preserves existing faction IDs when membership overlaps > 50%.
    Returns list of newly-formed faction dicts for logging.
    """
    active = sorted(
        [c for c in colonists if c.is_active()], key=lambda c: c.id)
    if len(active) < MIN_FACTION_SIZE * 2:
        return []

    # Cluster by dominant stat
    clusters: dict[str, list[str]] = {}
    for c in active:
        ds = _dominant_stat(c)
        clusters.setdefault(ds, []).append(c.id)

    # Merge small clusters into nearest large cluster by avg trust
    final_clusters: list[list[str]] = []
    overflow: list[str] = []
    for stat, members in sorted(clusters.items()):
        if len(members) >= MIN_FACTION_SIZE:
            final_clusters.append(sorted(members))
        else:
            overflow.extend(members)

    # Assign overflow members to cluster with highest avg trust
    for orphan in sorted(overflow):
        best_cluster = None
        best_trust = -1.0
        for cluster in final_clusters:
            avg_t = sum(
                social.get(orphan, cid).trust
                for cid in cluster if cid != orphan
            ) / max(1, len(cluster))
            if avg_t > best_trust:
                best_trust = avg_t
                best_cluster = cluster
        if best_cluster is not None:
            best_cluster.append(orphan)
        elif final_clusters:
            final_clusters[0].append(orphan)

    if not final_clusters:
        return []

    # Match new clusters to existing factions (>50% overlap keeps ID)
    old_factions = dict(state.factions)
    new_factions: dict[str, Faction] = {}
    used_names: set[str] = {f.name for f in old_factions.values()}
    formed: list[dict] = []

    for cluster in final_clusters:
        cluster_set = set(cluster)
        best_match: str | None = None
        best_overlap = 0.0
        for fid, faction in old_factions.items():
            old_set = set(faction.member_ids)
            if not old_set:
                continue
            overlap = len(cluster_set & old_set) / len(old_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = fid

        if best_match and best_overlap > 0.5:
            faction = old_factions[best_match]
            faction.member_ids = sorted(cluster)
            faction.dominant_value = _cluster_dominant(colonists, cluster)
            faction.cohesion = _cluster_cohesion(social, cluster)
            new_factions[best_match] = faction
            used_names.add(faction.name)
        else:
            fid = f"faction-{state.next_faction_id}"
            state.next_faction_id += 1
            name = _pick_name(used_names, rng)
            used_names.add(name)
            dominant = _cluster_dominant(colonists, cluster)
            cohesion = _cluster_cohesion(social, cluster)
            ambassador = _pick_ambassador(colonists, cluster)
            faction = Faction(
                id=fid, name=name, formed_year=year,
                member_ids=sorted(cluster), dominant_value=dominant,
                cohesion=cohesion, ambassador_id=ambassador,
            )
            new_factions[fid] = faction
            formed.append(faction.to_dict())

    # Update state
    state.factions = new_factions
    state.faction_membership = {}
    for fid, faction in new_factions.items():
        for mid in faction.member_ids:
            state.faction_membership[mid] = fid
    state.last_recompute_year = year

    return formed


def maintain_factions(
    colonists: list[Any],
    social: Any,
    state: DiplomacyState,
    year: int,
    rng: _random_module.Random,
) -> None:
    """Annual lightweight maintenance: prune dead, assign newcomers."""
    active_ids = {c.id for c in colonists if c.is_active()}

    # Prune dead/exiled members
    for faction in state.factions.values():
        faction.member_ids = sorted(
            mid for mid in faction.member_ids if mid in active_ids)
        if faction.ambassador_id and faction.ambassador_id not in active_ids:
            faction.ambassador_id = (
                _pick_ambassador(colonists, faction.member_ids)
                if faction.member_ids else None)
        faction.cohesion = _cluster_cohesion(social, faction.member_ids)

    # Assign unaffiliated colonists to nearest faction by trust
    affiliated = set(state.faction_membership.keys()) & active_ids
    unaffiliated = sorted(active_ids - affiliated)
    for cid in unaffiliated:
        best_fid: str | None = None
        best_trust = -1.0
        for fid, faction in state.factions.items():
            if not faction.member_ids:
                continue
            avg_t = sum(
                social.get(cid, mid).trust
                for mid in faction.member_ids if mid != cid
            ) / max(1, len(faction.member_ids))
            if avg_t > best_trust:
                best_trust = avg_t
                best_fid = fid
        if best_fid:
            state.factions[best_fid].member_ids.append(cid)
            state.factions[best_fid].member_ids.sort()
            state.faction_membership[cid] = best_fid

    # Remove dead membership entries
    dead_ids = set(state.faction_membership.keys()) - active_ids
    for did in dead_ids:
        del state.faction_membership[did]

    # Dissolve empty factions
    empty = [fid for fid, f in state.factions.items() if not f.member_ids]
    for fid in empty:
        del state.factions[fid]


# ---------------------------------------------------------------------------
# Treaties
# ---------------------------------------------------------------------------


def propose_treaties(
    state: DiplomacyState,
    social: Any,
    year: int,
    rng: _random_module.Random,
) -> list[dict]:
    """Propose treaties between factions with sufficient inter-faction trust."""
    proposed: list[dict] = []
    faction_ids = sorted(state.factions.keys())
    existing_pairs: set[tuple[str, str]] = set()
    for t in state.active_treaties():
        existing_pairs.add((t.party_a, t.party_b))
        existing_pairs.add((t.party_b, t.party_a))

    for i, fid_a in enumerate(faction_ids):
        for fid_b in faction_ids[i + 1:]:
            if (fid_a, fid_b) in existing_pairs:
                continue
            fa = state.factions[fid_a]
            fb = state.factions[fid_b]
            avg_trust = _inter_faction_trust(social, fa.member_ids, fb.member_ids)
            if avg_trust < TREATY_PROPOSE_TRUST:
                continue
            if rng.random() > 0.25:
                continue
            treaty_type = rng.choice(TREATY_TYPES)
            tid = f"treaty-y{year}-{fid_a}-{fid_b}"
            treaty = Treaty(
                id=tid, treaty_type=treaty_type,
                party_a=fid_a, party_b=fid_b,
                formed_year=year,
                expires_year=year + TREATY_DEFAULT_DURATION,
            )
            state.treaties[tid] = treaty
            existing_pairs.add((fid_a, fid_b))
            proposed.append(treaty.to_dict())

    return proposed


def check_violations(
    state: DiplomacyState,
    actions: dict[str, str],
    social: Any,
    year: int,
) -> tuple[list[dict], list[dict]]:
    """Check for treaty violations. Returns (violations, dissolved)."""
    violations: list[dict] = []
    dissolved: list[dict] = []

    for treaty in list(state.active_treaties()):
        fa = state.factions.get(treaty.party_a)
        fb = state.factions.get(treaty.party_b)
        if not fa or not fb:
            treaty.status = "dissolved_no_party"
            dissolved.append(treaty.to_dict())
            continue

        violated = False
        if treaty.treaty_type == "non_aggression":
            # Sabotage against treaty partner members is a violation
            for mid in fa.member_ids:
                if actions.get(mid) == "sabotage":
                    violated = True
                    break
            if not violated:
                for mid in fb.member_ids:
                    if actions.get(mid) == "sabotage":
                        violated = True
                        break
        elif treaty.treaty_type == "mutual_defense":
            # Not cooperating during crisis (low resources) is a violation
            # Checked via hoard action during resource scarcity
            a_hoarding = sum(1 for m in fa.member_ids if actions.get(m) == "hoard")
            b_hoarding = sum(1 for m in fb.member_ids if actions.get(m) == "hoard")
            if a_hoarding > len(fa.member_ids) // 2 or b_hoarding > len(fb.member_ids) // 2:
                violated = True

        if violated:
            treaty.violations += 1
            violations.append({
                "treaty_id": treaty.id, "year": year,
                "type": treaty.treaty_type, "violations": treaty.violations,
            })
            if treaty.violations >= TREATY_VIOLATION_LIMIT:
                treaty.status = "dissolved_violations"
                dissolved.append(treaty.to_dict())
                # Trust penalty between factions
                for mid_a in fa.member_ids:
                    for mid_b in fb.member_ids:
                        social.update_from_conflict(mid_a, mid_b,
                                                     _random_module.Random(year))

        # Expire old treaties
        if treaty.expires_year and year >= treaty.expires_year:
            treaty.status = "expired"
            dissolved.append(treaty.to_dict())

    return violations, dissolved


# ---------------------------------------------------------------------------
# Diplomatic pressure (lagged — computed this year, applied next year)
# ---------------------------------------------------------------------------


def compute_diplomatic_pressure(state: DiplomacyState) -> dict[str, float]:
    """Compute action-weight modifiers from active treaties.

    Returns a dict mapping action names to weight adjustments.
    Applied with one-year lag by the engine.
    """
    pressure: dict[str, float] = {}
    active = state.active_treaties()
    if not active:
        return pressure

    for treaty in active:
        if treaty.treaty_type == "resource_sharing":
            pressure["cooperate"] = pressure.get("cooperate", 0.0) + 0.05
            pressure["hoard"] = pressure.get("hoard", 0.0) - 0.05
        elif treaty.treaty_type == "knowledge_exchange":
            pressure["research"] = pressure.get("research", 0.0) + 0.08
            pressure["code"] = pressure.get("code", 0.0) + 0.03
        elif treaty.treaty_type == "mutual_defense":
            pressure["cooperate"] = pressure.get("cooperate", 0.0) + 0.03
            pressure["sabotage"] = pressure.get("sabotage", 0.0) - 0.10
        elif treaty.treaty_type == "non_aggression":
            pressure["sabotage"] = pressure.get("sabotage", 0.0) - 0.15
            pressure["mediate"] = pressure.get("mediate", 0.0) + 0.05

    # Clamp
    for k in pressure:
        pressure[k] = max(-DIPLO_PRESSURE_CAP, min(DIPLO_PRESSURE_CAP, pressure[k]))
    return pressure


def compute_resource_modifiers(state: DiplomacyState) -> dict[str, float]:
    """Compute resource modifiers from treaties (lagged one year).

    resource_sharing → food/water spoilage reduced
    knowledge_exchange → power production bonus
    """
    mods: dict[str, float] = {}
    for treaty in state.active_treaties():
        if treaty.treaty_type == "resource_sharing":
            mods["food_spoilage_mult"] = mods.get("food_spoilage_mult", 1.0) - 0.1
            mods["water_spoilage_mult"] = mods.get("water_spoilage_mult", 1.0) - 0.1
        elif treaty.treaty_type == "knowledge_exchange":
            # Slightly boost research effectiveness
            pass  # Handled via action pressure, not production
    # Clamp spoilage mults to [0.5, 1.0]
    for k in ("food_spoilage_mult", "water_spoilage_mult"):
        if k in mods:
            mods[k] = max(0.5, min(1.0, mods[k]))
    return mods


# ---------------------------------------------------------------------------
# Diplomatic crises
# ---------------------------------------------------------------------------


def check_crises(
    state: DiplomacyState,
    resource_avg: float,
    event_severity: float,
    year: int,
    rng: _random_module.Random,
) -> list[dict]:
    """Generate diplomatic crises from resource scarcity or events."""
    crises: list[dict] = []
    if len(state.factions) < 2:
        return crises

    # Resource scarcity crisis: who gets the last ration?
    if resource_avg < 0.25 and rng.random() < 0.3:
        faction_ids = sorted(state.factions.keys())
        if len(faction_ids) >= 2:
            involved = rng.sample(faction_ids, 2)
            crisis = {
                "year": year, "type": "resource_scarcity",
                "factions": involved,
                "desc": "Resource scarcity forces hard allocation choices",
            }
            crises.append(crisis)
            state.crises_log.append(crisis)

    # Major event triggers inter-faction tension
    if event_severity > 0.7 and rng.random() < 0.2:
        faction_ids = sorted(state.factions.keys())
        if len(faction_ids) >= 2:
            involved = rng.sample(faction_ids, 2)
            crisis = {
                "year": year, "type": "event_tension",
                "factions": involved,
                "desc": "Major event creates inter-faction blame",
            }
            crises.append(crisis)
            state.crises_log.append(crisis)

    return crises


# ---------------------------------------------------------------------------
# Governance integration: bloc voting
# ---------------------------------------------------------------------------


def compute_bloc_vote_bias(
    colonist_id: str,
    proposer_id: str,
    state: DiplomacyState,
) -> float:
    """Return a voting bias based on faction alignment.

    Same faction as proposer → +0.15
    Allied faction (treaty exists) → +0.08
    Rival faction (crisis history) → -0.10
    """
    c_faction = state.faction_membership.get(colonist_id)
    p_faction = state.faction_membership.get(proposer_id)
    if not c_faction or not p_faction:
        return 0.0
    if c_faction == p_faction:
        return 0.15

    # Check for alliance (any active treaty between factions)
    for treaty in state.active_treaties():
        parties = {treaty.party_a, treaty.party_b}
        if c_faction in parties and p_faction in parties:
            return 0.08

    # Check for rivalry (recent crisis involving both)
    recent_crises = [
        cr for cr in state.crises_log
        if cr.get("year", 0) > 0
        and c_faction in cr.get("factions", [])
        and p_faction in cr.get("factions", [])
    ]
    if recent_crises:
        return -0.10

    return 0.0


# ---------------------------------------------------------------------------
# Main tick
# ---------------------------------------------------------------------------


def tick_diplomacy(
    state: DiplomacyState,
    colonists: list[Any],
    social: Any,
    actions: dict[str, str],
    resource_avg: float,
    event_severity: float,
    year: int,
    rng: _random_module.Random,
) -> DiplomacyTickResult:
    """Advance diplomacy by one year."""
    result = DiplomacyTickResult()

    # Annual maintenance
    maintain_factions(colonists, social, state, year, rng)

    # Structural recompute every N years or during high stress
    should_recompute = (
        year - state.last_recompute_year >= FACTION_RECOMPUTE_INTERVAL
        or (resource_avg < 0.25 and year - state.last_recompute_year >= 3)
        or not state.factions
    )
    # Don't recompute before year 5
    if should_recompute and year >= 5:
        formed = detect_factions(colonists, social, state, year, rng)
        result.factions_formed = formed

    # Propose new treaties
    if state.factions and year >= 5:
        proposed = propose_treaties(state, social, year, rng)
        result.treaties_proposed = proposed

    # Check treaty violations
    violations, dissolved = check_violations(state, actions, social, year)
    result.violations = violations
    result.treaties_dissolved = dissolved

    # Check for crises
    crises = check_crises(state, resource_avg, event_severity, year, rng)
    result.crises = crises

    # Compute pressure for next year (lagged)
    result.pressure = compute_diplomatic_pressure(state)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cluster_dominant(colonists: list[Any], member_ids: list[str]) -> str:
    """Find the dominant stat across a cluster of colonists."""
    totals: dict[str, float] = {s: 0.0 for s in _STAT_NAMES}
    count = 0
    for c in colonists:
        if c.id in member_ids and c.is_active():
            for s in _STAT_NAMES:
                totals[s] += getattr(c.stats, s, 0.0)
            count += 1
    if count == 0:
        return "resolve"
    return max(totals, key=lambda s: totals[s])


def _cluster_cohesion(social: Any, member_ids: list[str]) -> float:
    """Average internal trust within a cluster."""
    if len(member_ids) < 2:
        return 1.0
    total = 0.0
    pairs = 0
    for a in member_ids:
        for b in member_ids:
            if a != b:
                total += social.get(a, b).trust
                pairs += 1
    return total / max(1, pairs)


def _inter_faction_trust(
    social: Any,
    members_a: list[str],
    members_b: list[str],
) -> float:
    """Average trust between two faction member sets."""
    if not members_a or not members_b:
        return 0.0
    total = 0.0
    pairs = 0
    for a in members_a:
        for b in members_b:
            total += social.get(a, b).trust
            pairs += 1
    return total / max(1, pairs)


def _pick_name(used: set[str], rng: _random_module.Random) -> str:
    """Pick an unused faction name."""
    available = [n for n in FACTION_NAMES if n not in used]
    if available:
        return rng.choice(available)
    return f"Faction-{rng.randint(100, 999)}"


def _pick_ambassador(colonists: list[Any], member_ids: list[str]) -> str | None:
    """Pick the colonist with highest empathy+mediation as ambassador."""
    best_id: str | None = None
    best_score = -1.0
    for c in colonists:
        if c.id in member_ids and c.is_active():
            score = c.stats.empathy + getattr(c.skills, "mediation", 0.0)
            if score > best_score:
                best_score = score
                best_id = c.id
    return best_id
