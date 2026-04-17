"""
Diplomacy engine for Mars-100.

Factions form organically from social graph clustering (mutual trust).
Colonists negotiate individual treaties — non-aggression, cooperation,
mutual defense.  Betrayals are detected from structured action outcomes
and have severe social consequences.

Factions are stable across years via overlap-based continuation (hysteresis).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


TREATY_TYPES = ("non_aggression", "cooperation", "mutual_defense")
FACTION_TRUST_THRESHOLD = 0.55
FACTION_MIN_SIZE = 2
FACTION_OVERLAP_THRESHOLD = 0.5


@dataclass
class Faction:
    """An organic cluster of colonists with aligned trust."""
    id: str
    member_ids: list[str]
    coherence: float
    dominant_value: str
    formed_year: int
    name: str = ""
    dissolved_year: int | None = None

    def is_active(self) -> bool:
        """Check if this faction is still active."""
        return self.dissolved_year is None

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        d: dict[str, Any] = {
            "id": self.id, "member_ids": self.member_ids,
            "coherence": round(self.coherence, 4),
            "dominant_value": self.dominant_value,
            "formed_year": self.formed_year, "name": self.name,
        }
        if self.dissolved_year is not None:
            d["dissolved_year"] = self.dissolved_year
        return d


@dataclass
class Treaty:
    """A bilateral agreement between two colonists."""
    id: str
    party_a: str
    party_b: str
    treaty_type: str
    year_signed: int
    duration: int
    active: bool = True
    violations: list[dict] = field(default_factory=list)
    year_expired: int | None = None

    def parties(self) -> set[str]:
        """Return both party IDs as a set."""
        return {self.party_a, self.party_b}

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "id": self.id, "party_a": self.party_a, "party_b": self.party_b,
            "treaty_type": self.treaty_type, "year_signed": self.year_signed,
            "duration": self.duration, "active": self.active,
            "violations": list(self.violations),
            "year_expired": self.year_expired,
        }


@dataclass
class ActionOutcome:
    """Structured record of a colonist action with optional target."""
    actor_id: str
    action: str
    target_id: str | None = None

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        d: dict[str, Any] = {"actor_id": self.actor_id, "action": self.action}
        if self.target_id:
            d["target_id"] = self.target_id
        return d


@dataclass
class DiplomacyEvent:
    """A notable diplomacy event in a given year."""
    year: int
    event_type: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {"year": self.year, "event_type": self.event_type,
                "details": self.details}


@dataclass
class DiplomacyState:
    """Colony-wide diplomacy state."""
    factions: list[Faction] = field(default_factory=list)
    treaties: list[Treaty] = field(default_factory=list)
    betrayals: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    next_faction_id: int = 0
    next_treaty_id: int = 0

    def active_factions(self) -> list[Faction]:
        """Return currently active factions."""
        return [f for f in self.factions if f.is_active()]

    def active_treaties(self) -> list[Treaty]:
        """Return currently active treaties."""
        return [t for t in self.treaties if t.active]

    def treaties_for(self, colonist_id: str) -> list[Treaty]:
        """Return active treaties involving a specific colonist."""
        return [t for t in self.active_treaties()
                if colonist_id in t.parties()]

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "factions": [f.to_dict() for f in self.factions],
            "treaties": [t.to_dict() for t in self.treaties],
            "betrayals": list(self.betrayals),
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiplomacyState:
        """Deserialize from dict."""
        state = cls()
        for fd in d.get("factions", []):
            state.factions.append(Faction(
                id=fd["id"], member_ids=fd["member_ids"],
                coherence=fd["coherence"], dominant_value=fd["dominant_value"],
                formed_year=fd["formed_year"], name=fd.get("name", ""),
                dissolved_year=fd.get("dissolved_year"),
            ))
        for td in d.get("treaties", []):
            state.treaties.append(Treaty(
                id=td["id"], party_a=td["party_a"], party_b=td["party_b"],
                treaty_type=td["treaty_type"], year_signed=td["year_signed"],
                duration=td["duration"], active=td.get("active", True),
                violations=td.get("violations", []),
                year_expired=td.get("year_expired"),
            ))
        state.betrayals = list(d.get("betrayals", []))
        state.history = list(d.get("history", []))
        if state.factions:
            state.next_faction_id = max(
                int(f.id.split("-")[-1]) for f in state.factions
            ) + 1
        if state.treaties:
            state.next_treaty_id = max(
                int(t.id.split("-")[-1]) for t in state.treaties
            ) + 1
        return state


FACTION_NAMES: dict[str, list[str]] = {
    "resolve": ["The Resolute", "Iron Pact", "Steel Covenant"],
    "improvisation": ["The Innovators", "Chaos Guild", "Free Sparks"],
    "empathy": ["The Empaths", "Heart Circle", "Kindred Bonds"],
    "hoarding": ["The Stockpilers", "Vault Keepers", "Resource Guard"],
    "faith": ["The Faithful", "Star Seekers", "Dust Prophets"],
    "paranoia": ["The Vigilant", "Shadow Watch", "Eyes of Mars"],
}


def _mutual_trust(social_graph: Any, a: str, b: str) -> float:
    """Compute average trust in both directions."""
    rel_ab = social_graph.get(a, b)
    rel_ba = social_graph.get(b, a)
    return (rel_ab.trust + rel_ba.trust) / 2


def detect_factions(active_ids: list[str], social_graph: Any,
                    colonists: list[Any],
                    threshold: float = FACTION_TRUST_THRESHOLD) -> list[set[str]]:
    """Find clusters of colonists with high mutual trust.

    Uses connected-component discovery on a mutual trust adjacency graph.
    Filters by internal density to avoid bridge-edge merges.
    """
    adj: dict[str, set[str]] = {cid: set() for cid in active_ids}
    for i, a in enumerate(active_ids):
        for b in active_ids[i + 1:]:
            mt = _mutual_trust(social_graph, a, b)
            if mt >= threshold:
                adj[a].add(b)
                adj[b].add(a)

    visited: set[str] = set()
    components: list[set[str]] = []
    for start in active_ids:
        if start in visited:
            continue
        component: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        if len(component) >= FACTION_MIN_SIZE:
            components.append(component)

    # Filter by internal density
    dense: list[set[str]] = []
    for comp in components:
        members = sorted(comp)
        if len(members) < 2:
            continue
        total = 0.0
        pairs = 0
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                total += _mutual_trust(social_graph, a, b)
                pairs += 1
        density = total / max(1, pairs)
        if density >= threshold * 0.9:
            dense.append(comp)

    return dense


def _dominant_stat(member_ids: list[str], colonists: list[Any]) -> str:
    """Find the stat with highest average across faction members."""
    from src.mars100.colonist import STAT_NAMES
    members = [c for c in colonists if c.id in set(member_ids) and c.is_active()]
    if not members:
        return "resolve"
    totals: dict[str, float] = {s: 0.0 for s in STAT_NAMES}
    for c in members:
        for s in STAT_NAMES:
            totals[s] += getattr(c.stats, s)
    return max(totals, key=lambda s: totals[s])


def _faction_name(dominant: str, rng: random.Random) -> str:
    """Generate a thematic faction name from the dominant stat."""
    names = FACTION_NAMES.get(dominant, ["The Unnamed"])
    return rng.choice(names)


def update_factions(state: DiplomacyState, active_ids: list[str],
                    social_graph: Any, colonists: list[Any],
                    year: int, rng: random.Random) -> list[DiplomacyEvent]:
    """Detect factions with hysteresis — existing factions continue if overlap is high."""
    events: list[DiplomacyEvent] = []
    new_clusters = detect_factions(active_ids, social_graph, colonists)

    matched_existing: set[str] = set()
    matched_clusters: set[int] = set()

    for faction in state.active_factions():
        old_set = set(faction.member_ids) & set(active_ids)
        if not old_set:
            faction.dissolved_year = year
            events.append(DiplomacyEvent(year, "faction_dissolved",
                                         {"faction_id": faction.id,
                                          "reason": "no_active_members"}))
            continue

        best_overlap = 0.0
        best_idx = -1
        for i, cluster in enumerate(new_clusters):
            if i in matched_clusters:
                continue
            overlap = len(old_set & cluster) / max(1, len(old_set | cluster))
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        if best_overlap >= FACTION_OVERLAP_THRESHOLD and best_idx >= 0:
            faction.member_ids = sorted(new_clusters[best_idx])
            dom = _dominant_stat(faction.member_ids, colonists)
            faction.dominant_value = dom
            total_trust = 0.0
            pairs = 0
            for i, a in enumerate(faction.member_ids):
                for b in faction.member_ids[i + 1:]:
                    total_trust += _mutual_trust(social_graph, a, b)
                    pairs += 1
            faction.coherence = total_trust / max(1, pairs)
            matched_existing.add(faction.id)
            matched_clusters.add(best_idx)
        else:
            faction.dissolved_year = year
            events.append(DiplomacyEvent(year, "faction_dissolved",
                                         {"faction_id": faction.id,
                                          "reason": "membership_diverged"}))

    for i, cluster in enumerate(new_clusters):
        if i in matched_clusters:
            continue
        fid = f"faction-{state.next_faction_id}"
        state.next_faction_id += 1
        members = sorted(cluster)
        dom = _dominant_stat(members, colonists)
        name = _faction_name(dom, rng)
        total_trust = 0.0
        pairs = 0
        for j, a in enumerate(members):
            for b in members[j + 1:]:
                total_trust += _mutual_trust(social_graph, a, b)
                pairs += 1
        coherence = total_trust / max(1, pairs)
        faction = Faction(id=fid, member_ids=members, coherence=coherence,
                          dominant_value=dom, formed_year=year, name=name)
        state.factions.append(faction)
        events.append(DiplomacyEvent(year, "faction_formed",
                                     {"faction_id": fid, "name": name,
                                      "members": members,
                                      "coherence": round(coherence, 4)}))

    return events


def _should_propose_treaty(a_id: str, b_id: str, social_graph: Any,
                           existing: list[Treaty],
                           rng: random.Random) -> str | None:
    """Determine if two colonists should propose a treaty and of what type."""
    for t in existing:
        if t.active and {a_id, b_id} == t.parties():
            return None

    mt = _mutual_trust(social_graph, a_id, b_id)
    if mt < 0.5:
        return None

    base_prob = (mt - 0.5) * 0.3
    if rng.random() > base_prob:
        return None

    if mt > 0.7:
        return rng.choice(["cooperation", "mutual_defense"])
    return "non_aggression"


def propose_treaties(state: DiplomacyState, active_ids: list[str],
                     social_graph: Any, year: int,
                     rng: random.Random) -> list[DiplomacyEvent]:
    """Generate treaty proposals between high-trust colonist pairs."""
    events: list[DiplomacyEvent] = []
    max_new = 2

    pairs: list[tuple[str, str, str, float]] = []
    for i, a in enumerate(active_ids):
        for b in active_ids[i + 1:]:
            treaty_type = _should_propose_treaty(
                a, b, social_graph, state.treaties, rng)
            if treaty_type:
                mt = _mutual_trust(social_graph, a, b)
                pairs.append((a, b, treaty_type, mt))

    pairs.sort(key=lambda x: x[3], reverse=True)

    for a, b, ttype, _ in pairs[:max_new]:
        tid = f"treaty-{state.next_treaty_id}"
        state.next_treaty_id += 1
        duration = rng.choice([5, 10, 15, 20])
        treaty = Treaty(id=tid, party_a=a, party_b=b, treaty_type=ttype,
                        year_signed=year, duration=duration)
        state.treaties.append(treaty)
        events.append(DiplomacyEvent(year, "treaty_signed",
                                     {"treaty_id": tid, "parties": [a, b],
                                      "type": ttype, "duration": duration}))

    return events


def expire_treaties(state: DiplomacyState, year: int) -> list[DiplomacyEvent]:
    """Expire treaties that have exceeded their duration."""
    events: list[DiplomacyEvent] = []
    for treaty in state.active_treaties():
        if year >= treaty.year_signed + treaty.duration:
            treaty.active = False
            treaty.year_expired = year
            events.append(DiplomacyEvent(year, "treaty_expired",
                                         {"treaty_id": treaty.id,
                                          "parties": sorted(treaty.parties())}))
    return events


def check_betrayals(state: DiplomacyState,
                    outcomes: list[ActionOutcome],
                    year: int) -> list[DiplomacyEvent]:
    """Detect treaty violations from structured action outcomes.

    Non-aggression: sabotage targeting a treaty partner.
    Cooperation: hoarding while partner cooperated.
    Mutual defense: resting while partner labors.
    """
    events: list[DiplomacyEvent] = []
    actions_by_id = {o.actor_id: o for o in outcomes}

    for treaty in state.active_treaties():
        a_action = actions_by_id.get(treaty.party_a)
        b_action = actions_by_id.get(treaty.party_b)
        if not a_action or not b_action:
            continue

        violations: list[tuple[str, str]] = []

        if treaty.treaty_type == "non_aggression":
            if (a_action.action == "sabotage"
                    and a_action.target_id == treaty.party_b):
                violations.append((treaty.party_a, "sabotage_against_partner"))
            if (b_action.action == "sabotage"
                    and b_action.target_id == treaty.party_a):
                violations.append((treaty.party_b, "sabotage_against_partner"))

        elif treaty.treaty_type == "cooperation":
            if a_action.action == "hoard" and b_action.action == "cooperate":
                violations.append((treaty.party_a, "hoarding_while_partner_cooperates"))
            if b_action.action == "hoard" and a_action.action == "cooperate":
                violations.append((treaty.party_b, "hoarding_while_partner_cooperates"))

        elif treaty.treaty_type == "mutual_defense":
            if a_action.action == "rest" and b_action.action in ("terraform", "farm", "code"):
                violations.append((treaty.party_a, "resting_during_partner_labor"))
            if b_action.action == "rest" and a_action.action in ("terraform", "farm", "code"):
                violations.append((treaty.party_b, "resting_during_partner_labor"))

        for violator_id, reason in violations:
            betrayal = {
                "treaty_id": treaty.id, "violator_id": violator_id,
                "year": year, "reason": reason,
                "treaty_type": treaty.treaty_type,
            }
            treaty.violations.append(betrayal)
            state.betrayals.append(betrayal)
            events.append(DiplomacyEvent(year, "betrayal",
                                         {"treaty_id": treaty.id,
                                          "violator": violator_id,
                                          "reason": reason}))

    return events


def apply_betrayal_consequences(state: DiplomacyState,
                                social_graph: Any,
                                year: int,
                                rng: random.Random) -> None:
    """Apply social consequences for betrayals detected this year."""
    year_betrayals = [b for b in state.betrayals if b["year"] == year]
    for betrayal in year_betrayals:
        violator = betrayal["violator_id"]
        treaty = next((t for t in state.treaties
                       if t.id == betrayal["treaty_id"]), None)
        if not treaty:
            continue
        partner = (treaty.party_a if violator == treaty.party_b
                   else treaty.party_b)

        # Trust collapse with treaty partner
        for pair in [(violator, partner), (partner, violator)]:
            rel = social_graph.get(pair[0], pair[1])
            if rel:
                rel.trust = max(0.0, rel.trust - 0.2 - rng.uniform(0, 0.05))
                rel.respect = max(0.0, rel.respect - 0.15 - rng.uniform(0, 0.03))

        # Mild trust loss with all faction mates of the betrayed partner
        for faction in state.active_factions():
            if partner in faction.member_ids:
                for member in faction.member_ids:
                    if member != violator and member != partner:
                        rel = social_graph.get(member, violator)
                        if rel:
                            rel.trust = max(0.0, rel.trust - 0.08 - rng.uniform(0, 0.02))


def tick_diplomacy(state: DiplomacyState, colonists: list[Any],
                   outcomes: list[ActionOutcome], social_graph: Any,
                   year: int, rng: random.Random) -> list[DiplomacyEvent]:
    """Run one year of diplomacy. Returns all diplomatic events."""
    active = [c for c in colonists if c.is_active()]
    active_ids = [c.id for c in active]

    all_events: list[DiplomacyEvent] = []

    all_events.extend(expire_treaties(state, year))
    all_events.extend(update_factions(state, active_ids, social_graph,
                                      colonists, year, rng))
    all_events.extend(propose_treaties(state, active_ids, social_graph,
                                       year, rng))
    all_events.extend(check_betrayals(state, outcomes, year))
    apply_betrayal_consequences(state, social_graph, year, rng)

    for ev in all_events:
        state.history.append(ev.to_dict())

    return all_events
