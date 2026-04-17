"""
Diplomacy organ for Mars-100 (engine v11.0).

Models internal faction formation, inter-faction treaties, and
sub-sim-enhanced negotiation. Factions emerge organically from social
graph trust clusters with hysteresis to prevent churn.

Key concepts:
  - Factions form when 3+ colonists share high mutual trust + trait similarity
  - Treaties modify resource flow, cohesion, and psychology
  - Faction leaders run LisPy subsims to model negotiation outcomes
  - Depth-2/3 subsims surface meta-insights about diplomacy itself
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.subsim import SubSimBudget, SubSimResult, spawn_subsim

IDEOLOGY_TYPES = ("cooperative", "militant", "isolationist", "technophile", "spiritual")
TREATY_TYPES = ("alliance", "trade_pact", "non_aggression", "resource_sharing")

FORM_THRESHOLD = 0.55
DISSOLVE_THRESHOLD = 0.45
DISSOLVE_GRACE_YEARS = 2
MIN_FACTION_SIZE = 3
MAX_TREATIES_PER_PAIR = 1
MAX_NEGOTIATIONS_PER_YEAR = 4
MAX_FACTIONS = 6
TREATY_DURATION_RANGE = (5, 15)


@dataclass
class Faction:
    """A group of colonists with shared ideology and mutual trust."""
    id: str
    name: str
    ideology: str
    member_ids: list[str]
    influence: float
    formed_year: int
    dissolved_year: int | None = None
    below_threshold_years: int = 0

    def is_active(self) -> bool:
        return self.dissolved_year is None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id, "name": self.name, "ideology": self.ideology,
            "member_ids": self.member_ids, "influence": round(self.influence, 4),
            "formed_year": self.formed_year,
        }
        if self.dissolved_year is not None:
            d["dissolved_year"] = self.dissolved_year
        return d


@dataclass
class Treaty:
    """Agreement between two factions."""
    id: str
    faction_a: str
    faction_b: str
    treaty_type: str
    terms: dict[str, float]
    signed_year: int
    expires_year: int
    broken: bool = False
    broken_year: int | None = None
    subsim_score: float | None = None

    def is_active(self, year: int) -> bool:
        return not self.broken and year < self.expires_year

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id, "faction_a": self.faction_a,
            "faction_b": self.faction_b, "treaty_type": self.treaty_type,
            "terms": self.terms, "signed_year": self.signed_year,
            "expires_year": self.expires_year, "broken": self.broken,
        }
        if self.broken_year is not None:
            d["broken_year"] = self.broken_year
        if self.subsim_score is not None:
            d["subsim_score"] = round(self.subsim_score, 4)
        return d


@dataclass
class DiplomacyState:
    """Complete diplomacy state for the colony."""
    factions: list[Faction] = field(default_factory=list)
    treaties: list[Treaty] = field(default_factory=list)
    next_faction_id: int = 0
    next_treaty_id: int = 0

    def active_factions(self) -> list[Faction]:
        return [f for f in self.factions if f.is_active()]

    def active_treaties(self, year: int) -> list[Treaty]:
        return [t for t in self.treaties if t.is_active(year)]

    def faction_by_id(self, fid: str) -> Faction | None:
        return next((f for f in self.factions if f.id == fid), None)

    def treaties_between(self, fa: str, fb: str, year: int) -> list[Treaty]:
        return [t for t in self.active_treaties(year)
                if {t.faction_a, t.faction_b} == {fa, fb}]

    def to_dict(self) -> dict[str, Any]:
        return {
            "factions": [f.to_dict() for f in self.factions],
            "treaties": [t.to_dict() for t in self.treaties],
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiplomacyState:
        state = cls()
        for fd in d.get("factions", []):
            state.factions.append(Faction(
                id=fd["id"], name=fd["name"], ideology=fd["ideology"],
                member_ids=fd["member_ids"],
                influence=fd.get("influence", 0.5),
                formed_year=fd["formed_year"],
                dissolved_year=fd.get("dissolved_year"),
            ))
        for td in d.get("treaties", []):
            state.treaties.append(Treaty(
                id=td["id"], faction_a=td["faction_a"],
                faction_b=td["faction_b"], treaty_type=td["treaty_type"],
                terms=td.get("terms", {}),
                signed_year=td["signed_year"],
                expires_year=td["expires_year"],
                broken=td.get("broken", False),
                broken_year=td.get("broken_year"),
                subsim_score=td.get("subsim_score"),
            ))
        return state


@dataclass
class DiplomacyTickResult:
    """Result of one year's diplomacy tick."""
    factions_formed: list[dict] = field(default_factory=list)
    factions_dissolved: list[dict] = field(default_factory=list)
    treaties_signed: list[dict] = field(default_factory=list)
    treaties_broken: list[dict] = field(default_factory=list)
    treaties_expired: list[dict] = field(default_factory=list)
    negotiation_subsims: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factions_formed": self.factions_formed,
            "factions_dissolved": self.factions_dissolved,
            "treaties_signed": self.treaties_signed,
            "treaties_broken": self.treaties_broken,
            "treaties_expired": self.treaties_expired,
            "negotiation_subsims": self.negotiation_subsims,
        }


# ── Faction Detection ────────────────────────────────────────────────

FACTION_NAMES = [
    "Iron Circle", "Dust Covenant", "Water Keepers", "Solar Collective",
    "Stone Pact", "Wind Accord", "Fire Watch", "Root Alliance",
    "Sky Treaty", "Sand Council", "Frost Union", "Ember Guild",
]

IDEOLOGY_STAT_MAP: dict[str, list[str]] = {
    "cooperative": ["empathy", "faith"],
    "militant": ["resolve", "paranoia"],
    "isolationist": ["hoarding", "paranoia"],
    "technophile": ["improvisation", "resolve"],
    "spiritual": ["faith", "empathy"],
}


def _compute_affinity(social_graph: Any, a_id: str, b_id: str,
                      colonists_by_id: dict[str, Any]) -> float:
    """Compute pairwise affinity: trust + trait similarity."""
    rel_ab = social_graph.get(a_id, b_id)
    rel_ba = social_graph.get(b_id, a_id)
    trust = (rel_ab.trust + rel_ba.trust) / 2.0

    ca = colonists_by_id.get(a_id)
    cb = colonists_by_id.get(b_id)
    if ca is None or cb is None:
        return trust

    from src.mars100.colonist import STAT_NAMES
    diff_sum = sum(abs(getattr(ca.stats, s) - getattr(cb.stats, s))
                   for s in STAT_NAMES)
    similarity = 1.0 - (diff_sum / len(STAT_NAMES))
    return trust * 0.6 + similarity * 0.4


def _determine_ideology(member_ids: list[str],
                         colonists_by_id: dict[str, Any]) -> str:
    """Pick ideology based on dominant traits of members."""
    trait_sums: dict[str, float] = {}
    from src.mars100.colonist import STAT_NAMES
    for cid in member_ids:
        c = colonists_by_id.get(cid)
        if c is None:
            continue
        for s in STAT_NAMES:
            trait_sums[s] = trait_sums.get(s, 0.0) + getattr(c.stats, s)

    best_ideology = "cooperative"
    best_score = -1.0
    for ideology, traits in IDEOLOGY_STAT_MAP.items():
        score = sum(trait_sums.get(t, 0.0) for t in traits)
        if score > best_score:
            best_score = score
            best_ideology = ideology
    return best_ideology


def detect_factions(
    social_graph: Any,
    colonists: list[Any],
    state: DiplomacyState,
    year: int,
    rng: random.Random,
) -> list[dict]:
    """Detect new factions from social graph clusters.

    Uses greedy agglomerative clustering with hysteresis:
    - Form threshold: mutual affinity > FORM_THRESHOLD for all pairs
    - Dissolve only after DISSOLVE_GRACE_YEARS below DISSOLVE_THRESHOLD
    - Match new clusters to existing factions by member overlap
    """
    active = [c for c in colonists if c.is_active()]
    if len(active) < MIN_FACTION_SIZE:
        return []

    active_ids = [c.id for c in active]
    colonists_by_id = {c.id: c for c in active}
    existing_member_sets = {
        f.id: set(f.member_ids)
        for f in state.active_factions()
    }
    assigned = set()
    for f in state.active_factions():
        assigned.update(f.member_ids)

    # Build affinity matrix for unassigned colonists
    unassigned = [cid for cid in active_ids if cid not in assigned]
    if len(unassigned) < MIN_FACTION_SIZE:
        return []

    # Greedy cluster: start from highest-affinity pair, grow
    affinities: list[tuple[str, str, float]] = []
    for i, a in enumerate(unassigned):
        for b in unassigned[i + 1:]:
            aff = _compute_affinity(social_graph, a, b, colonists_by_id)
            if aff >= FORM_THRESHOLD:
                affinities.append((a, b, aff))
    affinities.sort(key=lambda x: x[2], reverse=True)

    new_factions: list[dict] = []
    used = set()
    for a, b, _ in affinities:
        if a in used or b in used:
            continue
        if len(state.active_factions()) + len(new_factions) >= MAX_FACTIONS:
            break
        cluster = {a, b}
        # Try to grow cluster
        for cid in unassigned:
            if cid in used or cid in cluster:
                continue
            fits = all(
                _compute_affinity(social_graph, cid, m, colonists_by_id) >= FORM_THRESHOLD
                for m in cluster
            )
            if fits:
                cluster.add(cid)
        if len(cluster) < MIN_FACTION_SIZE:
            continue

        # Check overlap with existing dissolved factions for ID reuse
        members = sorted(cluster)
        fid = f"faction-{state.next_faction_id}"
        state.next_faction_id += 1
        name = rng.choice(FACTION_NAMES)
        ideology = _determine_ideology(members, colonists_by_id)
        influence = len(members) / max(1, len(active))

        faction = Faction(
            id=fid, name=name, ideology=ideology,
            member_ids=members, influence=influence,
            formed_year=year,
        )
        state.factions.append(faction)
        used.update(cluster)
        new_factions.append(faction.to_dict())

    return new_factions


def update_faction_membership(
    state: DiplomacyState,
    social_graph: Any,
    colonists: list[Any],
    year: int,
) -> list[dict]:
    """Update existing factions: refresh membership, dissolve stale ones."""
    active_ids = {c.id for c in colonists if c.is_active()}
    colonists_by_id = {c.id: c for c in colonists if c.is_active()}
    dissolved: list[dict] = []

    for faction in state.active_factions():
        # Remove dead/exiled members
        faction.member_ids = [m for m in faction.member_ids if m in active_ids]

        if len(faction.member_ids) < 2:
            faction.dissolved_year = year
            dissolved.append({"id": faction.id, "reason": "insufficient_members",
                              "year": year})
            continue

        # Check average internal affinity
        total_aff = 0.0
        count = 0
        for i, a in enumerate(faction.member_ids):
            for b in faction.member_ids[i + 1:]:
                total_aff += _compute_affinity(social_graph, a, b, colonists_by_id)
                count += 1
        avg_aff = total_aff / max(1, count)

        if avg_aff < DISSOLVE_THRESHOLD:
            faction.below_threshold_years += 1
            if faction.below_threshold_years >= DISSOLVE_GRACE_YEARS:
                faction.dissolved_year = year
                dissolved.append({"id": faction.id, "reason": "low_cohesion",
                                  "year": year})
        else:
            faction.below_threshold_years = 0

        # Update influence
        faction.influence = len(faction.member_ids) / max(1, len(active_ids))
        # Refresh ideology
        faction.ideology = _determine_ideology(faction.member_ids, colonists_by_id)

    return dissolved


# ── Treaties ─────────────────────────────────────────────────────────

def _generate_treaty_terms(treaty_type: str, rng: random.Random) -> dict[str, float]:
    """Generate treaty terms based on type."""
    if treaty_type == "alliance":
        return {"cohesion_bonus": round(rng.uniform(0.02, 0.05), 4),
                "defense_mult": round(rng.uniform(1.1, 1.3), 4)}
    elif treaty_type == "trade_pact":
        return {"trade_bonus": round(rng.uniform(0.01, 0.04), 4),
                "resource_sharing": round(rng.uniform(0.01, 0.03), 4)}
    elif treaty_type == "non_aggression":
        return {"paranoia_reduction": round(rng.uniform(0.01, 0.03), 4),
                "sabotage_penalty": round(rng.uniform(0.05, 0.15), 4)}
    elif treaty_type == "resource_sharing":
        return {"food_sharing": round(rng.uniform(0.01, 0.03), 4),
                "water_sharing": round(rng.uniform(0.01, 0.03), 4)}
    return {}


def _build_negotiation_expr(treaty_type: str, fa: Faction, fb: Faction) -> str:
    """Build a LisPy expression to model negotiation outcome."""
    n_a = len(fa.member_ids)
    n_b = len(fb.member_ids)
    inf_a = fa.influence
    inf_b = fb.influence

    if treaty_type == "alliance":
        return (f"(let ((combined-strength (+ {inf_a:.3f} {inf_b:.3f})) "
                f"(size-ratio (/ {min(n_a, n_b)} {max(n_a, n_b)}))) "
                f"(if (> (* combined-strength size-ratio) 0.4) "
                f"(+ combined-strength 0.1) (- combined-strength 0.2)))")
    elif treaty_type == "trade_pact":
        return (f"(let ((surplus (- {inf_a:.3f} (* {n_a} 0.05))) "
                f"(demand (* {n_b} 0.04))) "
                f"(if (> surplus demand) (+ surplus 0.1) (* surplus 0.5)))")
    elif treaty_type == "non_aggression":
        return (f"(let ((trust-sum (+ {inf_a:.3f} {inf_b:.3f})) "
                f"(threat (* {max(n_a, n_b)} 0.03))) "
                f"(if (> trust-sum threat) 1 0))")
    else:
        return (f"(let ((pool (+ {inf_a:.3f} {inf_b:.3f})) "
                f"(need (* (+ {n_a} {n_b}) 0.03))) "
                f"(if (> pool need) (- pool need) (/ pool 2)))")


def negotiate_treaty(
    fa: Faction, fb: Faction,
    state: DiplomacyState,
    year: int,
    subsim_budget: SubSimBudget,
    subsim_log: list[SubSimResult],
    rng: random.Random,
) -> Treaty | None:
    """Attempt to negotiate a treaty between two factions.

    Uses LisPy subsim to model the outcome. If the subsim predicts
    benefit, the treaty is signed.
    """
    # Check existing treaties
    if len(state.treaties_between(fa.id, fb.id, year)) >= MAX_TREATIES_PER_PAIR:
        return None

    treaty_type = rng.choice(TREATY_TYPES)
    leader_id = fa.member_ids[0] if fa.member_ids else "unknown"

    # Run subsim to evaluate
    expr = _build_negotiation_expr(treaty_type, fa, fb)
    bindings: dict[str, Any] = {
        "faction-a-size": len(fa.member_ids),
        "faction-b-size": len(fb.member_ids),
        "faction-a-influence": fa.influence,
        "faction-b-influence": fb.influence,
    }

    result = spawn_subsim(
        expression=expr, colonist_id=leader_id,
        year=year, bindings=bindings,
        depth=1, budget=subsim_budget, log=subsim_log,
    )

    subsim_score: float | None = None
    if result.succeeded and isinstance(result.result, (int, float)):
        subsim_score = float(result.result)

        # Depth 2: if result is interesting, model counter-offer
        if abs(subsim_score) > 0.3 and subsim_budget.can_spawn(leader_id):
            d2_expr = (f"(let ((offer {subsim_score:.4f}) "
                       f"(counter (* offer 0.9))) "
                       f"(if (> counter 0.2) (+ offer counter) "
                       f"(- offer 0.1)))")
            d2 = spawn_subsim(
                expression=d2_expr, colonist_id=leader_id,
                year=year, bindings=bindings,
                depth=2, budget=subsim_budget, log=subsim_log,
            )
            result.children.append(d2)

            # Depth 3: meta-negotiation (very rare)
            if (d2.succeeded and isinstance(d2.result, (int, float))
                    and abs(d2.result) > 0.6
                    and subsim_budget.can_spawn(leader_id)):
                d3_expr = (f"(let ((meta (+ {d2.result:.4f} "
                           f"(* sim-depth 0.01)))) "
                           f"(if (> meta 0.5) "
                           f"(+ meta (* sim-depth 0.1)) "
                           f"(- meta 0.05)))")
                d3 = spawn_subsim(
                    expression=d3_expr, colonist_id=leader_id,
                    year=year, bindings=bindings,
                    depth=3, budget=subsim_budget, log=subsim_log,
                )
                d2.children.append(d3)

    # Decide whether to sign based on subsim
    if subsim_score is not None and subsim_score <= 0:
        return None
    if subsim_score is None and rng.random() > 0.3:
        return None

    duration = rng.randint(*TREATY_DURATION_RANGE)
    terms = _generate_treaty_terms(treaty_type, rng)
    tid = f"treaty-{state.next_treaty_id}"
    state.next_treaty_id += 1

    treaty = Treaty(
        id=tid, faction_a=fa.id, faction_b=fb.id,
        treaty_type=treaty_type, terms=terms,
        signed_year=year, expires_year=year + duration,
        subsim_score=subsim_score,
    )
    state.treaties.append(treaty)
    return treaty


def check_treaty_violations(
    state: DiplomacyState,
    actions: dict[str, str],
    year: int,
    rng: random.Random,
) -> list[dict]:
    """Check if any faction member's actions violate active treaties."""
    broken: list[dict] = []
    faction_members: dict[str, set[str]] = {}
    for f in state.active_factions():
        faction_members[f.id] = set(f.member_ids)

    for treaty in state.active_treaties(year):
        if treaty.treaty_type == "non_aggression":
            members_a = faction_members.get(treaty.faction_a, set())
            members_b = faction_members.get(treaty.faction_b, set())
            saboteurs_a = sum(1 for m in members_a
                              if actions.get(m) == "sabotage")
            saboteurs_b = sum(1 for m in members_b
                              if actions.get(m) == "sabotage")
            if saboteurs_a + saboteurs_b > 0 and rng.random() < 0.5:
                treaty.broken = True
                treaty.broken_year = year
                broken.append({"treaty_id": treaty.id,
                                "reason": "sabotage_violation",
                                "year": year})
        elif treaty.treaty_type == "alliance":
            members_a = faction_members.get(treaty.faction_a, set())
            members_b = faction_members.get(treaty.faction_b, set())
            hoarders = sum(1 for m in (members_a | members_b)
                           if actions.get(m) == "hoard")
            if hoarders >= 2 and rng.random() < 0.3:
                treaty.broken = True
                treaty.broken_year = year
                broken.append({"treaty_id": treaty.id,
                                "reason": "hoarding_violation",
                                "year": year})
    return broken


def expire_treaties(state: DiplomacyState, year: int) -> list[dict]:
    """Mark expired treaties."""
    expired: list[dict] = []
    for treaty in state.treaties:
        if not treaty.broken and treaty.expires_year <= year:
            expired.append({"treaty_id": treaty.id, "year": year})
    return expired


# ── Modifiers ────────────────────────────────────────────────────────

def compute_diplomatic_modifiers(
    state: DiplomacyState, year: int,
) -> dict[str, float]:
    """Compute resource and social modifiers from active treaties.

    Returns a dict of modifier keys → values that the engine applies.
    """
    mods: dict[str, float] = {
        "cohesion_bonus": 0.0,
        "paranoia_reduction": 0.0,
        "trade_bonus": 0.0,
        "food_bonus": 0.0,
        "water_bonus": 0.0,
        "broken_treaty_paranoia": 0.0,
    }

    for treaty in state.active_treaties(year):
        terms = treaty.terms
        if treaty.treaty_type == "alliance":
            mods["cohesion_bonus"] += terms.get("cohesion_bonus", 0.0)
        elif treaty.treaty_type == "trade_pact":
            mods["trade_bonus"] += terms.get("trade_bonus", 0.0)
        elif treaty.treaty_type == "non_aggression":
            mods["paranoia_reduction"] += terms.get("paranoia_reduction", 0.0)
        elif treaty.treaty_type == "resource_sharing":
            mods["food_bonus"] += terms.get("food_sharing", 0.0)
            mods["water_bonus"] += terms.get("water_sharing", 0.0)

    # Broken treaties increase paranoia
    recent_broken = sum(1 for t in state.treaties
                        if t.broken and t.broken_year is not None
                        and year - t.broken_year <= 5)
    mods["broken_treaty_paranoia"] = recent_broken * 0.02

    return mods


def compute_faction_pressure(
    colonist_id: str,
    state: DiplomacyState,
    year: int,
) -> dict[str, float]:
    """Compute action weight modifiers from faction membership."""
    pressure: dict[str, float] = {}
    faction = next((f for f in state.active_factions()
                    if colonist_id in f.member_ids), None)
    if faction is None:
        return pressure

    if faction.ideology == "cooperative":
        pressure["cooperate"] = 0.3
        pressure["mediate"] = 0.2
        pressure["sabotage"] = -0.3
    elif faction.ideology == "militant":
        pressure["sabotage"] = 0.2
        pressure["hoard"] = 0.2
        pressure["cooperate"] = -0.1
    elif faction.ideology == "isolationist":
        pressure["hoard"] = 0.3
        pressure["rest"] = 0.1
        pressure["cooperate"] = -0.2
    elif faction.ideology == "technophile":
        pressure["code"] = 0.3
        pressure["research"] = 0.3
        pressure["pray"] = -0.1
    elif faction.ideology == "spiritual":
        pressure["pray"] = 0.3
        pressure["mediate"] = 0.2
        pressure["sabotage"] = -0.2

    return pressure


# ── Main Tick ────────────────────────────────────────────────────────

def tick_diplomacy(
    state: DiplomacyState,
    social_graph: Any,
    colonists: list[Any],
    actions: dict[str, str],
    year: int,
    subsim_budget: SubSimBudget,
    subsim_log: list[SubSimResult],
    rng: random.Random,
) -> DiplomacyTickResult:
    """Advance diplomacy by one Martian year.

    Phase order:
      1. Update existing faction membership (prune dead members, dissolve)
      2. Detect new factions from unassigned colonists
      3. Expire old treaties
      4. Check treaty violations from this year's actions
      5. Negotiate new treaties between faction pairs
    """
    result = DiplomacyTickResult()

    # Factions only emerge after year 8 (colony needs time to cluster)
    if year < 8:
        return result

    # Phase 1: update existing factions
    dissolved = update_faction_membership(state, social_graph, colonists, year)
    result.factions_dissolved = dissolved

    # Phase 2: detect new factions
    formed = detect_factions(social_graph, colonists, state, year, rng)
    result.factions_formed = formed

    # Phase 3: expire treaties
    expired = expire_treaties(state, year)
    result.treaties_expired = expired

    # Phase 4: check violations
    broken = check_treaty_violations(state, actions, year, rng)
    result.treaties_broken = broken

    # Phase 5: negotiate new treaties
    active = state.active_factions()
    negotiations = 0
    for i, fa in enumerate(active):
        if negotiations >= MAX_NEGOTIATIONS_PER_YEAR:
            break
        for fb in active[i + 1:]:
            if negotiations >= MAX_NEGOTIATIONS_PER_YEAR:
                break
            if rng.random() > 0.35:
                continue
            treaty = negotiate_treaty(
                fa, fb, state, year,
                subsim_budget, subsim_log, rng,
            )
            if treaty:
                result.treaties_signed.append(treaty.to_dict())
                result.negotiation_subsims.append({
                    "factions": [fa.id, fb.id],
                    "treaty_type": treaty.treaty_type,
                    "subsim_score": treaty.subsim_score,
                    "year": year,
                })
            negotiations += 1

    return result
