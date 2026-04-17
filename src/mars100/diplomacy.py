"""
Mars-100 diplomacy engine — factions, treaties, schisms, vote modifiers.

Colonists form factions based on shared dominant stats. Factions can negotiate
treaties (resource-sharing pacts), experience schisms, and influence governance
votes. Each tick updates faction membership, evaluates treaty proposals, and
checks for schisms.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# ── constants ──────────────────────────────────────────────────────────────

MAX_FACTIONS = 4
MIN_FACTION_SIZE = 2
FACTION_TRUST_THRESHOLD = 0.35
SCHISM_COHESION_THRESHOLD = 0.25
TREATY_BASE_DURATION = 15  # years
TREATY_PROPOSAL_PROBABILITY = 0.3
SCHISM_PROBABILITY = 0.4
VOTE_MODIFIER_CAP = 0.25

FACTION_NAMES = [
    "Terraformers Guild", "Hydro Collective", "Steel Covenant",
    "Skyward Pact", "Pathfinder Union", "Iron Hearth",
    "Dust Runners", "Beacon Circle",
]

# Maps dominant stat → preferred governance type
GOVERNANCE_PREFERENCE: dict[str, str] = {
    "resolve": "dictator",
    "improvisation": "lottery",
    "empathy": "consensus",
    "hoarding": "council",
    "faith": "consensus",
    "paranoia": "anarchy",
}

TREATY_TYPES: list[dict[str, str | float]] = [
    {"id": "labour", "label": "Labour Exchange", "bonus_key": "research_bonus", "bonus": 0.15},
    {"id": "emergency_air", "label": "Emergency Air Pact", "bonus_key": "air_crisis_bonus", "bonus": 0.10},
    {"id": "infra_priority", "label": "Infrastructure Priority", "bonus_key": "build_speed_bonus", "bonus": 0.12},
]


# ── data classes ───────────────────────────────────────────────────────────

@dataclass
class Faction:
    """A political faction within the colony."""
    name: str
    dominant_stat: str
    member_ids: list[str] = field(default_factory=list)
    cohesion: float = 0.5
    formed_year: int = 0

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dominant_stat": self.dominant_stat,
            "member_ids": list(self.member_ids),
            "cohesion": round(self.cohesion, 4),
            "size": self.size,
            "formed_year": self.formed_year,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Faction:
        return cls(
            name=data["name"],
            dominant_stat=data["dominant_stat"],
            member_ids=list(data.get("member_ids", [])),
            cohesion=data.get("cohesion", 0.5),
            formed_year=data.get("formed_year", 0),
        )


@dataclass
class Treaty:
    """A pact between two factions."""
    treaty_type: str
    faction_a: str
    faction_b: str
    signed_year: int
    duration: int = TREATY_BASE_DURATION
    bonus_key: str = ""
    bonus_value: float = 0.0

    @property
    def expires_year(self) -> int:
        return self.signed_year + self.duration

    def is_active(self, current_year: int) -> bool:
        return current_year < self.expires_year

    def to_dict(self) -> dict:
        return {
            "treaty_type": self.treaty_type,
            "faction_a": self.faction_a,
            "faction_b": self.faction_b,
            "signed_year": self.signed_year,
            "duration": self.duration,
            "expires_year": self.expires_year,
            "bonus_key": self.bonus_key,
            "bonus_value": round(self.bonus_value, 4),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Treaty:
        return cls(
            treaty_type=data["treaty_type"],
            faction_a=data["faction_a"],
            faction_b=data["faction_b"],
            signed_year=data["signed_year"],
            duration=data.get("duration", TREATY_BASE_DURATION),
            bonus_key=data.get("bonus_key", ""),
            bonus_value=data.get("bonus_value", 0.0),
        )


@dataclass
class DiplomacyState:
    """Persistent diplomacy state across years."""
    factions: list[Faction] = field(default_factory=list)
    treaties: list[Treaty] = field(default_factory=list)
    schism_log: list[dict] = field(default_factory=list)
    faction_name_idx: int = 0

    def to_dict(self) -> dict:
        return {
            "factions": [f.to_dict() for f in self.factions],
            "treaties": [t.to_dict() for t in self.treaties],
            "schism_log": list(self.schism_log),
            "faction_name_idx": self.faction_name_idx,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DiplomacyState:
        return cls(
            factions=[Faction.from_dict(f) for f in data.get("factions", [])],
            treaties=[Treaty.from_dict(t) for t in data.get("treaties", [])],
            schism_log=list(data.get("schism_log", [])),
            faction_name_idx=data.get("faction_name_idx", 0),
        )

    def _next_faction_name(self) -> str:
        name = FACTION_NAMES[self.faction_name_idx % len(FACTION_NAMES)]
        self.faction_name_idx += 1
        return name


@dataclass
class DiplomacyTickResult:
    """What happened diplomatically in one year."""
    factions_formed: list[dict] = field(default_factory=list)
    schisms: list[dict] = field(default_factory=list)
    treaties_proposed: list[dict] = field(default_factory=list)
    treaties_signed: list[dict] = field(default_factory=list)
    treaties_expired: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factions_formed": self.factions_formed,
            "schisms": self.schisms,
            "treaties_proposed": self.treaties_proposed,
            "treaties_signed": self.treaties_signed,
            "treaties_expired": self.treaties_expired,
        }


# ── core functions ─────────────────────────────────────────────────────────

def _dominant_stat(colonist: object) -> str:
    """Get the dominant stat name of a colonist."""
    return colonist.stats.dominant()  # type: ignore[union-attr]


def _active_member_ids(faction: Faction, active_ids: set[str]) -> list[str]:
    """Filter faction members to only those still active."""
    return [mid for mid in faction.member_ids if mid in active_ids]


def detect_factions(
    state: DiplomacyState,
    colonists: list,
    social: object,
    year: int,
    rng: random.Random,
) -> list[dict]:
    """Detect new factions from colonists with shared dominant stats and trust."""
    active = [c for c in colonists if c.is_active()]
    active_ids = {c.id for c in active}

    # Prune dead members from existing factions
    for faction in state.factions:
        faction.member_ids = _active_member_ids(faction, active_ids)

    # Remove empty factions
    state.factions = [f for f in state.factions if f.size >= 1]

    # Build set of already-factioned colonist ids
    factioned = set()
    for f in state.factions:
        factioned.update(f.member_ids)

    # Group unfactioned colonists by dominant stat
    unaffiliated: dict[str, list[str]] = {}
    for c in active:
        if c.id not in factioned:
            dom = _dominant_stat(c)
            unaffiliated.setdefault(dom, []).append(c.id)

    formed: list[dict] = []
    if len(state.factions) >= MAX_FACTIONS:
        return formed

    for stat, members in unaffiliated.items():
        if len(members) < MIN_FACTION_SIZE:
            continue
        if len(state.factions) >= MAX_FACTIONS:
            break
        # Check average pairwise trust
        trust_sum = 0.0
        pairs = 0
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                rel = social.get(a, b)  # type: ignore[union-attr]
                trust_sum += rel.trust
                pairs += 1
        avg_trust = trust_sum / max(pairs, 1)
        if avg_trust < FACTION_TRUST_THRESHOLD:
            continue
        name = state._next_faction_name()
        faction = Faction(
            name=name, dominant_stat=stat,
            member_ids=list(members), cohesion=avg_trust,
            formed_year=year,
        )
        state.factions.append(faction)
        formed.append(faction.to_dict())
    return formed


def update_cohesion(state: DiplomacyState, social: object) -> None:
    """Recompute faction cohesion from current pairwise trust."""
    for faction in state.factions:
        if faction.size < 2:
            faction.cohesion = 0.5
            continue
        trust_sum = 0.0
        pairs = 0
        for i, a in enumerate(faction.member_ids):
            for b in faction.member_ids[i + 1:]:
                rel = social.get(a, b)  # type: ignore[union-attr]
                trust_sum += rel.trust
                pairs += 1
        faction.cohesion = trust_sum / max(pairs, 1)


def check_schisms(
    state: DiplomacyState, year: int, rng: random.Random,
) -> list[dict]:
    """Check for faction schisms when cohesion is low."""
    schisms: list[dict] = []
    new_factions: list[Faction] = []
    for faction in list(state.factions):
        if faction.cohesion >= SCHISM_COHESION_THRESHOLD:
            continue
        # Need enough members that both halves survive
        if faction.size <= MIN_FACTION_SIZE + 1:
            continue
        if rng.random() > SCHISM_PROBABILITY:
            continue
        if len(state.factions) + len(new_factions) >= MAX_FACTIONS:
            continue
        # Split: first half stays, second half forms new faction
        mid = faction.size // 2
        shuffled = list(faction.member_ids)
        rng.shuffle(shuffled)
        stay = shuffled[:mid]
        leave = shuffled[mid:]
        if len(stay) < MIN_FACTION_SIZE or len(leave) < MIN_FACTION_SIZE:
            continue
        faction.member_ids = stay
        new_name = state._next_faction_name()
        new_faction = Faction(
            name=new_name, dominant_stat=faction.dominant_stat,
            member_ids=leave, cohesion=0.5, formed_year=year,
        )
        new_factions.append(new_faction)
        record = {
            "year": year, "parent": faction.name,
            "child": new_name,
            "parent_members": list(stay), "child_members": list(leave),
        }
        state.schism_log.append(record)
        schisms.append(record)
    state.factions.extend(new_factions)
    return schisms


def propose_treaty(
    state: DiplomacyState, year: int, rng: random.Random,
) -> list[dict]:
    """Attempt to propose treaties between faction pairs."""
    proposed: list[dict] = []
    if len(state.factions) < 2:
        return proposed
    pairs = []
    for i, fa in enumerate(state.factions):
        for fb in state.factions[i + 1:]:
            # Skip if they already have an active treaty
            existing = any(
                t.is_active(year) and
                {t.faction_a, t.faction_b} == {fa.name, fb.name}
                for t in state.treaties
            )
            if existing:
                continue
            pairs.append((fa, fb))
    for fa, fb in pairs:
        if rng.random() > TREATY_PROPOSAL_PROBABILITY:
            continue
        ttype = rng.choice(TREATY_TYPES)
        proposed.append({
            "treaty_type": ttype["id"],
            "faction_a": fa.name,
            "faction_b": fb.name,
            "bonus_key": ttype["bonus_key"],
            "bonus_value": ttype["bonus"],
            "label": ttype["label"],
        })
    return proposed


def sign_treaty(
    state: DiplomacyState, proposal: dict, year: int,
) -> Treaty:
    """Sign a proposed treaty and add it to state."""
    treaty = Treaty(
        treaty_type=proposal["treaty_type"],
        faction_a=proposal["faction_a"],
        faction_b=proposal["faction_b"],
        signed_year=year,
        bonus_key=proposal.get("bonus_key", ""),
        bonus_value=proposal.get("bonus_value", 0.0),
    )
    state.treaties.append(treaty)
    return treaty


def expire_treaties(
    state: DiplomacyState, year: int,
) -> list[dict]:
    """Remove expired treaties, return what expired."""
    expired: list[dict] = []
    remaining: list[Treaty] = []
    for t in state.treaties:
        if t.is_active(year):
            remaining.append(t)
        else:
            expired.append(t.to_dict())
    state.treaties = remaining
    return expired


def compute_treaty_effects(state: DiplomacyState) -> dict[str, float]:
    """Sum active treaty bonuses into a dict of effects."""
    effects: dict[str, float] = {}
    for t in state.treaties:
        if t.bonus_key:
            effects[t.bonus_key] = effects.get(t.bonus_key, 0.0) + t.bonus_value
    return effects


def faction_vote_modifier(
    colonist_id: str, gov_type: str, factions: list[Faction],
) -> float:
    """Compute how much a colonist's faction membership modifies their vote."""
    for faction in factions:
        if colonist_id not in faction.member_ids:
            continue
        preferred = GOVERNANCE_PREFERENCE.get(faction.dominant_stat)
        if preferred is None:
            return 0.0
        if gov_type == preferred:
            modifier = faction.cohesion * VOTE_MODIFIER_CAP
        else:
            modifier = -faction.cohesion * VOTE_MODIFIER_CAP * 0.5
        return max(-VOTE_MODIFIER_CAP, min(VOTE_MODIFIER_CAP, modifier))
    return 0.0


def tick_diplomacy(
    state: DiplomacyState,
    colonists: list,
    social: object,
    year: int,
    rng: random.Random,
) -> DiplomacyTickResult:
    """Run one year of diplomacy. Returns what happened."""
    result = DiplomacyTickResult()

    # 1. Detect new factions
    result.factions_formed = detect_factions(state, colonists, social, year, rng)

    # 2. Update cohesion from current trust
    update_cohesion(state, social)

    # 3. Check for schisms
    result.schisms = check_schisms(state, year, rng)

    # 4. Propose and sign treaties
    result.treaties_proposed = propose_treaty(state, year, rng)
    for prop in result.treaties_proposed:
        # Acceptance based on combined cohesion of the two factions
        fa = next((f for f in state.factions if f.name == prop["faction_a"]), None)
        fb = next((f for f in state.factions if f.name == prop["faction_b"]), None)
        if fa and fb:
            acceptance = (fa.cohesion + fb.cohesion) / 2.0
            if rng.random() < acceptance:
                treaty = sign_treaty(state, prop, year)
                result.treaties_signed.append(treaty.to_dict())

    # 5. Expire old treaties
    result.treaties_expired = expire_treaties(state, year)

    return result
