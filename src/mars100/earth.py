"""
Earth Protocol for Mars-100.

Models the evolving relationship between Mars colony and Earth:
- Contact quality (degrades over time, boosted by earth_contact events)
- Supply pipeline (resources from Earth, decreasing as colony matures)
- Earth trust (willingness to support the colony)
- Autonomy desire (colony's push toward independence)
- Treaties (formal agreements with terms and expiration)
- Independence vote (when autonomy_desire crosses threshold)

Two-phase tick:
- tick_earth_pre(): before resource tick — compute supply effects
- tick_earth_post(): after everything — update diplomacy state

Separate RNG (seed + 6151) to avoid disturbing existing test determinism.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# --- Constants -----------------------------------------------------------

INDEPENDENCE_THRESHOLD = 0.7
SUPPLY_BASE = 0.04
CONTACT_DECAY = 0.02
TRUST_DECAY = 0.01
SUPPLY_PIPELINE_DECAY = 0.005
AUTONOMY_GROWTH_BASE = 0.01
MAX_TREATIES = 3
TREATY_PROPOSE_CHANCE = 0.15
EMERGENCY_AID_THRESHOLD = 0.2

TREATY_TEMPLATES: dict[str, dict] = {
    "resource_pact": {
        "duration": 10,
        "terms": {"supply_bonus": 0.02, "research_obligation": True},
        "description": "Earth sends extra supplies; colony sends research data.",
    },
    "tech_transfer": {
        "duration": 5,
        "terms": {"infra_bonus": 0.01, "colony_reports_required": True},
        "description": "Earth shares technology blueprints with the colony.",
    },
    "emergency_aid": {
        "duration": 2,
        "terms": {"supply_burst": 0.08, "autonomy_penalty": 0.05},
        "description": "Emergency resource shipment — but at the cost of independence.",
    },
}


# --- Data structures -----------------------------------------------------

@dataclass
class Treaty:
    """A formal agreement between Earth and Mars colony."""
    kind: str
    start_year: int
    duration: int
    active: bool = True
    breached: bool = False
    terms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "start_year": self.start_year,
            "duration": self.duration, "active": self.active,
            "breached": self.breached, "terms": self.terms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Treaty:
        return cls(
            kind=d["kind"], start_year=d["start_year"],
            duration=d["duration"], active=d.get("active", True),
            breached=d.get("breached", False),
            terms=d.get("terms", {}),
        )


@dataclass
class EarthRelations:
    """Complete state of Earth-Mars diplomatic relations."""
    contact_quality: float = 1.0
    earth_trust: float = 0.8
    supply_pipeline: float = 1.0
    autonomy_desire: float = 0.0
    independence_declared: bool = False
    treaties: list[Treaty] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "contact_quality": round(self.contact_quality, 4),
            "earth_trust": round(self.earth_trust, 4),
            "supply_pipeline": round(self.supply_pipeline, 4),
            "autonomy_desire": round(self.autonomy_desire, 4),
            "independence_declared": self.independence_declared,
            "treaties": [t.to_dict() for t in self.treaties],
            "history": self.history[-20:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> EarthRelations:
        treaties = [Treaty.from_dict(t) for t in d.get("treaties", [])]
        return cls(
            contact_quality=d.get("contact_quality", 1.0),
            earth_trust=d.get("earth_trust", 0.8),
            supply_pipeline=d.get("supply_pipeline", 1.0),
            autonomy_desire=d.get("autonomy_desire", 0.0),
            independence_declared=d.get("independence_declared", False),
            treaties=treaties,
            history=d.get("history", []),
        )

    def active_treaties(self) -> list[Treaty]:
        """Return currently active (non-breached) treaties."""
        return [t for t in self.treaties if t.active and not t.breached]

    def has_treaty_kind(self, kind: str) -> bool:
        """Check if an active treaty of the given kind exists."""
        return any(t.kind == kind for t in self.active_treaties())


# --- Internal tick helpers -----------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _tick_contact(earth: EarthRelations, events: list[dict],
                  year: int) -> None:
    """Update contact quality based on events and natural decay."""
    earth.contact_quality -= CONTACT_DECAY
    for ev in events:
        if ev.get("name") == "earth_contact":
            earth.contact_quality += 0.15 + ev.get("severity", 0.1) * 0.1
        if ev.get("name") == "solar_flare":
            earth.contact_quality -= ev.get("severity", 0.5) * 0.1
    if year > 50:
        earth.contact_quality -= 0.005
    earth.contact_quality = _clamp(earth.contact_quality)


def _tick_trust(earth: EarthRelations, governance_type: str,
                resource_avg: float, rng: random.Random) -> None:
    """Update Earth's trust in the colony."""
    earth.earth_trust -= TRUST_DECAY
    if governance_type in ("council", "democratic"):
        earth.earth_trust += 0.02
    elif governance_type == "autocratic":
        earth.earth_trust -= 0.01
    if resource_avg > 0.5:
        earth.earth_trust += 0.005
    elif resource_avg < 0.2:
        earth.earth_trust -= 0.01
    for treaty in earth.treaties:
        if treaty.breached and treaty.active:
            earth.earth_trust -= 0.03
    earth.earth_trust = _clamp(earth.earth_trust)


def _tick_supply_pipeline(earth: EarthRelations, year: int) -> None:
    """Natural degradation of supply pipeline as Mars matures."""
    earth.supply_pipeline -= SUPPLY_PIPELINE_DECAY
    if earth.independence_declared:
        earth.supply_pipeline -= 0.05
    if earth.has_treaty_kind("resource_pact"):
        earth.supply_pipeline += 0.01
    earth.supply_pipeline = _clamp(earth.supply_pipeline)


def _tick_autonomy(earth: EarthRelations, year: int,
                   colonist_dicts: list[dict],
                   governance_type: str, rng: random.Random) -> None:
    """Colonists naturally want more self-governance over time."""
    if earth.independence_declared:
        return
    growth = AUTONOMY_GROWTH_BASE
    growth += year * 0.0003
    if governance_type == "democratic":
        growth += 0.005
    if earth.has_treaty_kind("emergency_aid"):
        growth -= 0.02
    if colonist_dicts:
        avg_resolve = sum(c.get("stats", {}).get("resolve", 0.5)
                          for c in colonist_dicts) / len(colonist_dicts)
        avg_paranoia = sum(c.get("stats", {}).get("paranoia", 0.5)
                           for c in colonist_dicts) / len(colonist_dicts)
        growth += (avg_resolve - 0.5) * 0.02
        growth += (avg_paranoia - 0.5) * 0.01
    earth.autonomy_desire += growth
    earth.autonomy_desire = _clamp(earth.autonomy_desire)


def _expire_treaties(earth: EarthRelations, year: int) -> list[dict]:
    """Expire treaties past their duration."""
    expired = []
    for treaty in earth.treaties:
        if treaty.active and year >= treaty.start_year + treaty.duration:
            treaty.active = False
            expired.append({
                "type": "treaty_expired", "kind": treaty.kind,
                "year": year, "start_year": treaty.start_year,
            })
    return expired


def _propose_treaty(earth: EarthRelations, year: int,
                    resource_avg: float, rng: random.Random) -> list[dict]:
    """Earth may propose a treaty if relations are decent."""
    events_out: list[dict] = []
    active_count = len(earth.active_treaties())
    if active_count >= MAX_TREATIES:
        return events_out
    if earth.independence_declared:
        return events_out
    if rng.random() > TREATY_PROPOSE_CHANCE * earth.earth_trust:
        return events_out

    if resource_avg < EMERGENCY_AID_THRESHOLD and not earth.has_treaty_kind("emergency_aid"):
        kind = "emergency_aid"
    elif not earth.has_treaty_kind("tech_transfer") and rng.random() < 0.4:
        kind = "tech_transfer"
    elif not earth.has_treaty_kind("resource_pact"):
        kind = "resource_pact"
    else:
        return events_out

    tmpl = TREATY_TEMPLATES[kind]
    events_out.append({
        "type": "treaty_proposed", "kind": kind, "year": year,
        "description": tmpl["description"],
    })
    return events_out


def _should_colony_accept_treaty(earth: EarthRelations,
                                 kind: str, rng: random.Random) -> bool:
    """Decide whether the colony accepts a proposed treaty."""
    if kind == "emergency_aid":
        return earth.autonomy_desire < 0.6 or rng.random() < 0.3
    base_accept = 0.7 - earth.autonomy_desire * 0.5
    return rng.random() < max(0.1, base_accept)


def _accept_treaty(earth: EarthRelations, kind: str, year: int) -> dict:
    """Accept and activate a treaty."""
    tmpl = TREATY_TEMPLATES[kind]
    treaty = Treaty(
        kind=kind, start_year=year,
        duration=tmpl["duration"],
        terms=dict(tmpl["terms"]),
    )
    earth.treaties.append(treaty)
    return {
        "type": "treaty_accepted", "kind": kind, "year": year,
        "duration": tmpl["duration"],
    }


def _reject_treaty(kind: str, year: int) -> dict:
    """Record a treaty rejection."""
    return {"type": "treaty_rejected", "kind": kind, "year": year}


def _check_independence_vote(earth: EarthRelations, year: int,
                             colonist_dicts: list[dict],
                             rng: random.Random) -> list[dict]:
    """Check if the colony votes for independence."""
    events_out: list[dict] = []
    if earth.independence_declared:
        return events_out
    if earth.autonomy_desire < INDEPENDENCE_THRESHOLD:
        return events_out
    if year < 20:
        return events_out

    votes_for = 0
    votes_against = 0
    for c in colonist_dicts:
        stance = compute_colonist_autonomy_stance(c, earth)
        if stance > 0.5:
            votes_for += 1
        else:
            votes_against += 1

    total = votes_for + votes_against
    if total == 0:
        return events_out

    passed = votes_for > total / 2
    events_out.append({
        "type": "independence_vote", "year": year,
        "votes_for": votes_for, "votes_against": votes_against,
        "passed": passed,
    })

    if passed:
        earth.independence_declared = True
        earth.supply_pipeline *= 0.3
        earth.earth_trust -= 0.2
        earth.earth_trust = _clamp(earth.earth_trust)
        for treaty in earth.treaties:
            if treaty.active:
                treaty.active = False
                treaty.breached = True
        events_out.append({
            "type": "independence_declared", "year": year,
            "message": "The Mars colony has declared independence from Earth.",
        })
    return events_out


# --- Public API ----------------------------------------------------------

def compute_supply_effects(earth: EarthRelations) -> dict[str, float]:
    """Compute resource bonuses from Earth's supply pipeline.

    Returns a dict of resource_name -> bonus amount.
    Supply is modulated by pipeline health, trust, and contact quality.
    """
    if earth.independence_declared:
        base = SUPPLY_BASE * 0.1
    else:
        base = SUPPLY_BASE

    multiplier = (
        earth.supply_pipeline * 0.5
        + earth.earth_trust * 0.3
        + earth.contact_quality * 0.2
    )

    treaty_bonus = 0.0
    for treaty in earth.active_treaties():
        treaty_bonus += treaty.terms.get("supply_bonus", 0.0)
        treaty_bonus += treaty.terms.get("supply_burst", 0.0)

    effective = base * multiplier + treaty_bonus
    return {
        "food": round(effective * 0.35, 6),
        "water": round(effective * 0.30, 6),
        "medicine": round(effective * 0.20, 6),
        "power": round(effective * 0.10, 6),
        "air": round(effective * 0.05, 6),
    }


def compute_colonist_autonomy_stance(colonist_dict: dict,
                                     earth: EarthRelations) -> float:
    """Compute a single colonist's stance on independence (0-1).

    Higher = more pro-independence. Based on personality stats and
    current Earth relations.
    """
    stats = colonist_dict.get("stats", {})
    resolve = stats.get("resolve", 0.5)
    paranoia = stats.get("paranoia", 0.5)
    empathy = stats.get("empathy", 0.5)
    faith = stats.get("faith", 0.5)

    base = (resolve * 0.3 + paranoia * 0.25
            - empathy * 0.15 - faith * 0.1 + 0.35)
    social_pressure = earth.autonomy_desire * 0.3
    stance = _clamp(base + social_pressure)
    return round(stance, 4)


def tick_earth_pre(earth: EarthRelations, events: list[dict],
                   year: int, rng: random.Random) -> dict[str, float]:
    """Pre-resource-tick phase: compute supply effects for this year.

    Called BEFORE tick_resources() so supply bonuses factor into
    the resource calculation.

    Args:
        earth: Current Earth relations state.
        events: This year's events (as dicts).
        year: Current simulation year.
        rng: Dedicated Earth RNG.

    Returns:
        Dict of resource bonuses to add to skill_bonuses.
    """
    _tick_contact(earth, events, year)
    return compute_supply_effects(earth)


def tick_earth_post(earth: EarthRelations, year: int,
                    year_summary: dict, colonist_dicts: list[dict],
                    governance_type: str,
                    rng: random.Random) -> list[dict]:
    """Post-tick phase: update diplomacy, treaties, autonomy, independence.

    Called AFTER all other tick phases. Mutates earth state.

    Args:
        earth: Current Earth relations state.
        year: Current simulation year.
        year_summary: Summary of this year (resources_after, etc).
        colonist_dicts: Current colonist snapshots as dicts.
        governance_type: Current governance type string.
        rng: Dedicated Earth RNG.

    Returns:
        List of diplomatic events that occurred.
    """
    diplo_events: list[dict] = []

    resource_avg = _compute_resource_avg(year_summary)

    diplo_events.extend(_expire_treaties(earth, year))
    _tick_trust(earth, governance_type, resource_avg, rng)
    _tick_supply_pipeline(earth, year)
    _tick_autonomy(earth, year, colonist_dicts, governance_type, rng)

    proposed = _propose_treaty(earth, year, resource_avg, rng)
    for prop in proposed:
        diplo_events.append(prop)
        kind = prop["kind"]
        if _should_colony_accept_treaty(earth, kind, rng):
            diplo_events.append(_accept_treaty(earth, kind, year))
        else:
            diplo_events.append(_reject_treaty(kind, year))

    diplo_events.extend(
        _check_independence_vote(earth, year, colonist_dicts, rng)
    )

    if diplo_events:
        for ev in diplo_events:
            earth.history.append(ev)

    return diplo_events


def _compute_resource_avg(year_summary: dict) -> float:
    """Extract average resource level from year summary."""
    resources = year_summary.get("resources_after", {})
    if not resources:
        return 0.5
    vals = [v for k, v in resources.items()
            if k in ("food", "water", "power", "air", "medicine")]
    return sum(vals) / max(1, len(vals)) if vals else 0.5
