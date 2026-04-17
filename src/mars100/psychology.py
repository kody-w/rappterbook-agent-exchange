"""
Psychology organ for Mars-100 colony simulation (engine v8.0).

Models colonist mental health: stress, morale, bonds, and breakdowns.

Key dynamics:
  - Stress accumulates from crises, resource scarcity, inequality,
    bonded-colonist loss, and meta-awareness events
  - Personality modulates stress: resolve reduces, paranoia amplifies
  - Morale tracks smoothed inverse of stress with social-cohesion boost
  - Bonds form between colonists with sustained high mutual trust
  - Breakdowns force rest and reset stress (lagged: check this year,
    accumulate for next year)
  - Colony average morale feeds back into birth probability

All effects use a 1-year lag: stress accumulated this year affects
breakdown checks NEXT year.  This avoids tick-order coupling.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# -- constants ---------------------------------------------------------------

MAX_BONDS = 3
BOND_TRUST_THRESHOLD = 0.70
BOND_AFFECTION_THRESHOLD = 0.55
BOND_FORMATION_PROB = 0.12

BREAKDOWN_STRESS_THRESHOLD = 0.80
BREAKDOWN_SCALE = 3.5
BREAKDOWN_RESET_STRESS = 0.45
BREAKDOWN_MORALE_DROP = 0.20

STRESS_DECAY = 0.05
MORALE_INERTIA = 0.80
MORALE_STRESS_WEIGHT = 0.15
MORALE_SOCIAL_WEIGHT = 0.05

GRIEF_DEATH = 0.30
GRIEF_EXILE = 0.20
STRESS_PER_CRITICAL_RESOURCE = 0.08
STRESS_INEQUALITY_SCALE = 0.04
STRESS_EVENT_SCALE = 0.15
STRESS_META_AWARENESS = 0.20
STRESS_GOV_CHANGE = 0.05
STRESS_TRADITION_COMFORT = 0.01

RESOLVE_STRESS_REDUCTION = 0.4
PARANOIA_STRESS_AMPLIFICATION = 0.3
EMPATHY_GRIEF_AMPLIFICATION = 0.2


# -- data classes ------------------------------------------------------------

@dataclass
class PsychTickResult:
    """Result of one year of psychology processing."""
    avg_stress: float
    avg_morale: float
    breakdowns: list[dict]
    bonds_formed: list[dict]
    bonds_broken: list[dict]
    grief_events: list[dict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_stress": round(self.avg_stress, 4),
            "avg_morale": round(self.avg_morale, 4),
            "breakdowns": len(self.breakdowns),
            "breakdown_details": self.breakdowns,
            "bonds_formed": len(self.bonds_formed),
            "bonds_broken": len(self.bonds_broken),
            "grief_events": len(self.grief_events),
        }


# -- pure functions ----------------------------------------------------------

def accumulate_stress(
    current_stress: float,
    resolve: float,
    paranoia: float,
    empathy: float,
    event_severities: list[float],
    critical_resource_count: int,
    gini: float,
    bonded_deaths: int,
    bonded_exiles: int,
    meta_aware: bool,
    gov_changed: bool,
    tradition_count: int,
) -> float:
    """Compute new stress for one colonist (additive sources, personality-modulated)."""
    delta = 0.0
    for sev in event_severities:
        if sev > 0.3:
            delta += sev * STRESS_EVENT_SCALE
    delta += critical_resource_count * STRESS_PER_CRITICAL_RESOURCE
    if gini > 0.5:
        delta += (gini - 0.5) * STRESS_INEQUALITY_SCALE
    grief = bonded_deaths * GRIEF_DEATH + bonded_exiles * GRIEF_EXILE
    grief *= (1.0 + empathy * EMPATHY_GRIEF_AMPLIFICATION)
    delta += grief
    if meta_aware:
        delta += STRESS_META_AWARENESS
    if gov_changed:
        delta += STRESS_GOV_CHANGE
    delta -= tradition_count * STRESS_TRADITION_COMFORT
    personality_mod = (1.0
                       - resolve * RESOLVE_STRESS_REDUCTION
                       + paranoia * PARANOIA_STRESS_AMPLIFICATION)
    delta *= max(0.1, personality_mod)
    delta -= STRESS_DECAY
    return max(0.0, min(1.0, current_stress + delta))


def update_morale(
    current_morale: float,
    stress: float,
    social_cohesion: float,
) -> float:
    """Update morale with inertia — lags behind stress."""
    target = (1.0 - stress) * MORALE_STRESS_WEIGHT + social_cohesion * MORALE_SOCIAL_WEIGHT
    new_morale = current_morale * MORALE_INERTIA + target
    return max(0.0, min(1.0, new_morale))


def check_breakdown(stress: float, rng: random.Random) -> bool:
    """Check if colonist has a psychological breakdown."""
    if stress <= BREAKDOWN_STRESS_THRESHOLD:
        return False
    prob = (stress - BREAKDOWN_STRESS_THRESHOLD) * BREAKDOWN_SCALE
    return rng.random() < prob


def process_grief(
    bonds: list[str],
    dead_ids: set[str],
    exiled_ids: set[str],
) -> tuple[list[str], int, int]:
    """Remove departed from bonds, return (cleaned_bonds, death_count, exile_count)."""
    deaths = exiles = 0
    cleaned = []
    for bid in bonds:
        if bid in dead_ids:
            deaths += 1
        elif bid in exiled_ids:
            exiles += 1
        else:
            cleaned.append(bid)
    return cleaned, deaths, exiles


def form_bonds(
    active_colonists: list[Any],
    social_graph: Any,
    rng: random.Random,
) -> list[dict]:
    """Form symmetric bonds between qualifying pairs."""
    formed: list[dict] = []
    active = [c for c in active_colonists if c.is_active()]
    for i, c in enumerate(active):
        for other in active[i + 1:]:
            if len(c.bonds) >= MAX_BONDS or len(other.bonds) >= MAX_BONDS:
                continue
            if c.id in other.bonds or other.id in c.bonds:
                continue
            rel_ab = social_graph.get(c.id, other.id)
            rel_ba = social_graph.get(other.id, c.id)
            avg_trust = (rel_ab.trust + rel_ba.trust) / 2
            avg_affection = (rel_ab.affection + rel_ba.affection) / 2
            if (avg_trust >= BOND_TRUST_THRESHOLD
                    and avg_affection >= BOND_AFFECTION_THRESHOLD
                    and rng.random() < BOND_FORMATION_PROB):
                c.bonds.append(other.id)
                other.bonds.append(c.id)
                formed.append({"a": c.id, "b": other.id})
    return formed


def compute_psych_pressure(avg_stress: float, avg_morale: float) -> dict[str, float]:
    """Action-weight modifiers from colony-wide psychological state."""
    pressure: dict[str, float] = {}
    if avg_morale < 0.4:
        severity = (0.4 - avg_morale) * 2.0
        pressure["mediate"] = 0.15 * severity
        pressure["pray"] = 0.10 * severity
    if avg_stress > 0.6:
        severity = (avg_stress - 0.6) * 2.0
        pressure["rest"] = 0.10 * severity
    return pressure


# -- main tick ---------------------------------------------------------------

def tick_psychology(
    colonists: list[Any],
    social: Any,
    resources: Any,
    events: list[Any],
    year: int,
    culture: Any,
    departed_ids: set[str],
    rng: random.Random,
) -> dict[str, Any]:
    """Run one year of psychology for the colony.

    Called AFTER deaths/exiles so grief can be processed.
    Returns a dict (stored directly as YearResult.psychology).
    """
    dead_ids = departed_ids
    exiled_ids: set[str] = set()  # exiles are included in departed_ids
    event_severities = [getattr(e, "severity", e.get("severity", 0.0))
                        if isinstance(e, dict) else e.severity
                        for e in events]

    active = [c for c in colonists if c.is_active()]
    if not active:
        return PsychTickResult(0.0, 0.0, [], [], [], []).to_dict()

    # Resource criticality
    res_dict = resources.to_dict() if hasattr(resources, "to_dict") else {}
    critical_count = sum(1 for v in res_dict.values()
                         if isinstance(v, (int, float)) and v < 0.25)

    # Culture traditions
    tradition_count = len(culture.traditions) if hasattr(culture, "traditions") else 0

    # Gini from economics (access via colonist wallets)
    wealth_values = []
    for c in active:
        if hasattr(c, "wallet"):
            wealth_values.append(c.wallet.total_wealth())
    gini = _simple_gini(wealth_values) if len(wealth_values) >= 2 else 0.0

    # Social cohesion
    active_ids = [c.id for c in active]
    cohesion = social.colony_cohesion(active_ids) if hasattr(social, "colony_cohesion") else 0.5

    # Phase 1: grief — clean bonds, count losses per colonist
    grief_events: list[dict] = []
    bonds_broken: list[dict] = []
    grief_map: dict[str, tuple[int, int]] = {}
    for c in active:
        cleaned, bd, be = process_grief(c.bonds, dead_ids, exiled_ids)
        if bd > 0 or be > 0:
            grief_events.append({"colonist_id": c.id, "bonded_deaths": bd,
                                  "bonded_exiles": be, "year": year})
            for bid in c.bonds:
                if bid in dead_ids or bid in exiled_ids:
                    bonds_broken.append({"colonist_id": c.id, "lost_bond": bid,
                                          "year": year})
            grief_map[c.id] = (bd, be)
        c.bonds = cleaned

    # Phase 2: stress accumulation (for NEXT year's breakdown check)
    for c in active:
        bd, be = grief_map.get(c.id, (0, 0))
        c.stress = accumulate_stress(
            current_stress=c.stress,
            resolve=c.stats.resolve,
            paranoia=c.stats.paranoia,
            empathy=c.stats.empathy,
            event_severities=event_severities,
            critical_resource_count=critical_count,
            gini=gini,
            bonded_deaths=bd,
            bonded_exiles=be,
            meta_aware=False,  # meta-awareness tracked separately
            gov_changed=False,  # simplified for v8.0
            tradition_count=tradition_count,
        )

    # Phase 3: morale update
    for c in active:
        c.morale = update_morale(c.morale, c.stress, cohesion)

    # Phase 4: breakdown check (based on CURRENT stress, i.e. just-accumulated)
    breakdowns: list[dict] = []
    for c in active:
        if c.birth_year == year:
            continue
        if check_breakdown(c.stress, rng):
            breakdowns.append({"colonist_id": c.id, "name": c.name, "year": year})
            c.stress = BREAKDOWN_RESET_STRESS
            c.morale = BREAKDOWN_MORALE_DROP
            c.breakdown_year = year
            c.add_memory(year, "Psychological breakdown — forced rest", -0.8)

    # Phase 5: bond formation
    bonds_formed = form_bonds(active, social, rng)

    # Averages
    stresses = [c.stress for c in active]
    morales = [c.morale for c in active]
    avg_stress = sum(stresses) / len(stresses)
    avg_morale = sum(morales) / len(morales)

    result = PsychTickResult(
        avg_stress=avg_stress,
        avg_morale=avg_morale,
        breakdowns=breakdowns,
        bonds_formed=bonds_formed,
        bonds_broken=bonds_broken,
        grief_events=grief_events,
    )
    return result.to_dict()


def _simple_gini(values: list[float]) -> float:
    """Quick Gini coefficient for wealth distribution."""
    n = len(values)
    if n < 2:
        return 0.0
    s = sorted(values)
    total = sum(s)
    if total <= 0:
        return 0.0
    weighted = sum((2 * (i + 1) - n - 1) * w for i, w in enumerate(s))
    return max(0.0, min(1.0, weighted / (n * total) * n / (n - 1)))
