"""
Psychology organ for Mars-100 colony simulation (engine v8.0).

Tracks per-colonist psychological state: stress, loneliness, purpose.
Morale is a *derived* summary -- not an independent axis -- preventing
double-counting.  Stable personality traits (resolve, paranoia, faith)
modify recovery rates, not duplicate outcomes.

Phase 1 scope:
  - PsychState per colonist (stress, loneliness, purpose -> morale)
  - tick_psychology(): update psych state from year context
  - Mental health crises with cooldown anti-spam
  - ONE downstream hook: morale affects death-rate multiplier
  - Defer: action-selection perturbation (v9+)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# -- constants ---------------------------------------------------------------

STRESS_DECAY = 0.10
LONELINESS_DECAY = 0.05
PURPOSE_DECAY = 0.03

STRESS_CAP_DELTA = 0.25
LONELINESS_CAP_DELTA = 0.20
PURPOSE_CAP_DELTA = 0.20

CRISIS_THRESHOLD = 0.85
CRISIS_PROBABILITY = 0.30
CRISIS_COOLDOWN = 3

MORALE_DEATH_THRESHOLD = 0.20
MORALE_DEATH_MULTIPLIER = 2.0

ACTION_STRESS: dict[str, float] = {
    "terraform": 0.04, "farm": 0.02, "mediate": -0.03, "code": 0.03,
    "pray": -0.05, "sabotage": 0.08, "cooperate": -0.02, "hoard": 0.03,
    "explore": 0.06, "rest": -0.08, "research": 0.04,
}

ACTION_PURPOSE: dict[str, float] = {
    "terraform": 0.04, "farm": 0.03, "mediate": 0.03, "code": 0.04,
    "pray": 0.02, "sabotage": -0.04, "cooperate": 0.03, "hoard": -0.02,
    "explore": 0.05, "rest": -0.01, "research": 0.05,
}


# -- data classes ------------------------------------------------------------

@dataclass
class PsychState:
    """Per-colonist psychological state.  Volatile (changes each year).

    stress:     0 = calm, 1 = breaking point
    loneliness: 0 = connected, 1 = isolated
    purpose:    0 = aimless, 1 = deeply driven
    last_crisis_year: year of most recent mental health crisis (-999 = never)
    """
    stress: float = 0.15
    loneliness: float = 0.20
    purpose: float = 0.50
    last_crisis_year: int = -999

    @property
    def morale(self) -> float:
        """Derived morale: high purpose and low stress/loneliness = high morale."""
        raw = (self.purpose * 0.5
               + (1.0 - self.stress) * 0.30
               + (1.0 - self.loneliness) * 0.20)
        return max(0.0, min(1.0, raw))

    def to_dict(self) -> dict:
        return {
            "stress": round(self.stress, 4),
            "loneliness": round(self.loneliness, 4),
            "purpose": round(self.purpose, 4),
            "morale": round(self.morale, 4),
            "last_crisis_year": self.last_crisis_year,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PsychState:
        return cls(
            stress=d.get("stress", 0.15),
            loneliness=d.get("loneliness", 0.20),
            purpose=d.get("purpose", 0.50),
            last_crisis_year=d.get("last_crisis_year", -999),
        )


@dataclass
class CrisisEvent:
    """A mental health crisis for one colonist."""
    colonist_id: str
    year: int
    stress_level: float
    forced_rest: bool = True

    def to_dict(self) -> dict:
        return {
            "colonist_id": self.colonist_id,
            "year": self.year,
            "stress_level": round(self.stress_level, 4),
            "forced_rest": self.forced_rest,
        }


@dataclass
class PsychTickResult:
    """Result of one year's psychology tick."""
    snapshots: dict[str, dict] = field(default_factory=dict)
    crises: list[CrisisEvent] = field(default_factory=list)
    colony_morale: float = 0.5
    colony_stress: float = 0.15
    bottom_quartile_morale: float = 0.5

    def to_dict(self) -> dict:
        return {
            "snapshots": self.snapshots,
            "crises": [c.to_dict() for c in self.crises],
            "colony_morale": round(self.colony_morale, 4),
            "colony_stress": round(self.colony_stress, 4),
            "bottom_quartile_morale": round(self.bottom_quartile_morale, 4),
        }


# -- pure helpers ------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _cap_delta(delta: float, cap: float) -> float:
    """Clamp a per-tick delta to +/-cap."""
    return max(-cap, min(cap, delta))


def compute_stress_delta(
    action: str,
    event_severity: float,
    resource_avg: float,
    resolve: float,
) -> float:
    """Compute stress change for one colonist-year.

    Resolve modifies recovery: high-resolve colonists shed stress faster.
    """
    action_effect = ACTION_STRESS.get(action, 0.0)
    event_effect = event_severity * 0.15
    resource_effect = max(0.0, 0.3 - resource_avg) * 0.2
    natural_decay = -STRESS_DECAY * (0.5 + resolve * 0.5)
    raw = action_effect + event_effect + resource_effect + natural_decay
    return _cap_delta(raw, STRESS_CAP_DELTA)


def compute_loneliness_delta(
    social_connections: int,
    avg_trust: float,
    earth_contact: bool,
    empathy: float,
) -> float:
    """Compute loneliness change for one colonist-year.

    Empathy modifies: empathetic colonists form bonds faster.
    """
    connection_effect = -0.02 * min(social_connections, 5)
    trust_effect = -avg_trust * 0.10
    earth_effect = -0.05 if earth_contact else 0.03
    empathy_bonus = -empathy * 0.03
    natural_decay = -LONELINESS_DECAY
    raw = connection_effect + trust_effect + earth_effect + empathy_bonus + natural_decay
    return _cap_delta(raw, LONELINESS_CAP_DELTA)


def compute_purpose_delta(
    action: str,
    infra_completed: bool,
    gov_participated: bool,
    subsim_ran: bool,
    faith: float,
) -> float:
    """Compute purpose change for one colonist-year.

    Faith modifies: faithful colonists maintain purpose in adversity.
    """
    action_effect = ACTION_PURPOSE.get(action, 0.0)
    infra_bonus = 0.06 if infra_completed else 0.0
    gov_bonus = 0.03 if gov_participated else 0.0
    subsim_bonus = 0.04 if subsim_ran else 0.0
    faith_floor = faith * 0.02
    natural_decay = -PURPOSE_DECAY
    raw = action_effect + infra_bonus + gov_bonus + subsim_bonus + faith_floor + natural_decay
    return _cap_delta(raw, PURPOSE_CAP_DELTA)


def check_crisis(
    psych: PsychState,
    year: int,
    rng: random.Random,
) -> bool:
    """Check if a colonist enters a mental health crisis."""
    if psych.stress < CRISIS_THRESHOLD:
        return False
    if year - psych.last_crisis_year < CRISIS_COOLDOWN:
        return False
    return rng.random() < CRISIS_PROBABILITY


def compute_colony_morale(psych_states: list[PsychState]) -> float:
    """Mean morale across all active colonists."""
    if not psych_states:
        return 0.5
    return sum(p.morale for p in psych_states) / len(psych_states)


def compute_bottom_quartile_morale(psych_states: list[PsychState]) -> float:
    """Mean morale of the bottom 25% -- surfaces suffering minorities."""
    if not psych_states:
        return 0.5
    morales = sorted(p.morale for p in psych_states)
    q_size = max(1, len(morales) // 4)
    return sum(morales[:q_size]) / q_size


def death_rate_modifier(morale: float) -> float:
    """Compute death-rate multiplier from morale.

    Returns 1.0 (no change) for healthy morale, up to
    MORALE_DEATH_MULTIPLIER for critically low morale.
    """
    if morale >= MORALE_DEATH_THRESHOLD:
        return 1.0
    t = morale / MORALE_DEATH_THRESHOLD
    return 1.0 + (MORALE_DEATH_MULTIPLIER - 1.0) * (1.0 - t)


# -- main tick ---------------------------------------------------------------

@dataclass
class ColonistPsychContext:
    """Year context for a single colonist's psych update."""
    colonist_id: str
    action: str
    event_severity: float
    resource_avg: float
    social_connections: int
    avg_trust: float
    earth_contact: bool
    infra_completed: bool
    gov_participated: bool
    subsim_ran: bool
    resolve: float
    empathy: float
    faith: float
    paranoia: float


def tick_psychology(
    psych_map: dict[str, PsychState],
    contexts: list[ColonistPsychContext],
    year: int,
    rng: random.Random,
) -> PsychTickResult:
    """Run one year of psychological updates.  Mutates psych_map in place."""
    crises: list[CrisisEvent] = []
    snapshots: dict[str, dict] = {}

    for ctx in contexts:
        cid = ctx.colonist_id
        psych = psych_map.get(cid)
        if psych is None:
            psych = PsychState()
            psych_map[cid] = psych

        stress_d = compute_stress_delta(
            ctx.action, ctx.event_severity, ctx.resource_avg, ctx.resolve)
        loneliness_d = compute_loneliness_delta(
            ctx.social_connections, ctx.avg_trust, ctx.earth_contact, ctx.empathy)
        purpose_d = compute_purpose_delta(
            ctx.action, ctx.infra_completed, ctx.gov_participated,
            ctx.subsim_ran, ctx.faith)

        if stress_d > 0:
            stress_d *= (1.0 + ctx.paranoia * 0.3)
            stress_d = _cap_delta(stress_d, STRESS_CAP_DELTA)

        psych.stress = _clamp(psych.stress + stress_d)
        psych.loneliness = _clamp(psych.loneliness + loneliness_d)
        psych.purpose = _clamp(psych.purpose + purpose_d)

        if check_crisis(psych, year, rng):
            crisis = CrisisEvent(colonist_id=cid, year=year,
                                 stress_level=psych.stress)
            crises.append(crisis)
            psych.last_crisis_year = year
            psych.stress = _clamp(psych.stress - 0.20)

        snapshots[cid] = psych.to_dict()

    active_psych = [psych_map[ctx.colonist_id] for ctx in contexts]
    colony_morale = compute_colony_morale(active_psych)
    colony_stress_avg = (sum(p.stress for p in active_psych) / len(active_psych)
                         if active_psych else 0.15)
    bq_morale = compute_bottom_quartile_morale(active_psych)

    return PsychTickResult(
        snapshots=snapshots,
        crises=crises,
        colony_morale=colony_morale,
        colony_stress=colony_stress_avg,
        bottom_quartile_morale=bq_morale,
    )
