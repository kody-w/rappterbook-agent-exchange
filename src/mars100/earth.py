"""
Earth Contact System for Mars-100.

Models the evolving relationship between Mars colony and Earth:
directives, supply shipments, compliance/defiance, and the path to independence.

Communication has a 1-year delay. Earth reacts to colony responses
from the previous year. Directives arrive at the start of each year
and force governance decisions before other colony business.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


EARTH_MOODS = ("supportive", "neutral", "demanding", "hostile", "collapsed")

DIRECTIVE_TYPES = (
    "supply_mission",       # Earth sends supplies — colony must allocate docking resources
    "science_experiment",   # Earth requests specific research — costs colony labor
    "personnel_transfer",   # Earth demands colonists return or accept new arrivals
    "budget_cut",           # Earth reduces support — supply probability drops
    "governance_mandate",   # Earth dictates governance change
    "recall_order",         # Earth orders full colony evacuation (late-game, hostile only)
)

RESPONSE_TYPES = ("comply", "negotiate", "reject", "ignore")


@dataclass
class Message:
    """A message in transit between Earth and Mars."""
    content: dict[str, Any]
    sent_year: int
    arrives_year: int
    direction: str  # "earth_to_mars" or "mars_to_earth"

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "sent_year": self.sent_year,
            "arrives_year": self.arrives_year,
            "direction": self.direction,
        }


@dataclass
class Directive:
    """An Earth directive that the colony must respond to."""
    dtype: str
    year_issued: int
    year_received: int
    description: str
    resource_cost: dict[str, float]
    resource_reward: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.dtype,
            "year_issued": self.year_issued,
            "year_received": self.year_received,
            "description": self.description,
            "resource_cost": self.resource_cost,
            "resource_reward": self.resource_reward,
        }


@dataclass
class EarthState:
    """Earth's evolving relationship with the Mars colony."""
    mood: str = "supportive"
    interest: float = 0.8          # 0-1, how much Earth cares about Mars
    budget: float = 0.7            # 0-1, supply ship probability
    compliance_score: float = 0.5  # rolling average of colony compliance
    directives_sent: int = 0
    directives_complied: int = 0
    directives_rejected: int = 0
    recent_rejections: int = 0     # consecutive recent rejections
    earth_crisis: bool = False     # whether Earth is in crisis
    communication_active: bool = True
    independence_declared: bool = False
    independence_year: int | None = None
    message_queue: list[Message] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mood": self.mood,
            "interest": round(self.interest, 3),
            "budget": round(self.budget, 3),
            "compliance_score": round(self.compliance_score, 3),
            "directives_sent": self.directives_sent,
            "directives_complied": self.directives_complied,
            "directives_rejected": self.directives_rejected,
            "recent_rejections": self.recent_rejections,
            "earth_crisis": self.earth_crisis,
            "communication_active": self.communication_active,
            "independence_declared": self.independence_declared,
            "independence_year": self.independence_year,
            "pending_messages": len(self.message_queue),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EarthState:
        return cls(
            mood=d.get("mood", "supportive"),
            interest=d.get("interest", 0.8),
            budget=d.get("budget", 0.7),
            compliance_score=d.get("compliance_score", 0.5),
            directives_sent=d.get("directives_sent", 0),
            directives_complied=d.get("directives_complied", 0),
            directives_rejected=d.get("directives_rejected", 0),
            recent_rejections=d.get("recent_rejections", 0),
            earth_crisis=d.get("earth_crisis", False),
            communication_active=d.get("communication_active", True),
            independence_declared=d.get("independence_declared", False),
            independence_year=d.get("independence_year"),
        )


def _mood_from_scores(interest: float, compliance: float,
                      crisis: bool) -> str:
    """Derive Earth's mood from its state variables."""
    if crisis:
        return "collapsed"
    if interest < 0.2:
        return "hostile"
    if compliance < 0.3:
        return "hostile"
    if compliance < 0.45:
        return "demanding"
    if interest < 0.5 or compliance < 0.55:
        return "neutral"
    return "supportive"


def earth_tick(state: EarthState, year: int,
               rng: random.Random) -> Directive | None:
    """Advance Earth's state by one year and maybe generate a directive.

    Returns a Directive if Earth has something to say, None otherwise.
    After independence or communication loss, returns None.
    """
    if state.independence_declared or not state.communication_active:
        return None

    # Deliver messages that arrive this year
    arrived = [m for m in state.message_queue if m.arrives_year <= year]
    state.message_queue = [m for m in state.message_queue if m.arrives_year > year]
    for msg in arrived:
        if msg.direction == "mars_to_earth":
            _process_colony_response(state, msg.content, rng)

    # Earth crisis check (rare — war, economic collapse, pandemic)
    if not state.earth_crisis and year > 20 and rng.random() < 0.008:
        state.earth_crisis = True
        state.interest *= 0.3
        state.budget *= 0.2

    # Earth crisis recovery (slow)
    if state.earth_crisis and rng.random() < 0.05:
        state.earth_crisis = False
        state.interest = min(1.0, state.interest + 0.1)
        state.budget = min(1.0, state.budget + 0.1)

    # Natural interest decay — Earth has its own problems
    if year > 5:
        state.interest = max(0.0, state.interest - rng.uniform(0.002, 0.008))

    # Compliance feedback: compliance boosts interest, rejection erodes it
    if state.compliance_score > 0.6:
        state.interest = min(1.0, state.interest + 0.005)
    elif state.compliance_score < 0.35:
        state.interest = max(0.0, state.interest - 0.01)

    # Budget tracks interest with noise
    target_budget = state.interest * 0.9
    state.budget += (target_budget - state.budget) * 0.2 + rng.gauss(0, 0.02)
    state.budget = max(0.0, min(1.0, state.budget))

    # Update mood
    state.mood = _mood_from_scores(state.interest, state.compliance_score,
                                   state.earth_crisis)

    # Generate directive if Earth is engaged enough
    return _maybe_generate_directive(state, year, rng)


def _process_colony_response(state: EarthState, content: dict,
                             rng: random.Random) -> None:
    """Process a colony response that has arrived at Earth."""
    response = content.get("response", "ignore")
    if response in ("comply", "negotiate"):
        state.compliance_score = min(
            1.0, state.compliance_score * 0.8 + 0.2 * (1.0 if response == "comply" else 0.7))
        state.recent_rejections = 0
        if response == "comply":
            state.directives_complied += 1
    elif response in ("reject", "ignore"):
        state.compliance_score = max(0.0, state.compliance_score * 0.8)
        state.recent_rejections += 1
        state.directives_rejected += 1


def _maybe_generate_directive(state: EarthState, year: int,
                              rng: random.Random) -> Directive | None:
    """Possibly generate a directive from Earth."""
    if state.earth_crisis:
        return None  # Earth too busy
    if state.interest < 0.15:
        return None  # Earth doesn't care

    # Directive probability scales with interest
    prob = 0.3 + state.interest * 0.3
    if state.mood == "demanding":
        prob += 0.15
    elif state.mood == "hostile":
        prob += 0.2

    if rng.random() > prob:
        return None

    return _generate_directive(state, year, rng)


def _generate_directive(state: EarthState, year: int,
                        rng: random.Random) -> Directive:
    """Generate a specific directive based on Earth's mood and state."""
    state.directives_sent += 1

    # Weight directive types by mood
    weights: dict[str, float] = {
        "supply_mission": 3.0 if state.mood == "supportive" else 1.0,
        "science_experiment": 2.0,
        "personnel_transfer": 1.5 if year > 15 else 0.5,
        "budget_cut": 2.5 if state.mood in ("demanding", "hostile") else 0.3,
        "governance_mandate": 1.5 if state.mood == "hostile" else 0.2,
        "recall_order": 1.0 if state.mood == "hostile" and year > 50 else 0.0,
    }

    total = sum(weights.values())
    r = rng.random() * total
    cumulative = 0.0
    chosen = "supply_mission"
    for dtype, w in weights.items():
        cumulative += w
        if r <= cumulative:
            chosen = dtype
            break

    return _build_directive(chosen, year, state, rng)


def _build_directive(dtype: str, year: int, state: EarthState,
                     rng: random.Random) -> Directive:
    """Construct a directive with description, costs, and rewards."""
    templates: dict[str, dict[str, Any]] = {
        "supply_mission": {
            "descriptions": [
                "Earth confirms supply ship for next launch window.",
                "Cargo vessel en route — ETA 9 months with equipment and rations.",
                "Emergency resupply authorized by UNSA council.",
            ],
            "cost": {"power": 0.02},
            "reward": {"food": 0.15, "medicine": 0.1, "water": 0.05},
        },
        "science_experiment": {
            "descriptions": [
                "Earth requests geological survey of Elysium basin.",
                "UNSA mandates atmospheric composition study — report due next year.",
                "University consortium needs regolith samples for Earth lab analysis.",
            ],
            "cost": {"food": 0.03, "power": 0.05},
            "reward": {"medicine": 0.05},
        },
        "personnel_transfer": {
            "descriptions": [
                "Earth requests two colonists return for debriefing.",
                "Three new personnel assigned to colony — prepare quarters.",
                "Medical specialist reassigned to Lunar Gateway — must comply.",
            ],
            "cost": {"food": 0.05, "medicine": 0.03},
            "reward": {"food": 0.02},
        },
        "budget_cut": {
            "descriptions": [
                "UNSA budget reduced 30% — supply frequency halved.",
                "Political shift on Earth: Mars program funding slashed.",
                "Economic recession on Earth — non-essential programs suspended.",
            ],
            "cost": {},
            "reward": {},
        },
        "governance_mandate": {
            "descriptions": [
                "Earth orders colony to adopt appointed governor model.",
                "UNSA demands veto power over colony constitutional changes.",
                "Earth insists on quarterly governance audits.",
            ],
            "cost": {"food": 0.02},
            "reward": {"medicine": 0.05},
        },
        "recall_order": {
            "descriptions": [
                "UNSA orders full colony evacuation — return vessel incoming.",
                "Earth declares Mars program terminated — all personnel recalled.",
                "Emergency: Earth demands immediate colony shutdown.",
            ],
            "cost": {},
            "reward": {},
        },
    }

    tmpl = templates.get(dtype, templates["supply_mission"])
    desc = rng.choice(tmpl["descriptions"])
    cost = {k: v * rng.uniform(0.8, 1.2) for k, v in tmpl["cost"].items()}
    reward = {k: v * rng.uniform(0.8, 1.2) for k, v in tmpl["reward"].items()}

    # Budget cuts reduce state
    if dtype == "budget_cut":
        state.budget = max(0.0, state.budget - rng.uniform(0.1, 0.25))

    return Directive(
        dtype=dtype, year_issued=year,
        year_received=year + 1,  # 1-year communication delay
        description=desc,
        resource_cost=cost, resource_reward=reward,
    )


def colony_decides(directive: Directive,
                   gov_type: str,
                   leader_id: str | None,
                   council_ids: list[str],
                   active_colonists: list,
                   social_cohesion: float,
                   resource_avg: float,
                   rng: random.Random) -> str:
    """Determine how the colony responds to an Earth directive.

    Returns one of: 'comply', 'negotiate', 'reject', 'ignore'.
    Decision method depends on governance type.
    """
    # Base disposition: how much does the colony WANT to comply?
    disposition = 0.5

    # Supply missions are welcome, recall orders are not
    type_bias: dict[str, float] = {
        "supply_mission": 0.3,
        "science_experiment": 0.1,
        "personnel_transfer": -0.1,
        "budget_cut": -0.2,
        "governance_mandate": -0.3,
        "recall_order": -0.5,
    }
    disposition += type_bias.get(directive.dtype, 0.0)

    # Low resources → more likely to comply (need supplies)
    if resource_avg < 0.3:
        disposition += 0.2
    elif resource_avg > 0.7:
        disposition -= 0.1

    # High cohesion → colony acts collectively → more likely to reject (solidarity)
    if social_cohesion > 0.7:
        disposition -= 0.1
    elif social_cohesion < 0.3:
        disposition += 0.1

    # Governance type modifies decision process
    noise = rng.gauss(0, 0.1)
    if gov_type == "dictator" and leader_id is not None:
        # Dictator decides alone — personality-dependent (approximated by random)
        score = disposition + noise + rng.gauss(0, 0.15)
    elif gov_type == "council":
        # Council averages — less variance
        score = disposition + noise * 0.5
    elif gov_type == "consensus":
        # Consensus needs strong agreement — pulls toward center
        score = disposition * 0.7 + noise * 0.3
    elif gov_type == "anarchy":
        # Anarchy is chaotic
        score = disposition + noise + rng.gauss(0, 0.25)
    else:
        score = disposition + noise

    if score > 0.6:
        return "comply"
    elif score > 0.35:
        return "negotiate"
    elif score > 0.1:
        return "reject"
    return "ignore"


def apply_directive_effects(directive: Directive, response: str,
                            resources_dict: dict[str, float]) -> dict[str, float]:
    """Apply resource effects of a directive based on colony response.

    Returns dict of resource deltas applied.
    """
    deltas: dict[str, float] = {}

    if response == "comply":
        for k, v in directive.resource_cost.items():
            if k in resources_dict:
                deltas[k] = deltas.get(k, 0.0) - v
        for k, v in directive.resource_reward.items():
            if k in resources_dict:
                deltas[k] = deltas.get(k, 0.0) + v
    elif response == "negotiate":
        # Partial compliance — 50% costs, 70% rewards
        for k, v in directive.resource_cost.items():
            if k in resources_dict:
                deltas[k] = deltas.get(k, 0.0) - v * 0.5
        for k, v in directive.resource_reward.items():
            if k in resources_dict:
                deltas[k] = deltas.get(k, 0.0) + v * 0.7
    # reject/ignore: no effects

    return deltas


def check_independence(state: EarthState, colony_population: int,
                       resource_avg: float, social_cohesion: float,
                       year: int, rng: random.Random) -> bool:
    """Check whether the colony declares independence from Earth.

    Requires: recent rejection streak >= 3, colony is self-sufficient
    (resources > 0.5, population > 8), year > 25, and a random roll
    weighted by cohesion and streak length.
    """
    if state.independence_declared:
        return False
    if state.recent_rejections < 3:
        return False
    if year < 25:
        return False
    if resource_avg < 0.45:
        return False
    if colony_population < 8:
        return False

    prob = (
        0.05
        + (state.recent_rejections - 3) * 0.08
        + social_cohesion * 0.15
        + (1.0 - state.interest) * 0.1
    )
    prob = min(0.6, prob)  # cap to avoid certainty

    if rng.random() < prob:
        state.independence_declared = True
        state.independence_year = year
        state.communication_active = True  # comms stay open, just no directives
        return True
    return False


def supply_ship_arrives(state: EarthState, year: int,
                        rng: random.Random) -> dict[str, float] | None:
    """Check if an Earth supply ship arrives this year.

    Post-independence: no supply ships.
    Budget determines probability.
    Returns resource bonuses if ship arrives, None otherwise.
    """
    if state.independence_declared:
        return None
    if state.earth_crisis:
        return None
    if rng.random() > state.budget * 0.4:  # base ~28% chance at full budget
        return None

    supplies: dict[str, float] = {
        "food": rng.uniform(0.05, 0.12),
        "medicine": rng.uniform(0.03, 0.08),
        "water": rng.uniform(0.02, 0.05),
        "power": rng.uniform(0.01, 0.03),
    }
    return supplies
