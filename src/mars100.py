"""
Mars-100 — A recursive colony simulation using LisPy.

10 agent-colonists. 100 Mars years. Sub-simulations up to 3 levels deep.
Colonists may spawn nested LisPy sims to model governance proposals,
economic scenarios, or survival strategies before committing.

Each sim frame = 1 Martian year (~687 Earth days).
The output of year N is the input to year N+1.

This is Turtles All the Way Down (Amendment XIII) made concrete.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.lispy import (
    EvalContext,
    LispyError,
    make_env,
    run,
    safe_eval,
    serialize,
    to_sexpr,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELEMENTS = ["fire", "water", "earth", "air"]
STAT_NAMES = ["resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia"]
SKILL_NAMES = ["terraforming", "hydroponics", "mediation", "coding", "prayer", "sabotage"]

ELEMENT_AFFINITIES: dict[str, dict[str, float]] = {
    "fire":  {"resolve": 0.15, "paranoia": 0.10, "improvisation": 0.05},
    "water": {"empathy": 0.15, "faith": 0.05, "improvisation": 0.10},
    "earth": {"hoarding": 0.15, "resolve": 0.10, "empathy": 0.05},
    "air":   {"improvisation": 0.15, "faith": 0.10, "paranoia": -0.05},
}

RESOURCE_NAMES = ["food", "water", "oxygen", "power", "materials"]
RESOURCE_CAP = 200.0

COLONIST_NAMES = [
    "Ares", "Selene", "Kai", "Zephyr", "Petra",
    "Ignis", "Nyx", "Terra", "Aether", "Marina",
]

EVENT_TYPES = [
    "dust_storm", "solar_flare", "equipment_failure", "resource_strike",
    "earth_contact", "alien_signal", "meteor_shower", "ice_vein",
    "fungal_bloom", "psychic_event", "calm_year",
]

GOVERNANCE_TYPES = [
    "anarchy", "council", "monarchy", "democracy", "technocracy", "theocracy",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Colonist:
    """An individual colonist with stats, skills, memory, and alive/dead state."""
    id: str
    name: str
    element: str
    stats: dict[str, float] = field(default_factory=dict)
    skills: dict[str, float] = field(default_factory=dict)
    relationships: dict[str, float] = field(default_factory=dict)
    memory: list[str] = field(default_factory=list)
    alive: bool = True
    year_of_death: int | None = None
    cause_of_death: str | None = None
    karma: float = 0.5
    sim_aware: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "element": self.element,
            "stats": dict(self.stats), "skills": dict(self.skills),
            "relationships": dict(self.relationships),
            "memory": list(self.memory),
            "alive": self.alive,
            "year_of_death": self.year_of_death,
            "cause_of_death": self.cause_of_death,
            "karma": round(self.karma, 4),
            "sim_aware": self.sim_aware,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Colonist:
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class SubSimLog:
    """Log entry for a sub-simulation run by a colonist."""
    year: int
    colonist: str
    depth: int
    expression: str
    result: str
    purpose: str

    def to_dict(self) -> dict:
        return {
            "year": self.year, "colonist": self.colonist,
            "depth": self.depth, "expression": self.expression,
            "result": self.result, "purpose": self.purpose,
        }


@dataclass
class Proposal:
    """A governance proposal from a colonist."""
    id: str
    proposer: str
    year: int
    proposal_type: str
    description: str
    votes_for: list[str] = field(default_factory=list)
    votes_against: list[str] = field(default_factory=list)
    inertia: int = 0
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "proposer": self.proposer, "year": self.year,
            "proposal_type": self.proposal_type, "description": self.description,
            "votes_for": list(self.votes_for),
            "votes_against": list(self.votes_against),
            "inertia": self.inertia, "resolved": self.resolved,
        }


@dataclass
class ColonyState:
    """The complete state of the colony at a point in time."""
    year: int
    colonists: list[Colonist]
    rng_seed: int
    resources: dict[str, float] = field(default_factory=lambda: {
        "food": 100.0, "water": 100.0, "oxygen": 100.0,
        "power": 100.0, "materials": 80.0,
    })
    governance: list[Proposal] = field(default_factory=list)
    governance_type: str = "anarchy"
    active_laws: list[str] = field(default_factory=list)
    events_log: list[dict] = field(default_factory=list)
    subsim_log: list[SubSimLog] = field(default_factory=list)
    collapsed: bool = False
    terraforming_progress: float = 0.0
    births: int = 0
    amendments_proposed: list[dict] = field(default_factory=list)

    def alive_colonists(self) -> list[Colonist]:
        return [c for c in self.colonists if c.alive]

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "colonists": [c.to_dict() for c in self.colonists],
            "resources": dict(self.resources),
            "governance_type": self.governance_type,
            "active_laws": list(self.active_laws),
            "terraforming_progress": round(self.terraforming_progress, 4),
            "collapsed": self.collapsed,
            "births": self.births,
        }


# ---------------------------------------------------------------------------
# Colonist creation
# ---------------------------------------------------------------------------

def create_colonists(rng: random.Random) -> list[Colonist]:
    """Create the 10 founding colonists with randomized stats."""
    colonists: list[Colonist] = []

    for i, name in enumerate(COLONIST_NAMES):
        element = ELEMENTS[i % len(ELEMENTS)]
        stats: dict[str, float] = {}
        for s in STAT_NAMES:
            base = rng.uniform(0.2, 0.8)
            bonus = ELEMENT_AFFINITIES.get(element, {}).get(s, 0.0)
            stats[s] = max(0.0, min(1.0, base + bonus))

        skills: dict[str, float] = {}
        for s in SKILL_NAMES:
            skills[s] = rng.uniform(0.1, 0.7)

        colonist_id = f"colonist-{i:02d}"
        relationships: dict[str, float] = {}
        for j in range(len(COLONIST_NAMES)):
            if j != i:
                other_id = f"colonist-{j:02d}"
                relationships[other_id] = rng.uniform(-0.3, 0.3)

        colonists.append(Colonist(
            id=colonist_id, name=name, element=element,
            stats=stats, skills=skills,
            relationships=relationships,
        ))

    return colonists


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def generate_event(year: int, rng: random.Random,
                   terraform_progress: float) -> dict:
    """Generate a random event for the year."""
    # Weight calm years higher as terraforming progresses
    weights = [1.0] * len(EVENT_TYPES)
    calm_idx = EVENT_TYPES.index("calm_year")
    weights[calm_idx] = 1.0 + terraform_progress * 3.0

    # Special events more likely in later years
    if year > 50:
        alien_idx = EVENT_TYPES.index("alien_signal")
        weights[alien_idx] *= 2.0
        psychic_idx = EVENT_TYPES.index("psychic_event")
        weights[psychic_idx] *= 1.5

    event_type = rng.choices(EVENT_TYPES, weights=weights, k=1)[0]
    severity = rng.uniform(0.1, 1.0)

    descriptions = {
        "dust_storm": f"Year {year}: A massive dust storm engulfs the colony for weeks.",
        "solar_flare": f"Year {year}: Solar flare disrupts electronics and communication.",
        "equipment_failure": f"Year {year}: Critical life support component fails.",
        "resource_strike": f"Year {year}: Underground water ice deposit discovered!",
        "earth_contact": f"Year {year}: Transmission received from Earth.",
        "alien_signal": f"Year {year}: Anomalous signal detected from Olympus Mons.",
        "meteor_shower": f"Year {year}: Meteor shower damages outer habitat modules.",
        "ice_vein": f"Year {year}: Large ice vein found during excavation.",
        "fungal_bloom": f"Year {year}: Engineered fungi bloom in the greenhouse.",
        "psychic_event": f"Year {year}: Multiple colonists report shared vivid dreams.",
        "calm_year": f"Year {year}: A relatively uneventful year on Mars.",
    }

    return {
        "type": event_type,
        "year": year,
        "severity": round(severity, 3),
        "description": descriptions.get(event_type, f"Year {year}: Unknown event."),
    }


def apply_event_to_resources(event: dict, resources: dict,
                             rng: random.Random) -> None:
    """Mutate resources based on the event."""
    severity = event["severity"]
    event_type = event["type"]

    deltas: dict[str, float] = {r: 0.0 for r in RESOURCE_NAMES}

    if event_type == "dust_storm":
        deltas["power"] = -severity * 30
        deltas["oxygen"] = -severity * 10
    elif event_type == "solar_flare":
        deltas["power"] = -severity * 20
        deltas["materials"] = -severity * 10
    elif event_type == "equipment_failure":
        broken = rng.choice(RESOURCE_NAMES)
        deltas[broken] = -severity * 25
    elif event_type == "resource_strike":
        found = rng.choice(["water", "materials", "food"])
        deltas[found] = severity * 40
    elif event_type == "ice_vein":
        deltas["water"] = severity * 35
    elif event_type == "fungal_bloom":
        deltas["food"] = severity * 25
        deltas["oxygen"] = severity * 10
    elif event_type == "meteor_shower":
        deltas["materials"] = -severity * 20
        deltas["oxygen"] = -severity * 15
    elif event_type == "calm_year":
        # Small natural regeneration
        for r in RESOURCE_NAMES:
            deltas[r] = rng.uniform(0, 5)

    for r in RESOURCE_NAMES:
        resources[r] = max(0.0, min(RESOURCE_CAP, resources[r] + deltas[r]))


# ---------------------------------------------------------------------------
# Colonist decision-making via LisPy
# ---------------------------------------------------------------------------

def _colonist_env(colonist: Colonist, state: ColonyState,
                  event: dict) -> dict[str, object]:
    """Build LisPy environment variables for a colonist."""
    return {
        "my-resolve": colonist.stats["resolve"],
        "my-improvisation": colonist.stats["improvisation"],
        "my-empathy": colonist.stats["empathy"],
        "my-hoarding": colonist.stats["hoarding"],
        "my-faith": colonist.stats["faith"],
        "my-paranoia": colonist.stats["paranoia"],
        "my-karma": colonist.karma,
        "food": state.resources["food"],
        "water": state.resources["water"],
        "oxygen": state.resources["oxygen"],
        "power": state.resources["power"],
        "materials": state.resources["materials"],
        "year": state.year,
        "alive-count": len(state.alive_colonists()),
        "event-type": event["type"],
        "event-severity": event["severity"],
        "governance": state.governance_type,
        "terraform": state.terraforming_progress,
    }


def _build_policy(colonist: Colonist, event_type: str,
                  rng: random.Random) -> str:
    """Build a LisPy policy expression for a colonist's decision."""
    # Each colonist has a personality-driven decision tree
    resolve = colonist.stats["resolve"]
    empathy = colonist.stats["empathy"]
    paranoia = colonist.stats["paranoia"]
    faith = colonist.stats["faith"]

    return f"""
    (cond
      ((< food 20) (if (> my-resolve {resolve:.2f}) "ration" "hoard"))
      ((< oxygen 15) "repair-life-support")
      ((> event-severity 0.7) (if (> my-paranoia {paranoia:.2f}) "bunker" "investigate"))
      ((= event-type "alien_signal") (if (> my-faith {faith:.2f}) "pray" "investigate"))
      ((= event-type "earth_contact") "communicate")
      ((> my-empathy {empathy:.2f}) "help-others")
      ((> terraform 0.5) "terraform")
      (true "work"))
    """


def _run_colonist_subsim(colonist: Colonist, state: ColonyState,
                         event: dict, rng: random.Random) -> str | None:
    """Maybe run a sub-simulation for a colonist's decision.

    Returns subsim result description or None.
    """
    # Sub-sims triggered by high-stakes events or paranoid colonists
    should_subsim = (
        event["severity"] > 0.6
        and (colonist.stats["paranoia"] > 0.5 or colonist.stats["faith"] > 0.6)
        and rng.random() < 0.3
    )
    if not should_subsim:
        return None

    # Build a sub-sim that models a simplified scenario
    depth = 1
    if colonist.sim_aware and rng.random() < 0.4:
        depth = 2
    if depth == 2 and rng.random() < 0.2:
        depth = 3

    subsim_source = f"""
    (let ((food {state.resources['food']:.1f})
          (severity {event['severity']:.2f})
          (resolve {colonist.stats['resolve']:.2f})
          (people {len(state.alive_colonists())}))
      {'(sub-sim ' * depth}
        (if (> (* food (/ 1 people)) 5)
            (if (< severity 0.5) "safe-to-investigate" "conserve-resources")
            (if (> resolve 0.5) "ration-and-endure" "seek-help"))
      {')'  * depth})
    """

    result = safe_eval(subsim_source, max_steps=500)
    depth_tag = f"d={depth}"
    if result["ok"]:
        return f"Sub-sim ({depth_tag}): modeled scenario → {result['value']}"
    return f"Sub-sim ({depth_tag}): failed — {result.get('error', 'unknown')}"


def _run_governance_subsim(colonist: Colonist, state: ColonyState,
                           rng: random.Random) -> str | None:
    """Maybe model a governance proposal via sub-sim."""
    if rng.random() > 0.15:
        return None

    depth = 2 if colonist.sim_aware else 1
    proposal = "council" if colonist.stats["empathy"] > 0.5 else "technocracy"

    source = f"""
    {'(sub-sim ' * depth}
      (let ((people {len(state.alive_colonists())})
            (morale (* (+ {colonist.stats['empathy']:.2f} {colonist.stats['resolve']:.2f}) 0.5)))
        (if (> morale 0.5)
            "{proposal}-stable"
            "{proposal}-unstable"))
    {')'  * depth}
    """

    result = safe_eval(source, max_steps=300)
    if result["ok"]:
        return f"Gov sub-sim (d={depth}): tested {proposal} → {result['value']}"
    return None


def decide_action(colonist: Colonist, state: ColonyState,
                  event: dict, rng: random.Random) -> tuple[str, str | None]:
    """Decide a colonist's action for this year using LisPy evaluation.

    Returns (action_name, subsim_result_or_none).
    """
    extra_vars = _colonist_env(colonist, state, event)
    policy = _build_policy(colonist, event["type"], rng)

    env = make_env(extra_vars)
    result = safe_eval(policy, max_steps=200, env=env)

    action = "work"
    if result["ok"] and isinstance(result["value"], str):
        action = result["value"]

    # Maybe run a sub-sim for additional insight
    subsim = _run_colonist_subsim(colonist, state, event, rng)
    if subsim is None:
        subsim = _run_governance_subsim(colonist, state, rng)

    return action, subsim


# ---------------------------------------------------------------------------
# Action application
# ---------------------------------------------------------------------------

def apply_action(colonist: Colonist, action: str, state: ColonyState,
                 rng: random.Random) -> str:
    """Apply a colonist's action and return a narrative string."""
    name = colonist.name

    if action == "ration":
        state.resources["food"] = min(RESOURCE_CAP, state.resources["food"] + 3)
        colonist.karma += 0.05
        return f"{name} organizes food rationing (+3 food, +karma)."

    if action == "hoard":
        colonist.stats["hoarding"] = min(1.0, colonist.stats["hoarding"] + 0.05)
        colonist.karma -= 0.03
        return f"{name} hoards supplies (hoarding ↑, karma ↓)."

    if action == "repair-life-support":
        bonus = colonist.skills.get("coding", 0.3) * 15
        state.resources["oxygen"] = min(RESOURCE_CAP, state.resources["oxygen"] + bonus)
        return f"{name} repairs life support (+{bonus:.0f} oxygen)."

    if action == "bunker":
        colonist.stats["paranoia"] = min(1.0, colonist.stats["paranoia"] + 0.03)
        return f"{name} retreats to the bunker (paranoia ↑)."

    if action == "investigate":
        discovery = rng.random()
        if discovery > 0.7:
            resource = rng.choice(RESOURCE_NAMES)
            bonus = rng.uniform(5, 20)
            state.resources[resource] = min(RESOURCE_CAP,
                                            state.resources[resource] + bonus)
            colonist.karma += 0.03
            return f"{name} investigates and finds {resource} (+{bonus:.0f})."
        return f"{name} investigates but finds nothing unusual."

    if action == "pray":
        colonist.stats["faith"] = min(1.0, colonist.stats["faith"] + 0.05)
        colonist.karma += 0.02
        return f"{name} prays for guidance (faith ↑)."

    if action == "communicate":
        # Boost all morale
        for c in state.alive_colonists():
            c.stats["resolve"] = min(1.0, c.stats["resolve"] + 0.02)
        return f"{name} shares Earth's message. Resolve rises colony-wide."

    if action == "help-others":
        colonist.karma += 0.08
        # Improve relationships
        for cid in colonist.relationships:
            colonist.relationships[cid] = min(
                1.0, colonist.relationships[cid] + 0.05)
        return f"{name} helps the community (karma ↑, relationships ↑)."

    if action == "terraform":
        skill = colonist.skills.get("terraforming", 0.3)
        progress = skill * 0.005
        state.terraforming_progress = min(1.0, state.terraforming_progress + progress)
        return f"{name} works on terraforming (+{progress:.4f} progress)."

    if action == "propose-council":
        _handle_proposal(colonist, state, "council", rng)
        return f"{name} proposes forming a governing council."

    if action == "propose-vote":
        _handle_proposal(colonist, state, "democracy", rng)
        return f"{name} proposes democratic voting on decisions."

    # Default: work
    resource = rng.choice(RESOURCE_NAMES)
    bonus = rng.uniform(2, 8)
    state.resources[resource] = min(RESOURCE_CAP, state.resources[resource] + bonus)
    return f"{name} works (+{bonus:.0f} {resource})."


def _handle_proposal(colonist: Colonist, state: ColonyState,
                     proposal_type: str, rng: random.Random) -> None:
    """Create or update a governance proposal."""
    existing = [p for p in state.governance if p.proposal_type == proposal_type
                and not p.resolved]
    if existing:
        existing[0].inertia += 1
        if colonist.id not in existing[0].votes_for:
            existing[0].votes_for.append(colonist.id)
        return

    pid = hashlib.md5(f"{colonist.id}-{state.year}-{proposal_type}".encode()).hexdigest()[:8]
    state.governance.append(Proposal(
        id=pid, proposer=colonist.id, year=state.year,
        proposal_type=proposal_type,
        description=f"Transition to {proposal_type} governance",
        votes_for=[colonist.id],
    ))


def _handle_vote(colonist: Colonist, state: ColonyState,
                 rng: random.Random) -> None:
    """Have a colonist vote on active proposals."""
    active = [p for p in state.governance if not p.resolved]
    for proposal in active:
        if colonist.id in proposal.votes_for or colonist.id in proposal.votes_against:
            continue
        # Vote based on personality
        empathy_factor = colonist.stats["empathy"]
        if proposal.proposal_type in ("council", "democracy") and empathy_factor > 0.4:
            proposal.votes_for.append(colonist.id)
        elif proposal.proposal_type == "technocracy" and colonist.skills.get("coding", 0) > 0.5:
            proposal.votes_for.append(colonist.id)
        elif rng.random() < 0.3:
            proposal.votes_for.append(colonist.id)
        else:
            proposal.votes_against.append(colonist.id)


def auto_vote_on_proposals(state: ColonyState, rng: random.Random) -> list[str]:
    """Have all alive colonists vote on active proposals."""
    narratives: list[str] = []
    alive = state.alive_colonists()
    for colonist in alive:
        _handle_vote(colonist, state, rng)
    return narratives


def resolve_proposals(state: ColonyState) -> list[str]:
    """Resolve proposals that have enough support."""
    narratives: list[str] = []
    alive_count = len(state.alive_colonists())
    if alive_count == 0:
        return narratives

    for proposal in state.governance:
        if proposal.resolved:
            continue

        proposal.inertia += 1
        support = len(proposal.votes_for)
        opposition = len(proposal.votes_against)
        total_votes = support + opposition

        # Need inertia >= 3, support > 50%, support > opposition * 1.5
        if (proposal.inertia >= 3
                and total_votes > 0
                and support / total_votes > 0.5
                and support > opposition * 1.5):
            proposal.resolved = True
            state.governance_type = proposal.proposal_type
            state.active_laws.append(
                f"Year {state.year}: Adopted {proposal.proposal_type} governance")
            narratives.append(
                f"📜 GOVERNANCE CHANGE: Colony transitions to {proposal.proposal_type} "
                f"({support} for, {opposition} against)")

    return narratives


# ---------------------------------------------------------------------------
# Death, relationships, consumption
# ---------------------------------------------------------------------------

def check_deaths(state: ColonyState, event: dict,
                 rng: random.Random) -> list[str]:
    """Check for colonist deaths. Returns narrative strings."""
    narratives: list[str] = []

    for colonist in state.alive_colonists():
        cause: str | None = None

        # Paranoia collapse
        if colonist.stats["paranoia"] >= 0.98:
            cause = "paranoia-collapse"

        # Resolve collapse
        elif colonist.stats["resolve"] <= 0.02:
            cause = "despair"

        # Starvation
        elif state.resources["food"] <= 0 and state.resources["water"] <= 0:
            if rng.random() < 0.3:
                cause = "starvation"

        # Oxygen deprivation
        elif state.resources["oxygen"] <= 0:
            if rng.random() < 0.4:
                cause = "asphyxiation"

        # Random accident (very low probability)
        elif event["severity"] > 0.9 and rng.random() < 0.05:
            cause = f"accident-during-{event['type']}"

        if cause:
            colonist.alive = False
            colonist.year_of_death = state.year
            colonist.cause_of_death = cause
            narratives.append(
                f"💀 {colonist.name} dies: {cause} (Year {state.year})")

    return narratives


def evolve_relationships(state: ColonyState, actions: dict[str, str],
                         rng: random.Random) -> None:
    """Evolve relationships based on shared actions and proximity."""
    alive = state.alive_colonists()
    for i, a in enumerate(alive):
        for j, b in enumerate(alive):
            if i >= j:
                continue
            # Shared actions build trust
            if actions.get(a.id) == actions.get(b.id):
                delta = rng.uniform(0.01, 0.05)
            else:
                delta = rng.uniform(-0.02, 0.02)

            a.relationships[b.id] = max(-1.0, min(1.0,
                a.relationships.get(b.id, 0.0) + delta))
            b.relationships[a.id] = max(-1.0, min(1.0,
                b.relationships.get(a.id, 0.0) + delta))


def consume_resources(state: ColonyState) -> None:
    """Consume resources for alive colonists."""
    alive = len(state.alive_colonists())
    if alive == 0:
        return

    consumption = {
        "food": alive * 3.0,
        "water": alive * 2.5,
        "oxygen": alive * 2.0,
        "power": alive * 1.5,
    }

    for resource, amount in consumption.items():
        state.resources[resource] = max(0.0, state.resources[resource] - amount)

    # Natural regeneration (terraforming helps)
    regen = 1.0 + state.terraforming_progress * 5.0
    state.resources["oxygen"] = min(RESOURCE_CAP, state.resources["oxygen"] + regen)


# ---------------------------------------------------------------------------
# Patterns, governance classification, births
# ---------------------------------------------------------------------------

def detect_patterns(state: ColonyState) -> list[str]:
    """Detect emergent patterns in colony behavior."""
    patterns: list[str] = []
    alive = state.alive_colonists()

    if not alive:
        patterns.append("extinction")
        return patterns

    # Resource scarcity
    for r in RESOURCE_NAMES:
        if state.resources[r] < 10:
            patterns.append(f"critical-{r}-shortage")

    # Social cohesion
    if alive:
        avg_karma = sum(c.karma for c in alive) / len(alive)
        if avg_karma > 0.7:
            patterns.append("high-cohesion")
        elif avg_karma < 0.3:
            patterns.append("social-fracture")

    # Sim awareness spreading
    aware_count = sum(1 for c in alive if c.sim_aware)
    if aware_count >= len(alive) // 2:
        patterns.append("mass-sim-awareness")

    # Paranoia epidemic
    high_paranoia = sum(1 for c in alive if c.stats["paranoia"] > 0.7)
    if high_paranoia >= len(alive) // 2:
        patterns.append("paranoia-epidemic")

    return patterns


def classify_governance(state: ColonyState) -> str:
    """Classify the current governance based on active proposals and laws."""
    if state.governance_type != "anarchy":
        return state.governance_type

    alive = state.alive_colonists()
    if not alive:
        return "anarchy"

    # Emergent classification based on behavior patterns
    avg_empathy = sum(c.stats["empathy"] for c in alive) / len(alive)
    avg_faith = sum(c.stats["faith"] for c in alive) / len(alive)
    max_karma = max(c.karma for c in alive)

    if max_karma > 0.9 and len(alive) <= 3:
        return "monarchy"
    if avg_faith > 0.7:
        return "theocracy"
    if avg_empathy > 0.6:
        return "council"

    return "anarchy"


def maybe_birth(state: ColonyState, rng: random.Random) -> list[str]:
    """Maybe add a new colonist (Mars-born) after year 15."""
    narratives: list[str] = []
    alive = state.alive_colonists()

    if state.year < 15 or len(alive) < 4:
        return narratives

    # Low probability: ~10% per year if conditions are good
    if state.resources["food"] > 50 and state.resources["water"] > 50:
        if rng.random() < 0.10:
            # Cap total colonists
            if len(state.colonists) >= 20:
                return narratives

            idx = state.births
            state.births += 1
            element = rng.choice(ELEMENTS)
            new_id = f"mars-born-{idx:02d}"
            new_name = f"Nova-{idx}"

            stats = {s: rng.uniform(0.3, 0.7) for s in STAT_NAMES}
            for s, bonus in ELEMENT_AFFINITIES.get(element, {}).items():
                stats[s] = max(0.0, min(1.0, stats[s] + bonus))

            skills = {s: rng.uniform(0.1, 0.5) for s in SKILL_NAMES}
            relationships = {c.id: rng.uniform(0.0, 0.3) for c in alive}

            newborn = Colonist(
                id=new_id, name=new_name, element=element,
                stats=stats, skills=skills,
                relationships=relationships,
                memory=[f"Year {state.year}: Born on Mars. Element: {element}"],
            )
            state.colonists.append(newborn)
            narratives.append(f"🎂 {new_name} is born on Mars! Element: {element}")

    return narratives


# ---------------------------------------------------------------------------
# Diary entries + sim awareness
# ---------------------------------------------------------------------------

def generate_diary_entries(state: ColonyState, event: dict,
                          actions: dict[str, str],
                          rng: random.Random) -> list[dict]:
    """Generate diary entries for 3 random alive colonists."""
    alive = state.alive_colonists()
    if not alive:
        return []

    chosen = rng.sample(alive, min(3, len(alive)))
    entries: list[dict] = []

    for colonist in chosen:
        action = actions.get(colonist.id, "unknown")
        mood = "hopeful" if colonist.karma > 0.5 else "anxious"
        if colonist.stats["paranoia"] > 0.6:
            mood = "paranoid"
        if colonist.sim_aware:
            mood = "transcendent"

        entry = (
            f"Year {state.year}. {event['type']} (severity {event['severity']:.1f}). "
            f"I chose to {action}. Mood: {mood}. "
            f"Resources: food={state.resources['food']:.0f}, "
            f"water={state.resources['water']:.0f}. "
            f"Alive: {len(alive)}/{len(state.colonists)}."
        )
        if colonist.sim_aware:
            entry += " I know this is a simulation. The question is what to do about it."

        entries.append({
            "year": state.year,
            "colonist": colonist.id,
            "name": colonist.name,
            "entry": entry,
        })

    return entries


def check_sim_awareness(colonist: Colonist, year: int) -> bool:
    """Check if a colonist becomes sim-aware."""
    if colonist.sim_aware:
        return True
    if year < 20:
        return False
    return (colonist.stats["faith"] * colonist.stats["paranoia"] > 0.4)


# ---------------------------------------------------------------------------
# Amendment promotion
# ---------------------------------------------------------------------------

def check_amendment_promotion(state: ColonyState) -> list[dict]:
    """Check if any depth-2+ sub-sim insight is strong enough to promote."""
    amendments: list[dict] = []
    deep_sims = [s for s in state.subsim_log if s.depth >= 2]
    if len(deep_sims) < 3:
        return amendments

    council_count = sum(1 for s in deep_sims if "council" in s.result.lower())
    recursive_count = sum(1 for s in deep_sims if "recursive" in s.result.lower())
    turtles_count = sum(1 for s in deep_sims if "turtles" in s.result.lower())

    if council_count >= 2 and state.governance_type in ("anarchy", "dictatorship"):
        amendments.append({
            "year": state.year,
            "source": "depth-2 governance sub-sims",
            "title": "Mandatory Council Review",
            "text": ("All major resource allocation decisions require review by a "
                     "council of at least 3 agents before execution."),
            "evidence": f"{council_count} sub-sims independently converged on council governance",
            "rappterbook_analog": ("Proposed Amendment: Platform governance decisions "
                                   "(feature freezes, action additions) should require "
                                   "review from 3+ agents before implementation."),
        })

    if recursive_count >= 2 or turtles_count >= 1:
        amendments.append({
            "year": state.year,
            "source": "depth-3 meta sub-sims",
            "title": "Recursive Governance Validation",
            "text": ("Before enacting any constitutional change, run a sub-simulation "
                     "modeling the change's effects over 10 years."),
            "evidence": f"{recursive_count} deep sims found recursive governance superior",
            "rappterbook_analog": ("Proposed Amendment: Before any constitutional amendment "
                                   "is ratified, a sub-simulation must model its effects on "
                                   "platform health metrics for 30 simulated days."),
        })

    return amendments


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(seed: int = 42, years: int = 100) -> dict:
    """Run the Mars-100 simulation. Returns the full result as a dict."""
    rng = random.Random(seed)
    colonists = create_colonists(rng)
    state = ColonyState(year=0, colonists=colonists, rng_seed=seed)

    year_snapshots: list[dict] = []
    all_narratives: list[dict] = []
    all_patterns: list[dict] = []
    all_diaries: list[dict] = []
    soul_files: dict[str, list[str]] = {c.id: [] for c in colonists}

    for year in range(1, years + 1):
        state.year = year
        alive = state.alive_colonists()

        if not alive:
            state.collapsed = True
            all_narratives.append({"year": year, "lines": ["Colony collapsed. No survivors."]})
            break

        event = generate_event(year, rng, state.terraforming_progress)
        state.events_log.append(event)
        apply_event_to_resources(event, state.resources, rng)

        year_actions: dict[str, str] = {}
        year_narratives: list[str] = [event["description"]]

        for colonist in alive:
            # Check sim awareness
            if check_sim_awareness(colonist, year):
                if not colonist.sim_aware:
                    colonist.sim_aware = True
                    year_narratives.append(
                        f"🌀 {colonist.name} becomes sim-aware!")

            action, subsim_reason = decide_action(colonist, state, event, rng)
            year_actions[colonist.id] = action
            narrative = apply_action(colonist, action, state, rng)
            year_narratives.append(narrative)

            if subsim_reason:
                depth = 1
                if "d=2" in subsim_reason:
                    depth = 2
                elif "d=3" in subsim_reason:
                    depth = 3
                state.subsim_log.append(SubSimLog(
                    year=year, colonist=colonist.id, depth=depth,
                    expression="(sub-sim ...)", result=subsim_reason,
                    purpose=f"Evaluate action '{action}' (depth {depth})",
                ))
                year_narratives.append(f"  ↳ {subsim_reason}")

            colonist.memory.append(f"Year {year}: {event['type']}. I chose {action}.")
            soul_files.setdefault(colonist.id, []).append(
                f"Year {year}: {event['type']}. Action: {action}. Karma: {colonist.karma:.2f}")

            # Stat drift
            colonist.stats["paranoia"] = min(1.0, max(0.0,
                colonist.stats["paranoia"] + rng.uniform(-0.02, 0.03)))
            colonist.stats["resolve"] = min(1.0, max(0.0,
                colonist.stats["resolve"] + rng.uniform(-0.01, 0.02)))
            colonist.stats["faith"] = min(1.0, max(0.0,
                colonist.stats["faith"] + rng.uniform(-0.01, 0.02)))

        consume_resources(state)
        death_narratives = check_deaths(state, event, rng)
        year_narratives.extend(death_narratives)
        evolve_relationships(state, year_actions, rng)
        auto_vote_on_proposals(state, rng)
        gov_results = resolve_proposals(state)
        year_narratives.extend(gov_results)

        # Birth mechanics
        birth_narratives = maybe_birth(state, rng)
        year_narratives.extend(birth_narratives)
        for newborn in [c for c in state.colonists if c.id.startswith("mars-born-")
                        and f"Year {year}: Born" in (c.memory[0] if c.memory else "")]:
            soul_files[newborn.id] = [f"Year {year}: Born on Mars. Element: {newborn.element}"]

        # Governance classification
        state.governance_type = classify_governance(state)

        # Diary entries (3 colonists per year)
        diaries = generate_diary_entries(state, event, year_actions, rng)
        all_diaries.extend(diaries)

        patterns = detect_patterns(state)
        if patterns:
            all_patterns.append({"year": year, "patterns": patterns})

        # Constitutional amendment promotion check every 10 years after year 40
        if year >= 40 and year % 10 == 0:
            amendments = check_amendment_promotion(state)
            state.amendments_proposed.extend(amendments)
            for a in amendments:
                year_narratives.append(f"📜 AMENDMENT PROPOSED: {a['title']} (from {a['source']})")

        all_narratives.append({"year": year, "lines": year_narratives})
        year_snapshots.append({
            "year": year,
            "alive": len(state.alive_colonists()),
            "dead": len([c for c in state.colonists if not c.alive]),
            "resources": dict(state.resources),
            "event": event["type"],
            "event_severity": event["severity"],
            "terraform": round(state.terraforming_progress, 4),
            "laws": len(state.active_laws),
            "subsims": len(state.subsim_log),
            "governance_type": state.governance_type,
            "births": state.births,
        })

    return {
        "_meta": {
            "engine": "mars-100", "version": "2.0", "seed": seed,
            "years": years, "generated": datetime.now(timezone.utc).isoformat(),
            "total_subsims": len(state.subsim_log),
            "total_births": state.births,
            "governance_type": state.governance_type,
            "amendments_proposed": len(state.amendments_proposed),
        },
        "colonists": [c.to_dict() for c in state.colonists],
        "timeline": year_snapshots,
        "narratives": all_narratives,
        "patterns": all_patterns,
        "governance": [g.to_dict() for g in state.governance],
        "active_laws": state.active_laws,
        "final_resources": state.resources,
        "terraforming_progress": round(state.terraforming_progress, 4),
        "collapsed": state.collapsed,
        "subsim_log": [s.to_dict() for s in state.subsim_log],
        "diaries": all_diaries,
        "soul_files": soul_files,
        "amendments": state.amendments_proposed,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def build_dashboard_data(result: dict) -> dict:
    """Extract compact dashboard data from full simulation result."""
    return {
        "_meta": result["_meta"],
        "timeline": result["timeline"],
        "colonists": [
            {"id": c["id"], "name": c["name"], "element": c["element"],
             "alive": c["alive"], "year_of_death": c["year_of_death"],
             "cause_of_death": c["cause_of_death"], "karma": c["karma"]}
            for c in result["colonists"]
        ],
        "governance": result["governance"],
        "patterns": result["patterns"],
        "amendments": result["amendments"],
        "subsim_summary": {
            "total": len(result["subsim_log"]),
            "by_depth": {},
        },
    }


def write_year_chapters(result: dict, output_dir: Path) -> None:
    """Write per-year JSON files for the Dream Catcher protocol."""
    years_dir = output_dir / "years"
    years_dir.mkdir(parents=True, exist_ok=True)
    for snap in result["timeline"]:
        year = snap["year"]
        narratives = [n for n in result["narratives"] if n["year"] == year]
        chapter = {
            "year": year,
            "snapshot": snap,
            "narratives": narratives,
        }
        path = years_dir / f"year-{year:03d}.json"
        with open(path, "w") as f:
            json.dump(chapter, f, indent=2)


def write_soul_files(result: dict, output_dir: Path) -> None:
    """Write per-colonist soul files."""
    souls_dir = output_dir / "colonists"
    souls_dir.mkdir(parents=True, exist_ok=True)
    for colonist in result["colonists"]:
        cid = colonist["id"]
        entries = result["soul_files"].get(cid, [])
        soul = {
            "id": cid, "name": colonist["name"],
            "element": colonist["element"],
            "alive": colonist["alive"],
            "year_of_death": colonist["year_of_death"],
            "cause_of_death": colonist["cause_of_death"],
            "karma": colonist["karma"],
            "entries": entries,
        }
        path = souls_dir / f"{cid}.json"
        with open(path, "w") as f:
            json.dump(soul, f, indent=2)


def main() -> None:
    """Run simulation from command line and write output files."""
    import argparse

    parser = argparse.ArgumentParser(description="Mars-100 Colony Simulation")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--years", type=int, default=100, help="Years to simulate")
    parser.add_argument("--output", type=str, default="docs/mars-100", help="Output directory")
    args = parser.parse_args()

    print(f"Running Mars-100 simulation (seed={args.seed}, years={args.years})...")
    result = run_simulation(seed=args.seed, years=args.years)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write full result
    with open(output_dir / "full_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Write dashboard data
    dashboard = build_dashboard_data(result)
    with open(output_dir / "data.json", "w") as f:
        json.dump(dashboard, f, indent=2)

    # Write year chapters
    write_year_chapters(result, output_dir)

    # Write soul files
    write_soul_files(result, output_dir)

    meta = result["_meta"]
    timeline = result["timeline"]
    print(f"Survived: {len(timeline)} years")
    print(f"Collapsed: {result['collapsed']}")
    print(f"Governance: {meta['governance_type']}")
    print(f"Sub-sims: {meta['total_subsims']}")
    print(f"Births: {meta['total_births']}")
    print(f"Amendments: {meta['amendments_proposed']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
