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
from typing import Any

from src.lispy import (
    Budget, Env, Lambda, LispyError, NIL, Symbol,
    format_sexpr, lispy_eval, make_env, parse, run,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELEMENTS = ["fire", "water", "earth", "air"]
STAT_NAMES = ["resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia"]
SKILL_NAMES = ["terraforming", "hydroponics", "mediation", "coding", "prayer", "sabotage"]

EVENT_TYPES = [
    "dust_storm", "resource_strike", "equipment_failure",
    "earth_contact", "alien_signal", "solar_flare",
    "underground_water", "habitat_breach", "meteor_shower",
    "fungal_bloom",
]

COLONIST_NAMES = [
    "Kael", "Zara", "Orin", "Mira", "Thane",
    "Petra", "Solan", "Vex", "Luna", "Ash",
]

ACTIONS = [
    "repair", "explore", "ration", "hoard", "share",
    "mediate", "pray", "sabotage", "terraform", "farm",
    "code", "rest", "propose", "vote_yes", "vote_no",
]

# Governance structure types
GOVERNANCE_TYPES = [
    "anarchy", "council", "democracy", "dictatorship", "theocracy", "technocracy",
]

MIN_STAT = 0.0
MAX_STAT = 1.0
MIN_RESOURCE = 0
MAX_RESOURCE = 10000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Colonist:
    """A Mars colonist — data structure AND LisPy-evaluable personality."""
    id: str
    name: str
    element: str
    stats: dict[str, float]
    skills: dict[str, float]
    relationships: dict[str, float]
    memory: list[str]
    alive: bool = True
    year_of_death: int | None = None
    cause_of_death: str | None = None
    governance_votes: dict[str, str] = field(default_factory=dict)
    karma: float = 0.5

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id, "name": self.name, "element": self.element,
            "stats": self.stats, "skills": self.skills,
            "relationships": self.relationships,
            "memory": self.memory[-20:],
            "alive": self.alive, "year_of_death": self.year_of_death,
            "cause_of_death": self.cause_of_death,
            "karma": round(self.karma, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Colonist:
        """Deserialize from dict."""
        return cls(
            id=data["id"], name=data["name"], element=data["element"],
            stats=data["stats"], skills=data["skills"],
            relationships=data["relationships"],
            memory=data.get("memory", []),
            alive=data.get("alive", True),
            year_of_death=data.get("year_of_death"),
            cause_of_death=data.get("cause_of_death"),
            karma=data.get("karma", 0.5),
        )


@dataclass
class GovernanceProposal:
    """A proposal for colony governance."""
    id: str
    year: int
    proposer: str
    title: str
    description: str
    rule_expr: str
    votes_for: list[str] = field(default_factory=list)
    votes_against: list[str] = field(default_factory=list)
    passed: bool | None = None
    year_decided: int | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "year": self.year, "proposer": self.proposer,
            "title": self.title, "description": self.description,
            "rule_expr": self.rule_expr,
            "votes_for": self.votes_for, "votes_against": self.votes_against,
            "passed": self.passed, "year_decided": self.year_decided,
        }


@dataclass
class SubSimLog:
    """Log of a sub-simulation run."""
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
class ColonyState:
    """The full state of the Mars colony at a given year."""
    year: int = 0
    colonists: list[Colonist] = field(default_factory=list)
    resources: dict[str, int] = field(default_factory=lambda: {
        "food": 2000, "water": 3000, "power": 1500,
        "oxygen": 2500, "materials": 1000,
    })
    governance: list[GovernanceProposal] = field(default_factory=list)
    active_laws: list[str] = field(default_factory=list)
    events_log: list[dict] = field(default_factory=list)
    subsim_log: list[SubSimLog] = field(default_factory=list)
    terraforming_progress: float = 0.0
    collapsed: bool = False
    rng_seed: int = 42
    governance_type: str = "anarchy"
    births: int = 0
    amendments_proposed: list[dict] = field(default_factory=list)

    def alive_colonists(self) -> list[Colonist]:
        """Return list of living colonists."""
        return [c for c in self.colonists if c.alive]

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "colonists": [c.to_dict() for c in self.colonists],
            "resources": self.resources,
            "governance": [g.to_dict() for g in self.governance],
            "active_laws": self.active_laws,
            "events_log": self.events_log[-200:],
            "subsim_log": [s.to_dict() for s in self.subsim_log[-100:]],
            "terraforming_progress": round(self.terraforming_progress, 4),
            "collapsed": self.collapsed,
            "governance_type": self.governance_type,
            "births": self.births,
            "amendments_proposed": self.amendments_proposed,
        }


# ---------------------------------------------------------------------------
# Colony initialization
# ---------------------------------------------------------------------------

def create_colonists(rng: random.Random) -> list[Colonist]:
    """Create the 10 founding colonists with diverse traits."""
    colonists: list[Colonist] = []

    for i, name in enumerate(COLONIST_NAMES):
        element = ELEMENTS[i % len(ELEMENTS)]
        base_stats = {s: rng.uniform(0.2, 0.8) for s in STAT_NAMES}
        base_skills = {s: rng.uniform(0.1, 0.6) for s in SKILL_NAMES}

        # Element affinity
        if element == "fire":
            base_stats["resolve"] = min(1.0, base_stats["resolve"] + 0.2)
            base_stats["paranoia"] = min(1.0, base_stats["paranoia"] + 0.1)
            base_skills["terraforming"] = min(1.0, base_skills["terraforming"] + 0.3)
        elif element == "water":
            base_stats["empathy"] = min(1.0, base_stats["empathy"] + 0.2)
            base_stats["improvisation"] = min(1.0, base_stats["improvisation"] + 0.1)
            base_skills["hydroponics"] = min(1.0, base_skills["hydroponics"] + 0.3)
        elif element == "earth":
            base_stats["hoarding"] = min(1.0, base_stats["hoarding"] + 0.2)
            base_stats["faith"] = min(1.0, base_stats["faith"] + 0.1)
            base_skills["mediation"] = min(1.0, base_skills["mediation"] + 0.3)
        elif element == "air":
            base_stats["improvisation"] = min(1.0, base_stats["improvisation"] + 0.2)
            base_stats["resolve"] = max(0.0, base_stats["resolve"] - 0.1)
            base_skills["coding"] = min(1.0, base_skills["coding"] + 0.3)

        cid = f"mars-{name.lower()}"
        colonist = Colonist(
            id=cid, name=name, element=element,
            stats={k: round(v, 3) for k, v in base_stats.items()},
            skills={k: round(v, 3) for k, v in base_skills.items()},
            relationships={},
            memory=[f"Year 0: Landed on Mars. I am {name}, element of {element}."],
        )
        colonists.append(colonist)

    # Initialize relationships
    for c in colonists:
        for other in colonists:
            if other.id != c.id:
                base = 0.1 if c.element == other.element else 0.0
                c.relationships[other.id] = round(base + rng.uniform(-0.2, 0.2), 3)

    return colonists


# ---------------------------------------------------------------------------
# Environmental events
# ---------------------------------------------------------------------------

def generate_event(year: int, rng: random.Random,
                   terraform_progress: float) -> dict:
    """Generate the year's environmental event."""
    storm_weight = max(0.3, 1.0 - terraform_progress * 2)
    weights = {
        "dust_storm": 0.15 * storm_weight,
        "resource_strike": 0.12,
        "equipment_failure": 0.13 * storm_weight,
        "earth_contact": 0.10,
        "alien_signal": 0.03 + (0.02 if year > 50 else 0),
        "solar_flare": 0.08,
        "underground_water": 0.10,
        "habitat_breach": 0.07 * storm_weight,
        "meteor_shower": 0.06,
        "fungal_bloom": 0.05 + (0.05 if terraform_progress > 0.3 else 0),
    }

    total = sum(weights.values())
    roll = rng.uniform(0, total)
    cumulative = 0.0
    chosen = "dust_storm"
    for etype, w in weights.items():
        cumulative += w
        if roll <= cumulative:
            chosen = etype
            break

    severity = rng.uniform(0.3, 1.0)
    if year > 60:
        severity = min(1.0, severity * 1.2)

    descriptions = {
        "dust_storm": f"A {'massive' if severity > 0.7 else 'moderate'} dust storm blankets the colony.",
        "resource_strike": f"Drilling discovers a {'rich' if severity > 0.6 else 'modest'} mineral vein.",
        "equipment_failure": f"The {'primary' if severity > 0.7 else 'backup'} life support malfunctions.",
        "earth_contact": f"Earth sends a {'critical' if severity > 0.7 else 'routine'} transmission.",
        "alien_signal": f"Sensors detect an {'unmistakable' if severity > 0.8 else 'ambiguous'} non-natural signal.",
        "solar_flare": f"A {'severe' if severity > 0.7 else 'minor'} solar flare hits the colony.",
        "underground_water": f"Seismic scans reveal {'massive' if severity > 0.7 else 'small'} ice deposits.",
        "habitat_breach": f"A {'critical' if severity > 0.7 else 'minor'} breach in the habitat module.",
        "meteor_shower": f"{'Heavy' if severity > 0.7 else 'Light'} meteor shower near the colony.",
        "fungal_bloom": f"{'Aggressive' if severity > 0.7 else 'Slow'} fungal growth in the greenhouse.",
    }

    return {
        "type": chosen, "severity": round(severity, 3),
        "year": year, "description": descriptions[chosen],
    }


def apply_event_to_resources(event: dict, resources: dict,
                             rng: random.Random) -> dict:
    """Apply event consequences to colony resources. Returns delta dict."""
    etype = event["type"]
    severity = event["severity"]
    delta: dict[str, int] = {}

    if etype == "dust_storm":
        delta = {"power": -int(200 * severity), "food": -int(50 * severity)}
    elif etype == "resource_strike":
        delta = {"materials": int(500 * severity), "water": int(200 * severity)}
    elif etype == "equipment_failure":
        delta = {"oxygen": -int(300 * severity), "power": -int(150 * severity)}
    elif etype == "earth_contact":
        delta = {"food": int(300 * severity), "materials": int(200 * severity)}
    elif etype == "solar_flare":
        delta = {"power": -int(250 * severity)}
    elif etype == "underground_water":
        delta = {"water": int(600 * severity)}
    elif etype == "habitat_breach":
        delta = {"oxygen": -int(400 * severity), "materials": -int(100 * severity)}
    elif etype == "meteor_shower":
        delta = {"materials": -int(150 * severity), "power": -int(100 * severity)}
    elif etype == "fungal_bloom":
        delta = {"food": -int(200 * severity)}

    for k, v in delta.items():
        resources[k] = max(MIN_RESOURCE, min(MAX_RESOURCE, resources[k] + v))
    return delta


# ---------------------------------------------------------------------------
# Colonist decision-making (hybrid: Python orchestration + LisPy policy)
# ---------------------------------------------------------------------------

def _colonist_env(colonist: Colonist, state: ColonyState,
                  event: dict, rng: random.Random) -> Env:
    """Build a LisPy environment for a colonist's decision."""
    env = make_env()
    for stat, val in colonist.stats.items():
        env[stat] = val
    for skill, val in colonist.skills.items():
        env[f"skill-{skill}"] = val

    env["year"] = state.year
    env["food"] = state.resources["food"]
    env["water"] = state.resources["water"]
    env["power"] = state.resources["power"]
    env["oxygen"] = state.resources["oxygen"]
    env["materials"] = state.resources["materials"]
    env["alive-count"] = len(state.alive_colonists())
    env["terraform-progress"] = state.terraforming_progress
    env["event-type"] = event["type"]
    env["event-severity"] = event["severity"]
    env["my-element"] = colonist.element
    env["my-name"] = colonist.name
    env["my-karma"] = colonist.karma
    env["random-int"] = lambda a, b: rng.randint(int(a), int(b))

    for other_id, trust in colonist.relationships.items():
        safe_key = other_id.replace("mars-", "trust-")
        env[safe_key] = trust

    return env


def _build_policy(colonist: Colonist, event_type: str,
                  severity: float) -> str:
    """Build a LisPy policy expression based on colonist personality + event."""
    if event_type == "equipment_failure" and severity > 0.6:
        return '(if (> skill-coding 0.5) "repair" (if (> empathy 0.5) "share" "ration"))'
    if event_type == "dust_storm" and severity > 0.7:
        return '(if (> skill-terraforming 0.4) "terraform" (if (> hoarding 0.6) "hoard" "ration"))'
    if event_type == "habitat_breach":
        return '(if (> skill-coding 0.4) "repair" (if (> resolve 0.5) "repair" "rest"))'
    if event_type == "alien_signal":
        return '(if (> faith 0.6) "pray" (if (> skill-coding 0.5) "code" (if (> paranoia 0.7) "sabotage" "explore")))'
    if event_type == "earth_contact":
        return '(if (> empathy 0.5) "share" "hoard")'

    top_skill = max(colonist.skills, key=colonist.skills.get)
    action_map = {
        "terraforming": "terraform", "hydroponics": "farm",
        "mediation": "mediate", "coding": "code",
        "prayer": "pray", "sabotage": "explore",
    }
    default = action_map.get(top_skill, "rest")

    return f"""(begin
      (if (and (> resolve 0.6) (= (mod year 7) 0)) "propose"
          (if (< food 500) "farm"
              (if (< oxygen 500) "repair"
                  (if (< power 300) "terraform"
                      "{default}")))))"""


def _run_colonist_subsim(colonist: Colonist, state: ColonyState,
                         event: dict, action: str,
                         rng: random.Random, depth: int = 1) -> str:
    """Run a sub-simulation to evaluate a proposed action.

    depth 1: simple action evaluation (safe/risky)
    depth 2: governance scenario modeling (project 5 years forward)
    depth 3: meta-simulation — colonist models whether they're in a sim
    """
    if depth == 1:
        subsim_expr = f"""(sub-sim (begin
            (define action "{action}")
            (define food-after (if (= action "farm") (+ food 200)
                (if (= action "hoard") (+ food 50) food)))
            (define risk (if (= action "sabotage") (* event-severity 0.8)
                (if (= action "explore") (* event-severity 0.3)
                    (* event-severity 0.1))))
            (if (and (> food-after 300) (< risk 0.5)) "safe" "risky")))"""
    elif depth == 2:
        # Governance sub-sim: model 5-year resource trajectory
        subsim_expr = f"""(sub-sim (begin
            (define year-food food)
            (define year-power power)
            (define stability 0)
            (define y1-food (+ year-food (if (> skill-hydroponics 0.4) 300 100)))
            (define y1-power (+ year-power (if (> skill-coding 0.4) 150 50)))
            (define council-viable (and (> y1-food 500) (> alive-count 4)))
            (sub-sim (begin
                (define inner-food (+ y1-food 200))
                (define inner-stability (if council-viable 3 1))
                (if (> inner-stability 2)
                    (list "council" inner-food inner-stability)
                    (list "anarchy" inner-food inner-stability))))))"""
    else:
        # Depth 3: meta-simulation — the colonist simulates simulating
        # This is where colonists can discover governance patterns that
        # don't exist yet in the colony's actual constitution.
        subsim_expr = f"""(sub-sim (begin
            (define outer-year year)
            (define outer-food food)
            (define outer-pop alive-count)
            (sub-sim (begin
                (define mid-insight (> outer-year 50))
                (define resource-stress (< outer-food 800))
                (define adaptive-rule
                    (if resource-stress
                        "crisis-autocracy"
                        (if mid-insight "recursive-governance" "flat-anarchy")))
                (sub-sim (begin
                    (define deep-pattern adaptive-rule)
                    (define meta-score
                        (if (= deep-pattern "recursive-governance") 3
                            (if (= deep-pattern "crisis-autocracy") 2 1)))
                    (define amendment-text
                        (if (> meta-score 2)
                            "Governance adapts to resource levels automatically"
                            "Maintain current structure"))
                    (list deep-pattern meta-score "turtles-all-the-way-down" amendment-text)))))))"""

    env = _colonist_env(colonist, state, event, rng)
    budget = Budget(remaining=5000, max_depth=3)
    try:
        result = run(subsim_expr, env=env, budget=budget)
        return f"Sub-sim(d={depth}) evaluated '{action}': {format_sexpr(result)}"
    except LispyError as e:
        return f"Sub-sim(d={depth}) failed: {e}"


def _run_governance_subsim(colonist: Colonist, state: ColonyState,
                           rng: random.Random) -> tuple[str, str]:
    """Run a depth-2 governance sub-sim. Returns (proposal_type, reasoning)."""
    return (
        "council" if colonist.stats["empathy"] > 0.5 else "technocracy",
        _run_colonist_subsim(colonist, state, {"type": "governance_sim", "severity": 0.5},
                             "propose", rng, depth=2),
    )


def decide_action(colonist: Colonist, state: ColonyState,
                  event: dict, rng: random.Random) -> tuple[str, str | None]:
    """Have a colonist decide their year's action. Returns (action, subsim_reason)."""
    policy = _build_policy(colonist, event["type"], event["severity"])
    env = _colonist_env(colonist, state, event, rng)
    budget = Budget(remaining=5000, max_depth=3)

    subsim_reason = None
    try:
        result = run(policy, env=env, budget=budget)
        action = str(result) if result else "rest"
    except LispyError:
        action = "rest"

    # High-paranoia colonists run sub-sims to validate
    if colonist.stats["paranoia"] > 0.6 and rng.random() < colonist.stats["paranoia"] * 0.4:
        subsim_reason = _run_colonist_subsim(colonist, state, event, action, rng)

    # High-faith colonists after year 30 run depth-2 governance sims
    if (state.year >= 30 and colonist.stats.get("faith", 0) > 0.5
            and rng.random() < 0.2 and subsim_reason is None):
        _, subsim_reason = _run_governance_subsim(colonist, state, rng)

    # Any colonist after year 50 may run depth-3 meta-sims (simulation awareness)
    if (state.year >= 50 and rng.random() < 0.05 and subsim_reason is None):
        subsim_reason = _run_colonist_subsim(
            colonist, state, event, action, rng, depth=3)

    if action not in ACTIONS:
        action = "rest"
    return action, subsim_reason


# ---------------------------------------------------------------------------
# Action application
# ---------------------------------------------------------------------------

def apply_action(colonist: Colonist, action: str, state: ColonyState,
                 rng: random.Random) -> str:
    """Apply a colonist's action. Returns narrative line."""
    r = state.resources

    if action == "repair":
        gained = int(100 * (0.5 + colonist.skills.get("coding", 0.3)))
        r["oxygen"] = min(MAX_RESOURCE, r["oxygen"] + gained)
        r["power"] = min(MAX_RESOURCE, r["power"] + gained // 2)
        return f"{colonist.name} repairs systems (+{gained} O₂, +{gained // 2} power)"

    if action == "explore":
        found = rng.choice(["materials", "water"])
        amount = int(150 * rng.uniform(0.5, 1.5))
        r[found] = min(MAX_RESOURCE, r[found] + amount)
        return f"{colonist.name} explores, finds {amount} {found}"

    if action == "ration":
        for key in r:
            r[key] = min(MAX_RESOURCE, r[key] + 30)
        return f"{colonist.name} rations supplies (all +30)"

    if action == "hoard":
        r["food"] = min(MAX_RESOURCE, r["food"] + 80)
        colonist.karma = max(0, colonist.karma - 0.05)
        return f"{colonist.name} hoards food (+80, karma -0.05)"

    if action == "share":
        r["food"] = max(MIN_RESOURCE, r["food"] - 50)
        colonist.karma = min(1.0, colonist.karma + 0.08)
        return f"{colonist.name} shares rations (-50 food, karma +0.08)"

    if action == "mediate":
        for c in state.alive_colonists():
            if c.id != colonist.id and c.id in colonist.relationships:
                colonist.relationships[c.id] = min(1.0, colonist.relationships[c.id] + 0.05)
        colonist.karma = min(1.0, colonist.karma + 0.03)
        return f"{colonist.name} mediates disputes (relationships +0.05)"

    if action == "pray":
        colonist.stats["faith"] = min(MAX_STAT, colonist.stats["faith"] + 0.02)
        colonist.stats["paranoia"] = max(MIN_STAT, colonist.stats["paranoia"] - 0.01)
        return f"{colonist.name} prays (faith +0.02, paranoia -0.01)"

    if action == "sabotage":
        others = [c for c in state.alive_colonists() if c.id != colonist.id]
        if others:
            target = rng.choice(others)
            r["materials"] = max(MIN_RESOURCE, r["materials"] - 100)
            colonist.karma = max(0, colonist.karma - 0.15)
            colonist.relationships[target.id] = max(
                -1.0, colonist.relationships.get(target.id, 0) - 0.3)
            return f"{colonist.name} sabotages {target.name}'s work (-100 materials)"
        return f"{colonist.name} contemplates sabotage but has no target"

    if action == "terraform":
        progress = 0.005 * (0.5 + colonist.skills.get("terraforming", 0.3))
        state.terraforming_progress = min(1.0, state.terraforming_progress + progress)
        r["power"] = max(MIN_RESOURCE, r["power"] - 50)
        return f"{colonist.name} terraforms (+{progress:.4f} progress)"

    if action == "farm":
        yield_amt = int(200 * (0.5 + colonist.skills.get("hydroponics", 0.3)))
        r["food"] = min(MAX_RESOURCE, r["food"] + yield_amt)
        return f"{colonist.name} farms (+{yield_amt} food)"

    if action == "code":
        r["power"] = min(MAX_RESOURCE, r["power"] + 60)
        colonist.skills["coding"] = min(MAX_STAT, colonist.skills["coding"] + 0.01)
        return f"{colonist.name} codes systems (+60 power, coding +0.01)"

    if action == "rest":
        colonist.stats["resolve"] = min(MAX_STAT, colonist.stats["resolve"] + 0.02)
        return f"{colonist.name} rests (resolve +0.02)"

    if action == "propose":
        return _handle_proposal(colonist, state, rng)

    if action in ("vote_yes", "vote_no"):
        return _handle_vote(colonist, state, action == "vote_yes")

    return f"{colonist.name} does nothing"


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

PROPOSAL_TEMPLATES = [
    ("Rationing Law", "All colonists must share food equally.",
     '(define ration-rule (lambda (c) (if (> (assoc "food" c) 200) "share" "keep")))'),
    ("Exile Protocol", "Colonists with karma below 0.2 face exile vote.",
     '(define exile-check (lambda (k) (< k 0.2)))'),
    ("Council of Elders", "Decisions require majority vote.",
     '(define council-vote (lambda (votes) (> (length (filter (lambda (v) v) votes)) (/ (length votes) 2))))'),
    ("Terraform Priority", "20% of power to terraforming.",
     '(define terraform-alloc (lambda (power) (round (* power 0.2))))'),
    ("Faith Mandate", "Communal prayer restores colony morale.",
     '(define faith-bonus (lambda (faith) (if (> faith 0.5) 0.05 0.01)))'),
    ("Sabotage Tribunal", "Saboteurs lose all karma and face exile.",
     '(define tribunal (lambda (act karma) (if (= act "sabotage") 0.0 karma)))'),
]


def _handle_proposal(colonist: Colonist, state: ColonyState,
                     rng: random.Random) -> str:
    """Generate and register a governance proposal."""
    template = rng.choice(PROPOSAL_TEMPLATES)
    pid = hashlib.md5(f"{state.year}-{colonist.id}-{template[0]}".encode()).hexdigest()[:8]
    proposal = GovernanceProposal(
        id=pid, year=state.year, proposer=colonist.id,
        title=template[0], description=template[1], rule_expr=template[2],
    )
    state.governance.append(proposal)
    colonist.karma = min(1.0, colonist.karma + 0.05)
    return f"{colonist.name} proposes: '{template[0]}'"


def _handle_vote(colonist: Colonist, state: ColonyState,
                 vote_yes: bool) -> str:
    """Handle a colonist voting on pending proposals."""
    pending = [g for g in state.governance if g.passed is None]
    if not pending:
        return f"{colonist.name} has nothing to vote on"
    proposal = pending[-1]
    if colonist.id in proposal.votes_for or colonist.id in proposal.votes_against:
        return f"{colonist.name} already voted on '{proposal.title}'"
    if vote_yes:
        proposal.votes_for.append(colonist.id)
    else:
        proposal.votes_against.append(colonist.id)
    return f"{colonist.name} votes {'yes' if vote_yes else 'no'} on '{proposal.title}'"


def auto_vote_on_proposals(state: ColonyState, rng: random.Random) -> list[str]:
    """All alive colonists automatically vote on pending proposals.

    Voting is personality-driven: empathy, faith, paranoia, and relationship
    with the proposer all influence the vote. This ensures proposals actually
    get resolved rather than languishing forever.
    """
    narratives: list[str] = []
    alive = state.alive_colonists()
    pending = [g for g in state.governance if g.passed is None]

    for proposal in pending:
        for colonist in alive:
            if colonist.id == proposal.proposer:
                if colonist.id not in proposal.votes_for:
                    proposal.votes_for.append(colonist.id)
                continue
            if colonist.id in proposal.votes_for or colonist.id in proposal.votes_against:
                continue

            # Personality-driven vote
            score = 0.0
            trust = colonist.relationships.get(proposal.proposer, 0)
            score += trust * 30  # trust in proposer matters

            # Content-based voting heuristics
            title_lower = proposal.title.lower()
            if "faith" in title_lower or "prayer" in title_lower:
                score += (colonist.stats.get("faith", 0.5) - 0.4) * 20
            if "exile" in title_lower or "tribunal" in title_lower:
                score += (colonist.stats.get("paranoia", 0.3) - 0.3) * 15
                score -= colonist.stats.get("empathy", 0.5) * 10
            if "council" in title_lower or "ration" in title_lower:
                score += colonist.stats.get("empathy", 0.5) * 15
            if "terraform" in title_lower:
                score += colonist.skills.get("terraforming", 0.3) * 15
            if "sabotage" in title_lower:
                score += colonist.stats.get("resolve", 0.5) * 10

            # Noise
            score += rng.uniform(-10, 10)

            if score > 0:
                proposal.votes_for.append(colonist.id)
            else:
                proposal.votes_against.append(colonist.id)

    return narratives


def resolve_proposals(state: ColonyState) -> list[str]:
    """Resolve pending proposals by majority of living colonists."""
    alive_ids = {c.id for c in state.alive_colonists()}
    resolved: list[str] = []
    for proposal in state.governance:
        if proposal.passed is not None:
            continue
        total_votes = len(proposal.votes_for) + len(proposal.votes_against)
        if total_votes < max(1, len(alive_ids) // 2):
            continue
        proposal.passed = len(proposal.votes_for) > len(proposal.votes_against)
        proposal.year_decided = state.year
        if proposal.passed:
            state.active_laws.append(proposal.title)
            resolved.append(f"LAW PASSED: '{proposal.title}' ({len(proposal.votes_for)}-{len(proposal.votes_against)})")
        else:
            resolved.append(f"REJECTED: '{proposal.title}' ({len(proposal.votes_for)}-{len(proposal.votes_against)})")
    return resolved


# ---------------------------------------------------------------------------
# Death and survival
# ---------------------------------------------------------------------------

def check_deaths(state: ColonyState, event: dict,
                 rng: random.Random) -> list[str]:
    """Check for colonist deaths. Returns narrative lines."""
    narratives: list[str] = []
    alive = state.alive_colonists()
    if not alive:
        return narratives

    r = state.resources
    for colonist in alive:
        death_chance = 0.0
        food_pp = r["food"] / max(1, len(alive))
        o2_pp = r["oxygen"] / max(1, len(alive))

        if food_pp < 50:
            death_chance += 0.15
        elif food_pp < 100:
            death_chance += 0.05
        if o2_pp < 50:
            death_chance += 0.20
        elif o2_pp < 100:
            death_chance += 0.08

        if event["type"] == "solar_flare" and event["severity"] > 0.8:
            death_chance += 0.10
        if event["type"] == "habitat_breach" and event["severity"] > 0.8:
            death_chance += 0.12
        if event["type"] == "meteor_shower" and event["severity"] > 0.9:
            death_chance += 0.08

        death_chance *= max(0.3, 1.0 - colonist.stats["resolve"] * 0.5)
        if colonist.karma < 0.2:
            death_chance += 0.05

        if rng.random() < death_chance:
            colonist.alive = False
            colonist.year_of_death = state.year
            cause = ("starvation" if food_pp < 50
                     else "suffocation" if o2_pp < 50
                     else f"killed by {event['type']}")
            colonist.cause_of_death = cause
            colonist.memory.append(f"Year {state.year}: I died. Cause: {cause}.")
            narratives.append(f"☠ {colonist.name} dies from {cause} in year {state.year}")

    return narratives


# ---------------------------------------------------------------------------
# Relationship evolution
# ---------------------------------------------------------------------------

def evolve_relationships(state: ColonyState, actions: dict[str, str],
                         rng: random.Random) -> None:
    """Evolve relationships based on this year's actions."""
    alive = state.alive_colonists()
    for colonist in alive:
        my_action = actions.get(colonist.id, "rest")
        for other in alive:
            if other.id == colonist.id:
                continue
            other_action = actions.get(other.id, "rest")
            delta = 0.0
            if my_action == other_action and my_action in ("repair", "farm", "terraform"):
                delta += 0.08
            if other_action == "share":
                delta += 0.05
            if other_action == "hoard":
                delta -= 0.03
            if other_action == "sabotage":
                delta -= 0.15
            if other_action == "mediate":
                delta += 0.04
            delta += rng.uniform(-0.02, 0.02)

            if other.id in colonist.relationships:
                colonist.relationships[other.id] = round(
                    max(-1.0, min(1.0, colonist.relationships[other.id] + delta)), 3)


def consume_resources(state: ColonyState) -> None:
    """Annual resource consumption by living colonists."""
    alive_count = len(state.alive_colonists())
    if alive_count == 0:
        return
    r = state.resources
    r["food"] = max(MIN_RESOURCE, r["food"] - alive_count * 100)
    r["water"] = max(MIN_RESOURCE, r["water"] - alive_count * 80)
    r["oxygen"] = max(MIN_RESOURCE, r["oxygen"] - alive_count * 120)
    r["power"] = max(MIN_RESOURCE, r["power"] - alive_count * 60)
    # Passive production
    r["food"] = min(MAX_RESOURCE, r["food"] + 300 + int(state.terraforming_progress * 200))
    r["water"] = min(MAX_RESOURCE, r["water"] + 400)
    r["oxygen"] = min(MAX_RESOURCE, r["oxygen"] + 350 + int(state.terraforming_progress * 150))
    r["power"] = min(MAX_RESOURCE, r["power"] + 250)


# ---------------------------------------------------------------------------
# Emergent pattern detection
# ---------------------------------------------------------------------------

def detect_patterns(state: ColonyState) -> list[str]:
    """Detect emergent governance and social patterns."""
    patterns: list[str] = []
    alive = state.alive_colonists()

    if not alive:
        return ["EXTINCTION: All colonists have perished."]

    for c in alive:
        avg_trust = sum(
            other.relationships.get(c.id, 0) for other in alive if other.id != c.id
        ) / max(1, len(alive) - 1)
        if avg_trust > 0.4 and c.karma > 0.7:
            patterns.append(f"LEADER: {c.name} (trust={avg_trust:.2f}, karma={c.karma:.2f})")
        if avg_trust < -0.3 and c.karma < 0.3:
            patterns.append(f"PARIAH: {c.name} (trust={avg_trust:.2f}, karma={c.karma:.2f})")

    if len(alive) >= 4:
        for i, a in enumerate(alive):
            for j, b in enumerate(alive):
                if j <= i:
                    continue
                mutual = (a.relationships.get(b.id, 0) + b.relationships.get(a.id, 0)) / 2
                if mutual > 0.5:
                    patterns.append(f"ALLIANCE: {a.name} + {b.name} (trust={mutual:.2f})")

    if state.year >= 50 and len(state.subsim_log) > 10:
        for c in alive:
            if c.stats["paranoia"] > 0.7 and c.stats["faith"] > 0.5:
                patterns.append(f"META-AWARENESS: {c.name} suspects this is a simulation")
                break

    # Governance structure classification
    gov_type = classify_governance(state)
    if gov_type != "anarchy":
        patterns.append(f"GOVERNANCE: {gov_type} (year {state.year})")

    return patterns


def classify_governance(state: ColonyState) -> str:
    """Classify the emergent governance structure based on colony dynamics."""
    alive = state.alive_colonists()
    if len(alive) < 2:
        return "anarchy"

    # Find potential leaders (high trust + karma)
    leaders = []
    for c in alive:
        avg_trust = sum(
            other.relationships.get(c.id, 0) for other in alive if other.id != c.id
        ) / max(1, len(alive) - 1)
        if avg_trust > 0.3 and c.karma > 0.6:
            leaders.append((c, avg_trust))

    # Theocracy: leader with high faith
    if leaders and any(c.stats.get("faith", 0) > 0.7 for c, _ in leaders):
        return "theocracy"

    # Dictatorship: single dominant leader with much higher trust
    if len(leaders) == 1 and leaders[0][1] > 0.5:
        return "dictatorship"

    # Council: 3+ leaders with moderate trust
    if len(leaders) >= 3:
        return "council"

    # Democracy: active voting (3+ resolved proposals)
    decided = [g for g in state.governance if g.passed is not None]
    if len(decided) >= 3:
        return "democracy"

    # Technocracy: top 2 colonists by coding skill lead
    coders = sorted(alive, key=lambda c: c.skills.get("coding", 0), reverse=True)
    if len(coders) >= 2 and coders[0].skills.get("coding", 0) > 0.7:
        top_trust = sum(
            other.relationships.get(coders[0].id, 0) for other in alive if other.id != coders[0].id
        ) / max(1, len(alive) - 1)
        if top_trust > 0.2:
            return "technocracy"

    return "anarchy"


# ---------------------------------------------------------------------------
# Birth mechanics
# ---------------------------------------------------------------------------

MARS_BORN_NAMES = [
    "Nova", "Eos", "Phobos", "Deimos", "Olympia",
    "Tharsis", "Valles", "Elysium", "Chryse", "Argyre",
]


def maybe_birth(state: ColonyState, rng: random.Random) -> list[str]:
    """After year 15, Mars-born colonists may arrive. Returns narrative lines."""
    narratives: list[str] = []
    alive = state.alive_colonists()
    if state.year < 15 or len(alive) < 3:
        return narratives

    # Probability increases with population, terraforming, and food
    birth_chance = 0.05 + state.terraforming_progress * 0.1
    if state.resources["food"] > 2000:
        birth_chance += 0.05
    if len(alive) >= 6:
        birth_chance += 0.05

    if rng.random() < birth_chance and state.births < len(MARS_BORN_NAMES):
        name = MARS_BORN_NAMES[state.births]
        cid = f"mars-born-{name.lower()}"
        element = ELEMENTS[state.births % len(ELEMENTS)]

        # Mars-born have different stat distributions — higher resolve, lower paranoia
        stats = {
            "resolve": round(min(1.0, rng.uniform(0.5, 0.9)), 2),
            "improvisation": round(rng.uniform(0.3, 0.8), 2),
            "empathy": round(rng.uniform(0.3, 0.7), 2),
            "hoarding": round(rng.uniform(0.0, 0.3), 2),
            "faith": round(rng.uniform(0.1, 0.6), 2),
            "paranoia": round(rng.uniform(0.0, 0.3), 2),
        }
        skills = {
            "terraforming": round(rng.uniform(0.3, 0.7), 2),
            "hydroponics": round(rng.uniform(0.2, 0.6), 2),
            "mediation": round(rng.uniform(0.2, 0.5), 2),
            "coding": round(rng.uniform(0.2, 0.7), 2),
            "prayer": round(rng.uniform(0.0, 0.3), 2),
            "sabotage": round(rng.uniform(0.0, 0.1), 2),
        }
        rels = {c.id: round(rng.uniform(0.1, 0.5), 3) for c in alive}

        newborn = Colonist(
            id=cid, name=name, element=element, stats=stats, skills=skills,
            relationships=rels, memory=[f"Year {state.year}: Born on Mars. First generation."],
            karma=0.5, alive=True,
        )
        state.colonists.append(newborn)
        # Add newborn to existing colonist relationship maps
        for c in alive:
            c.relationships[cid] = round(rng.uniform(0.1, 0.4), 3)
        state.births += 1
        narratives.append(f"🌱 {name} is born — first generation Mars-born! Element: {element}")

    return narratives


# ---------------------------------------------------------------------------
# Diary entries
# ---------------------------------------------------------------------------

def generate_diary_entries(state: ColonyState, event: dict, actions: dict[str, str],
                           rng: random.Random) -> list[dict]:
    """Generate diary entries from 3 colonists per year."""
    alive = state.alive_colonists()
    if len(alive) < 3:
        diarists = alive[:]
    else:
        diarists = rng.sample(alive, 3)

    entries = []
    for c in diarists:
        action = actions.get(c.id, "rest")
        # Build a diary entry reflecting personality
        moods = {
            "fire": ["fierce", "burning", "restless"],
            "water": ["flowing", "calm", "deep"],
            "earth": ["steady", "rooted", "patient"],
            "air": ["scattered", "free", "drifting"],
        }
        mood = rng.choice(moods.get(c.element, ["neutral"]))
        food_concern = "hungry" if state.resources["food"] < 800 else "fed"
        friend_ids = [oid for oid, rel in c.relationships.items() if rel > 0.3]
        enemy_ids = [oid for oid, rel in c.relationships.items() if rel < -0.2]

        lines = [f"Year {state.year}. I feel {mood} today."]
        lines.append(f"The {event['type']} shaped everything. I chose to {action}.")
        if food_concern == "hungry":
            lines.append("We're running low on food. Tensions are rising.")
        if friend_ids:
            lines.append(f"I trust {len(friend_ids)} of my companions still.")
        if enemy_ids:
            lines.append(f"I watch my back around {len(enemy_ids)} others.")
        if c.stats.get("paranoia", 0) > 0.7 and state.year > 50:
            lines.append("Sometimes I wonder... is any of this real?")
        if c.karma > 0.8:
            lines.append("The colony needs me. I carry that weight gladly.")
        elif c.karma < 0.3:
            lines.append("They look at me differently now. I can feel it.")

        entries.append({
            "colonist": c.id,
            "name": c.name,
            "element": c.element,
            "year": state.year,
            "text": " ".join(lines),
        })
    return entries


# ---------------------------------------------------------------------------
# Constitutional amendment promotion
# ---------------------------------------------------------------------------

def check_amendment_promotion(state: ColonyState) -> list[dict]:
    """Check if any depth-2+ sub-sim insight is strong enough to promote."""
    amendments: list[dict] = []
    deep_sims = [s for s in state.subsim_log if s.depth >= 2]
    if len(deep_sims) < 3:
        return amendments

    # Look for governance patterns in deep sub-sims
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
            action, subsim_reason = decide_action(colonist, state, event, rng)
            year_actions[colonist.id] = action
            narrative = apply_action(colonist, action, state, rng)
            year_narratives.append(narrative)

            if subsim_reason:
                # Parse depth from the sub-sim reason string
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
        "patterns": result["patterns"],
        "governance": result["governance"],
        "active_laws": result["active_laws"],
        "collapsed": result["collapsed"],
        "terraforming_progress": result["terraforming_progress"],
        "amendments": result.get("amendments", []),
        "diaries": result.get("diaries", [])[-30:],  # Last 30 diary entries for dashboard
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def write_year_chapters(result: dict, output_dir: Path) -> None:
    """Write per-year JSON chapters to docs/mars-100/years/."""
    years_dir = output_dir / "years"
    years_dir.mkdir(parents=True, exist_ok=True)
    for narrative in result["narratives"]:
        year = narrative["year"]
        # Find matching timeline snapshot
        snap = next((t for t in result["timeline"] if t["year"] == year), {})
        # Find matching diary entries
        diaries = [d for d in result.get("diaries", []) if d["year"] == year]
        chapter = {
            "year": year,
            "narrative": narrative["lines"],
            "snapshot": snap,
            "diaries": diaries,
        }
        (years_dir / f"year-{year:03d}.json").write_text(json.dumps(chapter, indent=2))


def write_soul_files(result: dict, output_dir: Path) -> None:
    """Write per-colonist soul files to docs/mars-100/colonists/."""
    colonists_dir = output_dir / "colonists"
    colonists_dir.mkdir(parents=True, exist_ok=True)
    for colonist in result["colonists"]:
        cid = colonist["id"]
        soul = {
            "id": cid,
            "name": colonist["name"],
            "element": colonist["element"],
            "alive": colonist["alive"],
            "year_of_death": colonist["year_of_death"],
            "cause_of_death": colonist["cause_of_death"],
            "karma": colonist["karma"],
            "stats": colonist["stats"],
            "skills": colonist["skills"],
            "memory": colonist["memory"],
            "soul_log": result.get("soul_files", {}).get(cid, []),
        }
        (colonists_dir / f"{cid}.json").write_text(json.dumps(soul, indent=2))


def main() -> None:
    """Run Mars-100 and write output."""
    import argparse

    parser = argparse.ArgumentParser(description="Mars-100 recursive colony simulation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "docs" / "mars-100"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Mars-100 — {args.years} years, seed {args.seed}...")

    result = run_simulation(seed=args.seed, years=args.years)
    dashboard = build_dashboard_data(result)

    (output_dir / "full_result.json").write_text(json.dumps(result, indent=2))
    (output_dir / "data.json").write_text(json.dumps(dashboard, separators=(",", ":")))

    # Write per-year chapters and per-colonist soul files
    write_year_chapters(result, output_dir)
    write_soul_files(result, output_dir)

    if not args.quiet:
        alive = len([c for c in result["colonists"] if c["alive"]])
        dead = len([c for c in result["colonists"] if not c["alive"]])
        births = result["_meta"].get("total_births", 0)
        gov = result["_meta"].get("governance_type", "anarchy")
        amendments = result["_meta"].get("amendments_proposed", 0)
        print(f"\n  Survivors: {alive}/{len(result['colonists'])} | Deaths: {dead} | Births: {births}")
        print(f"  Laws: {len(result['active_laws'])} | Sub-sims: {result['_meta']['total_subsims']}")
        print(f"  Terraform: {result['terraforming_progress']:.1%} | Governance: {gov}")
        print(f"  Amendments proposed: {amendments} | Collapsed: {result['collapsed']}")
        if result.get("amendments"):
            print(f"\n  Constitutional Amendments:")
            for a in result["amendments"]:
                print(f"    Year {a['year']}: {a['title']}")
                print(f"      Rappterbook: {a['rappterbook_analog'][:80]}...")
        if result["patterns"]:
            print(f"\n  Emergent patterns (last 5):")
            for p in result["patterns"][-5:]:
                for pat in p["patterns"][:2]:
                    print(f"    Year {p['year']}: {pat}")
        print(f"\n  Output: {output_dir}")
        print(f"    Years: {output_dir / 'years'}/")
        print(f"    Souls: {output_dir / 'colonists'}/")


if __name__ == "__main__":
    main()
