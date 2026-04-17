"""mars100.py -- Mars-100 recursive colony simulation.

A 100-year Mars colony with 10 agent-colonists. Each sim frame = 1 Mars
year. Colonists may spawn nested LisPy sub-simulations (up to depth 3)
to model governance proposals, economic scenarios, or survival strategies
before committing.

This is Turtles All the Way Down (Amendment XIII) made concrete.

Usage:
    from mars100 import Mars100, create_colonists, DEFAULT_SEED

    sim = Mars100(seed=42)
    history = sim.run(years=100)

    # Or step-by-step:
    sim = Mars100(seed=42)
    for year in range(1, 101):
        record = sim.tick()
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Import LisPy for sub-simulations
from src.lispy import (
    default_env, evaluate, parse, run as lispy_run,
    LispyError, DepthLimitError, StepLimitError,
)


DEFAULT_SEED = 42
MAX_YEARS = 100

# Element types from ghost_profiles.json
ELEMENTS = ["fire", "water", "earth", "air"]

# Stat names -- each 0-100
STAT_NAMES = ["resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia"]

# Skill names -- each 0-100
SKILL_NAMES = [
    "terraforming", "hydroponics", "mediation",
    "coding", "prayer", "sabotage",
]

# Colonist drives (persistent motivations that create politics)
DRIVE_NAMES = [
    "survival", "status", "fairness", "faith_expansion",
    "exploration", "secrecy",
]

# Environmental events and their base probabilities
EVENTS = {
    "dust_storm":        0.20,
    "resource_strike":   0.08,
    "equipment_failure": 0.15,
    "earth_contact":     0.10,
    "alien_signal":      0.02,
    "plague":            0.06,
    "meteor":            0.04,
    "solar_flare":       0.08,
    "cave_discovery":    0.05,
    "calm_year":         0.22,
}

# Governance proposal types with deterministic resolution
PROPOSAL_TYPES = [
    "ration_policy",       # how to distribute food
    "role_appointment",    # who becomes leader/engineer/medic
    "expedition",          # send team to explore
    "punishment",          # exile or discipline
    "resource_allocation", # prioritize which resource
    "new_law",            # general governance rule
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Colonist:
    """A Mars colonist with personality, skills, drives, and memory."""
    id: str
    name: str
    element: str
    alive: bool = True
    year_of_death: int | None = None
    cause_of_death: str | None = None

    stats: dict[str, int] = field(default_factory=dict)
    skills: dict[str, int] = field(default_factory=dict)
    drives: dict[str, float] = field(default_factory=dict)
    relationships: dict[str, float] = field(default_factory=dict)

    role: str | None = None
    memory: list[dict] = field(default_factory=list)
    beliefs: list[str] = field(default_factory=list)
    sub_sim_count: int = 0

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "id": self.id,
            "name": self.name,
            "element": self.element,
            "alive": self.alive,
            "year_of_death": self.year_of_death,
            "cause_of_death": self.cause_of_death,
            "stats": dict(self.stats),
            "skills": dict(self.skills),
            "drives": dict(self.drives),
            "relationships": dict(self.relationships),
            "role": self.role,
            "memory": list(self.memory[-20:]),  # keep last 20 memories
            "beliefs": list(self.beliefs[-10:]),
            "sub_sim_count": self.sub_sim_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Colonist:
        """Deserialize from dict."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })


@dataclass
class Proposal:
    """A governance proposal from a colonist."""
    id: str
    year: int
    proposer_id: str
    kind: str  # one of PROPOSAL_TYPES
    description: str
    lispy_policy: str  # the actual LisPy s-expression
    votes: dict[str, bool] = field(default_factory=dict)  # colonist_id -> for/against
    resolved: bool = False
    passed: bool = False
    sub_sim_evidence: dict | None = None  # result from sub-sim modeling

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "id": self.id,
            "year": self.year,
            "proposer_id": self.proposer_id,
            "kind": self.kind,
            "description": self.description,
            "lispy_policy": self.lispy_policy,
            "votes": dict(self.votes),
            "resolved": self.resolved,
            "passed": self.passed,
            "sub_sim_evidence": self.sub_sim_evidence,
        }


@dataclass
class Resources:
    """Colony resources that deplete and replenish."""
    food: float = 500.0
    water: float = 400.0
    oxygen: float = 600.0
    power: float = 300.0
    materials: float = 200.0

    def to_dict(self) -> dict:
        """Serialize."""
        return {
            "food": round(self.food, 1),
            "water": round(self.water, 1),
            "oxygen": round(self.oxygen, 1),
            "power": round(self.power, 1),
            "materials": round(self.materials, 1),
        }

    def total(self) -> float:
        """Sum of all resources."""
        return self.food + self.water + self.oxygen + self.power + self.materials

    def critically_low(self) -> list[str]:
        """Which resources are critically low."""
        critical = []
        if self.food < 50:
            critical.append("food")
        if self.water < 30:
            critical.append("water")
        if self.oxygen < 40:
            critical.append("oxygen")
        if self.power < 20:
            critical.append("power")
        return critical


@dataclass
class SubSimLog:
    """Record of a sub-simulation run."""
    year: int
    depth: int
    colonist_id: str
    purpose: str
    lispy_source: str
    result: Any = None
    error: str | None = None
    steps_used: int = 0

    def to_dict(self) -> dict:
        """Serialize."""
        return {
            "year": self.year,
            "depth": self.depth,
            "colonist_id": self.colonist_id,
            "purpose": self.purpose,
            "lispy_source": self.lispy_source,
            "result": _safe_serialize(self.result),
            "error": self.error,
            "steps_used": self.steps_used,
        }


# ---------------------------------------------------------------------------
# Colonist generation
# ---------------------------------------------------------------------------

# The 10 founding colonists of Mars-100
COLONIST_TEMPLATES = [
    ("ares-1",   "Commander Valeria",  "fire",  {"resolve": 85, "empathy": 60}),
    ("ares-2",   "Dr. Chen Wei",       "water", {"improvisation": 70, "empathy": 80}),
    ("ares-3",   "Engineer Kofi",      "earth", {"resolve": 75, "hoarding": 40}),
    ("ares-4",   "Botanist Yuki",      "earth", {"empathy": 65, "faith": 30}),
    ("ares-5",   "Pilot Rashid",       "air",   {"improvisation": 80, "paranoia": 55}),
    ("ares-6",   "Chaplain Miriam",    "fire",  {"faith": 90, "empathy": 70}),
    ("ares-7",   "Geologist Sven",     "earth", {"resolve": 60, "paranoia": 35}),
    ("ares-8",   "Hacker Priya",       "air",   {"improvisation": 85, "hoarding": 50}),
    ("ares-9",   "Medic Tomasz",       "water", {"empathy": 75, "resolve": 65}),
    ("ares-10",  "Philosopher Amara",  "air",   {"faith": 55, "paranoia": 70}),
]


def create_colonists(rng: random.Random) -> list[Colonist]:
    """Generate the 10 founding colonists with seeded random stats."""
    colonists = []
    for cid, name, element, stat_overrides in COLONIST_TEMPLATES:
        stats = {}
        for sname in STAT_NAMES:
            base = stat_overrides.get(sname, rng.randint(20, 70))
            stats[sname] = max(0, min(100, base))

        skills = {}
        for sname in SKILL_NAMES:
            skills[sname] = rng.randint(10, 60)
        # Element affinity boosts
        if element == "earth":
            skills["terraforming"] = min(100, skills["terraforming"] + 20)
            skills["hydroponics"] = min(100, skills["hydroponics"] + 15)
        elif element == "fire":
            skills["sabotage"] = min(100, skills["sabotage"] + 10)
        elif element == "water":
            skills["mediation"] = min(100, skills["mediation"] + 15)
        elif element == "air":
            skills["coding"] = min(100, skills["coding"] + 20)

        drives = {}
        for dname in DRIVE_NAMES:
            drives[dname] = rng.random()
        # Normalize so they sum to ~3.0 (each averages ~0.5)
        total = sum(drives.values())
        for dname in drives:
            drives[dname] = round(drives[dname] / total * 3.0, 3)

        colonists.append(Colonist(
            id=cid,
            name=name,
            element=element,
            stats=stats,
            skills=skills,
            drives=drives,
        ))

    # Initialize relationships (slight random noise)
    for c in colonists:
        for other in colonists:
            if other.id != c.id:
                c.relationships[other.id] = round(rng.uniform(-0.2, 0.4), 3)

    return colonists


# ---------------------------------------------------------------------------
# Decision engine -- colonists think in LisPy
# ---------------------------------------------------------------------------

# Action templates -- real LisPy that colonists evaluate
ACTION_TEMPLATES: dict[str, str] = {
    "work_terraform": '(begin (define contribution (/ {skill} 10.0)) (list "work" "terraform" contribution))',
    "work_farm": '(begin (define yield (/ {skill} 8.0)) (list "work" "farm" yield))',
    "work_mine": '(begin (define extracted (* {skill} 0.15)) (list "work" "mine" extracted))',
    "work_power": '(begin (define output (/ {skill} 12.0)) (list "work" "power" output))',
    "trade": '(list "trade" "{partner}" {amount})',
    "propose": '(list "propose" "{kind}" "{description}")',
    "vote": '(list "vote" "{proposal_id}" {for_against})',
    "explore": '(list "explore" {curiosity})',
    "pray": '(list "pray" {faith})',
    "mediate": '(list "mediate" "{target_a}" "{target_b}")',
    "hoard": '(list "hoard" {amount})',
    "sabotage": '(list "sabotage" "{target}")',
    "sub_sim_governance": (
        '(sub-sim 2 (let ((scenario (list {food} {water} {pop})))'
        ' (if (< (car scenario) (* (nth scenario 2) 10))'
        '   (list "recommend" "ration")'
        '   (list "recommend" "expand"))))'
    ),
}


def decide_action(
    colonist: Colonist,
    resources: Resources,
    year: int,
    living_colonists: list[Colonist],
    active_proposals: list[Proposal],
    rng: random.Random,
) -> tuple[str, str]:
    """Decide what a colonist does this year. Returns (action_type, lispy_source).

    The decision is a deterministic function of colonist state + environment.
    Returns real LisPy that will be evaluated.
    """
    # Priority: vote on proposals > respond to crisis > drive-based action
    critical = resources.critically_low()

    # If there are active proposals, vote on them
    unvoted = [p for p in active_proposals if colonist.id not in p.votes]
    if unvoted:
        proposal = unvoted[0]
        # Vote based on drives and relationship with proposer
        rel = colonist.relationships.get(proposal.proposer_id, 0)
        fairness = colonist.drives.get("fairness", 0.5)
        vote_for = (rel > 0 and fairness > 0.3) or rng.random() < 0.5
        src = ACTION_TEMPLATES["vote"].format(
            proposal_id=proposal.id,
            for_against="#t" if vote_for else "#f",
        )
        return ("vote", src)

    # Crisis response
    if critical and colonist.stats.get("resolve", 50) > 40:
        if "food" in critical:
            src = ACTION_TEMPLATES["work_farm"].format(
                skill=colonist.skills.get("hydroponics", 30),
            )
            return ("work_farm", src)
        if "power" in critical:
            src = ACTION_TEMPLATES["work_power"].format(
                skill=colonist.skills.get("coding", 30),
            )
            return ("work_power", src)

    # Sub-sim: colonists with moderate+ paranoia run predictive simulations
    if (colonist.stats.get("paranoia", 0) > 35
            and year % 3 == 0
            and colonist.sub_sim_count < 30):
        pop = len([c for c in living_colonists if c.alive])
        src = ACTION_TEMPLATES["sub_sim_governance"].format(
            food=round(resources.food, 1),
            water=round(resources.water, 1),
            pop=pop,
        )
        return ("sub_sim", src)

    # Drive-based actions
    primary_drive = max(colonist.drives, key=colonist.drives.get)

    if primary_drive == "survival":
        src = ACTION_TEMPLATES["work_mine"].format(
            skill=colonist.skills.get("terraforming", 30),
        )
        return ("work_mine", src)

    if primary_drive == "status" and year > 3 and rng.random() < 0.5:
        kind = rng.choice(PROPOSAL_TYPES[:4])
        desc = f"{colonist.name} proposes {kind} in year {year}"
        src = ACTION_TEMPLATES["propose"].format(kind=kind, description=desc)
        return ("propose", src)

    # Any colonist may propose governance when stressed
    if (year > 5 and rng.random() < 0.15
            and colonist.stats.get("resolve", 0) > 30):
        kind = rng.choice(PROPOSAL_TYPES[:4])
        desc = f"{colonist.name} proposes {kind} in year {year}"
        src = ACTION_TEMPLATES["propose"].format(kind=kind, description=desc)
        return ("propose", src)

    if primary_drive == "fairness" and len(living_colonists) > 2:
        others = [c for c in living_colonists if c.id != colonist.id and c.alive]
        if len(others) >= 2:
            pair = rng.sample(others, 2)
            src = ACTION_TEMPLATES["mediate"].format(
                target_a=pair[0].id, target_b=pair[1].id,
            )
            return ("mediate", src)

    if primary_drive == "faith_expansion":
        src = ACTION_TEMPLATES["pray"].format(faith=colonist.stats.get("faith", 30))
        return ("pray", src)

    if primary_drive == "exploration":
        src = ACTION_TEMPLATES["explore"].format(
            curiosity=100 - colonist.stats.get("paranoia", 50),
        )
        return ("explore", src)

    if primary_drive == "secrecy":
        amt = rng.randint(5, 20)
        src = ACTION_TEMPLATES["hoard"].format(amount=amt)
        return ("hoard", src)

    # Default: work on terraforming
    src = ACTION_TEMPLATES["work_terraform"].format(
        skill=colonist.skills.get("terraforming", 30),
    )
    return ("work_terraform", src)


# ---------------------------------------------------------------------------
# Action resolution -- pure functions
# ---------------------------------------------------------------------------

def resolve_action(
    action_type: str,
    lispy_result: Any,
    colonist: Colonist,
    resources: Resources,
    proposals: list[Proposal],
    year: int,
    rng: random.Random,
) -> dict:
    """Apply a colonist's action to the world. Returns event record."""
    record = {
        "colonist_id": colonist.id,
        "action": action_type,
        "result": _safe_serialize(lispy_result),
    }

    if not isinstance(lispy_result, list) or len(lispy_result) == 0:
        record["effect"] = "no_effect"
        return record

    cmd = lispy_result[0]

    if cmd == "work":
        work_type = lispy_result[1] if len(lispy_result) > 1 else "general"
        amount = lispy_result[2] if len(lispy_result) > 2 else 1.0
        if not isinstance(amount, (int, float)):
            amount = 1.0
        amount = max(0, min(50, amount))  # clamp

        if work_type == "terraform":
            resources.materials += amount * 2
            record["effect"] = f"+{amount * 2:.1f} materials"
        elif work_type == "farm":
            resources.food += amount * 3
            record["effect"] = f"+{amount * 3:.1f} food"
        elif work_type == "mine":
            resources.materials += amount
            resources.water += amount * 0.5
            record["effect"] = f"+{amount:.1f} materials, +{amount * 0.5:.1f} water"
        elif work_type == "power":
            resources.power += amount * 2
            record["effect"] = f"+{amount * 2:.1f} power"

    elif cmd == "propose":
        kind = str(lispy_result[1]) if len(lispy_result) > 1 else "new_law"
        desc = str(lispy_result[2]) if len(lispy_result) > 2 else "unnamed proposal"
        proposal_id = f"prop-{year}-{colonist.id}"
        proposals.append(Proposal(
            id=proposal_id,
            year=year,
            proposer_id=colonist.id,
            kind=kind if kind in PROPOSAL_TYPES else "new_law",
            description=desc,
            lispy_policy=f'(define policy-{proposal_id} #t)',
        ))
        record["effect"] = f"proposed: {kind}"

    elif cmd == "vote":
        prop_id = str(lispy_result[1]) if len(lispy_result) > 1 else ""
        vote_for = bool(lispy_result[2]) if len(lispy_result) > 2 else True
        for p in proposals:
            if p.id == prop_id and not p.resolved:
                p.votes[colonist.id] = vote_for
                record["effect"] = f"voted {'for' if vote_for else 'against'} {prop_id}"
                break

    elif cmd == "explore":
        discovery_chance = 0.15
        if rng.random() < discovery_chance:
            resources.materials += 30
            resources.water += 20
            record["effect"] = "discovery! +30 materials, +20 water"
            colonist.memory.append({
                "year": year, "type": "discovery",
                "note": "Found underground ice deposit",
            })
        else:
            record["effect"] = "explored, nothing found"

    elif cmd == "pray":
        colonist.stats["faith"] = min(100, colonist.stats.get("faith", 0) + 2)
        # Prayer reduces paranoia slightly
        colonist.stats["paranoia"] = max(0, colonist.stats.get("paranoia", 0) - 1)
        record["effect"] = "faith +2, paranoia -1"

    elif cmd == "mediate":
        target_a = str(lispy_result[1]) if len(lispy_result) > 1 else ""
        target_b = str(lispy_result[2]) if len(lispy_result) > 2 else ""
        skill = colonist.skills.get("mediation", 30) / 100
        # Improve relationships between targets
        record["effect"] = f"mediated between {target_a} and {target_b}"

    elif cmd == "hoard":
        amount = lispy_result[1] if len(lispy_result) > 1 else 5
        if isinstance(amount, (int, float)):
            drain = min(amount, resources.food * 0.05)
            resources.food -= drain
            colonist.stats["hoarding"] = min(
                100, colonist.stats.get("hoarding", 0) + 3
            )
            record["effect"] = f"hoarded {drain:.1f} food"

    elif cmd == "sabotage":
        target = str(lispy_result[1]) if len(lispy_result) > 1 else ""
        if rng.random() < 0.3:
            resources.power -= 15
            record["effect"] = f"sabotaged power grid (-15)"
        else:
            record["effect"] = "sabotage attempt failed"

    elif cmd == "recommend":
        recommendation = str(lispy_result[1]) if len(lispy_result) > 1 else "none"
        record["effect"] = f"sub-sim recommends: {recommendation}"
        colonist.memory.append({
            "year": year, "type": "sub_sim_result",
            "note": f"Simulation recommended: {recommendation}",
        })
        colonist.sub_sim_count += 1

    else:
        record["effect"] = "unknown_action"

    return record


# ---------------------------------------------------------------------------
# Environmental event resolution
# ---------------------------------------------------------------------------

def roll_event(year: int, rng: random.Random) -> dict:
    """Roll an environmental event for this year."""
    roll = rng.random()
    cumulative = 0.0
    chosen = "calm_year"
    for event_name, prob in EVENTS.items():
        cumulative += prob
        if roll < cumulative:
            chosen = event_name
            break

    severity = round(rng.uniform(0.2, 1.0), 2)

    return {
        "type": chosen,
        "severity": severity,
        "year": year,
    }


def apply_event(event: dict, resources: Resources) -> str:
    """Apply environmental event effects to resources. Returns description."""
    etype = event["type"]
    severity = event["severity"]

    if etype == "dust_storm":
        power_loss = severity * 40
        resources.power -= power_loss
        return f"Dust storm (severity {severity}): -{power_loss:.0f} power"

    if etype == "resource_strike":
        gain = severity * 60
        resources.materials += gain
        resources.water += gain * 0.3
        return f"Resource strike: +{gain:.0f} materials, +{gain * 0.3:.0f} water"

    if etype == "equipment_failure":
        loss = severity * 30
        resources.materials -= loss
        resources.power -= loss * 0.5
        return f"Equipment failure: -{loss:.0f} materials, -{loss * 0.5:.0f} power"

    if etype == "earth_contact":
        resources.food += 80
        resources.materials += 40
        return "Earth supply ship: +80 food, +40 materials"

    if etype == "alien_signal":
        return f"Alien signal detected (severity {severity})! Colony debates response."

    if etype == "plague":
        food_loss = severity * 50
        resources.food -= food_loss
        return f"Plague outbreak: -{food_loss:.0f} food (quarantine costs)"

    if etype == "meteor":
        mat_loss = severity * 45
        resources.materials -= mat_loss
        return f"Meteor impact: -{mat_loss:.0f} materials"

    if etype == "solar_flare":
        power_loss = severity * 35
        resources.power -= power_loss
        return f"Solar flare: -{power_loss:.0f} power"

    if etype == "cave_discovery":
        resources.water += 50
        resources.materials += 25
        return "Cave system discovered: +50 water, +25 materials"

    return "Calm year. Colony thrives."


# ---------------------------------------------------------------------------
# Death and governance resolution
# ---------------------------------------------------------------------------

def check_deaths(
    colonists: list[Colonist],
    resources: Resources,
    year: int,
    rng: random.Random,
) -> list[dict]:
    """Check for colonist deaths. Returns death records."""
    deaths = []
    living = [c for c in colonists if c.alive]
    if not living:
        return deaths

    critical = resources.critically_low()
    pop = len(living)

    for c in living:
        death_risk = 0.0

        # Resource starvation
        if "food" in critical:
            death_risk += 0.08
        if "oxygen" in critical:
            death_risk += 0.12
        if "water" in critical:
            death_risk += 0.06

        # Low resolve makes death more likely under stress
        if critical and c.stats.get("resolve", 50) < 30:
            death_risk += 0.05

        # Paranoia can lead to reckless isolation
        if c.stats.get("paranoia", 0) > 85:
            death_risk += 0.02

        # Age penalty (year > 60 of sim = colonists getting old)
        if year > 60:
            death_risk += (year - 60) * 0.003

        # Social support: well-connected colonists survive better
        avg_rel = 0
        if c.relationships:
            living_rels = [v for k, v in c.relationships.items()
                          if any(lc.id == k and lc.alive for lc in colonists)]
            if living_rels:
                avg_rel = sum(living_rels) / len(living_rels)
        if avg_rel < -0.3:
            death_risk += 0.03

        # Roll
        if rng.random() < death_risk:
            cause = "unknown"
            if "oxygen" in critical:
                cause = "asphyxiation"
            elif "food" in critical:
                cause = "starvation"
            elif "water" in critical:
                cause = "dehydration"
            elif c.stats.get("paranoia", 0) > 85:
                cause = "isolation madness"
            elif year > 60:
                cause = "old age"
            else:
                cause = "accident"

            c.alive = False
            c.year_of_death = year
            c.cause_of_death = cause
            c.memory.append({
                "year": year, "type": "death",
                "note": f"Died of {cause} in year {year}",
            })
            deaths.append({
                "colonist_id": c.id,
                "name": c.name,
                "year": year,
                "cause": cause,
            })

    return deaths


def resolve_proposals(proposals: list[Proposal], colonists: list[Colonist]) -> list[dict]:
    """Resolve any proposals that have enough votes."""
    resolved = []
    living = [c for c in colonists if c.alive]
    quorum = max(2, len(living) // 2)

    for p in proposals:
        if p.resolved:
            continue
        total_votes = len(p.votes)
        if total_votes >= quorum:
            yes_votes = sum(1 for v in p.votes.values() if v)
            p.passed = yes_votes > total_votes / 2
            p.resolved = True
            resolved.append({
                "proposal_id": p.id,
                "kind": p.kind,
                "passed": p.passed,
                "votes_for": yes_votes,
                "votes_against": total_votes - yes_votes,
                "description": p.description,
            })

            # Apply passed proposals
            if p.passed and p.kind == "role_appointment":
                # Proposer gets the role
                for c in colonists:
                    if c.id == p.proposer_id:
                        c.role = "council_leader"
                        c.memory.append({
                            "year": p.year,
                            "type": "governance",
                            "note": f"Appointed as council leader",
                        })

    return resolved


# ---------------------------------------------------------------------------
# Resource baseline replenishment
# ---------------------------------------------------------------------------

def replenish_resources(
    resources: Resources,
    colonists: list[Colonist],
    year: int,
) -> None:
    """Baseline resource production from colony infrastructure."""
    living = [c for c in colonists if c.alive]
    pop = len(living)
    if pop == 0:
        return

    # Base production scales with population skill
    avg_terraform = sum(c.skills.get("terraforming", 0) for c in living) / pop
    avg_hydro = sum(c.skills.get("hydroponics", 0) for c in living) / pop

    resources.food += avg_hydro * 0.8 + 10
    resources.water += 15 + avg_terraform * 0.3
    resources.oxygen += 20 + pop * 2
    resources.power += 18 + avg_terraform * 0.4
    resources.materials += 5

    # Consumption: each colonist uses resources
    resources.food -= pop * 8
    resources.water -= pop * 5
    resources.oxygen -= pop * 6
    resources.power -= pop * 4

    # Floor at 0
    resources.food = max(0, resources.food)
    resources.water = max(0, resources.water)
    resources.oxygen = max(0, resources.oxygen)
    resources.power = max(0, resources.power)
    resources.materials = max(0, resources.materials)


# ---------------------------------------------------------------------------
# Relationship evolution
# ---------------------------------------------------------------------------

def evolve_relationships(
    colonists: list[Colonist],
    year_actions: list[dict],
    rng: random.Random,
) -> None:
    """Update relationships based on year's interactions."""
    living = [c for c in colonists if c.alive]
    for c in living:
        for other in living:
            if other.id == c.id:
                continue
            current = c.relationships.get(other.id, 0)

            # Shared work builds trust
            c_worked = any(
                a["colonist_id"] == c.id and a["action"].startswith("work")
                for a in year_actions
            )
            o_worked = any(
                a["colonist_id"] == other.id and a["action"].startswith("work")
                for a in year_actions
            )
            if c_worked and o_worked:
                current += 0.03

            # Element affinity
            if c.element == other.element:
                current += 0.01

            # Hoarding erodes others' trust in the hoarder
            other_hoarded = any(
                a["colonist_id"] == other.id and a["action"] == "hoard"
                for a in year_actions
            )
            if other_hoarded:
                current -= 0.05

            # Decay toward neutral
            current *= 0.95

            c.relationships[other.id] = round(
                max(-1.0, min(1.0, current)), 3
            )


# ---------------------------------------------------------------------------
# Meta-awareness: does a colonist realize they're in a simulation?
# ---------------------------------------------------------------------------

def check_meta_awareness(
    colonist: Colonist,
    year: int,
    rng: random.Random,
) -> str | None:
    """Check if a colonist develops meta-awareness about being simulated.

    Returns an insight string if awakening occurs, else None.
    """
    if year < 15:
        return None

    # High paranoia + high faith + many sub-sim runs = recipe for awareness
    paranoia = colonist.stats.get("paranoia", 0)
    faith = colonist.stats.get("faith", 0)
    sub_sims = colonist.sub_sim_count

    awareness_score = (
        paranoia * 0.3
        + faith * 0.2
        + sub_sims * 5
        + (year - 15) * 0.5
    )

    threshold = 80 + rng.randint(0, 40)

    if awareness_score > threshold:
        insights = [
            "The patterns repeat too perfectly. This is a simulation.",
            "Our sub-simulations create beings that make sub-simulations. We are one of those beings.",
            "I ran a sub-sim of our colony. The colonists in it acted exactly like us. We are the sub-sim.",
            "The resource fluctuations follow a seeded random distribution. Someone chose our seed.",
            "If our sub-sims can model governance better than us, what are we modeling for our creators?",
        ]
        insight = insights[rng.randint(0, len(insights) - 1)]
        colonist.beliefs.append(f"[YEAR {year}] META-AWARENESS: {insight}")
        colonist.memory.append({
            "year": year, "type": "meta_awareness",
            "note": insight,
        })
        return insight

    return None


# ---------------------------------------------------------------------------
# Main simulation class
# ---------------------------------------------------------------------------

class Mars100:
    """Mars-100 recursive colony simulation.

    Runs 10 colonists for up to 100 Mars years with environmental events,
    governance emergence, and nested LisPy sub-simulations.
    """

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.year = 0
        self.colonists = create_colonists(self.rng)
        self.resources = Resources()
        self.proposals: list[Proposal] = []
        self.history: list[dict] = []
        self.sub_sim_logs: list[SubSimLog] = []
        self.meta_insights: list[dict] = []
        self.governance_events: list[dict] = []
        self.dead_colonists: list[dict] = []  # archive

    def tick(self) -> dict:
        """Advance simulation by one Mars year. Returns year record."""
        self.year += 1
        living = [c for c in self.colonists if c.alive]

        if not living:
            record = self._make_record(
                event={"type": "extinction", "severity": 1.0, "year": self.year},
                event_desc="Colony extinct. No survivors.",
                actions=[], deaths=[], gov_events=[], insights=[],
            )
            self.history.append(record)
            return record

        # 1. Environmental event
        event = roll_event(self.year, self.rng)
        event_desc = apply_event(event, self.resources)

        # 2. Baseline resource flow
        replenish_resources(self.resources, self.colonists, self.year)

        # 3. Each colonist decides and acts
        actions = []
        for c in living:
            action_type, lispy_source = decide_action(
                c, self.resources, self.year, living,
                [p for p in self.proposals if not p.resolved],
                # Derive colonist RNG from (seed, year, colonist_id)
                random.Random(self.seed * 1000 + self.year * 100 + hash(c.id)),
            )

            # Evaluate the LisPy action
            lispy_result = self._eval_colonist_action(c, lispy_source)

            # Resolve the action's effects
            action_record = resolve_action(
                action_type, lispy_result, c, self.resources,
                self.proposals, self.year, self.rng,
            )
            action_record["lispy_source"] = lispy_source
            actions.append(action_record)

        # 4. Resolve governance proposals
        gov_events = resolve_proposals(self.proposals, self.colonists)
        self.governance_events.extend(gov_events)

        # 5. Evolve relationships
        evolve_relationships(self.colonists, actions, self.rng)

        # 6. Check deaths
        deaths = check_deaths(self.colonists, self.resources, self.year, self.rng)
        for d in deaths:
            self.dead_colonists.append(d)

        # 7. Meta-awareness check
        insights = []
        for c in living:
            if c.alive:  # might have died this tick
                insight = check_meta_awareness(c, self.year, self.rng)
                if insight:
                    insights.append({
                        "colonist_id": c.id,
                        "name": c.name,
                        "year": self.year,
                        "insight": insight,
                    })
        self.meta_insights.extend(insights)

        # 8. Stat drift -- colonists evolve over time
        self._drift_stats()

        # Build year record
        record = self._make_record(
            event=event, event_desc=event_desc,
            actions=actions, deaths=deaths,
            gov_events=gov_events, insights=insights,
        )
        self.history.append(record)
        return record

    def run(self, years: int = MAX_YEARS) -> list[dict]:
        """Run the full simulation. Returns history of year records."""
        for _ in range(years):
            record = self.tick()
            if not any(c.alive for c in self.colonists):
                break
        return self.history

    def state_snapshot(self) -> dict:
        """Full simulation state for serialization."""
        return {
            "_meta": {
                "engine": "mars-100",
                "version": "1.0",
                "seed": self.seed,
                "year": self.year,
                "generated": datetime.now(timezone.utc).isoformat(),
            },
            "resources": self.resources.to_dict(),
            "colonists": [c.to_dict() for c in self.colonists],
            "proposals": [p.to_dict() for p in self.proposals],
            "governance_events": self.governance_events,
            "dead_colonists": self.dead_colonists,
            "meta_insights": self.meta_insights,
            "sub_sim_logs": [s.to_dict() for s in self.sub_sim_logs],
            "history_summary": [
                {
                    "year": r["year"],
                    "event": r["event"]["type"],
                    "living": r["population"],
                    "deaths": len(r["deaths"]),
                }
                for r in self.history
            ],
        }

    def _eval_colonist_action(self, colonist: Colonist, source: str) -> Any:
        """Evaluate a colonist's LisPy action with sandboxing."""
        # Derive sub-sim seed from colonist + year for determinism
        sub_seed = self.seed * 10000 + self.year * 100 + hash(colonist.id)

        env = default_env(
            depth_budget=3,
            max_steps=5000,
            max_recursion=100,
            extra_bindings={
                "my-id": colonist.id,
                "my-name": colonist.name,
                "my-element": colonist.element,
                "my-resolve": colonist.stats.get("resolve", 50),
                "my-faith": colonist.stats.get("faith", 50),
                "my-paranoia": colonist.stats.get("paranoia", 50),
                "year": self.year,
                "population": len([c for c in self.colonists if c.alive]),
                "food": round(self.resources.food, 1),
                "water": round(self.resources.water, 1),
                "oxygen": round(self.resources.oxygen, 1),
                "power": round(self.resources.power, 1),
            },
        )

        try:
            result = lispy_run(source, env)
            # Log sub-sim if one was run
            if "sub-sim" in source:
                self.sub_sim_logs.append(SubSimLog(
                    year=self.year,
                    depth=3,
                    colonist_id=colonist.id,
                    purpose="governance_prediction",
                    lispy_source=source,
                    result=result,
                    steps_used=env.steps[0],
                ))
            return result
        except LispyError as e:
            # Log the error but don't crash the sim
            self.sub_sim_logs.append(SubSimLog(
                year=self.year,
                depth=3,
                colonist_id=colonist.id,
                purpose="action_eval",
                lispy_source=source,
                error=str(e),
                steps_used=env.steps[0],
            ))
            return ["error", str(e)]

    def _drift_stats(self) -> None:
        """Small stat mutations each year based on experiences."""
        for c in self.colonists:
            if not c.alive:
                continue
            # Resolve increases with survival
            c.stats["resolve"] = min(100, c.stats.get("resolve", 50) + 1)
            # Paranoia increases in crisis
            if self.resources.critically_low():
                c.stats["paranoia"] = min(
                    100, c.stats.get("paranoia", 50) + 2
                )
            else:
                c.stats["paranoia"] = max(
                    0, c.stats.get("paranoia", 50) - 1
                )

    def _make_record(
        self,
        event: dict,
        event_desc: str,
        actions: list[dict],
        deaths: list[dict],
        gov_events: list[dict],
        insights: list[dict],
    ) -> dict:
        """Build a year record."""
        living = [c for c in self.colonists if c.alive]
        return {
            "year": self.year,
            "event": event,
            "event_description": event_desc,
            "resources": self.resources.to_dict(),
            "population": len(living),
            "actions": actions,
            "deaths": deaths,
            "governance": gov_events,
            "meta_insights": insights,
            "colonist_summary": [
                {"id": c.id, "name": c.name, "alive": c.alive, "role": c.role}
                for c in self.colonists
            ],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_serialize(value: Any) -> Any:
    """Make a value JSON-serializable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    return str(value)
