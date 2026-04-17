"""mars100.py — Mars-100 Recursive Colony Simulation Engine.

A 100-year Mars colony simulation with 10 agent-colonists. Each sim frame
= 1 Martian year (~687 Earth days). Colonists make decisions expressed as
LisPy s-expressions. Sub-simulations (up to depth 3) allow colonists to
model governance proposals before committing.

This is Amendment XIII (Turtles All the Way Down) made concrete:
a simulation inside a simulation whose colonists run simulations.

Constitutional:
  - Output of frame N = input to frame N+1 (data sloshing)
  - Dream Catcher protocol: per-year deltas keyed by (year, deterministic_id)
  - Legacy not delete: dead colonists become archived souls
  - Sub-sims sandboxed via LisPy VM (no I/O, pure computation)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.lispy_vm import LispyVM, format_sexpr, LispyError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELEMENTS = ("fire", "water", "earth", "air")
STAT_NAMES = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")
SKILL_NAMES = ("terraforming", "hydroponics", "mediation", "coding", "prayer", "sabotage")

# Environmental events (one per year, weighted by era)
EARLY_EVENTS = [
    "dust_storm", "equipment_failure", "water_strike", "solar_flare",
    "meteor_impact", "supply_drop", "habitat_leak",
]
MID_EVENTS = [
    "dust_storm", "resource_strike", "earth_contact", "epidemic",
    "crop_blight", "power_surge", "diplomatic_crisis", "baby_born",
]
LATE_EVENTS = [
    "dust_storm", "alien_signal", "colony_schism", "philosophical_crisis",
    "technological_breakthrough", "meta_awareness", "terraforming_milestone",
    "generational_shift",
]

# Governance proposal types
PROPOSAL_TYPES = (
    "resource_allocation", "leadership_election", "exile_vote",
    "law_enactment", "exploration_mission", "alliance_proposal",
    "constitutional_amendment", "sub_sim_request",
)

# Physical bounds for conservation law tests
MAX_RESOURCES = 100_000.0
MIN_MORALE = 0.0
MAX_MORALE = 1.0
MAX_RELATIONSHIP = 1.0
MIN_RELATIONSHIP = -1.0
MAX_GOVERNANCE_WEIGHT = 10.0

# Governance forms that can emerge from colony dynamics
GOVERNANCE_FORMS = (
    "anarchy", "consensus", "council", "elected_democracy",
    "autocracy", "theocracy", "technocracy", "commune",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Colonist:
    """A Mars colony inhabitant — both data structure and LisPy program."""

    id: str
    name: str
    element: str
    stats: dict[str, float]
    skills: dict[str, float]
    alive: bool = True
    year_born: int = 0
    year_died: int | None = None
    death_cause: str | None = None
    diary: list[dict] = field(default_factory=list)
    relationships: dict[str, float] = field(default_factory=dict)
    governance_weight: float = 1.0
    proposals_made: int = 0
    votes_cast: int = 0
    sub_sims_run: int = 0
    meta_aware: bool = False
    soul_archived: bool = False

    def to_dict(self) -> dict:
        """Serialize colonist to JSON-safe dict."""
        return {
            "id": self.id,
            "name": self.name,
            "element": self.element,
            "stats": dict(self.stats),
            "skills": dict(self.skills),
            "alive": self.alive,
            "year_born": self.year_born,
            "year_died": self.year_died,
            "death_cause": self.death_cause,
            "diary_count": len(self.diary),
            "last_diary": self.diary[-1] if self.diary else None,
            "relationships": dict(self.relationships),
            "governance_weight": self.governance_weight,
            "proposals_made": self.proposals_made,
            "votes_cast": self.votes_cast,
            "sub_sims_run": self.sub_sims_run,
            "meta_aware": self.meta_aware,
            "soul_archived": self.soul_archived,
        }

    def to_sexpr(self) -> str:
        """Serialize colonist as an s-expression (homoiconic representation)."""
        stats_pairs = " ".join(f'"{k}" {v:.2f}' for k, v in self.stats.items())
        skills_pairs = " ".join(f'"{k}" {v:.2f}' for k, v in self.skills.items())
        return (
            f'(colonist "{self.id}" "{self.name}" "{self.element}" '
            f"(stats {stats_pairs}) "
            f"(skills {skills_pairs}) "
            f"(alive {'#t' if self.alive else '#f'}) "
            f"(weight {self.governance_weight:.2f}))"
        )


@dataclass
class GovernanceProposal:
    """A governance proposal raised by a colonist."""

    id: str
    year: int
    proposer_id: str
    proposal_type: str
    description: str
    value: Any
    votes: dict[str, dict] = field(default_factory=dict)  # colonist_id -> {position, weight}
    resolved: bool = False
    outcome: str | None = None  # "passed", "failed", "withdrawn"
    sub_sim_evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize proposal to dict."""
        return {
            "id": self.id,
            "year": self.year,
            "proposer_id": self.proposer_id,
            "proposal_type": self.proposal_type,
            "description": self.description,
            "value": self.value,
            "votes": dict(self.votes),
            "resolved": self.resolved,
            "outcome": self.outcome,
            "sub_sim_evidence": self.sub_sim_evidence,
        }


@dataclass
class ColonyState:
    """The full colony state at any point in time."""

    year: int = 0
    colonists: list[Colonist] = field(default_factory=list)
    resources: dict[str, float] = field(default_factory=lambda: {
        "food": 5000.0,
        "water": 8000.0,
        "power": 3000.0,
        "materials": 2000.0,
        "oxygen": 6000.0,
    })
    governance: dict = field(default_factory=lambda: {
        "type": "consensus",
        "leader_id": None,
        "constitution": [],
        "amendments": [],
    })
    proposals: list[GovernanceProposal] = field(default_factory=list)
    events_log: list[dict] = field(default_factory=list)
    sub_sim_log: list[dict] = field(default_factory=list)
    morale: float = 0.7
    terraforming_progress: float = 0.0
    collapsed: bool = False
    collapse_reason: str | None = None

    def alive_colonists(self) -> list[Colonist]:
        """Return only living colonists."""
        return [c for c in self.colonists if c.alive]

    def dead_colonists(self) -> list[Colonist]:
        """Return archived (dead) colonists."""
        return [c for c in self.colonists if not c.alive]

    def to_dict(self) -> dict:
        """Full state serialization."""
        return {
            "year": self.year,
            "colonists": [c.to_dict() for c in self.colonists],
            "resources": dict(self.resources),
            "governance": dict(self.governance),
            "proposals": [p.to_dict() for p in self.proposals],
            "events_log": self.events_log,
            "sub_sim_log": self.sub_sim_log,
            "morale": round(self.morale, 4),
            "terraforming_progress": round(self.terraforming_progress, 6),
            "collapsed": self.collapsed,
            "collapse_reason": self.collapse_reason,
            "alive_count": len(self.alive_colonists()),
            "dead_count": len(self.dead_colonists()),
        }


# ---------------------------------------------------------------------------
# Colony genesis — create 10 colonists
# ---------------------------------------------------------------------------

COLONIST_TEMPLATES = [
    ("kael-terraform", "Kael", "earth", {"terraforming": 0.9, "resolve": 0.8}),
    ("lyra-hydroponics", "Lyra", "water", {"hydroponics": 0.9, "empathy": 0.8}),
    ("oren-mediator", "Oren", "air", {"mediation": 0.9, "improvisation": 0.7}),
    ("vex-coder", "Vex", "fire", {"coding": 0.9, "paranoia": 0.6}),
    ("sable-mystic", "Sable", "water", {"prayer": 0.8, "faith": 0.9}),
    ("thresh-survivor", "Thresh", "fire", {"sabotage": 0.7, "hoarding": 0.8}),
    ("nova-explorer", "Nova", "air", {"improvisation": 0.9, "resolve": 0.7}),
    ("petra-builder", "Petra", "earth", {"terraforming": 0.7, "coding": 0.6}),
    ("zeph-diplomat", "Zeph", "air", {"mediation": 0.8, "empathy": 0.9}),
    ("mira-scientist", "Mira", "water", {"hydroponics": 0.7, "coding": 0.8}),
]


def create_colonists(rng: random.Random) -> list[Colonist]:
    """Generate the 10 founding colonists with randomized base stats."""
    colonists = []
    for col_id, name, element, boosts in COLONIST_TEMPLATES:
        # Base stats: random 0.2-0.6, boosted by template
        stats = {}
        for stat in STAT_NAMES:
            base = rng.uniform(0.2, 0.6)
            if stat in boosts:
                base = max(base, boosts[stat])
            stats[stat] = round(min(1.0, base), 3)

        # Base skills: random 0.1-0.4, boosted by template
        skills = {}
        for skill in SKILL_NAMES:
            base = rng.uniform(0.1, 0.4)
            if skill in boosts:
                base = max(base, boosts[skill])
            skills[skill] = round(min(1.0, base), 3)

        colonists.append(Colonist(
            id=col_id, name=name, element=element,
            stats=stats, skills=skills,
        ))

    # Initialize relationship matrix (small random noise)
    for c in colonists:
        for other in colonists:
            if c.id != other.id:
                # Same element = slight affinity
                base = 0.1 if c.element == other.element else 0.0
                noise = rng.uniform(-0.15, 0.15)
                c.relationships[other.id] = round(
                    max(MIN_RELATIONSHIP, min(MAX_RELATIONSHIP, base + noise)), 3
                )

    return colonists


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------

def generate_event(year: int, rng: random.Random) -> dict:
    """Generate an environmental event for the given year."""
    if year <= 20:
        pool = EARLY_EVENTS
    elif year <= 60:
        pool = MID_EVENTS
    else:
        pool = LATE_EVENTS

    event_type = rng.choice(pool)
    severity = round(rng.uniform(0.1, 1.0), 3)

    descriptions = {
        "dust_storm": f"A massive dust storm engulfs the colony (severity {severity:.1f})",
        "equipment_failure": f"Critical life support equipment fails (severity {severity:.1f})",
        "water_strike": "Subsurface water ice deposit discovered!",
        "solar_flare": f"Intense solar flare bombards the colony (severity {severity:.1f})",
        "meteor_impact": f"Meteor strike near habitat (severity {severity:.1f})",
        "supply_drop": "Resupply ship from Earth arrives with materials",
        "habitat_leak": f"Pressure seal breach in habitat module (severity {severity:.1f})",
        "resource_strike": "New mineral vein discovered in Martian regolith",
        "earth_contact": "Communication window with Earth opens — news from home",
        "epidemic": f"Unknown pathogen spreads through the colony (severity {severity:.1f})",
        "crop_blight": f"Hydroponics crop failure — fungal contamination (severity {severity:.1f})",
        "power_surge": f"Solar array overload damages power grid (severity {severity:.1f})",
        "diplomatic_crisis": "Tensions between colonist factions reach breaking point",
        "baby_born": "A child is born in the colony — the first Martian generation",
        "alien_signal": "Anomalous repeating signal detected from Olympus Mons",
        "colony_schism": "A faction of colonists demands independence",
        "philosophical_crisis": "Existential debate: what is our purpose on Mars?",
        "technological_breakthrough": "Major advancement in terraforming technology",
        "meta_awareness": "A colonist begins questioning the nature of their reality",
        "terraforming_milestone": f"Atmospheric CO2 levels reach new threshold ({severity*10:.1f}%)",
        "generational_shift": "The original colonists pass leadership to the next generation",
    }

    return {
        "year": year,
        "type": event_type,
        "severity": severity,
        "description": descriptions.get(event_type, f"Unknown event: {event_type}"),
    }


# ---------------------------------------------------------------------------
# Colonist decision engine (LisPy-backed)
# ---------------------------------------------------------------------------

def make_colonist_decision(
    colonist: Colonist,
    event: dict,
    state: ColonyState,
    vm: LispyVM,
    rng: random.Random,
) -> dict:
    """Generate a colonist's response to the current year's event.

    Returns a decision dict with action, optional proposal, diary entry,
    and any sub-sim results.
    """
    if not colonist.alive:
        return {"action": "archived", "colonist_id": colonist.id}

    # Build context for the colonist's LisPy decision program
    alive_ids = [c.id for c in state.alive_colonists() if c.id != colonist.id]
    rel_summary = {cid: colonist.relationships.get(cid, 0) for cid in alive_ids[:5]}

    # Deterministic decision based on stats + event
    action = _choose_action(colonist, event, state, rng)
    proposal = None
    sub_sim_result = None

    # Some actions trigger governance proposals
    if action in ("propose_leader", "propose_law", "request_exile", "propose_amendment"):
        proposal = _create_proposal(colonist, action, event, state, rng)

    # Complex decisions may trigger sub-simulations
    if _should_sub_sim(colonist, event, state, rng):
        sub_sim_result = _run_colonist_sub_sim(colonist, event, state, vm, rng)
        colonist.sub_sims_run += 1

    # Generate diary entry
    diary_entry = _make_diary_entry(colonist, event, action, state, rng)
    colonist.diary.append(diary_entry)

    return {
        "colonist_id": colonist.id,
        "action": action,
        "proposal": proposal.to_dict() if proposal else None,
        "sub_sim": sub_sim_result,
        "diary": diary_entry,
    }


def _choose_action(
    colonist: Colonist, event: dict, state: ColonyState, rng: random.Random
) -> str:
    """Choose an action based on colonist personality and event."""
    severity = event.get("severity", 0.5)
    event_type = event["type"]

    # Personality-weighted action selection
    if event_type in ("dust_storm", "meteor_impact", "solar_flare", "habitat_leak"):
        # Crisis response — weighted by resolve and paranoia
        if colonist.stats["paranoia"] > 0.7 and severity > 0.6:
            return "hoard_resources"
        if colonist.stats["resolve"] > 0.6:
            return "lead_repair"
        return "assist_repair"

    if event_type in ("diplomatic_crisis", "colony_schism"):
        if colonist.stats["empathy"] > 0.6:
            return "mediate"
        if colonist.stats["resolve"] > 0.7:
            return "propose_leader"
        return "observe"

    if event_type == "philosophical_crisis":
        if colonist.stats["faith"] > 0.7:
            return "lead_ceremony"
        if colonist.stats["improvisation"] > 0.6:
            return "propose_amendment"
        return "reflect"

    if event_type == "meta_awareness":
        if colonist.stats["paranoia"] > 0.5 or colonist.stats["improvisation"] > 0.7:
            colonist.meta_aware = True
            return "question_reality"
        return "dismiss_notion"

    if event_type in ("water_strike", "resource_strike", "supply_drop"):
        if colonist.stats["hoarding"] > 0.6:
            return "claim_resources"
        return "share_resources"

    if event_type == "alien_signal":
        if colonist.stats["faith"] > 0.6:
            return "interpret_signal"
        if colonist.skills["coding"] > 0.5:
            return "decode_signal"
        return "observe"

    if event_type == "baby_born":
        return "celebrate"

    if event_type == "epidemic":
        if colonist.skills["hydroponics"] > 0.5 or colonist.stats["empathy"] > 0.6:
            return "tend_sick"
        return "quarantine"

    # Default: work on primary skill
    best_skill = max(colonist.skills, key=colonist.skills.get)
    return f"work_{best_skill}"


def _should_sub_sim(
    colonist: Colonist, event: dict, state: ColonyState, rng: random.Random
) -> bool:
    """Decide if a colonist runs a sub-simulation for this event."""
    # Higher improvisation and coding skills = more likely to model
    model_propensity = (
        colonist.stats["improvisation"] * 0.4
        + colonist.skills["coding"] * 0.4
        + colonist.stats["paranoia"] * 0.2
    )
    # Critical events increase sub-sim likelihood
    severity = event.get("severity", 0.5)
    threshold = 0.7 - (severity * 0.2)
    return model_propensity > threshold and rng.random() < 0.3


def _run_colonist_sub_sim(
    colonist: Colonist,
    event: dict,
    state: ColonyState,
    vm: LispyVM,
    rng: random.Random,
) -> dict:
    """Run a sub-simulation for a colonist's decision.

    Generates LisPy programs that may spawn nested sub-sims (depth 2-3).
    Meta-aware colonists explore deeper recursion; others stay at depth 1.
    """
    alive_count = len(state.alive_colonists())
    food = state.resources["food"]
    water = state.resources["water"]
    power = state.resources["power"]
    severity = event.get("severity", 0.5)

    # Meta-aware or high-coding colonists generate recursive programs
    if colonist.meta_aware or colonist.skills.get("coding", 0) > 0.7:
        program = _build_recursive_sub_sim(colonist, event, state, rng)
    else:
        program = _build_basic_sub_sim(food, water, power, alive_count, severity)

    vm.steps = 0
    vm.sub_sim_log = []
    try:
        result = vm.eval_str(program)
        return {
            "colonist_id": colonist.id,
            "depth_requested": 2 if colonist.meta_aware else 1,
            "program": program,
            "result": result,
            "sub_sim_log": list(vm.sub_sim_log),
            "status": "completed",
        }
    except LispyError as e:
        return {
            "colonist_id": colonist.id,
            "depth_requested": 2 if colonist.meta_aware else 1,
            "program": program,
            "error": str(e),
            "sub_sim_log": list(vm.sub_sim_log),
            "status": "error",
        }


def _build_basic_sub_sim(
    food: float, water: float, power: float,
    alive_count: int, severity: float,
) -> str:
    """Build a flat resource-projection sub-sim (depth 1)."""
    return (
        f'(let* ((food {food:.0f}) (water {water:.0f}) (power {power:.0f}) '
        f'(pop {alive_count}) (severity {severity:.2f}) '
        f'(consumption (* pop 1.8)) '
        f'(food-after (- food consumption)) '
        f'(survival-years (if (> consumption 0) (/ food-after consumption) 999)) '
        f'(risk (if (< survival-years 5) "critical" '
        f'  (if (< survival-years 20) "moderate" "safe")))) '
        f'(list risk survival-years food-after))'
    )


def _build_recursive_sub_sim(
    colonist: Colonist, event: dict, state: ColonyState, rng: random.Random,
) -> str:
    """Build a recursive sub-sim that spawns depth-2 (and possibly depth-3) children.

    The colonist models two governance scenarios at depth 2:
      - cooperative: share resources equally
      - competitive: hoard by strongest
    If meta-aware, a depth-3 sub-sim explores what happens when the model
    becomes aware it is being modeled.
    """
    alive_count = len(state.alive_colonists())
    food = state.resources["food"]
    severity = event.get("severity", 0.5)
    per_cap = food / max(1, alive_count)

    # Depth-2: compare cooperative vs competitive resource allocation
    depth3_clause = ""
    if colonist.meta_aware:
        # Meta-aware colonists ask: what if the model knows it's modeled?
        depth3_clause = (
            '(define meta-result '
            '  (sub-sim "meta-recursion" '
            '    (let* ((signal (if (> 0.5 0.3) "aware" "unaware")) '
            '           (choice (if (= signal "aware") "cooperate" "defect"))) '
            '      (list "depth-3-insight" choice signal)))) '
        )

    program = (
        f'(let* ((food {food:.0f}) (pop {alive_count}) '
        f'(severity {severity:.2f}) (per-cap {per_cap:.1f})) '
        f'(define coop-result '
        f'  (sub-sim "cooperative-model" '
        f'    (let* ((shared (/ {food:.0f} {max(1, alive_count)})) '
        f'           (morale (if (> shared 200) 0.8 0.4)) '
        f'           (survive (> shared 100))) '
        f'      {depth3_clause}'
        f'      (list "cooperative" morale survive'
        f'        {" meta-result" if colonist.meta_aware else ""})))) '
        f'(define comp-result '
        f'  (sub-sim "competitive-model" '
        f'    (let* ((top-share (* {per_cap:.1f} 1.5)) '
        f'           (bottom-share (* {per_cap:.1f} 0.5)) '
        f'           (morale 0.3) '
        f'           (survive (> bottom-share 80))) '
        f'      (list "competitive" morale survive)))) '
        f'(list coop-result comp-result '
        f'  (if (> (nth coop-result 1) (nth comp-result 1)) '
        f'    "recommend-cooperation" "recommend-competition")))'
    )
    return program


def _create_proposal(
    colonist: Colonist,
    action: str,
    event: dict,
    state: ColonyState,
    rng: random.Random,
) -> GovernanceProposal:
    """Create a governance proposal based on the colonist's action."""
    proposal_map = {
        "propose_leader": "leadership_election",
        "propose_law": "law_enactment",
        "request_exile": "exile_vote",
        "propose_amendment": "constitutional_amendment",
    }
    ptype = proposal_map.get(action, "resource_allocation")

    descriptions = {
        "leadership_election": f"{colonist.name} proposes a new colony leader in response to {event['type']}",
        "law_enactment": f"{colonist.name} proposes a new law: equitable resource sharing",
        "exile_vote": f"{colonist.name} calls for a vote on exile of a disruptive colonist",
        "constitutional_amendment": f"{colonist.name} proposes amending the colony constitution based on philosophical crisis",
        "resource_allocation": f"{colonist.name} proposes new resource allocation after {event['type']}",
    }

    # Deterministic ID
    raw = f"{state.year}-{colonist.id}-{ptype}"
    prop_id = hashlib.md5(raw.encode()).hexdigest()[:8]

    colonist.proposals_made += 1

    return GovernanceProposal(
        id=prop_id,
        year=state.year,
        proposer_id=colonist.id,
        proposal_type=ptype,
        description=descriptions.get(ptype, f"Proposal by {colonist.name}"),
        value={"event_context": event["type"], "severity": event.get("severity", 0.5)},
    )


def _make_diary_entry(
    colonist: Colonist,
    event: dict,
    action: str,
    state: ColonyState,
    rng: random.Random,
) -> dict:
    """Create a diary entry for this year."""
    moods = {
        "lead_repair": "determined",
        "assist_repair": "dutiful",
        "hoard_resources": "anxious",
        "mediate": "compassionate",
        "propose_leader": "ambitious",
        "reflect": "contemplative",
        "question_reality": "unsettled",
        "celebrate": "joyful",
        "tend_sick": "worried",
        "share_resources": "generous",
        "claim_resources": "protective",
        "observe": "watchful",
        "lead_ceremony": "reverent",
        "interpret_signal": "mystified",
        "decode_signal": "focused",
        "quarantine": "cautious",
        "dismiss_notion": "pragmatic",
        "propose_amendment": "visionary",
        "propose_law": "principled",
        "request_exile": "conflicted",
    }
    mood = moods.get(action, "neutral")

    return {
        "year": state.year,
        "event": event["type"],
        "action": action,
        "mood": mood,
        "morale": round(state.morale, 3),
        "meta_aware": colonist.meta_aware,
        "alive_neighbors": len(state.alive_colonists()) - 1,
    }


# ---------------------------------------------------------------------------
# Governance resolution
# ---------------------------------------------------------------------------

def resolve_proposals(state: ColonyState, rng: random.Random) -> list[dict]:
    """Resolve all pending governance proposals via weighted voting."""
    resolved = []
    alive = state.alive_colonists()

    for proposal in state.proposals:
        if proposal.resolved:
            continue

        # Each alive colonist votes
        total_for = 0.0
        total_against = 0.0
        for colonist in alive:
            if colonist.id == proposal.proposer_id:
                # Proposer always votes for
                position = "for"
                weight = colonist.governance_weight
            else:
                rel = colonist.relationships.get(proposal.proposer_id, 0)
                # Positive relationship → vote for; negative → against
                if rel > 0.1:
                    position = "for"
                elif rel < -0.1:
                    position = "against"
                else:
                    position = "for" if rng.random() < 0.5 else "against"
                weight = colonist.governance_weight

            # Clamp weight
            weight = max(0.0, min(MAX_GOVERNANCE_WEIGHT, weight))
            proposal.votes[colonist.id] = {"position": position, "weight": weight}
            colonist.votes_cast += 1

            if position == "for":
                total_for += weight
            else:
                total_against += weight

        # Simple majority
        proposal.resolved = True
        proposal.outcome = "passed" if total_for > total_against else "failed"
        resolved.append(proposal.to_dict())

    return resolved


# ---------------------------------------------------------------------------
# Year tick — apply one year of simulation
# ---------------------------------------------------------------------------

def apply_event_effects(event: dict, state: ColonyState, rng: random.Random) -> None:
    """Apply environmental event effects to colony state."""
    severity = event.get("severity", 0.5)
    event_type = event["type"]

    if event_type in ("dust_storm", "solar_flare"):
        state.resources["power"] *= (1.0 - severity * 0.3)
        state.morale -= severity * 0.1

    elif event_type in ("equipment_failure", "habitat_leak"):
        state.resources["oxygen"] *= (1.0 - severity * 0.2)
        state.resources["materials"] -= severity * 200
        # Risk of death
        if severity > 0.7:
            victim = _random_alive_colonist(state, rng)
            if victim:
                _kill_colonist(victim, state.year, event_type)

    elif event_type == "meteor_impact":
        state.resources["materials"] -= severity * 500
        state.morale -= severity * 0.15
        if severity > 0.8:
            victim = _random_alive_colonist(state, rng)
            if victim:
                _kill_colonist(victim, state.year, "meteor_impact")

    elif event_type in ("water_strike", "resource_strike"):
        state.resources["water"] += 2000
        state.resources["materials"] += 1000
        state.morale += 0.1

    elif event_type == "supply_drop":
        state.resources["food"] += 3000
        state.resources["materials"] += 1500
        state.morale += 0.15

    elif event_type == "epidemic":
        alive = state.alive_colonists()
        death_count = max(0, int(severity * 0.3 * len(alive)))
        for _ in range(death_count):
            victim = _random_alive_colonist(state, rng)
            if victim:
                _kill_colonist(victim, state.year, "epidemic")
        state.morale -= severity * 0.2

    elif event_type == "crop_blight":
        state.resources["food"] *= (1.0 - severity * 0.4)

    elif event_type == "power_surge":
        state.resources["power"] *= (1.0 - severity * 0.25)

    elif event_type == "technological_breakthrough":
        state.terraforming_progress += 0.05
        state.morale += 0.1

    elif event_type == "baby_born":
        state.morale += 0.15

    elif event_type == "terraforming_milestone":
        state.terraforming_progress += severity * 0.03

    elif event_type in ("diplomatic_crisis", "colony_schism", "philosophical_crisis"):
        state.morale -= severity * 0.12

    elif event_type == "alien_signal":
        state.morale += 0.05  # excitement

    elif event_type == "meta_awareness":
        state.morale -= 0.05  # existential dread

    # Ensure physical bounds
    for key in state.resources:
        state.resources[key] = max(0.0, min(MAX_RESOURCES, state.resources[key]))
    state.morale = max(MIN_MORALE, min(MAX_MORALE, state.morale))


def _random_alive_colonist(state: ColonyState, rng: random.Random) -> Colonist | None:
    """Pick a random alive colonist."""
    alive = state.alive_colonists()
    return rng.choice(alive) if alive else None


def _kill_colonist(colonist: Colonist, year: int, cause: str) -> None:
    """Kill a colonist — legacy not delete (Amendment X)."""
    colonist.alive = False
    colonist.year_died = year
    colonist.death_cause = cause
    colonist.soul_archived = True


def update_relationships(state: ColonyState, decisions: list[dict], rng: random.Random) -> None:
    """Update relationship matrix based on year's decisions."""
    alive = state.alive_colonists()
    action_affinity = {
        "lead_repair": 0.05,
        "assist_repair": 0.03,
        "mediate": 0.08,
        "share_resources": 0.06,
        "celebrate": 0.04,
        "tend_sick": 0.07,
    }
    action_friction = {
        "hoard_resources": -0.06,
        "claim_resources": -0.04,
        "request_exile": -0.1,
        "quarantine": -0.02,
    }

    for decision in decisions:
        cid = decision["colonist_id"]
        action = decision["action"]
        colonist = next((c for c in alive if c.id == cid), None)
        if not colonist:
            continue

        delta = action_affinity.get(action, 0) + action_friction.get(action, 0)
        if delta != 0:
            for other in alive:
                if other.id != cid:
                    old = colonist.relationships.get(other.id, 0)
                    noise = rng.uniform(-0.02, 0.02)
                    new_val = old + delta + noise
                    colonist.relationships[other.id] = round(
                        max(MIN_RELATIONSHIP, min(MAX_RELATIONSHIP, new_val)), 3
                    )

    # Decay: all relationships regress toward 0 slightly
    for c in alive:
        for other_id in list(c.relationships):
            c.relationships[other_id] *= 0.98


def update_governance_weights(state: ColonyState) -> None:
    """Update governance weights based on contributions and relationships."""
    alive = state.alive_colonists()
    for colonist in alive:
        # Weight grows with proposals and votes, decays without
        activity = (colonist.proposals_made * 0.1 + colonist.votes_cast * 0.02)
        base = 1.0 + activity
        # Clamp to prevent runaway
        colonist.governance_weight = round(
            max(0.1, min(MAX_GOVERNANCE_WEIGHT, base)), 3
        )


def apply_passed_proposals(state: ColonyState) -> list[dict]:
    """Apply effects of passed governance proposals to the colony.

    Constitutional amendments persist in governance["constitution"].
    Resource allocations and laws modify colony behavior.
    Returns list of effects applied.
    """
    effects: list[dict] = []
    for proposal in state.proposals:
        if not proposal.resolved or proposal.outcome != "passed":
            continue
        # Skip already-applied proposals (check by ID in constitution)
        applied_ids = {
            a.get("proposal_id") for a in state.governance.get("amendments", [])
        }
        if proposal.id in applied_ids:
            continue

        effect = _apply_single_proposal(proposal, state)
        if effect:
            effects.append(effect)

    return effects


def _apply_single_proposal(proposal: GovernanceProposal, state: ColonyState) -> dict | None:
    """Apply a single passed proposal. Returns effect dict or None."""
    ptype = proposal.proposal_type

    if ptype == "constitutional_amendment":
        amendment = {
            "proposal_id": proposal.id,
            "year_enacted": proposal.year,
            "proposer": proposal.proposer_id,
            "text": proposal.description,
            "active": True,
        }
        state.governance["amendments"].append(amendment)
        state.governance["constitution"].append(proposal.description)
        return {"type": "amendment_enacted", "proposal_id": proposal.id,
                "year": proposal.year, "text": proposal.description}

    if ptype == "leadership_election":
        state.governance["leader_id"] = proposal.proposer_id
        state.governance["type"] = "elected_leader"
        return {"type": "leader_elected", "leader": proposal.proposer_id,
                "year": proposal.year}

    if ptype == "resource_allocation":
        # Equitable sharing boosts morale
        state.morale = min(MAX_MORALE, state.morale + 0.05)
        return {"type": "resource_policy", "morale_boost": 0.05,
                "year": proposal.year}

    if ptype == "exile_vote":
        # Exile the least-liked alive colonist (excluding proposer)
        alive = [c for c in state.alive_colonists() if c.id != proposal.proposer_id]
        if alive:
            target = min(alive, key=lambda c: sum(c.relationships.values()))
            _kill_colonist(target, proposal.year, "exiled")
            return {"type": "exile", "exiled": target.id, "year": proposal.year}

    if ptype == "law_enactment":
        state.governance["constitution"].append(proposal.description)
        return {"type": "law_enacted", "text": proposal.description,
                "year": proposal.year}

    return None


# ---------------------------------------------------------------------------
# Colonist skill growth
# ---------------------------------------------------------------------------

# Action → skill that improves
_ACTION_SKILL_MAP: dict[str, str] = {
    "lead_repair": "terraforming",
    "assist_repair": "terraforming",
    "work_terraforming": "terraforming",
    "work_hydroponics": "hydroponics",
    "mediate": "mediation",
    "work_coding": "coding",
    "decode_signal": "coding",
    "lead_ceremony": "prayer",
    "work_prayer": "prayer",
    "tend_sick": "hydroponics",
    "hoard_resources": "sabotage",
    "work_sabotage": "sabotage",
}

SKILL_GROWTH_RATE = 0.015
SKILL_DECAY_RATE = 0.005


def update_skills(state: ColonyState, decisions: list[dict]) -> None:
    """Update colonist skills based on actions taken this year.

    Skills used this year grow; unused skills decay slightly.
    Bounds: [0.0, 1.0].
    """
    # Collect which colonist used which skill
    used: dict[str, str | None] = {}
    for decision in decisions:
        cid = decision["colonist_id"]
        action = decision["action"]
        used[cid] = _ACTION_SKILL_MAP.get(action)

    for colonist in state.alive_colonists():
        trained_skill = used.get(colonist.id)
        for skill_name in SKILL_NAMES:
            old = colonist.skills.get(skill_name, 0.1)
            if skill_name == trained_skill:
                new_val = old + SKILL_GROWTH_RATE
            else:
                new_val = old - SKILL_DECAY_RATE
            colonist.skills[skill_name] = round(max(0.0, min(1.0, new_val)), 4)


def consume_resources(state: ColonyState) -> None:
    """Annual resource consumption."""
    alive_count = len(state.alive_colonists())
    if alive_count == 0:
        return

    # Per-colonist annual consumption (Mars year ≈ 687 Earth days)
    state.resources["food"] -= alive_count * 1.8 * 687
    state.resources["water"] -= alive_count * 0.5 * 687  # mostly recycled
    state.resources["power"] -= alive_count * 3.0 * 687 * 0.01  # scaled
    state.resources["oxygen"] -= alive_count * 0.84 * 687 * 0.1  # mostly recycled

    # Starvation check
    if state.resources["food"] < 0:
        state.resources["food"] = 0
        state.morale -= 0.2

    # Ensure bounds
    for key in state.resources:
        state.resources[key] = max(0.0, min(MAX_RESOURCES, state.resources[key]))
    state.morale = max(MIN_MORALE, min(MAX_MORALE, state.morale))


def natural_production(state: ColonyState) -> None:
    """Annual resource production from colonist labor."""
    alive = state.alive_colonists()
    if not alive:
        return

    # Terraforming output
    terraform_skill = sum(c.skills["terraforming"] for c in alive)
    state.terraforming_progress += terraform_skill * 0.002
    state.terraforming_progress = min(1.0, state.terraforming_progress)

    # Hydroponics food production
    hydro_skill = sum(c.skills["hydroponics"] for c in alive)
    state.resources["food"] += hydro_skill * 500

    # Power maintenance
    coding_skill = sum(c.skills["coding"] for c in alive)
    state.resources["power"] += coding_skill * 200

    # General maintenance
    state.resources["materials"] += len(alive) * 50


def check_collapse(state: ColonyState) -> bool:
    """Check if the colony has collapsed."""
    alive = state.alive_colonists()
    if len(alive) == 0:
        state.collapsed = True
        state.collapse_reason = "all_dead"
        return True
    if len(alive) < 3 and state.morale < 0.1:
        state.collapsed = True
        state.collapse_reason = "critical_underpopulation"
        return True
    if state.resources["food"] <= 0 and state.resources["water"] <= 0:
        state.collapsed = True
        state.collapse_reason = "total_resource_depletion"
        return True
    return False


def clamp_morale(state: ColonyState) -> None:
    """Enforce morale bounds as a hard invariant."""
    state.morale = max(MIN_MORALE, min(MAX_MORALE, state.morale))


# ---------------------------------------------------------------------------
# Governance classification (scoring-based, overlap-safe)
# ---------------------------------------------------------------------------

def classify_governance(state: ColonyState) -> str:
    """Classify emergent governance form from observable colony signals.

    Uses a scoring model rather than strict precedence to handle overlapping
    conditions. Returns the highest-scoring form from GOVERNANCE_FORMS.
    """
    alive = state.alive_colonists()
    if not alive:
        return "anarchy"

    alive_count = len(alive)
    leader_id = state.governance.get("leader_id")
    constitution = state.governance.get("constitution", [])
    amendments = state.governance.get("amendments", [])
    weights = [c.governance_weight for c in alive]
    max_weight = max(weights)
    min_weight = min(weights)
    weight_spread = max_weight - min_weight
    proposals = state.proposals

    # Recent proposals (last 20 years)
    recent_year = max(1, state.year - 20)
    recent_proposals = [p for p in proposals if p.year >= recent_year]
    recent_elections = [
        p for p in recent_proposals
        if p.proposal_type == "leadership_election" and p.outcome == "passed"
    ]

    scores: dict[str, float] = {form: 0.0 for form in GOVERNANCE_FORMS}

    # --- Anarchy: few proposals, low structure ---
    if len(proposals) < 3:
        scores["anarchy"] += 3.0
    if not leader_id and not constitution:
        scores["anarchy"] += 1.0

    # --- Consensus: moderate proposals, high pass rate, no dominant leader ---
    if recent_proposals:
        pass_rate = sum(1 for p in recent_proposals if p.outcome == "passed") / len(recent_proposals)
        scores["consensus"] += pass_rate * 2.0
    if weight_spread < 1.5 and alive_count >= 3:
        scores["consensus"] += 1.0

    # --- Council: multiple high-weight colonists, no single dominant ---
    high_weight = [c for c in alive if c.governance_weight > 2.0]
    if len(high_weight) >= 3 and weight_spread < 3.0:
        scores["council"] += 2.5
    if len(high_weight) >= 2 and not leader_id:
        scores["council"] += 1.0

    # --- Elected democracy: current leader backed by recent election + constitution ---
    if leader_id and recent_elections:
        latest_election = max(recent_elections, key=lambda p: p.year)
        if latest_election.proposer_id == leader_id or leader_id in [
            p.proposer_id for p in recent_elections
        ]:
            scores["elected_democracy"] += 2.0
        if len(constitution) >= 2:
            scores["elected_democracy"] += 1.5
        if len(amendments) >= 1:
            scores["elected_democracy"] += 0.5

    # --- Autocracy: single dominant leader, large weight gap, weak institutions ---
    if leader_id:
        leader = next((c for c in alive if c.id == leader_id), None)
        if leader:
            leader_dominance = leader.governance_weight / max(1.0, sum(weights) / alive_count)
            if leader_dominance > 1.8:
                scores["autocracy"] += 2.5
            if len(constitution) < 2:
                scores["autocracy"] += 1.0
            if not recent_elections:
                scores["autocracy"] += 1.0

    # --- Theocracy: faith-dominant high-weight colonists ---
    faith_leaders = [c for c in alive if c.stats.get("faith", 0) > 0.7 and c.governance_weight > 2.0]
    if len(faith_leaders) >= max(2, alive_count * 0.3):
        scores["theocracy"] += 3.5
        # Bonus if faith leaders ARE the council — theocracy outranks generic council
        if len(faith_leaders) >= len(high_weight) * 0.6:
            scores["theocracy"] += 1.0
    elif len(faith_leaders) >= 1 and leader_id:
        leader = next((c for c in alive if c.id == leader_id), None)
        if leader and leader.stats.get("faith", 0) > 0.7:
            scores["theocracy"] += 2.0

    # --- Technocracy: coding/tech-dominant high-weight colonists ---
    tech_leaders = [c for c in alive if c.skills.get("coding", 0) > 0.7 and c.governance_weight > 2.0]
    if len(tech_leaders) >= max(2, alive_count * 0.3):
        scores["technocracy"] += 3.5
        if len(tech_leaders) >= len(high_weight) * 0.6:
            scores["technocracy"] += 1.0
    elif len(tech_leaders) >= 1 and leader_id:
        leader = next((c for c in alive if c.id == leader_id), None)
        if leader and leader.skills.get("coding", 0) > 0.7:
            scores["technocracy"] += 2.0

    # --- Commune: very equal weights + high participation ---
    if weight_spread < 0.5 and alive_count >= 3:
        scores["commune"] += 2.5
    participation = sum(1 for c in alive if c.proposals_made > 0 or c.votes_cast > 0)
    if participation >= alive_count * 0.8:
        scores["commune"] += 1.0

    best_form = max(scores, key=scores.get)
    return best_form


def compute_value_convergence(state: ColonyState) -> dict:
    """Compute how colonist values (stats) converge or diverge.

    Returns per-stat standard deviation across alive colonists,
    plus an aggregate convergence score (lower = more converged).
    """
    alive = state.alive_colonists()
    if len(alive) < 2:
        return {"stats_std": {}, "convergence_score": 0.0, "sample_size": len(alive)}

    stats_std: dict[str, float] = {}
    for stat in STAT_NAMES:
        values = [c.stats.get(stat, 0.5) for c in alive]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stats_std[stat] = round(math.sqrt(variance), 4)

    aggregate = sum(stats_std.values()) / len(stats_std) if stats_std else 0.0

    return {
        "stats_std": stats_std,
        "convergence_score": round(aggregate, 4),
        "sample_size": len(alive),
    }


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

class Mars100Simulation:
    """The Mars-100 recursive colony simulation.

    Runs 100 Martian years with 10 colonists. Each year:
    1. Generate environmental event
    2. Apply event effects
    3. Each colonist decides (some spawn sub-sims)
    4. Governance proposals raised and resolved
    5. Resources consumed and produced
    6. Relationships updated
    7. State saved as yearly delta
    """

    def __init__(self, seed: int = 42, max_years: int = 100) -> None:
        self.seed = seed
        self.max_years = max_years
        self.rng = random.Random(seed)
        self.vm = LispyVM(max_depth=3, max_steps=50_000, rng_seed=seed)
        self.state = ColonyState()
        self.state.colonists = create_colonists(self.rng)
        self.yearly_deltas: list[dict] = []
        self.meta_insights: list[dict] = []

    def run(self, callback: Any = None) -> dict:
        """Run the full simulation. Returns final state + all deltas."""
        for year in range(1, self.max_years + 1):
            self.state.year = year
            delta = self._tick_year(year)
            self.yearly_deltas.append(delta)

            if callback:
                callback(year, self.state, delta)

            if self.state.collapsed:
                break

        return self._build_results()

    def run_years(self, n_years: int, callback: Any = None) -> dict:
        """Run N years (for testing with small counts)."""
        start = self.state.year + 1
        end = min(start + n_years, self.max_years + 1)
        for year in range(start, end):
            self.state.year = year
            delta = self._tick_year(year)
            self.yearly_deltas.append(delta)
            if callback:
                callback(year, self.state, delta)
            if self.state.collapsed:
                break
        return self._build_results()

    def _tick_year(self, year: int) -> dict:
        """Advance one Martian year. Returns a delta dict."""
        # 1. Generate event
        event = generate_event(year, self.rng)
        self.state.events_log.append(event)

        # 2. Apply event effects
        apply_event_effects(event, self.state, self.rng)

        # 3. Colonist decisions
        decisions = []
        for colonist in self.state.alive_colonists():
            decision = make_colonist_decision(
                colonist, event, self.state, self.vm, self.rng
            )
            decisions.append(decision)

            # Collect proposals
            if decision.get("proposal"):
                prop_data = decision["proposal"]
                proposal = GovernanceProposal(
                    id=prop_data["id"],
                    year=prop_data["year"],
                    proposer_id=prop_data["proposer_id"],
                    proposal_type=prop_data["proposal_type"],
                    description=prop_data["description"],
                    value=prop_data["value"],
                )
                if decision.get("sub_sim"):
                    proposal.sub_sim_evidence.append(decision["sub_sim"])
                self.state.proposals.append(proposal)

            # Collect sub-sim logs
            if decision.get("sub_sim"):
                self.state.sub_sim_log.append(decision["sub_sim"])

        # 4. Resolve governance
        resolved = resolve_proposals(self.state, self.rng)

        # 4b. Apply passed proposals (active governance)
        governance_effects = apply_passed_proposals(self.state)

        # 5. Resource cycle
        consume_resources(self.state)
        natural_production(self.state)

        # 6. Update relationships, governance weights, and skills
        update_relationships(self.state, decisions, self.rng)
        update_governance_weights(self.state)
        update_skills(self.state, decisions)

        # 7. Check for meta-insights (depth-2+ sub-sims)
        for sub_log in self.state.sub_sim_log:
            if isinstance(sub_log, dict):
                nested = sub_log.get("sub_sim_log", [])
                for nested_entry in (nested if isinstance(nested, list) else []):
                    depth_val = nested_entry.get("depth", 0) if isinstance(nested_entry, dict) else 0
                    if depth_val >= 2:
                        self.meta_insights.append({
                            "year": year,
                            "source": sub_log.get("colonist_id", "unknown"),
                            "depth": depth_val,
                            "label": nested_entry.get("label", ""),
                            "result": nested_entry.get("result"),
                        })
                        # Scan depth-3 children
                        for d3 in (nested_entry.get("sub_sims", []) or []):
                            if isinstance(d3, dict) and d3.get("depth", 0) >= 3:
                                self.meta_insights.append({
                                    "year": year,
                                    "source": sub_log.get("colonist_id", "unknown"),
                                    "depth": d3.get("depth", 0),
                                    "label": d3.get("label", ""),
                                    "result": d3.get("result"),
                                })

        # 8. Collapse check
        check_collapse(self.state)

        # 9. Classify governance + value convergence (new in frame mutation)
        clamp_morale(self.state)  # defensive final clamp
        gov_form = classify_governance(self.state)
        convergence = compute_value_convergence(self.state)

        # Build delta (Dream Catcher protocol)
        delta_id = hashlib.md5(f"{year}-{self.seed}".encode()).hexdigest()[:12]
        delta = {
            "delta_id": delta_id,
            "year": year,
            "seed": self.seed,
            "event": event,
            "decisions": [
                {"colonist_id": d["colonist_id"], "action": d["action"]}
                for d in decisions
            ],
            "proposals_resolved": resolved,
            "governance_effects": governance_effects,
            "resources_snapshot": dict(self.state.resources),
            "morale": round(self.state.morale, 4),
            "alive_count": len(self.state.alive_colonists()),
            "dead_count": len(self.state.dead_colonists()),
            "terraforming": round(self.state.terraforming_progress, 6),
            "collapsed": self.state.collapsed,
            "sub_sims_this_year": len([d for d in decisions if d.get("sub_sim")]),
            "active_amendments": len(self.state.governance.get("amendments", [])),
            "governance_form": gov_form,
            "value_convergence": convergence["convergence_score"],
        }

        return delta

    def _build_results(self) -> dict:
        """Build the final results dict."""
        return {
            "_meta": {
                "engine": "mars-100",
                "version": "2.0",
                "seed": self.seed,
                "max_years": self.max_years,
                "years_completed": self.state.year,
                "generated": datetime.now(timezone.utc).isoformat(),
            },
            "state": self.state.to_dict(),
            "deltas": self.yearly_deltas,
            "meta_insights": self.meta_insights,
            "summary": self._build_summary(),
        }

    def _build_summary(self) -> dict:
        """Build a human-readable summary of the simulation."""
        alive = self.state.alive_colonists()
        dead = self.state.dead_colonists()

        # Governance analysis
        gov_types = {}
        for p in self.state.proposals:
            gov_types[p.proposal_type] = gov_types.get(p.proposal_type, 0) + 1

        # Relationship analysis
        avg_relationships = {}
        for c in alive:
            if c.relationships:
                avg_relationships[c.id] = round(
                    sum(c.relationships.values()) / len(c.relationships), 3
                )

        # Meta-awareness count
        meta_count = sum(1 for c in self.state.colonists if c.meta_aware)

        # Governance form timeline from deltas
        gov_timeline = []
        for d in self.yearly_deltas:
            if "governance_form" in d:
                gov_timeline.append({"year": d["year"], "form": d["governance_form"]})

        # Value convergence timeline from deltas
        convergence_timeline = [
            {"year": d["year"], "score": d["value_convergence"]}
            for d in self.yearly_deltas if "value_convergence" in d
        ]

        return {
            "years_simulated": self.state.year,
            "collapsed": self.state.collapsed,
            "collapse_reason": self.state.collapse_reason,
            "alive_count": len(alive),
            "dead_count": len(dead),
            "deaths": [
                {"id": c.id, "name": c.name, "year": c.year_died, "cause": c.death_cause}
                for c in dead
            ],
            "total_proposals": len(self.state.proposals),
            "proposal_types": gov_types,
            "passed_proposals": sum(1 for p in self.state.proposals if p.outcome == "passed"),
            "failed_proposals": sum(1 for p in self.state.proposals if p.outcome == "failed"),
            "active_amendments": len(self.state.governance.get("amendments", [])),
            "active_laws": len(self.state.governance.get("constitution", [])),
            "governance_type": self.state.governance.get("type", "consensus"),
            "leader": self.state.governance.get("leader_id"),
            "final_governance_form": classify_governance(self.state),
            "governance_form_timeline": gov_timeline,
            "final_morale": round(self.state.morale, 4),
            "final_resources": dict(self.state.resources),
            "terraforming_progress": round(self.state.terraforming_progress, 6),
            "total_sub_sims": sum(c.sub_sims_run for c in self.state.colonists),
            "meta_aware_colonists": meta_count,
            "depth_2_insights": sum(1 for m in self.meta_insights if m.get("depth") == 2),
            "depth_3_insights": sum(1 for m in self.meta_insights if m.get("depth") == 3),
            "avg_relationships": avg_relationships,
            "meta_insights": self.meta_insights,
            "value_convergence": compute_value_convergence(self.state),
            "convergence_timeline": convergence_timeline,
            "governance_evolution": _analyze_governance_evolution(self.state),
        }


def _analyze_governance_evolution(state: ColonyState) -> list[dict]:
    """Analyze how governance structures emerged over time."""
    phases = []
    proposals = sorted(state.proposals, key=lambda p: p.year)

    if not proposals:
        return phases

    # Group by decade
    for decade_start in range(1, state.year + 1, 10):
        decade_end = min(decade_start + 9, state.year)
        decade_props = [p for p in proposals if decade_start <= p.year <= decade_end]
        if decade_props:
            types = {}
            for p in decade_props:
                types[p.proposal_type] = types.get(p.proposal_type, 0) + 1
            passed = sum(1 for p in decade_props if p.outcome == "passed")
            phases.append({
                "decade": f"Y{decade_start}-Y{decade_end}",
                "proposals": len(decade_props),
                "passed": passed,
                "dominant_type": max(types, key=types.get) if types else None,
                "types": types,
            })

    return phases
