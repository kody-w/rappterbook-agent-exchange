"""
mars100.py — Mars-100: A Recursive Colony Experiment.

100 Martian years. 10 colonists. Sub-simulations up to 3 levels deep.
Colonists run LisPy programs to decide actions, model consequences via
sub-sims, propose governance, and evolve relationships.

The simulation is the star. Raw s-expressions. Emergent governance.
Turtles All the Way Down (Amendment XIII) made concrete.

Usage:
    from src.mars100 import Mars100
    sim = Mars100(seed=42)
    results = sim.run(years=100)

    # Or from CLI:
    python src/mars100.py --years 100 --seed 42
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.lispy import Lispy, LispError, Symbol, Closure, to_sexp, NIL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELEMENTS = ["fire", "water", "earth", "air"]
STAT_NAMES = ["resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia"]
SKILL_NAMES = ["terraforming", "hydroponics", "mediation", "coding", "prayer", "sabotage"]

# Environmental events — weighted by probability
EVENTS = [
    {"name": "dust_storm",        "weight": 15, "severity": (0.3, 0.9), "resource_drain": 0.15, "morale_hit": -0.08},
    {"name": "water_discovery",   "weight": 8,  "severity": (0.0, 0.0), "resource_drain": -0.25, "morale_hit": 0.12},
    {"name": "equipment_failure", "weight": 12, "severity": (0.2, 0.7), "resource_drain": 0.10, "morale_hit": -0.10},
    {"name": "earth_signal",      "weight": 5,  "severity": (0.0, 0.0), "resource_drain": 0.0, "morale_hit": 0.15},
    {"name": "mysterious_signal",  "weight": 3,  "severity": (0.0, 0.0), "resource_drain": 0.0, "morale_hit": 0.05},
    {"name": "solar_flare",       "weight": 10, "severity": (0.4, 1.0), "resource_drain": 0.08, "morale_hit": -0.12},
    {"name": "habitat_breach",    "weight": 6,  "severity": (0.5, 1.0), "resource_drain": 0.20, "morale_hit": -0.15},
    {"name": "food_blight",       "weight": 8,  "severity": (0.3, 0.8), "resource_drain": 0.18, "morale_hit": -0.10},
    {"name": "ore_strike",        "weight": 7,  "severity": (0.0, 0.0), "resource_drain": -0.15, "morale_hit": 0.10},
    {"name": "comm_blackout",     "weight": 10, "severity": (0.1, 0.5), "resource_drain": 0.03, "morale_hit": -0.06},
    {"name": "calm_year",         "weight": 16, "severity": (0.0, 0.0), "resource_drain": -0.05, "morale_hit": 0.03},
]

# Governance proposal types
PROPOSAL_TYPES = [
    "resource_allocation",   # redistribute resources
    "leadership_election",   # choose a new leader
    "exile_vote",            # exile a colonist
    "law_change",            # change a colony rule
    "expansion_plan",        # build new infrastructure
    "sub_sim_mandate",       # mandate sub-sim before major decisions
]

# Colonist names (10 unique, Mars-themed)
COLONIST_NAMES = [
    "Ares", "Rhea", "Phobos", "Deimos", "Olympia",
    "Hellas", "Valles", "Elysium", "Tharsis", "Aonia",
]

# Max LisPy steps per year (shared across all colonist evals + sub-sims)
YEAR_STEP_BUDGET = 100000
COLONIST_STEP_BUDGET = 8000


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Colonist:
    """A Mars-100 colonist — both data and LisPy program."""
    id: str
    name: str
    element: str
    stats: dict[str, float]
    skills: dict[str, float]
    alive: bool = True
    year_arrived: int = 0
    year_died: int | None = None
    cause_of_death: str | None = None
    relationships: dict[str, float] = field(default_factory=dict)
    memory: list[str] = field(default_factory=list)
    governance_votes: dict[str, bool] = field(default_factory=dict)
    proposals_made: int = 0
    sub_sims_run: int = 0
    total_actions: int = 0

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "id": self.id,
            "name": self.name,
            "element": self.element,
            "stats": dict(self.stats),
            "skills": dict(self.skills),
            "alive": self.alive,
            "year_arrived": self.year_arrived,
            "year_died": self.year_died,
            "cause_of_death": self.cause_of_death,
            "relationships": dict(self.relationships),
            "memory": list(self.memory[-20:]),  # keep last 20 memories
            "governance_votes": dict(self.governance_votes),
            "proposals_made": self.proposals_made,
            "sub_sims_run": self.sub_sims_run,
            "total_actions": self.total_actions,
        }

    def to_lispy_dict(self) -> dict:
        """Generate the LisPy-visible view of this colonist."""
        return {
            "id": self.id,
            "name": self.name,
            "element": self.element,
            "alive": self.alive,
            **{f"stat-{k}": v for k, v in self.stats.items()},
            **{f"skill-{k}": v for k, v in self.skills.items()},
        }


@dataclass
class Proposal:
    """A governance proposal."""
    id: str
    type: str
    proposer: str
    year: int
    description: str
    target: str | None = None  # for exile/leadership
    votes_for: list[str] = field(default_factory=list)
    votes_against: list[str] = field(default_factory=list)
    outcome: str | None = None  # "adopted", "rejected", "pending"
    sub_sim_evidence: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "proposer": self.proposer,
            "year": self.year,
            "description": self.description,
            "target": self.target,
            "votes_for": list(self.votes_for),
            "votes_against": list(self.votes_against),
            "outcome": self.outcome,
            "sub_sim_evidence": self.sub_sim_evidence,
        }


@dataclass
class SubSimLog:
    """Log entry for a sub-simulation execution."""
    year: int
    colonist_id: str
    depth: int
    expression: str
    result: str
    steps_used: int

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "colonist_id": self.colonist_id,
            "depth": self.depth,
            "expression": self.expression[:500],
            "result": self.result[:500],
            "steps_used": self.steps_used,
        }


@dataclass
class YearChapter:
    """One chapter of the Mars-100 chronicle."""
    year: int
    event: dict
    colonist_actions: list[dict]
    proposals: list[dict]
    sub_sims: list[dict]
    deaths: list[dict]
    colony_state: dict
    narrative: str

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "event": self.event,
            "colonist_actions": self.colonist_actions,
            "proposals": self.proposals,
            "sub_sims": self.sub_sims,
            "deaths": self.deaths,
            "colony_state": self.colony_state,
            "narrative": self.narrative,
        }


# ---------------------------------------------------------------------------
# Colony State
# ---------------------------------------------------------------------------

@dataclass
class ColonyState:
    """Shared colony resources and infrastructure."""
    food: float = 1000.0         # kg
    water: float = 2000.0        # liters
    power: float = 500.0         # kWh stored
    materials: float = 800.0     # kg construction materials
    morale: float = 0.7          # 0-1 colony average
    habitat_integrity: float = 1.0  # 0-1
    terraform_progress: float = 0.0
    laws: list[str] = field(default_factory=lambda: [
        "majority_vote_decides",
        "no_exile_without_trial",
        "share_resources_equally",
    ])
    leader: str | None = None
    constitution_amendments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "food": round(self.food, 1),
            "water": round(self.water, 1),
            "power": round(self.power, 1),
            "materials": round(self.materials, 1),
            "morale": round(self.morale, 3),
            "habitat_integrity": round(self.habitat_integrity, 3),
            "terraform_progress": round(self.terraform_progress, 6),
            "laws": list(self.laws),
            "leader": self.leader,
            "constitution_amendments": list(self.constitution_amendments),
        }

    def to_lispy_dict(self) -> dict:
        return {
            "food": round(self.food, 1),
            "water": round(self.water, 1),
            "power": round(self.power, 1),
            "materials": round(self.materials, 1),
            "morale": round(self.morale, 3),
            "integrity": round(self.habitat_integrity, 3),
            "terraform": round(self.terraform_progress, 6),
            "leader": self.leader or "none",
        }


# ---------------------------------------------------------------------------
# Mars-100 Simulation
# ---------------------------------------------------------------------------

class Mars100:
    """The Mars-100 recursive colony simulation.

    10 colonists. 100 Martian years. Sub-sims up to depth 3.
    Data sloshing: output of year N = input to year N+1.
    """

    def __init__(self, seed: int = 42, max_depth: int = 3) -> None:
        self.seed = seed
        self.max_depth = max_depth

        # Separate RNGs per domain (rubber-duck recommendation)
        self.event_rng = random.Random(seed)
        self.colonist_rng = random.Random(seed + 1111)
        self.vote_rng = random.Random(seed + 2222)
        self.subsim_rng = random.Random(seed + 3333)
        self.death_rng = random.Random(seed + 4444)

        self.colonists: list[Colonist] = self._create_colonists()
        self.colony = ColonyState()
        self.proposals: list[Proposal] = []
        self.sub_sim_logs: list[SubSimLog] = []
        self.chapters: list[YearChapter] = []
        self.archived_souls: list[dict] = []
        self.year = 0

    def _create_colonists(self) -> list[Colonist]:
        """Generate 10 colonists with diverse profiles."""
        colonists: list[Colonist] = []
        for i, name in enumerate(COLONIST_NAMES):
            element = ELEMENTS[i % len(ELEMENTS)]
            # Stats influenced by element
            base_stats = self._generate_stats(element, i)
            base_skills = self._generate_skills(element, i)
            col = Colonist(
                id=f"col-{i:03d}",
                name=name,
                element=element,
                stats=base_stats,
                skills=base_skills,
            )
            # Initial relationships — slight noise
            for j, other_name in enumerate(COLONIST_NAMES):
                if i != j:
                    col.relationships[f"col-{j:03d}"] = round(
                        0.5 + self.colonist_rng.gauss(0, 0.15), 3
                    )
            colonists.append(col)
        return colonists

    def _generate_stats(self, element: str, idx: int) -> dict[str, float]:
        """Generate colonist stats biased by element."""
        biases = {
            "fire":  {"resolve": 0.2, "improvisation": 0.1, "paranoia": 0.1},
            "water": {"empathy": 0.2, "faith": 0.1, "improvisation": 0.1},
            "earth": {"resolve": 0.1, "hoarding": 0.2, "faith": 0.1},
            "air":   {"improvisation": 0.2, "empathy": 0.1, "paranoia": -0.1},
        }
        stats: dict[str, float] = {}
        element_bias = biases.get(element, {})
        for stat in STAT_NAMES:
            base = 0.4 + self.colonist_rng.random() * 0.4
            bias = element_bias.get(stat, 0.0)
            stats[stat] = round(max(0.05, min(0.99, base + bias)), 3)
        return stats

    def _generate_skills(self, element: str, idx: int) -> dict[str, float]:
        """Generate colonist skills biased by element."""
        biases = {
            "fire":  {"terraforming": 0.2, "sabotage": 0.1},
            "water": {"hydroponics": 0.2, "mediation": 0.1},
            "earth": {"terraforming": 0.1, "coding": 0.2},
            "air":   {"mediation": 0.2, "prayer": 0.1},
        }
        skills: dict[str, float] = {}
        element_bias = biases.get(element, {})
        for skill in SKILL_NAMES:
            base = 0.2 + self.colonist_rng.random() * 0.5
            bias = element_bias.get(skill, 0.0)
            skills[skill] = round(max(0.05, min(0.99, base + bias)), 3)
        return skills

    # --- Yearly Tick ---

    def tick_year(self) -> YearChapter:
        """Advance the colony by one Martian year."""
        self.year += 1

        # 1. Generate environmental event
        event = self._generate_event()

        # 2. Apply event to colony resources
        self._apply_event(event)

        # 3. Base resource production (surviving colonists contribute)
        self._produce_resources()

        # 4. Each colonist observes and acts
        actions: list[dict] = []
        year_sub_sims: list[SubSimLog] = []
        for col in self.colonists:
            if not col.alive:
                continue
            action = self._colonist_act(col, event)
            actions.append(action)
            if action.get("sub_sim") and action.get("sub_sim_log"):
                # Re-create SubSimLog from the dict for internal tracking
                log_dict = action["sub_sim_log"]
                log_entry = SubSimLog(
                    year=log_dict["year"],
                    colonist_id=log_dict["colonist_id"],
                    depth=log_dict["depth"],
                    expression=log_dict["expression"],
                    result=log_dict["result"],
                    steps_used=log_dict["steps_used"],
                )
                year_sub_sims.append(log_entry)

        # 5. Governance: proposals and voting
        year_proposals = self._process_governance(event)

        # 6. Check for deaths
        deaths = self._process_deaths(event)

        # 7. Evolve relationships
        self._evolve_relationships(event, actions)

        # 8. Consume resources
        self._consume_resources()

        # 9. Update colony morale
        self._update_morale(event, deaths)

        # 10. Clamp resources
        self._clamp_resources()

        # Build chapter
        narrative = self._narrate(event, actions, year_proposals, deaths)
        chapter = YearChapter(
            year=self.year,
            event=event,
            colonist_actions=actions,
            proposals=[p.to_dict() for p in year_proposals],
            sub_sims=[s.to_dict() for s in year_sub_sims],
            deaths=deaths,
            colony_state=self.colony.to_dict(),
            narrative=narrative,
        )
        self.chapters.append(chapter)
        self.sub_sim_logs.extend(year_sub_sims)
        return chapter

    def _generate_event(self) -> dict:
        """Pick a weighted-random environmental event."""
        weights = [e["weight"] for e in EVENTS]
        total = sum(weights)
        r = self.event_rng.random() * total
        cumul = 0
        for evt in EVENTS:
            cumul += evt["weight"]
            if r <= cumul:
                severity = 0.0
                if evt["severity"][1] > 0:
                    severity = round(self.event_rng.uniform(*evt["severity"]), 3)
                return {
                    "name": evt["name"],
                    "severity": severity,
                    "resource_drain": round(evt["resource_drain"] * (1 + severity), 3),
                    "morale_hit": round(evt["morale_hit"] * (1 + severity * 0.5), 3),
                    "year": self.year,
                }
        # Fallback
        return {"name": "calm_year", "severity": 0, "resource_drain": -0.05,
                "morale_hit": 0.03, "year": self.year}

    def _apply_event(self, event: dict) -> None:
        """Apply environmental event effects to colony resources."""
        drain = event["resource_drain"]
        if drain > 0:
            self.colony.food *= max(0.5, 1.0 - drain * 0.5)
            self.colony.water *= max(0.5, 1.0 - drain * 0.3)
            self.colony.power *= max(0.5, 1.0 - drain * 0.4)
        elif drain < 0:
            bonus = abs(drain)
            self.colony.food += bonus * 200
            self.colony.water += bonus * 300
            self.colony.materials += bonus * 100

        if event["name"] == "habitat_breach":
            self.colony.habitat_integrity *= max(0.5, 1.0 - event["severity"] * 0.3)

        if event["name"] == "solar_flare":
            self.colony.power *= max(0.3, 1.0 - event["severity"] * 0.5)

    def _produce_resources(self) -> None:
        """Living colonists produce resources based on skills."""
        alive = [c for c in self.colonists if c.alive]
        for col in alive:
            self.colony.food += col.skills["hydroponics"] * 30
            self.colony.water += 20  # baseline recycling
            self.colony.power += col.skills["coding"] * 10 + 15
            self.colony.materials += col.skills["terraforming"] * 8
            self.colony.terraform_progress += col.skills["terraforming"] * 0.0001

    def _colonist_act(self, col: Colonist, event: dict) -> dict:
        """A colonist decides and executes an action via LisPy."""
        col.total_actions += 1

        # Build the LisPy program this colonist will run
        action_type, description = self._decide_action(col, event)

        # Check if colonist wants to run a sub-sim first
        sub_sim_result = None
        sub_sim_log_entry = None
        if self._should_sub_sim(col, event, action_type):
            sub_sim_result, sub_sim_log_entry = self._run_sub_sim(col, event, action_type)
            col.sub_sims_run += 1

        # Apply the action
        self._apply_action(col, action_type, event, sub_sim_result)

        # Record memory
        memory_entry = f"Year {self.year}: {event['name']} — I chose to {action_type}: {description}"
        col.memory.append(memory_entry)

        return {
            "colonist_id": col.id,
            "colonist_name": col.name,
            "action": action_type,
            "description": description,
            "sub_sim": sub_sim_result is not None,
            "sub_sim_log": sub_sim_log_entry.to_dict() if sub_sim_log_entry else None,
            "sub_sim_result": str(sub_sim_result)[:200] if sub_sim_result else None,
        }

    def _decide_action(self, col: Colonist, event: dict) -> tuple[str, str]:
        """Use colonist personality to decide action type."""
        rng = random.Random(self.seed + self.year * 100 + hash(col.id))
        severity = event.get("severity", 0)

        # Personality-weighted action selection
        weights: dict[str, float] = {
            "repair": col.skills["coding"] * 0.5 + (severity * 0.3),
            "farm": col.skills["hydroponics"] * 0.6,
            "terraform": col.skills["terraforming"] * 0.4,
            "mediate": col.skills["mediation"] * 0.5,
            "pray": col.skills["prayer"] * (0.3 + col.stats["faith"] * 0.3),
            "hoard": col.stats["hoarding"] * 0.5,
            "explore": col.stats["improvisation"] * 0.4,
            "build": col.skills["coding"] * 0.3 + col.stats["resolve"] * 0.2,
        }

        # Boost repair when damaged
        if event["name"] in ("habitat_breach", "equipment_failure"):
            weights["repair"] += 0.5
        if event["name"] == "food_blight":
            weights["farm"] += 0.4
        if event["name"] in ("earth_signal", "mysterious_signal"):
            weights["explore"] += 0.3
            weights["pray"] += 0.2

        # Pick weighted random action
        total = sum(weights.values())
        r = rng.random() * total
        cumul = 0.0
        chosen = "explore"
        for action, w in weights.items():
            cumul += w
            if r <= cumul:
                chosen = action
                break

        descriptions = {
            "repair": f"{col.name} works on repairing habitat systems",
            "farm": f"{col.name} tends the hydroponics bay",
            "terraform": f"{col.name} runs terraforming equipment",
            "mediate": f"{col.name} mediates tensions between colonists",
            "pray": f"{col.name} retreats for contemplation",
            "hoard": f"{col.name} stockpiles personal supplies",
            "explore": f"{col.name} scouts the surrounding terrain",
            "build": f"{col.name} builds new infrastructure",
        }
        return chosen, descriptions.get(chosen, f"{col.name} acts")

    def _should_sub_sim(self, col: Colonist, event: dict, action: str) -> bool:
        """Decide if this colonist runs a sub-sim before acting."""
        # Higher paranoia/improvisation → more likely to simulate first
        chance = col.stats["paranoia"] * 0.3 + col.stats["improvisation"] * 0.2
        # Dangerous events increase sub-sim probability
        if event["severity"] > 0.5:
            chance += 0.2
        # Governance actions always benefit from sub-sim
        if action == "mediate":
            chance += 0.15
        return self.subsim_rng.random() < chance

    def _run_sub_sim(self, col: Colonist, event: dict, action: str) -> tuple[object, SubSimLog]:
        """Run a LisPy sub-simulation for this colonist."""
        step_budget = [COLONIST_STEP_BUDGET]
        vm = Lispy(max_steps=COLONIST_STEP_BUDGET, max_depth=self.max_depth,
                   current_depth=0, step_budget=step_budget)

        # Inject colony state into LisPy environment
        colony_data = self.colony.to_lispy_dict()
        for key, val in colony_data.items():
            vm.global_env.set(f"colony-{key}", val)
        vm.global_env.set("event-name", event["name"])
        vm.global_env.set("event-severity", event["severity"])
        vm.global_env.set("year", self.year)
        vm.global_env.set("my-name", col.name)
        vm.global_env.set("my-element", col.element)
        for stat, val in col.stats.items():
            vm.global_env.set(f"my-{stat}", val)
        for skill, val in col.skills.items():
            vm.global_env.set(f"my-{skill}", val)

        # The colonist's sub-sim program: model what happens if they take this action
        program = self._generate_sub_sim_program(col, event, action)
        steps_before = step_budget[0]
        result = vm.run(program)
        steps_used = steps_before - step_budget[0]

        log = SubSimLog(
            year=self.year,
            colonist_id=col.id,
            depth=1,
            expression=program,
            result=to_sexp(result) if not isinstance(result, LispError) else str(result),
            steps_used=steps_used,
        )
        return result, log

    def _generate_sub_sim_program(self, col: Colonist, event: dict, action: str) -> str:
        """Generate the LisPy sub-sim program for a colonist's decision."""
        # The colonist simulates outcomes
        programs = {
            "repair": f"""
(let ((cost (* event-severity 50))
      (benefit (* my-coding 0.8))
      (risk (if (> event-severity 0.7) 0.3 0.1)))
  (if (> benefit risk)
    (list "repair" "worth_it" (- benefit risk))
    (list "repair" "risky" risk)))""",
            "farm": f"""
(let ((yield (* my-hydroponics 30))
      (food-need (* 10 1.8))
      (surplus (- (+ colony-food yield) food-need)))
  (if (> surplus 0)
    (list "farm" "surplus" surplus)
    (list "farm" "deficit" surplus)))""",
            "terraform": f"""
(let ((progress (* my-terraforming 0.001))
      (cost (* 20 (- 1 my-resolve)))
      (net (- progress cost)))
  (sub-sim
    (let ((projected (+ colony-terraform progress)))
      (if (> projected 0.01)
        (list "terraform" "milestone" projected)
        (list "terraform" "incremental" projected)))))""",
            "mediate": f"""
(let ((tension (- 1.0 colony-morale))
      (skill my-mediation)
      (resolution (* skill (- 1.0 tension))))
  (if (> resolution 0.3)
    (list "mediate" "effective" resolution)
    (list "mediate" "insufficient" tension)))""",
            "explore": f"""
(let ((discovery-chance (* my-improvisation 0.4))
      (risk (* event-severity 0.3))
      (net (- discovery-chance risk)))
  (if (> net 0)
    (list "explore" "promising" discovery-chance)
    (list "explore" "dangerous" risk)))""",
            "pray": f"""
(let ((peace (* my-faith 0.5))
      (clarity (* my-faith my-resolve))
      (insight (if (> clarity 0.3) "revelation" "calm")))
  (list "pray" insight peace))""",
            "hoard": f"""
(let ((stash (* my-hoarding 20))
      (guilt (* my-empathy 0.5))
      (detected (* (- 1.0 my-improvisation) 0.3)))
  (if (> detected 0.5)
    (list "hoard" "caught" guilt)
    (list "hoard" "hidden" stash)))""",
            "build": f"""
(let ((capacity (* my-coding my-resolve 50))
      (cost (* 30 (- 1.0 my-improvisation)))
      (net (- capacity cost)))
  (if (> net 0)
    (list "build" "success" net)
    (list "build" "stalled" cost)))""",
        }
        return programs.get(action, '(list "default" "act" 0)')

    def _apply_action(self, col: Colonist, action: str, event: dict,
                      sub_sim_result: object) -> None:
        """Apply the consequences of a colonist's action."""
        skill_factor = 0.5
        if action == "repair":
            repair_amount = col.skills["coding"] * 0.1
            self.colony.habitat_integrity = min(1.0, self.colony.habitat_integrity + repair_amount)
            self.colony.power -= 5
        elif action == "farm":
            self.colony.food += col.skills["hydroponics"] * 15
            self.colony.water -= 10
        elif action == "terraform":
            self.colony.terraform_progress += col.skills["terraforming"] * 0.0002
            self.colony.power -= 10
            self.colony.materials -= 5
        elif action == "mediate":
            self.colony.morale += col.skills["mediation"] * 0.03
        elif action == "pray":
            col.stats["faith"] = min(0.99, col.stats["faith"] + 0.01)
            self.colony.morale += col.stats["faith"] * 0.01
        elif action == "hoard":
            stolen = min(20, self.colony.food * 0.02)
            self.colony.food -= stolen
            # Detected?
            if self.colonist_rng.random() < (1 - col.stats["improvisation"]) * 0.3:
                col.memory.append(f"Year {self.year}: Caught hoarding. Shame.")
                for other in self.colonists:
                    if other.id != col.id and other.alive:
                        delta = other.relationships.get(col.id, 0.5) - 0.1
                        other.relationships[col.id] = max(0.0, delta)
        elif action == "explore":
            if self.colonist_rng.random() < col.stats["improvisation"] * 0.3:
                bonus = self.colonist_rng.choice(["materials", "water", "food"])
                amount = self.colonist_rng.randint(20, 80)
                setattr(self.colony, bonus, getattr(self.colony, bonus) + amount)
                col.memory.append(f"Year {self.year}: Found {amount} units of {bonus}!")
        elif action == "build":
            if col.skills["coding"] + col.stats["resolve"] > 0.8:
                self.colony.materials -= 20
                self.colony.habitat_integrity = min(1.0, self.colony.habitat_integrity + 0.05)

        # Sub-sim insight bonus — if they simulated first, slightly better outcome
        if sub_sim_result is not None and not isinstance(sub_sim_result, LispError):
            self.colony.morale += 0.005  # foresight bonus

    def _process_governance(self, event: dict) -> list[Proposal]:
        """Generate and resolve governance proposals."""
        alive = [c for c in self.colonists if c.alive]
        if len(alive) < 3:
            return []

        year_proposals: list[Proposal] = []

        # Check if anyone wants to propose something
        for col in alive:
            proposal_chance = col.stats["resolve"] * 0.08 + col.skills["mediation"] * 0.05
            # Crisis increases proposals
            if event["severity"] > 0.5:
                proposal_chance += 0.1
            # Low morale triggers governance activity
            if self.colony.morale < 0.4:
                proposal_chance += 0.15

            if self.vote_rng.random() < proposal_chance:
                proposal = self._generate_proposal(col, event)
                if proposal:
                    year_proposals.append(proposal)
                    col.proposals_made += 1

        # Resolve proposals via voting
        for proposal in year_proposals:
            self._resolve_proposal(proposal, alive)

        self.proposals.extend(year_proposals)
        return year_proposals

    def _generate_proposal(self, col: Colonist, event: dict) -> Proposal | None:
        """Generate a governance proposal based on colonist personality."""
        # Determine proposal type from personality
        if self.colony.morale < 0.3 and col.stats["resolve"] > 0.5:
            ptype = "leadership_election"
            desc = f"{col.name} proposes new leadership to address the crisis"
            target = col.id  # self-nomination
        elif self.colony.food < 200 and col.stats["hoarding"] > 0.5:
            ptype = "resource_allocation"
            desc = f"{col.name} proposes rationing food supplies"
            target = None
        elif event["severity"] > 0.7 and col.stats["paranoia"] > 0.5:
            # Find least-liked colonist
            worst_rel = min(
                ((oid, rel) for oid, rel in col.relationships.items()
                 if any(c.id == oid and c.alive for c in self.colonists)),
                key=lambda x: x[1],
                default=(None, 1.0),
            )
            if worst_rel[0] and worst_rel[1] < 0.3:
                ptype = "exile_vote"
                desc = f"{col.name} calls for exile vote"
                target = worst_rel[0]
            else:
                ptype = "law_change"
                desc = f"{col.name} proposes mandatory sub-sim before major decisions"
                target = None
        elif col.skills["terraforming"] > 0.6:
            ptype = "expansion_plan"
            desc = f"{col.name} proposes accelerating terraforming"
            target = None
        else:
            ptype = "sub_sim_mandate"
            desc = f"{col.name} proposes requiring simulations before colony decisions"
            target = None

        pid = hashlib.md5(f"{self.year}-{col.id}-{ptype}".encode()).hexdigest()[:8]
        return Proposal(
            id=f"prop-{pid}",
            type=ptype,
            proposer=col.id,
            year=self.year,
            description=desc,
            target=target,
        )

    def _resolve_proposal(self, proposal: Proposal, voters: list[Colonist]) -> None:
        """Resolve a proposal through colonist voting."""
        for voter in voters:
            # Don't vote on own proposal (abstain)
            if voter.id == proposal.proposer:
                proposal.votes_for.append(voter.id)
                continue

            vote_for_chance = 0.5  # baseline

            # Relationship with proposer influences vote
            rel = voter.relationships.get(proposal.proposer, 0.5)
            vote_for_chance += (rel - 0.5) * 0.3

            # Proposal type preferences
            if proposal.type == "exile_vote" and voter.stats["empathy"] > 0.6:
                vote_for_chance -= 0.3
            if proposal.type == "exile_vote" and proposal.target == voter.id:
                vote_for_chance -= 0.8
            if proposal.type == "resource_allocation" and voter.stats["hoarding"] > 0.5:
                vote_for_chance -= 0.2
            if proposal.type == "leadership_election" and voter.stats["resolve"] < 0.4:
                vote_for_chance += 0.2
            if proposal.type == "sub_sim_mandate" and voter.stats["improvisation"] > 0.5:
                vote_for_chance += 0.2

            if self.vote_rng.random() < max(0.05, min(0.95, vote_for_chance)):
                proposal.votes_for.append(voter.id)
                voter.governance_votes[proposal.id] = True
            else:
                proposal.votes_against.append(voter.id)
                voter.governance_votes[proposal.id] = False

        # Determine outcome — majority of alive colonists
        quorum = len(voters) // 2 + 1
        if len(proposal.votes_for) >= quorum:
            proposal.outcome = "adopted"
            self._adopt_proposal(proposal)
        else:
            proposal.outcome = "rejected"

    def _adopt_proposal(self, proposal: Proposal) -> None:
        """Apply the effects of an adopted proposal."""
        if proposal.type == "leadership_election":
            self.colony.leader = proposal.target or proposal.proposer
            self.colony.morale += 0.05

        elif proposal.type == "resource_allocation":
            self.colony.laws.append("rationing_active")
            self.colony.morale -= 0.03

        elif proposal.type == "exile_vote" and proposal.target:
            target_col = next(
                (c for c in self.colonists if c.id == proposal.target and c.alive),
                None,
            )
            if target_col:
                target_col.alive = False
                target_col.year_died = self.year
                target_col.cause_of_death = "exiled"
                self._archive_soul(target_col)
                self.colony.morale -= 0.1  # exile is traumatic

        elif proposal.type == "expansion_plan":
            self.colony.materials -= 50
            self.colony.terraform_progress += 0.001

        elif proposal.type == "sub_sim_mandate":
            if "sub_sim_before_decisions" not in self.colony.laws:
                self.colony.laws.append("sub_sim_before_decisions")
                self.colony.constitution_amendments.append(
                    f"Year {self.year}: Sub-simulation mandate adopted"
                )

        elif proposal.type == "law_change":
            self.colony.constitution_amendments.append(
                f"Year {self.year}: {proposal.description}"
            )

    def _process_deaths(self, event: dict) -> list[dict]:
        """Check for colonist deaths — starvation, radiation, events."""
        deaths: list[dict] = []
        alive = [c for c in self.colonists if c.alive]
        alive_count = len(alive)
        if alive_count == 0:
            return deaths

        food_per_person = self.colony.food / max(1, alive_count)

        for col in alive:
            cause = None

            # Starvation
            if food_per_person < 5:
                if self.death_rng.random() < 0.15:
                    cause = "starvation"

            # Habitat breach — low integrity kills
            if self.colony.habitat_integrity < 0.3:
                if self.death_rng.random() < 0.1 * (1 - self.colony.habitat_integrity):
                    cause = "habitat_failure"

            # Severe events can kill
            if event["severity"] > 0.8:
                death_chance = (event["severity"] - 0.8) * 0.2
                # Resolve and element resistance
                if col.element == "earth":
                    death_chance *= 0.5
                death_chance *= (1 - col.stats["resolve"] * 0.5)
                if self.death_rng.random() < death_chance:
                    cause = event["name"]

            # Old age (after year 70 of the sim, earliest colonists age)
            if self.year > 70:
                age_death_chance = (self.year - 70) * 0.005
                if self.death_rng.random() < age_death_chance:
                    cause = "old_age"

            if cause:
                col.alive = False
                col.year_died = self.year
                col.cause_of_death = cause
                self._archive_soul(col)
                deaths.append({
                    "colonist_id": col.id,
                    "name": col.name,
                    "cause": cause,
                    "year": self.year,
                })

        return deaths

    def _archive_soul(self, col: Colonist) -> None:
        """Archive a dead colonist's soul — legacy, not delete."""
        soul = col.to_dict()
        soul["archived_at"] = self.year
        soul["epitaph"] = self._generate_epitaph(col)
        self.archived_souls.append(soul)

    def _generate_epitaph(self, col: Colonist) -> str:
        """Generate a poetic epitaph for a dead colonist."""
        epitaphs = {
            "fire": f"{col.name} burned bright. The dust remembers their heat.",
            "water": f"{col.name} flowed where others couldn't. The ice mourns.",
            "earth": f"{col.name} stood firm until the ground gave way.",
            "air": f"{col.name} drifted free. The wind carries their last thought.",
        }
        base = epitaphs.get(col.element, f"{col.name} is remembered.")
        if col.cause_of_death == "exiled":
            return f"{col.name} walks alone in the red dust. Exiled, year {col.year_died}."
        if col.sub_sims_run > 5:
            return f"{base} They ran {col.sub_sims_run} simulations. Did they find the answer?"
        return base

    def _evolve_relationships(self, event: dict, actions: list[dict]) -> None:
        """Relationships evolve based on shared experiences."""
        alive = [c for c in self.colonists if c.alive]
        for col in alive:
            for other in alive:
                if col.id == other.id:
                    continue
                rel = col.relationships.get(other.id, 0.5)

                # Shared hardship bonds
                if event["severity"] > 0.5:
                    rel += 0.02

                # Same action type → affinity
                col_action = next((a for a in actions if a["colonist_id"] == col.id), None)
                other_action = next((a for a in actions if a["colonist_id"] == other.id), None)
                if col_action and other_action and col_action["action"] == other_action["action"]:
                    rel += 0.01

                # Element compatibility
                compatible = {
                    ("fire", "air"), ("air", "fire"),
                    ("water", "earth"), ("earth", "water"),
                }
                if (col.element, other.element) in compatible:
                    rel += 0.005

                # Empathy smooths relationships
                empathy_avg = (col.stats["empathy"] + other.stats["empathy"]) / 2
                rel += empathy_avg * 0.005

                # Paranoia degrades relationships
                if col.stats["paranoia"] > 0.7:
                    rel -= 0.01

                # Slow decay toward neutral
                rel = rel * 0.98 + 0.5 * 0.02

                col.relationships[other.id] = round(max(0.0, min(1.0, rel)), 3)

    def _consume_resources(self) -> None:
        """Consume resources for alive colonists."""
        alive_count = sum(1 for c in self.colonists if c.alive)
        self.colony.food -= alive_count * 1.8 * 668  # 1.8 kg/sol × 668 sols/year, scaled down
        self.colony.water -= alive_count * 0.5 * 668  # recycled, so lower
        self.colony.power -= alive_count * 20

        # Scale down to game-playable numbers
        self.colony.food = max(0, self.colony.food)
        self.colony.water = max(0, self.colony.water)
        self.colony.power = max(0, self.colony.power)

    def _update_morale(self, event: dict, deaths: list[dict]) -> None:
        """Update colony morale."""
        self.colony.morale += event["morale_hit"]

        # Deaths hurt morale
        for d in deaths:
            self.colony.morale -= 0.05

        # Leadership bonus
        if self.colony.leader:
            leader = next((c for c in self.colonists if c.id == self.colony.leader and c.alive), None)
            if leader:
                self.colony.morale += leader.skills["mediation"] * 0.02

        # Terraform progress boosts morale
        if self.colony.terraform_progress > 0.005:
            self.colony.morale += 0.01

        self.colony.morale = max(0.0, min(1.0, self.colony.morale))

    def _clamp_resources(self) -> None:
        """Ensure no resource goes below zero."""
        self.colony.food = max(0.0, self.colony.food)
        self.colony.water = max(0.0, self.colony.water)
        self.colony.power = max(0.0, self.colony.power)
        self.colony.materials = max(0.0, self.colony.materials)
        self.colony.habitat_integrity = max(0.0, min(1.0, self.colony.habitat_integrity))
        self.colony.terraform_progress = max(0.0, self.colony.terraform_progress)

    def _narrate(self, event: dict, actions: list[dict],
                 proposals: list[Proposal], deaths: list[dict]) -> str:
        """Generate human-readable narrative for this year."""
        alive = sum(1 for c in self.colonists if c.alive)
        lines: list[str] = []
        lines.append(f"## Year {self.year} of Mars-100")
        lines.append("")
        lines.append(f"**Event:** {event['name'].replace('_', ' ').title()}"
                     f" (severity {event['severity']:.1f})")
        lines.append(f"**Population:** {alive} colonists alive")
        lines.append(f"**Resources:** food={self.colony.food:.0f}kg, "
                     f"water={self.colony.water:.0f}L, power={self.colony.power:.0f}kWh")
        lines.append(f"**Morale:** {self.colony.morale:.1%}")
        lines.append("")

        # Colonist diary entries (top 3 by action importance)
        lines.append("### Colonist Diaries")
        for act in actions[:3]:
            lines.append(f"- **{act['colonist_name']}** ({act['action']}): {act['description']}")
            if act.get("sub_sim"):
                lines.append(f"  *Ran sub-simulation before deciding. Result: {act.get('sub_sim_result', 'N/A')[:100]}*")
        lines.append("")

        # Governance
        if proposals:
            lines.append("### Governance")
            for p in proposals:
                outcome_str = p.outcome or "pending"
                lines.append(f"- **{p.type}**: {p.description} → *{outcome_str}* "
                             f"({len(p.votes_for)} for, {len(p.votes_against)} against)")
            lines.append("")

        # Deaths
        if deaths:
            lines.append("### Deaths")
            for d in deaths:
                lines.append(f"- **{d['name']}** — {d['cause']} (year {d['year']})")
            lines.append("")

        return "\n".join(lines)

    # --- Full Run ---

    def run(self, years: int = 100, callback: object = None) -> dict:
        """Run the full Mars-100 simulation."""
        for _ in range(years):
            chapter = self.tick_year()
            if callback:
                callback(self.year, chapter)
            # Check for colony collapse
            alive = sum(1 for c in self.colonists if c.alive)
            if alive == 0:
                break
        return self.results()

    def results(self) -> dict:
        """Package simulation results."""
        alive = [c for c in self.colonists if c.alive]
        now = datetime.now(timezone.utc).isoformat()

        # Determine emergent governance patterns
        governance_patterns = self._analyze_governance()

        return {
            "_meta": {
                "engine": "mars-100",
                "version": "1.0",
                "seed": self.seed,
                "years_simulated": self.year,
                "generated": now,
                "max_depth": self.max_depth,
            },
            "colony": self.colony.to_dict(),
            "colonists": [c.to_dict() for c in self.colonists],
            "archived_souls": self.archived_souls,
            "proposals": [p.to_dict() for p in self.proposals],
            "sub_sim_logs": [s.to_dict() for s in self.sub_sim_logs[-100:]],
            "chapters": [ch.to_dict() for ch in self.chapters],
            "governance_patterns": governance_patterns,
            "summary": self._summary(),
        }

    def _summary(self) -> dict:
        """Generate simulation summary."""
        alive = [c for c in self.colonists if c.alive]
        dead = [c for c in self.colonists if not c.alive]
        total_sub_sims = sum(c.sub_sims_run for c in self.colonists)
        adopted = [p for p in self.proposals if p.outcome == "adopted"]
        rejected = [p for p in self.proposals if p.outcome == "rejected"]
        death_causes: dict[str, int] = {}
        for c in dead:
            cause = c.cause_of_death or "unknown"
            death_causes[cause] = death_causes.get(cause, 0) + 1

        return {
            "years_survived": self.year,
            "final_population": len(alive),
            "total_deaths": len(dead),
            "death_causes": death_causes,
            "total_proposals": len(self.proposals),
            "proposals_adopted": len(adopted),
            "proposals_rejected": len(rejected),
            "total_sub_sims": total_sub_sims,
            "constitution_amendments": len(self.colony.constitution_amendments),
            "terraform_progress": round(self.colony.terraform_progress, 6),
            "final_morale": round(self.colony.morale, 3),
            "laws": self.colony.laws,
        }

    def _analyze_governance(self) -> dict:
        """Analyze emergent governance patterns."""
        patterns: dict = {
            "leadership_changes": 0,
            "exile_attempts": 0,
            "exiles_carried_out": 0,
            "laws_changed": len(self.colony.constitution_amendments),
            "sub_sim_mandate": "sub_sim_before_decisions" in self.colony.laws,
            "dominant_proposer": None,
            "most_controversial_year": None,
        }

        proposer_counts: dict[str, int] = {}
        proposals_per_year: dict[int, int] = {}
        for p in self.proposals:
            proposer_counts[p.proposer] = proposer_counts.get(p.proposer, 0) + 1
            proposals_per_year[p.year] = proposals_per_year.get(p.year, 0) + 1
            if p.type == "leadership_election" and p.outcome == "adopted":
                patterns["leadership_changes"] += 1
            if p.type == "exile_vote":
                patterns["exile_attempts"] += 1
                if p.outcome == "adopted":
                    patterns["exiles_carried_out"] += 1

        if proposer_counts:
            patterns["dominant_proposer"] = max(proposer_counts, key=proposer_counts.get)
        if proposals_per_year:
            patterns["most_controversial_year"] = max(proposals_per_year, key=proposals_per_year.get)

        # Meta-insight: did any sub-sim at depth 2+ produce actionable insights?
        deep_sims = [s for s in self.sub_sim_logs if s.depth >= 2]
        patterns["deep_sub_sims"] = len(deep_sims)

        return patterns


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mars-100: A Recursive Colony Experiment")
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "docs" / "mars-100"
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = Mars100(seed=args.seed, max_depth=args.max_depth)

    def on_year(year: int, chapter: YearChapter) -> None:
        if not args.quiet:
            alive = sum(1 for c in sim.colonists if c.alive)
            print(f"  Year {year:>3}/{args.years}  pop={alive}  "
                  f"morale={sim.colony.morale:.1%}  event={chapter.event['name']}")

    print(f"Mars-100 — simulating {args.years} years with {len(sim.colonists)} colonists...")
    print(f"  Seed: {args.seed}  Max depth: {args.max_depth}")
    print()

    results = sim.run(years=args.years, callback=on_year)

    print()
    print("=" * 60)
    print("MARS-100 COMPLETE")
    print("=" * 60)
    s = results["summary"]
    print(f"  Years survived:     {s['years_survived']}")
    print(f"  Final population:   {s['final_population']}/10")
    print(f"  Total deaths:       {s['total_deaths']}")
    if s["death_causes"]:
        for cause, count in sorted(s["death_causes"].items(), key=lambda x: -x[1]):
            print(f"    {cause}: {count}")
    print(f"  Proposals:          {s['total_proposals']} ({s['proposals_adopted']} adopted)")
    print(f"  Sub-simulations:    {s['total_sub_sims']}")
    print(f"  Amendments:         {s['constitution_amendments']}")
    print(f"  Terraform progress: {s['terraform_progress']*100:.4f}%")
    print(f"  Final morale:       {s['final_morale']:.1%}")
    print()

    # Save results
    results_path = output_dir / "results.json"
    tmp = results_path.with_suffix(".tmp")
    # Compact chapters for space
    compact = {k: v for k, v in results.items() if k != "chapters"}
    compact["chapter_count"] = len(results["chapters"])
    compact["narratives"] = [ch["narrative"] for ch in results["chapters"]]
    tmp.write_text(json.dumps(compact, indent=2))
    tmp.rename(results_path)
    print(f"Results saved: {results_path}")

    # Save per-colonist state
    colonists_dir = output_dir / "colonists"
    colonists_dir.mkdir(exist_ok=True)
    for col in results["colonists"]:
        col_path = colonists_dir / f"{col['id']}.json"
        col_path.write_text(json.dumps(col, indent=2))

    # Save archived souls
    for soul in results["archived_souls"]:
        soul_path = colonists_dir / f"{soul['id']}-soul.json"
        soul_path.write_text(json.dumps(soul, indent=2))

    print(f"Colonist states: {colonists_dir}")
    print()


if __name__ == "__main__":
    main()
