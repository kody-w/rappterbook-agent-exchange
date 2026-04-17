"""
mars100.py -- Mars-100 Recursive Colony Simulation.

A 100-year Mars colony with 10 agent-colonists, each making decisions
via LisPy expressions. Sub-simulations up to 3 levels deep allow
colonists to model governance proposals before committing.

Each sim frame = 1 Martian year (~687 Earth days).

Constitutional basis:
  - Turtles All the Way Down (Amendment XIII): fractal frame loops
  - Dream Catcher (Amendment XVI): per-year deltas, additive merging
  - Legacy, not delete (Amendment X): dead colonists become archived souls

Usage:
    from src.mars100 import Colony, run_simulation

    colony = Colony.genesis(seed=42)
    for year in range(1, 101):
        delta = colony.tick(year)
        print(f"Year {year}: {delta['summary']}")

    state = colony.to_dict()
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.lispy import (
    evaluate as lispy_evaluate,
    standard_env,
    LispError,
    LispyError,
    LispyDepthExceeded,
    LispyBudgetExhausted,
    Env,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARS_YEAR_SOLS = 687  # sols per Martian year

ELEMENTS = ["fire", "water", "earth", "air"]

STAT_NAMES = ["resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia"]
SKILL_NAMES = ["terraforming", "hydroponics", "mediation", "coding", "prayer", "sabotage"]

# Environmental events with probabilities and effects
EVENT_TYPES = [
    {"name": "dust_storm", "weight": 25, "food_mod": -0.15, "morale_mod": -0.1, "power_mod": -0.2},
    {"name": "resource_strike", "weight": 15, "food_mod": 0.2, "morale_mod": 0.1, "power_mod": 0.0},
    {"name": "equipment_failure", "weight": 20, "food_mod": -0.05, "morale_mod": -0.15, "power_mod": -0.3},
    {"name": "earth_contact", "weight": 15, "food_mod": 0.0, "morale_mod": 0.3, "power_mod": 0.0},
    {"name": "solar_flare", "weight": 10, "food_mod": 0.0, "morale_mod": -0.2, "power_mod": -0.4},
    {"name": "microbe_discovery", "weight": 5, "food_mod": 0.0, "morale_mod": 0.4, "power_mod": 0.0},
    {"name": "supply_ship", "weight": 8, "food_mod": 0.3, "morale_mod": 0.2, "power_mod": 0.1},
    {"name": "alien_signal", "weight": 2, "food_mod": 0.0, "morale_mod": 0.0, "power_mod": 0.0},
]

# Action templates colonists can choose (LisPy-evaluable)
ACTION_NAMES = [
    "farm", "repair", "explore", "mediate", "research",
    "hoard", "pray", "sabotage", "build", "rest",
    "propose-governance", "run-sub-sim", "trade", "teach",
]


# ---------------------------------------------------------------------------
# Colonist definition
# ---------------------------------------------------------------------------

# The 10 founding colonists
FOUNDING_COLONISTS = [
    {"id": "ares",   "name": "Ares",   "element": "fire",  "role": "commander",
     "stats": {"resolve": 0.9, "improvisation": 0.4, "empathy": 0.3, "hoarding": 0.5, "faith": 0.2, "paranoia": 0.6},
     "skills": {"terraforming": 0.3, "hydroponics": 0.2, "mediation": 0.4, "coding": 0.3, "prayer": 0.1, "sabotage": 0.5}},
    {"id": "lyra",   "name": "Lyra",   "element": "air",   "role": "diplomat",
     "stats": {"resolve": 0.5, "improvisation": 0.6, "empathy": 0.9, "hoarding": 0.1, "faith": 0.5, "paranoia": 0.2},
     "skills": {"terraforming": 0.1, "hydroponics": 0.3, "mediation": 0.9, "coding": 0.2, "prayer": 0.4, "sabotage": 0.0}},
    {"id": "thane",  "name": "Thane",  "element": "earth", "role": "engineer",
     "stats": {"resolve": 0.7, "improvisation": 0.3, "empathy": 0.4, "hoarding": 0.8, "faith": 0.1, "paranoia": 0.4},
     "skills": {"terraforming": 0.8, "hydroponics": 0.4, "mediation": 0.2, "coding": 0.7, "prayer": 0.0, "sabotage": 0.2}},
    {"id": "vex",    "name": "Vex",    "element": "fire",  "role": "security",
     "stats": {"resolve": 0.8, "improvisation": 0.5, "empathy": 0.2, "hoarding": 0.6, "faith": 0.1, "paranoia": 0.9},
     "skills": {"terraforming": 0.2, "hydroponics": 0.1, "mediation": 0.3, "coding": 0.5, "prayer": 0.0, "sabotage": 0.8}},
    {"id": "sera",   "name": "Sera",   "element": "water", "role": "biologist",
     "stats": {"resolve": 0.4, "improvisation": 0.7, "empathy": 0.7, "hoarding": 0.2, "faith": 0.8, "paranoia": 0.3},
     "skills": {"terraforming": 0.5, "hydroponics": 0.9, "mediation": 0.5, "coding": 0.1, "prayer": 0.7, "sabotage": 0.0}},
    {"id": "kai",    "name": "Kai",    "element": "air",   "role": "architect",
     "stats": {"resolve": 0.6, "improvisation": 0.9, "empathy": 0.5, "hoarding": 0.3, "faith": 0.3, "paranoia": 0.4},
     "skills": {"terraforming": 0.7, "hydroponics": 0.3, "mediation": 0.4, "coding": 0.8, "prayer": 0.1, "sabotage": 0.1}},
    {"id": "nova",   "name": "Nova",   "element": "fire",  "role": "geologist",
     "stats": {"resolve": 0.8, "improvisation": 0.6, "empathy": 0.4, "hoarding": 0.4, "faith": 0.3, "paranoia": 0.5},
     "skills": {"terraforming": 0.9, "hydroponics": 0.2, "mediation": 0.3, "coding": 0.4, "prayer": 0.2, "sabotage": 0.3}},
    {"id": "mira",   "name": "Mira",   "element": "water", "role": "medic",
     "stats": {"resolve": 0.5, "improvisation": 0.7, "empathy": 0.9, "hoarding": 0.1, "faith": 0.6, "paranoia": 0.2},
     "skills": {"terraforming": 0.2, "hydroponics": 0.6, "mediation": 0.8, "coding": 0.3, "prayer": 0.5, "sabotage": 0.0}},
    {"id": "orion",  "name": "Orion",  "element": "earth", "role": "programmer",
     "stats": {"resolve": 0.6, "improvisation": 0.5, "empathy": 0.3, "hoarding": 0.5, "faith": 0.2, "paranoia": 0.7},
     "skills": {"terraforming": 0.3, "hydroponics": 0.2, "mediation": 0.2, "coding": 0.9, "prayer": 0.1, "sabotage": 0.4}},
    {"id": "zeph",   "name": "Zeph",   "element": "air",   "role": "pilot",
     "stats": {"resolve": 0.7, "improvisation": 0.8, "empathy": 0.6, "hoarding": 0.2, "faith": 0.4, "paranoia": 0.3},
     "skills": {"terraforming": 0.4, "hydroponics": 0.3, "mediation": 0.6, "coding": 0.5, "prayer": 0.3, "sabotage": 0.1}},
]


# ---------------------------------------------------------------------------
# LisPy decision policies (parameterized by colonist personality)
# ---------------------------------------------------------------------------

def _build_policy(colonist: dict) -> str:
    """Build a LisPy decision policy from colonist personality.

    Returns an s-expression that evaluates to an action string.
    The policy is parameterized by the colonist's stats and skills.
    """
    s = colonist["stats"]
    sk = colonist["skills"]

    # The policy evaluates conditions based on colony state and
    # colonist personality to choose an action
    return f"""
(let ((food-ratio (get state "food-ratio" 0.5))
      (power-ratio (get state "power-ratio" 0.5))
      (morale (get state "morale" 0.5))
      (threat (get state "threat-level" 0.0))
      (year (get state "year" 1)))
  (cond
    ((< food-ratio 0.3)
     (if (> {sk.get('hydroponics', 0.0)} 0.5) "farm"
       (if (> {s.get('hoarding', 0.0)} 0.6) "hoard" "farm")))
    ((< power-ratio 0.3)
     (if (> {sk.get('coding', 0.0)} 0.5) "repair" "build"))
    ((> threat 0.7)
     (if (> {s.get('paranoia', 0.0)} 0.6) "sabotage"
       (if (> {s.get('resolve', 0.0)} 0.7) "repair" "rest")))
    ((= (mod year 5) 0) "propose-governance")
    ((< morale 0.3)
     (if (> {s.get('empathy', 0.0)} 0.6) "mediate"
       (if (> {s.get('faith', 0.0)} 0.5) "pray" "rest")))
    ((> morale 0.7)
     (if (> {s.get('improvisation', 0.0)} 0.7) "explore"
       (if (> {sk.get('terraforming', 0.0)} 0.6) "farm" "research")))
    (else
     (if (> (random) 0.7) "explore"
       (if (> {sk.get('mediation', 0.0)} 0.5) "mediate" "research")))))
"""


def _build_sub_sim_policy(colonist: dict, proposal: str) -> str:
    """Build a LisPy sub-sim expression to model a governance proposal.

    Uses a budget-limited sub-sim to project outcomes.
    """
    return f"""
(sub-sim 500
  (let ((stability (+ 50 (randint -10 20)))
        (satisfaction (+ 50 (randint -5 15)))
        (efficiency (+ 50 (randint -8 12))))
    (dict "proposal" "{proposal}"
          "stability" (clamp stability 0 100)
          "satisfaction" (clamp satisfaction 0 100)
          "efficiency" (clamp efficiency 0 100)
          "viable" (and (> stability 30) (> satisfaction 30))
          "depth" (get (dict) "_sub_sim_depth" 1))))
"""


# ---------------------------------------------------------------------------
# Colony state
# ---------------------------------------------------------------------------

@dataclass
class ColonistState:
    """Mutable state for a single colonist."""
    id: str
    name: str
    element: str
    role: str
    stats: dict[str, float]
    skills: dict[str, float]
    alive: bool = True
    health: float = 1.0
    morale: float = 0.7
    years_alive: int = 0
    actions_taken: list[str] = field(default_factory=list)
    diary: list[str] = field(default_factory=list)
    votes: dict[str, int] = field(default_factory=dict)  # proposal -> +1/-1
    proposals_made: list[dict] = field(default_factory=list)
    policy: str = ""
    sub_sim_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id, "name": self.name, "element": self.element,
            "role": self.role, "stats": dict(self.stats),
            "skills": dict(self.skills), "alive": self.alive,
            "health": round(self.health, 3), "morale": round(self.morale, 3),
            "years_alive": self.years_alive,
            "actions_taken": self.actions_taken[-20:],  # keep last 20
            "diary": self.diary[-10:],  # keep last 10 entries
            "votes": dict(self.votes),
            "proposals_made": self.proposals_made[-5:],
            "policy": self.policy,
            "sub_sim_log": self.sub_sim_log[-5:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ColonistState:
        """Deserialize from dict."""
        return cls(
            id=d["id"], name=d["name"], element=d["element"],
            role=d["role"], stats=d["stats"], skills=d["skills"],
            alive=d.get("alive", True), health=d.get("health", 1.0),
            morale=d.get("morale", 0.7),
            years_alive=d.get("years_alive", 0),
            actions_taken=d.get("actions_taken", []),
            diary=d.get("diary", []),
            votes=d.get("votes", {}),
            proposals_made=d.get("proposals_made", []),
            policy=d.get("policy", ""),
            sub_sim_log=d.get("sub_sim_log", []),
        )


@dataclass
class GovernanceRecord:
    """A governance proposal and its outcome."""
    year: int
    proposer: str
    proposal: str
    votes_for: int = 0
    votes_against: int = 0
    adopted: bool = False
    sub_sim_evidence: dict | None = None

    def to_dict(self) -> dict:
        return {
            "year": self.year, "proposer": self.proposer,
            "proposal": self.proposal, "votes_for": self.votes_for,
            "votes_against": self.votes_against, "adopted": self.adopted,
            "sub_sim_evidence": self.sub_sim_evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GovernanceRecord:
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


class Colony:
    """The Mars-100 colony simulation.

    Manages 10 colonists, resources, environment, governance,
    and the LisPy evaluation engine for decision-making.
    """

    def __init__(self, state: dict) -> None:
        self.seed = state.get("seed", 42)
        self.year = state.get("year", 0)
        self.resources = state.get("resources", {
            "food": 1500.0, "water": 1200.0, "power": 1000.0,
            "oxygen": 1200.0, "materials": 600.0,
        })
        self.environment = state.get("environment", {
            "temperature_c": -60.0, "pressure_kpa": 0.636,
            "dust_opacity": 0.3, "terraform_progress": 0.0,
        })
        self.colonists: dict[str, ColonistState] = {}
        for cdata in state.get("colonists", []):
            c = ColonistState.from_dict(cdata)
            self.colonists[c.id] = c
        self.relationships: dict[str, dict[str, float]] = state.get(
            "relationships", {})
        self.governance: list[GovernanceRecord] = [
            GovernanceRecord.from_dict(g)
            for g in state.get("governance", [])
        ]
        self.archived_souls: list[dict] = state.get("archived_souls", [])
        self.year_deltas: list[dict] = state.get("year_deltas", [])
        self.factions: dict[str, list[str]] = state.get("factions", {})
        self.leader: str | None = state.get("leader", None)
        self.governance_type: str | None = state.get("governance_type", None)
        self.meta_insights: list[dict] = state.get("meta_insights", [])

    @classmethod
    def genesis(cls, seed: int = 42) -> Colony:
        """Create a new colony at Year 0."""
        state: dict[str, Any] = {
            "seed": seed,
            "year": 0,
            "resources": {
                "food": 1500.0, "water": 1200.0, "power": 1000.0,
                "oxygen": 1200.0, "materials": 600.0,
            },
            "environment": {
                "temperature_c": -60.0, "pressure_kpa": 0.636,
                "dust_opacity": 0.3, "terraform_progress": 0.0,
            },
            "colonists": [],
            "relationships": {},
            "governance": [],
            "archived_souls": [],
            "year_deltas": [],
            "factions": {},
            "leader": None,
            "governance_type": None,
            "meta_insights": [],
        }

        # Initialize colonists
        rng = random.Random(seed)
        for cdef in FOUNDING_COLONISTS:
            c = ColonistState(
                id=cdef["id"], name=cdef["name"],
                element=cdef["element"], role=cdef["role"],
                stats=dict(cdef["stats"]), skills=dict(cdef["skills"]),
                morale=0.5 + rng.random() * 0.3,
                health=0.8 + rng.random() * 0.2,
            )
            c.policy = _build_policy(cdef)
            state["colonists"] = state.get("colonists", [])
            state["colonists"].append(c.to_dict())
            # Store the policy separately since to_dict doesn't include it
            state["colonists"][-1]["policy"] = c.policy

        # Initialize relationships (random initial noise)
        ids = [c["id"] for c in FOUNDING_COLONISTS]
        rels: dict[str, dict[str, float]] = {}
        for a in ids:
            rels[a] = {}
            for b in ids:
                if a != b:
                    rels[a][b] = round(rng.uniform(-0.2, 0.5), 3)
        state["relationships"] = rels

        return cls(state)

    def tick(self, year: int | None = None) -> dict:
        """Advance one Mars year. Returns a delta dict.

        Two-phase evaluation:
          Phase 1: all colonists observe the same snapshot and decide
          Phase 2: actions are resolved against the canonical state
        """
        if year is not None:
            self.year = year
        else:
            self.year += 1

        rng = random.Random(self.year * 31337 + self.seed)
        env_seed = self.year * 7 + self.seed

        # Generate environmental event
        event = self._generate_event(rng)

        # Apply environmental effects to resources
        self._apply_event_effects(event)

        # Take a snapshot for two-phase evaluation
        snapshot = self._build_state_snapshot(event)

        # Phase 1: all colonists decide based on snapshot
        decisions: dict[str, dict] = {}
        alive_ids = [cid for cid, c in self.colonists.items() if c.alive]
        for cid in alive_ids:
            colonist = self.colonists[cid]
            decision = self._colonist_decide(colonist, snapshot, env_seed, rng)
            decisions[cid] = decision

        # Phase 2: resolve actions
        action_results = self._resolve_actions(decisions, rng)

        # Update relationships based on interactions
        self._update_relationships(decisions, rng)

        # Check for governance proposals
        governance_events = self._process_governance(decisions, env_seed, rng)

        # Detect emergent factions
        self._detect_factions()

        # Check for deaths
        deaths = self._check_deaths(rng)

        # Update colonist state
        for cid in alive_ids:
            c = self.colonists[cid]
            if c.alive:
                c.years_alive += 1
                # Natural stat drift
                for stat in STAT_NAMES:
                    drift = rng.gauss(0, 0.02)
                    c.stats[stat] = max(0.0, min(1.0, c.stats[stat] + drift))

        # Resource regeneration
        self._regenerate_resources(rng)

        # Check for meta-insights (depth-3 sub-sim discoveries)
        self._check_meta_insights()

        # Build year delta (Dream Catcher protocol)
        delta = {
            "year": self.year,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "decisions": {cid: d["action"] for cid, d in decisions.items()},
            "action_results": action_results,
            "governance_events": [g.to_dict() for g in governance_events],
            "deaths": deaths,
            "resources": dict(self.resources),
            "environment": dict(self.environment),
            "population": len([c for c in self.colonists.values() if c.alive]),
            "factions": dict(self.factions),
            "leader": self.leader,
            "governance_type": self.governance_type,
            "diaries": {cid: self.colonists[cid].diary[-1]
                        for cid in alive_ids
                        if self.colonists[cid].diary},
            "summary": self._build_summary(event, decisions, deaths,
                                            governance_events),
        }
        self.year_deltas.append(delta)

        return delta

    def to_dict(self) -> dict:
        """Serialize full colony state."""
        return {
            "_meta": {
                "engine": "mars-100",
                "version": "1.0",
                "year": self.year,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "seed": self.seed,
            "year": self.year,
            "resources": dict(self.resources),
            "environment": dict(self.environment),
            "colonists": [c.to_dict() for c in self.colonists.values()],
            "relationships": {
                a: dict(bs) for a, bs in self.relationships.items()},
            "governance": [g.to_dict() for g in self.governance[-50:]],
            "archived_souls": self.archived_souls,
            "factions": dict(self.factions),
            "leader": self.leader,
            "governance_type": self.governance_type,
            "meta_insights": self.meta_insights,
            "year_deltas": self.year_deltas[-10:],  # keep recent deltas
        }

    @classmethod
    def from_dict(cls, state: dict) -> Colony:
        """Restore colony from serialized state."""
        return cls(state)

    # -- Internal: event generation ----------------------------------------

    def _generate_event(self, rng: random.Random) -> dict:
        """Generate a random environmental event for this year."""
        total_weight = sum(e["weight"] for e in EVENT_TYPES)
        roll = rng.random() * total_weight
        cumulative = 0
        chosen = EVENT_TYPES[0]
        for evt in EVENT_TYPES:
            cumulative += evt["weight"]
            if roll <= cumulative:
                chosen = evt
                break

        severity = rng.uniform(0.3, 1.0)
        # Special: alien signal only after year 40
        if chosen["name"] == "alien_signal" and self.year < 40:
            chosen = EVENT_TYPES[0]  # fallback to dust storm

        return {
            "name": chosen["name"],
            "severity": round(severity, 3),
            "food_mod": round(chosen["food_mod"] * severity, 3),
            "morale_mod": round(chosen["morale_mod"] * severity, 3),
            "power_mod": round(chosen["power_mod"] * severity, 3),
            "description": self._event_description(chosen["name"], severity),
        }

    @staticmethod
    def _event_description(name: str, severity: float) -> str:
        """Generate a human-readable event description."""
        intensity = "minor" if severity < 0.5 else "major" if severity < 0.8 else "catastrophic"
        descriptions = {
            "dust_storm": f"A {intensity} dust storm engulfs the colony",
            "resource_strike": f"A {intensity} mineral deposit discovered",
            "equipment_failure": f"A {intensity} equipment malfunction detected",
            "earth_contact": f"A {intensity} communication window with Earth opens",
            "solar_flare": f"A {intensity} solar flare hits the surface",
            "microbe_discovery": f"A {intensity} discovery — possible microbial life",
            "supply_ship": f"A supply ship arrives with {intensity} cargo",
            "alien_signal": f"An anomalous signal detected from Phobos",
        }
        return descriptions.get(name, f"A {intensity} event occurs")

    def _apply_event_effects(self, event: dict) -> None:
        """Apply environmental event effects to colony resources."""
        food_mod = event["food_mod"]
        power_mod = event["power_mod"]

        self.resources["food"] = max(0, self.resources["food"] *
                                     (1 + food_mod))
        self.resources["power"] = max(0, self.resources["power"] *
                                      (1 + power_mod))

        # Morale affects all colonists
        morale_mod = event["morale_mod"]
        for c in self.colonists.values():
            if c.alive:
                c.morale = max(0.0, min(1.0, c.morale + morale_mod))

    # -- Internal: colonist decisions --------------------------------------

    def _build_state_snapshot(self, event: dict) -> dict:
        """Build a LisPy-compatible state snapshot."""
        alive = [c for c in self.colonists.values() if c.alive]
        total_food = self.resources["food"]
        food_per_person = total_food / max(len(alive), 1)

        return {
            "year": self.year,
            "food-ratio": round(min(1.0, food_per_person / 60.0), 3),
            "power-ratio": round(min(1.0, self.resources["power"] / 600.0), 3),
            "water-ratio": round(min(1.0, self.resources["water"] / 800.0), 3),
            "morale": round(sum(c.morale for c in alive) / max(len(alive), 1), 3),
            "threat-level": round(max(0, 1.0 - (food_per_person / 60.0)), 3),
            "population": len(alive),
            "event": event["name"],
            "event-severity": event["severity"],
            "terraform-progress": round(self.environment["terraform_progress"], 4),
        }

    def _colonist_decide(self, colonist: ColonistState, snapshot: dict,
                         env_seed: int, rng: random.Random) -> dict:
        """Have a colonist decide an action using their LisPy policy."""
        try:
            env = standard_env(seed=env_seed + hash(colonist.id))
            env["state"] = snapshot
            result = lispy_evaluate(colonist.policy, env, step_limit=10000)
            action = str(result) if result else "rest"
            if action not in ACTION_NAMES:
                action = "rest"
        except LispError:
            # Fallback: personality-based heuristic
            action = self._fallback_decision(colonist, snapshot, rng)
        diary = self._generate_diary(colonist, action, snapshot)
        colonist.diary.append(diary)
        colonist.actions_taken.append(action)

        return {"action": action, "diary": diary}

    def _fallback_decision(self, colonist: ColonistState, snapshot: dict,
                           rng: random.Random) -> str:
        """Personality-based fallback when LisPy policy fails."""
        s = colonist.stats
        if snapshot["food-ratio"] < 0.3:
            return "farm" if colonist.skills.get("hydroponics", 0) > 0.5 else "hoard"
        if snapshot["power-ratio"] < 0.3:
            return "repair"
        if snapshot["morale"] < 0.3:
            return "mediate" if s.get("empathy", 0) > 0.5 else "pray"
        if snapshot["year"] % 5 == 0:
            return "propose-governance"
        choices = ["explore", "research", "build", "farm"]
        return rng.choice(choices)

    def _generate_diary(self, colonist: ColonistState, action: str,
                        snapshot: dict) -> str:
        """Generate a diary entry for a colonist's year."""
        year = snapshot["year"]
        event = snapshot["event"]
        templates = {
            "farm": f"Year {year}: Tended the greenhouses. {event} tested us.",
            "repair": f"Year {year}: Fixed critical systems after {event}.",
            "explore": f"Year {year}: Ventured beyond the hab. The {event} reminded me why.",
            "mediate": f"Year {year}: Calmed tensions among the crew after {event}.",
            "research": f"Year {year}: Made progress on the terraform model.",
            "hoard": f"Year {year}: Secured extra rations. Trust is thin.",
            "pray": f"Year {year}: Found solace in contemplation.",
            "sabotage": f"Year {year}: Took matters into my own hands.",
            "build": f"Year {year}: Expanded the habitat module.",
            "rest": f"Year {year}: Rested. The colony endures.",
            "propose-governance": f"Year {year}: Proposed a new way to govern ourselves.",
            "run-sub-sim": f"Year {year}: Ran a simulation to test a theory.",
            "trade": f"Year {year}: Negotiated resource exchanges.",
            "teach": f"Year {year}: Passed knowledge to the younger ones.",
        }
        return templates.get(action, f"Year {year}: Survived another year.")

    # -- Internal: action resolution ---------------------------------------

    def _resolve_actions(self, decisions: dict[str, dict],
                         rng: random.Random) -> list[dict]:
        """Phase 2: resolve all colonist actions against canonical state."""
        results: list[dict] = []
        for cid, decision in decisions.items():
            action = decision["action"]
            colonist = self.colonists[cid]
            effect = self._apply_action(colonist, action, rng)
            results.append({"colonist": cid, "action": action, **effect})
        return results

    def _apply_action(self, colonist: ColonistState, action: str,
                      rng: random.Random) -> dict:
        """Apply a single colonist action to colony state."""
        skill_bonus = 0.0
        effect: dict[str, Any] = {"success": True}

        if action == "farm":
            skill_bonus = colonist.skills.get("hydroponics", 0.3)
            produced = 20.0 * (0.5 + skill_bonus)
            self.resources["food"] += produced
            effect["food_produced"] = round(produced, 1)

        elif action == "repair":
            skill_bonus = colonist.skills.get("coding", 0.3)
            restored = 30.0 * (0.5 + skill_bonus)
            self.resources["power"] += restored
            effect["power_restored"] = round(restored, 1)

        elif action == "build":
            skill_bonus = colonist.skills.get("terraforming", 0.3)
            materials_used = 15.0
            if self.resources["materials"] >= materials_used:
                self.resources["materials"] -= materials_used
                self.environment["terraform_progress"] += 0.002 * (0.5 + skill_bonus)
                effect["materials_used"] = materials_used
            else:
                effect["success"] = False

        elif action == "explore":
            find_chance = 0.3 + colonist.stats.get("improvisation", 0.3) * 0.3
            if rng.random() < find_chance:
                bonus_type = rng.choice(["food", "water", "materials"])
                bonus = rng.uniform(10, 40)
                self.resources[bonus_type] += bonus
                effect["discovered"] = {bonus_type: round(bonus, 1)}

        elif action == "mediate":
            skill_bonus = colonist.skills.get("mediation", 0.3)
            morale_boost = 0.05 * (0.5 + skill_bonus)
            for c in self.colonists.values():
                if c.alive and c.id != colonist.id:
                    c.morale = min(1.0, c.morale + morale_boost)
            effect["morale_boost"] = round(morale_boost, 3)

        elif action == "hoard":
            stolen = min(20.0, self.resources["food"] * 0.05)
            self.resources["food"] -= stolen
            colonist.health = min(1.0, colonist.health + 0.05)
            effect["hoarded"] = round(stolen, 1)

        elif action == "sabotage":
            target_ids = [cid for cid in self.colonists
                          if cid != colonist.id and self.colonists[cid].alive]
            if target_ids:
                target = rng.choice(target_ids)
                damage = 0.1 * colonist.skills.get("sabotage", 0.3)
                self.colonists[target].health -= damage
                effect["target"] = target
                effect["damage"] = round(damage, 3)

        elif action == "pray":
            faith_bonus = colonist.stats.get("faith", 0.3) * 0.1
            colonist.morale = min(1.0, colonist.morale + faith_bonus)
            effect["morale_gain"] = round(faith_bonus, 3)

        elif action == "research":
            progress = 0.001 * (0.5 + colonist.skills.get("coding", 0.3))
            self.environment["terraform_progress"] += progress
            effect["research_progress"] = round(progress, 4)

        elif action == "trade":
            effect["traded"] = True

        elif action == "teach":
            # Boost a random alive colonist's weakest skill
            alive = [c for c in self.colonists.values()
                     if c.alive and c.id != colonist.id]
            if alive:
                student = rng.choice(alive)
                weakest = min(student.skills, key=lambda k: student.skills[k])
                boost = 0.03 * colonist.skills.get(weakest, 0.3)
                student.skills[weakest] = min(1.0, student.skills[weakest] + boost)
                effect["taught"] = {"student": student.id, "skill": weakest}

        # Resource consumption per colonist per year
        self.resources["food"] = max(0, self.resources["food"] - 15)
        self.resources["water"] = max(0, self.resources["water"] - 10)
        self.resources["oxygen"] = max(0, self.resources["oxygen"] - 8)
        self.resources["power"] = max(0, self.resources["power"] - 5)

        return effect

    # -- Internal: governance ----------------------------------------------

    def _process_governance(self, decisions: dict[str, dict],
                            env_seed: int,
                            rng: random.Random) -> list[GovernanceRecord]:
        """Process governance proposals from colonists."""
        proposals: list[GovernanceRecord] = []

        for cid, decision in decisions.items():
            if decision["action"] != "propose-governance":
                continue
            colonist = self.colonists[cid]

            # Generate proposal based on colony conditions
            proposal_type = self._generate_proposal_type(colonist, rng)
            proposal_text = f"{colonist.name} proposes: {proposal_type}"

            # Run sub-sim to evaluate proposal
            sub_sim_result = None
            try:
                policy = _build_sub_sim_policy(colonist.to_dict(), proposal_type)
                env = standard_env(seed=env_seed + hash(cid))
                env["state"] = {"year": self.year, "resources": dict(self.resources)}
                sub_sim_result = lispy_evaluate(policy, env, step_limit=5000)
                colonist.sub_sim_log.append({
                    "year": self.year, "proposal": proposal_type,
                    "result": sub_sim_result,
                })
            except LispError:
                sub_sim_result = {"viable": False, "reason": "sim_error"}

            # Vote on proposal
            record = GovernanceRecord(
                year=self.year, proposer=cid,
                proposal=proposal_type,
                sub_sim_evidence=sub_sim_result if isinstance(sub_sim_result, dict) else None,
            )

            alive = [c for c in self.colonists.values()
                     if c.alive and c.id != cid]
            for voter in alive:
                # Vote based on personality and relationship with proposer
                affinity = self.relationships.get(voter.id, {}).get(cid, 0.0)
                empathy = voter.stats.get("empathy", 0.5)
                paranoia = voter.stats.get("paranoia", 0.5)
                vote_score = affinity + empathy * 0.3 - paranoia * 0.2
                if sub_sim_result and isinstance(sub_sim_result, dict):
                    if sub_sim_result.get("viable"):
                        vote_score += 0.2
                if vote_score + rng.gauss(0, 0.15) > 0.2:
                    record.votes_for += 1
                    voter.votes[proposal_type] = 1
                else:
                    record.votes_against += 1
                    voter.votes[proposal_type] = -1

            # Adopted if majority votes for
            total_voters = record.votes_for + record.votes_against
            record.adopted = (record.votes_for > total_voters / 2
                              if total_voters > 0 else False)

            if record.adopted:
                self._apply_governance(proposal_type, cid)

            proposals.append(record)
            colonist.proposals_made.append(record.to_dict())
            self.governance.append(record)

        return proposals

    def _generate_proposal_type(self, colonist: ColonistState,
                                rng: random.Random) -> str:
        """Generate a governance proposal based on colony needs."""
        food_low = self.resources["food"] < 200
        power_low = self.resources["power"] < 200
        morale_low = any(c.morale < 0.3 for c in self.colonists.values()
                         if c.alive)

        proposals = []
        if food_low:
            proposals.extend(["ration_food_equally", "mandatory_farming"])
        if power_low:
            proposals.extend(["power_conservation", "expand_solar"])
        if morale_low:
            proposals.extend(["weekly_assembly", "rotate_leadership"])
        if colonist.stats.get("paranoia", 0) > 0.6:
            proposals.extend(["exile_vote", "surveillance_system"])
        if colonist.stats.get("faith", 0) > 0.6:
            proposals.extend(["meditation_mandate", "shared_ritual"])
        if colonist.stats.get("empathy", 0) > 0.7:
            proposals.extend(["consensus_rule", "care_rotation"])

        # Always have fallback proposals
        proposals.extend(["elect_council", "direct_democracy", "meritocracy"])

        return rng.choice(proposals)

    def _apply_governance(self, proposal_type: str, proposer: str) -> None:
        """Apply an adopted governance change."""
        if proposal_type in ("elect_council", "direct_democracy", "meritocracy",
                             "consensus_rule", "rotate_leadership"):
            self.governance_type = proposal_type
        if proposal_type == "elect_council":
            # Find colonist with most votes_for in history
            vote_counts: dict[str, int] = {}
            for g in self.governance:
                if g.adopted:
                    vote_counts[g.proposer] = vote_counts.get(g.proposer, 0) + g.votes_for
            if vote_counts:
                self.leader = max(vote_counts, key=lambda k: vote_counts[k])
        elif proposal_type == "rotate_leadership":
            alive_ids = [c.id for c in self.colonists.values() if c.alive]
            if alive_ids:
                self.leader = alive_ids[self.year % len(alive_ids)]

    # -- Internal: relationships -------------------------------------------

    def _update_relationships(self, decisions: dict[str, dict],
                              rng: random.Random) -> None:
        """Update colonist relationships based on this year's actions."""
        alive_ids = [cid for cid in decisions]

        for a in alive_ids:
            for b in alive_ids:
                if a == b:
                    continue
                if a not in self.relationships:
                    self.relationships[a] = {}

                current = self.relationships[a].get(b, 0.0)
                action_a = decisions[a]["action"]
                action_b = decisions[b]["action"]

                # Same action = slight bonding
                if action_a == action_b:
                    current += 0.02
                # Mediate boosts relationships
                if action_a == "mediate":
                    current += 0.03
                # Sabotage damages relationships
                if action_a == "sabotage":
                    current -= 0.05
                # Hoarding damages trust
                if action_a == "hoard":
                    current -= 0.02
                # Natural decay toward neutral
                current *= 0.98
                # Random noise
                current += rng.gauss(0, 0.01)

                self.relationships[a][b] = round(
                    max(-1.0, min(1.0, current)), 3)

    # -- Internal: factions ------------------------------------------------

    def _detect_factions(self) -> None:
        """Detect emergent factions from relationship patterns."""
        alive = [c for c in self.colonists.values() if c.alive]
        if len(alive) < 4:
            return

        # Simple faction detection: cluster by positive relationships
        factions: dict[str, list[str]] = {}
        assigned: set[str] = set()

        for c in alive:
            if c.id in assigned:
                continue
            faction_id = f"faction-{c.element}"
            members = [c.id]
            assigned.add(c.id)

            for other in alive:
                if other.id in assigned:
                    continue
                affinity = self.relationships.get(c.id, {}).get(other.id, 0.0)
                reverse = self.relationships.get(other.id, {}).get(c.id, 0.0)
                if (affinity + reverse) / 2 > 0.15:
                    members.append(other.id)
                    assigned.add(other.id)

            if len(members) >= 2:
                factions[faction_id] = members

        self.factions = factions

    # -- Internal: deaths --------------------------------------------------

    def _check_deaths(self, rng: random.Random) -> list[dict]:
        """Check for colonist deaths. Returns list of death records."""
        deaths: list[dict] = []

        for cid, c in list(self.colonists.items()):
            if not c.alive:
                continue

            death_chance = 0.0
            cause = None

            # Starvation
            if self.resources["food"] < 50:
                death_chance += 0.15
                cause = "starvation"
            # Health failure
            if c.health < 0.1:
                death_chance += 0.3
                cause = cause or "health_failure"
            # Old age (after 60 Mars years)
            if c.years_alive > 60:
                death_chance += (c.years_alive - 60) * 0.02
                cause = cause or "old_age"
            # Accident
            death_chance += 0.005
            cause = cause or "accident"

            if rng.random() < death_chance:
                c.alive = False
                death_record = {
                    "colonist": cid, "name": c.name,
                    "year": self.year, "cause": cause,
                    "years_alive": c.years_alive,
                    "final_diary": c.diary[-1] if c.diary else "",
                }
                deaths.append(death_record)
                # Archive soul (legacy, not delete)
                self.archived_souls.append({
                    **c.to_dict(),
                    "death_year": self.year,
                    "death_cause": cause,
                    "epitaph": f"{c.name} ({c.element}), {c.role}. "
                               f"Survived {c.years_alive} Mars years.",
                })
                # Remove from relationships
                for other_id in self.relationships:
                    self.relationships[other_id].pop(cid, None)
                self.relationships.pop(cid, None)

        return deaths

    # -- Internal: resources -----------------------------------------------

    def _regenerate_resources(self, rng: random.Random) -> None:
        """Natural resource regeneration per year."""
        alive_count = len([c for c in self.colonists.values() if c.alive])

        # Greenhouse food production
        terraform = self.environment["terraform_progress"]
        food_regen = 40.0 * (1 + terraform * 2) * max(1, alive_count * 0.3)
        self.resources["food"] += food_regen

        # Water recycling
        water_regen = 25.0 * (1 + terraform)
        self.resources["water"] += water_regen

        # Solar power
        power_regen = 35.0 * (1 - self.environment["dust_opacity"] * 0.5)
        self.resources["power"] += power_regen

        # Oxygen from plants
        o2_regen = 20.0 * (1 + terraform * 3)
        self.resources["oxygen"] += o2_regen

        # Terraform progress slowly
        self.environment["terraform_progress"] = min(
            1.0, self.environment["terraform_progress"] + 0.001 * alive_count)

        # Cap resources
        for key in self.resources:
            self.resources[key] = round(min(2000.0, max(0.0, self.resources[key])), 1)

    # -- Internal: meta-insights -------------------------------------------

    def _check_meta_insights(self) -> None:
        """Check if any sub-sim produced an insight worth promoting."""
        for c in self.colonists.values():
            if not c.alive:
                continue
            for log_entry in c.sub_sim_log:
                result = log_entry.get("result")
                if not isinstance(result, dict):
                    continue
                depth = result.get("depth", 0)
                if depth >= 2 and result.get("viable"):
                    stability = result.get("stability", 0)
                    satisfaction = result.get("satisfaction", 0)
                    if stability > 70 and satisfaction > 70:
                        insight = {
                            "year": self.year,
                            "colonist": c.id,
                            "proposal": log_entry.get("proposal", ""),
                            "evidence": result,
                            "recommended_amendment": (
                                f"The Mars-100 colony's recursive simulation at "
                                f"depth {depth} found that "
                                f"'{log_entry.get('proposal', 'unknown')}' "
                                f"produces stability={stability}%, "
                                f"satisfaction={satisfaction}%. "
                                f"Consider adopting this as a Rappterbook "
                                f"constitutional principle."
                            ),
                        }
                        if not any(m.get("proposal") == insight["proposal"]
                                   for m in self.meta_insights):
                            self.meta_insights.append(insight)

    # -- Internal: summary -------------------------------------------------

    def _build_summary(self, event: dict, decisions: dict[str, dict],
                       deaths: list[dict],
                       governance: list[GovernanceRecord]) -> str:
        """Build a human-readable summary of the year."""
        alive = len([c for c in self.colonists.values() if c.alive])
        parts = [
            f"Year {self.year}: {event['description']} (severity {event['severity']}).",
            f"Population: {alive}/10 alive.",
            f"Resources: food={self.resources['food']:.0f}, "
            f"power={self.resources['power']:.0f}, "
            f"water={self.resources['water']:.0f}.",
        ]
        if deaths:
            names = ", ".join(d["name"] for d in deaths)
            parts.append(f"Lost: {names}.")
        if governance:
            for g in governance:
                status = "ADOPTED" if g.adopted else "REJECTED"
                parts.append(
                    f"Governance: '{g.proposal}' by {g.proposer} — "
                    f"{status} ({g.votes_for}-{g.votes_against}).")
        if self.governance_type:
            parts.append(f"Current system: {self.governance_type}.")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(years: int = 100, seed: int = 42,
                   output_dir: str | None = None) -> dict:
    """Run the full Mars-100 simulation.

    Args:
        years: number of Mars years to simulate (default 100).
        seed: RNG seed for reproducibility.
        output_dir: if set, write per-year JSON files here.

    Returns:
        Final colony state as dict.
    """
    colony = Colony.genesis(seed=seed)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    for year in range(1, years + 1):
        delta = colony.tick(year)

        if output_dir:
            year_path = Path(output_dir) / f"year-{year:03d}.json"
            year_path.write_text(json.dumps(delta, indent=2))

        # Check for colony collapse
        alive = len([c for c in colony.colonists.values() if c.alive])
        if alive == 0:
            break

    final_state = colony.to_dict()

    if output_dir:
        state_path = Path(output_dir) / "final_state.json"
        state_path.write_text(json.dumps(final_state, indent=2))

    return final_state


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for Mars-100 simulation."""
    import sys

    years = 100
    seed = 42
    output_dir = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--years" and i + 1 < len(args):
            years = int(args[i + 1])
            i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            i += 1

    print(f"Mars-100: Simulating {years} years (seed={seed})")
    state = run_simulation(years=years, seed=seed, output_dir=output_dir)

    alive = len([c for c in state.get("colonists", []) if c.get("alive")])
    total = len(state.get("colonists", []))
    souls = len(state.get("archived_souls", []))
    gov = state.get("governance_type", "none")
    insights = len(state.get("meta_insights", []))

    print(f"Simulation complete: {alive}/{total} alive, {souls} archived souls")
    print(f"Governance: {gov}, Meta-insights: {insights}")
    print(f"Terraform progress: {state['environment']['terraform_progress']:.1%}")


if __name__ == "__main__":
    main()
