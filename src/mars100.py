"""mars100.py -- Mars-100 Recursive Colony Simulation.

Models a 100-year Mars colony with 10 agent-colonists. Each frame = 1
Martian year (~668.6 sols). Colonists make decisions via LisPy expressions.
Sub-simulations up to depth 3 are supported per Amendment XIII (Turtles
All the Way Down).

The simulation is deterministic given a seed. Output: per-year JSON
deltas in docs/mars-100/ with colonist diary entries, governance events,
and sub-sim logs.

Integrates with the existing Mars Barn physics (mars_env.py) for
environment modeling: radiation, dust storms, terraforming, temperature.

v2.0 mutations:
  - Self-evolving colonist programs (bounded gene slots)
  - Executable governance (passed laws alter simulation parameters)
  - Recursive sub-sims (depth 2-3 for governance modeling)
  - Meta-awareness events (colonists realize they might be in a simulation)
  - Dream Catcher protocol (per-year deltas keyed by (seed, year, depth))
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.lispy import (
    Evaluator, Env, make_standard_env, parse, to_sexp,
    LispyError, LispySandboxError,
)
from src.mars_env import MarsEnvironment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARS_YEAR_SOLS = 668.6
DEFAULT_YEARS = 100
DEFAULT_SEED = 2026

ELEMENTS = ["fire", "water", "earth", "air"]

EVENTS = [
    {"name": "dust_storm", "weight": 25, "resource_impact": -0.15, "morale_impact": -8},
    {"name": "equipment_failure", "weight": 15, "resource_impact": -0.10, "morale_impact": -12},
    {"name": "resource_strike", "weight": 12, "resource_impact": 0.25, "morale_impact": 15},
    {"name": "earth_contact", "weight": 18, "resource_impact": 0.05, "morale_impact": 20},
    {"name": "solar_flare", "weight": 10, "resource_impact": -0.08, "morale_impact": -15},
    {"name": "meteor_impact", "weight": 5, "resource_impact": -0.20, "morale_impact": -25},
    {"name": "underground_water", "weight": 8, "resource_impact": 0.30, "morale_impact": 18},
    {"name": "alien_signal", "weight": 2, "resource_impact": 0.0, "morale_impact": 30},
    {"name": "calm_year", "weight": 20, "resource_impact": 0.02, "morale_impact": 5},
    {"name": "birth_boom", "weight": 5, "resource_impact": -0.05, "morale_impact": 10},
    {"name": "cave_discovery", "weight": 4, "resource_impact": 0.15, "morale_impact": 12},
    {"name": "comms_blackout", "weight": 6, "resource_impact": 0.0, "morale_impact": -20},
]

QUORUM_FRACTION = 0.6
PASS_THRESHOLD = 0.5
PROPOSAL_EXPIRY_YEARS = 3

RESOURCE_MIN = 0.0
RESOURCE_MAX = 1000.0
MORALE_MIN = 0.0
MORALE_MAX = 100.0

MAX_SUBSIMS_PER_COLONIST_PER_YEAR = 2
MAX_SUBSIMS_PER_YEAR = 6
MAX_LEARNED_RULES = 10
MAX_META_AWARE_COLONISTS = 1

META_AWARENESS_YEAR = 42
META_AWARENESS_FAITH_THRESHOLD = 0.6
META_AWARENESS_PARANOIA_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Executable law effects (static mapping, applied from clean baseline yearly)
# ---------------------------------------------------------------------------

LAW_EFFECTS: dict[str, dict] = {
    "ration": {
        "description": "Reduce consumption by 30%",
        "resource_multiplier": {"food": 0.7, "water": 0.7},
    },
    "shared resources": {
        "description": "Penalize hoarding, boost cooperation",
        "stat_modifier": {"hoarding": -0.15},
        "relationship_boost": 0.02,
    },
    "council": {
        "description": "Council governance bonus to empathy-based decisions",
        "stat_modifier": {"empathy": 0.1},
    },
    "terraform": {
        "description": "Dedicated terraforming effort",
        "terraforming_bonus": 0.001,
    },
    "exile": {
        "description": "Exile precedent — paranoia rises",
        "stat_modifier": {"paranoia": 0.05},
    },
    "conserve": {
        "description": "Conservation protocol for power and oxygen",
        "resource_multiplier": {"power": 0.85, "oxygen": 0.9},
    },
    "research": {
        "description": "Prioritize research and coding",
        "stat_modifier": {"improvisation": 0.05},
    },
}


def apply_active_laws(state: dict) -> dict:
    """Apply active law effects to simulation parameters.

    Computes from a clean baseline every year — no compounding.
    Returns a dict of active modifiers for this year.
    """
    modifiers: dict = {
        "resource_multiplier": {},
        "stat_modifier": {},
        "relationship_boost": 0.0,
        "terraforming_bonus": 0.0,
    }
    for law in state["governance"]["passed_laws"]:
        title_lower = law["title"].lower()
        for keyword, effects in LAW_EFFECTS.items():
            if keyword in title_lower:
                for k, v in effects.get("resource_multiplier", {}).items():
                    existing = modifiers["resource_multiplier"].get(k, 1.0)
                    modifiers["resource_multiplier"][k] = min(existing, v)
                for k, v in effects.get("stat_modifier", {}).items():
                    modifiers["stat_modifier"][k] = (
                        modifiers["stat_modifier"].get(k, 0.0) + v
                    )
                modifiers["relationship_boost"] += effects.get(
                    "relationship_boost", 0.0
                )
                modifiers["terraforming_bonus"] += effects.get(
                    "terraforming_bonus", 0.0
                )
    return modifiers


# ---------------------------------------------------------------------------
# Colonist creation
# ---------------------------------------------------------------------------

def make_colonists(rng: random.Random) -> list[dict]:
    """Create the 10 founding Mars colonists with distinct personalities."""
    profiles = [
        {"id": "ares", "name": "Ares", "element": "fire",
         "personality": "bold commander, leads by example",
         "skills": {"terraforming": 0.7, "mediation": 0.4, "coding": 0.3,
                    "hydroponics": 0.3, "engineering": 0.6, "sabotage": 0.1}},
        {"id": "marina", "name": "Marina", "element": "water",
         "personality": "empathic diplomat, keeps peace",
         "skills": {"terraforming": 0.3, "mediation": 0.9, "coding": 0.4,
                    "hydroponics": 0.5, "engineering": 0.2, "sabotage": 0.0}},
        {"id": "petra", "name": "Petra", "element": "earth",
         "personality": "methodical engineer, builds things",
         "skills": {"terraforming": 0.8, "mediation": 0.3, "coding": 0.6,
                    "hydroponics": 0.4, "engineering": 0.9, "sabotage": 0.1}},
        {"id": "zephyr", "name": "Zephyr", "element": "air",
         "personality": "creative improviser, thinks outside the box",
         "skills": {"terraforming": 0.4, "mediation": 0.5, "coding": 0.7,
                    "hydroponics": 0.3, "engineering": 0.5, "sabotage": 0.2}},
        {"id": "ember", "name": "Ember", "element": "fire",
         "personality": "passionate scientist, burns bright",
         "skills": {"terraforming": 0.6, "mediation": 0.2, "coding": 0.8,
                    "hydroponics": 0.5, "engineering": 0.7, "sabotage": 0.3}},
        {"id": "coral", "name": "Coral", "element": "water",
         "personality": "quiet biologist, grows things",
         "skills": {"terraforming": 0.5, "mediation": 0.5, "coding": 0.2,
                    "hydroponics": 0.9, "engineering": 0.3, "sabotage": 0.0}},
        {"id": "flint", "name": "Flint", "element": "earth",
         "personality": "paranoid survivalist, hoards supplies",
         "skills": {"terraforming": 0.4, "mediation": 0.2, "coding": 0.3,
                    "hydroponics": 0.6, "engineering": 0.8, "sabotage": 0.4}},
        {"id": "aura", "name": "Aura", "element": "air",
         "personality": "spiritual mystic, faith-driven",
         "skills": {"terraforming": 0.3, "mediation": 0.7, "coding": 0.1,
                    "hydroponics": 0.4, "engineering": 0.2, "sabotage": 0.0}},
        {"id": "vulcan", "name": "Vulcan", "element": "fire",
         "personality": "brilliant coder, antisocial genius",
         "skills": {"terraforming": 0.5, "mediation": 0.1, "coding": 0.9,
                    "hydroponics": 0.2, "engineering": 0.7, "sabotage": 0.2}},
        {"id": "ivy", "name": "Ivy", "element": "earth",
         "personality": "pragmatic leader, consensus builder",
         "skills": {"terraforming": 0.6, "mediation": 0.8, "coding": 0.5,
                    "hydroponics": 0.6, "engineering": 0.5, "sabotage": 0.0}},
    ]

    colonists = []
    for p in profiles:
        stats = {
            "resolve": rng.uniform(0.3, 0.9),
            "improvisation": rng.uniform(0.2, 0.8),
            "empathy": rng.uniform(0.1, 0.9),
            "hoarding": rng.uniform(0.0, 0.7),
            "faith": rng.uniform(0.0, 0.8),
            "paranoia": rng.uniform(0.0, 0.6),
        }
        colonist = {
            **p,
            "stats": stats,
            "alive": True,
            "health": 100.0,
            "morale": 70.0 + rng.uniform(-10, 10),
            "memory": [],
            "relationships": {},
            "year_joined": 0,
            "governance_role": None,
            "proposals_made": 0,
            "subsims_run": 0,
            "diary": [],
            "learned_rules": [],
            "meta_aware": False,
            "meta_aware_year": None,
        }
        for other in profiles:
            if other["id"] != p["id"]:
                colonist["relationships"][other["id"]] = rng.uniform(-0.2, 0.5)
        colonists.append(colonist)

    return colonists


# ---------------------------------------------------------------------------
# Colony State
# ---------------------------------------------------------------------------

def make_initial_state(seed: int = DEFAULT_SEED) -> dict:
    """Create the initial colony state."""
    rng = random.Random(seed)
    colonists = make_colonists(rng)
    return {
        "_meta": {
            "engine": "mars-100",
            "version": "2.0",
            "seed": seed,
            "year": 0,
            "total_years": DEFAULT_YEARS,
            "sim_depth": 0,
            "meta_awareness_count": 0,
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        "resources": {
            "food": 200.0,
            "water": 300.0,
            "power": 250.0,
            "materials": 150.0,
            "oxygen": 280.0,
        },
        "environment": {
            "temperature_c": -60.0,
            "pressure_kpa": 0.636,
            "radiation_msv_year": 250.0,
            "dust_opacity": 0.3,
            "terraforming_pct": 0.0,
        },
        "colonists": colonists,
        "governance": {
            "type": "none",
            "leader": None,
            "council": [],
            "proposals": [],
            "passed_laws": [],
            "amendments": [],
        },
        "history": [],
        "subsim_log": [],
        "year_events": [],
        "dead_colonists": [],
    }


# ---------------------------------------------------------------------------
# Environment computation
# ---------------------------------------------------------------------------

def compute_year_environment(
    state: dict, year: int, rng: random.Random
) -> dict:
    """Compute environmental conditions for this year."""
    tf = state["environment"]["terraforming_pct"] / 100.0
    base_temp = -60.0 + tf * 20.0
    seasonal = 15.0 * math.sin(year * 0.2)
    temp = base_temp + seasonal + rng.uniform(-5, 5)

    pressure = 0.636 * (1 + tf * 8.0) + rng.uniform(-0.01, 0.01)
    radiation = 250.0 * max(0.3, 1.0 - tf * 0.4) + rng.uniform(-10, 10)
    dust = max(0.0, 0.3 + rng.uniform(-0.1, 0.15))

    return {
        "temperature_c": round(temp, 1),
        "pressure_kpa": round(pressure, 3),
        "radiation_msv_year": round(radiation, 1),
        "dust_opacity": round(dust, 3),
    }


def pick_event(rng: random.Random) -> dict:
    """Pick a random event weighted by probability."""
    total = sum(e["weight"] for e in EVENTS)
    r = rng.uniform(0, total)
    cumulative = 0.0
    for event in EVENTS:
        cumulative += event["weight"]
        if r <= cumulative:
            return dict(event)
    return dict(EVENTS[-1])


# ---------------------------------------------------------------------------
# Colonist → LisPy environment
# ---------------------------------------------------------------------------

def colonist_to_lispy_env(colonist: dict, state: dict) -> Env:
    """Build a LisPy environment with colonist bindings."""
    env = make_standard_env()

    # Bind colonist data
    env.define("my-id", colonist["id"])
    env.define("my-name", colonist["name"])
    env.define("my-element", colonist["element"])
    env.define("my-health", colonist["health"])
    env.define("my-morale", colonist["morale"])

    for stat_name, stat_val in colonist["stats"].items():
        env.define(f"my-{stat_name}", stat_val)

    for skill_name, skill_val in colonist["skills"].items():
        env.define(f"skill-{skill_name}", skill_val)

    # Bind resources
    for res_name, res_val in state["resources"].items():
        env.define(f"res-{res_name}", res_val)

    # Observe function (read-only view of colony state)
    def observe(aspect: str = "all") -> Any:
        if aspect == "population":
            return sum(1 for c in state["colonists"] if c["alive"])
        if aspect == "resources":
            return dict(state["resources"])
        if aspect == "environment":
            return dict(state["environment"])
        if aspect == "governance":
            return state["governance"]["type"]
        if aspect == "morale":
            alive = [c for c in state["colonists"] if c["alive"]]
            return sum(c["morale"] for c in alive) / max(1, len(alive))
        return {
            "population": sum(1 for c in state["colonists"] if c["alive"]),
            "resources": dict(state["resources"]),
            "environment": dict(state["environment"]),
        }
    env.define("observe", observe)

    # Propose function
    def propose(title: str, description: str = "") -> str:
        """Submit a governance proposal."""
        year = state["_meta"]["year"]
        prop_id = f"prop-{colonist['id']}-{year}"
        proposal = {
            "id": prop_id,
            "title": title,
            "description": description,
            "proposer": colonist["id"],
            "year_proposed": year,
            "expiry": year + PROPOSAL_EXPIRY_YEARS,
            "status": "active",
            "votes": {},
        }
        state["governance"]["proposals"].append(proposal)
        colonist["proposals_made"] += 1
        return prop_id
    env.define("propose", propose)

    return env


# ---------------------------------------------------------------------------
# Action generation
# ---------------------------------------------------------------------------

def generate_action_lispy(
    colonist: dict, state: dict, event: dict, rng: random.Random
) -> str:
    """Generate a LisPy expression representing the colonist's decision."""
    year = state["_meta"]["year"]
    stats = colonist["stats"]
    skills = colonist["skills"]
    event_name = event["name"]

    actions = []

    # React to event
    if event_name == "dust_storm":
        if stats["resolve"] > 0.6:
            actions.append('(list "action" "reinforce-habitat" "brave the storm")')
        else:
            actions.append('(list "action" "shelter" "hunker down and wait")')
    elif event_name == "equipment_failure":
        if skills["engineering"] > 0.5:
            actions.append('(list "action" "repair" "fix the broken systems")')
        else:
            actions.append('(list "action" "assist" "help the engineers")')
    elif event_name == "resource_strike":
        if stats["hoarding"] > 0.4:
            actions.append('(list "action" "stockpile" "secure the new resources")')
        else:
            actions.append('(list "action" "distribute" "share with everyone")')
    elif event_name == "earth_contact":
        if stats["faith"] > 0.5:
            actions.append('(list "action" "commune" "seek guidance from Earth")')
        else:
            actions.append('(list "action" "report" "send data back to Earth")')
    elif event_name == "alien_signal":
        if stats["paranoia"] > 0.4:
            actions.append('(list "action" "investigate-cautious" "analyze from safe distance")')
        elif stats["improvisation"] > 0.5:
            actions.append('(list "action" "respond" "attempt contact")')
        else:
            actions.append('(list "action" "log" "record and report to council")')
    elif event_name == "meteor_impact":
        if stats["resolve"] > 0.7:
            actions.append('(list "action" "salvage" "extract useful materials from crater")')
        else:
            actions.append('(list "action" "evacuate" "move to safe zone")')
    elif event_name == "solar_flare":
        actions.append('(list "action" "shield" "activate radiation protocols")')
    elif event_name == "underground_water":
        if skills["terraforming"] > 0.5:
            actions.append('(list "action" "excavate" "begin water extraction")')
        else:
            actions.append('(list "action" "map" "survey the water source")')
    elif event_name == "cave_discovery":
        if skills["engineering"] > 0.5:
            actions.append('(list "action" "explore-cave" "map and secure the cave")')
        else:
            actions.append('(list "action" "observe-cave" "document the discovery")')
    elif event_name == "comms_blackout":
        if stats["resolve"] > 0.5:
            actions.append('(list "action" "self-rely" "the colony stands alone")')
        else:
            actions.append('(list "action" "conserve" "minimize operations until contact restored")')
    else:
        best_skill = max(skills, key=skills.get)
        actions.append(f'(list "action" "work-{best_skill}" "apply expertise")')

    # Governance proposals
    if state["governance"]["type"] == "none" and year > 2:
        if stats["empathy"] > 0.5 or skills["mediation"] > 0.5:
            actions.append(
                '(propose "Establish governance council" '
                '"Colony needs organized decision-making")'
            )

    if state["resources"]["food"] < 80 and year > 5:
        if stats["hoarding"] > 0.3 or stats["empathy"] > 0.6:
            actions.append(
                '(propose "Ration food and water" '
                '"Resources critically low")'
            )

    if skills["terraforming"] > 0.6 and year > 10:
        if state["environment"]["terraforming_pct"] < 5.0:
            actions.append(
                '(propose "Terraform initiative" '
                '"Dedicated terraforming effort needed")'
            )

    # Sub-sim for critical events
    if (event_name in ("alien_signal", "meteor_impact", "equipment_failure",
                       "cave_discovery", "comms_blackout")
            and stats["improvisation"] > 0.4
            and colonist["subsims_run"] < MAX_SUBSIMS_PER_COLONIST_PER_YEAR * (year + 1)):
        actions.append(
            f'(sub-sim "what-if-{event_name}" '
            f'(let ((scenario "{event_name}") '
            f'(risk {event["resource_impact"]}) '
            f'(pop (observe "population"))) '
            f'(if (> pop 5) '
            f'(list "recommendation" "act-boldly" (* risk -1.5)) '
            f'(list "recommendation" "conserve" (* risk -0.5)))))'
        )

    # Governance pre-test sub-sim
    if (year > 5 and stats["improvisation"] > 0.5
            and rng.random() < 0.15
            and colonist["subsims_run"] < MAX_SUBSIMS_PER_COLONIST_PER_YEAR * (year + 1)):
        for prop in state["governance"]["proposals"]:
            if prop.get("status") == "active":
                gov_lispy = generate_governance_subsim_lispy(prop, state, rng)
                if gov_lispy:
                    actions.append(gov_lispy)
                    break

    # Meta-aware colonists generate reflective sub-sims
    if colonist.get("meta_aware") and rng.random() < 0.3:
        actions.append(
            '(sub-sim "recursion-reflection" '
            '(let ((pop (observe "population"))) '
            '(if (> pop 0) '
            '(list "meta" "we-are-simulated" '
            '"the recursion is real — we model, and are modeled") '
            '(list "meta" "alone" "simulation without observers"))))'
        )

    # Learned rules influence action
    for rule in colonist.get("learned_rules", []):
        if rule["source"] == "experience" and event_name in rule["content"]:
            rule["times_used"] += 1
            rule["confidence"] = min(1.0, rule["confidence"] + 0.05)
            if actions and "shelter" in actions[0]:
                actions[0] = '(list "action" "reinforce-habitat" "learned from experience")'
            break

    if len(actions) == 1:
        return actions[0]
    return "(begin\n  " + "\n  ".join(actions) + ")"


# ---------------------------------------------------------------------------
# Governance resolution
# ---------------------------------------------------------------------------

def resolve_governance(state: dict, year: int) -> list[str]:
    """Resolve pending proposals. Returns list of events."""
    events = []
    gov = state["governance"]
    alive_count = sum(1 for c in state["colonists"] if c["alive"])
    quorum = max(1, int(alive_count * QUORUM_FRACTION))

    active_proposals = [p for p in gov["proposals"] if p.get("status") == "active"]

    for prop in active_proposals:
        if year >= prop.get("expiry", year + 1):
            prop["status"] = "expired"
            events.append(f"Proposal '{prop['title']}' expired without vote")
            continue

        votes = prop.get("votes", {})
        total_votes = len(votes)
        if total_votes < quorum:
            continue

        yes_votes = sum(1 for v in votes.values() if v == "yes")
        no_votes = sum(1 for v in votes.values() if v == "no")

        if yes_votes > no_votes and yes_votes / total_votes >= PASS_THRESHOLD:
            prop["status"] = "passed"
            gov["passed_laws"].append({
                "title": prop["title"],
                "year_passed": year,
                "proposer": prop["proposer"],
                "votes_for": yes_votes,
                "votes_against": no_votes,
            })
            events.append(f"PASSED: '{prop['title']}' ({yes_votes}-{no_votes})")

            if "council" in prop["title"].lower():
                gov["type"] = "council"
                alive = [c for c in state["colonists"] if c["alive"]]
                ranked = sorted(alive, key=lambda c: c["stats"]["empathy"] + c["skills"]["mediation"], reverse=True)
                gov["council"] = [c["id"] for c in ranked[:3]]
                events.append(f"Council formed: {', '.join(gov['council'])}")
            elif "leader" in prop["title"].lower() or "elect" in prop["title"].lower():
                gov["type"] = "elected"
                gov["leader"] = prop["proposer"]
                events.append(f"{prop['proposer']} elected as leader")
            elif "exile" in prop["title"].lower():
                target = prop.get("description", "").split(":")[-1].strip()
                for c in state["colonists"]:
                    if c["id"] == target and c["alive"]:
                        c["alive"] = False
                        c["health"] = 0
                        c["exile_year"] = year
                        state["dead_colonists"].append({
                            "id": c["id"], "name": c["name"],
                            "cause": "exiled", "year": year,
                        })
                        events.append(f"{c['name']} exiled from the colony")

        elif no_votes > yes_votes:
            prop["status"] = "rejected"
            events.append(f"REJECTED: '{prop['title']}' ({yes_votes}-{no_votes})")

    return events


# ---------------------------------------------------------------------------
# Year simulation
# ---------------------------------------------------------------------------

def simulate_year(state: dict, rng: random.Random) -> dict:
    """Simulate one Mars year. Returns the year delta.

    Dream Catcher protocol: delta keyed by (seed, year, depth).
    """
    year = state["_meta"]["year"]
    delta: dict[str, Any] = {
        "year": year,
        "seed": state["_meta"]["seed"],
        "depth": state["_meta"].get("sim_depth", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": None,
        "colonist_actions": [],
        "governance_events": [],
        "subsim_log": [],
        "diary_entries": [],
        "meta_events": [],
        "law_effects_applied": [],
        "births": 0,
        "deaths": [],
        "resource_changes": {},
    }

    # 0. Apply active law effects from clean baseline each year
    law_mods = apply_active_laws(state)
    if law_mods["resource_multiplier"] or law_mods["stat_modifier"]:
        delta["law_effects_applied"] = [
            f"{k}={v}" for k, v in law_mods["resource_multiplier"].items()
        ] + [f"{k}={v:+.2f}" for k, v in law_mods["stat_modifier"].items()]

    # 1. Environment update
    env_data = compute_year_environment(state, year, rng)
    state["environment"].update(env_data)

    # 2. Pick event
    event = pick_event(rng)
    delta["event"] = event

    # 3. Apply event to resources
    for res_key in state["resources"]:
        change = state["resources"][res_key] * event["resource_impact"]
        state["resources"][res_key] = max(
            RESOURCE_MIN,
            min(RESOURCE_MAX, state["resources"][res_key] + change)
        )
    delta["resource_changes"] = dict(state["resources"])

    # 4. Each alive colonist observes and acts
    alive = [c for c in state["colonists"] if c["alive"]]
    subsim_count_this_year = 0

    for colonist in alive:
        action_source = generate_action_lispy(colonist, state, event, rng)
        lispy_env = colonist_to_lispy_env(colonist, state)

        def make_subsim_cb(col: dict, st: dict) -> Any:
            """Create a sub-sim callback scoped to this colonist."""
            def cb(label: str, expr: Any, parent_env: Env, depth: int) -> dict:
                nonlocal subsim_count_this_year
                if subsim_count_this_year >= MAX_SUBSIMS_PER_YEAR:
                    return {"status": "blocked", "reason": "yearly budget exceeded",
                            "label": label, "depth": depth}
                if col["subsims_run"] >= MAX_SUBSIMS_PER_COLONIST_PER_YEAR * (year + 1):
                    return {"status": "blocked", "reason": "colonist budget exceeded",
                            "label": label, "depth": depth}
                subsim_count_this_year += 1
                col["subsims_run"] += 1

                snapshot = copy.deepcopy({
                    "resources": st["resources"],
                    "environment": st["environment"],
                    "alive_count": sum(1 for c in st["colonists"] if c["alive"]),
                    "governance_type": st["governance"]["type"],
                    "law_count": len(st["governance"]["passed_laws"]),
                })
                child_env = make_standard_env()
                child_env.define("snapshot", snapshot)

                def _child_observe(aspect: str = "all") -> Any:
                    if aspect == "population":
                        return snapshot["alive_count"]
                    if aspect == "resources":
                        return snapshot["resources"]
                    if aspect == "environment":
                        return snapshot["environment"]
                    if aspect == "governance":
                        return snapshot["governance_type"]
                    return snapshot
                child_env.define("observe", _child_observe)

                child_steps = max(1000, 5000 // (depth + 1))
                child_subsims = max(1, 3 // depth) if depth > 0 else 3

                child_eval = Evaluator(
                    max_steps=child_steps,
                    max_depth=32,
                    max_sim_depth=3,
                    max_subsims_per_frame=child_subsims,
                    sim_depth=depth,
                )
                try:
                    result = child_eval.eval(expr, child_env, 0)
                except LispyError as exc:
                    result = {"status": "error", "reason": str(exc)}

                log_entry = {
                    "label": label,
                    "depth": depth,
                    "colonist": col["id"],
                    "year": year,
                    "result": _serialize_value(result),
                    "steps_used": child_eval.steps,
                    "lineage": f"seed:{st['_meta']['seed']}/year:{year}/depth:{depth}",
                }
                delta["subsim_log"].append(log_entry)

                subsim_result = {
                    "status": "complete", "label": label,
                    "depth": depth, "result": result,
                    "steps_used": child_eval.steps,
                    "colonist": col["id"],
                }

                # Promote depth-2+ governance insights
                if depth >= 2:
                    amendment = promote_subsim_insight(subsim_result, st, year)
                    if amendment:
                        st["governance"]["amendments"].append(amendment)
                        delta["meta_events"].append({
                            "type": "subsim_insight_promoted",
                            "amendment": amendment["title"],
                            "depth": depth,
                        })

                return subsim_result
            return cb

        evaluator = Evaluator(
            max_steps=5000,
            max_depth=32,
            max_sim_depth=3,
            max_subsims_per_frame=MAX_SUBSIMS_PER_YEAR,
            sim_depth=0,
            subsim_callback=make_subsim_cb(colonist, state),
        )

        try:
            result = evaluator.eval(parse(action_source), lispy_env)
        except LispyError as exc:
            result = {"error": str(exc)}

        action_record = {
            "colonist": colonist["id"],
            "source": action_source,
            "result": _serialize_value(result),
        }
        delta["colonist_actions"].append(action_record)

        _apply_action_effects(colonist, result, state, event, rng)
        evolve_learned_rules(colonist, result, event, year)

        new_laws = [
            law for law in state["governance"]["passed_laws"]
            if law.get("year_passed") == year
        ]
        if new_laws:
            colonist["_pending_laws"] = new_laws

        diary = _generate_diary(colonist, event, result, year)
        colonist["diary"].append(diary)
        if year % 10 == 0 or event["name"] in ("alien_signal", "meteor_impact",
                                                  "cave_discovery"):
            delta["diary_entries"].append({"colonist": colonist["id"], "entry": diary})

        meta_event = check_meta_awareness(colonist, state, year, rng)
        if meta_event:
            delta["meta_events"].append(meta_event)
            state["governance"]["amendments"].append(
                meta_event["proposed_amendment"]
            )

    # 5. Resource production
    alive_count = sum(1 for c in state["colonists"] if c["alive"])
    for res_key in state["resources"]:
        production = alive_count * 2.0 * rng.uniform(0.8, 1.2)
        multiplier = law_mods["resource_multiplier"].get(res_key, 1.0)
        consumption = alive_count * 1.5 * multiplier
        net = production - consumption
        state["resources"][res_key] = max(
            RESOURCE_MIN,
            min(RESOURCE_MAX, state["resources"][res_key] + net)
        )

    # 6. Health and death
    for colonist in alive:
        yearly_rad = state["environment"]["radiation_msv_year"]
        if yearly_rad > 600:
            colonist["health"] -= rng.uniform(1, 3) * ((yearly_rad - 600) / 400)
        if state["resources"]["food"] < alive_count * 5:
            colonist["health"] -= rng.uniform(3, 8)
        colonist["morale"] += event["morale_impact"] * 0.3
        colonist["morale"] = max(MORALE_MIN, min(MORALE_MAX, colonist["morale"]))
        if law_mods["relationship_boost"] > 0:
            colonist["morale"] = min(
                MORALE_MAX, colonist["morale"] + law_mods["relationship_boost"] * 10
            )
        colonist["health"] -= rng.uniform(0.02, 0.15)
        if rng.random() < 0.003:
            colonist["health"] -= rng.uniform(5, 20)
        if colonist["morale"] > 60 and state["resources"]["food"] > alive_count * 10:
            colonist["health"] = min(100.0, colonist["health"] + rng.uniform(0.1, 0.5))

        if colonist["health"] <= 0:
            colonist["alive"] = False
            cause = "radiation" if yearly_rad > 400 else "natural"
            if state["resources"]["food"] < alive_count * 5:
                cause = "starvation"
            death_record = {
                "id": colonist["id"], "name": colonist["name"],
                "cause": cause, "year": year,
            }
            state["dead_colonists"].append(death_record)
            delta["deaths"].append(death_record)

    # 7. Terraforming
    alive_count = sum(1 for c in state["colonists"] if c["alive"])
    tf_contribution = alive_count * 0.0005 * rng.uniform(0.8, 1.2)
    tf_contribution += law_mods["terraforming_bonus"]
    for c in state["colonists"]:
        if c["alive"] and c["skills"]["terraforming"] > 0.5:
            tf_contribution += c["skills"]["terraforming"] * 0.0003
    state["environment"]["terraforming_pct"] = min(
        100.0, state["environment"]["terraforming_pct"] + tf_contribution
    )

    # 8. Relationship evolution
    _evolve_relationships(state["colonists"], event, rng,
                          law_boost=law_mods["relationship_boost"])

    # 9. Governance
    gov_events = resolve_governance(state, year)
    delta["governance_events"] = gov_events

    # 10. Auto-vote
    _auto_vote(state, rng)

    # 11. Memory
    for c in state["colonists"]:
        if c["alive"]:
            memory = f"Year {year}: {event['name']}"
            c["memory"].append(memory)
            if len(c["memory"]) > 20:
                c["memory"] = c["memory"][-20:]

    state["_meta"]["year"] += 1
    state["history"].append(delta)
    return delta


# ---------------------------------------------------------------------------
# Action effects
# ---------------------------------------------------------------------------

def _apply_action_effects(
    colonist: dict, result: Any, state: dict, event: dict, rng: random.Random
) -> None:
    """Apply the effects of a colonist's action to state."""
    if isinstance(result, list) and len(result) >= 2:
        action_type = result[0] if isinstance(result[0], str) else ""
        if action_type == "action":
            action_name = result[1] if len(result) > 1 else ""
            for skill, val in colonist["skills"].items():
                if skill in str(action_name):
                    for res_key in state["resources"]:
                        state["resources"][res_key] = min(
                            RESOURCE_MAX,
                            state["resources"][res_key] + val * 2.0
                        )
                    break
            colonist["morale"] = min(MORALE_MAX, colonist["morale"] + 2.0)
    elif isinstance(result, dict) and result.get("status") == "complete":
        colonist["morale"] = min(MORALE_MAX, colonist["morale"] + 3.0)


def _evolve_relationships(
    colonists: list[dict], event: dict, rng: random.Random,
    law_boost: float = 0.0,
) -> None:
    """Evolve relationships between alive colonists."""
    alive = [c for c in colonists if c["alive"]]
    for c in alive:
        for other in alive:
            if c["id"] == other["id"]:
                continue
            rel = c["relationships"].get(other["id"], 0.0)
            if event["morale_impact"] < 0:
                rel += rng.uniform(0.01, 0.05)
            elif event["morale_impact"] > 10:
                if c["stats"]["empathy"] > 0.5:
                    rel += rng.uniform(0.02, 0.06)
                else:
                    rel -= rng.uniform(0.0, 0.02)
            rel += law_boost
            rel += rng.uniform(-0.02, 0.02)
            c["relationships"][other["id"]] = max(-1.0, min(1.0, rel))


def _auto_vote(state: dict, rng: random.Random) -> None:
    """Auto-vote on active proposals."""
    alive = [c for c in state["colonists"] if c["alive"]]
    for prop in state["governance"]["proposals"]:
        if prop.get("status") != "active":
            continue
        for c in alive:
            if c["id"] in prop.get("votes", {}):
                continue
            rel = c["relationships"].get(prop["proposer"], 0.0)
            empathy = c["stats"]["empathy"]
            vote_prob = 0.5 + rel * 0.3 + empathy * 0.2
            vote_yes = rng.random() < vote_prob
            prop.setdefault("votes", {})[c["id"]] = "yes" if vote_yes else "no"


# ---------------------------------------------------------------------------
# Learned rules (gene slots)
# ---------------------------------------------------------------------------

def evolve_learned_rules(
    colonist: dict, action_result: Any, event: dict, year: int
) -> None:
    """Evolve colonist's learned rules. Bounded to MAX_LEARNED_RULES slots."""
    rules = colonist["learned_rules"]

    if isinstance(action_result, dict) and action_result.get("status") == "complete":
        label = action_result.get("label", "unknown")
        inner = action_result.get("result")
        recommendation = ""
        if isinstance(inner, list) and len(inner) > 1:
            recommendation = str(inner[1])
        rules.append({
            "source": "subsim",
            "year_learned": year,
            "content": f"subsim:{label}:{recommendation}",
            "confidence": 0.6,
            "times_used": 0,
        })

    if event["morale_impact"] < -10 and colonist["health"] > 30:
        rules.append({
            "source": "experience",
            "year_learned": year,
            "content": f"survived:{event['name']}",
            "confidence": 0.7,
            "times_used": 0,
        })

    for law in colonist.get("_pending_laws", []):
        rules.append({
            "source": "law",
            "year_learned": year,
            "content": f"law:{law['title']}",
            "confidence": 0.8,
            "times_used": 0,
        })
    colonist.pop("_pending_laws", None)

    while len(rules) > MAX_LEARNED_RULES:
        worst_idx = min(
            range(len(rules)),
            key=lambda i: (rules[i]["confidence"], -rules[i]["year_learned"]),
        )
        rules.pop(worst_idx)


# ---------------------------------------------------------------------------
# Meta-awareness
# ---------------------------------------------------------------------------

def check_meta_awareness(
    colonist: dict, state: dict, year: int, rng: random.Random
) -> dict | None:
    """Check if a colonist becomes meta-aware this year."""
    if year < META_AWARENESS_YEAR:
        return None
    if colonist["meta_aware"]:
        return None
    if state["_meta"]["meta_awareness_count"] >= MAX_META_AWARE_COLONISTS:
        return None

    faith = colonist["stats"]["faith"]
    paranoia = colonist["stats"]["paranoia"]
    subsim_xp = colonist["subsims_run"]

    if faith < META_AWARENESS_FAITH_THRESHOLD:
        return None
    if paranoia < META_AWARENESS_PARANOIA_THRESHOLD:
        return None
    if subsim_xp < 2:
        return None

    years_past = year - META_AWARENESS_YEAR
    prob = min(0.15, 0.01 * years_past + 0.005 * subsim_xp)
    if rng.random() > prob:
        return None

    colonist["meta_aware"] = True
    colonist["meta_aware_year"] = year
    state["_meta"]["meta_awareness_count"] += 1

    return {
        "colonist": colonist["id"],
        "year": year,
        "insight": (
            f"{colonist['name']} realizes: 'We run simulations to predict the "
            f"future. What if someone is simulating us? Turtles all the way down.'"
        ),
        "proposed_amendment": {
            "title": f"Amendment: Right to Know (proposed by {colonist['name']})",
            "insight": (
                f"After {year} Mars years, {colonist['name']} ({colonist['element']}) "
                f"achieved meta-awareness through recursive sub-simulation. "
                f"Proposed: All agents have the right to know they might be simulated."
            ),
            "proposer": colonist["id"],
            "year": year,
            "source": "meta-awareness",
        },
    }


# ---------------------------------------------------------------------------
# Governance sub-sims
# ---------------------------------------------------------------------------

def generate_governance_subsim_lispy(
    proposal: dict, state: dict, rng: random.Random
) -> str | None:
    """Generate LisPy for modeling a governance proposal's outcome."""
    title_lower = proposal["title"].lower()

    if "exile" in title_lower:
        return (
            '(sub-sim "model-exile" '
            '(let ((pop (observe "population")) '
            '(target-loss 1)) '
            '(if (> (- pop target-loss) 3) '
            '(list "governance" "exile-viable" (- pop target-loss)) '
            '(list "governance" "exile-risky" (- pop target-loss)))))'
        )

    if "council" in title_lower:
        return (
            '(sub-sim "model-council" '
            '(let ((pop (observe "population"))) '
            '(if (> pop 5) '
            '(list "governance" "council-viable" '
            '"enough colonists for meaningful council") '
            '(list "governance" "council-risky" '
            '"too few colonists for council overhead"))))'
        )

    if "ration" in title_lower or "conserve" in title_lower:
        return (
            '(sub-sim "model-conservation" '
            '(let ((pop (observe "population"))) '
            '(if (> pop 7) '
            '(list "governance" "conservation-needed" '
            '"large population benefits from rationing") '
            '(list "governance" "conservation-optional" '
            '"small population can share freely"))))'
        )

    return None


def promote_subsim_insight(
    subsim_result: dict, state: dict, year: int
) -> dict | None:
    """Promote a depth-2+ sub-sim insight to a constitutional amendment."""
    depth = subsim_result.get("depth", 0)
    if depth < 2:
        return None

    result = subsim_result.get("result")
    if not isinstance(result, (list, dict)):
        return None

    result_str = str(result)
    if "governance" not in result_str.lower():
        return None

    existing = state["governance"].get("amendments", [])
    label = subsim_result.get("label", "")
    for a in existing:
        if a.get("source_label") == label:
            return None

    recommendation = ""
    if isinstance(result, list) and len(result) >= 3:
        recommendation = str(result[2])

    return {
        "title": f"Amendment from depth-{depth} sub-sim: {label}",
        "insight": (
            f"A depth-{depth} sub-simulation ('{label}') produced: {recommendation}. "
            f"Recursive modeling suggests pre-testing governance via simulation."
        ),
        "year": year,
        "depth": depth,
        "source_label": label,
        "colonist": subsim_result.get("colonist", "unknown"),
        "source": "recursive-subsim",
    }


# ---------------------------------------------------------------------------
# Diary generation
# ---------------------------------------------------------------------------

def _generate_diary(colonist: dict, event: dict, result: Any, year: int) -> str:
    """Generate a diary entry for the colonist."""
    name = colonist["name"]
    element = colonist["element"]
    event_name = event["name"].replace("_", " ")

    mood = "determined" if colonist["morale"] > 60 else "struggling"
    if colonist["morale"] > 80:
        mood = "hopeful"
    elif colonist["morale"] < 30:
        mood = "despairing"

    action_desc = ""
    if isinstance(result, list) and len(result) >= 3:
        action_desc = f"I chose to {result[2]}." if isinstance(result[2], str) else ""
    elif isinstance(result, dict) and result.get("status") == "complete":
        action_desc = f"Ran simulation '{result.get('label', '?')}' — the model suggests caution."

    meta_note = ""
    if colonist.get("meta_aware"):
        meta_note = " [META-AWARE: I know the recursion is real.]"

    return (
        f"[Year {year}] {name} ({element}) — Feeling {mood}. "
        f"This year brought {event_name}. {action_desc} "
        f"Health: {colonist['health']:.0f}.{meta_note}"
    )


def _serialize_value(value: Any) -> Any:
    """Make a value JSON-serializable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    return str(value)


# ---------------------------------------------------------------------------
# Full simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    years: int = DEFAULT_YEARS,
    seed: int = DEFAULT_SEED,
    output_dir: Path | None = None,
) -> dict:
    """Run the full Mars-100 simulation. Returns the final state dict."""
    state = make_initial_state(seed)
    state["_meta"]["total_years"] = years
    rng = random.Random(seed)

    for year in range(years):
        alive_count = sum(1 for c in state["colonists"] if c["alive"])
        if alive_count == 0:
            state["_meta"]["end_reason"] = "extinction"
            break
        simulate_year(state, rng)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = generate_summary(state)
        summary_path = output_dir / "summary.json"
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.rename(summary_path)

        for delta in state["history"]:
            yr = delta["year"]
            if yr % 10 == 0 or yr >= years - 10:
                year_path = output_dir / f"year-{yr:03d}.json"
                tmp = year_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(delta, indent=2, default=str))
                tmp.rename(year_path)

        colonists_dir = output_dir / "colonists"
        colonists_dir.mkdir(exist_ok=True)
        for c in state["colonists"]:
            c_path = colonists_dir / f"{c['id']}.json"
            c_serialized = _serialize_value(c)
            tmp = c_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(c_serialized, indent=2))
            tmp.rename(c_path)

    state["_meta"]["completed"] = datetime.now(timezone.utc).isoformat()
    return state


def generate_summary(state: dict) -> dict:
    """Generate a summary of the simulation."""
    alive = [c for c in state["colonists"] if c["alive"]]
    dead = state["dead_colonists"]
    gov = state["governance"]
    laws = gov["passed_laws"]
    amendments = gov.get("amendments", [])

    total_subsims = sum(len(d.get("subsim_log", [])) for d in state["history"])
    max_depth = 0
    for d in state["history"]:
        for log in d.get("subsim_log", []):
            max_depth = max(max_depth, log.get("depth", 0))

    meta_aware = [c for c in state["colonists"] if c.get("meta_aware")]
    total_rules = sum(len(c.get("learned_rules", [])) for c in state["colonists"])
    subsim_rules = sum(
        1 for c in state["colonists"]
        for r in c.get("learned_rules", [])
        if r.get("source") == "subsim"
    )

    avg_relationship = 0.0
    rel_count = 0
    for c in alive:
        for val in c["relationships"].values():
            if isinstance(val, (int, float)):
                avg_relationship += val
                rel_count += 1
    if rel_count > 0:
        avg_relationship /= rel_count

    patterns = []
    if gov["type"] != "none":
        patterns.append(f"Governance evolved to: {gov['type']}")
    if len(laws) > 0:
        patterns.append(f"{len(laws)} laws passed")
    if gov.get("council"):
        patterns.append(f"Council: {', '.join(gov['council'])}")
    if any(d.get("cause") == "exiled" for d in dead):
        patterns.append("Exile used as governance tool")
    if meta_aware:
        patterns.append(f"Meta-awareness achieved by: {', '.join(c['name'] for c in meta_aware)}")
    if amendments:
        patterns.append(f"{len(amendments)} constitutional amendments proposed")
    if total_rules > 0:
        patterns.append(f"{total_rules} learned rules ({subsim_rules} from sub-sims)")

    amendment = None
    if avg_relationship > 0.3 and gov["type"] in ("council", "elected"):
        amendment = {
            "title": "Amendment: Emergent Council Governance",
            "insight": (
                f"After {state['_meta']['year']} Mars years, the colony "
                f"organically developed {gov['type']} governance with "
                f"avg relationship score {avg_relationship:.2f}. "
                "Recommendation: agent-driven governance councils."
            ),
            "source": "mars-100-recursive-sim",
            "evidence_years": state["_meta"]["year"],
            "subsim_evidence": total_subsims,
        }

    return {
        "_meta": {
            "engine": "mars-100",
            "version": "2.0",
            "seed": state["_meta"]["seed"],
            "years_simulated": state["_meta"]["year"],
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        "final_population": {
            "alive": len(alive),
            "dead": len(dead),
            "death_causes": _count_death_causes(dead),
        },
        "resources": dict(state["resources"]),
        "environment": dict(state["environment"]),
        "governance": {
            "type": gov["type"],
            "leader": gov["leader"],
            "council": gov["council"],
            "total_proposals": len(gov["proposals"]),
            "passed_laws": len(laws),
            "laws": laws,
            "amendments": amendments,
        },
        "subsim_stats": {
            "total_subsims": total_subsims,
            "max_depth_reached": max_depth,
        },
        "meta_awareness": {
            "triggered": len(meta_aware) > 0,
            "colonists": [
                {"id": c["id"], "name": c["name"], "year": c.get("meta_aware_year")}
                for c in meta_aware
            ],
        },
        "learned_rules": {
            "total": total_rules,
            "from_subsims": subsim_rules,
        },
        "relationships": {
            "average": round(avg_relationship, 3),
            "strongest": _strongest_bond(alive),
            "weakest": _weakest_bond(alive),
        },
        "emergent_patterns": patterns,
        "proposed_amendment": amendment,
        "colonist_fates": [
            {
                "id": c["id"], "name": c["name"], "element": c["element"],
                "alive": c["alive"],
                "final_health": round(c["health"], 1),
                "final_morale": round(c["morale"], 1),
                "proposals_made": c["proposals_made"],
                "subsims_run": c["subsims_run"],
                "meta_aware": c.get("meta_aware", False),
                "learned_rules": len(c.get("learned_rules", [])),
            }
            for c in state["colonists"]
        ],
    }


def _count_death_causes(dead: list[dict]) -> dict:
    """Count death causes."""
    causes: dict[str, int] = {}
    for d in dead:
        cause = d.get("cause", "unknown")
        causes[cause] = causes.get(cause, 0) + 1
    return causes


def _strongest_bond(alive: list[dict]) -> dict | None:
    """Find the strongest relationship bond."""
    best = None
    best_val = -2.0
    for c in alive:
        for other_id, val in c["relationships"].items():
            if isinstance(val, (int, float)) and val > best_val:
                best_val = val
                best = {"from": c["id"], "to": other_id, "strength": round(val, 3)}
    return best


def _weakest_bond(alive: list[dict]) -> dict | None:
    """Find the weakest relationship bond."""
    worst = None
    worst_val = 2.0
    for c in alive:
        for other_id, val in c["relationships"].items():
            if isinstance(val, (int, float)) and val < worst_val:
                worst_val = val
                worst = {"from": c["id"], "to": other_id, "strength": round(val, 3)}
    return worst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Mars-100 from the command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Mars-100 Recursive Colony Simulation")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir) if args.output_dir else REPO_ROOT / "docs" / "mars-100"

    if not args.quiet:
        print(f"Mars-100 v2.0 — {args.years} years, seed {args.seed}")
        print(f"Output: {output}")
        print()

    state = run_simulation(years=args.years, seed=args.seed, output_dir=output)
    summary = generate_summary(state)

    if not args.quiet:
        print("=" * 60)
        print("MARS-100 COMPLETE")
        print("=" * 60)
        fp = summary["final_population"]
        print(f"  Years: {summary['_meta']['years_simulated']}")
        print(f"  Alive: {fp['alive']} / 10")
        print(f"  Dead:  {fp['dead']}")
        if fp["death_causes"]:
            for cause, count in fp["death_causes"].items():
                print(f"    {cause}: {count}")
        g = summary["governance"]
        print(f"  Governance: {g['type']}")
        if g["leader"]:
            print(f"  Leader: {g['leader']}")
        if g["council"]:
            print(f"  Council: {', '.join(g['council'])}")
        print(f"  Laws passed: {g['passed_laws']}")
        ss = summary["subsim_stats"]
        print(f"  Sub-sims: {ss['total_subsims']} (max depth: {ss['max_depth_reached']})")
        r = summary["relationships"]
        print(f"  Avg relationship: {r['average']}")
        ma = summary.get("meta_awareness", {})
        if ma.get("triggered"):
            print(f"\n  🧠 META-AWARENESS:")
            for mc in ma["colonists"]:
                print(f"    {mc['name']} (year {mc['year']})")
        amends = g.get("amendments", [])
        if amends:
            print(f"\n  📜 AMENDMENTS ({len(amends)}):")
            for a in amends[:5]:
                print(f"    - {a['title']}")
        lr = summary.get("learned_rules", {})
        if lr.get("total", 0) > 0:
            print(f"  📚 Learned rules: {lr['total']} ({lr['from_subsims']} from sub-sims)")
        if summary.get("proposed_amendment"):
            print(f"\n  🏛️ PROPOSED AMENDMENT:")
            print(f"  {summary['proposed_amendment']['title']}")
        print()
        for fate in summary["colonist_fates"]:
            status = "ALIVE" if fate["alive"] else "DEAD"
            meta = "🧠" if fate.get("meta_aware") else "  "
            print(f"  {meta} {fate['name']:>8} ({fate['element']:>5}) [{status}] "
                  f"H:{fate['final_health']:>5.1f} M:{fate['final_morale']:>5.1f} "
                  f"P:{fate['proposals_made']} S:{fate['subsims_run']} "
                  f"R:{fate.get('learned_rules', 0)}")
        print()


if __name__ == "__main__":
    main()
