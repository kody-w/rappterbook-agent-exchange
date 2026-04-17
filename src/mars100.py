"""
Mars-100 — a recursive colony simulation with 10 agent-colonists.

Each frame = 1 Martian year (~687 Earth days).
Colonists have stats, skills, elements, relationships.
They observe events, deliberate via LisPy expressions, and
may spawn sub-simulations (up to depth 3) to model governance
proposals before committing.

State is deterministic given a base seed. RNG is derived per
(base_seed, year, colonist_id) for reproducibility.
"""
from __future__ import annotations

import copy
import math
import random as _random_mod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.lispy import (
    Budget,
    DepthLimitExceeded,
    Env,
    LispError,
    Symbol,
    make_env,
    run,
    run_in_env,
    to_sexp,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELEMENTS = ["fire", "water", "earth", "air"]
STAT_NAMES = ["resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia"]
SKILL_NAMES = ["terraforming", "hydroponics", "mediation", "coding", "prayer", "sabotage"]

EVENTS = [
    {"id": "dust_storm",       "severity": 0.6, "desc": "A massive dust storm engulfs the colony"},
    {"id": "resource_strike",  "severity": 0.3, "desc": "Mineral deposits discovered nearby"},
    {"id": "equipment_failure","severity": 0.7, "desc": "Critical life support malfunction"},
    {"id": "earth_contact",    "severity": 0.1, "desc": "Radio contact with Earth restored"},
    {"id": "alien_signal",     "severity": 0.5, "desc": "Anomalous signal detected from subsurface"},
    {"id": "solar_flare",      "severity": 0.8, "desc": "Intense solar radiation event"},
    {"id": "ice_discovery",    "severity": 0.2, "desc": "Underground ice reservoir found"},
    {"id": "hab_breach",       "severity": 0.9, "desc": "Habitat pressure breach detected"},
    {"id": "crop_blight",      "severity": 0.5, "desc": "Fungal infection in hydroponics bay"},
    {"id": "meteorite",        "severity": 0.7, "desc": "Meteorite impact near settlement"},
    {"id": "geothermal_vent",  "severity": 0.2, "desc": "Geothermal energy source discovered"},
    {"id": "dust_devil",       "severity": 0.3, "desc": "Dust devil damages solar panels"},
    {"id": "aurora",           "severity": 0.0, "desc": "Rare Martian aurora illuminates the sky"},
    {"id": "supply_ship",      "severity": 0.1, "desc": "Supply ship arrives from Earth"},
    {"id": "cave_system",      "severity": 0.2, "desc": "Lava tube cave system mapped"},
]

GOVERNANCE_TYPES = [
    "exile", "ration_allocation", "leadership_election",
    "resource_priority", "expansion_vote", "research_directive",
    "emergency_powers", "constitution_amendment",
]

COLONIST_NAMES = [
    "Kaelen", "Mira", "Rho", "Vesper", "Thorn",
    "Lux", "Sage", "Ember", "Drift", "Echo",
    "Zara", "Orion", "Cinder", "Moss", "Volt",
]

INITIAL_RESOURCES = {
    "food": 1200.0,
    "water": 1500.0,
    "power": 500.0,
    "oxygen": 1000.0,
    "materials": 600.0,
    "morale": 70.0,
}


# ---------------------------------------------------------------------------
# Colonist
# ---------------------------------------------------------------------------

def create_colonist(colonist_id: int, rng: _random_mod.Random) -> dict:
    """Create a colonist with randomized stats, skills, and element."""
    name = COLONIST_NAMES[colonist_id % len(COLONIST_NAMES)]
    element = ELEMENTS[rng.randint(0, 3)]
    stats = {s: rng.randint(20, 80) for s in STAT_NAMES}
    skills = {s: rng.randint(10, 60) for s in SKILL_NAMES}

    if element == "fire":
        stats["resolve"] = min(100, stats["resolve"] + 15)
        skills["sabotage"] = min(100, skills["sabotage"] + 10)
    elif element == "water":
        stats["empathy"] = min(100, stats["empathy"] + 15)
        skills["mediation"] = min(100, skills["mediation"] + 10)
    elif element == "earth":
        stats["hoarding"] = min(100, stats["hoarding"] + 15)
        skills["terraforming"] = min(100, skills["terraforming"] + 10)
    elif element == "air":
        stats["improvisation"] = min(100, stats["improvisation"] + 15)
        skills["coding"] = min(100, skills["coding"] + 10)

    return {
        "id": colonist_id,
        "name": name,
        "element": element,
        "stats": stats,
        "skills": skills,
        "alive": True,
        "year_born": 0,
        "year_died": None,
        "cause_of_death": None,
        "memories": [],
        "governance_votes": [],
        "sub_sims_run": 0,
        "relationships": {},
    }


def init_relationships(colonists: list[dict]) -> None:
    """Initialize relationship scores as id-keyed dicts."""
    rng = _random_mod.Random(99)
    alive_ids = [c["id"] for c in colonists if c["alive"]]
    for c in colonists:
        if not c["alive"]:
            continue
        for other_id in alive_ids:
            if other_id != c["id"]:
                c["relationships"][str(other_id)] = rng.randint(-10, 10)


# ---------------------------------------------------------------------------
# Colony state
# ---------------------------------------------------------------------------

def create_colony(seed: int = 42, n_colonists: int = 10) -> dict:
    """Create initial colony state."""
    colonists = [create_colonist(i, _random_mod.Random(seed + i * 7)) for i in range(n_colonists)]
    init_relationships(colonists)

    return {
        "_meta": {
            "engine": "mars-100",
            "version": "1.0",
            "seed": seed,
            "created": datetime.now(timezone.utc).isoformat(),
        },
        "year": 0,
        "colonists": colonists,
        "resources": dict(INITIAL_RESOURCES),
        "governance": {
            "system": "direct_democracy",
            "leader": None,
            "constitution": [
                "All colonists have equal vote",
                "Exile requires 2/3 majority",
                "Leader serves 5-year terms",
                "Emergency powers last 1 year max",
            ],
            "amendments": [],
        },
        "history": [],
        "sub_sim_log": [],
        "dead_souls": [],
        "born_log": [],
        "proposals_pending": [],
    }


# ---------------------------------------------------------------------------
# LisPy bridge: inject colonist state into a LisPy env
# ---------------------------------------------------------------------------

def colonist_to_env(colonist: dict, colony: dict, event: dict, env: Env) -> None:
    """Inject colonist and colony state into a LisPy environment."""
    env["my-id"] = colonist["id"]
    env["my-name"] = colonist["name"]
    env["my-element"] = colonist["element"]
    for stat, val in colonist["stats"].items():
        env[f"my-{stat}"] = val
    for skill, val in colonist["skills"].items():
        env[f"my-{skill}"] = val

    rels = [[int(k), v] for k, v in colonist["relationships"].items()]
    env["my-relationships"] = rels

    for res, val in colony["resources"].items():
        env[f"colony-{res}"] = val
    env["colony-year"] = colony["year"]
    env["year-num"] = colony["year"]
    env["colony-population"] = sum(1 for c in colony["colonists"] if c["alive"])
    env["colony-leader"] = colony["governance"]["leader"]
    env["governance-system"] = colony["governance"]["system"]

    env["event-id"] = event["id"]
    env["event-severity"] = event["severity"]
    env["event-desc"] = event["desc"]

    # Action primitives (return action dicts)
    env["propose"] = lambda gtype, detail="": {"type": "propose", "governance_type": gtype, "detail": detail}
    env["vote"] = lambda prop_id, yes_no: {"type": "vote", "proposal_id": prop_id, "vote": yes_no}
    env["work"] = lambda skill: {"type": "work", "skill": skill}
    env["repair"] = lambda: {"type": "repair"}
    env["explore"] = lambda: {"type": "explore"}
    env["pray"] = lambda: {"type": "pray"}
    env["sabotage"] = lambda target_id: {"type": "sabotage", "target": target_id}
    env["mediate"] = lambda id_a, id_b: {"type": "mediate", "between": [id_a, id_b]}
    env["hoard"] = lambda resource: {"type": "hoard", "resource": resource}
    env["share"] = lambda resource, amount: {"type": "share", "resource": resource, "amount": amount}


# ---------------------------------------------------------------------------
# Action generation (personality-biased LisPy expressions)
# ---------------------------------------------------------------------------

def generate_action_expr(colonist: dict, event: dict, year: int) -> str:
    """Generate a LisPy expression for a colonist's action based on personality."""
    stats = colonist["stats"]
    skills = colonist["skills"]

    lines = ["(cond"]
    if stats["paranoia"] > 50:
        lines.append('  ((> event-severity 0.6) (hoard "food"))')
        lines.append('  ((= event-id "alien_signal") (begin (work "coding") (propose "emergency_powers" "lockdown")))')
    if stats["empathy"] > 50:
        lines.append('  ((< colony-morale 40) (share "food" 20))')
        lines.append('  ((< colony-morale 25) (propose "morale_initiative" "community bonding sessions"))')
    if stats["resolve"] > 60:
        best_skill = max(skills, key=lambda s: skills[s])
        lines.append(f'  ((> event-severity 0.3) (work "{best_skill}"))')
        if year > 5 and stats["resolve"] > 70:
            lines.append(f'  ((< colony-food 200) (propose "resource_priority" "prioritize hydroponics"))')
    if stats["faith"] > 50:
        lines.append('  ((> event-severity 0.7) (pray))')
        if year > 10:
            lines.append('  ((> colony-morale 60) (propose "constitutional_amendment" "right to spiritual practice"))')
    if stats["improvisation"] > 50:
        lines.append('  ((= event-id "cave_system") (explore))')
        lines.append('  ((= event-id "ice_discovery") (explore))')
        if year > 3:
            lines.append('  ((> event-severity 0.5) (propose "research_directive" "study the anomaly"))')
    if year > 15 and stats["resolve"] > 50:
        lines.append('  ((> year-num 15) (propose "leadership_election" "we need new leadership"))')
    best_skill = max(skills, key=lambda s: skills[s])
    lines.append(f'  (else (work "{best_skill}"))')
    lines.append(")")
    return "\n".join(lines)


def generate_subsim_expr(colonist: dict, proposal: dict) -> str:
    """Generate a sub-sim LisPy expression to model a governance proposal."""
    ptype = proposal.get("governance_type", "unknown")
    return f"""(sub-sim
  (let ((init-morale colony-morale)
        (init-food colony-food)
        (pop colony-population))
    (cond
      ((= "{ptype}" "exile")
       (list "exile-result" (- pop 1) (+ init-morale 5) (+ init-food 50)))
      ((= "{ptype}" "ration_allocation")
       (list "ration-result" pop (- init-morale 10) (+ init-food 100)))
      ((= "{ptype}" "leadership_election")
       (list "election-result" pop (+ init-morale 15) init-food))
      ((= "{ptype}" "emergency_powers")
       (list "emergency-result" pop (- init-morale 20) (+ init-food 30)))
      (else
       (list "default-result" pop init-morale init-food)))))"""


# ---------------------------------------------------------------------------
# Event system
# ---------------------------------------------------------------------------

def pick_event(year: int, rng: _random_mod.Random) -> dict:
    """Select this year's environmental event."""
    weights = [1.0] * len(EVENTS)
    if year < 20:
        weights[0] *= 2.0
    if year < 30:
        weights[4] = 0.0
    if year % 5 != 0:
        weights[13] = 0.0
    total = sum(weights)
    weights = [w / total for w in weights]
    return rng.choices(EVENTS, weights=weights, k=1)[0]


def apply_event_effects(colony: dict, event: dict, rng: _random_mod.Random) -> list[str]:
    """Apply environmental event effects to colony resources."""
    effects = []
    eid = event["id"]
    sev = event["severity"]
    res = colony["resources"]

    if eid == "dust_storm":
        loss = sev * 50
        res["power"] = max(0, res["power"] - loss)
        effects.append(f"Power reduced by {loss:.0f} kWh")
    elif eid == "resource_strike":
        gain = 100 + rng.random() * 200
        res["materials"] += gain
        effects.append(f"Materials +{gain:.0f}")
    elif eid == "equipment_failure":
        loss = sev * 30
        res["oxygen"] = max(0, res["oxygen"] - loss)
        effects.append(f"Oxygen reduced by {loss:.0f}")
    elif eid == "earth_contact":
        res["morale"] = min(100, res["morale"] + 15)
        effects.append("Morale +15 from Earth contact")
    elif eid == "alien_signal":
        res["morale"] = min(100, res["morale"] + 5)
        effects.append("Mysterious signal boosts curiosity")
    elif eid == "solar_flare":
        loss = sev * 80
        res["power"] = max(0, res["power"] - loss)
        effects.append(f"Solar flare: power -{loss:.0f}")
    elif eid == "ice_discovery":
        gain = 200 + rng.random() * 300
        res["water"] += gain
        effects.append(f"Ice discovery: water +{gain:.0f}")
    elif eid == "hab_breach":
        o_loss = sev * 50
        m_loss = sev * 30
        res["oxygen"] = max(0, res["oxygen"] - o_loss)
        res["materials"] = max(0, res["materials"] - m_loss)
        res["morale"] = max(0, res["morale"] - 15)
        effects.append(f"Hab breach: oxygen -{o_loss:.0f}, materials -{m_loss:.0f}")
    elif eid == "crop_blight":
        loss = sev * 100
        res["food"] = max(0, res["food"] - loss)
        effects.append(f"Crop blight: food -{loss:.0f}")
    elif eid == "meteorite":
        loss = sev * 40
        res["materials"] = max(0, res["materials"] - loss)
        effects.append(f"Meteorite impact: materials -{loss:.0f}")
    elif eid == "geothermal_vent":
        res["power"] += 60
        effects.append("Geothermal vent: power +60")
    elif eid == "dust_devil":
        res["power"] = max(0, res["power"] - 20)
        effects.append("Dust devil: power -20")
    elif eid == "aurora":
        res["morale"] = min(100, res["morale"] + 10)
        effects.append("Aurora: morale +10")
    elif eid == "supply_ship":
        res["food"] += 200
        res["water"] += 200
        res["materials"] += 100
        effects.append("Supply ship: food +200, water +200, materials +100")
    elif eid == "cave_system":
        res["materials"] += 80
        effects.append("Cave system: materials +80")
    return effects


# ---------------------------------------------------------------------------
# Colonist action processing
# ---------------------------------------------------------------------------

def process_action(colonist: dict, action, colony: dict, rng: _random_mod.Random) -> str:
    """Process a colonist's action and return narrative string."""
    if action is None or not isinstance(action, dict):
        return f"{colonist['name']} stands idle."

    atype = action.get("type", "idle")
    res = colony["resources"]

    if atype == "work":
        skill = action.get("skill", "terraforming")
        skill_val = colonist["skills"].get(skill, 10)
        prod = skill_val * 0.5
        if skill == "hydroponics":
            res["food"] += prod
            return f"{colonist['name']} tends crops (+{prod:.0f} food)"
        elif skill == "terraforming":
            res["materials"] += prod * 0.3
            return f"{colonist['name']} terraforms (+{prod * 0.3:.0f} materials)"
        elif skill == "coding":
            res["power"] += prod * 0.4
            return f"{colonist['name']} optimizes systems (+{prod * 0.4:.0f} power)"
        else:
            res["morale"] = min(100, res["morale"] + 1)
            return f"{colonist['name']} works on {skill}"
    elif atype == "repair":
        gain = colonist["skills"].get("coding", 10) * 0.3
        res["oxygen"] += gain
        return f"{colonist['name']} repairs life support (+{gain:.0f} oxygen)"
    elif atype == "explore":
        if rng.random() < 0.3:
            find = rng.choice(["water", "materials", "food"])
            amount = rng.randint(10, 50)
            res[find] += amount
            return f"{colonist['name']} explores and finds {amount} {find}"
        return f"{colonist['name']} explores but finds nothing"
    elif atype == "pray":
        boost = colonist["stats"].get("faith", 10) * 0.1
        res["morale"] = min(100, res["morale"] + boost)
        return f"{colonist['name']} prays (morale +{boost:.1f})"
    elif atype == "hoard":
        resource = action.get("resource", "food")
        res[resource] = max(0, res.get(resource, 0) - 5)
        res["morale"] = max(0, res["morale"] - 2)
        return f"{colonist['name']} hoards 5 {resource}"
    elif atype == "share":
        resource = action.get("resource", "food")
        amount = min(action.get("amount", 10), res.get(resource, 0))
        res["morale"] = min(100, res["morale"] + 3)
        return f"{colonist['name']} shares {amount} {resource} (morale +3)"
    elif atype == "mediate":
        between = action.get("between", [])
        if len(between) == 2:
            for c in colony["colonists"]:
                if c["id"] in between and c["alive"]:
                    other_id = str(between[0] if c["id"] == between[1] else between[1])
                    c["relationships"][other_id] = min(100, c["relationships"].get(other_id, 0) + 10)
            return f"{colonist['name']} mediates between colonists {between}"
        return f"{colonist['name']} attempts mediation"
    elif atype == "sabotage":
        target = action.get("target")
        if target is not None:
            for c in colony["colonists"]:
                if c["id"] == target and c["alive"]:
                    c["relationships"][str(colonist["id"])] = max(
                        -100, c["relationships"].get(str(colonist["id"]), 0) - 20)
                    res["morale"] = max(0, res["morale"] - 5)
                    return f"{colonist['name']} sabotages {c['name']}"
        return f"{colonist['name']} plots but acts on nothing"
    elif atype == "propose":
        gtype = action.get("governance_type", "resource_priority")
        detail = action.get("detail", "")
        proposal = {
            "id": len(colony["proposals_pending"]),
            "year": colony["year"],
            "proposer": colonist["id"],
            "governance_type": gtype,
            "detail": detail,
            "votes_for": [colonist["id"]],
            "votes_against": [],
            "resolved": False,
        }
        colony["proposals_pending"].append(proposal)
        return f"{colonist['name']} proposes {gtype}: {detail}"
    return f"{colonist['name']} does nothing."


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

def resolve_proposals(colony: dict, rng: _random_mod.Random) -> list[str]:
    """Resolve all pending proposals by majority vote."""
    effects = []
    alive = [c for c in colony["colonists"] if c["alive"]]
    alive_ids = {c["id"] for c in alive}
    majority = len(alive) // 2 + 1

    for proposal in colony["proposals_pending"]:
        if proposal["resolved"]:
            continue
        for c in alive:
            if c["id"] == proposal["proposer"]:
                continue
            if c["id"] in proposal["votes_for"] or c["id"] in proposal["votes_against"]:
                continue
            rel = c["relationships"].get(str(proposal["proposer"]), 0)
            vote_chance = 0.5 + (rel / 100.0) * 0.3
            if rng.random() < vote_chance:
                proposal["votes_for"].append(c["id"])
            else:
                proposal["votes_against"].append(c["id"])

        proposal["resolved"] = True
        yes = len(proposal["votes_for"])
        no = len(proposal["votes_against"])
        gtype = proposal["governance_type"]

        if gtype == "exile" and yes >= len(alive) * 2 // 3:
            worst_id, worst_avg = None, 999
            for c in alive:
                if c["id"] == proposal["proposer"]:
                    continue
                avg_rel = sum(c["relationships"].get(str(oid), 0) for oid in alive_ids if oid != c["id"]) / max(1, len(alive_ids) - 1)
                if avg_rel < worst_avg:
                    worst_avg = avg_rel
                    worst_id = c["id"]
            if worst_id is not None:
                for c in colony["colonists"]:
                    if c["id"] == worst_id:
                        c["alive"] = False
                        c["year_died"] = colony["year"]
                        c["cause_of_death"] = "exiled"
                        colony["dead_souls"].append(copy.deepcopy(c))
                        effects.append(f"Exile passed ({yes}-{no}): {c['name']} banished")
                        break
        elif gtype == "leadership_election" and yes >= majority:
            colony["governance"]["leader"] = proposal["proposer"]
            colony["resources"]["morale"] = min(100, colony["resources"]["morale"] + 10)
            name = next((c["name"] for c in alive if c["id"] == proposal["proposer"]), "?")
            effects.append(f"Election passed ({yes}-{no}): {name} elected leader")
        elif gtype == "emergency_powers" and yes >= majority:
            colony["governance"]["system"] = "emergency_powers"
            colony["governance"]["leader"] = proposal["proposer"]
            effects.append(f"Emergency powers granted ({yes}-{no})")
        elif gtype == "ration_allocation" and yes >= majority:
            colony["resources"]["food"] += 20
            colony["resources"]["morale"] = max(0, colony["resources"]["morale"] - 5)
            effects.append(f"Rationing enacted ({yes}-{no})")
        elif gtype == "expansion_vote" and yes >= majority:
            colony["resources"]["materials"] = max(0, colony["resources"]["materials"] - 50)
            effects.append(f"Expansion approved ({yes}-{no})")
        elif gtype == "research_directive" and yes >= majority:
            skill = rng.choice(SKILL_NAMES)
            for c in alive:
                c["skills"][skill] = min(100, c["skills"][skill] + 5)
            effects.append(f"Research directive ({yes}-{no}): all gain +5 {skill}")
        elif gtype == "constitution_amendment" and yes >= len(alive) * 2 // 3:
            amendment = proposal.get("detail", "New amendment")
            colony["governance"]["amendments"].append({
                "year": colony["year"],
                "text": amendment,
                "proposer": proposal["proposer"],
            })
            effects.append(f"Amendment ratified ({yes}-{no}): {amendment}")
        else:
            effects.append(f"Proposal {gtype} failed ({yes}-{no})")

    colony["proposals_pending"] = [p for p in colony["proposals_pending"] if not p["resolved"]]
    return effects


# ---------------------------------------------------------------------------
# Resources and survival
# ---------------------------------------------------------------------------

def consume_resources(colony: dict) -> list[str]:
    """Annual resource consumption and death checks."""
    effects = []
    alive = [c for c in colony["colonists"] if c["alive"]]
    n = len(alive)
    if n == 0:
        return ["Colony extinct."]

    res = colony["resources"]
    # Scaled consumption for game balance (10-person colony)
    res["food"] -= n * 12
    res["water"] -= n * 15
    res["power"] -= n * 8
    res["oxygen"] -= n * 10
    # Base production (hydroponics, recyclers, solar)
    res["food"] += n * 10
    res["water"] += n * 13
    res["power"] += n * 7
    res["oxygen"] += n * 9

    if res["food"] < 0:
        victim = min(alive, key=lambda c: c["stats"]["resolve"])
        victim["alive"] = False
        victim["year_died"] = colony["year"]
        victim["cause_of_death"] = "starvation"
        colony["dead_souls"].append(copy.deepcopy(victim))
        effects.append(f"{victim['name']} dies of starvation")
        res["food"] = 0

    if res["oxygen"] < 0:
        victim = min((c for c in colony["colonists"] if c["alive"]), key=lambda c: c["stats"]["resolve"], default=None)
        if victim:
            victim["alive"] = False
            victim["year_died"] = colony["year"]
            victim["cause_of_death"] = "asphyxiation"
            colony["dead_souls"].append(copy.deepcopy(victim))
            effects.append(f"{victim['name']} dies of asphyxiation")
        res["oxygen"] = 0

    if res["food"] < 100:
        res["morale"] = max(0, res["morale"] - 5)
    if res["water"] < 100:
        res["morale"] = max(0, res["morale"] - 3)

    for key in res:
        if isinstance(res[key], (int, float)):
            res[key] = max(0.0, res[key])
    return effects


def evolve_relationships(colony: dict, rng: _random_mod.Random) -> None:
    """Evolve relationships based on shared experiences."""
    alive = [c for c in colony["colonists"] if c["alive"]]
    for c in alive:
        for other in alive:
            if c["id"] == other["id"]:
                continue
            key = str(other["id"])
            current = c["relationships"].get(key, 0)
            drift = rng.gauss(0, 3)
            if c["element"] == other["element"]:
                drift += 2
            if colony["resources"]["morale"] < 30:
                drift -= 3
            c["relationships"][key] = round(max(-100, min(100, current + drift)), 1)


def check_births(colony: dict, rng: _random_mod.Random) -> list[str]:
    """Check for births."""
    effects = []
    alive = [c for c in colony["colonists"] if c["alive"]]
    n = len(alive)
    if n < 2:
        return effects
    birth_chance = 0.08 * (colony["resources"]["morale"] / 100.0)
    if n >= 15:
        birth_chance *= 0.3
    if rng.random() < birth_chance:
        new_id = max(c["id"] for c in colony["colonists"]) + 1
        baby = create_colonist(new_id, rng)
        baby["year_born"] = colony["year"]
        for c in alive:
            baby["relationships"][str(c["id"])] = rng.randint(5, 30)
            c["relationships"][str(new_id)] = rng.randint(5, 30)
        colony["colonists"].append(baby)
        colony["born_log"].append({"id": new_id, "name": baby["name"], "year": colony["year"]})
        effects.append(f"{baby['name']} is born (colonist #{new_id})")
    return effects


def check_meta_awareness(colony: dict, year: int) -> str | None:
    """Check if any colonist realizes they might be in a simulation."""
    if year < 15:
        return None
    for c in (c for c in colony["colonists"] if c["alive"]):
        score = (c["stats"]["improvisation"] + c["stats"]["faith"]) / 2 + c["sub_sims_run"] * 5
        threshold = 120 - year * 0.5
        if score > threshold:
            return c["name"]
    return None


# ---------------------------------------------------------------------------
# Year tick
# ---------------------------------------------------------------------------

def tick_year(colony: dict, base_seed: int) -> dict:
    """Advance the colony by one Martian year. Returns year delta."""
    year = colony["year"] + 1
    colony["year"] = year
    year_rng = _random_mod.Random(base_seed * 10000 + year)

    delta = {
        "year": year,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": None,
        "event_effects": [],
        "colonist_actions": [],
        "governance_results": [],
        "resource_effects": [],
        "birth_effects": [],
        "sub_sims": [],
        "meta_awareness": None,
        "population": 0,
        "resources_snapshot": {},
        "diary_entries": [],
    }

    alive = [c for c in colony["colonists"] if c["alive"]]
    if not alive:
        delta["population"] = 0
        delta["resources_snapshot"] = dict(colony["resources"])
        return delta

    # 1. Environmental event
    event = pick_event(year, year_rng)
    delta["event"] = {"id": event["id"], "desc": event["desc"], "severity": event["severity"]}
    delta["event_effects"] = apply_event_effects(colony, event, year_rng)

    # 2. Each colonist acts via LisPy
    for c in alive:
        if not c["alive"]:
            continue
        c_seed = base_seed * 10000 + year * 100 + c["id"]
        env, ctx = make_env(seed=c_seed)
        colonist_to_env(c, colony, event, env)
        expr_str = generate_action_expr(c, event, year)
        try:
            action = run_in_env(expr_str, env, ctx)
        except (LispError, Exception):
            action = {"type": "work", "skill": "terraforming"}
        narrative = process_action(c, action, colony, year_rng)
        delta["colonist_actions"].append({"colonist": c["name"], "action": narrative})

        # Sub-sim for governance proposals
        if colony["proposals_pending"] and c["stats"]["improvisation"] > 40:
            for prop in colony["proposals_pending"][-1:]:
                subsim_expr = generate_subsim_expr(c, prop)
                try:
                    sub_env, sub_ctx = make_env(seed=c_seed + 999)
                    colonist_to_env(c, colony, event, sub_env)
                    result = run_in_env(subsim_expr, sub_env, sub_ctx)
                    c["sub_sims_run"] += 1
                    delta["sub_sims"].append({
                        "colonist": c["name"],
                        "depth": 1,
                        "proposal": prop["governance_type"],
                        "result": str(result),
                        "s_expr": subsim_expr[:200],
                    })
                except DepthLimitExceeded:
                    delta["sub_sims"].append({
                        "colonist": c["name"],
                        "depth": "exceeded",
                        "proposal": prop.get("governance_type", "?"),
                        "result": "depth limit reached",
                    })
                except (LispError, Exception):
                    pass

    # 3. Governance
    delta["governance_results"] = resolve_proposals(colony, year_rng)

    # 4. Resource consumption
    delta["resource_effects"] = consume_resources(colony)

    # 5. Births
    delta["birth_effects"] = check_births(colony, year_rng)

    # 6. Relationship evolution
    evolve_relationships(colony, year_rng)

    # 7. Meta-awareness
    aware_name = check_meta_awareness(colony, year)
    if aware_name:
        delta["meta_awareness"] = f"{aware_name} suspects they are in a simulation"

    # 8. Diary entries (from 3 colonists)
    alive_now = [c for c in colony["colonists"] if c["alive"]]
    diarists = alive_now[:3] if len(alive_now) >= 3 else alive_now
    for c in diarists:
        entry = _generate_diary(c, event, colony, year)
        delta["diary_entries"].append({"colonist": c["name"], "entry": entry})
        c["memories"].append({"year": year, "memory": entry[:100]})
        if len(c["memories"]) > 20:
            c["memories"] = c["memories"][-20:]

    delta["population"] = sum(1 for c in colony["colonists"] if c["alive"])
    delta["resources_snapshot"] = {k: round(v, 1) if isinstance(v, float) else v
                                   for k, v in colony["resources"].items()}
    colony["history"].append(delta)
    return delta


def _generate_diary(colonist: dict, event: dict, colony: dict, year: int) -> str:
    """Generate a diary entry for a colonist."""
    name = colonist["name"]
    mood = "anxious" if colony["resources"]["morale"] < 40 else "hopeful" if colony["resources"]["morale"] > 70 else "steady"
    pop = sum(1 for c in colony["colonists"] if c["alive"])
    entry = f"Year {year} — {name}'s Log\n"
    entry += f"  Event: {event['desc']}\n"
    entry += f"  Mood: {mood} | Population: {pop}\n"
    entry += f"  Food: {colony['resources']['food']:.0f} | Water: {colony['resources']['water']:.0f}\n"
    if colonist["stats"]["paranoia"] > 60:
        entry += "  I don't trust the others. Something is wrong.\n"
    if colonist["stats"]["faith"] > 60:
        entry += "  We must keep faith. Mars tests us.\n"
    if colonist["stats"]["empathy"] > 60:
        entry += "  We need each other now more than ever.\n"
    if year > 30 and colonist["sub_sims_run"] > 2:
        entry += "  The simulations within simulations... are we in one too?\n"
    return entry


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------

def run_simulation(years: int = 100, seed: int = 42) -> dict:
    """Run the full Mars-100 simulation for N years."""
    colony = create_colony(seed=seed)
    deltas = []
    for _ in range(years):
        alive = [c for c in colony["colonists"] if c["alive"]]
        if not alive:
            break
        delta = tick_year(colony, seed)
        deltas.append(delta)
    return {
        "colony": colony,
        "deltas": deltas,
        "summary": _build_summary(colony, deltas),
    }


def _build_summary(colony: dict, deltas: list[dict]) -> dict:
    """Build summary of the simulation."""
    total_subsims = sum(len(d["sub_sims"]) for d in deltas)
    meta_events = [d["meta_awareness"] for d in deltas if d["meta_awareness"]]
    pop_curve = [d["population"] for d in deltas]
    morale_curve = [d["resources_snapshot"].get("morale", 0) for d in deltas]
    return {
        "years_survived": len(deltas),
        "final_population": sum(1 for c in colony["colonists"] if c["alive"]),
        "total_births": len(colony["born_log"]),
        "total_deaths": len(colony["dead_souls"]),
        "total_sub_simulations": total_subsims,
        "meta_awareness_events": meta_events,
        "constitutional_amendments": colony["governance"]["amendments"],
        "governance_system": colony["governance"]["system"],
        "leader": colony["governance"]["leader"],
        "population_curve": pop_curve,
        "morale_curve": morale_curve,
    }
