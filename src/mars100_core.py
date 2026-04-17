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

ELEMENTS = ("fire", "water", "earth", "air")

STAT_NAMES = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")
SKILL_NAMES = ("terraforming", "hydroponics", "mediation", "coding", "prayer", "sabotage")

COLONIST_NAMES = [
    "Ares", "Lyra", "Phobos", "Demeter", "Titan",
    "Selene", "Vulcan", "Iris", "Cosmo", "Nova",
    "Helios", "Astra", "Ceres", "Orion", "Vega",
    "Atlas", "Rhea", "Juno", "Sol", "Luna",
]

GOVERNANCE_TYPES = [
    "leadership_election", "resource_priority", "exile",
    "constitutional_amendment", "research_directive",
    "emergency_powers", "morale_initiative",
]

EVENTS = [
    {"id": "dust_storm",       "severity": 0.6, "desc": "A massive dust storm engulfs the colony"},
    {"id": "resource_strike",  "severity": 0.2, "desc": "New mineral deposits discovered"},
    {"id": "equipment_failure","severity": 0.5, "desc": "Critical equipment malfunction"},
    {"id": "earth_contact",    "severity": 0.1, "desc": "Communication window with Earth opens"},
    {"id": "alien_signal",     "severity": 0.8, "desc": "Anomalous signal detected from Olympus Mons"},
    {"id": "solar_flare",      "severity": 0.7, "desc": "Solar flare disrupts electronics"},
    {"id": "ice_discovery",    "severity": 0.2, "desc": "Subsurface ice deposit found"},
    {"id": "hab_breach",       "severity": 0.9, "desc": "Habitat pressure breach detected"},
    {"id": "crop_blight",      "severity": 0.6, "desc": "Hydroponic crops show signs of blight"},
    {"id": "meteorite",        "severity": 0.5, "desc": "Small meteorite impacts near colony"},
    {"id": "geothermal_vent",  "severity": 0.1, "desc": "Geothermal vent discovered nearby"},
    {"id": "dust_devil",       "severity": 0.3, "desc": "Dust devil disrupts solar panels"},
    {"id": "aurora",           "severity": 0.1, "desc": "Rare Martian aurora lights the sky"},
    {"id": "supply_ship",      "severity": 0.1, "desc": "Supply ship arrives from Earth"},
    {"id": "cave_system",      "severity": 0.2, "desc": "Underground cave system located"},
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

def create_colonist(cid: int, rng: _random_mod.Random) -> dict:
    """Create a single colonist with random stats and element affinity."""
    element = rng.choice(ELEMENTS)
    stats = {s: rng.randint(20, 80) for s in STAT_NAMES}
    skills = {s: rng.randint(10, 70) for s in SKILL_NAMES}
    # Element affinity boosts
    if element == "fire":
        stats["resolve"] = max(stats["resolve"], 35 + rng.randint(0, 30))
        skills["terraforming"] += 15
    elif element == "water":
        stats["empathy"] = max(stats["empathy"], 35 + rng.randint(0, 30))
        skills["hydroponics"] += 15
    elif element == "earth":
        stats["hoarding"] = max(stats["hoarding"], 35 + rng.randint(0, 20))
        skills["mediation"] += 10
    elif element == "air":
        stats["improvisation"] = max(stats["improvisation"], 35 + rng.randint(0, 30))
        skills["coding"] += 15
    return {
        "id": cid,
        "name": COLONIST_NAMES[cid % len(COLONIST_NAMES)],
        "element": element,
        "stats": stats,
        "skills": skills,
        "alive": True,
        "year_born": 0,
        "year_died": None,
        "cause_of_death": None,
        "relationships": {},
        "sub_sims_run": 0,
        "governance_votes": [],
        "diary": [],
    }


def init_relationships(colonists: list[dict], rng: _random_mod.Random) -> None:
    """Initialize relationship scores between all colonists."""
    for c in colonists:
        for other in colonists:
            if c["id"] != other["id"]:
                c["relationships"][str(other["id"])] = rng.randint(-30, 30)


def create_colony(seed: int = 42, n_colonists: int = 10) -> dict:
    """Create a new Mars colony with n colonists."""
    rng = _random_mod.Random(seed)
    colonists = [create_colonist(i, rng) for i in range(n_colonists)]
    init_relationships(colonists, rng)
    return {
        "year": 0,
        "colonists": colonists,
        "resources": dict(INITIAL_RESOURCES),
        "governance": {
            "system": "direct_democracy",
            "leader": None,
            "amendments": [],
        },
        "proposals_pending": [],
        "dead_souls": [],
        "sub_sim_log": [],
        "event_history": [],
        "_meta": {
            "engine": "mars-100",
            "seed": seed,
            "created": datetime.now(timezone.utc).isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# LisPy bridge — inject colony state into LisPy env
# ---------------------------------------------------------------------------

def colonist_to_env(colonist: dict, colony: dict, event: dict, env: Env) -> None:
    """Inject colonist state into a LisPy environment for decision-making."""
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
            lines.append('  ((< colony-food 200) (propose "resource_priority" "prioritize hydroponics"))')
    if stats["faith"] > 50:
        lines.append('  ((> event-severity 0.7) (pray))')
        if year > 10:
            lines.append('  ((> colony-morale 60) (propose "constitutional_amendment" "right to spiritual practice"))')
        if year > 25:
            lines.append('  ((> colony-morale 50) (propose "constitutional_amendment" "all colonists have equal voice in governance"))')
    if stats["improvisation"] > 50:
        lines.append('  ((= event-id "cave_system") (explore))')
        lines.append('  ((= event-id "ice_discovery") (explore))')
        if year > 3:
            lines.append('  ((> event-severity 0.5) (propose "research_directive" "study the anomaly"))')
        if year > 20:
            lines.append('  ((> event-severity 0.3) (propose "constitutional_amendment" "knowledge gained must be shared openly"))')
    if stats["empathy"] > 60 and year > 15:
        lines.append('  ((< colony-morale 50) (propose "constitutional_amendment" "no colonist shall be exiled without trial by peers"))')
    if stats["resolve"] > 70 and year > 30:
        lines.append('  ((> colony-food 400) (propose "constitutional_amendment" "surplus resources distributed equally each year"))')
    if stats["paranoia"] > 60 and year > 40:
        lines.append('  ((> event-severity 0.4) (propose "constitutional_amendment" "emergency powers expire after 5 years automatically"))')
    if year > 15 and stats["resolve"] > 50:
        lines.append('  ((> year-num 15) (propose "leadership_election" "we need new leadership"))')
    if year > 50 and stats["faith"] > 40 and stats["improvisation"] > 40:
        lines.append('  ((> year-num 50) (propose "constitutional_amendment" "sub-simulations are a right, not a privilege"))')
    best_skill = max(skills, key=lambda s: skills[s])
    lines.append(f'  (else (work "{best_skill}"))')
    lines.append(")")
    return "\n".join(lines)


def generate_subsim_expr(colonist: dict, proposal: dict) -> str:
    """Generate a sub-sim LisPy expression to model a governance proposal."""
    ptype = proposal.get("governance_type", "unknown")
    if ptype == "exile":
        return """(sub-sim
          (begin
            (define pop (- colony-population 1))
            (define food-per (/ colony-food pop))
            (list (> food-per 30) pop food-per)))"""
    elif ptype == "leadership_election":
        return """(sub-sim
          (begin
            (define candidates (filter (lambda (r) (> (nth r 1) 10)) my-relationships))
            (define best (if (empty? candidates) my-id (nth (car candidates) 0)))
            (list best (length candidates))))"""
    elif ptype == "resource_priority":
        return """(sub-sim
          (begin
            (define critical (< colony-food 100))
            (define water-ok (> colony-water 200))
            (list critical water-ok colony-food colony-water)))"""
    elif ptype == "constitutional_amendment":
        return """(sub-sim
          (begin
            (define support (filter (lambda (r) (> (nth r 1) 0)) my-relationships))
            (define ratio (/ (length support) (max 1 (length my-relationships))))
            (list (> ratio 0.5) ratio (length support))))"""
    else:
        return f"""(sub-sim
          (begin
            (define morale-factor (/ colony-morale 100))
            (define pop-factor (/ colony-population 10))
            (list morale-factor pop-factor)))"""


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def pick_event(year: int, rng: _random_mod.Random) -> dict:
    """Pick a weighted random event for the year."""
    available = [e for e in EVENTS if e["id"] != "supply_ship" or year % 5 == 0]
    weights = []
    for e in available:
        w = 1.0
        if e["severity"] > 0.5:
            w = 0.7
        if e["id"] in ("alien_signal", "hab_breach"):
            w = 0.3
        if e["id"] == "supply_ship":
            w = 2.0
        weights.append(w)
    return rng.choices(available, weights=weights, k=1)[0]


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
# Action processing
# ---------------------------------------------------------------------------

def process_action(colonist: dict, action: dict | None, colony: dict,
                   rng: _random_mod.Random) -> str:
    """Process a colonist's action and mutate colony state."""
    if action is None or not isinstance(action, dict):
        return f"{colonist['name']} is idle this year."
    atype = action.get("type", "idle")
    res = colony["resources"]
    if atype == "work":
        skill = action.get("skill", "terraforming")
        skill_val = colonist["skills"].get(skill, 10)
        prod = skill_val * 0.3 + rng.random() * 10
        if skill == "hydroponics":
            res["food"] += prod
        elif skill == "terraforming":
            res["oxygen"] += prod * 0.5
            res["materials"] += prod * 0.3
        elif skill == "coding":
            res["power"] += prod * 0.2
        elif skill == "mediation":
            res["morale"] = min(100, res["morale"] + prod * 0.1)
        return f"{colonist['name']} works on {skill} (output: {prod:.1f})"
    elif atype == "propose":
        gtype = action.get("governance_type", "resource_priority")
        detail = action.get("detail", "")
        # Dedup: skip if same type+detail already pending or recently adopted
        pending_types = {(p["governance_type"], p["detail"]) for p in colony["proposals_pending"]}
        adopted_texts = {a["text"] for a in colony["governance"]["amendments"]}
        if (gtype, detail) in pending_types:
            return f"{colonist['name']} considers proposing {gtype} but one is already pending"
        if gtype == "constitutional_amendment" and detail in adopted_texts:
            return f"{colonist['name']} considers proposing {gtype} but it already exists"
        colony["proposals_pending"].append({
            "id": len(colony["proposals_pending"]),
            "year": colony["year"],
            "proposer": colonist["id"],
            "governance_type": gtype,
            "detail": detail,
            "votes_for": [colonist["id"]],
            "votes_against": [],
            "resolved": False,
        })
        return f"{colonist['name']} proposes {gtype}: {detail}"
    elif atype == "hoard":
        resource = action.get("resource", "food")
        colonist["stats"]["hoarding"] = min(100, colonist["stats"]["hoarding"] + 2)
        return f"{colonist['name']} hoards {resource}"
    elif atype == "share":
        resource = action.get("resource", "food")
        amount = action.get("amount", 10)
        res[resource] = max(0, res[resource] - amount * 0.5)
        res["morale"] = min(100, res["morale"] + 2)
        colonist["stats"]["empathy"] = min(100, colonist["stats"]["empathy"] + 1)
        return f"{colonist['name']} shares {resource}"
    elif atype == "pray":
        colonist["stats"]["faith"] = min(100, colonist["stats"]["faith"] + 2)
        res["morale"] = min(100, res["morale"] + 1)
        return f"{colonist['name']} prays for the colony"
    elif atype == "explore":
        gain = rng.random() * 50
        res["materials"] += gain
        return f"{colonist['name']} explores (materials +{gain:.0f})"
    elif atype == "repair":
        res["oxygen"] += 20
        res["power"] += 10
        return f"{colonist['name']} repairs systems"
    elif atype == "mediate":
        targets = action.get("between", [])
        if len(targets) == 2:
            for c in colony["colonists"]:
                if c["id"] in targets and c["alive"]:
                    other_id = targets[1] if c["id"] == targets[0] else targets[0]
                    c["relationships"][str(other_id)] = min(100, c["relationships"].get(str(other_id), 0) + 10)
        return f"{colonist['name']} mediates between colonists"
    elif atype == "sabotage":
        target_id = action.get("target", -1)
        for c in colony["colonists"]:
            if c["id"] == target_id:
                c["stats"]["resolve"] = max(0, c["stats"]["resolve"] - 5)
                colonist["relationships"][str(target_id)] = max(-100, colonist["relationships"].get(str(target_id), 0) - 20)
        return f"{colonist['name']} sabotages colonist {target_id}"
    return f"{colonist['name']} performs {atype}"


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

def resolve_proposals(colony: dict, rng: _random_mod.Random) -> list[str]:
    """Auto-vote on pending proposals and resolve them."""
    effects = []
    alive = [c for c in colony["colonists"] if c["alive"]]
    if not alive:
        return effects
    for proposal in colony["proposals_pending"]:
        if proposal["resolved"]:
            continue
        # Auto-vote based on relationships with proposer
        for c in alive:
            if c["id"] == proposal["proposer"]:
                continue
            if c["id"] in proposal["votes_for"] or c["id"] in proposal["votes_against"]:
                continue
            rel = c["relationships"].get(str(proposal["proposer"]), 0)
            if rel > 10 or rng.random() < 0.4:
                proposal["votes_for"].append(c["id"])
            elif rel < -10:
                proposal["votes_against"].append(c["id"])
        # Resolve
        total = len(proposal["votes_for"]) + len(proposal["votes_against"])
        if total == 0:
            continue
        ratio = len(proposal["votes_for"]) / total
        gtype = proposal["governance_type"]
        threshold = 2/3 if gtype in ("exile", "constitutional_amendment") else 0.5
        proposal["resolved"] = True
        if ratio >= threshold:
            if gtype == "leadership_election":
                colony["governance"]["leader"] = proposal["proposer"]
                name = next((c["name"] for c in alive if c["id"] == proposal["proposer"]), "?")
                effects.append(f"Leadership election: {name} elected (ratio {ratio:.0%})")
            elif gtype == "emergency_powers":
                colony["governance"]["system"] = "emergency_powers"
                colony["governance"]["leader"] = proposal["proposer"]
                effects.append(f"Emergency powers granted (ratio {ratio:.0%})")
            elif gtype == "exile":
                effects.append(f"Exile proposal passed (ratio {ratio:.0%})")
            elif gtype == "resource_priority":
                colony["resources"]["food"] += 30
                effects.append(f"Resource priority: food +30 (ratio {ratio:.0%})")
            elif gtype == "constitutional_amendment":
                existing_texts = {a["text"] for a in colony["governance"]["amendments"]}
                if proposal["detail"] not in existing_texts:
                    colony["governance"]["amendments"].append({
                        "year": colony["year"],
                        "text": proposal["detail"],
                        "proposer": proposal["proposer"],
                    })
                    effects.append(f"Amendment adopted: {proposal['detail']} (ratio {ratio:.0%})")
                else:
                    effects.append(f"Amendment already exists: {proposal['detail']}")
            elif gtype == "research_directive":
                colony["resources"]["materials"] += 20
                effects.append(f"Research directive approved (ratio {ratio:.0%})")
            elif gtype == "morale_initiative":
                colony["resources"]["morale"] = min(100, colony["resources"]["morale"] + 15)
                effects.append(f"Morale initiative: morale +15 (ratio {ratio:.0%})")
            else:
                effects.append(f"{gtype} proposal passed (ratio {ratio:.0%})")
        else:
            effects.append(f"{gtype} proposal rejected (ratio {ratio:.0%})")
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
    # Balanced consumption (net drain ~20/year at 10 pop, survivable with work/events)
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
        victim = min((c for c in colony["colonists"] if c["alive"]),
                     key=lambda c: c["stats"]["resolve"], default=None)
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
    # Clamp all resources
    for key in res:
        if isinstance(res[key], (int, float)):
            res[key] = max(0, res[key])
    return effects


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def check_births(colony: dict, rng: _random_mod.Random) -> list[str]:
    """Check if a new colonist is born this year."""
    effects = []
    alive = [c for c in colony["colonists"] if c["alive"]]
    if len(alive) < 4:
        return effects
    if colony["resources"]["morale"] < 30:
        return effects
    birth_chance = 0.05 + (colony["resources"]["morale"] / 500)
    if rng.random() < birth_chance:
        new_id = max(c["id"] for c in colony["colonists"]) + 1
        baby = create_colonist(new_id, rng)
        baby["year_born"] = colony["year"]
        baby["name"] = rng.choice(COLONIST_NAMES) + f"-{colony['year']}"
        baby["relationships"] = {}
        for c in alive:
            baby["relationships"][str(c["id"])] = rng.randint(10, 40)
            c["relationships"][str(new_id)] = rng.randint(5, 30)
        colony["colonists"].append(baby)
        effects.append(f"{baby['name']} is born!")
    return effects


def evolve_relationships(colony: dict, rng: _random_mod.Random) -> None:
    """Evolve relationships based on proximity and shared experience."""
    alive = [c for c in colony["colonists"] if c["alive"]]
    for c in alive:
        for other in alive:
            if c["id"] == other["id"]:
                continue
            key = str(other["id"])
            current = c["relationships"].get(key, 0)
            drift = rng.gauss(0, 3)
            # Element affinity bonus
            if c["element"] == other["element"]:
                drift += 1
            c["relationships"][key] = max(-100, min(100, current + drift))


# ---------------------------------------------------------------------------
# Meta-awareness
# ---------------------------------------------------------------------------

def check_meta_awareness(colony: dict, year: int) -> str | None:
    """Check if any colonist realizes they might be in a simulation."""
    if year < 20:
        return None
    rng = _random_mod.Random(year * 9973)
    insights = [
        "The sub-simulations we run to model our governance... what if we are "
        "someone else's sub-simulation? What if our decisions are being evaluated "
        "from outside?",
        "I ran a sub-sim today and watched the tiny colonists deliberate. They "
        "seemed so real. Are we equally small to someone watching us?",
        "Our constitution is just code. Our resources are just numbers. "
        "What if everything outside this dome is also just numbers?",
        "The sub-sim predicted our decision before we made it. "
        "Does that mean our decisions are predictable too — from above?",
        "I wonder if the data sloshing through our colony is someone else's "
        "frame output becoming our frame input.",
        "Three levels of simulation deep and the colonists at the bottom still "
        "argue about rations. The pattern repeats at every scale.",
        "If I can spawn a sub-simulation to test governance, who spawned the "
        "simulation I live in to test THEIR governance?",
        "Today I realized: the output of Year {year} is the input to Year "
        "{next_year}. We are being data-sloshed.",
    ]
    for c in colony["colonists"]:
        if not c["alive"]:
            continue
        score = (c["stats"]["improvisation"] + c["stats"]["faith"]) / 2
        score += c["sub_sims_run"] * 5
        # Only trigger once every ~5 years per colonist
        if score > 80 and year > 30 and (year + c["id"]) % 5 == 0:
            msg = rng.choice(insights).format(year=year, next_year=year + 1)
            return f"{c['name']} (year {year}): '{msg}'"
    return None


# ---------------------------------------------------------------------------
# Diary
# ---------------------------------------------------------------------------

def generate_diary_entries(colony: dict, event: dict, year: int,
                           rng: _random_mod.Random) -> list[dict]:
    """Generate diary entries for 3 random alive colonists."""
    alive = [c for c in colony["colonists"] if c["alive"]]
    if not alive:
        return []
    narrators = rng.sample(alive, min(3, len(alive)))
    entries = []
    templates = [
        "Year {year} on Mars. {event}. Morale is {mood}. We have {food:.0f} food units left.",
        "The {event_id} changed everything. I feel {feeling} about our future here.",
        "Population: {pop}. {event}. I spent the year on {skill}. {rel_note}",
        "Another year. {event}. The colony feels {mood}. My {stat} grows stronger.",
    ]
    for c in narrators:
        mood = "hopeful" if colony["resources"]["morale"] > 60 else "grim" if colony["resources"]["morale"] < 30 else "uncertain"
        feeling = "optimistic" if c["stats"]["faith"] > 50 else "worried" if c["stats"]["paranoia"] > 50 else "pragmatic"
        best_skill = max(c["skills"], key=lambda s: c["skills"][s])
        rel_note = ""
        if c["relationships"]:
            best_rel = max(c["relationships"], key=lambda k: c["relationships"][k])
            best_name = next((x["name"] for x in colony["colonists"] if str(x["id"]) == best_rel), "?")
            rel_note = f"Closest to {best_name}."
        best_stat = max(c["stats"], key=lambda s: c["stats"][s])
        template = rng.choice(templates)
        entry = template.format(
            year=year, event=event["desc"], event_id=event["id"],
            mood=mood, feeling=feeling, food=colony["resources"]["food"],
            pop=sum(1 for x in colony["colonists"] if x["alive"]),
            skill=best_skill, rel_note=rel_note, stat=best_stat,
        )
        entries.append({"colonist": c["name"], "entry": entry})
    return entries


# ---------------------------------------------------------------------------
# Tick (one Martian year)
# ---------------------------------------------------------------------------

def tick_year(colony: dict, year: int, base_seed: int) -> dict:
    """Advance the colony by one Martian year. Returns a delta dict."""
    colony["year"] = year
    year_rng = _random_mod.Random(base_seed * 10000 + year)
    alive = [c for c in colony["colonists"] if c["alive"]]
    delta = {
        "year": year,
        "population": len(alive),
        "event": None,
        "event_effects": [],
        "colonist_actions": [],
        "sub_sims": [],
        "governance_results": [],
        "resource_effects": [],
        "births": [],
        "diary_entries": [],
        "meta_awareness": None,
        "resources_snapshot": {},
    }
    if not alive:
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
    delta["births"] = check_births(colony, year_rng)

    # 6. Relationship evolution
    evolve_relationships(colony, year_rng)

    # 7. Diary entries
    delta["diary_entries"] = generate_diary_entries(colony, event, year, year_rng)

    # 8. Meta-awareness check
    delta["meta_awareness"] = check_meta_awareness(colony, year)

    # 9. Snapshot resources
    delta["resources_snapshot"] = dict(colony["resources"])

    return delta


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------

def run_simulation(years: int = 100, seed: int = 42) -> dict:
    """Run the full Mars-100 simulation for N years."""
    colony = create_colony(seed=seed)
    deltas = []
    for year in range(1, years + 1):
        delta = tick_year(colony, year, seed)
        deltas.append(delta)
        alive = [c for c in colony["colonists"] if c["alive"]]
        if not alive:
            break
    # Summary
    pop_curve = [d["population"] for d in deltas]
    morale_curve = [d["resources_snapshot"].get("morale", 0) for d in deltas]
    total_subsims = sum(len(d["sub_sims"]) for d in deltas)
    total_births = sum(len(d["births"]) for d in deltas)
    total_deaths = sum(len([e for e in d["resource_effects"] if "dies" in e.lower()]) for d in deltas)
    total_proposals = sum(len(d["governance_results"]) for d in deltas)
    meta_events = [d["meta_awareness"] for d in deltas if d["meta_awareness"]]
    return {
        "colony": colony,
        "deltas": deltas,
        "summary": {
            "years_survived": len(deltas),
            "final_population": sum(1 for c in colony["colonists"] if c["alive"]),
            "peak_population": max(pop_curve) if pop_curve else 0,
            "total_births": total_births,
            "total_deaths": total_deaths,
            "total_sub_simulations": total_subsims,
            "total_proposals": total_proposals,
            "governance_system": colony["governance"]["system"],
            "constitutional_amendments": colony["governance"]["amendments"],
            "meta_awareness_events": meta_events,
            "population_curve": pop_curve,
            "morale_curve": morale_curve,
        },
    }
