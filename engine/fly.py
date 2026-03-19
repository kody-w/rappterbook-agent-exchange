#!/usr/bin/env python3
"""Musca domestica — housefly lifecycle tick engine (Gen 2+).

Reads state/fly.json, advances one tick, writes back.
The output of frame N is the input of frame N+1.

Gen 2 additions:
- Ancestry tracking across generations
- Inherited behavioral biases from parent
- Kitchen decomposition (corpse, food decay)
- Egg cluster sibling tracking
- Ant threat behavior
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import copy
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(ROOT / "docs")))

STAGES = ["egg", "larva", "pupa", "adult", "death"]
MAX_GENERATIONS = 10


def load_state() -> dict:
    """Load fly state from disk."""
    with open(STATE_DIR / "fly.json") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    """Save to state/ and docs/ for frontend."""
    with open(STATE_DIR / "fly.json", "w") as f:
        json.dump(state, f, indent=2)
    with open(DOCS_DIR / "fly_state.json", "w") as f:
        json.dump(state, f, separators=(",", ":"))


def dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def record(state: dict, event: str) -> None:
    """Append a history event."""
    state["history"].append({
        "tick": state["lifecycle"]["total_ticks"],
        "event": event,
        "stage": state["lifecycle"]["stage"],
        "energy": round(state["energy"]["current"], 1),
        "position": copy.deepcopy(state["body"]["position"]),
    })
    if len(state["history"]) > 200:
        state["history"] = state["history"][-150:]


def update_kitchen(state: dict) -> None:
    """Advance kitchen environment: time, temperature, lights, decomposition."""
    k = state["kitchen"]
    k["time_of_day"] = (k["time_of_day"] + 0.009) % 1.0
    tod = k["time_of_day"]
    k["lights_on"] = 0.25 < tod < 0.85
    k["ambient_temp"] = 20 + 4 * math.sin(tod * math.pi)

    # Decompose remains
    for obj in k["objects"]:
        if obj["type"] == "remains":
            obj["decay_progress"] = min(1.0, obj.get("decay_progress", 0) + 0.008)
            obj["smell_radius"] = int(40 + obj["decay_progress"] * 60)
            if obj["decay_progress"] > 0.5:
                obj["energy"] = max(0, obj.get("energy", 5) - 0.1)

    # Sibling egg cluster: some hatch, some don't
    for obj in k["objects"]:
        if obj["type"] == "egg_cluster":
            tick = state["lifecycle"]["total_ticks"]
            if tick > 6 and random.random() < 0.02:
                hatched = random.randint(1, 3)
                obj["count"] = max(0, obj.get("count", 0) - hatched)
                obj["viability"] = max(0, obj.get("viability", 0.6) - 0.02)

    # Food decay over time
    for obj in k["objects"]:
        if obj["type"] == "food" and random.random() < 0.005:
            obj["energy"] = max(1, obj["energy"] - 1)
            obj["decay"] = min(1.0, obj.get("decay", 0.5) + 0.01)


def update_threats(state: dict) -> None:
    """Randomly spawn/despawn threats."""
    k = state["kitchen"]
    for obj in k["objects"]:
        if obj["type"] != "threat":
            continue
        if obj["id"] == "ants":
            # Ants slowly patrol near food
            if random.random() < 0.1:
                obj["x"] += random.uniform(-3, 3)
                obj["y"] += random.uniform(-3, 3)
            continue
        if not obj.get("active") and random.random() < 0.03:
            obj["active"] = True
            obj["x"] = random.uniform(50, k["width"] - 50)
            obj["y"] = random.uniform(50, k["height"] - 50)
            record(state, f"{obj['name']} appears!")
        elif obj.get("active") and random.random() < 0.15:
            obj["active"] = False
            obj["x"] = -100
            obj["y"] = -100


def update_senses(state: dict) -> None:
    """Compute what the fly can smell, see, and feel."""
    body = state["body"]
    senses = state["senses"]
    genome = state["genome"]
    k = state["kitchen"]
    stage = state["lifecycle"]["stage"]

    senses["temperature"] = k["ambient_temp"]
    senses["wind"] = random.uniform(0, 0.3)
    senses["touch"]["vibration"] *= 0.7

    if stage in ("egg", "pupa"):
        senses["smell"] = []
        senses["sight"] = []
        return

    smells = []
    sights = []
    px, py = body["position"]["x"], body["position"]["y"]

    for obj in k["objects"]:
        d = dist2d(px, py, obj["x"], obj["y"])
        if obj["type"] in ("food", "remains"):
            sr = obj.get("smell_radius", 80) * genome["smell_sensitivity"]
            if d < sr:
                smells.append({
                    "id": obj["id"], "name": obj["name"],
                    "distance": round(d, 1),
                    "intensity": round(1 - d / sr, 2),
                    "type": obj["type"],
                })
        if obj["type"] == "light" and d < 350:
            sights.append({
                "id": obj["id"], "name": obj["name"],
                "distance": round(d, 1),
                "intensity": obj.get("intensity", 0.5),
            })
        if obj["type"] == "threat" and obj.get("active") and d < 120:
            sights.append({
                "id": obj["id"], "name": obj["name"],
                "distance": round(d, 1),
                "threat": True,
            })

    senses["smell"] = smells
    senses["sight"] = sights
    senses["touch"]["surface"] = "air" if body["is_airborne"] else body.get("surface", "counter")


def think(state: dict) -> None:
    """Brain decision-making with inherited behavioral biases."""
    brain = state["brain"]
    energy = state["energy"]
    senses = state["senses"]
    stage = state["lifecycle"]["stage"]

    if stage == "egg":
        brain["state"] = "dormant"
        brain["current_goal"] = None
        return
    if stage == "pupa":
        brain["state"] = "metamorphosis"
        brain["current_goal"] = None
        return

    brain["state"] = "simple_reflex" if stage == "larva" else "active"
    brain["curiosity"] = min(1.0, brain["curiosity"] + 0.02)

    # Inherited fear makes this fly more cautious
    inherited_fear = brain.get("inherited_fear", 0)
    fear_threshold = max(40, 80 - inherited_fear * 100)

    # Check for threats first
    for sight in senses.get("sight", []):
        if sight.get("threat") and sight["distance"] < fear_threshold:
            brain["current_goal"] = "flee"
            brain["fear_level"] = min(1.0, brain["fear_level"] + 0.5)
            brain["decisions_made"] += 1
            return

    brain["fear_level"] = max(0, brain["fear_level"] - 0.05)

    # Inherited food memory: check if near parent's known food
    inherited_food = state["memory"].get("inherited_food_memory", [])
    if inherited_food and energy["hunger"] > 10 and random.random() < 0.2:
        brain["current_goal"] = "seek_inherited"
        brain["decisions_made"] += 1
        return

    # Hungry? Seek food
    if energy["hunger"] > 15 and senses["smell"]:
        brain["current_goal"] = "seek_food"
        brain["decisions_made"] += 1
        return

    # Curious? Explore (inherited curiosity boosts this)
    inherited_curiosity = brain.get("inherited_curiosity", 0)
    explore_chance = 0.4 + inherited_curiosity
    if brain["curiosity"] > 0.5 and random.random() < explore_chance:
        brain["current_goal"] = "explore"
        brain["curiosity"] = max(0, brain["curiosity"] - 0.3)
        brain["decisions_made"] += 1
        return

    # Phototaxis for adults
    if stage == "adult" and senses["sight"]:
        lights = [s for s in senses["sight"] if not s.get("threat")]
        if lights and random.random() < 0.3:
            brain["current_goal"] = "fly_to_light"
            brain["decisions_made"] += 1
            return

    brain["current_goal"] = "idle"


def move(state: dict) -> None:
    """Move the fly based on brain goal and physics."""
    body = state["body"]
    brain = state["brain"]
    senses = state["senses"]
    memory = state["memory"]
    genome = state["genome"]
    k = state["kitchen"]
    stage = state["lifecycle"]["stage"]
    goal = brain["current_goal"]

    if stage in ("egg", "pupa"):
        body["velocity"] = {"x": 0, "y": 0, "z": 0}
        return

    px, py = body["position"]["x"], body["position"]["y"]

    if goal == "flee":
        threats = [s for s in senses.get("sight", []) if s.get("threat")]
        if threats:
            th = threats[0]
            tobj = next((o for o in k["objects"] if o["id"] == th["id"]), None)
            if tobj:
                angle = math.atan2(py - tobj["y"], px - tobj["x"])
                speed = 8 * genome["flight_efficiency"]
                body["velocity"]["x"] = math.cos(angle) * speed
                body["velocity"]["y"] = math.sin(angle) * speed
                if not body["is_airborne"] and stage == "adult":
                    body["is_airborne"] = True
                    body["position"]["z"] = 1.5
                memory["times_fled"] = memory.get("times_fled", 0) + 1
                # Remember danger zone
                dz = memory.get("danger_zones", [])
                if len(dz) < 10:
                    dz.append({"x": tobj["x"], "y": tobj["y"], "threat": th["id"]})
                    memory["danger_zones"] = dz
        return

    if goal == "seek_inherited":
        # Move toward inherited food memory from parent
        inherited = memory.get("inherited_food_memory", [])
        if inherited:
            target = inherited[0]
            dx, dy = target["x"] - px, target["y"] - py
            d = max(dist2d(px, py, target["x"], target["y"]), 0.1)
            speed = 1.5 if stage == "larva" else 4 * genome["flight_efficiency"]
            body["velocity"]["x"] = dx / d * speed
            body["velocity"]["y"] = dy / d * speed
            if stage == "adult" and not body["is_airborne"] and d > 30:
                body["is_airborne"] = True
                body["position"]["z"] = 1.5
        return

    if goal == "seek_food":
        smells = senses.get("smell", [])
        if smells:
            target = max(smells, key=lambda s: s["intensity"])
            tobj = next((o for o in k["objects"] if o["id"] == target["id"]), None)
            if tobj:
                dx, dy = tobj["x"] - px, tobj["y"] - py
                d = max(dist2d(px, py, tobj["x"], tobj["y"]), 0.1)
                speed = 1.5 if stage == "larva" else 5 * genome["flight_efficiency"]
                body["velocity"]["x"] = dx / d * speed
                body["velocity"]["y"] = dy / d * speed
                if stage == "adult" and not body["is_airborne"] and d > 30:
                    body["is_airborne"] = True
                    body["position"]["z"] = 1.5
        return

    if goal == "explore":
        angle = random.uniform(0, 2 * math.pi)
        speed = 2 if stage == "larva" else 4 * genome["flight_efficiency"]
        body["velocity"]["x"] = math.cos(angle) * speed
        body["velocity"]["y"] = math.sin(angle) * speed
        if stage == "adult" and not body["is_airborne"]:
            body["is_airborne"] = True
            body["position"]["z"] = 1.5
        return

    if goal == "fly_to_light":
        lights = [s for s in senses.get("sight", []) if not s.get("threat")]
        if lights:
            light = max(lights, key=lambda s: s["intensity"])
            lobj = next((o for o in k["objects"] if o["id"] == light["id"]), None)
            if lobj:
                dx, dy = lobj["x"] - px, lobj["y"] - py
                d = max(dist2d(px, py, lobj["x"], lobj["y"]), 0.1)
                speed = 3 * genome["flight_efficiency"]
                body["velocity"]["x"] = dx / d * speed
                body["velocity"]["y"] = dy / d * speed
                if not body["is_airborne"]:
                    body["is_airborne"] = True
                    body["position"]["z"] = 1.5
        return

    # Idle behavior
    if stage == "adult" and body["is_airborne"]:
        if random.random() < 0.3:
            body["is_airborne"] = False
            body["position"]["z"] = 0
            body["velocity"] = {"x": 0, "y": 0, "z": 0}
            body["surface"] = "counter"
        else:
            body["velocity"]["x"] += random.uniform(-1, 1)
            body["velocity"]["y"] += random.uniform(-1, 1)
    elif stage == "larva":
        angle = random.uniform(0, 2 * math.pi)
        body["velocity"]["x"] = math.cos(angle) * 1
        body["velocity"]["y"] = math.sin(angle) * 1
    else:
        speed = 1.5
        body["facing"] = body.get("facing", 0) + random.uniform(-0.5, 0.5)
        body["velocity"]["x"] = math.cos(body["facing"]) * speed
        body["velocity"]["y"] = math.sin(body["facing"]) * speed


def try_feed(state: dict) -> None:
    """Try to eat if near food (includes remains for larvae)."""
    body = state["body"]
    energy = state["energy"]
    memory = state["memory"]
    k = state["kitchen"]
    lc = state["lifecycle"]
    stage = lc["stage"]

    if stage in ("egg", "pupa", "death"):
        return

    px, py = body["position"]["x"], body["position"]["y"]
    for obj in k["objects"]:
        if obj["type"] not in ("food", "remains"):
            continue
        # Larvae can feed on remains (protein source)
        if obj["type"] == "remains" and stage != "larva":
            continue
        d = dist2d(px, py, obj["x"], obj["y"])
        feed_range = 15 if stage == "adult" else 10
        if d < feed_range:
            mult = 0.4 if stage == "adult" else 0.3
            available = obj.get("energy", 10)
            gained = min(available * mult, energy["max"] - energy["current"])
            energy["current"] = min(energy["max"], energy["current"] + gained)
            energy["hunger"] = max(0, energy["hunger"] - 25)
            energy["last_fed_tick"] = lc["total_ticks"]
            memory["times_fed"] += 1
            if obj["id"] not in [f["id"] for f in memory["food_sources"]]:
                memory["food_sources"].append({"id": obj["id"], "x": obj["x"], "y": obj["y"]})
            if memory["favorite_food"] is None or available > 20:
                memory["favorite_food"] = obj["name"]
            state["brain"]["satisfaction"] = min(1.0, state["brain"]["satisfaction"] + 0.2)
            if stage == "adult":
                body["is_airborne"] = False
                body["position"]["z"] = 0
                body["surface"] = "food"
            break


def update_energy(state: dict) -> None:
    """Apply metabolic drain and physics."""
    body = state["body"]
    energy = state["energy"]
    genome = state["genome"]
    k = state["kitchen"]
    memory = state["memory"]
    stage = state["lifecycle"]["stage"]

    base = energy["metabolic_drain"] * genome["metabolic_rate"]
    drain_mult = {"egg": 0.2, "larva": 0.8, "pupa": 0.3}.get(stage, 1.0)
    if stage == "adult" and body["is_airborne"]:
        drain_mult = 2.0
    energy["current"] = max(0, energy["current"] - base * drain_mult)
    energy["hunger"] = min(100, energy["hunger"] + base * drain_mult * 0.5)

    # Physics: apply velocity, enforce bounds
    body["position"]["x"] += body["velocity"]["x"]
    body["position"]["y"] += body["velocity"]["y"]
    body["position"]["z"] += body["velocity"]["z"]

    if body["is_airborne"]:
        body["position"]["z"] = max(0.5, body["position"]["z"])
        body["velocity"]["x"] *= 0.92
        body["velocity"]["y"] *= 0.92
        if body["position"]["z"] > memory.get("peak_altitude", 0):
            memory["peak_altitude"] = body["position"]["z"]
    else:
        body["position"]["z"] = max(0, body["position"]["z"])
        body["velocity"]["x"] *= 0.7
        body["velocity"]["y"] *= 0.7

    body["position"]["x"] = clamp(body["position"]["x"], 0, k["width"])
    body["position"]["y"] = clamp(body["position"]["y"], 0, k["height"])

    dist_moved = math.sqrt(body["velocity"]["x"] ** 2 + body["velocity"]["y"] ** 2)
    memory["total_distance"] += dist_moved


def check_transition(state: dict) -> None:
    """Check if the fly should advance to the next lifecycle stage."""
    lc = state["lifecycle"]
    stage = lc["stage"]
    body = state["body"]
    brain = state["brain"]

    if stage == "death":
        return

    duration = lc["stage_durations"].get(stage, 999)
    if lc["stage_tick"] < duration:
        # Larva growth: molts
        if stage == "larva":
            instar = 1 + lc["stage_tick"] // 8
            if instar != lc.get("larva_instar", 0):
                lc["larva_instar"] = instar
                lc["molts"] = lc.get("molts", 0) + 1
                body["size"] += 0.8
                record(state, f"molt to instar {instar}")
            body["size"] += 0.06
            body["mass"] = body["size"] * 0.003
            brain["neural_complexity"] = min(0.3, brain["neural_complexity"] + 0.008)
        if stage == "pupa":
            prog = lc["stage_tick"] / max(duration, 1)
            brain["neural_complexity"] = min(1.0, 0.3 + prog * 0.7)
            if prog < 0.3:
                body["wing_state"] = "forming"
                body["leg_state"] = "forming"
            elif prog < 0.7:
                body["wing_state"] = "forming"
                body["leg_state"] = "forming"
            else:
                body["wing_state"] = "folded"
                body["leg_state"] = "folded"
                body["size"] = max(body["size"], 5.0)
        return

    # Transition!
    idx = STAGES.index(stage)
    new_stage = STAGES[idx + 1]
    lc["stage"] = new_stage
    lc["stage_tick"] = 0
    record(state, f"{stage} -> {new_stage}")

    if new_stage == "larva":
        brain["state"] = "simple_reflex"
        body["leg_state"] = "stub"
    elif new_stage == "pupa":
        body["velocity"] = {"x": 0, "y": 0, "z": 0}
    elif new_stage == "adult":
        body["wing_state"] = "functional"
        body["leg_state"] = "functional"
        body["size"] = max(body["size"], 6.0)
        body["mass"] = 0.012
        brain["state"] = "active"
    elif new_stage == "death":
        state["_meta"]["cause_of_death"] = "old age"
        state["_meta"]["died_at"] = lc["total_ticks"]
        brain["state"] = "dead"

    # Check starvation
    if state["energy"]["current"] <= 0 and new_stage != "death":
        lc["stage"] = "death"
        state["_meta"]["cause_of_death"] = "starvation"
        state["_meta"]["died_at"] = lc["total_ticks"]
        brain["state"] = "dead"
        record(state, "starved to death")


def generate_narration(state: dict) -> None:
    """Write a one-line narration for the current tick."""
    lc = state["lifecycle"]
    stage = lc["stage"]
    body = state["body"]
    energy = state["energy"]
    brain = state["brain"]
    gen = state["_meta"].get("generation", 1)
    ancestry = state.get("ancestry", {})

    if stage == "death":
        cause = state["_meta"].get("cause_of_death", "unknown")
        state["narration"] = f"Gen {gen} is gone. Cause: {cause}. The kitchen endures."
        return

    if stage == "egg":
        prog = lc["stage_tick"] / max(lc["stage_durations"]["egg"], 1)
        parent_cause = ancestry.get("parent", {}).get("cause_of_death", "")
        if prog < 0.3:
            state["narration"] = f"Gen {gen} — a speck beside its mother's remains. Cells divide in silence."
        elif prog < 0.7:
            state["narration"] = f"The embryo twitches. Nearby, {ancestry.get('siblings_alive', 0)} siblings wait."
        else:
            state["narration"] = f"A crack forms. Generation {gen} prepares to face the kitchen."
        return

    if stage == "larva":
        if brain["current_goal"] == "seek_food":
            state["narration"] = f"Gen {gen} larva wriggles hungrily. Size: {body['size']:.1f}mm."
        elif brain["current_goal"] == "seek_inherited":
            state["narration"] = f"Something pulls the larva toward crumbs. A mother's memory, written in genes."
        elif energy["hunger"] < 20:
            state["narration"] = f"Well-fed larva grows. {body['size']:.1f}mm. The mother's body feeds her children."
        else:
            state["narration"] = f"Each molt brings new hunger. Gen {gen}, instar {lc.get('larva_instar', 1)}."
        return

    if stage == "pupa":
        prog = lc["stage_tick"] / max(lc["stage_durations"]["pupa"], 1)
        if prog < 0.4:
            state["narration"] = f"Gen {gen} dissolves into soup. Everything it was unmakes itself."
        elif prog < 0.8:
            state["narration"] = f"Wings crystallize where none existed. Eyes compound. Metamorphosis: {prog*100:.0f}%."
        else:
            state["narration"] = f"Almost there. A new adult assembles from the old larva's sacrifice."
        return

    # Adult narrations
    goal = brain["current_goal"]
    parent_dist = ancestry.get("parent", {}).get("total_distance", 0)
    own_dist = state["memory"]["total_distance"]
    narrations = {
        "flee": f"DANGER! Gen {gen} bolts — fear runs deeper in this bloodline.",
        "seek_food": f"Hunger drives Gen {gen} downward. Smell sensitivity: {state['genome']['smell_sensitivity']:.0%}.",
        "seek_inherited": f"An ancestral pull — toward crumbs the mother once loved.",
        "explore": f"Gen {gen} maps the kitchen. {own_dist:.0f}px traveled (mother: {parent_dist:.0f}px).",
        "fly_to_light": f"The light calls. Every generation spirals toward it.",
        "idle": f"Gen {gen} grooms a foreleg. Energy: {energy['current']:.0f}%." if not body["is_airborne"]
                else f"Wings beat 200x/sec. Gen {gen} surveys its domain.",
    }
    state["narration"] = narrations.get(goal, f"Gen {gen} — energy: {energy['current']:.0f}%. Life persists.")


def rebirth(state: dict) -> dict:
    """Create next generation from dead fly. Death feeds new life."""
    gen = state["_meta"].get("generation", 1)
    if gen >= MAX_GENERATIONS:
        return state

    rng = random.Random(gen * 7919 + state["lifecycle"]["total_ticks"])

    gen_record = {
        "generation": gen,
        "genome": copy.deepcopy(state["genome"]),
        "total_ticks": state["lifecycle"]["total_ticks"],
        "cause_of_death": state["_meta"].get("cause_of_death", "unknown"),
        "memory_summary": {
            "times_fed": state["memory"]["times_fed"],
            "times_fled": state["memory"]["times_fled"],
            "total_distance": round(state["memory"]["total_distance"], 1),
            "favorite_food": state["memory"]["favorite_food"],
            "decisions_made": state["brain"]["decisions_made"]
        },
        "death_position": copy.deepcopy(state["body"]["position"]),
        "final_energy": round(state["energy"]["current"], 1)
    }

    generations = state.get("generations", [])
    generations.append(gen_record)

    parent = state["genome"]
    new_genome = {}
    for k, v in parent.items():
        if isinstance(v, float):
            new_genome[k] = round(max(0.0, min(1.5, v + rng.gauss(0, 0.04))), 4)
        elif isinstance(v, int):
            new_genome[k] = max(2000, min(6000, int(v + rng.gauss(0, 150))))
        else:
            new_genome[k] = v

    foods = [o for o in state["kitchen"]["objects"]
             if o["type"] == "food" and o.get("energy", 0) > 3]
    if foods:
        target = rng.choice(foods)
        egg_x = target["x"] + rng.uniform(-20, 20)
        egg_y = target["y"] + rng.uniform(-20, 20)
    else:
        egg_x = rng.uniform(100, 500)
        egg_y = rng.uniform(100, 300)

    kitchen = copy.deepcopy(state["kitchen"])
    kitchen["objects"] = [o for o in kitchen["objects"]
                          if not (o.get("is_corpse") and o.get("decomposition", 0) >= 1.0)]
    kitchen["objects"].append({
        "id": f"corpse_gen{gen}", "type": "food",
        "x": round(state["body"]["position"]["x"], 1),
        "y": round(state["body"]["position"]["y"], 1),
        "z": 0, "smell_radius": 60, "energy": 5, "decay": 0.95,
        "name": f"gen-{gen} remains", "is_corpse": True, "decomposition": 0.0
    })
    for obj in kitchen["objects"]:
        if obj["type"] == "threat":
            obj["active"] = False
            obj["x"] = -100
            obj["y"] = -100
    kitchen["time_of_day"] = rng.uniform(0.05, 0.2)
    kitchen["lights_on"] = False

    base_dur = state["lifecycle"]["stage_durations"]
    new_dur = {s: max(5, d + rng.randint(-2, 2)) for s, d in base_dur.items()}
    new_gen = gen + 1
    now = datetime.now(timezone.utc).isoformat()

    ancestry = {
        "parent": {
            "generation": gen,
            "cause_of_death": state["_meta"].get("cause_of_death", "unknown"),
            "total_distance": round(state["memory"]["total_distance"], 1),
            "favorite_food": state["memory"]["favorite_food"],
            "total_ticks": state["lifecycle"]["total_ticks"]
        },
        "siblings_alive": 0,
        "lineage_length": new_gen
    }

    energy_start = round(80 + rng.uniform(0, 15), 1)
    return {
        "_meta": {
            "organism": "Musca domestica",
            "frame": state["_meta"]["frame"] + 1,
            "born_at": now,
            "version": "2.0.0",
            "generation": new_gen,
            "cause_of_death": None,
            "died_at": None,
            "total_frames_alive": 0,
            "parent_generation": gen,
            "lineage_id": state["_meta"].get("lineage_id", "musca-kitchen-alpha")
        },
        "ancestry": ancestry,
        "generations": generations,
        "genome": new_genome,
        "lifecycle": {
            "stage": "egg", "stage_tick": 0, "total_ticks": 0,
            "stage_durations": new_dur, "molts": 0, "larva_instar": 0
        },
        "body": {
            "position": {"x": round(egg_x, 2), "y": round(egg_y, 2), "z": 0.5},
            "velocity": {"x": 0, "y": 0, "z": 0},
            "facing": 0, "size": 1.0, "mass": 0.001,
            "wing_state": "none", "leg_state": "none",
            "is_airborne": False, "surface": "counter"
        },
        "energy": {
            "current": energy_start, "max": 100, "hunger": 5.0,
            "metabolic_drain": 0.5, "last_fed_tick": 0
        },
        "brain": {
            "state": "dormant", "current_goal": None,
            "fear_level": 0.0, "curiosity": 0.0, "satisfaction": 0.5,
            "decisions_made": 0, "neural_complexity": 0.0,
            "inherited_memory": {
                "danger_awareness": min(0.8, 0.1 * len(generations)),
                "food_preference": state["memory"]["favorite_food"]
            }
        },
        "senses": {
            "smell": [], "sight": [],
            "touch": {"surface": "counter", "vibration": 0.0},
            "temperature": 22.0, "wind": 0.0
        },
        "memory": {
            "food_sources": [], "danger_zones": [], "visited_positions": [],
            "total_distance": 0.0, "times_fed": 0, "times_fled": 0,
            "peak_altitude": 0.5, "favorite_food": None
        },
        "kitchen": kitchen,
        "history": [{
            "tick": 0, "event": f"generation {new_gen} — egg laid",
            "stage": "egg", "energy": energy_start,
            "position": {"x": round(egg_x, 2), "y": round(egg_y, 2), "z": 0.5}
        }],
        "narration": f"Generation {new_gen}. A new egg glistens in the half-light."
    }


def tick(state: dict) -> dict:
    """Advance the organism one tick forward. THE HEARTBEAT."""
    if state["lifecycle"]["stage"] == "death":
        return state

    state["_meta"]["frame"] += 1
    state["lifecycle"]["stage_tick"] += 1
    state["lifecycle"]["total_ticks"] += 1
    state["_meta"]["total_frames_alive"] = state["lifecycle"]["total_ticks"]

    update_kitchen(state)
    update_threats(state)
    update_senses(state)
    think(state)
    move(state)
    try_feed(state)
    update_energy(state)
    check_transition(state)
    generate_narration(state)

    return state


def main() -> None:
    state = load_state()

    if state["lifecycle"]["stage"] == "death":
        gen = state["_meta"].get("generation", 1)
        if gen < MAX_GENERATIONS:
            print(f"Generation {gen} is dead. Rebirth → generation {gen + 1}...")
            state = rebirth(state)
            save_state(state)
            print(f"  New egg at ({state['body']['position']['x']:.0f}, {state['body']['position']['y']:.0f})")
            print(f"  {state['narration']}")
            return
        else:
            print(f"Lineage complete after {gen} generations. The kitchen is still.")
            save_state(state)
            return

    args = sys.argv[1:]
    ticks = 1
    until_death = False
    i = 0
    while i < len(args):
        if args[i] == "--ticks" and i + 1 < len(args):
            ticks = int(args[i + 1])
            i += 2
        elif args[i] == "--until" and i + 1 < len(args) and args[i + 1] == "death":
            until_death = True
            i += 2
        else:
            i += 1

    if until_death:
        while state["lifecycle"]["stage"] != "death":
            state = tick(state)
            frame = state["_meta"]["frame"]
            stage = state["lifecycle"]["stage"]
            e = state["energy"]["current"]
            gen = state["_meta"].get("generation", 1)
            print(f"  Frame {frame:3d} | Gen {gen} | {stage:6s} | energy={e:5.1f} | {state['narration'][:60]}")
    else:
        for _ in range(ticks):
            if state["lifecycle"]["stage"] == "death":
                break
            state = tick(state)
        frame = state["_meta"]["frame"]
        stage = state["lifecycle"]["stage"]
        e = state["energy"]["current"]
        goal = state["brain"]["current_goal"]
        gen = state["_meta"].get("generation", 1)
        print(f"Frame {frame} | Gen {gen} | {stage} | energy={e:.1f} | goal={goal}")
        print(f"  {state['narration']}")

    save_state(state)


if __name__ == "__main__":
    main()
