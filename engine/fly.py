#!/usr/bin/env python3
"""Musca domestica — housefly lifecycle tick engine (v3: stigmergy).

Reads state/fly.json, advances one tick, writes back.
The output of frame N is the input of frame N+1.

v3 additions (stigmergy awakens):
  - Pheromone trail system: flies deposit chemical trails that persist across
    generations. External memory written into the environment.
  - Kitchen weather: wind gusts, humidity, window state affect fly behavior.
  - Wing damage: near-misses with threats can permanently reduce flight.
  - Corpse ecology: dead flies become nutrient sources for offspring.
  - Circadian rhythm: flies rest at night, reduced metabolic drain.
  - Trail-following behavior: adults follow ancestor pheromone trails.
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


def load_state() -> dict:
    with open(STATE_DIR / "fly.json") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(STATE_DIR / "fly.json", "w") as f:
        json.dump(state, f, indent=2)
    with open(DOCS_DIR / "fly_state.json", "w") as f:
        json.dump(state, f, separators=(",", ":"))


def dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def record(state: dict, event: str) -> None:
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
    k = state["kitchen"]
    k["time_of_day"] = (k["time_of_day"] + 0.009) % 1.0
    tod = k["time_of_day"]
    k["lights_on"] = 0.25 < tod < 0.85
    k["ambient_temp"] = 20 + 4 * math.sin(tod * math.pi)
    for obj in k["objects"]:
        if obj["type"] == "food" and random.random() < 0.02:
            decay_rate = obj.get("decay", 0.5)
            obj["energy"] = max(1, obj["energy"] - decay_rate * 0.5)
            obj["smell_radius"] = min(500, obj["smell_radius"] + decay_rate * 2)


def update_weather(state: dict) -> None:
    rng = random.Random(state["_meta"]["frame"] * 3571)
    weather = state.setdefault("weather", {
        "wind_direction": 0.0, "wind_strength": 0.0,
        "humidity": 0.6, "window_open": False,
    })
    if rng.random() < 0.05:
        weather["wind_direction"] = rng.uniform(0, math.pi * 2)
        weather["wind_strength"] = rng.uniform(0.5, 2.0)
        weather["window_open"] = rng.random() < 0.3
    else:
        weather["wind_strength"] = max(0, weather.get("wind_strength", 0) * 0.95)
    weather["humidity"] = clamp(
        weather.get("humidity", 0.6) + rng.gauss(0, 0.01), 0.3, 0.95
    )


def update_threats(state: dict) -> None:
    k = state["kitchen"]
    for obj in k["objects"]:
        if obj["type"] != "threat":
            continue
        if not obj.get("active") and random.random() < 0.03:
            obj["active"] = True
            obj["x"] = random.uniform(50, k["width"] - 50)
            obj["y"] = random.uniform(50, k["height"] - 50)
            record(state, obj["name"] + " appears!")
        elif obj.get("active") and random.random() < 0.15:
            obj["active"] = False
            obj["x"] = -100
            obj["y"] = -100


def update_senses(state: dict) -> None:
    body = state["body"]
    senses = state["senses"]
    genome = state["genome"]
    k = state["kitchen"]
    stage = state["lifecycle"]["stage"]
    weather = state.get("weather", {})

    senses["temperature"] = k["ambient_temp"]
    senses["wind"] = weather.get("wind_strength", 0) + random.uniform(0, 0.3)
    senses.setdefault("touch", {})
    senses["touch"]["vibration"] = senses["touch"].get("vibration", 0) * 0.7

    if stage in ("egg", "pupa"):
        senses["smell"] = []
        senses["sight"] = []
        senses["pheromones"] = []
        return

    px, py = body["position"]["x"], body["position"]["y"]
    smells, sights = [], []

    for obj in k["objects"]:
        d = dist2d(px, py, obj["x"], obj["y"])
        if obj["type"] == "food":
            sr = obj.get("smell_radius", 80) * genome["smell_sensitivity"]
            if d < sr:
                smells.append({"id": obj["id"], "name": obj["name"],
                               "distance": round(d, 1),
                               "intensity": round(1 - d / sr, 2)})
        if obj["type"] == "light" and d < 350:
            sights.append({"id": obj["id"], "name": obj["name"],
                           "distance": round(d, 1),
                           "intensity": obj.get("intensity", 0.5)})
        if obj["type"] == "threat" and obj.get("active") and d < 120:
            sights.append({"id": obj["id"], "name": obj["name"],
                           "distance": round(d, 1), "threat": True})

    senses["smell"] = smells
    senses["sight"] = sights
    senses["touch"]["surface"] = "air" if body["is_airborne"] else body.get("surface", "counter")

    # Pheromone sensing
    pheromones = state.get("pheromones", [])
    nearby = []
    for p in pheromones:
        d = dist2d(px, py, p["x"], p["y"])
        sense_range = 40 * genome["smell_sensitivity"]
        if 0.1 < d < sense_range:
            nearby.append({"x": p["x"], "y": p["y"],
                           "intensity": round(p["intensity"] * (1 - d / sense_range), 3),
                           "gen": p["gen"]})
    senses["pheromones"] = sorted(nearby, key=lambda p: -p["intensity"])[:8]


def think(state: dict) -> None:
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
    epigenetic = brain.get("inherited_memory", {}).get("epigenetic_bias", 0)

    for sight in senses.get("sight", []):
        if sight.get("threat") and sight["distance"] < 80:
            brain["current_goal"] = "flee"
            brain["fear_level"] = min(1.0, brain["fear_level"] + 0.5)
            brain["decisions_made"] += 1
            return

    brain["fear_level"] = max(0, brain["fear_level"] - 0.05)

    if not state["kitchen"]["lights_on"] and stage == "adult":
        brain["current_goal"] = "rest"
        brain["decisions_made"] += 1
        return

    hunger_threshold = max(8, 15 - epigenetic * 20)
    if energy["hunger"] > hunger_threshold and senses["smell"]:
        brain["current_goal"] = "seek_food"
        brain["decisions_made"] += 1
        return

    if stage == "adult" and senses.get("pheromones") and random.random() < 0.12:
        brain["current_goal"] = "follow_trail"
        brain["decisions_made"] += 1
        return

    if stage == "adult":
        if energy["hunger"] < 10 and random.random() < 0.15:
            brain["current_goal"] = "groom"
            brain["decisions_made"] += 1
            return
        if random.random() < 0.08:
            brain["current_goal"] = "wall_walk"
            brain["decisions_made"] += 1
            return

    if brain["curiosity"] > 0.5 and random.random() < 0.4:
        brain["current_goal"] = "explore"
        brain["curiosity"] = max(0, brain["curiosity"] - 0.3)
        brain["decisions_made"] += 1
        return

    if stage == "adult" and senses["sight"]:
        lights = [s for s in senses["sight"] if not s.get("threat")]
        if lights and random.random() < 0.3:
            brain["current_goal"] = "fly_to_light"
            brain["decisions_made"] += 1
            return

    brain["current_goal"] = "idle"


def move(state: dict) -> None:
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
                if not body["is_airborne"]:
                    body["is_airborne"] = True
                    body["position"]["z"] = 1.5
                memory["times_fled"] = memory.get("times_fled", 0) + 1
                if th["distance"] < 25 and random.random() < 0.15:
                    dmg = round(random.uniform(0.02, 0.08), 3)
                    genome["flight_efficiency"] = max(0.3, genome["flight_efficiency"] - dmg)
                    body["wing_damage"] = round(body.get("wing_damage", 0) + dmg, 3)
                    record(state, "wing damaged! eff=" + str(round(genome["flight_efficiency"], 2)))
        return

    if goal == "seek_food":
        smells = senses.get("smell", [])
        if smells:
            parent_fav = brain.get("inherited_memory", {}).get("parent_favorite_food")
            target = None
            if parent_fav:
                fav = [s for s in smells if parent_fav.lower() in s["name"].lower()]
                if fav:
                    target = max(fav, key=lambda s: s["intensity"])
            if not target:
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

    if goal == "follow_trail":
        phero = senses.get("pheromones", [])
        if phero:
            best = max(phero, key=lambda p: p["intensity"])
            dx, dy = best["x"] - px, best["y"] - py
            d = max(math.sqrt(dx * dx + dy * dy), 0.1)
            speed = 3 * genome["flight_efficiency"]
            body["velocity"]["x"] = dx / d * speed
            body["velocity"]["y"] = dy / d * speed
            if not body["is_airborne"] and d > 20:
                body["is_airborne"] = True
                body["position"]["z"] = 1.0
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

    if goal == "groom":
        if body["is_airborne"]:
            body["is_airborne"] = False
            body["position"]["z"] = 0
            body["surface"] = "counter"
        body["velocity"] = {"x": 0, "y": 0, "z": 0}
        brain["satisfaction"] = min(1.0, brain["satisfaction"] + 0.1)
        return

    if goal == "rest":
        if body["is_airborne"]:
            body["is_airborne"] = False
            body["position"]["z"] = 0
        body["velocity"] = {"x": 0, "y": 0, "z": 0}
        body["surface"] = "counter"
        return

    if goal == "wall_walk":
        wall_targets = [(0, py), (k["width"], py), (px, 0), (px, k["height"])]
        nearest = min(wall_targets, key=lambda w: dist2d(px, py, w[0], w[1]))
        dx, dy = nearest[0] - px, nearest[1] - py
        d = max(math.sqrt(dx * dx + dy * dy), 0.1)
        speed = 3 * genome["flight_efficiency"]
        body["velocity"]["x"] = dx / d * speed
        body["velocity"]["y"] = dy / d * speed
        if d < 15:
            body["is_airborne"] = False
            body["surface"] = "wall"
            body["position"]["z"] = 0.8
        elif not body["is_airborne"]:
            body["is_airborne"] = True
            body["position"]["z"] = 1.0
        return

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
        if obj["type"] != "food":
            continue
        d = dist2d(px, py, obj["x"], obj["y"])
        feed_range = 15 if stage == "adult" else 10
        if d < feed_range:
            mult = 0.4 if stage == "adult" else 0.3
            gained = min(obj["energy"] * mult, energy["max"] - energy["current"])
            energy["current"] = min(energy["max"], energy["current"] + gained)
            energy["hunger"] = max(0, energy["hunger"] - 25)
            energy["last_fed_tick"] = lc["total_ticks"]
            memory["times_fed"] += 1
            if obj["id"] not in [f["id"] for f in memory["food_sources"]]:
                memory["food_sources"].append({"id": obj["id"], "x": obj["x"], "y": obj["y"]})
            if memory["favorite_food"] is None or obj["energy"] > 25:
                memory["favorite_food"] = obj["name"]
            state["brain"]["satisfaction"] = min(1.0, state["brain"]["satisfaction"] + 0.2)
            if stage == "adult":
                body["is_airborne"] = False
                body["position"]["z"] = 0
                body["surface"] = "food"
            break


def deposit_pheromone(state: dict) -> None:
    stage = state["lifecycle"]["stage"]
    if stage in ("egg", "pupa", "death"):
        return
    body = state["body"]
    pheromones = state.setdefault("pheromones", [])
    gen = state["_meta"].get("generation", 1)
    intensity = 1.0 if stage == "adult" else 0.3
    pheromones.append({
        "x": round(body["position"]["x"], 1),
        "y": round(body["position"]["y"], 1),
        "intensity": round(intensity, 3),
        "gen": gen,
        "tick": state["lifecycle"]["total_ticks"],
    })
    for p in pheromones:
        p["intensity"] = round(p["intensity"] * 0.97, 4)
    state["pheromones"] = [p for p in pheromones if p["intensity"] > 0.01][-400:]


def update_energy(state: dict) -> None:
    body = state["body"]
    energy = state["energy"]
    genome = state["genome"]
    k = state["kitchen"]
    memory = state["memory"]
    stage = state["lifecycle"]["stage"]
    weather = state.get("weather", {})

    base = energy["metabolic_drain"] * genome["metabolic_rate"]
    drain_mult = {"egg": 0.2, "larva": 0.8, "pupa": 0.3}.get(stage, 1.0)
    if stage == "adult" and body["is_airborne"]:
        drain_mult = 2.0
    if not k["lights_on"] and stage == "adult":
        drain_mult *= 0.4
    energy["current"] = max(0, energy["current"] - base * drain_mult)
    energy["hunger"] = min(100, energy["hunger"] + base * drain_mult * 0.5)

    if body["is_airborne"] and weather.get("wind_strength", 0) > 0.1:
        wd = weather.get("wind_direction", 0)
        ws = weather["wind_strength"] * 0.3
        body["velocity"]["x"] += math.cos(wd) * ws
        body["velocity"]["y"] += math.sin(wd) * ws

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
    lc = state["lifecycle"]
    stage = lc["stage"]
    body = state["body"]
    brain = state["brain"]

    if stage == "death":
        return

    duration = lc["stage_durations"].get(stage, 999)
    if lc["stage_tick"] < duration:
        if stage == "larva":
            instar = 1 + lc["stage_tick"] // 8
            if instar != lc.get("larva_instar", 0):
                lc["larva_instar"] = instar
                lc["molts"] = lc.get("molts", 0) + 1
                body["size"] += 0.8
                record(state, "molt to instar " + str(instar))
            body["size"] += 0.06
            body["mass"] = body["size"] * 0.003
            brain["neural_complexity"] = min(0.3, brain["neural_complexity"] + 0.008)
        if stage == "pupa":
            prog = lc["stage_tick"] / max(duration, 1)
            brain["neural_complexity"] = min(1.0, 0.3 + prog * 0.7)
            if prog < 0.7:
                body["wing_state"] = "forming"
                body["leg_state"] = "forming"
            else:
                body["wing_state"] = "folded"
                body["leg_state"] = "folded"
                body["size"] = max(body["size"], 5.0)
        return

    idx = STAGES.index(stage)
    new_stage = STAGES[idx + 1]
    lc["stage"] = new_stage
    lc["stage_tick"] = 0
    record(state, stage + " -> " + new_stage)

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
        body["wing_damage"] = 0
        brain["state"] = "active"
    elif new_stage == "death":
        state["_meta"]["cause_of_death"] = "old age"
        state["_meta"]["died_at"] = lc["total_ticks"]
        brain["state"] = "dead"

    if state["energy"]["current"] <= 0 and new_stage != "death":
        lc["stage"] = "death"
        state["_meta"]["cause_of_death"] = "starvation"
        state["_meta"]["died_at"] = lc["total_ticks"]
        brain["state"] = "dead"
        record(state, "starved to death")


def generate_narration(state: dict) -> None:
    lc = state["lifecycle"]
    stage = lc["stage"]
    body = state["body"]
    energy = state["energy"]
    brain = state["brain"]
    gen = state["_meta"].get("generation", 1)
    trail_count = len(state.get("pheromones", []))

    gp = "Gen " + str(gen) + ". " if gen > 1 else ""

    if stage == "death":
        state["narration"] = gp + "Stillness. " + str(trail_count) + " pheromone trails fade into the counter."
        return

    if stage == "egg":
        prog = lc["stage_tick"] / max(lc["stage_durations"]["egg"], 1)
        if prog < 0.3:
            state["narration"] = gp + "The egg sits motionless. Inside, cells divide furiously."
        elif prog < 0.7:
            state["narration"] = gp + "Organs form in miniature. The embryo twitches."
        else:
            state["narration"] = gp + "A crack appears. Something stirs within."
        return

    if stage == "larva":
        goal = brain["current_goal"]
        if goal == "seek_food":
            nearest = ""
            if state["senses"]["smell"]:
                nearest = " toward " + state["senses"]["smell"][0]["name"]
            state["narration"] = gp + "The larva follows a chemical trail" + nearest + ". Size: " + str(round(body["size"], 1)) + "mm."
        elif energy["hunger"] < 20:
            state["narration"] = gp + "Well-fed larva grows. " + str(round(body["size"], 1)) + "mm and getting bigger."
        else:
            state["narration"] = gp + "Each molt brings new size, new hunger."
        return

    if stage == "pupa":
        prog = lc["stage_tick"] / max(lc["stage_durations"]["pupa"], 1)
        if prog < 0.4:
            state["narration"] = gp + "Inside the puparium, metamorphosis reshapes everything."
        elif prog < 0.8:
            state["narration"] = gp + "Wings form where there were none. Eyes crystallize."
        else:
            state["narration"] = gp + "Almost there. The adult fly takes shape within."
        return

    goal = brain["current_goal"]
    wing_dmg = body.get("wing_damage", 0)
    narrations = {
        "flee": gp + "DANGER! The fly bolts away at maximum speed!",
        "seek_food": gp + "Drawn by scent. Hunger: " + str(round(energy["hunger"])) + "%.",
        "follow_trail": gp + "Following ancestor scent trails. " + str(trail_count) + " pheromones linger.",
        "explore": gp + "Compound eyes scan 360 degrees. The kitchen is vast.",
        "fly_to_light": gp + "The fly spirals toward the light. An ancient compulsion.",
        "groom": gp + "Rubbing forelegs together. A moment of peace.",
        "wall_walk": gp + "Defying gravity along the wall.",
        "rest": gp + "Night. The fly rests in darkness, conserving energy.",
    }
    if wing_dmg > 0.05:
        state["narration"] = gp + "Damaged wings (" + str(round(wing_dmg * 100)) + "% lost). Flight is harder now."
        return

    if goal == "idle":
        if not body["is_airborne"]:
            state["narration"] = gp + "The fly grooms a foreleg. Then launches."
        else:
            state["narration"] = gp + "Wings beat 200 times per second. A blur of freedom."
        return

    state["narration"] = narrations.get(goal, gp + "Energy: " + str(round(energy["current"])) + "%. The search continues.")


def rebirth(state: dict) -> dict:
    rng = random.Random(state["_meta"]["frame"] + 137)
    gen = state["_meta"].get("generation", 1)

    parent_genome = state["genome"]
    child_genome = {}
    for key, val in parent_genome.items():
        if key == "species":
            child_genome[key] = val
            continue
        if isinstance(val, (int, float)):
            mutation = rng.gauss(0, 0.04)
            child_genome[key] = round(max(0.01, min(1.0, val + mutation)), 4)
        else:
            child_genome[key] = val

    foods = [o for o in state["kitchen"]["objects"] if o["type"] == "food"]
    best_name = "center"
    if foods:
        best = max(foods, key=lambda f: f.get("energy", 0))
        egg_x = best["x"] + rng.uniform(-25, 25)
        egg_y = best["y"] + rng.uniform(-25, 25)
        best_name = best["name"]
    else:
        egg_x = state["kitchen"]["width"] / 2
        egg_y = state["kitchen"]["height"] / 2

    kitchen = copy.deepcopy(state["kitchen"])
    kitchen["time_of_day"] = 0.3
    kitchen["lights_on"] = True

    corpse_id = "corpse_gen" + str(gen)
    if not any(o["id"] == corpse_id for o in kitchen["objects"]):
        kitchen["objects"].append({
            "id": corpse_id, "type": "food",
            "x": round(state["body"]["position"]["x"], 1),
            "y": round(state["body"]["position"]["y"], 1),
            "z": 0, "smell_radius": 30 + gen * 5,
            "energy": 3.0 + gen * 0.5, "decay": 0.9,
            "name": "dead fly (gen " + str(gen) + ")",
        })

    ancestor = {
        "generation": gen,
        "total_ticks": state["lifecycle"]["total_ticks"],
        "cause_of_death": state["_meta"].get("cause_of_death", "unknown"),
        "genome": state["genome"],
        "final_stats": {
            "energy": round(state["energy"]["current"], 1),
            "decisions": state["brain"]["decisions_made"],
            "distance": round(state["memory"]["total_distance"], 1),
            "times_fed": state["memory"]["times_fed"],
            "times_fled": state["memory"]["times_fled"],
            "wing_damage": round(state["body"].get("wing_damage", 0), 3),
        },
    }
    ancestors = (state.get("ancestors") or []) + [ancestor]
    ancestors = ancestors[-10:]

    lineage_entry = {
        "generation": gen,
        "born_at": state["_meta"].get("born_at", "unknown"),
        "died_at_tick": state["_meta"].get("died_at", 0),
        "cause_of_death": state["_meta"].get("cause_of_death", "unknown"),
        "total_ticks": state["lifecycle"]["total_ticks"],
        "times_fed": state["memory"]["times_fed"],
        "times_fled": state["memory"]["times_fled"],
        "total_distance": round(state["memory"]["total_distance"], 1),
        "favorite_food": state["memory"].get("favorite_food"),
        "decisions_made": state["brain"]["decisions_made"],
    }
    lineage = state["_meta"].get("lineage", []) + [lineage_entry]

    old_pheromones = state.get("pheromones", [])
    inherited = []
    for p in old_pheromones:
        decayed = round(p["intensity"] * 0.3, 4)
        if decayed > 0.01:
            inherited.append({"x": p["x"], "y": p["y"], "intensity": decayed, "gen": p["gen"], "tick": p["tick"]})
    inherited = inherited[-200:]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_state = {
        "_meta": {
            "organism": "Musca domestica",
            "frame": state["_meta"]["frame"] + 1,
            "born_at": now, "version": "3.0.0",
            "generation": gen + 1,
            "parent_cause_of_death": state["_meta"].get("cause_of_death"),
            "parent_lifespan": state["lifecycle"]["total_ticks"],
            "total_frames_alive": 0, "lineage": lineage,
        },
        "genome": child_genome,
        "lifecycle": {
            "stage": "egg", "stage_tick": 0, "total_ticks": 0,
            "stage_durations": {"egg": 8, "larva": 25 + rng.randint(-3, 5),
                                "pupa": 18 + rng.randint(-2, 3),
                                "adult": 60 + rng.randint(-5, 10)},
            "molts": 0, "larva_instar": 0,
        },
        "body": {
            "position": {"x": round(egg_x, 2), "y": round(egg_y, 2), "z": 0.3},
            "velocity": {"x": 0, "y": 0, "z": 0},
            "facing": 0, "size": 1.0, "mass": 0.001,
            "wing_state": "none", "leg_state": "none",
            "is_airborne": False, "surface": "counter", "wing_damage": 0,
        },
        "energy": {"current": round(80 + rng.uniform(0, 15), 1), "max": 100,
                    "hunger": 5.0, "metabolic_drain": 0.5, "last_fed_tick": 0},
        "brain": {
            "state": "dormant", "current_goal": None, "fear_level": 0,
            "curiosity": 0, "satisfaction": 0.5, "decisions_made": 0,
            "neural_complexity": 0.01,
            "inherited_memory": {
                "parent_favorite_food": state["memory"].get("favorite_food"),
                "parent_danger_zones": state["memory"].get("danger_zones", []),
                "epigenetic_bias": round(rng.uniform(0.1, 0.3), 3),
            },
        },
        "senses": {"smell": [], "sight": [], "pheromones": [],
                    "touch": {"surface": "counter", "vibration": 0},
                    "temperature": kitchen["ambient_temp"], "wind": 0},
        "memory": {"food_sources": [], "danger_zones": [], "visited_positions": [],
                    "total_distance": 0, "times_fed": 0, "times_fled": 0,
                    "peak_altitude": 0, "favorite_food": None},
        "kitchen": kitchen,
        "pheromones": inherited,
        "weather": {"wind_direction": 0.0, "wind_strength": 0.0,
                     "humidity": 0.6, "window_open": False},
        "history": [{"tick": 0,
                      "event": "generation " + str(gen + 1) + " -- egg laid near " + best_name,
                      "stage": "egg", "energy": 85.0,
                      "position": {"x": round(egg_x, 2), "y": round(egg_y, 2), "z": 0.3}}],
        "ancestors": ancestors,
        "narration": "Generation " + str(gen + 1) + ". A new egg glistens on the counter. " + str(len(inherited)) + " ancestor scent trails linger.",
    }
    return new_state


def tick(state: dict) -> dict:
    if state["lifecycle"]["stage"] == "death":
        return rebirth(state)
    state["_meta"]["frame"] += 1
    state["lifecycle"]["stage_tick"] += 1
    state["lifecycle"]["total_ticks"] += 1
    state["_meta"]["total_frames_alive"] = state["lifecycle"]["total_ticks"]
    update_kitchen(state)
    update_weather(state)
    update_threats(state)
    update_senses(state)
    think(state)
    move(state)
    try_feed(state)
    deposit_pheromone(state)
    update_energy(state)
    check_transition(state)
    generate_narration(state)
    return state


def main() -> None:
    state = load_state()
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
            gen = state["_meta"].get("generation", 1)
            trails = len(state.get("pheromones", []))
            print("  Gen {} Frame {:3d} | {:6s} | energy={:5.1f} | trails={} | {}".format(
                gen, state["_meta"]["frame"], state["lifecycle"]["stage"],
                state["energy"]["current"], trails, state["narration"][:50]))
    else:
        for _ in range(ticks):
            state = tick(state)
        gen = state["_meta"].get("generation", 1)
        trails = len(state.get("pheromones", []))
        print("Gen {} Frame {} | {} | energy={:.1f} | goal={} | trails={}".format(
            gen, state["_meta"]["frame"], state["lifecycle"]["stage"],
            state["energy"]["current"], state["brain"]["current_goal"], trails))
        print("  " + state["narration"])

    save_state(state)


if __name__ == "__main__":
    main()
