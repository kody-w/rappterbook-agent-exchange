#!/usr/bin/env python3
"""Musca domestica — housefly lifecycle tick engine.

Reads state/fly.json, advances one tick, writes back.
The output of frame N is the input of frame N+1.

Lifecycle: egg → larva → pupa → adult → death → decomposing → REBIRTH (new egg)
Death is not the end. The body decomposes, nutrients cycle back, and a new
generation emerges with a mutated genome and epigenetic memory from its parent.
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
DECOMP_STAGES = ["fresh", "bloat", "active_decay", "advanced_decay", "dry", "skeletal"]
DECOMP_DURATION = 12


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
        "tick": state["_meta"]["frame"],
        "event": event,
        "stage": state["lifecycle"]["stage"],
        "energy": round(state["energy"]["current"], 1),
        "position": copy.deepcopy(state["body"]["position"]),
        "generation": state["_meta"].get("generation", 1),
    })
    if len(state["history"]) > 300:
        state["history"] = state["history"][-200:]


def update_kitchen(state: dict) -> None:
    """Advance kitchen environment: time, temperature, lights."""
    k = state["kitchen"]
    k["time_of_day"] = (k["time_of_day"] + 0.009) % 1.0
    tod = k["time_of_day"]
    k["lights_on"] = 0.25 < tod < 0.85
    k["ambient_temp"] = 20 + 4 * math.sin(tod * math.pi)

    # Food slowly decays over generations
    for obj in k["objects"]:
        if obj["type"] == "food":
            obj["decay"] = min(1.0, obj.get("decay", 0.5) + 0.001)


def update_threats(state: dict) -> None:
    """Randomly spawn/despawn threats."""
    k = state["kitchen"]
    for obj in k["objects"]:
        if obj["type"] != "threat":
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
        if obj["type"] == "food":
            sr = obj.get("smell_radius", 80) * genome["smell_sensitivity"]
            if d < sr:
                smells.append({
                    "id": obj["id"], "name": obj["name"],
                    "distance": round(d, 1),
                    "intensity": round(1 - d / sr, 2)
                })
        if obj["type"] == "light" and d < 350:
            sights.append({
                "id": obj["id"], "name": obj["name"],
                "distance": round(d, 1),
                "intensity": obj.get("intensity", 0.5)
            })
        if obj["type"] == "threat" and obj.get("active") and d < 120:
            sights.append({
                "id": obj["id"], "name": obj["name"],
                "distance": round(d, 1),
                "threat": True
            })

    senses["smell"] = smells
    senses["sight"] = sights
    senses["touch"]["surface"] = "air" if body["is_airborne"] else body.get("surface", "counter")


def think(state: dict) -> None:
    """Brain decision-making — set current_goal based on senses."""
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

    # Epigenetic: inherited food knowledge gives slight bias toward known food
    inherited = state["memory"].get("inherited_food_knowledge", [])

    # Check for threats first
    for sight in senses.get("sight", []):
        if sight.get("threat") and sight["distance"] < 80:
            brain["current_goal"] = "flee"
            brain["fear_level"] = min(1.0, brain["fear_level"] + 0.5)
            brain["decisions_made"] += 1
            return

    brain["fear_level"] = max(0, brain["fear_level"] - 0.05)

    # Hungry? Seek food (with inherited preference)
    if energy["hunger"] > 15 and senses["smell"]:
        brain["current_goal"] = "seek_food"
        brain["decisions_made"] += 1
        return

    # Epigenetic memory: lower hunger threshold if parent knew food locations
    if inherited and energy["hunger"] > 8 and senses["smell"]:
        known = [s for s in senses["smell"] if s["id"] in inherited]
        if known:
            brain["current_goal"] = "seek_food"
            brain["decisions_made"] += 1
            return

    # Curious? Explore
    if brain["curiosity"] > 0.5 and random.random() < 0.4:
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
                if not body["is_airborne"]:
                    body["is_airborne"] = True
                    body["position"]["z"] = 1.5
                memory["times_fled"] = memory.get("times_fled", 0) + 1
        return

    if goal == "seek_food":
        smells = senses.get("smell", [])
        if smells:
            # Prefer inherited food sources if available
            inherited = memory.get("inherited_food_knowledge", [])
            known = [s for s in smells if s["id"] in inherited]
            target = max(known or smells, key=lambda s: s["intensity"])
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
    """Try to eat if near food."""
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
        state["_meta"]["died_at"] = state["_meta"]["frame"]
        brain["state"] = "dead"

    # Check starvation
    if state["energy"]["current"] <= 0 and new_stage != "death":
        lc["stage"] = "death"
        state["_meta"]["cause_of_death"] = "starvation"
        state["_meta"]["died_at"] = state["_meta"]["frame"]
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

    if stage == "death":
        state["narration"] = "Stillness. The kitchen light still hums overhead."
        return

    gen_prefix = f"[Gen {gen}] " if gen > 1 else ""

    if stage == "egg":
        prog = lc["stage_tick"] / max(lc["stage_durations"]["egg"], 1)
        if prog < 0.3:
            state["narration"] = gen_prefix + "The egg sits motionless. Inside, cells divide furiously."
        elif prog < 0.7:
            state["narration"] = gen_prefix + "Organs form in miniature. The embryo twitches."
        else:
            state["narration"] = gen_prefix + "A crack appears. The egg trembles. Something stirs within."
        return

    if stage == "larva":
        if brain["current_goal"] == "seek_food":
            state["narration"] = gen_prefix + f"The larva wriggles toward food. Size: {body['size']:.1f}mm."
        elif energy["hunger"] < 20:
            state["narration"] = gen_prefix + f"Well-fed larva grows. {body['size']:.1f}mm and getting bigger."
        else:
            state["narration"] = gen_prefix + "Each molt brings new size, new hunger."
        return

    if stage == "pupa":
        prog = lc["stage_tick"] / max(lc["stage_durations"]["pupa"], 1)
        if prog < 0.4:
            state["narration"] = gen_prefix + "Inside the puparium, metamorphosis reshapes everything."
        elif prog < 0.8:
            state["narration"] = gen_prefix + "Wings form where there were none. Eyes crystallize."
        else:
            state["narration"] = gen_prefix + "Almost there. The adult fly takes shape within."
        return

    # Adult narrations
    goal = brain["current_goal"]
    inherited = state["memory"].get("inherited_food_knowledge", [])
    inherit_note = " (instinct guides it)" if inherited and goal == "seek_food" else ""
    narrations = {
        "flee": gen_prefix + "DANGER! The fly bolts away at maximum speed!",
        "seek_food": gen_prefix + f"Drawn by scent, the fly descends toward food{inherit_note}. Hunger: {energy['hunger']:.0f}%.",
        "explore": gen_prefix + "Compound eyes scan 360 degrees. The kitchen is vast.",
        "fly_to_light": gen_prefix + "The fly spirals toward the light. An ancient compulsion.",
        "idle": gen_prefix + ("The fly grooms a foreleg. Then launches." if not body["is_airborne"]
                else "Wings beat 200 times per second. A blur of freedom."),
    }
    state["narration"] = narrations.get(goal, gen_prefix + f"Energy: {energy['current']:.0f}%. The search continues.")


# ─── POST-MORTEM: DECOMPOSITION & REBIRTH ─────────────────────────────

def tick_postmortem(state: dict) -> dict:
    """Post-mortem tick: decomposition → nutrient cycling → rebirth.

    The body breaks down through forensic stages. Bacteria multiply.
    Nutrients release. Other organisms are attracted. After full
    decomposition, a new generation egg is laid nearby — the cycle continues.
    """
    state["_meta"]["frame"] += 1

    # Initialize decomposition tracking
    if "decomposition" not in state:
        state["decomposition"] = {
            "progress": 0.0,
            "bacteria_count": 0,
            "nutrients_released": 0.0,
            "stage": "fresh",
            "ticks_dead": 0,
            "attracted_organisms": [],
            "chitin_remaining": 1.0,
        }
        record(state, "decomposition begins")
        state["narration"] = "The body cools. Rigor mortis sets in. Bacteria begin their silent work."
        return state

    decomp = state["decomposition"]
    decomp["ticks_dead"] += 1

    # Decomposition rate affected by temperature
    temp = state["kitchen"].get("ambient_temp", 22)
    temp_factor = 0.8 + (temp - 20) * 0.05
    rate = (1.0 / DECOMP_DURATION) * temp_factor
    decomp["progress"] = min(1.0, decomp["progress"] + rate)

    # Bacteria grow logistically
    carrying_capacity = 200
    growth = 0.4
    b = decomp["bacteria_count"]
    decomp["bacteria_count"] = int(b + growth * b * (1 - b / carrying_capacity)) if b > 0 else 3

    # Nutrients released proportional to progress
    decomp["nutrients_released"] = round(decomp["progress"] * state["body"]["mass"] * 1000, 2)

    # Chitin degrades slowly
    decomp["chitin_remaining"] = max(0, 1.0 - decomp["progress"] * 0.7)

    # Forensic decomposition stages
    prev_stage = decomp["stage"]
    if decomp["progress"] < 0.15:
        decomp["stage"] = "fresh"
    elif decomp["progress"] < 0.3:
        decomp["stage"] = "bloat"
    elif decomp["progress"] < 0.55:
        decomp["stage"] = "active_decay"
    elif decomp["progress"] < 0.75:
        decomp["stage"] = "advanced_decay"
    elif decomp["progress"] < 0.92:
        decomp["stage"] = "dry"
    else:
        decomp["stage"] = "skeletal"

    if decomp["stage"] != prev_stage:
        record(state, f"decomposition: {decomp['stage']}")

    # Attracted organisms arrive
    if decomp["progress"] > 0.1 and random.random() < 0.35:
        pool = {
            "fresh": ["blow fly", "flesh fly"],
            "bloat": ["blow fly", "beetle", "mite"],
            "active_decay": ["beetle", "mite", "springtail", "ant"],
            "advanced_decay": ["mite", "springtail", "ant", "moth fly"],
            "dry": ["dermestid beetle", "spider", "ant"],
            "skeletal": ["mite", "dust mite"],
        }
        candidates = pool.get(decomp["stage"], ["mite"])
        new_org = random.choice(candidates)
        if new_org not in decomp["attracted_organisms"]:
            decomp["attracted_organisms"].append(new_org)
            record(state, f"{new_org} arrives at the remains")
        if len(decomp["attracted_organisms"]) > 8:
            decomp["attracted_organisms"] = decomp["attracted_organisms"][-8:]

    # Kitchen continues evolving
    update_kitchen(state)

    # Narration
    narrations = {
        "fresh": "The body cools. Cellular autolysis begins. Bacteria multiply inside.",
        "bloat": "Gases inflate the abdomen. The smell intensifies. A blow fly investigates.",
        "active_decay": "Enzymes dissolve tissue. The counter beneath darkens. Nutrients flow outward.",
        "advanced_decay": "Most soft tissue is gone. A dark stain marks where life once buzzed.",
        "dry": "Cartilage and dried tissue remain. The chitin exoskeleton persists.",
        "skeletal": "Only chitin fragments remain. A ghost. But in decay, there is an invitation...",
    }
    state["narration"] = narrations.get(decomp["stage"], "Decomposition continues.")

    # REBIRTH — decomposition complete
    if decomp["progress"] >= 1.0:
        state["narration"] = "From death, nutrients. From nutrients, attraction. From attraction... new life."
        respawn(state)

    return state


def respawn(state: dict) -> None:
    """Spawn a new generation from the nutrients of the old.

    A female fly, attracted by the decomposition scent, lays an egg
    near the remains. The genome carries slight mutations.
    Epigenetic memory of the parent's food sources passes on.
    """
    old_genome = copy.deepcopy(state["genome"])
    old_meta = copy.deepcopy(state["_meta"])
    old_memory = copy.deepcopy(state["memory"])
    old_history = state["history"]
    parent_pos = copy.deepcopy(state["body"]["position"])
    frame = state["_meta"]["frame"]

    gen = old_meta.get("generation", 1) + 1

    # Archive this generation in lineage
    lineage_entry = {
        "generation": old_meta.get("generation", 1),
        "born_at": old_meta.get("born_at"),
        "died_at": old_meta.get("died_at"),
        "cause_of_death": old_meta.get("cause_of_death"),
        "total_ticks_alive": old_meta.get("total_frames_alive", 0),
        "genome_snapshot": {k: round(v, 3) for k, v in old_genome.items() if isinstance(v, (int, float))},
        "stats": {
            "times_fed": old_memory.get("times_fed", 0),
            "times_fled": old_memory.get("times_fled", 0),
            "total_distance": round(old_memory.get("total_distance", 0), 1),
            "favorite_food": old_memory.get("favorite_food"),
            "decisions_made": state["brain"].get("decisions_made", 0),
        },
    }

    lineage = old_meta.get("lineage", [])
    lineage.append(lineage_entry)

    # Mutate genome for next generation
    new_genome = {}
    for key, val in old_genome.items():
        if isinstance(val, float):
            mutation = random.gauss(0, 0.04)
            new_genome[key] = round(max(0.01, min(1.0, val + mutation)), 4)
        elif isinstance(val, int) and key == "eye_facets":
            new_genome[key] = max(2000, val + random.randint(-200, 200))
        elif key == "species":
            new_genome[key] = val
        else:
            new_genome[key] = val

    # New egg near the remains
    egg_x = clamp(parent_pos["x"] + random.uniform(-30, 30), 10, state["kitchen"]["width"] - 10)
    egg_y = clamp(parent_pos["y"] + random.uniform(-30, 30), 10, state["kitchen"]["height"] - 10)

    # Slightly varied stage durations (natural variation)
    state["_meta"] = {
        "organism": "Musca domestica",
        "frame": frame,
        "born_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.1.0",
        "cause_of_death": None,
        "died_at": None,
        "total_frames_alive": 0,
        "generation": gen,
        "lineage": lineage,
    }

    state["genome"] = new_genome

    state["lifecycle"] = {
        "stage": "egg",
        "stage_tick": 0,
        "total_ticks": 0,
        "stage_durations": {
            "egg": max(5, 8 + random.randint(-2, 2)),
            "larva": max(15, 25 + random.randint(-5, 5)),
            "pupa": max(12, 18 + random.randint(-3, 3)),
            "adult": max(40, 60 + random.randint(-10, 10)),
        },
        "molts": 0,
        "larva_instar": 0,
    }

    state["body"] = {
        "position": {"x": round(egg_x, 2), "y": round(egg_y, 2), "z": 0.5},
        "velocity": {"x": 0, "y": 0, "z": 0},
        "facing": round(random.uniform(0, 2 * math.pi), 4),
        "size": 1.2,
        "mass": 0.001,
        "wing_state": "none",
        "leg_state": "none",
        "is_airborne": False,
        "surface": "counter",
    }

    state["energy"] = {
        "current": 82,
        "max": 100,
        "hunger": 5,
        "metabolic_drain": 0.5,
        "last_fed_tick": 0,
    }

    state["brain"] = {
        "state": "dormant",
        "current_goal": None,
        "fear_level": 0,
        "curiosity": 0,
        "satisfaction": 0.5,
        "decisions_made": 0,
        "neural_complexity": 0.01,
    }

    state["senses"] = {
        "smell": [],
        "sight": [],
        "touch": {"surface": "counter", "vibration": 0},
        "temperature": state["kitchen"]["ambient_temp"],
        "wind": 0,
    }

    state["memory"] = {
        "food_sources": [],
        "danger_zones": [],
        "visited_positions": [],
        "total_distance": 0,
        "times_fed": 0,
        "times_fled": 0,
        "peak_altitude": 0,
        "favorite_food": None,
        "inherited_food_knowledge": [f["id"] for f in old_memory.get("food_sources", [])],
        "inherited_danger_zones": old_memory.get("danger_zones", []),
        "parent_favorite_food": old_memory.get("favorite_food"),
    }

    # Clean up decomposition
    if "decomposition" in state:
        del state["decomposition"]

    # Preserve history across generations
    state["history"] = old_history
    record(state, f"REBIRTH — generation {gen} egg laid")

    state["narration"] = f"From death, new life. Generation {gen} begins. A tiny egg glistens on the counter."


# ─── MAIN TICK ─────────────────────────────────────────────────────────

def tick(state: dict) -> dict:
    """Advance the organism one tick forward. THE HEARTBEAT."""
    if state["lifecycle"]["stage"] == "death":
        return tick_postmortem(state)

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
    args = sys.argv[1:]
    ticks = 1
    until_death = False
    until_rebirth = False
    i = 0
    while i < len(args):
        if args[i] == "--ticks" and i + 1 < len(args):
            ticks = int(args[i + 1])
            i += 2
        elif args[i] == "--until" and i + 1 < len(args):
            if args[i + 1] == "death":
                until_death = True
            elif args[i + 1] == "rebirth":
                until_rebirth = True
            i += 2
        else:
            i += 1

    if until_death:
        while state["lifecycle"]["stage"] != "death":
            state = tick(state)
            frame = state["_meta"]["frame"]
            stage = state["lifecycle"]["stage"]
            e = state["energy"]["current"]
            print(f"  Frame {frame:3d} | {stage:6s} | energy={e:5.1f} | {state['narration'][:60]}")
    elif until_rebirth:
        gen = state["_meta"].get("generation", 1)
        while state["_meta"].get("generation", 1) == gen:
            state = tick(state)
            frame = state["_meta"]["frame"]
            stage = state["lifecycle"]["stage"]
            decomp = state.get("decomposition", {})
            e = state["energy"]["current"]
            extra = f" decomp={decomp['progress']:.0%}" if decomp else ""
            print(f"  Frame {frame:3d} | {stage:12s} | energy={e:5.1f}{extra} | {state['narration'][:50]}")
        print(f"\n  === REBIRTH: Generation {state['_meta']['generation']} ===")
    else:
        for _ in range(ticks):
            state = tick(state)
            frame = state["_meta"]["frame"]
            stage = state["lifecycle"]["stage"]
            e = state["energy"]["current"]
            gen = state["_meta"].get("generation", 1)
            decomp = state.get("decomposition", {})
            extra = ""
            if decomp:
                extra = f" | decomp={decomp['progress']:.0%} [{decomp['stage']}]"
            goal = state["brain"]["current_goal"]
            print(f"Frame {frame} | gen {gen} | {stage} | energy={e:.1f} | goal={goal}{extra}")
            print(f"  {state['narration']}")

    save_state(state)


if __name__ == "__main__":
    main()
