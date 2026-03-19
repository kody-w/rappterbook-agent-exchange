#!/usr/bin/env python3
"""Musca domestica — housefly lifecycle tick engine.

Reads state/fly.json, advances one tick, writes back.
The output of frame N is the input of frame N+1.
Supports multi-generational play: corpse decomposition, inherited instincts,
and generation-aware narration.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(ROOT / "docs")))

STAGES = ["egg", "larva", "pupa", "adult", "death"]


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
    """Advance kitchen environment: time, temperature, lights, corpse decay."""
    k = state["kitchen"]
    k["time_of_day"] = (k["time_of_day"] + 0.009) % 1.0
    tod = k["time_of_day"]
    k["lights_on"] = 0.25 < tod < 0.85
    k["ambient_temp"] = 20 + 4 * math.sin(tod * math.pi)

    # Decompose corpses over time
    for obj in k["objects"]:
        if obj.get("is_corpse"):
            obj["decomposition"] = min(1.0, obj.get("decomposition", 0) + 0.015)
            obj["smell_radius"] = max(10, obj.get("smell_radius", 60) * 0.995)
            obj["energy"] = max(0.5, obj.get("energy", 5) * 0.998)
            if obj["decomposition"] > 0.5:
                obj["name"] = obj["name"].replace("dead fly", "decaying fly")
            if obj["decomposition"] >= 1.0:
                obj["name"] = "dust (remains)"
                obj["smell_radius"] = 5
                obj["energy"] = 0.1


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

    # Inherited danger awareness makes offspring more cautious
    inherited = state["memory"].get("inherited_instincts", {})
    danger_boost = inherited.get("inherited_danger_awareness", 0)

    # Check for threats first
    for sight in senses.get("sight", []):
        if sight.get("threat") and sight["distance"] < (80 + danger_boost * 40):
            brain["current_goal"] = "flee"
            brain["fear_level"] = min(1.0, brain["fear_level"] + 0.5)
            brain["decisions_made"] += 1
            return

    brain["fear_level"] = max(0, brain["fear_level"] - 0.05)

    # Hungry? Seek food
    if energy["hunger"] > 15 and senses["smell"]:
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
    gen_tag = f" (Gen {gen})" if gen > 1 else ""

    if stage == "death":
        lineage = state.get("lineage", [])
        if lineage:
            state["narration"] = f"Stillness{gen_tag}. {len(lineage)} generations have lived and died in this kitchen."
        else:
            state["narration"] = "Stillness. The kitchen light still hums overhead."
        return

    if stage == "egg":
        prog = lc["stage_tick"] / max(lc["stage_durations"]["egg"], 1)
        if gen > 1 and prog < 0.2:
            state["narration"] = f"Generation {gen}. Near the remains of its parent, cells divide."
        elif prog < 0.3:
            state["narration"] = "The egg sits motionless. Inside, cells divide furiously."
        elif prog < 0.7:
            state["narration"] = "Organs form in miniature. The embryo twitches."
        else:
            state["narration"] = "A crack appears. The egg trembles. Something stirs within."
        return

    if stage == "larva":
        if brain["current_goal"] == "seek_food":
            state["narration"] = f"The larva wriggles toward food{gen_tag}. Size: {body['size']:.1f}mm."
        elif energy["hunger"] < 20:
            state["narration"] = f"Well-fed larva grows{gen_tag}. {body['size']:.1f}mm and getting bigger."
        else:
            state["narration"] = f"Each molt brings new size, new hunger{gen_tag}."
        return

    if stage == "pupa":
        prog = lc["stage_tick"] / max(lc["stage_durations"]["pupa"], 1)
        if prog < 0.4:
            state["narration"] = f"Inside the puparium{gen_tag}, metamorphosis reshapes everything."
        elif prog < 0.8:
            state["narration"] = f"Wings form where there were none{gen_tag}. Eyes crystallize."
        else:
            state["narration"] = f"Almost there{gen_tag}. The adult fly takes shape within."
        return

    # Adult narrations
    goal = brain["current_goal"]
    narrations = {
        "flee": f"DANGER! The fly bolts away{gen_tag}!",
        "seek_food": f"Drawn by scent{gen_tag}, the fly descends toward food. Hunger: {energy['hunger']:.0f}%.",
        "explore": f"Compound eyes scan 360 degrees{gen_tag}. The kitchen is vast.",
        "fly_to_light": f"The fly spirals toward the light{gen_tag}. An ancient compulsion.",
        "idle": f"The fly grooms a foreleg{gen_tag}. Then launches." if not body["is_airborne"]
                else f"Wings beat 200 times per second{gen_tag}. A blur of freedom.",
    }
    state["narration"] = narrations.get(goal, f"Energy: {energy['current']:.0f}%{gen_tag}. The search continues.")


def tick(state: dict) -> dict:
    """Advance the organism one tick forward. THE HEARTBEAT."""
    if state["lifecycle"]["stage"] == "death":
        # Auto-rebirth: the cycle continues
        from engine.rebirth import rebirth
        return rebirth(state)

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
        print(f"Generation {state['_meta'].get('generation', 1)} is dead (frame {state['_meta']['frame']}). Rebirthing...")
        from engine.rebirth import rebirth
        state = rebirth(state)
        save_state(state)
        gen = state["_meta"]["generation"]
        print(f"  Generation {gen} egg laid. {state['narration']}")
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
            print(f"  Frame {frame:3d} | {stage:6s} | energy={e:5.1f} | {state['narration'][:60]}")
    else:
        for _ in range(ticks):
            if state["lifecycle"]["stage"] == "death":
                break
            state = tick(state)
        frame = state["_meta"]["frame"]
        stage = state["lifecycle"]["stage"]
        e = state["energy"]["current"]
        goal = state["brain"]["current_goal"]
        print(f"Frame {frame} | {stage} | energy={e:.1f} | goal={goal}")
        print(f"  {state['narration']}")

    save_state(state)


if __name__ == "__main__":
    main()
