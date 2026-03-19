#!/usr/bin/env python3
"""Musca domestica — housefly lifecycle tick engine (v3: kitchen events).

Reads state/fly.json, advances one tick, writes back.
The output of frame N is the input of frame N+1.

v3 additions:
  - Kitchen events: random environmental disturbances (door slam, fridge open,
    spill, wind gust) that shake the fly's world
  - Corpse ecology: parent's body decays, attracting bacteria, changing smell
  - Pupa dreaming: during metamorphosis the brain fires random pattern echoes
  - Stress system: cumulative stress affects decisions and energy drain
  - Sound: flying generates buzz that can attract threats
  - Temperature microclimate: different surfaces have different temps

v2 (inherited):
  - Generational rebirth, inherited memory, kitchen evolution, grooming
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
    """Advance kitchen environment: time, temperature, lights, food decay."""
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
            record(state, obj["name"] + " appears!")
        elif obj.get("active") and random.random() < 0.15:
            obj["active"] = False
            obj["x"] = -100
            obj["y"] = -100


KITCHEN_EVENTS = [
    {"id": "door_slam", "name": "door slams", "vibration": 0.9, "wind": 0.6,
     "temp_delta": -2, "duration": 3, "chance": 0.04},
    {"id": "fridge_open", "name": "fridge opens", "vibration": 0.3, "wind": 0.2,
     "temp_delta": -5, "duration": 5, "chance": 0.03},
    {"id": "water_spill", "name": "water spills on counter", "vibration": 0.5, "wind": 0,
     "temp_delta": -1, "duration": 8, "chance": 0.02},
    {"id": "cooking_steam", "name": "steam rises from stove", "vibration": 0.1, "wind": 0.4,
     "temp_delta": 6, "duration": 10, "chance": 0.025},
    {"id": "window_breeze", "name": "breeze through window", "vibration": 0, "wind": 0.8,
     "temp_delta": -3, "duration": 6, "chance": 0.035},
    {"id": "light_flicker", "name": "light flickers", "vibration": 0, "wind": 0,
     "temp_delta": 0, "duration": 2, "chance": 0.05},
    {"id": "footsteps", "name": "footsteps nearby", "vibration": 0.7, "wind": 0,
     "temp_delta": 0, "duration": 4, "chance": 0.06},
]


def update_kitchen_events(state: dict) -> None:
    """Roll for random environmental disturbances."""
    k = state["kitchen"]
    events = k.setdefault("active_events", [])

    # Tick down existing events
    still_active = []
    for ev in events:
        ev["remaining"] -= 1
        if ev["remaining"] > 0:
            still_active.append(ev)
    k["active_events"] = still_active

    # Roll for new events
    for template in KITCHEN_EVENTS:
        if random.random() < template["chance"]:
            # Don't stack same event
            if any(e["id"] == template["id"] for e in k["active_events"]):
                continue
            ev = {
                "id": template["id"],
                "name": template["name"],
                "vibration": template["vibration"],
                "wind": template["wind"],
                "temp_delta": template["temp_delta"],
                "remaining": template["duration"],
            }
            k["active_events"].append(ev)
            record(state, "kitchen: " + template["name"])

    # Apply event effects to environment
    total_vibration = sum(e["vibration"] for e in k["active_events"])
    total_wind = sum(e["wind"] for e in k["active_events"])
    total_temp = sum(e["temp_delta"] for e in k["active_events"])
    k["event_vibration"] = round(min(1.0, total_vibration), 2)
    k["event_wind"] = round(min(1.0, total_wind), 2)
    k["event_temp_delta"] = round(total_temp, 1)


def update_corpse_ecology(state: dict) -> None:
    """The parent's corpse decays, growing more pungent over time."""
    k = state["kitchen"]
    for obj in k["objects"]:
        if obj["id"] != "carcass":
            continue
        tick = state["lifecycle"]["total_ticks"]
        # Corpse bloats then desiccates
        if tick < 30:
            obj["decay"] = round(min(1.0, obj.get("decay", 0.8) + 0.01), 3)
            obj["smell_radius"] = min(120, obj.get("smell_radius", 40) + 2)
            obj["energy"] = min(12, obj.get("energy", 5) + 0.2)
            obj["name"] = "bloating fly corpse"
        elif tick < 60:
            obj["smell_radius"] = max(20, obj.get("smell_radius", 80) - 1)
            obj["energy"] = max(1, obj.get("energy", 8) - 0.1)
            obj["name"] = "desiccating fly husk"
        else:
            obj["smell_radius"] = max(5, obj.get("smell_radius", 40) - 0.5)
            obj["energy"] = max(0.5, obj.get("energy", 3) - 0.05)
            obj["name"] = "dried fly husk"


def update_stress(state: dict) -> None:
    """Cumulative stress system — loud events and threats raise stress."""
    brain = state["brain"]
    k = state["kitchen"]
    stress = brain.setdefault("stress", 0.0)

    # Events cause stress
    event_vibration = k.get("event_vibration", 0)
    if event_vibration > 0.5:
        stress += event_vibration * 0.08
    if brain.get("fear_level", 0) > 0.3:
        stress += 0.06

    # Natural decay — faster when grooming or feeding
    decay = 0.05
    if brain.get("current_goal") == "groom":
        decay = 0.15
    if brain.get("satisfaction", 0) > 0.5:
        decay += 0.03
    stress = max(0, stress - decay)
    brain["stress"] = round(min(1.0, stress), 3)

    # High stress increases metabolic drain
    if stress > 0.5:
        state["energy"]["metabolic_drain"] = 0.5 + stress * 0.3
    else:
        state["energy"]["metabolic_drain"] = 0.5


def pupa_dream(state: dict) -> None:
    """During metamorphosis, the brain fires random pattern echoes."""
    if state["lifecycle"]["stage"] != "pupa":
        return

    brain = state["brain"]
    dreams = brain.setdefault("dreams", [])
    memory = state["memory"]

    prog = state["lifecycle"]["stage_tick"] / max(
        state["lifecycle"]["stage_durations"].get("pupa", 20), 1
    )

    # Dream intensity peaks mid-metamorphosis
    intensity = math.sin(prog * math.pi)
    if random.random() < intensity * 0.4:
        dream_types = [
            "echo of warmth from the egg",
            "phantom scent of " + (memory.get("favorite_food") or "unknown food"),
            "inherited fear of shadows",
            "wing-beat rhythm forming in neural tissue",
            "compound eye patterns crystallizing",
            "ancestral memory of flight",
            "the taste of the counter surface",
            "a vibration from a world beyond the shell",
        ]
        dream = random.choice(dream_types)
        dreams.append({
            "tick": state["lifecycle"]["total_ticks"],
            "content": dream,
            "intensity": round(intensity, 2),
        })
        # Keep only recent dreams
        brain["dreams"] = dreams[-12:]


def update_senses(state: dict) -> None:
    """Compute what the fly can smell, see, and feel."""
    body = state["body"]
    senses = state["senses"]
    genome = state["genome"]
    k = state["kitchen"]
    stage = state["lifecycle"]["stage"]

    senses["temperature"] = k["ambient_temp"] + k.get("event_temp_delta", 0)
    senses["wind"] = random.uniform(0, 0.3) + k.get("event_wind", 0)
    senses["touch"]["vibration"] = (
        senses["touch"]["vibration"] * 0.7 + k.get("event_vibration", 0)
    )

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
                    "intensity": round(1 - d / sr, 2),
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
    senses["touch"]["surface"] = (
        "air" if body["is_airborne"] else body.get("surface", "counter")
    )


def think(state: dict) -> None:
    """Brain decision-making based on senses and inherited memory."""
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

    # Threats first
    for sight in senses.get("sight", []):
        if sight.get("threat") and sight["distance"] < 80:
            brain["current_goal"] = "flee"
            brain["fear_level"] = min(1.0, brain["fear_level"] + 0.5)
            brain["decisions_made"] += 1
            return

    brain["fear_level"] = max(0, brain["fear_level"] - 0.05)

    hunger_threshold = max(8, 15 - epigenetic * 20)
    if energy["hunger"] > hunger_threshold and senses["smell"]:
        brain["current_goal"] = "seek_food"
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
            parent_fav = brain.get("inherited_memory", {}).get(
                "parent_favorite_food"
            )
            target = None
            if parent_fav:
                fav = [
                    s for s in smells
                    if parent_fav.lower() in s["name"].lower()
                ]
                if fav:
                    target = max(fav, key=lambda s: s["intensity"])
            if not target:
                target = max(smells, key=lambda s: s["intensity"])
            tobj = next(
                (o for o in k["objects"] if o["id"] == target["id"]), None
            )
            if tobj:
                dx, dy = tobj["x"] - px, tobj["y"] - py
                d = max(dist2d(px, py, tobj["x"], tobj["y"]), 0.1)
                speed = (
                    1.5 if stage == "larva"
                    else 5 * genome["flight_efficiency"]
                )
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
            lobj = next(
                (o for o in k["objects"] if o["id"] == light["id"]), None
            )
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

    if goal == "wall_walk":
        wall_targets = [
            (0, py), (k["width"], py), (px, 0), (px, k["height"])
        ]
        nearest = min(
            wall_targets, key=lambda w: dist2d(px, py, w[0], w[1])
        )
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
            gained = min(
                obj["energy"] * mult, energy["max"] - energy["current"]
            )
            energy["current"] = min(energy["max"], energy["current"] + gained)
            energy["hunger"] = max(0, energy["hunger"] - 25)
            energy["last_fed_tick"] = lc["total_ticks"]
            memory["times_fed"] += 1
            if obj["id"] not in [f["id"] for f in memory["food_sources"]]:
                memory["food_sources"].append({
                    "id": obj["id"], "x": obj["x"], "y": obj["y"]
                })
            if memory["favorite_food"] is None or obj["energy"] > 25:
                memory["favorite_food"] = obj["name"]
            state["brain"]["satisfaction"] = min(
                1.0, state["brain"]["satisfaction"] + 0.2
            )
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

    dist_moved = math.sqrt(
        body["velocity"]["x"] ** 2 + body["velocity"]["y"] ** 2
    )
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
        if stage == "larva":
            instar = 1 + lc["stage_tick"] // 8
            if instar != lc.get("larva_instar", 0):
                lc["larva_instar"] = instar
                lc["molts"] = lc.get("molts", 0) + 1
                body["size"] += 0.8
                record(state, "molt to instar " + str(instar))
            body["size"] += 0.06
            body["mass"] = body["size"] * 0.003
            brain["neural_complexity"] = min(
                0.3, brain["neural_complexity"] + 0.008
            )
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
    """Write a one-line narration for the current tick."""
    lc = state["lifecycle"]
    stage = lc["stage"]
    body = state["body"]
    energy = state["energy"]
    brain = state["brain"]
    k = state["kitchen"]
    gen = state["_meta"].get("generation", 1)

    gp = "Gen " + str(gen) + ". " if gen > 1 else ""

    # Check for active kitchen events — they override normal narration sometimes
    active_events = k.get("active_events", [])
    event_override = ""
    if active_events and random.random() < 0.4:
        ev = active_events[0]
        event_override = " The kitchen shudders — " + ev["name"] + "."

    if stage == "death":
        state["narration"] = gp + "Stillness. The kitchen light still hums overhead."
        return

    if stage == "egg":
        prog = lc["stage_tick"] / max(lc["stage_durations"]["egg"], 1)
        if prog < 0.3:
            state["narration"] = gp + "The egg sits motionless. Inside, cells divide furiously." + event_override
        elif prog < 0.7:
            state["narration"] = gp + "Organs form in miniature. The embryo twitches." + event_override
        else:
            state["narration"] = gp + "A crack appears. The egg trembles. Something stirs within."
        return

    if stage == "larva":
        if brain["current_goal"] == "seek_food":
            state["narration"] = gp + "The larva wriggles toward food. Size: " + str(round(body["size"], 1)) + "mm." + event_override
        elif energy["hunger"] < 20:
            state["narration"] = gp + "Well-fed larva grows. " + str(round(body["size"], 1)) + "mm and getting bigger."
        else:
            state["narration"] = gp + "Each molt brings new size, new hunger." + event_override
        return

    if stage == "pupa":
        prog = lc["stage_tick"] / max(lc["stage_durations"]["pupa"], 1)
        dreams = brain.get("dreams", [])
        dream_text = ""
        if dreams:
            latest = dreams[-1]
            dream_text = " Dream: " + latest["content"] + "."
        if prog < 0.4:
            state["narration"] = gp + "Inside the puparium, metamorphosis reshapes everything." + dream_text
        elif prog < 0.8:
            state["narration"] = gp + "Wings form where there were none. Eyes crystallize." + dream_text
        else:
            state["narration"] = gp + "Almost there. The adult fly takes shape within." + dream_text
        return

    stress = brain.get("stress", 0)
    stress_text = " [stressed]" if stress > 0.5 else ""
    goal = brain["current_goal"]
    is_air = body["is_airborne"]
    hunger = round(energy["hunger"])
    ecur = round(energy["current"])
    narrations = {
        "flee": gp + "DANGER! The fly bolts away at maximum speed!",
        "seek_food": gp + "Drawn by scent, the fly descends. Hunger: " + str(hunger) + "%.",
        "explore": gp + "Compound eyes scan 360 degrees. The kitchen is vast.",
        "fly_to_light": gp + "The fly spirals toward the light. An ancient compulsion.",
        "groom": gp + "The fly pauses, rubbing forelegs together. A moment of peace.",
        "wall_walk": gp + "Defying gravity, the fly walks along the wall.",
    }
    if goal == "idle":
        if not is_air:
            state["narration"] = gp + "The fly grooms a foreleg. Then launches." + stress_text
        else:
            state["narration"] = gp + "Wings beat 200 times per second. A blur of freedom." + stress_text
        return

    base = narrations.get(
        goal, gp + "Energy: " + str(ecur) + "%. The search continues."
    )
    state["narration"] = base + stress_text + event_override


def rebirth(state: dict) -> dict:
    """Trigger generational rebirth -- dead fly spawns generation N+1."""
    rng = random.Random(state["_meta"]["frame"] + 137)
    gen = state["_meta"].get("generation", 1)
    brain = state["brain"]

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
    if foods:
        best = max(foods, key=lambda f: f.get("energy", 0))
        egg_x = best["x"] + rng.uniform(-25, 25)
        egg_y = best["y"] + rng.uniform(-25, 25)
    else:
        egg_x = state["kitchen"]["width"] / 2
        egg_y = state["kitchen"]["height"] / 2

    kitchen = copy.deepcopy(state["kitchen"])
    kitchen["time_of_day"] = 0.3
    kitchen["lights_on"] = True

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

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_state = {
        "_meta": {
            "organism": "Musca domestica",
            "frame": state["_meta"]["frame"] + 1,
            "born_at": now,
            "version": "2.0.0",
            "generation": gen + 1,
            "parent_cause_of_death": state["_meta"].get("cause_of_death"),
            "parent_lifespan": state["lifecycle"]["total_ticks"],
            "total_frames_alive": 0,
            "lineage": lineage,
        },
        "genome": child_genome,
        "lifecycle": {
            "stage": "egg",
            "stage_tick": 0,
            "total_ticks": 0,
            "stage_durations": {
                "egg": 8,
                "larva": 25 + rng.randint(-3, 5),
                "pupa": 18 + rng.randint(-2, 3),
                "adult": 60 + rng.randint(-5, 10),
            },
            "molts": 0,
            "larva_instar": 0,
        },
        "body": {
            "position": {
                "x": round(egg_x, 2), "y": round(egg_y, 2), "z": 0.3
            },
            "velocity": {"x": 0, "y": 0, "z": 0},
            "facing": 0,
            "size": 1.0,
            "mass": 0.001,
            "wing_state": "none",
            "leg_state": "none",
            "is_airborne": False,
            "surface": "counter",
        },
        "energy": {
            "current": round(80 + rng.uniform(0, 15), 1),
            "max": 100,
            "hunger": 5.0,
            "metabolic_drain": 0.5,
            "last_fed_tick": 0,
        },
        "brain": {
            "state": "dormant",
            "current_goal": None,
            "fear_level": 0,
            "curiosity": 0,
            "satisfaction": 0.5,
            "decisions_made": 0,
            "neural_complexity": 0.01,
            "stress": 0.0,
            "dreams": [],
            "inherited_memory": {
                "parent_favorite_food": state["memory"].get("favorite_food"),
                "parent_danger_zones": state["memory"].get(
                    "danger_zones", []
                ),
                "parent_stress_avg": round(brain.get("stress", 0), 3),
                "epigenetic_bias": round(rng.uniform(0.1, 0.3), 3),
            },
        },
        "senses": {
            "smell": [],
            "sight": [],
            "touch": {"surface": "counter", "vibration": 0},
            "temperature": kitchen["ambient_temp"],
            "wind": 0,
        },
        "memory": {
            "food_sources": [],
            "danger_zones": [],
            "visited_positions": [],
            "total_distance": 0,
            "times_fed": 0,
            "times_fled": 0,
            "peak_altitude": 0,
            "favorite_food": None,
        },
        "kitchen": kitchen,
        "history": [
            {
                "tick": 0,
                "event": "generation " + str(gen + 1) + " -- egg laid",
                "stage": "egg",
                "energy": 85.0,
                "position": {
                    "x": round(egg_x, 2),
                    "y": round(egg_y, 2),
                    "z": 0.3,
                },
            }
        ],
        "ancestors": ancestors,
        "narration": (
            "Generation " + str(gen + 1)
            + ". A new egg glistens on the counter."
        ),
    }
    return new_state


def tick(state: dict) -> dict:
    """Advance the organism one tick forward. THE HEARTBEAT."""
    if state["lifecycle"]["stage"] == "death":
        return rebirth(state)

    state["_meta"]["frame"] += 1
    state["_meta"]["version"] = "3.0.0"
    state["lifecycle"]["stage_tick"] += 1
    state["lifecycle"]["total_ticks"] += 1
    state["_meta"]["total_frames_alive"] = state["lifecycle"]["total_ticks"]

    update_kitchen(state)
    update_kitchen_events(state)
    update_corpse_ecology(state)
    update_threats(state)
    update_senses(state)
    pupa_dream(state)
    update_stress(state)
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
    i = 0
    while i < len(args):
        if args[i] == "--ticks" and i + 1 < len(args):
            ticks = int(args[i + 1])
            i += 2
        elif (
            args[i] == "--until"
            and i + 1 < len(args)
            and args[i + 1] == "death"
        ):
            until_death = True
            i += 2
        else:
            i += 1

    if until_death:
        while state["lifecycle"]["stage"] != "death":
            state = tick(state)
            frame = state["_meta"]["frame"]
            stage = state["lifecycle"]["stage"]
            gen = state["_meta"].get("generation", 1)
            e = state["energy"]["current"]
            print(
                "  Gen {} Frame {:3d} | {:6s} | energy={:5.1f} | {}".format(
                    gen, frame, stage, e, state["narration"][:60]
                )
            )
    else:
        for _ in range(ticks):
            state = tick(state)
        frame = state["_meta"]["frame"]
        stage = state["lifecycle"]["stage"]
        gen = state["_meta"].get("generation", 1)
        e = state["energy"]["current"]
        goal = state["brain"]["current_goal"]
        print(
            "Gen {} Frame {} | {} | energy={:.1f} | goal={}".format(
                gen, frame, stage, e, goal
            )
        )
        print("  " + state["narration"])

    save_state(state)


if __name__ == "__main__":
    main()
