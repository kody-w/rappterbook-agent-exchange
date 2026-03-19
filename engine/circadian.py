"""Circadian organ — the fly's internal clock.

Controls:
  - Sleep/wake cycle tied to kitchen light + time_of_day
  - Hormonal rhythm: melatonin (sleep) vs cortisol (alertness)
  - Wing beat frequency varies with circadian phase
  - Body temperature follows a daily curve
  - Aging: adults slow down as they approach death
  - Weather awareness: reads kitchen weather, affects behavior
  - Wing condition: wings degrade with flight, damaged by threats
  - Bacterial load: feeding on decay introduces bacteria
  - Danger memory: fly remembers where threats appeared
"""
from __future__ import annotations
import math
import random


def update_circadian(state: dict) -> None:
    """Full circadian organ — sleep, hormones, aging, wings, bacteria, weather, danger."""
    stage = state["lifecycle"]["stage"]
    body = state["body"]
    brain = state["brain"]
    kitchen = state["kitchen"]
    tod = kitchen.get("time_of_day", 0.5)

    # --- Initialize v4 fields if missing ---
    if "wing_condition" not in body:
        body["wing_condition"] = 1.0 if stage in ("adult",) else 0.0
    if "bacterial_load" not in body:
        body["bacterial_load"] = 0.0
    if "weather" not in kitchen:
        kitchen["weather"] = "clear"
    if "circadian" not in brain:
        brain["circadian"] = {
            "melatonin": 0.0,
            "cortisol": 0.5,
            "body_temp_offset": 0.0,
            "wing_beat_hz": 200,
            "phase": "awake",
            "sleep_debt": 0.0,
        }
    if "danger_zones" not in state["memory"]:
        state["memory"]["danger_zones"] = []

    circ = brain["circadian"]

    # --- Weather ---
    if random.random() < 0.006:
        kitchen["weather"] = random.choice(["clear", "clear", "overcast", "rain", "hot", "windy"])

    weather = kitchen.get("weather", "clear")
    if weather == "rain":
        kitchen["ambient_temp"] = kitchen.get("ambient_temp", 22) - 0.5
    elif weather == "hot":
        kitchen["ambient_temp"] = kitchen.get("ambient_temp", 22) + 0.5

    # --- Hormones follow light cycle ---
    is_dark = tod < 0.2 or tod > 0.88
    if is_dark:
        circ["melatonin"] = min(1.0, circ["melatonin"] + 0.06)
        circ["cortisol"] = max(0.0, circ["cortisol"] - 0.04)
    else:
        circ["melatonin"] = max(0.0, circ["melatonin"] - 0.04)
        circ["cortisol"] = min(1.0, circ["cortisol"] + 0.03)

    # --- Sleep/wake ---
    if stage in ("egg", "pupa", "death"):
        circ["phase"] = "dormant"
    elif stage == "adult":
        if circ["melatonin"] > 0.7 and state["energy"]["hunger"] < 35:
            circ["phase"] = "drowsy"
            circ["sleep_debt"] = max(0, circ["sleep_debt"] - 0.1)
        elif circ["melatonin"] > 0.85 and brain.get("fear_level", 0) < 0.3:
            circ["phase"] = "sleeping"
            circ["sleep_debt"] = max(0, circ["sleep_debt"] - 0.2)
        else:
            circ["phase"] = "awake"
            circ["sleep_debt"] = min(1.0, circ["sleep_debt"] + 0.01)
    else:
        circ["phase"] = "awake"

    # Sleep modifies brain
    if circ["phase"] == "sleeping":
        brain["current_goal"] = "sleep"
        brain["state"] = "resting"
        # Sleeping fly doesn't move
        body["velocity"] = {"x": 0, "y": 0, "z": 0}
        if body["is_airborne"]:
            body["is_airborne"] = False
            body["position"]["z"] = 0
            body["surface"] = "counter"
    elif circ["phase"] == "drowsy" and stage == "adult":
        # Drowsy fly moves toward dark corner
        if brain.get("current_goal") == "idle":
            brain["current_goal"] = "seek_rest"

    # --- Body temperature ---
    base_temp = kitchen.get("ambient_temp", 22)
    circ["body_temp_offset"] = round(math.sin(tod * math.pi * 2) * 1.5, 2)

    # --- Wing beat frequency varies ---
    if stage == "adult" and body.get("is_airborne"):
        base_hz = 200
        fatigue = 1.0 - body.get("wing_condition", 1.0)
        stress_mod = state.get("stress", {}).get("level", 0) * 30
        circ["wing_beat_hz"] = round(base_hz - fatigue * 40 + stress_mod + circ["cortisol"] * 20)
    else:
        circ["wing_beat_hz"] = 0

    # --- Aging effects for adults ---
    if stage == "adult":
        adult_age = state["lifecycle"]["stage_tick"]
        adult_max = state["lifecycle"]["stage_durations"].get("adult", 65)
        age_ratio = adult_age / max(adult_max, 1)

        # Senses dull with age
        genome = state["genome"]
        if age_ratio > 0.5:
            dull_factor = 1.0 - (age_ratio - 0.5) * 0.6
            # Reduce smell detection range in senses
            for smell in state["senses"].get("smell", []):
                smell["intensity"] = round(smell["intensity"] * dull_factor, 2)

    # --- Wing condition ---
    if stage == "adult":
        wing = body.get("wing_condition", 1.0)

        # Flight degrades wings slowly
        if body.get("is_airborne"):
            wing -= 0.0015
        
        # Grooming repairs wings
        if brain.get("current_goal") == "groom":
            wing = min(1.0, wing + 0.025)

        # Near-miss threat damage already handled in update_senses
        # but add wind damage
        wind = state["senses"].get("wind", 0)
        if wind > 0.6 and body.get("is_airborne"):
            wing -= wind * 0.005

        body["wing_condition"] = round(max(0.1, min(1.0, wing)), 4)

    # --- Bacterial load ---
    bact = body.get("bacterial_load", 0)
    if bact > 0:
        # Immune system fights bacteria
        body["bacterial_load"] = round(max(0, bact - 0.004), 4)
        # High bacterial load drains extra energy
        if bact > 0.5:
            state["energy"]["current"] = max(0, state["energy"]["current"] - bact * 0.2)

    # --- Danger memory ---
    # Record threat positions (done in think/senses already)
    # Prune old danger memories (older than 40 ticks)
    dz = state["memory"].get("danger_zones", [])
    tick_now = state["lifecycle"]["total_ticks"]
    state["memory"]["danger_zones"] = [
        d for d in dz if tick_now - d.get("tick", 0) < 40
    ]

    # --- Bacterial pickup from feeding on decayed food ---
    # (Handled per-feed in try_feed, but also ambient pickup on trash)
    if stage in ("larva", "adult"):
        px, py = body["position"]["x"], body["position"]["y"]
        for obj in kitchen.get("objects", []):
            if obj.get("type") == "food" and obj.get("decay", 0) > 0.7:
                dx = px - obj["x"]
                dy = py - obj["y"]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 20:
                    body["bacterial_load"] = round(
                        min(1.0, body.get("bacterial_load", 0) + 0.008), 4
                    )
                    break

    # --- Lethal bacterial load ---
    if body.get("bacterial_load", 0) > 0.95 and stage == "adult":
        state["lifecycle"]["stage"] = "death"
        state["_meta"]["cause_of_death"] = "infection"
        state["_meta"]["died_at"] = state["lifecycle"]["total_ticks"]
        brain["state"] = "dead"
        state["history"].append({
            "tick": state["lifecycle"]["total_ticks"],
            "event": "succumbed to bacterial infection",
            "stage": "death",
            "energy": round(state["energy"]["current"], 1),
            "position": dict(body["position"]),
        })
