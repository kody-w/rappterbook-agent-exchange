#!/usr/bin/env python3
"""Musca domestica — housefly lifecycle tick engine.

Reads state/fly.json, advances one tick, writes back.
The output of frame N is the input of frame N+1.
"""
from __future__ import annotations
import json, math, os, random, sys, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(ROOT / "docs")))

def load_state() -> dict:
    p = STATE_DIR / "fly.json"
    if not p.exists():
        print("No state/fly.json found — run genesis first", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        return json.load(f)

def save_state(state: dict) -> None:
    with open(STATE_DIR / "fly.json", "w") as f:
        json.dump(state, f, indent=2)
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(DOCS_DIR / "fly_state.json", "w") as f:
        json.dump(state, f, indent=2)

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def dist(a: dict, b: dict) -> float:
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2)

def add_event(state: dict, event: str, desc: str) -> None:
    state["history"].append({
        "frame": state["_meta"]["frame"],
        "tick": state["lifecycle"]["total_ticks"],
        "stage": state["lifecycle"]["stage"],
        "event": event,
        "description": desc
    })
    if len(state["history"]) > 200:
        state["history"] = state["history"][-200:]

# ── Lifecycle transitions ──────────────────────────────────

def check_transition(state: dict) -> None:
    lc = state["lifecycle"]
    stage = lc["stage"]
    dur = lc["stage_durations"].get(stage, 999)

    if stage == "death":
        return

    if stage == "egg" and lc["stage_tick"] >= dur:
        transition_to(state, "larva")
        add_event(state, "hatch", "The egg splits open. A tiny maggot writhes free.")
    elif stage == "larva":
        instar_dur = dur // 3
        if lc["stage_tick"] >= dur:
            transition_to(state, "pupa")
            add_event(state, "pupate", "The larva stops moving. Its skin hardens into a brown puparium.")
        elif lc["stage_tick"] % instar_dur == 0 and lc["stage_tick"] > 0 and lc["larva_instar"] < 3:
            lc["larva_instar"] += 1
            lc["molts"] += 1
            state["body"]["size"] += 0.3
            add_event(state, "molt", f"Instar {lc['larva_instar']}: the larva sheds its skin and grows.")
    elif stage == "pupa" and lc["stage_tick"] >= dur:
        transition_to(state, "adult")
        state["body"]["wing_state"] = "folded"
        state["body"]["leg_state"] = "six"
        state["body"]["size"] = 2.0
        state["brain"]["state"] = "alert"
        state["brain"]["neural_complexity"] = 1.0
        add_event(state, "emerge", "Wings unfurl. Compound eyes glitter. The adult fly is born.")
    elif stage == "adult":
        max_life = dur * state["genome"]["lifespan_modifier"]
        energy_dead = state["energy"]["current"] <= 0
        age_dead = lc["stage_tick"] >= max_life
        if energy_dead or age_dead:
            cause = "starvation" if energy_dead else "old age"
            transition_to(state, "death")
            state["_meta"]["cause_of_death"] = cause
            state["brain"]["state"] = "dead"
            add_event(state, "death", f"The fly is still. Cause: {cause}. {lc['total_ticks']} ticks lived.")

def transition_to(state: dict, new_stage: str) -> None:
    state["lifecycle"]["stage"] = new_stage
    state["lifecycle"]["stage_tick"] = 0

# ── Senses ─────────────────────────────────────────────────

def update_senses(state: dict) -> None:
    pos = state["body"]["position"]
    kit = state["kitchen"]
    senses = state["senses"]
    smell_sens = state["genome"]["smell_sensitivity"]

    smells = []
    sights = []
    for obj in kit["objects"]:
        d = dist(pos, obj)
        if obj["type"] == "food":
            sr = obj.get("smell_radius", 60)
            if d < sr * smell_sens:
                intensity = 1.0 - (d / (sr * smell_sens))
                smells.append({"id": obj["id"], "name": obj["name"], "intensity": round(intensity, 2), "distance": round(d, 1), "direction": math.atan2(obj["y"]-pos["y"], obj["x"]-pos["x"])})
        if obj["type"] == "light":
            sights.append({"id": obj["id"], "intensity": obj.get("intensity", 1) * max(0, 1 - d/400), "direction": math.atan2(obj["y"]-pos["y"], obj["x"]-pos["x"])})
        if obj["type"] == "threat" and obj.get("active"):
            d_threat = dist(pos, obj)
            if d_threat < 100:
                senses["touch"]["vibration"] = max(senses["touch"]["vibration"], 1.0 - d_threat/100)

    senses["smell"] = sorted(smells, key=lambda s: -s["intensity"])[:5]
    senses["sight"] = sorted(sights, key=lambda s: -s["intensity"])[:3]
    senses["temperature"] = kit["ambient_temp"] + random.uniform(-0.5, 0.5)

# ── Brain / behavior ───────────────────────────────────────

def think(state: dict) -> None:
    stage = state["lifecycle"]["stage"]
    brain = state["brain"]

    if stage in ("egg", "pupa", "death"):
        brain["current_goal"] = None
        return

    senses = state["senses"]
    energy = state["energy"]
    brain["decisions_made"] += 1

    # Fear from vibrations
    vib = senses["touch"]["vibration"]
    brain["fear_level"] = clamp(brain["fear_level"] * 0.8 + vib * 0.5, 0, 1)

    # Decide goal
    if brain["fear_level"] > 0.6:
        brain["current_goal"] = "flee"
        brain["state"] = "panicked"
    elif energy["hunger"] > 60:
        brain["current_goal"] = "find_food"
        brain["state"] = "hungry"
    elif stage == "adult" and random.random() < 0.15:
        brain["current_goal"] = random.choice(["explore", "fly_to_light", "rest", "groom"])
        brain["state"] = "curious"
    elif stage == "larva":
        if senses["smell"]:
            brain["current_goal"] = "crawl_to_food"
            brain["state"] = "feeding"
        else:
            brain["current_goal"] = "wander"
            brain["state"] = "searching"
    else:
        if brain["current_goal"] is None:
            brain["current_goal"] = "idle"
            brain["state"] = "resting"

# ── Movement ───────────────────────────────────────────────

def move(state: dict) -> None:
    stage = state["lifecycle"]["stage"]
    if stage in ("egg", "pupa", "death"):
        return

    body = state["body"]
    pos = body["position"]
    vel = body["velocity"]
    goal = state["brain"]["current_goal"]
    kit = state["kitchen"]

    speed = 1.0 if stage == "larva" else 4.0
    flight_eff = state["genome"]["flight_efficiency"]

    if goal == "flee":
        angle = random.uniform(0, 2*math.pi)
        vel["x"] = math.cos(angle) * speed * 3
        vel["y"] = math.sin(angle) * speed * 3
        if stage == "adult":
            vel["z"] = random.uniform(0.5, 2.0)
            body["is_airborne"] = True
            body["wing_state"] = "buzzing"
        state["memory"]["times_fled"] += 1

    elif goal in ("find_food", "crawl_to_food"):
        smells = state["senses"]["smell"]
        if smells:
            best = smells[0]
            angle = best["direction"]
            vel["x"] = math.cos(angle) * speed
            vel["y"] = math.sin(angle) * speed
        else:
            angle = random.uniform(0, 2*math.pi)
            vel["x"] = math.cos(angle) * speed * 0.5
            vel["y"] = math.sin(angle) * speed * 0.5

    elif goal == "fly_to_light":
        lights = state["senses"]["sight"]
        if lights and stage == "adult":
            best = lights[0]
            vel["x"] = math.cos(best["direction"]) * speed * flight_eff
            vel["y"] = math.sin(best["direction"]) * speed * flight_eff
            vel["z"] = 0.3
            body["is_airborne"] = True
            body["wing_state"] = "buzzing"

    elif goal == "explore":
        angle = body["facing"] + random.uniform(-0.5, 0.5)
        body["facing"] = angle
        vel["x"] = math.cos(angle) * speed * 0.7
        vel["y"] = math.sin(angle) * speed * 0.7
        if stage == "adult" and random.random() < 0.3:
            vel["z"] = random.uniform(-0.5, 1.0)
            body["is_airborne"] = True
            body["wing_state"] = "buzzing"

    elif goal == "rest" or goal == "groom":
        vel["x"] *= 0.3
        vel["y"] *= 0.3
        vel["z"] = -0.2 if body["is_airborne"] else 0
        if not body["is_airborne"]:
            body["wing_state"] = "folded"

    elif goal == "wander":
        body["facing"] += random.uniform(-0.8, 0.8)
        vel["x"] = math.cos(body["facing"]) * speed * 0.4
        vel["y"] = math.sin(body["facing"]) * speed * 0.4

    # Apply gravity
    if body["is_airborne"]:
        vel["z"] -= 0.15

    # Update position
    old_x, old_y = pos["x"], pos["y"]
    pos["x"] += vel["x"]
    pos["y"] += vel["y"]
    pos["z"] = max(0, pos["z"] + vel["z"])

    if pos["z"] <= 0:
        body["is_airborne"] = False
        pos["z"] = 0
        vel["z"] = 0
        if stage == "adult":
            body["wing_state"] = "folded"
        body["surface"] = "counter"

    # Boundary collision
    pos["x"] = clamp(pos["x"], 0, kit["width"])
    pos["y"] = clamp(pos["y"], 0, kit["height"])
    pos["z"] = clamp(pos["z"], 0, 5)

    # Track peak altitude
    state["memory"]["peak_altitude"] = max(state["memory"]["peak_altitude"], pos["z"])

    # Friction
    vel["x"] *= 0.85
    vel["y"] *= 0.85

    # Distance tracking
    d = math.sqrt((pos["x"]-old_x)**2 + (pos["y"]-old_y)**2)
    state["memory"]["total_distance"] += d

# ── Feeding ────────────────────────────────────────────────

def try_feed(state: dict) -> None:
    stage = state["lifecycle"]["stage"]
    if stage in ("egg", "pupa", "death"):
        return

    pos = state["body"]["position"]
    energy = state["energy"]

    for obj in state["kitchen"]["objects"]:
        if obj["type"] != "food":
            continue
        d = dist(pos, obj)
        if d < 20:
            gained = obj["energy"] * obj["decay"] * 0.3
            energy["current"] = min(energy["max"], energy["current"] + gained)
            energy["hunger"] = max(0, energy["hunger"] - 25)
            energy["last_fed_tick"] = state["lifecycle"]["total_ticks"]
            state["memory"]["times_fed"] += 1
            if not state["memory"]["favorite_food"] or gained > 5:
                state["memory"]["favorite_food"] = obj["name"]
            if state["memory"]["times_fed"] == 1:
                add_event(state, "first_meal", f"First meal: {obj['name']}. Energy surges.")
            # Remember food source
            known = [f["id"] for f in state["memory"]["food_sources"]]
            if obj["id"] not in known:
                state["memory"]["food_sources"].append({"id": obj["id"], "x": obj["x"], "y": obj["y"], "name": obj["name"]})
            break

# ── Threats ────────────────────────────────────────────────

def update_threats(state: dict) -> None:
    kit = state["kitchen"]
    tick = state["lifecycle"]["total_ticks"]

    for obj in kit["objects"]:
        if obj["type"] != "threat":
            continue
        # Random threat activation
        if not obj["active"] and random.random() < 0.02 and state["lifecycle"]["stage"] == "adult":
            obj["active"] = True
            obj["x"] = random.uniform(0, kit["width"])
            obj["y"] = random.uniform(0, kit["height"])
            add_event(state, "threat_appears", f"The {obj['name']} appears!")
        elif obj["active"]:
            # Chase fly
            pos = state["body"]["position"]
            dx = pos["x"] - obj["x"]
            dy = pos["y"] - obj["y"]
            d = max(1, math.sqrt(dx*dx + dy*dy))
            obj["x"] += (dx/d) * obj["speed"] * 0.3
            obj["y"] += (dy/d) * obj["speed"] * 0.3
            # Deactivate after a while or if far
            if random.random() < 0.1:
                obj["active"] = False
                obj["x"] = -100
                obj["y"] = -100

# ── Energy / metabolism ────────────────────────────────────

def update_energy(state: dict) -> None:
    stage = state["lifecycle"]["stage"]
    if stage == "death":
        return

    energy = state["energy"]
    meta = state["genome"]["metabolic_rate"]

    # Base drain by stage
    drain_map = {"egg": 0.1, "larva": 0.3, "pupa": 0.15, "adult": 0.5}
    drain = drain_map.get(stage, 0.2) * meta

    # Flying costs more
    if state["body"]["is_airborne"]:
        drain *= 2.0

    energy["current"] -= drain
    energy["current"] = max(0, round(energy["current"], 2))

    # Hunger increases
    ticks_since_food = state["lifecycle"]["total_ticks"] - max(0, energy["last_fed_tick"])
    energy["hunger"] = clamp(energy["hunger"] + 0.5 + (0.1 * ticks_since_food / 10), 0, 100)

    if energy["current"] < 15 and stage == "adult":
        state["brain"]["state"] = "desperate"
        if energy["current"] < 5:
            add_event(state, "near_death", "Energy critically low. The fly stumbles.")

# ── Kitchen environment ────────────────────────────────────

def update_kitchen(state: dict) -> None:
    kit = state["kitchen"]
    tick = state["lifecycle"]["total_ticks"]

    # Day/night cycle (100 ticks = 1 day)
    kit["time_of_day"] = (tick % 100) / 100.0
    tod = kit["time_of_day"]
    kit["lights_on"] = 0.25 < tod < 0.75
    kit["ambient_temp"] = 20 + 4 * math.sin(tod * math.pi)

    # Food decay
    for obj in kit["objects"]:
        if obj["type"] == "food":
            obj["decay"] = min(1.0, obj["decay"] + 0.002)
            obj["smell_radius"] = min(200, obj.get("smell_radius", 60) + obj["decay"] * 0.3)

# ── Narration ──────────────────────────────────────────────

def generate_narration(state: dict) -> None:
    stage = state["lifecycle"]["stage"]
    brain = state["brain"]
    energy = state["energy"]
    tick = state["lifecycle"]["stage_tick"]

    narrations = {
        "egg": [
            "The egg sits motionless. Inside, cells divide furiously.",
            "A translucent shell. Warmth from the counter seeps in.",
            "Patience. The embryo is not yet ready.",
            "Micro-tremors from within. Something stirs.",
        ],
        "larva": [
            "The maggot inches forward, blind but hungry.",
            "Chemoreceptors fire. Food is near.",
            "Growing. Always growing. The world is taste and touch.",
            "Each molt brings new size, new hunger.",
        ],
        "pupa": [
            "Inside the puparium, metamorphosis reshapes everything.",
            "Wings form where there were none. Eyes crystallize.",
            "The pupa is still. But inside, revolution.",
            "Imaginal discs unfold. An adult assembles itself.",
        ],
        "adult": [
            "Compound eyes scan 360 degrees. The kitchen is vast.",
            "Wings beat 200 times per second. A blur of freedom.",
            f"Energy: {energy['current']:.0f}%. The search continues.",
            "Halteres stabilize. A perfect flying machine.",
            "The fly grooms a foreleg. Then launches.",
        ],
        "death": [
            "Stillness. The kitchen light still hums overhead.",
            "A small body on the counter. Life measured in ticks.",
        ]
    }

    options = narrations.get(stage, ["..."])
    # Mix in brain-state specific narrations
    if brain["state"] == "panicked":
        options = ["DANGER. Every nerve fires. Evasive maneuvers.", "The buzz intensifies — pure survival instinct."]
    elif brain["state"] == "hungry" and stage == "adult":
        options = ["Hunger gnaws. The proboscis extends, tasting the air.", "Must find food. Energy reserves dwindling."]
    elif brain["state"] == "desperate":
        options = ["Spiraling. Crashing. The end approaches.", "Too weak to fly. Crawling toward any scent."]

    state["narration"] = random.choice(options)

# ── Main tick ──────────────────────────────────────────────

def tick(state: dict) -> dict:
    state["_meta"]["frame"] += 1
    state["lifecycle"]["stage_tick"] += 1
    state["lifecycle"]["total_ticks"] += 1
    state["senses"]["touch"]["vibration"] *= 0.7

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
        print(f"The fly is dead (frame {state['_meta']['frame']}). No more ticks.")
        save_state(state)
        return
    state = tick(state)
    frame = state["_meta"]["frame"]
    stage = state["lifecycle"]["stage"]
    energy = state["energy"]["current"]
    goal = state["brain"]["current_goal"]
    print(f"Frame {frame} | {stage} | energy={energy:.1f} | goal={goal}")
    print(f"  {state['narration']}")
    save_state(state)

if __name__ == "__main__":
    main()
