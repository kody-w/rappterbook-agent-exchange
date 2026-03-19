#!/usr/bin/env python3
"""
fly.py -- Musca domestica lifecycle engine.
Reads state/fly.json, advances one tick, writes back.
"""
from __future__ import annotations
import json, math, random, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state" / "fly.json"
DOCS = ROOT / "docs" / "fly_state.json"

def load(): 
    with open(STATE) as f: return json.load(f)

def save(s):
    for p in (STATE, DOCS):
        p.parent.mkdir(parents=True, exist_ok=True)
        t = p.with_suffix(".tmp")
        with open(t, "w") as f: json.dump(s, f, indent=2)
        t.rename(p)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def dist3(a, b):
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2)

def tick_egg(s):
    e, b = s["energy"], s["body"]
    e["current"] -= e["metabolism_drain_per_tick"] * 0.3
    e["food_in_gut"] -= 0.5
    b["size"] += 0.02
    b["appendages"]["eyes"]["facets_developed"] += 50
    temp = s["senses"]["touch"]["surface_temperature"]
    if temp < 15: e["current"] -= 1.0; return "development_slowed_cold"
    if temp > 35: e["current"] -= 2.0; return "heat_damage"
    p = s["lifecycle"]["stage_tick"] / s["lifecycle"]["stage_durations"]["egg"]
    if p > 0.5:
        b["integrity"] = max(0.5, b["integrity"] - 0.05)
        return "egg_cracking"
    return "cells_dividing"

def tick_larva(s):
    b, e, sn, br, m, k, g = s["body"], s["energy"], s["senses"], s["brain"], s["memory"], s["kitchen"], s["genome"]
    sn["smell"]["active"] = True; sn["smell"]["range"] = 30
    e["current"] -= e["metabolism_drain_per_tick"] * 1.5; b["size"] += 0.08
    if e["food_in_gut"] > 0:
        ab = min(e["digestion_rate"], e["food_in_gut"]); e["food_in_gut"] -= ab
        e["current"] = min(e["max"], e["current"] + ab * 0.8)
    det = []
    for o in k["objects"]:
        if "odor_strength" in o:
            d = dist3(b["position"], o["position"]); ef = sn["smell"]["range"] * o["odor_strength"]
            if d < ef: det.append({"source": o["id"], "strength": o["odor_strength"]*(1-d/max(ef,1)), "direction": {"x": o["position"]["x"]-b["position"]["x"], "y": o["position"]["y"]-b["position"]["y"], "z": o["position"]["z"]-b["position"]["z"]}})
    sn["smell"]["detected_odors"] = det
    sn["smell"]["strongest_odor"] = max(det, key=lambda x: x["strength"]) if det else None
    br["hunger_level"] = clamp(1-e["current"]/e["max"], 0, 1); br["state"] = "active"
    if br["hunger_level"] > 0.3 and sn["smell"]["strongest_odor"]:
        br["current_goal"] = "seek_food"; d = sn["smell"]["strongest_odor"]["direction"]
        mg = math.sqrt(d["x"]**2+d["y"]**2+d["z"]**2) or 1; sp = g["speed"]*0.3
        b["velocity"]["x"] = d["x"]/mg*sp; b["velocity"]["y"] = d["y"]/mg*sp; b["velocity"]["z"] = 0
    else:
        br["current_goal"] = "explore"
        b["velocity"]["x"] = random.uniform(-0.5, 0.5); b["velocity"]["y"] = random.uniform(-0.5, 0.5)
    b["position"]["x"] += b["velocity"]["x"]; b["position"]["y"] += b["velocity"]["y"]
    dm = k["dimensions"]; b["position"]["x"] = clamp(b["position"]["x"], 0, dm["width"]); b["position"]["y"] = clamp(b["position"]["y"], 0, dm["depth"])
    m["total_distance_traveled"] += math.sqrt(b["velocity"]["x"]**2 + b["velocity"]["y"]**2)
    for o in k["objects"]:
        if o.get("attractiveness", 0) > 0.5 and dist3(b["position"], o["position"]) < 5:
            eaten = min(15, e["gut_capacity"]-e["food_in_gut"]); e["food_in_gut"] += eaten
            e["last_fed_tick"] = s["lifecycle"]["total_ticks"]; m["times_fed"] += 1
            return "feeding_on_" + o["id"]
    p = s["lifecycle"]["stage_tick"] / s["lifecycle"]["stage_durations"]["larva"]
    if p > 0.3: b["appendages"]["antennae"]["state"] = "developing"
    if p > 0.6: b["appendages"]["proboscis"]["state"] = "forming"
    if p > 0.8: b["appendages"]["eyes"]["facets_developed"] = min(g["compound_eye_facets"], b["appendages"]["eyes"]["facets_developed"]+200)
    br["decisions_made"] += 1; br["neural_connections"] += 1
    return "crawling"

def tick_pupa(s):
    b, e, lc = s["body"], s["energy"], s["lifecycle"]
    e["current"] -= e["metabolism_drain_per_tick"] * 0.2; e["food_in_gut"] -= 0.3
    b["velocity"] = {"x": 0, "y": 0, "z": 0}
    p = lc["stage_tick"] / lc["stage_durations"]["pupa"]
    if p < 0.3:
        b["integrity"] -= 0.02; b["appendages"]["legs"]["state"] = "dissolving"; return "histolysis"
    elif p < 0.6:
        b["integrity"] += 0.03; b["appendages"]["wings"]["state"] = "forming_adult"
        b["appendages"]["legs"]["state"] = "reforming"; b["appendages"]["eyes"]["state"] = "compound_forming"
        b["appendages"]["eyes"]["facets_developed"] = int(s["genome"]["compound_eye_facets"]*p)
        return "histogenesis"
    else:
        b["integrity"] = min(1.0, b["integrity"]+0.05)
        for k in ("wings","legs","antennae","proboscis"): b["appendages"][k]["state"] = "ready"
        b["appendages"]["antennae"]["sensitivity"] = s["genome"]["antenna_sensitivity"]
        b["appendages"]["eyes"]["state"] = "compound_ready"
        b["appendages"]["eyes"]["facets_developed"] = s["genome"]["compound_eye_facets"]
        b["size"] = s["genome"]["size_modifier"]
        return "adult_forming"

def tick_adult(s):
    b, e, sn, br, m, k, g = s["body"], s["energy"], s["senses"], s["brain"], s["memory"], s["kitchen"], s["genome"]
    sn["smell"]["active"] = True; sn["smell"]["range"] = 80*g["antenna_sensitivity"]
    sn["sight"]["active"] = True; sn["sight"]["range"] = 150*(g["compound_eye_facets"]/4000)
    fd = 0.3 if sn["proprioception"]["is_flying"] else 0
    e["current"] -= e["metabolism_drain_per_tick"] + fd
    if e["food_in_gut"] > 0:
        ab = min(e["digestion_rate"], e["food_in_gut"]); e["food_in_gut"] -= ab
        e["current"] = min(e["max"], e["current"]+ab*0.6)
    br["hunger_level"] = clamp(1-e["current"]/e["max"], 0, 1); br["state"] = "active"
    br["curiosity"] = random.uniform(0.2, 0.8)
    br["comfort"] = clamp(1-br["fear_level"]-br["hunger_level"]*0.5, 0, 1)
    br["mating_drive"] = clamp(s["lifecycle"]["stage_tick"]/s["lifecycle"]["stage_durations"]["adult"]*0.8, 0, 1)
    det = []
    for o in k["objects"]:
        if "odor_strength" in o:
            d = dist3(b["position"], o["position"]); ef = sn["smell"]["range"]*o["odor_strength"]
            if d < ef: det.append({"source": o["id"], "strength": o["odor_strength"]*max(0,1-d/ef), "direction": {"x": o["position"]["x"]-b["position"]["x"], "y": o["position"]["y"]-b["position"]["y"], "z": o["position"]["z"]-b["position"]["z"]}})
    sn["smell"]["detected_odors"] = det
    sn["smell"]["strongest_odor"] = max(det, key=lambda x: x["strength"]) if det else None
    br["fear_level"] = 0; nt = None
    for t in k["threats"]:
        if not t.get("present") or "position" not in t: continue
        d = dist3(b["position"], t["position"]); f = t["danger_level"]*max(0,1-d/100)
        if f > br["fear_level"]: br["fear_level"] = f; nt = t
    ev = "flying"
    if br["fear_level"] > 0.5 and nt:
        br["current_goal"] = "flee"; fx = {"x": b["position"]["x"]-nt["position"]["x"], "y": b["position"]["y"]-nt["position"]["y"], "z": 20}
        mg = math.sqrt(fx["x"]**2+fx["y"]**2+fx["z"]**2) or 1; sp = g["speed"]*g["flight_agility"]*3
        b["velocity"] = {"x": fx["x"]/mg*sp, "y": fx["y"]/mg*sp, "z": fx["z"]/mg*sp}; m["times_fled"] += 1
        ev = "fleeing_" + nt["id"]
    elif br["hunger_level"] > 0.4 and sn["smell"]["strongest_odor"]:
        br["current_goal"] = "seek_food"; d = sn["smell"]["strongest_odor"]["direction"]
        mg = math.sqrt(d["x"]**2+d["y"]**2+d["z"]**2) or 1; sp = g["speed"]*1.5
        b["velocity"] = {"x": d["x"]/mg*sp, "y": d["y"]/mg*sp, "z": d["z"]/mg*sp}
        ev = "seeking_" + sn["smell"]["strongest_odor"]["source"]
    elif br["curiosity"] > 0.6:
        br["current_goal"] = "explore"
        lights = [o for o in k["objects"] if o.get("brightness", 0) > 0.5]
        if lights and random.random() < 0.4:
            tg = random.choice(lights); d = {"x": tg["position"]["x"]-b["position"]["x"], "y": tg["position"]["y"]-b["position"]["y"], "z": tg["position"]["z"]-b["position"]["z"]}
            mg = math.sqrt(d["x"]**2+d["y"]**2+d["z"]**2) or 1
            b["velocity"] = {"x": d["x"]/mg*g["speed"], "y": d["y"]/mg*g["speed"], "z": d["z"]/mg*g["speed"]}
            ev = "attracted_to_light"
        else: b["velocity"] = {"x": random.uniform(-2,2)*g["speed"], "y": random.uniform(-2,2)*g["speed"], "z": random.uniform(-1,1)*g["speed"]}; ev = "exploring"
    else:
        br["current_goal"] = "rest"
        if b["position"]["z"] > 5: b["velocity"]["z"] = -0.5
        else: b["velocity"] = {"x": 0, "y": 0, "z": 0}
        ev = "resting"
    b["position"]["x"] += b["velocity"]["x"]; b["position"]["y"] += b["velocity"]["y"]; b["position"]["z"] += b["velocity"]["z"]
    if b["position"]["z"] > 0: b["velocity"]["z"] -= 0.1
    dm = k["dimensions"]; b["position"]["x"] = clamp(b["position"]["x"], 0, dm["width"]); b["position"]["y"] = clamp(b["position"]["y"], 0, dm["depth"]); b["position"]["z"] = clamp(b["position"]["z"], 0, dm["height"])
    sn["proprioception"]["is_flying"] = b["position"]["z"] > 2; sn["proprioception"]["is_grounded"] = b["position"]["z"] <= 2
    for o in k["objects"]:
        if o.get("attractiveness", 0) > 0.3 and dist3(b["position"], o["position"]) < 8:
            eaten = min(10, e["gut_capacity"]-e["food_in_gut"]); e["food_in_gut"] += eaten
            e["last_fed_tick"] = s["lifecycle"]["total_ticks"]; m["times_fed"] += 1; ev = "feeding_on_"+o["id"]; break
    for t in k["threats"]:
        if t["type"] == "trap" and "position" in t and dist3(b["position"], t["position"]) < t.get("size", 10):
            e["current"] = 0; s["_meta"]["cause_of_death"] = "caught_in_"+t["id"]; ev = "caught_in_"+t["id"]; break
    m["total_distance_traveled"] += math.sqrt(sum(v**2 for v in b["velocity"].values()))
    if s["lifecycle"]["total_ticks"] % 5 == 0:
        m["visited_positions"].append({"x": round(b["position"]["x"],1), "y": round(b["position"]["y"],1), "z": round(b["position"]["z"],1), "tick": s["lifecycle"]["total_ticks"]})
        if len(m["visited_positions"]) > 50: m["visited_positions"] = m["visited_positions"][-50:]
    br["decisions_made"] += 1; br["neural_connections"] = min(5000, br["neural_connections"]+3)
    return ev

def advance_kitchen(s):
    k = s["kitchen"]; t = s["lifecycle"]["total_ticks"]
    k["hour"] = (8 + t*0.25) % 24; h = k["hour"]
    if 6 <= h < 12: k["time_of_day"] = "morning"; k["ambient_light"] = 0.6+(h-6)*0.05
    elif 12 <= h < 18: k["time_of_day"] = "afternoon"; k["ambient_light"] = 0.9
    elif 18 <= h < 21: k["time_of_day"] = "evening"; k["ambient_light"] = 0.7-(h-18)*0.1
    else: k["time_of_day"] = "night"; k["ambient_light"] = 0.15
    hu = next(x for x in k["threats"] if x["id"] == "human")
    hu["present"] = int(h) in hu.get("typical_hours", [])
    if hu["present"]: hu["position"] = {"x": random.uniform(50,350), "y": random.uniform(50,250), "z": random.uniform(80,170)}
    cat = next((x for x in k["threats"] if x["id"] == "cat"), None)
    if cat:
        if random.random() < 0.1: cat["state"] = random.choice(["sleeping","walking","watching","eating"])
        if cat["state"] == "walking":
            cat["position"]["x"] = clamp(cat["position"]["x"]+random.uniform(-10,10), 0, k["dimensions"]["width"])
            cat["position"]["y"] = clamp(cat["position"]["y"]+random.uniform(-10,10), 0, k["dimensions"]["depth"])
        cat["interest_in_flies"] = 0.8 if cat["state"] == "watching" else 0.2
        cat["danger_level"] = 0.9 if cat["state"] == "watching" else 0.3
    k["events"] = []
    if random.random() < 0.05: k["events"].append({"type": random.choice(["door_slam","faucet","dish_clatter","microwave"]), "tick": t})

def check_death(s):
    e, lc = s["energy"], s["lifecycle"]
    if e["current"] <= 0: s["_meta"]["cause_of_death"] = s["_meta"].get("cause_of_death") or "starvation"; return True
    if lc["stage"] == "adult" and lc["stage_tick"] >= lc["stage_durations"]["adult"]: s["_meta"]["cause_of_death"] = "old_age"; return True
    if s["body"]["integrity"] <= 0: s["_meta"]["cause_of_death"] = "body_failure"; return True
    return False

def describe(s, ev):
    stage, e, t = s["lifecycle"]["stage"], s["energy"]["current"], s["lifecycle"]["total_ticks"]
    d = {"cells_dividing": "Inside the egg, cells divide rapidly. A tiny organism takes shape.",
         "egg_cracking": "The egg shell thins. Hairline cracks appear.",
         "development_slowed_cold": "Cold slows development.",
         "heat_damage": "Excessive heat damages the egg.",
         "crawling": "The larva inches across the counter. Energy: {:.0f}".format(e),
         "histolysis": "Inside the pupal case, the larval body dissolves into cellular soup.",
         "histogenesis": "New structures emerge: wings, compound eyes, legs.",
         "adult_forming": "Transformation nears completion. A fly takes shape.",
         "flying": "The fly buzzes through kitchen air.",
         "exploring": "Curiosity drives the fly into new territory.",
         "resting": "The fly lands and rests, conserving energy.",
         "attracted_to_light": "The ceiling light draws the fly upward."}
    if ev and ev.startswith("feeding_on_"): return "Feeding on "+ev[11:].replace("_"," ")+"."
    if ev and ev.startswith("fleeing_"): return "DANGER! Fleeing from "+ev[8:]+"!"
    if ev and ev.startswith("seeking_"): return "The smell of "+ev[8:].replace("_"," ")+" guides the fly."
    if ev and ev.startswith("metamorphosis_"):
        parts = ev[14:].split("_to_"); return "METAMORPHOSIS: "+parts[0]+" transforms into "+parts[1]+"!"
    if ev and ev.startswith("death_"): return "The fly has died. Cause: "+ev[6:].replace("_"," ")+"."
    if ev and ev.startswith("caught_in_"): return "Trapped in "+ev[10:].replace("_"," ")+"!"
    return d.get(ev or "", "The {} persists. Tick {}.".format(stage, t))

def tick(s):
    lc, meta = s["lifecycle"], s["_meta"]
    if lc["stage"] == "death": return s
    advance_kitchen(s)
    handlers = {"egg": tick_egg, "larva": tick_larva, "pupa": tick_pupa, "adult": tick_adult}
    ev = handlers.get(lc["stage"], lambda x: "idle")(s)
    lc["stage_tick"] += 1; lc["total_ticks"] += 1
    lc["stage_progress"] = lc["stage_tick"] / lc["stage_durations"].get(lc["stage"], 1)
    stages = ["egg", "larva", "pupa", "adult"]
    idx = stages.index(lc["stage"]) if lc["stage"] in stages else -1
    if lc["stage"] in lc["stage_durations"] and lc["stage_tick"] >= lc["stage_durations"][lc["stage"]]:
        if idx < len(stages)-1:
            ns = stages[idx+1]
            lc["transitions"].append({"from": lc["stage"], "to": ns, "at_tick": lc["total_ticks"], "energy": s["energy"]["current"]})
            lc["stage"] = ns; lc["stage_tick"] = 0; lc["stage_progress"] = 0
            ev = "metamorphosis_"+stages[idx]+"_to_"+ns
    if check_death(s):
        prev = stages[idx] if idx >= 0 else "unknown"
        lc["stage"] = "death"
        lc["transitions"].append({"from": prev, "to": "death", "at_tick": lc["total_ticks"], "energy": s["energy"]["current"], "cause": meta.get("cause_of_death","unknown")})
        meta["died_at"] = datetime.now(timezone.utc).isoformat()
        ev = "death_"+meta.get("cause_of_death","unknown")
    meta["frame"] += 1; meta["total_frames_alive"] = lc["total_ticks"]
    s["history"].append({"frame": meta["frame"], "tick": lc["total_ticks"], "event": ev or "idle",
        "description": describe(s, ev), "position": {k: round(v,1) for k,v in s["body"]["position"].items()},
        "energy": round(s["energy"]["current"],1), "stage": lc["stage"]})
    if len(s["history"]) > 100: s["history"] = s["history"][-100:]
    return s

def main():
    s = load(); n = 1; ud = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "--until" and len(sys.argv) > 2 and sys.argv[2] == "death": ud = True
        elif sys.argv[1] == "--ticks" and len(sys.argv) > 2: n = int(sys.argv[2])
    if ud:
        while s["lifecycle"]["stage"] != "death":
            s = tick(s); print("Frame {:3d} | {:6s} | Energy: {:5.1f} | {}".format(s["_meta"]["frame"], s["lifecycle"]["stage"], s["energy"]["current"], s["history"][-1]["event"]))
        print("\nLived {} ticks. Cause: {}".format(s["_meta"]["total_frames_alive"], s["_meta"]["cause_of_death"]))
    else:
        for _ in range(n):
            s = tick(s)
            if s["lifecycle"]["stage"] == "death": break
        h = s["history"][-1]
        print("Frame {} | {} | Energy: {:.1f}".format(s["_meta"]["frame"], s["lifecycle"]["stage"], s["energy"]["current"]))
        print("  -> {}".format(h["description"]))
    save(s)

if __name__ == "__main__": main()
