#!/usr/bin/env python3
"""Musca Domestica tick engine. Advances fly_state.json by one tick."""
from __future__ import annotations
import json, math, random, sys
from datetime import datetime, timezone
from pathlib import Path

def load(p):
    with open(p) as f: return json.load(f)
def save(p, s):
    with open(p, "w") as f: json.dump(s, f, indent=2)
def dist3(a, b):
    return math.sqrt((a["x"]-b["x"])**2+(a["y"]-b["y"])**2+(a["z"]-b["z"])**2)
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def update_senses(fly, kitchen):
    pos, stage = fly["position"], fly["lifecycle"]
    best_s, best_d = None, float("inf")
    for s in kitchen["surfaces"]:
        b = s["bounds"]
        cp = {"x": clamp(pos["x"],b["x"],b["x"]+b["w"]), "y": clamp(pos["y"],b["y"],b["y"]+b["h"]), "z": clamp(pos["z"],b["z"],b["z"]+b["d"])}
        d = dist3(pos, cp)
        if d < best_d: best_d, best_s = d, s
    if best_s:
        fly["senses"]["touch"]["surface"] = best_s["type"]
        fly["senses"]["touch"]["temperature"] = best_s["temperature"]
    fly["senses"]["smell"] = []
    if stage in ("larva", "adult"):
        for food in kitchen["food_sources"]:
            if dist3(pos, food["position"]) < food["smell_radius"] * fly["genome"]["chemotaxis"]:
                fly["senses"]["smell"].append(food["id"])
    fly["senses"]["vision"] = []
    if stage == "adult":
        eq = fly["genome"]["compound_eyes"]
        for l in kitchen["light_sources"]:
            d = dist3(pos, l["position"])
            if d < 300*eq: fly["senses"]["vision"].append({"type":"light","id":l["id"],"distance":round(d,1)})
        for h in kitchen["hazards"]:
            if h["active"] and dist3(pos, h["position"]) < h["danger_radius"]*1.5*eq:
                fly["senses"]["vision"].append({"type":"hazard","id":h["id"],"distance":round(dist3(pos,h["position"]),1)})
    fly["senses"]["hearing"] = []
    if stage == "adult":
        for h in kitchen["hazards"]:
            if h["active"] and h["type"]=="fan" and dist3(pos,h["position"])<400:
                fly["senses"]["hearing"].append("fan_hum")

def decide(fly, kitchen):
    stage, n = fly["lifecycle"], fly["neural"]
    if stage in ("egg","pupa"): return None
    if stage == "larva":
        if fly["hunger"]>0.5 and fly["senses"]["smell"]: return "seek_food"
        return "wander"
    if stage == "adult":
        if n["fear"]>0.7: return "flee"
        if fly["hunger"]>0.6 and fly["senses"]["smell"]: return "seek_food"
        if fly["hydration"]<0.3: return "seek_water"
        if n["curiosity"]>0.6: return "explore"
        for v in fly["senses"]["vision"]:
            if v["type"]=="light" and fly["genome"]["phototaxis"]>0.5: return "approach_light"
        return "rest" if n["arousal"]<0.3 else "patrol"
    return None

def move(fly, kitchen, decision):
    if not decision or fly["lifecycle"] in ("egg","pupa","dead"): return 0.0
    pos, vel, stage = fly["position"], fly["velocity"], fly["lifecycle"]
    spd = (0.5 if stage=="larva" else 3.0) * fly["genome"]["flight_agility"]
    dx=dy=dz=0.0
    if decision=="seek_food" and fly["senses"]["smell"]:
        for food in kitchen["food_sources"]:
            if food["id"]==fly["senses"]["smell"][0]:
                d=dist3(pos,food["position"])
                if d>1: dx=(food["position"]["x"]-pos["x"])/d*spd; dz=(food["position"]["z"]-pos["z"])/d*spd
                break
    elif decision=="flee":
        for h in kitchen["hazards"]:
            if h["active"] and dist3(pos,h["position"])<h["danger_radius"]*2:
                d=max(dist3(pos,h["position"]),0.1)
                dx=(pos["x"]-h["position"]["x"])/d*spd*2; dz=(pos["z"]-h["position"]["z"])/d*spd*2
                if stage=="adult": dy=-spd
                break
    elif decision in ("explore","wander","patrol"):
        a=random.random()*math.pi*2; m=spd*(1 if decision!="wander" else 0.3)
        dx=math.cos(a)*m; dz=math.sin(a)*m
        if stage=="adult" and random.random()<0.3: dy=(random.random()-0.5)*spd
    elif decision=="approach_light":
        for l in kitchen["light_sources"]:
            d=dist3(pos,l["position"])
            if d>5: dx=(l["position"]["x"]-pos["x"])/d*spd; dy=(l["position"]["y"]-pos["y"])/d*spd; dz=(l["position"]["z"]-pos["z"])/d*spd
            break
    elif decision=="seek_water":
        for s in kitchen["surfaces"]:
            if s["type"]=="stainless_sink":
                t={"x":s["bounds"]["x"]+s["bounds"]["w"]/2,"y":s["bounds"]["y"],"z":s["bounds"]["z"]+s["bounds"]["d"]/2}
                d=dist3(pos,t)
                if d>1: dx=(t["x"]-pos["x"])/d*spd; dy=(t["y"]-pos["y"])/d*spd; dz=(t["z"]-pos["z"])/d*spd
                break
    w=kitchen["ambient"]["wind"]; dx+=w["x"]; dz+=w.get("z",0)
    if stage=="adult" and pos["y"]<60: dy+=0.2
    vel["x"]=vel["x"]*0.7+dx*0.3; vel["y"]=vel["y"]*0.7+dy*0.3; vel["z"]=vel["z"]*0.7+dz*0.3
    pos["x"]=clamp(pos["x"]+vel["x"],0,kitchen["width"]); pos["y"]=clamp(pos["y"]+vel["y"],-200,kitchen["height"]); pos["z"]=clamp(pos["z"]+vel["z"],0,kitchen["depth"])
    return math.sqrt(dx*dx+dy*dy+dz*dz)

def update_neural(fly, kitchen):
    n, stage = fly["neural"], fly["lifecycle"]
    if stage in ("egg","dead"): return
    n["arousal"]*=0.95; n["fear"]*=0.9; n["curiosity"]=clamp(n["curiosity"]+(random.random()-0.5)*0.05,0,1)
    if fly["senses"]["smell"]: n["arousal"]=clamp(n["arousal"]+0.1,0,1)
    if any(v.get("type")=="hazard" for v in fly["senses"]["vision"]): n["fear"]=clamp(n["fear"]+0.3,0,1)
    temp=fly["senses"]["touch"]["temperature"]; ideal=sum(fly["memory"]["comfortable_temps"])/2
    n["comfort"]=clamp(1.0-abs(temp-ideal)/15.0,0,1); n["satiation"]=clamp(1.0-fly["hunger"],0,1)
    n["valence"]=(n["comfort"]+n["satiation"]-n["fear"])/3
    if n["fear"]>0.5: n["drive"]="escape"
    elif fly["hunger"]>0.6: n["drive"]="feed"
    elif fly["hydration"]<0.3: n["drive"]="drink"
    elif n["curiosity"]>0.5: n["drive"]="explore"
    elif n["arousal"]<0.2: n["drive"]="rest"
    else: n["drive"]="idle"

def tick(state):
    fly, kitchen, meta, stats, tl = state["fly"], state["kitchen"], state["_meta"], state["stats"], state["timeline"]
    age, stage, th = fly["age_ticks"], fly["lifecycle"], fly["lifecycle_thresholds"]
    if stage == "dead": return state
    fly["age_ticks"] += 1; age = fly["age_ticks"]; stats["total_ticks"] = age
    met = fly["genome"]["metabolism_rate"]
    if stage == "egg": fly["energy"] -= 0.3 * met
    elif stage == "larva": fly["energy"] -= 0.8 * met; fly["hunger"] = clamp(fly["hunger"]+0.02,0,1); fly["hydration"] = clamp(fly["hydration"]-0.005,0,1)
    elif stage == "pupa": fly["energy"] -= 0.5 * met
    elif stage == "adult":
        dr = 1.2 if fly["current_behavior"] in ("flying","exploring","flee") else 0.6
        fly["energy"] -= dr * met; fly["hunger"] = clamp(fly["hunger"]+0.03,0,1); fly["hydration"] = clamp(fly["hydration"]-0.01,0,1)
    fly["energy"] = clamp(fly["energy"], 0, fly["max_energy"])
    ns = stage
    if stage == "egg" and age >= th["egg_to_larva"]:
        ns = "larva"; fly["physical"].update({"segments":6,"size_mm":2.0,"color":"pale_cream","spiracles_open":True})
        fly["energy"] = min(fly["energy"]+20, fly["max_energy"])
        tl.append({"tick":age,"event":"hatch","lifecycle":"larva","description":"The egg splits. A cream-colored larva, 2mm. It smells the banana.","energy":fly["energy"],"position":dict(fly["position"])})
    elif stage == "larva" and age >= th["larva_to_pupa"]:
        ns = "pupa"; fly["physical"].update({"size_mm":5.0,"color":"dark_brown","wing_state":"forming","segments":0})
        fly["current_behavior"] = "dormant"; fly["velocity"] = {"x":0,"y":0,"z":0}
        tl.append({"tick":age,"event":"pupate","lifecycle":"pupa","description":"The larva hardens. Metamorphosis begins.","energy":fly["energy"],"position":dict(fly["position"])})
    elif stage == "pupa" and age >= th["pupa_to_adult"]:
        ns = "adult"; fly["physical"].update({"size_mm":7.0,"color":"dark_grey","wing_state":"ready","legs_count":6,"eyes_state":"compound","antenna_state":"active"})
        fly["max_energy"] = 120; fly["energy"] = 90; fly["current_behavior"] = "resting"
        tl.append({"tick":age,"event":"emerge","lifecycle":"adult","description":"The pupal case cracks. A fly emerges, wings glistening.","energy":fly["energy"],"position":dict(fly["position"])})
    if fly["energy"] <= 0 or (stage == "adult" and age >= th["adult_death"]):
        ns = "dead"; cause = "starvation" if fly["energy"] <= 0 else "old_age"
        stats["cause_of_death"] = cause; fly["current_behavior"] = "dead"; fly["velocity"] = {"x":0,"y":0,"z":0}
        tl.append({"tick":age,"event":"death","lifecycle":"dead","description":"Stillness. "+cause+". "+str(age)+" ticks.","energy":0,"position":dict(fly["position"])})
    fly["lifecycle"] = ns; fly["lifecycle_progress"] = age / th["adult_death"]
    if ns == "dead": meta["last_tick"] = datetime.now(timezone.utc).isoformat(); meta["frame"] += 1; return state
    if ns == "larva":
        fly["physical"]["size_mm"] = min(fly["physical"]["size_mm"]+0.05*fly["genome"]["size_factor"], 8.0)
        if fly["physical"]["segments"] < 12 and random.random() < 0.1: fly["physical"]["segments"] += 1
    elif ns == "pupa":
        prog = (age-th["larva_to_pupa"]) / max(th["pupa_to_adult"]-th["larva_to_pupa"], 1)
        if prog > 0.3: fly["physical"]["eyes_state"] = "forming"
        if prog > 0.5: fly["physical"]["legs_count"] = 6
        if prog > 0.7: fly["physical"]["antenna_state"] = "forming"
    fly["physical"]["body_temp"] += (fly["senses"]["touch"]["temperature"]-fly["physical"]["body_temp"]) * 0.1
    update_senses(fly, kitchen); update_neural(fly, kitchen)
    decision = decide(fly, kitchen)
    if decision:
        fly["current_behavior"] = decision
        fly["decisions_log"].append({"tick":age,"decision":decision})
        if len(fly["decisions_log"]) > 50: fly["decisions_log"] = fly["decisions_log"][-50:]
        stats["total_decisions"] += 1
    d = move(fly, kitchen, decision); stats["total_distance_mm"] += d
    if decision == "seek_food" and fly["senses"]["smell"]:
        for food in kitchen["food_sources"]:
            if food["id"] in fly["senses"]["smell"] and dist3(fly["position"], food["position"]) < 10:
                eg = food["energy_value"] * 0.1; fly["energy"] = min(fly["energy"]+eg, fly["max_energy"])
                fly["hunger"] = clamp(fly["hunger"]-0.15, 0, 1); fly["memory"]["last_meal_tick"] = age
                if food["id"] not in [fl["id"] for fl in fly["memory"]["food_locations"]]:
                    fly["memory"]["food_locations"].append({"id":food["id"],"position":dict(food["position"])})
                stats["total_food_eaten"] += 1; stats["total_energy_consumed"] += eg
                tl.append({"tick":age,"event":"feed","lifecycle":ns,"description":"Feeding on "+food["label"].lower()+".","energy":fly["energy"],"position":dict(fly["position"])})
                break
    for h in kitchen["hazards"]:
        if h["active"]:
            dd = dist3(fly["position"], h["position"])
            if dd < h["danger_radius"]:
                stats["close_calls"] += 1
                fly["neural"]["fear"] = clamp(fly["neural"]["fear"]+0.5, 0, 1)
                if h["id"] not in [dl["id"] for dl in fly["memory"]["danger_locations"]]:
                    fly["memory"]["danger_locations"].append({"id":h["id"],"position":dict(h["position"])})
                if h["lethal"] and dd < h["danger_radius"]*0.3: fly["energy"] = 0
    if ns == "adult" and fly["position"]["y"] < 50:
        stats["flights_taken"] += 1; fly["memory"]["last_flight_tick"] = age
        stats["max_altitude_mm"] = max(stats["max_altitude_mm"], 82-fly["position"]["y"])
    cs = fly["senses"]["touch"]["surface"]
    if cs not in fly["memory"]["visited_surfaces"]:
        fly["memory"]["visited_surfaces"].append(cs); stats["surfaces_visited"] = len(fly["memory"]["visited_surfaces"])
    if len(tl) > 100: tl[:] = tl[:5] + tl[-95:]
    meta["last_tick"] = datetime.now(timezone.utc).isoformat(); meta["frame"] += 1
    return state

if __name__ == "__main__":
    sd = Path("docs")
    for i, a in enumerate(sys.argv[1:]):
        if a.startswith("--state-dir"):
            sd = Path(a.split("=")[1]) if "=" in a else Path(sys.argv[i+2])
    sp = sd / "fly_state.json"
    if not sp.exists(): print("No state at", sp); sys.exit(1)
    s = load(sp); old_s, old_t = s["fly"]["lifecycle"], s["fly"]["age_ticks"]
    s = tick(s); save(sp, s)
    f = s["fly"]
    print("Tick", old_t, "->", f["age_ticks"], "|", old_s, "->", f["lifecycle"], "| E:", round(f["energy"],1), "|", f["current_behavior"])
