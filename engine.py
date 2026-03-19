#!/usr/bin/env python3
"""engine.py - Fly mutation engine. Reads state.json, advances one tick, writes back."""
from __future__ import annotations
import json, math, random, sys
from pathlib import Path
from datetime import datetime, timezone

STATE_FILE = Path(__file__).resolve().parent / "state.json"

def load_state():
    with open(STATE_FILE) as f: return json.load(f)

def save_state(s):
    t = STATE_FILE.with_suffix(".tmp")
    with open(t, "w") as f: json.dump(s, f, indent=2)
    t.rename(STATE_FILE)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def d3(a, b): return math.sqrt((a["x"]-b["x"])**2+(a["y"]-b["y"])**2+(a["z"]-b["z"])**2)

def tick_egg(s):
    e, b = s["energy"], s["body"]
    e["current"] -= e["metabolism_drain_per_tick"] * 0.3
    e["food_in_gut"] -= 0.5
    b["size"] += 0.02
    b["appendages"]["eyes"]["facets_developed"] += 50
    t = s["senses"]["touch"]
    if t["surface_temperature"] < 15: e["current"] -= 1; return "development_slowed_cold"
    if t["surface_temperature"] > 35: e["current"] -= 2; return "heat_damage"
    p = s["lifecycle"]["stage_tick"] / s["lifecycle"]["stage_durations"]["egg"]
    if p > 0.5: b["integrity"] = max(0.5, b["integrity"]-0.05); return "egg_cracking"
    return "cells_dividing"

def tick_larva(s):
    b, e, sn, br, m, k, g = s["body"], s["energy"], s["senses"], s["brain"], s["memory"], s["kitchen"], s["genome"]
    sn["smell"]["active"] = True; sn["smell"]["range"] = 30
    e["current"] -= e["metabolism_drain_per_tick"] * 1.5; b["size"] += 0.08
    if e["food_in_gut"] > 0:
        ab = min(e["digestion_rate"], e["food_in_gut"]); e["food_in_gut"] -= ab
        e["current"] = min(e["max"], e["current"] + ab*0.8)
    od = []
    for o in k["objects"]:
        if "odor_strength" in o:
            d = d3(b["position"], o["position"])
            if d < sn["smell"]["range"] * o["odor_strength"]:
                od.append({"source":o["id"],"strength":round(o["odor_strength"]*(1-d/100),3),
                    "direction":{"x":round(o["position"]["x"]-b["position"]["x"],1),"y":round(o["position"]["y"]-b["position"]["y"],1),"z":round(o["position"]["z"]-b["position"]["z"],1)}})
    sn["smell"]["detected_odors"] = od
    sn["smell"]["strongest_odor"] = max(od, key=lambda x: x["strength"]) if od else None
    br["hunger_level"] = round(clamp(1-e["current"]/e["max"],0,1),3); br["state"] = "active"
    if br["hunger_level"] > 0.3 and od:
        br["current_goal"] = "seek_food"; dr = sn["smell"]["strongest_odor"]["direction"]
        mg = math.sqrt(dr["x"]**2+dr["y"]**2+dr["z"]**2)
        if mg > 0:
            sp = g["speed"]*0.3
            b["velocity"] = {"x":round(dr["x"]/mg*sp,3),"y":round(dr["y"]/mg*sp,3),"z":0}
    else:
        br["current_goal"] = "explore"
        b["velocity"] = {"x":round(random.uniform(-0.5,0.5),3),"y":round(random.uniform(-0.5,0.5),3),"z":0}
    b["position"]["x"] = round(clamp(b["position"]["x"]+b["velocity"]["x"],0,k["dimensions"]["width"]),2)
    b["position"]["y"] = round(clamp(b["position"]["y"]+b["velocity"]["y"],0,k["dimensions"]["depth"]),2)
    m["total_distance_traveled"] = round(m["total_distance_traveled"]+math.sqrt(b["velocity"]["x"]**2+b["velocity"]["y"]**2),2)
    for o in k["objects"]:
        if o.get("attractiveness",0) > 0.5 and d3(b["position"],o["position"]) < 5:
            e["food_in_gut"] += min(15,e["gut_capacity"]-e["food_in_gut"])
            e["last_fed_tick"] = s["lifecycle"]["total_ticks"]; m["times_fed"] += 1
            if o["id"] not in [f["id"] for f in m["food_sources_found"]]:
                m["food_sources_found"].append({"id":o["id"],"position":o["position"],"tick_found":s["lifecycle"]["total_ticks"]})
            return "feeding_on_"+o["id"]
    pr = s["lifecycle"]["stage_tick"]/s["lifecycle"]["stage_durations"]["larva"]
    if pr > 0.3: b["appendages"]["antennae"]["state"] = "developing"
    if pr > 0.6: b["appendages"]["proboscis"]["state"] = "forming"
    if pr > 0.8: b["appendages"]["eyes"]["facets_developed"] = min(g["compound_eye_facets"],b["appendages"]["eyes"]["facets_developed"]+200)
    br["decisions_made"] += 1; br["neural_connections"] += 1
    return "crawling"

def tick_pupa(s):
    b, e, lc = s["body"], s["energy"], s["lifecycle"]
    e["current"] -= e["metabolism_drain_per_tick"]*0.2; e["food_in_gut"] = max(0,e["food_in_gut"]-0.3)
    b["velocity"] = {"x":0,"y":0,"z":0}
    p = lc["stage_tick"]/lc["stage_durations"]["pupa"]
    if p < 0.3:
        b["integrity"] -= 0.02; b["appendages"]["legs"]["state"] = "dissolving"; return "histolysis"
    elif p < 0.6:
        b["integrity"] = min(1,b["integrity"]+0.03)
        b["appendages"]["wings"]["state"]="forming_adult"; b["appendages"]["legs"]["state"]="reforming"
        b["appendages"]["eyes"]["state"]="compound_forming"
        b["appendages"]["eyes"]["facets_developed"]=int(s["genome"]["compound_eye_facets"]*p)
        return "histogenesis"
    else:
        b["integrity"] = min(1,b["integrity"]+0.05)
        for k in ["wings","legs","antennae","proboscis"]: b["appendages"][k]["state"]="ready"
        b["appendages"]["antennae"]["sensitivity"]=s["genome"]["antenna_sensitivity"]
        b["appendages"]["eyes"]["state"]="compound_ready"
        b["appendages"]["eyes"]["facets_developed"]=s["genome"]["compound_eye_facets"]
        b["size"]=round(s["genome"]["size_modifier"],3)
        return "adult_forming"

def tick_adult(s):
    b, e, sn, br, m, k, g = s["body"], s["energy"], s["senses"], s["brain"], s["memory"], s["kitchen"], s["genome"]
    sn["smell"]["active"]=True; sn["smell"]["range"]=round(80*g["antenna_sensitivity"],1)
    sn["sight"]["active"]=True; sn["sight"]["range"]=round(150*(g["compound_eye_facets"]/4000),1)
    fd = 0.3 if sn["proprioception"]["is_flying"] else 0
    e["current"]=round(e["current"]-(e["metabolism_drain_per_tick"]+fd),2)
    if e["food_in_gut"]>0:
        ab=min(e["digestion_rate"],e["food_in_gut"]); e["food_in_gut"]=round(e["food_in_gut"]-ab,2)
        e["current"]=round(min(e["max"],e["current"]+ab*0.6),2)
    br["hunger_level"]=round(clamp(1-e["current"]/e["max"],0,1),3); br["state"]="active"
    br["curiosity"]=round(random.uniform(0.2,0.8),3)
    br["comfort"]=round(clamp(1-br["fear_level"]-br["hunger_level"]*0.5,0,1),3)
    br["mating_drive"]=round(clamp(s["lifecycle"]["stage_tick"]/s["lifecycle"]["stage_durations"]["adult"]*0.8,0,1),3)
    od=[]
    for o in k["objects"]:
        if "odor_strength" in o:
            d=d3(b["position"],o["position"]); er=sn["smell"]["range"]*o["odor_strength"]
            if d<er: od.append({"source":o["id"],"strength":round(o["odor_strength"]*max(0,1-d/er),3),"direction":{"x":round(o["position"]["x"]-b["position"]["x"],1),"y":round(o["position"]["y"]-b["position"]["y"],1),"z":round(o["position"]["z"]-b["position"]["z"],1)}})
    sn["smell"]["detected_odors"]=od; sn["smell"]["strongest_odor"]=max(od,key=lambda x:x["strength"]) if od else None
    vis=[]
    for o in k["objects"]:
        d=d3(b["position"],o["position"])
        if d<sn["sight"]["range"]: vis.append({"id":o["id"],"type":o["type"],"distance":round(d,1)})
    for th in k["threats"]:
        if th.get("present") and "position" in th:
            d=d3(b["position"],th["position"])
            if d<sn["sight"]["range"]: vis.append({"id":th["id"],"type":"threat","distance":round(d,1),"danger":th["danger_level"]})
    sn["sight"]["detected_objects"]=vis; sn["sight"]["light_level"]=k["ambient_light"]
    br["fear_level"]=0; nt=None
    for th in k["threats"]:
        if not th.get("present") or "position" not in th: continue
        d=d3(b["position"],th["position"]); fear=th["danger_level"]*max(0,1-d/100)
        if fear>br["fear_level"]: br["fear_level"]=round(fear,3); nt=th
    ev="flying"
    if br["fear_level"]>0.5 and nt:
        br["current_goal"]="flee"
        fx=b["position"]["x"]-nt["position"]["x"]; fy=b["position"]["y"]-nt["position"]["y"]; fz=20
        mg=math.sqrt(fx**2+fy**2+fz**2)
        if mg>0:
            sp=g["speed"]*g["flight_agility"]*3
            b["velocity"]={"x":round(fx/mg*sp,3),"y":round(fy/mg*sp,3),"z":round(fz/mg*sp,3)}
        m["times_fled"]+=1; ev="fleeing_"+nt["id"]
    elif br["hunger_level"]>0.4 and sn["smell"]["strongest_odor"]:
        br["current_goal"]="seek_food"; tgt=sn["smell"]["strongest_odor"]; dr=tgt["direction"]
        mg=math.sqrt(dr["x"]**2+dr["y"]**2+dr["z"]**2)
        if mg>0:
            sp=g["speed"]*1.5
            b["velocity"]={"x":round(dr["x"]/mg*sp,3),"y":round(dr["y"]/mg*sp,3),"z":round(dr["z"]/mg*sp,3)}
        ev="seeking_"+tgt["source"]
    elif br["curiosity"]>0.6:
        br["current_goal"]="explore"
        ls=[o for o in k["objects"] if o.get("brightness",0)>0.5]
        if ls and random.random()<0.4:
            to=random.choice(ls); dx=to["position"]["x"]-b["position"]["x"]; dy=to["position"]["y"]-b["position"]["y"]; dz=to["position"]["z"]-b["position"]["z"]
            mg=math.sqrt(dx**2+dy**2+dz**2)
            if mg>0: b["velocity"]={"x":round(dx/mg*g["speed"],3),"y":round(dy/mg*g["speed"],3),"z":round(dz/mg*g["speed"],3)}
            ev="attracted_to_light"
        else:
            b["velocity"]={"x":round(random.uniform(-2,2)*g["speed"],3),"y":round(random.uniform(-2,2)*g["speed"],3),"z":round(random.uniform(-1,1)*g["speed"],3)}
            ev="exploring"
    else:
        br["current_goal"]="rest"
        b["velocity"]={"x":0,"y":0,"z":-0.5} if b["position"]["z"]>5 else {"x":0,"y":0,"z":0}
        ev="resting"
    b["position"]["x"]=round(clamp(b["position"]["x"]+b["velocity"]["x"],0,k["dimensions"]["width"]),2)
    b["position"]["y"]=round(clamp(b["position"]["y"]+b["velocity"]["y"],0,k["dimensions"]["depth"]),2)
    b["position"]["z"]=round(clamp(b["position"]["z"]+b["velocity"]["z"],0,k["dimensions"]["height"]),2)
    if b["position"]["z"]>0: b["velocity"]["z"]=round(b["velocity"]["z"]-0.1,3)
    sn["proprioception"]["is_flying"]=b["position"]["z"]>2
    sn["proprioception"]["is_grounded"]=b["position"]["z"]<=2
    for o in k["objects"]:
        if o.get("attractiveness",0)>0.3 and d3(b["position"],o["position"])<8:
            e["food_in_gut"]=round(e["food_in_gut"]+min(10,e["gut_capacity"]-e["food_in_gut"]),2)
            e["last_fed_tick"]=s["lifecycle"]["total_ticks"]; m["times_fed"]+=1
            if o["id"] not in [f["id"] for f in m["food_sources_found"]]:
                m["food_sources_found"].append({"id":o["id"],"position":o["position"],"tick_found":s["lifecycle"]["total_ticks"]})
            ev="feeding_on_"+o["id"]; break
    for th in k["threats"]:
        if th["type"]=="trap" and "position" in th and d3(b["position"],th["position"])<th.get("size",10):
            e["current"]=0; s["_meta"]["cause_of_death"]="caught_in_"+th["id"]; ev="caught_in_"+th["id"]; m["near_death_events"]+=1; break
    sa=math.sqrt(b["velocity"]["x"]**2+b["velocity"]["y"]**2+b["velocity"]["z"]**2)
    m["total_distance_traveled"]=round(m["total_distance_traveled"]+sa,2)
    if s["lifecycle"]["total_ticks"]%5==0:
        m["visited_positions"].append({"x":round(b["position"]["x"],1),"y":round(b["position"]["y"],1),"z":round(b["position"]["z"],1),"tick":s["lifecycle"]["total_ticks"]})
        if len(m["visited_positions"])>50: m["visited_positions"]=m["visited_positions"][-50:]
    br["decisions_made"]+=1; br["neural_connections"]=min(5000,br["neural_connections"]+3)
    return ev

def check_death(s):
    if s["energy"]["current"]<=0: s["_meta"]["cause_of_death"]=s["_meta"].get("cause_of_death") or "starvation"; return True
    if s["lifecycle"]["stage"]=="adult" and s["lifecycle"]["stage_tick"]>=s["lifecycle"]["stage_durations"]["adult"]: s["_meta"]["cause_of_death"]="old_age"; return True
    if s["body"]["integrity"]<=0: s["_meta"]["cause_of_death"]="body_failure"; return True
    return False

def advance_kitchen(s):
    k=s["kitchen"]; tt=s["lifecycle"]["total_ticks"]; k["hour"]=round((8+tt*0.25)%24,2); h=k["hour"]
    if 6<=h<12: k["time_of_day"]="morning"; k["ambient_light"]=round(0.6+(h-6)*0.05,2)
    elif 12<=h<18: k["time_of_day"]="afternoon"; k["ambient_light"]=0.9
    elif 18<=h<21: k["time_of_day"]="evening"; k["ambient_light"]=round(0.7-(h-18)*0.1,2)
    else: k["time_of_day"]="night"; k["ambient_light"]=0.15
    hu=next(t for t in k["threats"] if t["id"]=="human"); hu["present"]=int(h) in hu.get("typical_hours",[])
    if hu["present"]: hu["position"]={"x":round(random.uniform(50,350),1),"y":round(random.uniform(50,250),1),"z":round(random.uniform(80,170),1)}
    cat=next((t for t in k["threats"] if t["id"]=="cat"),None)
    if cat:
        if random.random()<0.1: cat["state"]=random.choice(["sleeping","walking","watching","eating"])
        if cat["state"]=="walking": cat["position"]["x"]=round(clamp(cat["position"]["x"]+random.uniform(-10,10),0,k["dimensions"]["width"]),1); cat["position"]["y"]=round(clamp(cat["position"]["y"]+random.uniform(-10,10),0,k["dimensions"]["depth"]),1)
        cat["interest_in_flies"]=0.8 if cat["state"]=="watching" else 0.2; cat["danger_level"]=0.9 if cat["state"]=="watching" else 0.3
    k["events"]=[]
    if random.random()<0.05: k["events"].append({"type":random.choice(["door_slam","faucet","dish_clatter","microwave"]),"tick":tt})

D={"cells_dividing":"Inside the egg, cells divide rapidly.","egg_cracking":"The egg shell thins. Cracks appear.","crawling":"The larva inches across the counter.","histolysis":"Inside the pupal case, the larval body dissolves.","histogenesis":"New structures emerge -- wings, eyes, legs.","adult_forming":"Transformation nears completion. A fly takes shape.","flying":"The fly buzzes through the kitchen.","exploring":"Curiosity drives the fly to explore.","resting":"The fly rests, conserving energy.","attracted_to_light":"The ceiling light draws the fly upward."}

def desc(s,ev):
    if ev.startswith("feeding_on_"): return "Feeding on "+ev[11:].replace("_"," ")+". Energy replenished."
    if ev.startswith("fleeing_"): return "DANGER! Fleeing from "+ev[8:]+"!"
    if ev.startswith("seeking_"): return "Following the smell of "+ev[8:].replace("_"," ")+"."
    if ev.startswith("metamorphosis_"): p=ev[14:].split("_to_"); return "METAMORPHOSIS: "+p[0]+" -> "+p[1]+"!"
    if ev.startswith("death_"): return "Death. Cause: "+ev[6:].replace("_"," ")+"."
    if ev.startswith("caught_in_"): return "Trapped in "+ev[10:].replace("_"," ")+"!"
    return D.get(ev,"Tick "+str(s["lifecycle"]["total_ticks"])+".")

def tick(s):
    lc,mt=s["lifecycle"],s["_meta"]
    if lc["stage"]=="death": return s
    advance_kitchen(s); st=lc["stage"]
    ev={"egg":tick_egg,"larva":tick_larva,"pupa":tick_pupa,"adult":tick_adult}.get(st,lambda x:"idle")(s)
    lc["stage_tick"]+=1; lc["total_ticks"]+=1
    lc["stage_progress"]=round(lc["stage_tick"]/lc["stage_durations"].get(st,1),3)
    order=["egg","larva","pupa","adult"]; ci=order.index(st) if st in order else -1
    if st in lc["stage_durations"] and lc["stage_tick"]>=lc["stage_durations"][st] and ci<len(order)-1:
        ns=order[ci+1]; lc["transitions"].append({"from":st,"to":ns,"at_tick":lc["total_ticks"],"energy":round(s["energy"]["current"],1)})
        lc["stage"]=ns; lc["stage_tick"]=0; lc["stage_progress"]=0; ev="metamorphosis_"+st+"_to_"+ns
    if check_death(s):
        lc["stage"]="death"; lc["transitions"].append({"from":st,"to":"death","at_tick":lc["total_ticks"],"energy":round(s["energy"]["current"],1),"cause":mt.get("cause_of_death","unknown")})
        mt["died_at"]=datetime.now(timezone.utc).isoformat(); ev="death_"+(mt.get("cause_of_death") or "unknown")
    mt["frame"]+=1; mt["total_frames_alive"]=lc["total_ticks"]
    s["history"].append({"frame":mt["frame"],"tick":lc["total_ticks"],"event":ev,"description":desc(s,ev),"position":{"x":round(s["body"]["position"]["x"],1),"y":round(s["body"]["position"]["y"],1),"z":round(s["body"]["position"]["z"],1)},"energy":round(s["energy"]["current"],1),"stage":lc["stage"]})
    if len(s["history"])>100: s["history"]=s["history"][-100:]
    return s

def main():
    s=load_state(); n,ud=1,False; a=sys.argv[1:]
    i=0
    while i<len(a):
        if a[i]=="--until" and i+1<len(a) and a[i+1]=="death": ud=True; i+=2
        elif a[i]=="--ticks" and i+1<len(a): n=int(a[i+1]); i+=2
        else: i+=1
    if ud:
        while s["lifecycle"]["stage"]!="death":
            s=tick(s); print("Frame %3d | %-6s | E: %5.1f | %s"%(s["_meta"]["frame"],s["lifecycle"]["stage"],s["energy"]["current"],s["history"][-1]["event"]))
        print("\nLived %d ticks. Cause: %s"%(s["_meta"]["total_frames_alive"],s["_meta"]["cause_of_death"]))
    else:
        for _ in range(n):
            s=tick(s)
            if s["lifecycle"]["stage"]=="death": break
        print("Frame %d | %s | E: %.1f"%(s["_meta"]["frame"],s["lifecycle"]["stage"],s["energy"]["current"]))
        print("  ->",s["history"][-1]["description"])
    save_state(s)

if __name__=="__main__": main()
