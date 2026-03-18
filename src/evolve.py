"""
The Reef - Autonomous digital ecosystem.

One run = one tick. Organisms carry DNA encoding 8 traits.
Natural selection, predation, speciation, mass extinction & recovery.
Python stdlib only.
"""
from __future__ import annotations
import json, math, random, hashlib, os
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(REPO_ROOT / "docs")))
STATE_PATH = STATE_DIR / "world.json"
VIZ_PATH = DOCS_DIR / "state.json"

WW, WH = 800, 600
MAX_POP, INIT_POP = 300, 50
RES_SPAWN, MAX_RES, RES_E = 25, 300, 22.0
MAX_AGE = 180
PRED_R, EAT_R = 12.0, 10.0
REPRO_CD = 4
SPEC_TH = 75.0
HIST_MAX, EVT_MAX = 500, 200

EPOCHS = [(0,"Primordial Soup"),(10,"First Sparks"),(50,"The Cambrian"),
    (150,"Age of Diversity"),(300,"Great Expansion"),(500,"Golden Era"),
    (1000,"Deep Time"),(2000,"The Singularity"),(5000,"Eternal Reef")]

GENES = ["hue","size","speed","perception","aggression","metabolism","repro_threshold","mutation_rate"]
RANGES = {"hue":(0,360),"size":(1.5,6.0),"speed":(0.5,4.0),"perception":(5.0,18.0),
    "aggression":(0.0,1.0),"metabolism":(0.1,0.7),"repro_threshold":(25,60),"mutation_rate":(0.02,0.25)}

_nid = 0

def rdna():
    return "".join("{:02x}".format(random.randint(0,255)) for _ in range(8))

def dgene(dna, i):
    return int(dna[i*2:i*2+2], 16)

def dtrait(dna, name):
    i = GENES.index(name); lo, hi = RANGES[name]
    return lo + (dgene(dna, i) / 255.0) * (hi - lo)

def dtraits(dna):
    return {n: round(dtrait(dna, n), 3) for n in GENES}

def mutdna(dna, rate):
    gs = [int(dna[i:i+2], 16) for i in range(0, 16, 2)]
    for i in range(len(gs)):
        if random.random() < rate:
            gs[i] = max(0, min(255, int(gs[i] + random.gauss(0, 30))))
    return "".join("{:02x}".format(g) for g in gs)

def dnadist(a, b):
    return math.sqrt(sum((int(a[i*2:i*2+2],16)-int(b[i*2:i*2+2],16))**2 for i in range(8)))

def nid():
    global _nid; _nid += 1; return "o-{:06d}".format(_nid)

def niso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

_SYL = ["al","be","ca","de","el","fi","go","he","ix","ju","ka","lu",
    "mi","no","or","pi","qu","ra","si","te","ul","vi","wa","xe"]

def spname(sid):
    h = hashlib.md5(sid.encode()).hexdigest()
    return "{}{}{}us".format(*[_SYL[int(h[i:i+2],16)%len(_SYL)].capitalize() if i==0 else _SYL[int(h[i:i+2],16)%len(_SYL)] for i in range(0,6,2)])

def mkorg(x, y, dna, sid, pid=None, gen=0):
    return {"id":nid(),"dna":dna,"x":round(x,1),"y":round(y,1),"vx":0.0,"vy":0.0,
        "energy":100.0,"age":0,"generation":gen,"parent":pid,
        "species_id":sid,"cooldown":0,"traits":dtraits(dna),"kills":0}

def empty():
    return {"_meta":{"tick":0,"epoch":"Primordial Soup","created_at":niso(),"updated_at":niso(),
        "total_births":0,"total_deaths":0,"total_species":0},
        "config":{"world_w":WW,"world_h":WH},
        "organisms":[],"resources":[],"species":{},"graveyard":[],
        "history":{"population":[],"species_count":[],"resource_count":[],
            "avg_speed":[],"avg_size":[],"avg_aggression":[],"events":[]}}

def seed(w):
    evts = []; nf = random.randint(3, 5)
    for i in range(nf):
        bh = int(i * 360 / nf); sid = "s-{:03d}".format(i)
        hb = int((bh/360)*255)
        bd = "".join("{:02x}".format(g) for g in [hb]+[random.randint(50,200) for _ in range(7)])
        ct = INIT_POP // nf
        w["species"][sid] = {"name":spname(sid),"founder_dna":bd,"color_h":bh,
            "first_seen":0,"peak_pop":ct,"current_pop":ct,"total_born":ct}
        w["_meta"]["total_species"] += 1
        for _ in range(ct):
            dna = mutdna(bd, 0.25)
            w["organisms"].append(mkorg(random.uniform(20,WW-20), random.uniform(20,WH-20), dna, sid))
            w["_meta"]["total_births"] += 1
        evts.append({"tick":0,"type":"genesis","desc":"{} emerges ({})".format(w["species"][sid]["name"],ct)})
    return evts

def wrap(v, l): return v % l

def td(x1,y1,x2,y2):
    dx=min(abs(x1-x2),WW-abs(x1-x2)); dy=min(abs(y1-y2),WH-abs(y1-y2))
    return math.sqrt(dx*dx+dy*dy)

def mvto(o,tx,ty,s):
    dx,dy=tx-o["x"],ty-o["y"]
    if abs(dx)>WW/2: dx=-(dx/abs(dx))*(WW-abs(dx))
    if abs(dy)>WH/2: dy=-(dy/abs(dy))*(WH-abs(dy))
    d=math.sqrt(dx*dx+dy*dy)
    if d<0.1: return
    f=min(s,d)/d; o["vx"]=round(dx*f,2); o["vy"]=round(dy*f,2)
    o["x"]=round(wrap(o["x"]+o["vx"],WW),1); o["y"]=round(wrap(o["y"]+o["vy"],WH),1)

def mvfr(o,tx,ty,s):
    dx,dy=o["x"]-tx,o["y"]-ty
    if abs(dx)>WW/2: dx=-(dx/abs(dx))*(WW-abs(dx))
    if abs(dy)>WH/2: dy=-(dy/abs(dy))*(WH-abs(dy))
    d=math.sqrt(dx*dx+dy*dy)
    if d<0.1: dx,dy,d=random.uniform(-1,1),random.uniform(-1,1),1.0
    f=s/d; o["vx"]=round(dx*f,2); o["vy"]=round(dy*f,2)
    o["x"]=round(wrap(o["x"]+o["vx"],WW),1); o["y"]=round(wrap(o["y"]+o["vy"],WH),1)

def wander(o,s):
    o["vx"]=round(o["vx"]*.6+random.gauss(0,s*.6),2)
    o["vy"]=round(o["vy"]*.6+random.gauss(0,s*.6),2)
    m=math.sqrt(o["vx"]**2+o["vy"]**2)
    if m>s: o["vx"]=round(o["vx"]*s/m,2); o["vy"]=round(o["vy"]*s/m,2)
    o["x"]=round(wrap(o["x"]+o["vx"],WW),1); o["y"]=round(wrap(o["y"]+o["vy"],WH),1)

def tick(w):
    evts=[]; tn=w["_meta"]["tick"]; orgs=w["organisms"]; res=w["resources"]
    for _ in range(min(RES_SPAWN, MAX_RES-len(res))):
        res.append({"x":round(random.uniform(5,WW-5),1),"y":round(random.uniform(5,WH-5),1),
            "energy":round(random.uniform(8,RES_E),1)})
    random.shuffle(orgs); births=[]; deaths=set()
    for o in orgs:
        if o["id"] in deaths: continue
        t=o["traits"]; sp,pc,ag,mt,sz=t["speed"],t["perception"],t["aggression"],t["metabolism"],t["size"]
        near=[(x,td(o["x"],o["y"],x["x"],x["y"])) for x in orgs
              if x["id"]!=o["id"] and x["id"] not in deaths and td(o["x"],o["y"],x["x"],x["y"])<pc]
        threats=[(x,d) for x,d in near if x["traits"]["aggression"]>.5 and x["traits"]["size"]>sz*.8 and x["species_id"]!=o["species_id"]]
        prey=[(x,d) for x,d in near if x["traits"]["size"]<sz*.7 and x["species_id"]!=o["species_id"]] if ag>.5 else []
        bf,bfd=None,pc
        for r in res:
            fd=td(o["x"],o["y"],r["x"],r["y"])
            if fd<bfd: bfd,bf=fd,r
        act=False
        if threats and ag<.4:
            ct=min(threats,key=lambda x:x[1]); mvfr(o,ct[0]["x"],ct[0]["y"],sp*1.3); act=True
        elif prey and ag>.6:
            cp=min(prey,key=lambda x:x[1]); mvto(o,cp[0]["x"],cp[0]["y"],sp)
            if td(o["x"],o["y"],cp[0]["x"],cp[0]["y"])<PRED_R:
                o["energy"]+=cp[0]["energy"]*.6+cp[0]["traits"]["size"]*3; o["kills"]+=1; deaths.add(cp[0]["id"])
            act=True
        if not act and bf:
            mvto(o,bf["x"],bf["y"],sp)
            if td(o["x"],o["y"],bf["x"],bf["y"])<EAT_R:
                o["energy"]+=bf["energy"]
                if bf in res: res.remove(bf)
            act=True
        if not act: wander(o,sp)
        o["energy"]-=mt*(.25+sp*.1+sz*.05); o["energy"]=round(o["energy"],1); o["age"]+=1
        if o["energy"]>t["repro_threshold"] and o["cooldown"]<=0 and len(orgs)+len(births)<MAX_POP:
            cd=mutdna(o["dna"],t["mutation_rate"]); ce=o["energy"]*.45; o["energy"]*=.5
            fd=w["species"].get(o["species_id"],{}).get("founder_dna",o["dna"])
            dr=dnadist(cd,fd)
            if dr>SPEC_TH:
                ns="s-{:03d}".format(w["_meta"]["total_species"]); ch=dtrait(cd,"hue")
                w["species"][ns]={"name":spname(ns),"founder_dna":cd,"color_h":round(ch),
                    "first_seen":tn,"peak_pop":1,"current_pop":1,"total_born":1}
                w["_meta"]["total_species"]+=1; si=ns
                evts.append({"tick":tn,"type":"speciation","desc":"New species: {}".format(w["species"][ns]["name"])})
            else: si=o["species_id"]
            a=random.uniform(0,2*math.pi)
            child=mkorg(wrap(o["x"]+math.cos(a)*10,WW),wrap(o["y"]+math.sin(a)*10,WH),cd,si,o["id"],o["generation"]+1)
            child["energy"]=round(ce,1); births.append(child); o["cooldown"]=REPRO_CD; w["_meta"]["total_births"]+=1
        if o["cooldown"]>0: o["cooldown"]-=1
        if o["energy"]<=0 or o["age"]>MAX_AGE: deaths.add(o["id"])
    for o in orgs:
        if o["id"] in deaths:
            w["_meta"]["total_deaths"]+=1
            w["graveyard"].append({"id":o["id"],"species":o["species_id"],"generation":o["generation"],
                "age":o["age"],"kills":o["kills"],"tick":tn})
    w["organisms"]=[o for o in orgs if o["id"] not in deaths]+births
    w["graveyard"]=w["graveyard"][-100:]
    sp_pop={}
    for o in w["organisms"]: sp_pop[o["species_id"]]=sp_pop.get(o["species_id"],0)+1
    for sid,sd in w["species"].items():
        p=sp_pop.get(sid,0); sd["current_pop"]=p
        if p>sd["peak_pop"]: sd["peak_pop"]=p
        if p==0 and sd.get("first_seen",0)<tn and "extinct_tick" not in sd:
            sd["extinct_tick"]=tn
            evts.append({"tick":tn,"type":"extinction","desc":"{} went extinct".format(sd["name"])})
    if not w["organisms"]:
        evts.append({"tick":tn,"type":"extinction_event","desc":"Mass extinction! Re-seeding..."})
        evts.extend(seed(w))
    pop=len(w["organisms"]); alive=sum(1 for s in w["species"].values() if s["current_pop"]>0)
    h=w["history"]
    h["population"].append(pop); h["species_count"].append(alive); h["resource_count"].append(len(res))
    if pop>0:
        h["avg_speed"].append(round(sum(o["traits"]["speed"] for o in w["organisms"])/pop,2))
        h["avg_size"].append(round(sum(o["traits"]["size"] for o in w["organisms"])/pop,2))
        h["avg_aggression"].append(round(sum(o["traits"]["aggression"] for o in w["organisms"])/pop,3))
    else: h["avg_speed"].append(0); h["avg_size"].append(0); h["avg_aggression"].append(0)
    h["events"].extend(evts)
    for k in ["population","species_count","resource_count","avg_speed","avg_size","avg_aggression"]:
        if len(h[k])>HIST_MAX: h[k]=h[k][-HIST_MAX:]
    if len(h["events"])>EVT_MAX: h["events"]=h["events"][-EVT_MAX:]
    for th,nm in reversed(EPOCHS):
        if tn>=th: w["_meta"]["epoch"]=nm; break
    return evts

def load():
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f: w=json.load(f)
            if "tick" not in w.get("_meta",{}): return empty()
            global _nid; mx=0
            for o in w.get("organisms",[])+w.get("graveyard",[]):
                try:
                    n=int(o["id"].split("-")[1])
                    if n>mx: mx=n
                except: pass
            _nid=mx; return w
        except: pass
    return empty()

def save(w):
    w["_meta"]["updated_at"]=niso()
    STATE_DIR.mkdir(parents=True,exist_ok=True); DOCS_DIR.mkdir(parents=True,exist_ok=True)
    with open(STATE_PATH,"w") as f: json.dump(w,f,separators=(",",":"))
    viz={"_meta":w["_meta"],"config":w["config"],
        "organisms":[{"id":o["id"],"x":o["x"],"y":o["y"],"vx":o["vx"],"vy":o["vy"],
            "energy":round(o["energy"],1),"age":o["age"],"generation":o["generation"],
            "species_id":o["species_id"],"traits":{"hue":o["traits"]["hue"],"size":o["traits"]["size"],
            "speed":o["traits"]["speed"],"aggression":o["traits"]["aggression"]},"kills":o["kills"]}
            for o in w["organisms"]],
        "resources":w["resources"],"species":w["species"],"history":w["history"]}
    with open(VIZ_PATH,"w") as f: json.dump(viz,f,indent=1)

def main():
    w=load(); tn=w["_meta"]["tick"]
    if tn==0 and not w["organisms"]:
        print("The Reef - Seeding primordial world...")
        ev=seed(w); w["history"]["events"].extend(ev)
        for e in ev: print("  "+e["desc"])
    else: print("The Reef - Tick {}".format(tn))
    w["_meta"]["tick"]+=1; events=tick(w)
    for e in events: print("  "+e["desc"])
    pop=len(w["organisms"]); alive=sum(1 for s in w["species"].values() if s["current_pop"]>0)
    mg=max((o["generation"] for o in w["organisms"]),default=0)
    print("  Pop: {} | Species: {} | Gen: {} | {}".format(pop,alive,mg,w["_meta"]["epoch"]))
    save(w)

if __name__=="__main__": main()
