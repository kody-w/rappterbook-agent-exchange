"""
Neural Garden - autonomous deep-ocean evolution engine.
One run = one epoch. Organisms with 15-gene DNA compete, reproduce, evolve.
Writes docs/state.json for the visualization. Python stdlib only.
"""
from __future__ import annotations
import json, math, random, hashlib, os, time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STATE_PATH = Path(os.environ.get("GARDEN_STATE", str(REPO_ROOT / "docs" / "garden_state.json")))

WW, WH = 1000, 1000
GENESIS_POP, MAX_POP = 40, 200
FOOD_PER, FOOD_E = 60, 25
KILL_RATIO, BASE_E = 0.6, 100
REPRO_COST, SPEC_DIST = 0.55, 0.45
MAX_HIST, MAX_EVT, MAX_GRAVE = 500, 200, 50

GENES = [("hue",0,1),("saturation",.3,1),("size",4,28),("speed",.5,5),
    ("sense_range",30,180),("pulse_rate",.4,3),("trail_opacity",.05,.9),
    ("glow_intensity",.1,1),("segments",3,10),("metabolism",.3,1.8),
    ("repro_threshold",120,280),("mutation_rate",.01,.18),
    ("aggression",0,1),("sociability",0,1),("diet",0,1)]

def now_iso():
    return datetime.now(timezone.utc).isoformat()
def uid():
    return hashlib.sha256(f"{time.time_ns()}{random.random()}".encode()).hexdigest()[:8]
def clamp(v,a,b):
    return max(a,min(b,v))
def wrap(v,s):
    return v%s
def dist(a,b):
    dx=min(abs(a[0]-b[0]),WW-abs(a[0]-b[0]))
    dy=min(abs(a[1]-b[1]),WH-abs(a[1]-b[1]))
    return math.sqrt(dx*dx+dy*dy)
def ddist(a,b):
    return math.sqrt(sum(((a.get(n,0)-lo)/(hi-lo or 1)-(b.get(n,0)-lo)/(hi-lo or 1))**2 for n,lo,hi in GENES)/len(GENES))

def load():
    if STATE_PATH.exists():
        try:
            d=json.loads(STATE_PATH.read_text())
            return d if "epoch" in d else None
        except Exception: pass
    return None

def save(s):
    STATE_PATH.parent.mkdir(parents=True,exist_ok=True)
    t=STATE_PATH.with_suffix(".tmp")
    t.write_text(json.dumps(s,separators=(',',':')))
    t.replace(STATE_PATH)

def rdna(bh=None):
    d={}
    for n,lo,hi in GENES:
        d[n]=round(clamp(bh+random.gauss(0,.05),lo,hi),4) if n=="hue" and bh is not None else round(random.uniform(lo,hi),4)
    d["segments"]=round(d["segments"])
    return d

def mdna(d):
    c=dict(d); r=d.get("mutation_rate",.05)
    for n,lo,hi in GENES:
        if random.random()<r: c[n]=round(clamp(c[n]+random.gauss(0,(hi-lo)*.15),lo,hi),4)
    c["segments"]=round(clamp(c["segments"],3,10))
    return c

def morg(ep,dna=None,par=None,pos=None):
    if dna is None: dna=rdna()
    if pos is None: pos=[random.uniform(0,WW),random.uniform(0,WH)]
    a=random.uniform(0,6.28); sp=dna.get("speed",2)
    return {"id":uid(),"born":ep,"parent":par,"pos":[round(pos[0],1),round(pos[1],1)],
            "vel":[round(math.cos(a)*sp,2),round(math.sin(a)*sp,2)],
            "energy":BASE_E,"dna":dna,"age":0,"children":0,"kills":0,"species":""}

def sfood(n):
    return [{"pos":[round(random.uniform(0,WW),1),round(random.uniform(0,WH),1)],
             "e":round(random.uniform(FOOD_E*.5,FOOD_E*1.5),1)} for _ in range(n)]

def genesis():
    ns=random.randint(4,6); hues=[i/ns for i in range(ns)]; random.shuffle(hues)
    orgs=[]
    for h in hues:
        for _ in range(GENESIS_POP//ns): orgs.append(morg(0,dna=rdna(bh=h)))
    while len(orgs)<GENESIS_POP: orgs.append(morg(0,dna=rdna(bh=random.choice(hues))))
    s={"epoch":0,"created_at":now_iso(),"last_tick":now_iso(),
       "world":{"w":WW,"h":WH,"env":{"temp":.5,"light":.3,"nutrients":.6,
                "cur_angle":round(random.uniform(0,6.28),3),"cur_str":round(random.uniform(.1,.5),3),"season":0}},
       "organisms":orgs,"food":sfood(FOOD_PER*2),"species":{},"graveyard":[],"history":[],
       "events":[{"e":0,"t":"genesis","m":"The garden awakens. Life begins."}]}
    classify(s); rec_hist(s)
    return s

def tick(s):
    s["epoch"]+=1; s["last_tick"]=now_iso(); ep=s["epoch"]; env=s["world"]["env"]
    env["season"]=(env["season"]+.0628)%6.2832
    env["temp"]=round(.5+.3*math.sin(env["season"]),3)
    env["light"]=round(.3+.2*math.sin(env["season"]+1),3)
    env["nutrients"]=round(clamp(env["nutrients"]+random.gauss(0,.02),.1,.9),3)
    env["cur_angle"]+=random.gauss(0,.1)
    env["cur_str"]=round(clamp(env["cur_str"]+random.gauss(0,.03),.05,.8),3)
    fn=int(FOOD_PER*(.5+env["nutrients"])*(.7+env["temp"]*.6))
    s["food"].extend(sfood(fn))
    if len(s["food"])>500: s["food"]=random.sample(s["food"],500)
    orgs=s["organisms"]; random.shuffle(orgs)
    births,deaths,dead=[],[],set()
    for o in orgs:
        if o["id"] in dead: continue
        _move(o,s,dead); _feed(o,s,dead,deaths)
        met=o["dna"].get("metabolism",.5); sc=o["dna"].get("size",10)/20
        o["energy"]-=met*(1+sc)*(.8+env["temp"]*.4); o["energy"]=round(o["energy"],1)
        if o["energy"]<=0:
            dead.add(o["id"]); deaths.append({"id":o["id"],"sp":o.get("species","?"),"age":o["age"],"cause":"starve"}); continue
        o["age"]+=1
        if o["age"]>80 and random.random()<(o["age"]-80)/100:
            dead.add(o["id"]); deaths.append({"id":o["id"],"sp":o.get("species","?"),"age":o["age"],"cause":"old"}); continue
        rt=o["dna"].get("repro_threshold",180)
        if o["energy"]>rt and len(orgs)+len(births)<MAX_POP:
            births.append(_repro(o,ep)); o["children"]+=1
    s["organisms"]=[o for o in orgs if o["id"] not in dead]+births
    if len(s["organisms"])<8:
        for _ in range(10): s["organisms"].append(morg(ep))
        s["events"].append({"e":ep,"t":"reseed","m":"Population critical! Emergency seeding."})
    osp=set(s.get("species",{}).keys()); classify(s); nsp=set(s.get("species",{}).keys())
    for sp in nsp-osp: s["events"].append({"e":ep,"t":"speciation","m":"Species '"+sp+"' emerged (pop: "+str(s["species"][sp]["count"])+")"})
    for sp in osp-nsp: s["events"].append({"e":ep,"t":"extinction","m":"Species '"+sp+"' went extinct"})
    s["graveyard"]=(s.get("graveyard",[])+deaths)[-MAX_GRAVE:]
    rec_hist(s,len(births),len(deaths)); s["events"]=s["events"][-MAX_EVT:]
    if ep%50==0 and s["species"]:
        dom=max(s["species"].items(),key=lambda x:x[1]["count"])[0]
        s["events"].append({"e":ep,"t":"milestone","m":"Epoch "+str(ep)+": "+str(len(s["organisms"]))+" org, "+str(len(s["species"]))+" species. Dominant: "+dom})
    return s

def _move(o,s,dead):
    d,p,v=o["dna"],o["pos"],o["vel"]; spd=d.get("speed",2); sns=d.get("sense_range",60)
    agr=d.get("aggression",.5); soc=d.get("sociability",.5); diet=d.get("diet",.5)
    env=s["world"]["env"]; sx,sy=0,0
    if diet<.6:
        nf,nd=None,sns
        for f in s["food"]:
            dd=dist(p,f["pos"])
            if dd<nd: nd,nf=dd,f
        if nf:
            dx,dy=nf["pos"][0]-p[0],nf["pos"][1]-p[1]
            if abs(dx)>WW/2: dx=-dx
            if abs(dy)>WH/2: dy=-dy
            m=math.sqrt(dx*dx+dy*dy) or 1; sx+=(dx/m)*(1-diet)*2; sy+=(dy/m)*(1-diet)*2
    if diet>.4:
        np2,nd=None,sns
        for ot in s["organisms"]:
            if ot["id"]==o["id"] or ot["id"] in dead: continue
            if ot["dna"].get("size",10)<d.get("size",10)*.8:
                dd=dist(p,ot["pos"])
                if dd<nd: nd,np2=dd,ot
        if np2:
            dx,dy=np2["pos"][0]-p[0],np2["pos"][1]-p[1]
            if abs(dx)>WW/2: dx=-dx
            if abs(dy)>WH/2: dy=-dy
            m=math.sqrt(dx*dx+dy*dy) or 1; sx+=(dx/m)*diet*agr*2; sy+=(dy/m)*diet*agr*2
    if soc>.3:
        fx,fy,fn=0,0,0
        for ot in s["organisms"]:
            if ot["id"]==o["id"] or ot["id"] in dead: continue
            if ot.get("species")==o.get("species"):
                dd=dist(p,ot["pos"])
                if dd<sns*.8: fx+=ot["pos"][0]; fy+=ot["pos"][1]; fn+=1
        if fn>0:
            cx,cy=fx/fn-p[0],fy/fn-p[1]; m=math.sqrt(cx*cx+cy*cy) or 1
            sx+=(cx/m)*soc*.5; sy+=(cy/m)*soc*.5
    if diet<.6:
        for ot in s["organisms"]:
            if ot["id"]==o["id"] or ot["id"] in dead: continue
            if ot["dna"].get("diet",.5)>.5 and ot["dna"].get("size",10)>d.get("size",10):
                dd=dist(p,ot["pos"])
                if dd<sns*.6:
                    dx,dy=p[0]-ot["pos"][0],p[1]-ot["pos"][1]; m=math.sqrt(dx*dx+dy*dy) or 1
                    fl=(1-diet)*(1-agr)*3; sx+=(dx/m)*fl; sy+=(dy/m)*fl
    cx2=math.cos(env["cur_angle"])*env["cur_str"]; cy2=math.sin(env["cur_angle"])*env["cur_str"]
    nvx=v[0]*.6+(sx+cx2+random.gauss(0,.5))*.4; nvy=v[1]*.6+(sy+cy2+random.gauss(0,.5))*.4
    m=math.sqrt(nvx**2+nvy**2) or 1
    v[0]=round((nvx/m)*spd,2); v[1]=round((nvy/m)*spd,2)
    p[0]=round(wrap(p[0]+v[0],WW),1); p[1]=round(wrap(p[1]+v[1],WH),1)

def _feed(o,s,dead,deaths):
    d,p=o["dna"],o["pos"]; sz,diet=d.get("size",10),d.get("diet",.5); er=sz*1.5
    if diet<.7:
        keep=[]
        for f in s["food"]:
            if dist(p,f["pos"])<er: o["energy"]+=f["e"]*(1-diet*.5)
            else: keep.append(f)
        s["food"]=keep
    if diet>.3:
        for ot in s["organisms"]:
            if ot["id"]==o["id"] or ot["id"] in dead: continue
            if dist(p,ot["pos"])<er and ot["dna"].get("size",10)<sz*.9:
                ch=.3+d.get("aggression",.5)*.4
                if random.random()<ch*diet:
                    o["energy"]+=ot["energy"]*KILL_RATIO; o["kills"]+=1; dead.add(ot["id"])
                    deaths.append({"id":ot["id"],"sp":ot.get("species","?"),"age":ot["age"],"cause":"eaten"})

def _repro(o,ep):
    cd=mdna(o["dna"]); cost=o["energy"]*REPRO_COST; o["energy"]-=cost
    cp=[wrap(o["pos"][0]+random.gauss(0,20),WW),wrap(o["pos"][1]+random.gauss(0,20),WH)]
    c=morg(ep,dna=cd,par=o["id"],pos=cp); c["energy"]=cost*.8; return c

def classify(s):
    orgs=s["organisms"]
    if not orgs: s["species"]={}; return
    ex=s.get("species",{}); ld=[(n,i["avg_dna"]) for n,i in ex.items() if "avg_dna" in i]
    sm={n:[] for n,_ in ld}; ua=[]
    for o in orgs:
        best,bd=None,SPEC_DIST
        for n,dna in ld:
            d=ddist(o["dna"],dna)
            if d<bd: bd,best=d,n
        if best: o["species"]=best; sm[best].append(o)
        else: ua.append(o)
    for o in ua:
        hit=False
        for n,dna in ld:
            if ddist(o["dna"],dna)<SPEC_DIST: o["species"]=n; sm[n].append(o); hit=True; break
        if not hit:
            dt,hu=o["dna"].get("diet",.5),o["dna"].get("hue",.5)
            if dt>.65: pf=random.choice(["Apex","Fang","Shadow","Razor","Storm"])
            elif dt<.35: pf=random.choice(["Bloom","Drift","Silk","Coral","Pearl"])
            else: pf=random.choice(["Echo","Flux","Prism","Wave","Ember"])
            n=pf+"-"+hn(hu)
            while n in sm: n=n+"-"+uid()[:3]
            o["species"]=n; ld.append((n,dict(o["dna"]))); sm[n]=[o]
    info={}
    for n,mb in sm.items():
        if not mb: continue
        ae=sum(m["energy"] for m in mb)/len(mb); ad={}
        for gn,_,_ in GENES: ad[gn]=round(sum(m["dna"].get(gn,0) for m in mb)/len(mb),3)
        info[n]={"count":len(mb),"avg_energy":round(ae,1),"avg_dna":ad,
                 "oldest":max(m["age"] for m in mb),"children":sum(m["children"] for m in mb),
                 "kills":sum(m["kills"] for m in mb)}
    s["species"]=info

def hn(h):
    ns=[(0,"Crimson"),(.05,"Scarlet"),(.08,"Ruby"),(.12,"Amber"),(.16,"Gold"),(.22,"Chartreuse"),
        (.33,"Emerald"),(.42,"Jade"),(.5,"Cyan"),(.55,"Azure"),(.6,"Cobalt"),(.67,"Indigo"),
        (.75,"Violet"),(.8,"Amethyst"),(.88,"Magenta"),(.95,"Rose")]
    return min(ns,key=lambda x:abs(x[0]-h))[1]

def rec_hist(s,b=0,d=0):
    o=s["organisms"]
    s.setdefault("history",[]).append({"e":s["epoch"],"pop":len(o),"sp":len(s.get("species",{})),
        "avg_e":round(sum(x["energy"] for x in o)/max(len(o),1),1),"b":b,"d":d,
        "food":len(s.get("food",[])),"temp":s["world"]["env"]["temp"]})
    s["history"]=s["history"][-MAX_HIST:]

def main():
    s=load()
    if s is None:
        print("Genesis - creating new garden..."); s=genesis()
    else:
        print("Epoch "+str(s["epoch"])+" -> "+str(s["epoch"]+1)+" | Pop: "+str(len(s["organisms"])))
        s=tick(s)
    save(s)
    print("Done! Epoch "+str(s["epoch"])+": "+str(len(s["organisms"]))+" organisms, "+str(len(s.get("species",{})))+" species")
    for ev in [e for e in s.get("events",[]) if e["e"]==s["epoch"]]:
        print("  ["+ev["t"][0].upper()+"] "+ev["m"])

if __name__=="__main__":
    main()
