#!/usr/bin/env python3
"""Phosphene -- emergent neural ecosystem. One run = one tick.
Neurons self-organize via Kuramoto sync, Hebbian learning, reproduction, selection.
Python stdlib only.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(REPO_ROOT / "docs")))
STATE_PATH = STATE_DIR / "phosphene.json"
VIZ_PATH = DOCS_DIR / "phosphene.json"

WORLD_W, WORLD_H = 1000, 700
INIT_POP, MAX_POP, MIN_POP = 150, 350, 25
MAX_CONNS = 12
WEAK_THRESHOLD = 0.04
DT = 0.1
ENERGY_DRAIN = 0.25
SYNC_BONUS = 0.45
REPRO_THRESHOLD = 65.0
REPRO_COST = 35.0
DEATH_ENERGY = 3.0
MAX_AGE = 300
HISTORY_CAP, EVENT_CAP = 500, 200
TWO_PI = 2 * math.pi

GENE_NAMES = ["freq","coupling","excitability","decay","growth",
              "hue","size","reach","plasticity","mutation"]
GENE_RANGES = {
    "freq":(0.1,2.0),"coupling":(0.0,1.0),"excitability":(0.0,1.0),
    "decay":(0.01,0.5),"growth":(0.0,1.0),"hue":(0.0,360.0),
    "size":(2.0,8.0),"reach":(40.0,150.0),"plasticity":(0.0,0.5),
    "mutation":(0.01,0.3),
}
EPOCHS = [(0,"Void"),(10,"First Light"),(50,"Kindling"),(150,"The Weave"),
          (400,"Resonance"),(800,"Neural Spring"),(1500,"Synchrony"),
          (3000,"The Dreaming"),(6000,"Transcendence")]

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def uid(prefix="n"):
    h = hashlib.md5(f"{time.time_ns()}-{random.random()}".encode()).hexdigest()[:6]
    return f"{prefix}-{h}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def dist(a, b):
    dx, dy = a["x"] - b["x"], a["y"] - b["y"]
    return math.sqrt(dx*dx + dy*dy)

def species_hash(genome):
    q = (round(genome["freq"]*3)/3, round(genome["coupling"]*3)/3,
         round(genome["excitability"]*3)/3, round(genome["hue"]/60))
    return hashlib.md5(str(q).encode()).hexdigest()[:4]

def load_state():
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text())
            if data.get("_meta", {}).get("engine") == "phosphene":
                return data
        except Exception: pass
    return None

def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    state["_meta"]["last_tick"] = now_iso()
    for n in state["neurons"]:
        n.pop("_pp", None)
    for path in (STATE_PATH, VIZ_PATH):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, separators=(",", ":")))
        tmp.rename(path)

def random_genome():
    return {n: round(random.uniform(*GENE_RANGES[n]), 4) for n in GENE_NAMES}

def mutate_genome(parent, rate):
    child = {}
    for name in GENE_NAMES:
        lo, hi = GENE_RANGES[name]
        val = parent[name]
        if random.random() < rate:
            val = clamp(val + random.gauss(0, (hi-lo)*0.15), lo, hi)
        child[name] = round(val, 4)
    return child

def make_neuron(tick, genome=None, x=None, y=None, parent_id="", generation=0):
    g = genome or random_genome()
    return {
        "id": uid("n"),
        "x": x if x is not None else random.uniform(20, WORLD_W-20),
        "y": y if y is not None else random.uniform(20, WORLD_H-20),
        "vx": random.gauss(0, 0.3), "vy": random.gauss(0, 0.3),
        "phase": random.uniform(0, TWO_PI), "energy": 45.0,
        "genome": g, "conns": {},
        "age": 0, "generation": generation,
        "parent": parent_id, "species": species_hash(g),
        "fires": 0, "last_fire": -1,
    }

def genesis():
    neurons = [make_neuron(0) for _ in range(INIT_POP)]
    for i, n in enumerate(neurons):
        for j in range(i+1, len(neurons)):
            m = neurons[j]
            d = dist(n, m)
            if d < min(n["genome"]["reach"], m["genome"]["reach"]) and random.random() < 0.3:
                w = round(random.uniform(0.1, 0.4), 4)
                n["conns"][m["id"]] = w
                m["conns"][n["id"]] = w
    return {
        "_meta": {"created": now_iso(), "engine": "phosphene",
                  "version": "1.0", "last_tick": now_iso()},
        "tick": 0, "neurons": neurons, "history": [], "events": [],
        "clusters": [],
        "stats": {"total_births": INIT_POP, "total_deaths": 0,
                  "total_fires": 0, "total_avalanches": 0,
                  "max_pop": INIT_POP, "max_cluster": 0,
                  "largest_avalanche": 0},
    }

def tick(state):
    neurons = state["neurons"]
    if not neurons: return
    state["tick"] += 1
    t = state["tick"]
    stats, events = state["stats"], state["events"]
    by_id = {n["id"]: n for n in neurons}

    for n in neurons:
        g = n["genome"]
        omega = g["freq"] * TWO_PI
        delta, count = 0.0, 0
        for cid, w in list(n["conns"].items()):
            other = by_id.get(cid)
            if not other:
                del n["conns"][cid]; continue
            delta += w * g["coupling"] * math.sin(other["phase"] - n["phase"])
            count += 1
        if count > 0: delta /= count
        old_phase = n["phase"]
        n["phase"] = (n["phase"] + (omega + delta) * DT) % TWO_PI
        n["_pp"] = old_phase

    fired = []
    for n in neurons:
        old = n.get("_pp", n["phase"])
        new = n["phase"]
        wrapped = (old + n["genome"]["freq"] * TWO_PI * DT) > TWO_PI and new < old
        if wrapped and n["energy"] > 8:
            n["fires"] += 1
            n["last_fire"] = t
            n["energy"] -= 0.15
            fired.append(n)
            stats["total_fires"] += 1

    if len(fired) > len(neurons) * 0.12:
        stats["total_avalanches"] += 1
        stats["largest_avalanche"] = max(stats["largest_avalanche"], len(fired))
        if len(events) < EVENT_CAP:
            events.append({"t": t, "type": "avalanche", "size": len(fired)})

    for n in neurons:
        n["energy"] += 0.2
        sync, cc = 0.0, 0
        for cid, w in n["conns"].items():
            other = by_id.get(cid)
            if not other: continue
            dp = abs(n["phase"] - other["phase"])
            dp = min(dp, TWO_PI - dp)
            sync += (1.0 - dp / math.pi) * w
            cc += 1
        if cc > 0:
            sync /= cc
            n["energy"] += sync * SYNC_BONUS
        n["energy"] -= ENERGY_DRAIN * (0.5 + n["genome"]["decay"])
        n["age"] += 1

    for n in neurons:
        eta = n["genome"]["plasticity"]
        for cid in list(n["conns"].keys()):
            other = by_id.get(cid)
            if not other: continue
            dp = abs(n["phase"] - other["phase"])
            dp = min(dp, TWO_PI - dp)
            sv = math.cos(dp)
            w = clamp(n["conns"][cid] + eta * sv * 0.03, 0.0, 1.0)
            n["conns"][cid] = round(w, 4)
            if n["id"] in other["conns"]:
                other["conns"][n["id"]] = round(w, 4)

    for n in neurons:
        g = n["genome"]
        n["conns"] = {cid: w for cid, w in n["conns"].items()
                      if w >= WEAK_THRESHOLD and cid in by_id}
        if len(n["conns"]) < MAX_CONNS and random.random() < g["growth"] * 0.1:
            cands = [m for m in neurons if m["id"] != n["id"]
                     and m["id"] not in n["conns"] and dist(n,m) < g["reach"]]
            if cands:
                tgt = min(cands, key=lambda m: dist(n, m))
                w = round(random.uniform(0.05, 0.2), 4)
                n["conns"][tgt["id"]] = w
                tgt["conns"][n["id"]] = w

    for n in neurons:
        ax, ay = 0.0, 0.0
        for cid, w in n["conns"].items():
            other = by_id.get(cid)
            if other:
                dx, dy = other["x"]-n["x"], other["y"]-n["y"]
                d = math.sqrt(dx*dx + dy*dy) + 0.1
                ax += dx/d * w * 0.05
                ay += dy/d * w * 0.05
        drift = 0.3 + n["genome"]["freq"] * 0.2
        n["vx"] = n["vx"]*0.95 + ax + random.gauss(0, drift*0.1)
        n["vy"] = n["vy"]*0.95 + ay + random.gauss(0, drift*0.1)
        spd = math.sqrt(n["vx"]**2 + n["vy"]**2)
        if spd > drift: n["vx"] *= drift/spd; n["vy"] *= drift/spd
        n["x"] = clamp(n["x"]+n["vx"], 5, WORLD_W-5)
        n["y"] = clamp(n["y"]+n["vy"], 5, WORLD_H-5)

    new_neurons = []
    for n in neurons:
        if n["energy"] > REPRO_THRESHOLD and len(neurons)+len(new_neurons) < MAX_POP and n["age"] > 8:
            cg = mutate_genome(n["genome"], n["genome"]["mutation"])
            child = make_neuron(t, genome=cg, x=n["x"]+random.gauss(0,15),
                                y=n["y"]+random.gauss(0,15),
                                parent_id=n["id"], generation=n["generation"]+1)
            child["energy"] = REPRO_COST * 0.6
            n["energy"] -= REPRO_COST
            new_neurons.append(child)
            stats["total_births"] += 1
            if len(events) < EVENT_CAP:
                events.append({"t": t, "type": "birth", "id": child["id"]})
    neurons.extend(new_neurons)

    alive = []
    for n in neurons:
        max_age = MAX_AGE * (0.5 + n["genome"]["size"]/16)
        if n["energy"] <= DEATH_ENERGY or n["age"] > max_age:
            stats["total_deaths"] += 1
            if len(events) < EVENT_CAP:
                events.append({"t": t, "type": "death", "id": n["id"]})
            for other in neurons:
                other["conns"].pop(n["id"], None)
        else:
            alive.append(n)
    if len(alive) < MIN_POP:
        for _ in range(MIN_POP - len(alive)):
            n = make_neuron(t); n["energy"] = 55
            alive.append(n); stats["total_births"] += 1
    state["neurons"] = alive

    by_id_new = {n["id"]: n for n in alive}
    visited, clusters = set(), []
    for n in alive:
        if n["id"] in visited: continue
        cluster, queue = [], [n["id"]]
        while queue:
            nid = queue.pop(0)
            if nid in visited: continue
            visited.add(nid)
            node = by_id_new.get(nid)
            if not node: continue
            cluster.append(nid)
            for cid, w in node["conns"].items():
                if cid not in visited and w > 0.25: queue.append(cid)
        if len(cluster) >= 3:
            phases = [by_id_new[nid]["phase"] for nid in cluster if nid in by_id_new]
            if phases:
                cs = sum(math.cos(p) for p in phases)
                sn = sum(math.sin(p) for p in phases)
                r = math.sqrt(cs**2 + sn**2) / len(phases)
                ah = sum(by_id_new[nid]["genome"]["hue"] for nid in cluster if nid in by_id_new) / len(cluster)
                clusters.append({"neurons": cluster[:50], "size": len(cluster),
                                 "sync": round(r, 3), "hue": round(ah, 1)})
    state["clusters"] = sorted(clusters, key=lambda c: c["size"], reverse=True)[:20]
    stats["max_cluster"] = max(stats["max_cluster"], max((c["size"] for c in clusters), default=0))
    stats["max_pop"] = max(stats["max_pop"], len(alive))

    total_conns = sum(len(n["conns"]) for n in alive) // 2
    avg_sync = 0.0
    if clusters:
        tw = sum(c["size"] for c in clusters)
        if tw > 0: avg_sync = sum(c["sync"]*c["size"] for c in clusters) / tw
    sps = set(n["species"] for n in alive)
    state["history"].append({
        "t": t, "pop": len(alive), "conns": total_conns,
        "clusters": len(clusters), "sync": round(avg_sync, 3),
        "energy": round(sum(n["energy"] for n in alive) / max(len(alive), 1), 1),
        "species": len(sps), "fires": len(fired),
    })
    if len(state["history"]) > HISTORY_CAP:
        state["history"] = state["history"][-HISTORY_CAP:]
    if len(state["events"]) > EVENT_CAP:
        state["events"] = state["events"][-EVENT_CAP:]

def main():
    parser = argparse.ArgumentParser(description="Phosphene neural ecosystem")
    parser.add_argument("--ticks", type=int, default=1)
    parser.add_argument("--genesis", action="store_true")
    args = parser.parse_args()
    if args.genesis:
        state = genesis(); save_state(state)
        print(f"Genesis: {len(state['neurons'])} neurons"); return
    state = load_state()
    if state is None:
        print("No state found, running genesis..."); state = genesis()
    for _ in range(args.ticks):
        tick(state)
        pop = len(state["neurons"])
        conns = sum(len(n["conns"]) for n in state["neurons"]) // 2
        cls = len(state["clusters"])
        fires = state["history"][-1]["fires"] if state["history"] else 0
        print(f"Tick {state['tick']}: {pop} neurons, {conns} conns, {cls} clusters, {fires} fires")
    save_state(state)
    print(f"Saved to {STATE_PATH} and {VIZ_PATH}")

if __name__ == "__main__":
    main()
