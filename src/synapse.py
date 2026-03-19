#!/usr/bin/env python3
"""
The Synapse - a living neural network that evolves.
One run = one tick. Neurons fire, synapses strengthen/weaken (Hebbian learning),
the network grows new connections, prunes dead ones, and the whole topology
evolves through natural selection. Python stdlib only.
"""
from __future__ import annotations
import hashlib, json, math, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "docs")))
STATE_PATH = STATE_DIR / "synapse_state.json"

WORLD_W, WORLD_H = 1000, 1000
INITIAL_NEURONS = 120
MAX_NEURONS = 400
MIN_NEURONS = 30
MAX_SYNAPSES_PER = 12
SYNAPSE_RANGE = 180.0
STEPS_PER_TICK = 80
SIGNAL_SPEED = 3.0
HISTORY_CAP = 500
EVENT_CAP = 200

G_THRESHOLD = 0
G_DECAY = 1
G_FIRE_STRENGTH = 2
G_PLASTICITY = 3
G_GROWTH = 4
G_PRUNE = 5
G_REFRACTORY = 6
G_MUTATION = 7
G_EXCITABILITY = 8
G_INHIBITION = 9
G_RESONANCE = 10
G_ADAPTATION = 11
GENE_COUNT = 12

GENE_NAMES = [
    "threshold", "decay", "fire_strength", "plasticity",
    "growth", "prune", "refractory", "mutation",
    "excitability", "inhibition", "resonance", "adaptation",
]

EPOCHS = [
    (0, "Silent Void"), (20, "First Spark"), (100, "Kindling"),
    (300, "Chain Lightning"), (700, "Neural Dawn"), (1500, "The Dreaming"),
    (3000, "Deep Resonance"), (6000, "Transcendence"),
]


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def uid(prefix="n"):
    h = hashlib.sha256(f"{time.time_ns()}{random.random()}".encode()).hexdigest()[:6]
    return f"{prefix}-{h}"

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def dist(a, b):
    dx, dy = a["x"] - b["x"], a["y"] - b["y"]
    return math.sqrt(dx*dx + dy*dy)

def epoch_name(tick):
    name = EPOCHS[0][1]
    for threshold, label in EPOCHS:
        if tick >= threshold:
            name = label
    return name

def gene_val(genome, idx, lo, hi):
    return lo + genome[idx] * (hi - lo)


def make_neuron(tick, genome=None, parent_id="", generation=0, x=None, y=None):
    g = genome or [random.random() for _ in range(GENE_COUNT)]
    return {
        "id": uid("n"), "x": x if x is not None else random.uniform(50, WORLD_W-50),
        "y": y if y is not None else random.uniform(50, WORLD_H-50),
        "genome": [round(v, 4) for v in g], "potential": random.uniform(0, 0.3),
        "energy": 100.0, "fired": False, "fire_count": 0, "last_fire": -100,
        "age": 0, "generation": generation, "born_tick": tick,
        "parent_id": parent_id, "cluster_id": "",
    }

def make_synapse(src_id, dst_id, weight=0.5):
    return {"src": src_id, "dst": dst_id, "weight": round(clamp(weight, 0.01, 2.0), 4),
            "age": 0, "signal": 0.0, "last_active": 0}


def genesis():
    tick = 0
    neurons = []
    num_clusters = random.randint(5, 9)
    centers = [(random.uniform(150, WORLD_W-150), random.uniform(150, WORLD_H-150))
               for _ in range(num_clusters)]
    for i in range(INITIAL_NEURONS):
        cx, cy = random.choice(centers)
        x = clamp(cx + random.gauss(0, 60), 20, WORLD_W-20)
        y = clamp(cy + random.gauss(0, 60), 20, WORLD_H-20)
        genome = [random.random() for _ in range(GENE_COUNT)]
        genome[G_RESONANCE] = clamp(cx / WORLD_W + random.gauss(0, 0.1))
        genome[G_INHIBITION] = clamp(cy / WORLD_H * 0.5 + random.gauss(0, 0.1))
        neurons.append(make_neuron(tick, genome, x=x, y=y))
    synapses = _grow_synapses(neurons, [], tick)
    _assign_clusters(neurons, synapses)
    return {
        "tick": tick, "step": 0, "epoch": epoch_name(tick),
        "neurons": neurons, "synapses": synapses,
        "signals": [], "history": [], "events": [],
        "stats": {"total_fires": 0, "total_cascades": 0, "total_births": 0,
                  "total_deaths": 0, "peak_neurons": INITIAL_NEURONS,
                  "peak_synapses": len(synapses), "longest_cascade": 0, "total_ticks": 0},
        "dream": {"pattern": "void", "intensity": 0.0, "coherence": 0.0, "dominant_freq": 0.0},
        "updated_at": now_iso(),
    }


def _grow_synapses(neurons, existing, tick):
    synapse_set = {(s["src"], s["dst"]) for s in existing}
    out_count = {}
    for s in existing:
        out_count[s["src"]] = out_count.get(s["src"], 0) + 1
    new_synapses = list(existing)
    for n in neurons:
        if out_count.get(n["id"], 0) >= MAX_SYNAPSES_PER:
            continue
        if random.random() > gene_val(n["genome"], G_GROWTH, 0.02, 0.3):
            continue
        candidates = []
        for other in neurons:
            if other["id"] == n["id"]:
                continue
            d = dist(n, other)
            if d < SYNAPSE_RANGE and (n["id"], other["id"]) not in synapse_set:
                candidates.append((d, other))
        candidates.sort(key=lambda t: t[0])
        for d, other in candidates[:2]:
            if out_count.get(n["id"], 0) >= MAX_SYNAPSES_PER:
                break
            new_synapses.append(make_synapse(n["id"], other["id"], 0.3 + random.random() * 0.4))
            synapse_set.add((n["id"], other["id"]))
            out_count[n["id"]] = out_count.get(n["id"], 0) + 1
    return new_synapses


def _assign_clusters(neurons, synapses):
    nmap = {n["id"]: n for n in neurons}
    adj = {n["id"]: [] for n in neurons}
    for s in synapses:
        if s["src"] in adj and s["dst"] in adj:
            adj[s["src"]].append(s["dst"])
            adj[s["dst"]].append(s["src"])
    visited = set()
    cluster_id = 0
    for n in neurons:
        if n["id"] in visited:
            continue
        queue = [n["id"]]
        members = []
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            members.append(nid)
            for neighbor in adj.get(nid, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        cid = f"c-{cluster_id}"
        for mid in members:
            if mid in nmap:
                nmap[mid]["cluster_id"] = cid
        cluster_id += 1


def sim_step(world):
    neurons, synapses, signals = world["neurons"], world["synapses"], world["signals"]
    stats, step = world["stats"], world["step"]
    events = []
    nmap = {n["id"]: n for n in neurons}

    new_signals = []
    for sig in signals:
        sig["progress"] += SIGNAL_SPEED / max(sig["length"], 1)
        if sig["progress"] >= 1.0:
            dst = nmap.get(sig["dst"])
            if dst and not dst["fired"]:
                if sig.get("inhibitory", False):
                    dst["potential"] = max(0, dst["potential"] - sig["strength"] * 0.5)
                else:
                    dst["potential"] += sig["strength"]
        else:
            new_signals.append(sig)
    world["signals"] = new_signals

    fired_ids = []
    for n in neurons:
        threshold = gene_val(n["genome"], G_THRESHOLD, 0.2, 0.9)
        refractory = int(gene_val(n["genome"], G_REFRACTORY, 2, 15))
        excitability = gene_val(n["genome"], G_EXCITABILITY, 0.001, 0.05)
        if not n["fired"] and random.random() < excitability:
            n["potential"] += 0.3
        if n["potential"] >= threshold and (step - n["last_fire"]) > refractory:
            n["fired"] = True
            n["fire_count"] += 1
            n["last_fire"] = step
            n["energy"] += 5.0
            fired_ids.append(n["id"])
            stats["total_fires"] = stats.get("total_fires", 0) + 1
            fire_str = gene_val(n["genome"], G_FIRE_STRENGTH, 0.1, 0.8)
            inhib_frac = gene_val(n["genome"], G_INHIBITION, 0.0, 0.5)
            for syn in synapses:
                if syn["src"] == n["id"]:
                    dst_n = nmap.get(syn["dst"])
                    if dst_n:
                        length = dist(n, dst_n)
                        world["signals"].append({
                            "src": n["id"], "dst": syn["dst"],
                            "strength": round(fire_str * syn["weight"], 3),
                            "progress": 0.0, "length": round(length, 1),
                            "inhibitory": random.random() < inhib_frac,
                        })
                        syn["last_active"] = step
            n["potential"] = 0.0
        else:
            n["fired"] = False

    if len(fired_ids) > 5:
        stats["total_cascades"] = stats.get("total_cascades", 0) + 1
        cs = len(fired_ids)
        if cs > stats.get("longest_cascade", 0):
            stats["longest_cascade"] = cs
        events.append({"type": "cascade", "size": cs, "step": step, "tick": world["tick"]})

    for n in neurons:
        decay = gene_val(n["genome"], G_DECAY, 0.02, 0.15)
        n["potential"] = max(0, n["potential"] - decay)
        adapt = gene_val(n["genome"], G_ADAPTATION, 0.001, 0.02)
        if n["fire_count"] > 0 and (step - n["last_fire"]) < 10:
            n["genome"][G_THRESHOLD] = clamp(n["genome"][G_THRESHOLD] + adapt * 0.5)
        else:
            n["genome"][G_THRESHOLD] = clamp(n["genome"][G_THRESHOLD] - adapt * 0.2)

    for syn in synapses:
        src_n, dst_n = nmap.get(syn["src"]), nmap.get(syn["dst"])
        if not src_n or not dst_n:
            continue
        plasticity = gene_val(src_n["genome"], G_PLASTICITY, 0.005, 0.05)
        if syn["last_active"] == step:
            syn["weight"] = min(2.0, syn["weight"] + plasticity)
        else:
            prune_rate = gene_val(src_n["genome"], G_PRUNE, 0.001, 0.01)
            syn["weight"] = max(0.01, syn["weight"] - prune_rate)
        syn["age"] += 1

    for n in neurons:
        metabolism = 0.15 + n["genome"][G_FIRE_STRENGTH] * 0.2
        n["energy"] -= metabolism * 0.1
        n["age"] += 1

    world["step"] = step + 1
    return events


def evolve(world):
    neurons, synapses = world["neurons"], world["synapses"]
    stats, tick = world["stats"], world["tick"]
    events = []

    alive, dead_ids = [], set()
    for n in neurons:
        max_age = 400 + n["genome"][G_REFRACTORY] * 600
        if n["energy"] <= 0 or n["age"] > max_age:
            dead_ids.add(n["id"])
            stats["total_deaths"] = stats.get("total_deaths", 0) + 1
            events.append({"type": "death", "neuron": n["id"], "age": n["age"],
                          "fires": n["fire_count"], "tick": tick})
        else:
            alive.append(n)

    if dead_ids:
        synapses = [s for s in synapses if s["src"] not in dead_ids and s["dst"] not in dead_ids]
    synapses = [s for s in synapses if s["weight"] > 0.05 or s["age"] < 50]

    new_neurons = []
    for n in alive:
        if n["energy"] > 70 and len(alive) + len(new_neurons) < MAX_NEURONS:
            if random.random() < 0.35:
                mut_rate = gene_val(n["genome"], G_MUTATION, 0.02, 0.2)
                child_genome = [clamp(g + random.gauss(0, 0.15)) if random.random() < mut_rate else g
                                for g in n["genome"]]
                cx = clamp(n["x"] + random.gauss(0, 40), 20, WORLD_W-20)
                cy = clamp(n["y"] + random.gauss(0, 40), 20, WORLD_H-20)
                child = make_neuron(tick, child_genome, parent_id=n["id"],
                                    generation=n["generation"]+1, x=cx, y=cy)
                new_neurons.append(child)
                n["energy"] -= 20
                stats["total_births"] = stats.get("total_births", 0) + 1
                events.append({"type": "birth", "neuron": child["id"],
                              "parent": n["id"], "generation": child["generation"], "tick": tick})
    alive.extend(new_neurons)

    while len(alive) < MIN_NEURONS:
        alive.append(make_neuron(tick))
        stats["total_births"] = stats.get("total_births", 0) + 1
        events.append({"type": "spawn", "tick": tick})

    synapses = _grow_synapses(alive, synapses, tick)
    _assign_clusters(alive, synapses)

    stats["peak_neurons"] = max(stats.get("peak_neurons", 0), len(alive))
    stats["peak_synapses"] = max(stats.get("peak_synapses", 0), len(synapses))
    stats["total_ticks"] = stats.get("total_ticks", 0) + 1

    total_fire_rate = sum(n["fire_count"] for n in alive) / max(len(alive), 1)
    cluster_ids = set(n["cluster_id"] for n in alive if n["cluster_id"])
    resonances = [n["genome"][G_RESONANCE] for n in alive if n["fire_count"] > 0]
    dom_freq = sum(resonances) / max(len(resonances), 1) if resonances else 0

    patterns = ["void", "flicker", "pulse", "wave", "spiral", "bloom", "storm", "dream"]
    pattern_idx = min(int(total_fire_rate / 2), len(patterns) - 1)
    world["dream"] = {
        "pattern": patterns[pattern_idx],
        "intensity": round(clamp(total_fire_rate / 10), 3),
        "coherence": round(clamp(1.0 - len(cluster_ids) / max(len(alive) * 0.5, 1)), 3),
        "dominant_freq": round(dom_freq, 3),
    }
    world["neurons"] = alive
    world["synapses"] = synapses
    return events


def snapshot(world):
    neurons = world["neurons"]
    cluster_ids = set(n["cluster_id"] for n in neurons if n["cluster_id"])
    active = sum(1 for n in neurons if n["fire_count"] > 0)
    return {"tick": world["tick"], "neurons": len(neurons), "synapses": len(world["synapses"]),
            "clusters": len(cluster_ids), "active": active,
            "signals": len(world["signals"]), "dream": world["dream"]["pattern"]}


def load_world():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return None
    return None


def save_world(world):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    world["signals"] = world["signals"][:200]
    if len(world["history"]) > HISTORY_CAP:
        world["history"] = world["history"][-HISTORY_CAP:]
    if len(world["events"]) > EVENT_CAP:
        world["events"] = world["events"][-EVENT_CAP:]
    tmp = STATE_PATH.with_suffix(".tmp")
    data = json.dumps(world, separators=(",", ":"))
    tmp.write_text(data)
    tmp.replace(STATE_PATH)
    print(f"  Saved {len(data)} bytes -> {STATE_PATH}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="The Synapse - neural evolution engine")
    parser.add_argument("--ticks", type=int, default=1)
    parser.add_argument("--steps", type=int, default=STEPS_PER_TICK)
    parser.add_argument("--genesis", action="store_true")
    args = parser.parse_args()

    world = None if args.genesis else load_world()
    if world is None:
        print("Genesis - creating neural network...")
        world = genesis()
        save_world(world)

    for t in range(args.ticks):
        world["tick"] += 1
        world["epoch"] = epoch_name(world["tick"])
        all_events = []
        for s in range(args.steps):
            all_events.extend(sim_step(world))
        all_events.extend(evolve(world))
        world["history"].append(snapshot(world))
        world["events"].extend(all_events)
        world["updated_at"] = now_iso()
        nc = len(world["neurons"])
        sc = len(world["synapses"])
        sgc = len(world["signals"])
        fires = world["stats"].get("total_fires", 0)
        dream = world["dream"]["pattern"]
        epoch = world["epoch"]
        tick = world["tick"]
        print(f"  tick {tick:>4} | {epoch:>16} | neurons={nc:>3} synapses={sc:>4} signals={sgc:>3} fires={fires:>6} dream={dream}")

    save_world(world)
    tick_val = world["tick"]
    print(f"Done: {args.ticks} tick(s). Generation {tick_val}.")


if __name__ == "__main__":
    main()
