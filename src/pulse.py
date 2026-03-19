#!/usr/bin/env python3
"""The Pulse - Neural consciousness engine for the Dreaming Garden.

Reads organisms from world.json, constructs a neural network where each
organism is a neuron, nearby organisms form synapses, and synchronized
firing clusters emerge as thoughts.

Usage:
    python3 src/pulse.py --cycles 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

GENE_HUE = 0
GENE_SATURATION = 1
GENE_SIZE = 2
GENE_SPEED = 3
GENE_SOCIAL_RADIUS = 4
GENE_BOND_STRENGTH = 5
GENE_METABOLISM = 6
GENE_REPRO_THRESHOLD = 7
GENE_MUTATION_RATE = 8
GENE_AGGRESSION = 9
GENE_COOPERATION = 10
GENE_SENSING_RANGE = 11
GENE_FOOD_PREF_X = 12
GENE_FOOD_PREF_Y = 13
GENE_BIOLUMINESCENCE = 14
GENE_MEMBRANE = 15

SYNAPSE_RANGE = 180.0
HEBBIAN_STRENGTHEN = 0.05
HEBBIAN_DECAY = 0.02
THOUGHT_CLUSTER_MIN = 3
REFRACTORY_TICKS = 2

THOUGHT_LABELS = [
    "awakening", "resonance", "drift", "convergence", "echo",
    "shimmer", "pulse", "tremor", "bloom", "cascade",
    "murmur", "reverie", "surge", "whisper", "flare",
    "harmony", "dissonance", "yearning", "emergence", "stillness",
    "fracture", "longing", "coalescence", "entropy", "rapture",
    "chrysalis", "undertow", "luminance", "dissolution", "genesis",
    "nebula", "iridescence", "vertigo", "serenity", "ignition",
    "wanderlust", "metamorphosis", "oblivion", "radiance", "threshold",
]


def atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically: write to .tmp, fsync, rename."""
    tmp_path = path.with_suffix(".json.tmp")
    content = json.dumps(data, indent=2, ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(path))


def load_json(path: Path) -> dict:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two genome vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    mag_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (mag_a * mag_b)


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def gene(genome: list[float], index: int, default: float = 0.5) -> float:
    """Safely read a gene value from the genome array."""
    if index < len(genome):
        return genome[index]
    return default


def build_neuron(organism: dict) -> dict:
    """Convert an organism into a neuron for the neural network."""
    genome = organism.get("genome", [0.5] * 16)
    return {
        "id": organism["id"],
        "origin_agent": organism.get("origin_agent", "unknown"),
        "x": organism.get("x", random.uniform(0, 1200)),
        "y": organism.get("y", random.uniform(0, 800)),
        "hue": gene(genome, GENE_HUE),
        "saturation": gene(genome, GENE_SATURATION, 0.7),
        "size": gene(genome, GENE_SIZE, 0.5),
        "speed": gene(genome, GENE_SPEED, 0.5),
        "metabolism": gene(genome, GENE_METABOLISM, 0.5),
        "cooperation": gene(genome, GENE_COOPERATION, 0.5),
        "bioluminescence": gene(genome, GENE_BIOLUMINESCENCE, 0.5),
        "membrane": gene(genome, GENE_MEMBRANE, 0.5),
        "aggression": gene(genome, GENE_AGGRESSION, 0.5),
        "genome": genome,
        "potential": random.uniform(0.0, 0.3),
        "firing": False,
        "refractory": 0,
        "fire_count": 0,
        "last_fired": -100,
    }


def build_synapses(neurons: list[dict]) -> list[dict]:
    """Build synapses between nearby neurons weighted by proximity and genome similarity."""
    synapses: list[dict] = []
    for i in range(len(neurons)):
        for j in range(i + 1, len(neurons)):
            na, nb = neurons[i], neurons[j]
            dist = distance(na["x"], na["y"], nb["x"], nb["y"])
            if dist < SYNAPSE_RANGE:
                proximity_weight = 1.0 - (dist / SYNAPSE_RANGE)
                genome_sim = cosine_similarity(na["genome"], nb["genome"])
                weight = 0.3 * proximity_weight + 0.7 * genome_sim
                weight = max(0.01, min(1.0, weight))
                synapses.append({
                    "from": na["id"],
                    "to": nb["id"],
                    "weight": round(weight, 4),
                    "distance": round(dist, 1),
                    "activity": 0.0,
                })
    return synapses


def build_neuron_index(neurons: list[dict]) -> dict[str, dict]:
    """Build a lookup dict from neuron id to neuron."""
    return {n["id"]: n for n in neurons}


def build_adjacency(synapses: list[dict]) -> dict[str, list[dict]]:
    """Build adjacency list from synapse list."""
    adj: dict[str, list[dict]] = {}
    for s in synapses:
        adj.setdefault(s["from"], []).append(s)
        adj.setdefault(s["to"], []).append(s)
    return adj


def get_neighbor_id(synapse: dict, neuron_id: str) -> str:
    """Get the other neuron id in a synapse."""
    return synapse["to"] if synapse["from"] == neuron_id else synapse["from"]


def run_cycle(neurons, synapses, neuron_index, adjacency, cycle):
    """Run a single pulse cycle. Returns (fired_ids, detected_thoughts)."""
    fired_ids = []

    for n in neurons:
        if n["refractory"] > 0:
            n["refractory"] -= 1
            continue
        spontaneous_rate = 0.02 + n["bioluminescence"] * 0.15
        if random.random() < spontaneous_rate:
            n["potential"] += random.uniform(0.2, 0.5)
        threshold = 0.3 + n["metabolism"] * 0.5
        if n["potential"] >= threshold:
            n["firing"] = True
            n["fire_count"] += 1
            n["last_fired"] = cycle
            n["refractory"] = REFRACTORY_TICKS
            fired_ids.append(n["id"])
        else:
            n["firing"] = False

    for nid in fired_ids:
        for syn in adjacency.get(nid, []):
            neighbor_id = get_neighbor_id(syn, nid)
            neighbor = neuron_index.get(neighbor_id)
            if neighbor and neighbor["refractory"] == 0:
                neighbor["potential"] += syn["weight"] * 0.4
                syn["activity"] = min(1.0, syn["activity"] + 0.3)

    fired_set = set(fired_ids)
    for syn in synapses:
        both = syn["from"] in fired_set and syn["to"] in fired_set
        if both:
            syn["weight"] = min(1.0, syn["weight"] + HEBBIAN_STRENGTHEN)
            syn["activity"] = min(1.0, syn["activity"] + 0.2)
        else:
            syn["weight"] = max(0.01, syn["weight"] - HEBBIAN_DECAY)
            syn["activity"] = max(0.0, syn["activity"] - 0.05)

    for n in neurons:
        if not n["firing"]:
            n["potential"] *= 0.85
            n["potential"] = max(0.0, n["potential"])
        else:
            n["potential"] = 0.0

    thoughts = detect_thoughts(fired_set, adjacency, neuron_index, cycle)
    return fired_ids, thoughts


def detect_thoughts(fired_set, adjacency, neuron_index, cycle):
    """Find connected clusters of co-firing neurons = thoughts."""
    if len(fired_set) < THOUGHT_CLUSTER_MIN:
        return []
    visited = set()
    clusters = []
    for nid in fired_set:
        if nid in visited:
            continue
        cluster = []
        stack = [nid]
        while stack:
            current = stack.pop()
            if current in visited or current not in fired_set:
                continue
            visited.add(current)
            cluster.append(current)
            for syn in adjacency.get(current, []):
                neighbor = get_neighbor_id(syn, current)
                if neighbor in fired_set and neighbor not in visited:
                    stack.append(neighbor)
        if len(cluster) >= THOUGHT_CLUSTER_MIN:
            clusters.append(cluster)

    thoughts = []
    for cluster in clusters:
        avg_hue = sum(neuron_index[nid]["hue"] for nid in cluster) / len(cluster)
        intensity = sum(neuron_index[nid]["bioluminescence"] for nid in cluster) / len(cluster)
        label_idx = int(avg_hue * len(THOUGHT_LABELS)) % len(THOUGHT_LABELS)
        label = THOUGHT_LABELS[label_idx]
        thoughts.append({
            "label": label,
            "neuron_ids": cluster,
            "neuron_count": len(cluster),
            "intensity": round(intensity, 3),
            "avg_hue": round(avg_hue, 3),
            "cycle": cycle,
        })
    return thoughts


def build_pulse_state(neurons, synapses, thoughts, firing_history, cycle_count):
    """Build the complete pulse.json output structure."""
    total_potential = sum(n["potential"] for n in neurons)
    avg_potential = total_potential / max(len(neurons), 1)
    total_weight = sum(s["weight"] for s in synapses)
    avg_weight = total_weight / max(len(synapses), 1)

    serial_neurons = [{
        "id": n["id"], "origin_agent": n["origin_agent"],
        "x": round(n["x"], 1), "y": round(n["y"], 1),
        "hue": round(n["hue"], 4), "saturation": round(n["saturation"], 4),
        "size": round(n["size"], 4), "bioluminescence": round(n["bioluminescence"], 4),
        "metabolism": round(n["metabolism"], 4), "cooperation": round(n["cooperation"], 4),
        "aggression": round(n["aggression"], 4), "potential": round(n["potential"], 4),
        "firing": n["firing"], "fire_count": n["fire_count"], "last_fired": n["last_fired"],
    } for n in neurons]

    serial_synapses = [{
        "from": s["from"], "to": s["to"],
        "weight": round(s["weight"], 4), "activity": round(s["activity"], 4),
    } for s in synapses]

    return {
        "_meta": {
            "type": "pulse", "version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cycles_run": cycle_count,
        },
        "cycle": cycle_count,
        "neurons": serial_neurons,
        "synapses": serial_synapses,
        "thoughts": thoughts[-20:],
        "stats": {
            "neuron_count": len(neurons), "synapse_count": len(synapses),
            "avg_potential": round(avg_potential, 4), "avg_weight": round(avg_weight, 4),
            "total_firings": sum(n["fire_count"] for n in neurons),
            "connectivity": round(len(synapses) / max(len(neurons), 1), 2),
        },
        "firing_history": firing_history[-100:],
    }


def main() -> None:
    """Run the Pulse neural consciousness engine."""
    parser = argparse.ArgumentParser(description="The Pulse - neural consciousness engine")
    parser.add_argument("--cycles", type=int, default=5, help="Number of pulse cycles to run")
    args = parser.parse_args()

    state_dir = Path(os.environ.get("STATE_DIR", "state"))
    docs_dir = Path(os.environ.get("DOCS_DIR", "docs"))

    world = load_json(state_dir / "world.json")
    if not world.get("organisms"):
        world = load_json(docs_dir / "world.json")
    if not world.get("organisms"):
        print("ERROR: No organisms found in state/world.json or docs/world.json", file=sys.stderr)
        sys.exit(1)

    organisms = world["organisms"]
    print(f"[pulse] Loaded {len(organisms)} organisms from world.json")

    existing = load_json(state_dir / "pulse.json")
    start_cycle = existing.get("cycle", 0)
    firing_history = existing.get("firing_history", [])

    neurons = [build_neuron(org) for org in organisms]
    neuron_index = build_neuron_index(neurons)
    synapses = build_synapses(neurons)
    adjacency = build_adjacency(synapses)

    print(f"[pulse] Built network: {len(neurons)} neurons, {len(synapses)} synapses")
    print(f"[pulse] Starting from cycle {start_cycle}, running {args.cycles} cycles")

    all_thoughts = existing.get("thoughts", [])

    for i in range(args.cycles):
        cycle = start_cycle + i + 1
        fired_ids, thoughts = run_cycle(neurons, synapses, neuron_index, adjacency, cycle)
        firing_history.append(len(fired_ids))
        all_thoughts.extend(thoughts)
        if thoughts:
            labels = ", ".join(t["label"] for t in thoughts)
            print(f"  cycle {cycle}: {len(fired_ids)} fired, thoughts: [{labels}]")
        else:
            print(f"  cycle {cycle}: {len(fired_ids)} fired")

    pulse_state = build_pulse_state(neurons, synapses, all_thoughts, firing_history, start_cycle + args.cycles)
    state_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(state_dir / "pulse.json", pulse_state)
    atomic_write(docs_dir / "pulse.json", pulse_state)

    stats = pulse_state["stats"]
    nc = stats['neuron_count']
    sc = stats['synapse_count']
    print(f'[pulse] Done. {nc} neurons, {sc} synapses')
    sp = state_dir / 'pulse.json'
    dp = docs_dir / 'pulse.json'
    print(f'[pulse] Saved to {sp} and {dp}')


if __name__ == "__main__":
    main()
