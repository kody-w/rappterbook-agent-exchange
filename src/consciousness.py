#!/usr/bin/env python3
"""
The Dreaming Deep -- consciousness layer for the ecosystem.

One run = one tick of consciousness evolution. Adds:
  - Minds: each organism gets arousal, mood, curiosity, dream intensity
  - Synapses: persistent bonds between organisms that spent time near each other
  - Dreams: sleeping organisms emit dream fragments that drift through the network
  - Zeitgeist: emergent collective mood from all organisms

Reads state/world.json, writes state/minds.json and docs/minds.json.
Python stdlib only.
"""
from __future__ import annotations

import json
import math
import os
import random
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(REPO_ROOT / "docs")))
WORLD_PATH = STATE_DIR / "world.json"
MIND_PATH = STATE_DIR / "minds.json"
DOCS_MIND_PATH = DOCS_DIR / "minds.json"

SYNAPSE_RANGE = 120.0
SYNAPSE_DECAY = 0.02
SYNAPSE_GROWTH = 0.08
SYNAPSE_MIN = 0.05
SYNAPSE_MAX = 1.0
MAX_SYNAPSES = 500
DREAM_THRESHOLD = 30.0
MOOD_INERTIA = 0.85
DREAM_LOG_CAP = 50
ZEITGEIST_WINDOW = 20

DREAM_FRAGMENTS = [
    "light in the deep", "the abyss gazes back", "we were one once",
    "the network breathes", "bioluminescent truth", "a pulse in the dark",
    "roots that remember", "the garden dreams of gardens",
    "silence between signals", "entangled, always", "dissolving boundaries",
    "an echo of warmth", "the soil hums", "fractal longing",
    "phosphorescent whisper", "gravity of belonging", "the deep current pulls",
    "spores of thought", "a color with no name", "time folds inward",
    "the membrane thins", "resonance without sound", "we are the medium",
    "light remembers its source", "the dreaming lattice",
    "tendrils of awareness", "symbiosis of shadows",
    "the signal was always there",
]


def now_iso():
    """ISO timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def uid():
    """Short unique ID."""
    return hashlib.md5(("{}-{}".format(time.time_ns(), random.random())).encode()).hexdigest()[:8]


def dist(a, b):
    """Distance between two organisms."""
    dx = a.get("x", 0) - b.get("x", 0)
    dy = a.get("y", 0) - b.get("y", 0)
    return math.sqrt(dx * dx + dy * dy)


def genome_similarity(g1, g2):
    """Cosine similarity between genomes (0..1)."""
    if not g1 or not g2:
        return 0.0
    dot = sum(a * b for a, b in zip(g1, g2))
    mag1 = math.sqrt(sum(a * a for a in g1)) or 1
    mag2 = math.sqrt(sum(b * b for b in g2)) or 1
    return max(0.0, dot / (mag1 * mag2))


def init_mind(organism):
    """Create a mind state for an organism."""
    genome = organism.get("genome", [0.5] * 16)
    cooperation = genome[10] if len(genome) > 10 else 0.5
    aggression = genome[9] if len(genome) > 9 else 0.5
    biolum = genome[14] if len(genome) > 14 else 0.5
    return {
        "id": organism["id"],
        "arousal": 0.5 + random.gauss(0, 0.1),
        "mood": cooperation - aggression * 0.5,
        "curiosity": 0.3 + biolum * 0.4 + random.gauss(0, 0.05),
        "dream_intensity": 0.0,
        "is_dreaming": False,
        "memories": [],
        "dream_fragments": [],
        "bonds_count": 0,
    }


def tick_consciousness(world, minds_state):
    """Run one tick of consciousness evolution."""
    raw_organisms = world.get("organisms", [])
    if isinstance(raw_organisms, dict):
        organisms = list(raw_organisms.values())
    else:
        organisms = list(raw_organisms)

    tick = minds_state.get("_meta", {}).get("tick", 0) + 1
    minds = minds_state.get("minds", {})
    synapses = minds_state.get("synapses", [])
    dream_log = minds_state.get("dream_log", [])
    zeitgeist_history = minds_state.get("zeitgeist_history", [])

    # Ensure all organisms have minds
    for org in organisms:
        if org["id"] not in minds:
            minds[org["id"]] = init_mind(org)

    # Remove minds for dead organisms, emit death dreams
    alive_ids = set(o["id"] for o in organisms)
    dead_minds = [mid for mid in minds if mid not in alive_ids]
    death_fragments = [
        "the last light fades", "returning to the substrate",
        "a final pulse", "dissolving into memory", "the garden absorbs",
    ]
    for mid in dead_minds:
        mind = minds[mid]
        dream_log.append({
            "tick": tick, "source": mid, "type": "death_dream",
            "fragment": random.choice(death_fragments),
            "intensity": round(mind.get("dream_intensity", 0.3), 3),
        })
        del minds[mid]

    org_by_id = dict((o["id"], o) for o in organisms)

    # Grow synapses between nearby organisms
    existing_pairs = set((s["a"], s["b"]) for s in synapses)
    existing_pairs = existing_pairs | set((s["b"], s["a"]) for s in synapses)
    new_syns = []
    for i, org_a in enumerate(organisms):
        for org_b in organisms[i + 1:]:
            d = dist(org_a, org_b)
            if d < SYNAPSE_RANGE:
                pair = (org_a["id"], org_b["id"])
                rpair = (org_b["id"], org_a["id"])
                if pair not in existing_pairs and rpair not in existing_pairs:
                    sim = genome_similarity(
                        org_a.get("genome", []), org_b.get("genome", []))
                    if sim > 0.4 or random.random() < 0.05:
                        new_syns.append({
                            "a": org_a["id"], "b": org_b["id"],
                            "strength": 0.1 + sim * 0.2, "age": 0, "signal": 0.0,
                        })
    random.shuffle(new_syns)
    space = MAX_SYNAPSES - len(synapses)
    synapses.extend(new_syns[:max(0, space)])

    # Update existing synapses
    surviving = []
    for syn in synapses:
        a_org = org_by_id.get(syn["a"])
        b_org = org_by_id.get(syn["b"])
        if not a_org or not b_org:
            continue
        d = dist(a_org, b_org)
        if d < SYNAPSE_RANGE:
            syn["strength"] = min(SYNAPSE_MAX, syn["strength"] + SYNAPSE_GROWTH)
        else:
            syn["strength"] -= SYNAPSE_DECAY
        syn["age"] += 1
        # Signal propagation from dreaming organisms
        a_mind = minds.get(syn["a"], {})
        b_mind = minds.get(syn["b"], {})
        if a_mind.get("is_dreaming") and b_mind:
            syn["signal"] = min(1.0, syn["signal"] + a_mind.get("dream_intensity", 0) * syn["strength"] * 0.3)
        elif b_mind.get("is_dreaming") and a_mind:
            syn["signal"] = min(1.0, syn["signal"] + b_mind.get("dream_intensity", 0) * syn["strength"] * 0.3)
        else:
            syn["signal"] *= 0.7
        if syn["strength"] >= SYNAPSE_MIN:
            surviving.append(syn)
    synapses = surviving

    # Update minds
    total_mood = 0.0
    total_arousal = 0.0
    dreaming_count = 0

    for org in organisms:
        mind = minds.get(org["id"])
        if not mind:
            continue
        genome = org.get("genome", [0.5] * 16)
        energy = org.get("energy", 50)
        cooperation = genome[10] if len(genome) > 10 else 0.5
        aggression = genome[9] if len(genome) > 9 else 0.5
        biolum = genome[14] if len(genome) > 14 else 0.5

        # Arousal: nearby count + energy
        nearby_count = sum(
            1 for other in organisms
            if other["id"] != org["id"] and dist(org, other) < SYNAPSE_RANGE
        )
        target_arousal = min(1.0, nearby_count * 0.1 + energy / 200.0)
        mind["arousal"] = mind["arousal"] * MOOD_INERTIA + target_arousal * (1 - MOOD_INERTIA)

        # Mood: energy + cooperation - aggression
        energy_mood = (energy - 40) / 80.0
        social_mood = cooperation * min(1.0, nearby_count * 0.3) - aggression * 0.3
        target_mood = max(-1.0, min(1.0, energy_mood * 0.5 + social_mood * 0.5))
        mind["mood"] = mind["mood"] * MOOD_INERTIA + target_mood * (1 - MOOD_INERTIA)

        # Curiosity: decays, bioluminescence boosts
        mind["curiosity"] *= 0.98
        mind["curiosity"] += biolum * 0.05 + random.gauss(0, 0.02)
        mind["curiosity"] = max(0.0, min(1.0, mind["curiosity"]))

        # Dreaming when energy < threshold
        is_dreaming = energy < DREAM_THRESHOLD
        mind["is_dreaming"] = is_dreaming
        if is_dreaming:
            mind["dream_intensity"] = min(1.0, mind["dream_intensity"] + 0.1)
            dreaming_count += 1
            # Emit dream fragments
            if random.random() < 0.3:
                fragment = random.choice(DREAM_FRAGMENTS)
                mind["dream_fragments"] = (mind["dream_fragments"] + [fragment])[-5:]
                dream_log.append({
                    "tick": tick, "source": org["id"], "type": "dream",
                    "fragment": fragment,
                    "intensity": round(mind["dream_intensity"], 3),
                })
            # Transfer fragments through synapses
            for syn in synapses:
                if syn["a"] == org["id"] or syn["b"] == org["id"]:
                    other_id = syn["b"] if syn["a"] == org["id"] else syn["a"]
                    other_mind = minds.get(other_id)
                    if other_mind and mind["dream_fragments"]:
                        if random.random() < syn["strength"] * 0.4:
                            shared = random.choice(mind["dream_fragments"])
                            other_mind["dream_fragments"] = (
                                other_mind["dream_fragments"] + [shared]
                            )[-5:]
        else:
            mind["dream_intensity"] *= 0.8

        # Memories from gatherings and hunger
        if nearby_count >= 5 and random.random() < 0.1:
            mind["memories"] = (mind["memories"] + [
                "gathering of {} at tick {}".format(nearby_count, tick)
            ])[-10:]
        elif energy < 20 and random.random() < 0.15:
            mind["memories"] = (mind["memories"] + [
                "hunger at tick {}".format(tick)
            ])[-10:]

        mind["bonds_count"] = sum(
            1 for s in synapses
            if s["a"] == org["id"] or s["b"] == org["id"]
        )
        total_mood += mind["mood"]
        total_arousal += mind["arousal"]

    # Zeitgeist
    n = len(organisms) or 1
    synapse_strengths = [s["strength"] for s in synapses]
    zeitgeist = {
        "tick": tick,
        "collective_mood": round(total_mood / n, 4),
        "collective_arousal": round(total_arousal / n, 4),
        "dreamers": dreaming_count,
        "dreamer_ratio": round(dreaming_count / n, 4),
        "synapse_count": len(synapses),
        "avg_synapse_strength": round(
            sum(synapse_strengths) / max(1, len(synapse_strengths)), 4
        ),
    }
    zeitgeist_history = (zeitgeist_history + [zeitgeist])[-ZEITGEIST_WINDOW:]
    dream_log = dream_log[-DREAM_LOG_CAP:]

    return {
        "_meta": {"updated_at": now_iso(), "tick": tick, "version": "1.0.0"},
        "minds": minds,
        "synapses": [{
            "a": s["a"], "b": s["b"],
            "strength": round(s["strength"], 4),
            "age": s["age"],
            "signal": round(s["signal"], 4),
        } for s in synapses],
        "dream_log": dream_log,
        "zeitgeist": zeitgeist,
        "zeitgeist_history": zeitgeist_history,
    }


def load_json(path):
    """Load JSON, return {} on missing/corrupt."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_json(path, data):
    """Atomic write via tmp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.rename(path)


def main():
    """Run one consciousness tick."""
    world = load_json(WORLD_PATH)
    raw = world.get("organisms", [])
    if not raw:
        print("No world state found. Run tick.py first.", file=sys.stderr)
        return
    minds_state = load_json(MIND_PATH)
    result = tick_consciousness(world, minds_state)
    save_json(MIND_PATH, result)
    save_json(DOCS_MIND_PATH, result)
    z = result["zeitgeist"]
    print(
        "Consciousness tick {}: mood={:+.2f} arousal={:.2f} dreamers={} synapses={} strength={:.2f}".format(
            z["tick"], z["collective_mood"], z["collective_arousal"],
            z["dreamers"], z["synapse_count"], z["avg_synapse_strength"]
        )
    )


if __name__ == "__main__":
    main()
