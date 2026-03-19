#!/usr/bin/env python3
"""Rebirth — when a fly dies, the next generation begins.

Reads the dead fly's state, creates an offspring with a mutated genome,
places the egg near the parent's death site, adds the corpse as a kitchen
object, and stores the lineage.  The cycle of life continues.

The output of generation N's death is the input to generation N+1's birth.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(ROOT / "docs")))


def mutate_genome(parent_genome: dict, rng: random.Random) -> dict:
    """Create offspring genome via mutation of parent's genes.

    Each numeric gene has a chance to shift slightly.  Occasionally a gene
    makes a bigger jump — a hopeful monster.
    """
    child = {}
    for key, val in parent_genome.items():
        if key == "species":
            child[key] = val
            continue
        if key == "eye_facets":
            delta = rng.randint(-200, 200)
            child[key] = max(2000, min(6000, val + delta))
            continue
        if isinstance(val, (int, float)):
            # Normal mutation: small gaussian shift
            sigma = 0.04
            # Rare big mutation (5% chance)
            if rng.random() < 0.05:
                sigma = 0.15
            mutated = val + rng.gauss(0, sigma)
            # Clamp to reasonable ranges
            if key in ("metabolic_rate", "flight_efficiency", "smell_sensitivity",
                       "heat_tolerance", "lifespan_modifier"):
                mutated = max(0.3, min(1.5, mutated))
            elif key in ("wing_vein_pattern", "body_color_hue", "bristle_density"):
                mutated = max(0.0, min(1.0, mutated))
            child[key] = round(mutated, 4)
        else:
            child[key] = val
    return child


def inherit_instincts(parent_memory: dict) -> dict:
    """Extract survival instincts from parent's memory.

    Not full memory — just vague inherited knowledge about food and danger.
    Epigenetics in action.
    """
    instincts = {
        "inherited_food_bias": None,
        "inherited_danger_awareness": 0.0,
        "parent_peak_altitude": parent_memory.get("peak_altitude", 0),
        "parent_total_distance": parent_memory.get("total_distance", 0),
    }
    # If parent had a favorite food, offspring has a vague attraction to it
    if parent_memory.get("favorite_food"):
        instincts["inherited_food_bias"] = parent_memory["favorite_food"]
    # If parent fled a lot, offspring is more cautious
    times_fled = parent_memory.get("times_fled", 0)
    if times_fled > 3:
        instincts["inherited_danger_awareness"] = min(1.0, times_fled * 0.1)
    return instincts


def create_corpse(parent_state: dict) -> dict:
    """Create a corpse object from the dead fly for the kitchen."""
    pos = parent_state["body"]["position"]
    return {
        "id": "corpse_gen1",
        "type": "food",
        "x": round(pos["x"], 1),
        "y": round(pos["y"], 1),
        "z": 0,
        "smell_radius": 60,
        "energy": 5,
        "decay": 0.95,
        "name": "dead fly (gen 1)",
        "is_corpse": True,
        "decomposition": 0.0,
    }


def summarize_generation(state: dict) -> dict:
    """Create a compact summary of a completed generation."""
    return {
        "generation": state["_meta"].get("generation", 1),
        "born_at": state["_meta"].get("born_at"),
        "died_at_tick": state["_meta"].get("died_at", state["lifecycle"]["total_ticks"]),
        "cause_of_death": state["_meta"].get("cause_of_death", "unknown"),
        "total_ticks": state["lifecycle"]["total_ticks"],
        "genome": copy.deepcopy(state["genome"]),
        "final_energy": round(state["energy"]["current"], 1),
        "times_fed": state["memory"].get("times_fed", 0),
        "times_fled": state["memory"].get("times_fled", 0),
        "total_distance": round(state["memory"].get("total_distance", 0), 1),
        "peak_altitude": state["memory"].get("peak_altitude", 0),
        "decisions_made": state["brain"].get("decisions_made", 0),
        "favorite_food": state["memory"].get("favorite_food"),
        "death_position": copy.deepcopy(state["body"]["position"]),
        "history": copy.deepcopy(state["history"][-20:]),
    }


def rebirth(dead_state: dict) -> dict:
    """Create generation N+1 from dead generation N.

    This is the data slosh: the dead fly's state IS the input,
    the new egg's state IS the output.
    """
    rng = random.Random()
    gen = dead_state["_meta"].get("generation", 1)
    new_gen = gen + 1

    # Mutate genome
    child_genome = mutate_genome(dead_state["genome"], rng)

    # Egg position: near parent's death site (flies oviposit near food)
    parent_pos = dead_state["body"]["position"]
    egg_x = parent_pos["x"] + rng.uniform(-20, 20)
    egg_y = parent_pos["y"] + rng.uniform(-20, 20)
    egg_x = max(10, min(dead_state["kitchen"]["width"] - 10, egg_x))
    egg_y = max(10, min(dead_state["kitchen"]["height"] - 10, egg_y))

    # Inherit instincts from parent
    instincts = inherit_instincts(dead_state["memory"])

    # Build lineage
    lineage = dead_state.get("lineage", [])
    lineage.append(summarize_generation(dead_state))

    # Kitchen evolves: add corpse, shift time, maybe new food
    kitchen = copy.deepcopy(dead_state["kitchen"])
    # Remove old corpses
    kitchen["objects"] = [o for o in kitchen["objects"] if not o.get("is_corpse")]
    # Add parent's corpse
    kitchen["objects"].append(create_corpse(dead_state))
    # Time passes — advance time_of_day
    kitchen["time_of_day"] = (kitchen["time_of_day"] + 0.15) % 1.0
    tod = kitchen["time_of_day"]
    kitchen["lights_on"] = 0.25 < tod < 0.85
    kitchen["ambient_temp"] = 20 + 4 * math.sin(tod * math.pi)
    # Deactivate threats (fresh start)
    for obj in kitchen["objects"]:
        if obj["type"] == "threat":
            obj["active"] = False
            obj["x"] = -100
            obj["y"] = -100
    # Maybe add a new food source each generation
    new_foods = [
        {"id": "honey", "type": "food", "x": 180, "y": 90, "z": 0,
         "smell_radius": 250, "energy": 35, "decay": 0.4, "name": "honey drop"},
        {"id": "fruit", "type": "food", "x": 400, "y": 300, "z": 0,
         "smell_radius": 280, "energy": 28, "decay": 0.6, "name": "rotting fruit"},
        {"id": "milk", "type": "food", "x": 520, "y": 80, "z": 0,
         "smell_radius": 180, "energy": 22, "decay": 0.5, "name": "spilled milk"},
    ]
    existing_ids = {o["id"] for o in kitchen["objects"]}
    candidates = [f for f in new_foods if f["id"] not in existing_ids]
    if candidates:
        kitchen["objects"].append(rng.choice(candidates))

    # Stage durations: slightly randomized per generation
    stage_durations = {
        "egg": rng.randint(6, 10),
        "larva": rng.randint(20, 30),
        "pupa": rng.randint(15, 22),
        "adult": rng.randint(50, 70),
    }

    new_state = {
        "_meta": {
            "organism": "Musca domestica",
            "frame": dead_state["_meta"]["frame"] + 1,
            "born_at": f"2025-07-18T{new_gen:02d}:00:00Z",
            "version": "2.0.0",
            "generation": new_gen,
            "cause_of_death": None,
            "died_at": None,
            "total_frames_alive": 0,
        },
        "genome": child_genome,
        "lifecycle": {
            "stage": "egg",
            "stage_tick": 0,
            "total_ticks": 0,
            "stage_durations": stage_durations,
            "molts": 0,
            "larva_instar": 0,
        },
        "body": {
            "position": {"x": round(egg_x, 2), "y": round(egg_y, 2), "z": 0.5},
            "velocity": {"x": 0, "y": 0, "z": 0},
            "facing": rng.uniform(0, 2 * math.pi),
            "size": 1.0,
            "mass": 0.001,
            "wing_state": "none",
            "leg_state": "none",
            "is_airborne": False,
            "surface": "counter",
        },
        "energy": {
            "current": 85.0,
            "max": 100,
            "hunger": 0.0,
            "metabolic_drain": 0.5,
            "last_fed_tick": 0,
        },
        "brain": {
            "state": "dormant",
            "current_goal": None,
            "fear_level": instincts["inherited_danger_awareness"] * 0.3,
            "curiosity": 0.5,
            "satisfaction": 0.5,
            "decisions_made": 0,
            "neural_complexity": 0.01,
        },
        "senses": {
            "smell": [],
            "sight": [],
            "touch": {"surface": "counter", "vibration": 0.0},
            "temperature": kitchen["ambient_temp"],
            "wind": 0.0,
        },
        "memory": {
            "food_sources": [],
            "danger_zones": [],
            "visited_positions": [],
            "total_distance": 0.0,
            "times_fed": 0,
            "times_fled": 0,
            "peak_altitude": 0.0,
            "favorite_food": None,
            "inherited_instincts": instincts,
        },
        "kitchen": kitchen,
        "lineage": lineage,
        "history": [
            {
                "tick": 0,
                "event": f"generation {new_gen} egg laid",
                "stage": "egg",
                "energy": 85.0,
                "position": {"x": round(egg_x, 2), "y": round(egg_y, 2), "z": 0.5},
            }
        ],
        "narration": f"Near the remains of its parent, a new egg glistens. Generation {new_gen} begins.",
    }

    return new_state


def main() -> None:
    """Load dead fly, rebirth, save."""
    state_path = STATE_DIR / "fly.json"
    if not state_path.exists():
        print("No fly.json found.", file=sys.stderr)
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    if state["lifecycle"]["stage"] != "death":
        print(f"Fly is still alive (stage: {state['lifecycle']['stage']}). No rebirth needed.")
        return

    gen = state["_meta"].get("generation", 1)
    print(f"Generation {gen} is dead. Initiating rebirth...")

    new_state = rebirth(state)
    new_gen = new_state["_meta"]["generation"]

    # Save
    with open(state_path, "w") as f:
        json.dump(new_state, f, indent=2)
    with open(DOCS_DIR / "fly_state.json", "w") as f:
        json.dump(new_state, f, separators=(",", ":"))

    print(f"Generation {new_gen} egg laid at ({new_state['body']['position']['x']:.1f}, {new_state['body']['position']['y']:.1f})")
    print(f"  Genome mutations: {sum(1 for k in new_state['genome'] if k != 'species' and new_state['genome'][k] != state['genome'].get(k, None))} genes changed")
    print(f"  Lineage: {len(new_state['lineage'])} ancestors recorded")
    print(f"  Kitchen: {len(new_state['kitchen']['objects'])} objects ({sum(1 for o in new_state['kitchen']['objects'] if o['type'] == 'food')} food)")
    print(f"  {new_state['narration']}")


if __name__ == "__main__":
    main()
