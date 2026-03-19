#!/usr/bin/env python3
"""The Abyss -- evolution engine.  One run = one tick of evolution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORLD_W, WORLD_H = 1200, 800
GRID_W, GRID_H = 60, 40
CELL_W, CELL_H = WORLD_W / GRID_W, WORLD_H / GRID_H
INITIAL_POP = 80
GENE_COUNT = 16
MAX_POP = 400
HISTORY_CAP = 600
PHEROMONE_DEPOSIT = 0.08
PHEROMONE_DECAY = 0.97

EPOCHS = [
    (0,    "Primordial Soup"),
    (50,   "First Sparks"),
    (200,  "The Cambrian"),
    (500,  "Age of Predators"),
    (1000, "Symbiotic Era"),
    (2000, "Radiant Bloom"),
    (5000, "Deep Time"),
]

GENE_NAMES = [
    "hue", "saturation", "size", "speed", "social_radius",
    "bond_strength", "metabolism", "repro_threshold", "mutation_rate",
    "aggression", "cooperation", "sensing_range",
    "food_pref_x", "food_pref_y", "bioluminescence", "membrane",
]

STATE_PATH = Path(os.environ.get("STATE_DIR", "docs")) / "world.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uid(prefix: str = "org", tick: int = 0) -> str:
    h = hashlib.md5(f"{tick}-{time.time_ns()}-{random.random()}".encode()).hexdigest()[:6]
    return f"{prefix}-{tick}-{h}"


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def dist(a: dict, b: dict) -> float:
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return math.sqrt(dx * dx + dy * dy)


def genome_dist(g1: list[float], g2: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(g1, g2)))


def epoch_name(tick: int) -> str:
    name = EPOCHS[0][1]
    for threshold, label in EPOCHS:
        if tick >= threshold:
            name = label
    return name


# ---------------------------------------------------------------------------
# Organism factory
# ---------------------------------------------------------------------------

def make_organism(tick: int, genome: list[float] | None = None,
                  parent_id: str = "", generation: int = 0,
                  x: float | None = None, y: float | None = None) -> dict:
    """Create a new organism with a 16-gene genome."""
    g = genome or [random.random() for _ in range(GENE_COUNT)]
    return {
        "id": _uid("org", tick),
        "name": "",
        "x": x if x is not None else random.uniform(20, WORLD_W - 20),
        "y": y if y is not None else random.uniform(20, WORLD_H - 20),
        "vx": random.uniform(-1, 1),
        "vy": random.uniform(-1, 1),
        "energy": 50.0 + g[2] * 30,
        "genome": [round(v, 4) for v in g],
        "age": 0,
        "generation": generation,
        "species_id": "sp-0",
        "born_tick": tick,
        "parent_id": parent_id,
    }


# ---------------------------------------------------------------------------
# Nutrient grid
# ---------------------------------------------------------------------------

def make_grid() -> list[float]:
    """Create the initial 60x40 nutrient grid."""
    return [random.uniform(0.2, 0.6) for _ in range(GRID_W * GRID_H)]


def regrow(grid: list[float], tick: int) -> None:
    """Regrow nutrients with seasonal variation."""
    season = 0.5 + 0.3 * math.sin(tick * 0.02)
    rate = 0.005 * season
    for i in range(len(grid)):
        grid[i] = clamp(grid[i] + rate + random.uniform(0, 0.002))


def add_food_cluster(grid: list[float]) -> None:
    """Add a random nutrient cluster to the grid."""
    cx = random.randint(2, GRID_W - 3)
    cy = random.randint(2, GRID_H - 3)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            gx, gy = cx + dx, cy + dy
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                grid[gy * GRID_W + gx] = clamp(
                    grid[gy * GRID_W + gx] + random.uniform(0.1, 0.4)
                )


# ---------------------------------------------------------------------------
# Species clustering (every 10 ticks)
# ---------------------------------------------------------------------------

def cluster_species(organisms: list[dict], tick: int) -> dict[str, list[str]]:
    """Cluster organisms into species by genome similarity."""
    if not organisms:
        return {}
    threshold = 1.8
    species: dict[str, list[str]] = {}
    centroids: list[tuple[str, list[float]]] = []

    for org in organisms:
        assigned = False
        for sp_id, centroid in centroids:
            if genome_dist(org["genome"], centroid) < threshold:
                org["species_id"] = sp_id
                species.setdefault(sp_id, []).append(org["id"])
                assigned = True
                break
        if not assigned:
            sp_id = f"sp-{len(centroids) + 1}"
            org["species_id"] = sp_id
            centroids.append((sp_id, list(org["genome"])))
            species[sp_id] = [org["id"]]
    return species


# ---------------------------------------------------------------------------
# Simulation step
# ---------------------------------------------------------------------------

def _steer_to_nutrients(org: dict, grid: list[float]) -> None:
    """Steer organism toward the richest nearby nutrient cell."""
    gx = int(clamp(org["x"] / CELL_W, 0, GRID_W - 1))
    gy = int(clamp(org["y"] / CELL_H, 0, GRID_H - 1))
    best_val, best_dx, best_dy = -1.0, 0.0, 0.0
    for ddy in range(-2, 3):
        for ddx in range(-2, 3):
            nx, ny = gx + ddx, gy + ddy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                val = grid[ny * GRID_W + nx]
                if val > best_val:
                    best_val = val
                    best_dx = ddx * CELL_W
                    best_dy = ddy * CELL_H
    if best_val > 0:
        mag = math.sqrt(best_dx ** 2 + best_dy ** 2) + 1e-6
        org["vx"] += (best_dx / mag) * 0.3
        org["vy"] += (best_dy / mag) * 0.3


def _flock(org: dict, organisms: list[dict], dead_ids: set[str]) -> None:
    """Cooperative organisms flock with same-species neighbours."""
    g = org["genome"]
    coop = g[10]
    if coop <= 0.5:
        return
    fx, fy, count = 0.0, 0.0, 0
    social_r = 30 + g[4] * 80
    for other in organisms:
        if other["id"] == org["id"] or other["id"] in dead_ids:
            continue
        if other["species_id"] == org["species_id"]:
            d = dist(org, other)
            if d < social_r and d > 1:
                fx += (other["x"] - org["x"]) / d
                fy += (other["y"] - org["y"]) / d
                count += 1
    if count > 0:
        org["vx"] += fx / count * 0.15 * coop
        org["vy"] += fy / count * 0.15 * coop


def _flee(org: dict, organisms: list[dict], dead_ids: set[str]) -> None:
    """Flee from nearby larger aggressive organisms."""
    g = org["genome"]
    sense = 20 + g[11] * 120
    for other in organisms:
        if other["id"] == org["id"] or other["id"] in dead_ids:
            continue
        if other["genome"][9] > 0.6 and other["genome"][2] > g[2]:
            d = dist(org, other)
            if 0 < d < sense * 0.6:
                org["vx"] -= (other["x"] - org["x"]) / d * 0.5
                org["vy"] -= (other["y"] - org["y"]) / d * 0.5


def _steer_to_pheromones(org: dict, pheromones: list[float]) -> None:
    """Cooperative organisms follow pheromone trails left by others."""
    g = org["genome"]
    if g[10] <= 0.3:
        return
    gx = int(clamp(org["x"] / CELL_W, 0, GRID_W - 1))
    gy = int(clamp(org["y"] / CELL_H, 0, GRID_H - 1))
    best_val, best_dx, best_dy = -1.0, 0.0, 0.0
    for ddy in range(-2, 3):
        for ddx in range(-2, 3):
            if ddx == 0 and ddy == 0:
                continue
            nx, ny = gx + ddx, gy + ddy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                val = pheromones[ny * GRID_W + nx]
                if val > best_val:
                    best_val = val
                    best_dx = ddx * CELL_W
                    best_dy = ddy * CELL_H
    if best_val > 0.01:
        mag = math.sqrt(best_dx ** 2 + best_dy ** 2) + 1e-6
        org["vx"] += (best_dx / mag) * 0.2 * g[10]
        org["vy"] += (best_dy / mag) * 0.2 * g[10]


def _deposit_pheromone(org: dict, pheromones: list[float]) -> None:
    """Organism deposits pheromone at its grid cell."""
    gx = int(clamp(org["x"] / CELL_W, 0, GRID_W - 1))
    gy = int(clamp(org["y"] / CELL_H, 0, GRID_H - 1))
    idx = gy * GRID_W + gx
    strength = org["genome"][10] * PHEROMONE_DEPOSIT
    pheromones[idx] = clamp(pheromones[idx] + strength)


def _hunt(org: dict, organisms: list[dict], dead_ids: set[str],
          tick: int, events: list[dict]) -> None:
    """Aggressive organisms hunt and eat smaller organisms of other species."""
    g = org["genome"]
    aggr = g[9]
    if aggr <= 0.5:
        return
    sense = 20 + g[11] * 120
    size_val = 3 + g[2] * 12
    closest, cd = None, sense
    for other in organisms:
        if other["id"] == org["id"] or other["id"] in dead_ids:
            continue
        if other["species_id"] != org["species_id"] and other["genome"][2] < g[2]:
            d = dist(org, other)
            if d < cd:
                cd = d
                closest = other
    if closest is not None:
        d = cd + 1e-6
        org["vx"] += (closest["x"] - org["x"]) / d * 0.4 * aggr
        org["vy"] += (closest["y"] - org["y"]) / d * 0.4 * aggr
        if cd < size_val + 3:
            gained = closest["energy"] * 0.6
            org["energy"] += gained
            dead_ids.add(closest["id"])
            events.append({
                "tick": tick, "type": "death",
                "message": f"{closest['id']} consumed by {org['id']}"
            })


def step(organisms: list[dict], grid: list[float], tick: int,
         events: list[dict], pheromones: list[float] | None = None) -> list[dict]:
    """Run one simulation step — move, eat, hunt, reproduce, die."""
    new_orgs: list[dict] = []
    dead_ids: set[str] = set()
    if pheromones is None:
        pheromones = [0.0] * (GRID_W * GRID_H)

    for org in organisms:
        if org["id"] in dead_ids:
            continue

        g = org["genome"]
        speed = 0.5 + g[3] * 3.0
        metab = 0.1 + g[6] * 0.5
        size_val = 3 + g[2] * 12

        _steer_to_nutrients(org, grid)
        _flock(org, organisms, dead_ids)
        _flee(org, organisms, dead_ids)
        _steer_to_pheromones(org, pheromones)
        _hunt(org, organisms, dead_ids, tick, events)

        # velocity damping + move
        vmag = math.sqrt(org["vx"] ** 2 + org["vy"] ** 2) + 1e-6
        if vmag > speed:
            org["vx"] = org["vx"] / vmag * speed
            org["vy"] = org["vy"] / vmag * speed
        org["x"] += org["vx"]
        org["y"] += org["vy"]

        # boundary wrap
        if org["x"] < 0:
            org["x"] += WORLD_W
        if org["x"] >= WORLD_W:
            org["x"] -= WORLD_W
        if org["y"] < 0:
            org["y"] += WORLD_H
        if org["y"] >= WORLD_H:
            org["y"] -= WORLD_H

        # eat nutrients
        gx = int(clamp(org["x"] / CELL_W, 0, GRID_W - 1))
        gy = int(clamp(org["y"] / CELL_H, 0, GRID_H - 1))
        idx = gy * GRID_W + gx
        eaten = min(grid[idx], 0.15 + g[6] * 0.15)
        grid[idx] -= eaten
        org["energy"] += eaten * 30

        # metabolism cost
        org["energy"] -= metab + (speed * 0.05) + (size_val * 0.01)
        org["age"] += 1

        # starvation
        if org["energy"] <= 0:
            dead_ids.add(org["id"])
            events.append({
                "tick": tick, "type": "death",
                "message": f"{org['id']} starved (age {org['age']})"
            })
            continue

        # reproduction
        repro_thresh = 60 + g[7] * 80
        if org["energy"] > repro_thresh and len(organisms) + len(new_orgs) < MAX_POP:
            child_genome = list(g)
            mut_rate = g[8] * 0.15
            for i in range(GENE_COUNT):
                if random.random() < mut_rate:
                    child_genome[i] = clamp(child_genome[i] + random.gauss(0, 0.1))
            child = make_organism(
                tick, child_genome,
                parent_id=org["id"],
                generation=org["generation"] + 1,
                x=org["x"] + random.uniform(-15, 15),
                y=org["y"] + random.uniform(-15, 15),
            )
            cost = 30 + g[2] * 20
            org["energy"] -= cost
            child["energy"] = cost * 0.8
            new_orgs.append(child)
            events.append({
                "tick": tick, "type": "birth",
                "message": f"{child['id']} born from {org['id']}"
            })

    # Deposit pheromones and decay
    for org in organisms:
        if org["id"] not in dead_ids:
            _deposit_pheromone(org, pheromones)
    for i in range(len(pheromones)):
        pheromones[i] *= PHEROMONE_DECAY

    survivors = [o for o in organisms if o["id"] not in dead_ids]
    survivors.extend(new_orgs)
    return survivors


# ---------------------------------------------------------------------------
# World init / load / save
# ---------------------------------------------------------------------------

def new_world() -> dict:
    """Create a fresh world with initial organisms and nutrient grid."""
    tick = 0
    organisms = [make_organism(tick) for _ in range(INITIAL_POP)]
    grid = make_grid()
    species = cluster_species(organisms, tick)
    return {
        "_meta": {
            "world_width": WORLD_W,
            "world_height": WORLD_H,
            "nutrient_grid_w": GRID_W,
            "nutrient_grid_h": GRID_H,
            "epoch": epoch_name(tick),
        },
        "tick": tick,
        "organisms": organisms,
        "nutrients": [round(v, 3) for v in grid],
        "species": species,
        "history": {
            "population": [len(organisms)],
            "species_count": [len(species)],
            "avg_energy": [50.0],
        },
        "events": [{"tick": 0, "type": "genesis", "message": "The Abyss awakens..."}],
        "pheromones": [0.0] * (GRID_W * GRID_H),
    }


def load_world() -> dict:
    """Load world from disk or create new."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return new_world()


def save_world(world: dict) -> None:
    """Save world to disk as compact JSON."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(world, f, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Main tick loop
# ---------------------------------------------------------------------------

def run_tick(world: dict) -> dict:
    """Advance the world by one tick."""
    tick = world["tick"] + 1
    organisms = world["organisms"]
    grid = world["nutrients"]
    events = world["events"]
    history = world["history"]
    pheromones = world.get("pheromones", [0.0] * (GRID_W * GRID_H))

    # cap old events
    if len(events) > 200:
        events = events[-150:]

    # regrow nutrients + random clusters
    regrow(grid, tick)
    if random.random() < 0.15:
        add_food_cluster(grid)

    # evolve
    organisms = step(organisms, grid, tick, events, pheromones)

    # re-cluster species every 10 ticks
    if tick % 10 == 0:
        species = cluster_species(organisms, tick)
    else:
        species = {}
        for o in organisms:
            species.setdefault(o["species_id"], []).append(o["id"])

    # speciation events
    sp_count = len(species)
    old_sp = history["species_count"][-1] if history["species_count"] else 0
    if sp_count > old_sp and tick % 10 == 0:
        events.append({
            "tick": tick, "type": "speciation",
            "message": f"New species emerged! Now {sp_count} species"
        })

    # extinction rescue
    if len(organisms) < 5:
        for _ in range(20):
            organisms.append(make_organism(tick))
        events.append({
            "tick": tick, "type": "genesis",
            "message": "Mass extinction rescue -- new organisms seeded"
        })
        species = cluster_species(organisms, tick)

    # track history
    avg_e = sum(o["energy"] for o in organisms) / max(1, len(organisms))
    history["population"].append(len(organisms))
    history["species_count"].append(len(species))
    history["avg_energy"].append(round(avg_e, 1))
    for key in history:
        if len(history[key]) > HISTORY_CAP:
            history[key] = history[key][-HISTORY_CAP:]

    world["tick"] = tick
    world["organisms"] = organisms
    world["nutrients"] = [round(v, 3) for v in grid]
    world["pheromones"] = [round(v, 4) for v in pheromones]
    world["species"] = species
    world["history"] = history
    world["events"] = events
    world["_meta"]["epoch"] = epoch_name(tick)
    return world


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="The Abyss -- evolution engine")
    parser.add_argument("--ticks", type=int, default=1, help="Number of ticks")
    parser.add_argument("--genesis", action="store_true", help="Force new world")
    args = parser.parse_args()

    world = new_world() if args.genesis else load_world()
    print(f"[abyss] Starting at tick {world['tick']} with {len(world['organisms'])} organisms")

    for i in range(args.ticks):
        world = run_tick(world)
        if (i + 1) % 10 == 0 or i == args.ticks - 1:
            pop = len(world["organisms"])
            sp = len(world["species"])
            print(f"  tick {world['tick']}: pop={pop}  species={sp}  epoch={world['_meta']['epoch']}")

    save_world(world)
    print(f"[abyss] Saved to {STATE_PATH} -- {len(world['organisms'])} organisms, tick {world['tick']}")


if __name__ == "__main__":
    main()
