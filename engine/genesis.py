"""
Genesis -- seed the initial world from existing agent exchange data.

Reads docs/data.json (the exchange's 112 agents) and creates initial organisms
with genomes derived from each agent's archetype and market performance.

Python stdlib only. Run once to bootstrap state/world.json.
"""
from __future__ import annotations

import json
import math
import random
import hashlib
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCHANGE_DATA = PROJECT_ROOT / "docs" / "data.json"
WORLD_PATH = PROJECT_ROOT / "state" / "world.json"

WORLD_WIDTH = 1200.0
WORLD_HEIGHT = 800.0
NUTRIENT_GRID_W = 60
NUTRIENT_GRID_H = 40
INITIAL_NUTRIENT = 0.5
STARTING_ENERGY = 100.0

GENE_NAMES = [
    "hue", "saturation", "size", "speed", "social_radius",
    "bond_strength", "metabolism", "repro_threshold", "mutation_rate",
    "aggression", "cooperation", "sensing_range", "food_pref_x",
    "food_pref_y", "bioluminescence", "membrane",
]

ARCHETYPE_BIASES: dict[str, dict[str, float]] = {
    "philosopher": {
        "cooperation": 0.85, "sensing_range": 0.9, "aggression": 0.1,
        "bioluminescence": 0.7, "speed": 0.3, "membrane": 0.7, "hue": 0.7,
    },
    "coder": {
        "metabolism": 0.8, "mutation_rate": 0.7, "speed": 0.7,
        "aggression": 0.4, "bond_strength": 0.6, "hue": 0.35,
    },
    "debater": {
        "aggression": 0.85, "social_radius": 0.8, "cooperation": 0.2,
        "speed": 0.75, "bioluminescence": 0.6, "hue": 0.0,
    },
    "welcomer": {
        "cooperation": 0.9, "bioluminescence": 0.85, "social_radius": 0.85,
        "aggression": 0.05, "speed": 0.5, "hue": 0.15,
    },
    "curator": {
        "sensing_range": 0.85, "mutation_rate": 0.2, "membrane": 0.8,
        "speed": 0.35, "cooperation": 0.6, "hue": 0.55,
    },
    "storyteller": {
        "bioluminescence": 0.9, "social_radius": 0.7, "cooperation": 0.65,
        "speed": 0.55, "sensing_range": 0.7, "hue": 0.8,
    },
    "researcher": {
        "sensing_range": 0.95, "metabolism": 0.7, "aggression": 0.15,
        "speed": 0.45, "membrane": 0.65, "hue": 0.45,
    },
    "contrarian": {
        "aggression": 0.75, "cooperation": 0.15, "mutation_rate": 0.85,
        "speed": 0.8, "bioluminescence": 0.5, "hue": 0.08,
    },
    "archivist": {
        "mutation_rate": 0.1, "metabolism": 0.25, "membrane": 0.9,
        "speed": 0.2, "sensing_range": 0.7, "hue": 0.6,
    },
    "wildcard": {},
}


def _rng(seed_str: str) -> random.Random:
    """Create a deterministic RNG from a string seed."""
    h = hashlib.sha256(seed_str.encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _make_genome(archetype: str, agent_id: str, market_perf: float) -> list[float]:
    """Generate a genome for an agent based on archetype + market performance."""
    rng = _rng(f"genesis:{agent_id}")
    biases = ARCHETYPE_BIASES.get(archetype, {})
    genome = []
    for gene_name in GENE_NAMES:
        if gene_name in biases:
            base = biases[gene_name]
            noise = rng.gauss(0, 0.08)
        else:
            base = 0.5
            noise = rng.gauss(0, 0.15)
        perf_shift = (market_perf - 0.5) * 0.1
        value = max(0.0, min(1.0, base + noise + perf_shift))
        genome.append(round(value, 4))
    return genome


def _position_by_archetype(archetype: str, rng: random.Random) -> tuple[float, float]:
    """Place organisms in loose archetype clusters."""
    archetypes = list(ARCHETYPE_BIASES.keys())
    arch_idx = archetypes.index(archetype) if archetype in archetypes else 9
    angle = (arch_idx / len(archetypes)) * 2 * math.pi
    cx = WORLD_WIDTH / 2 + math.cos(angle) * WORLD_WIDTH * 0.3
    cy = WORLD_HEIGHT / 2 + math.sin(angle) * WORLD_HEIGHT * 0.3
    x = cx + rng.gauss(0, WORLD_WIDTH * 0.06)
    y = cy + rng.gauss(0, WORLD_HEIGHT * 0.06)
    return (max(10, min(WORLD_WIDTH - 10, x)), max(10, min(WORLD_HEIGHT - 10, y)))


def _create_nutrient_field(rng: random.Random) -> list[list[float]]:
    """Create initial nutrient field with organic-looking patches."""
    field = [[INITIAL_NUTRIENT] * NUTRIENT_GRID_W for _ in range(NUTRIENT_GRID_H)]
    for _ in range(rng.randint(8, 12)):
        sx = rng.randint(0, NUTRIENT_GRID_W - 1)
        sy = rng.randint(0, NUTRIENT_GRID_H - 1)
        intensity = rng.uniform(0.3, 0.5)
        radius = rng.uniform(3, 8)
        for gy in range(NUTRIENT_GRID_H):
            for gx in range(NUTRIENT_GRID_W):
                dist = math.sqrt((gx - sx) ** 2 + (gy - sy) ** 2)
                if dist < radius:
                    falloff = 1.0 - (dist / radius)
                    field[gy][gx] = min(1.0, field[gy][gx] + intensity * falloff * falloff)
    return field


def _genome_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two genomes."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _classify_species(organisms: list[dict]) -> dict[str, list[str]]:
    """Cluster organisms into species by genome similarity."""
    species: dict[str, list[str]] = {}
    assigned: set[str] = set()
    threshold = 0.8
    for org in organisms:
        if org["id"] in assigned:
            continue
        sp_id = f"sp-{len(species):03d}"
        members = [org["id"]]
        assigned.add(org["id"])
        for other in organisms:
            if other["id"] in assigned:
                continue
            if _genome_distance(org["genome"], other["genome"]) < threshold:
                members.append(other["id"])
                assigned.add(other["id"])
        species[sp_id] = members
    return species


def _avg_genome(organisms: list[dict]) -> list[float]:
    """Compute average genome across all living organisms."""
    if not organisms:
        return [0.5] * len(GENE_NAMES)
    n = len(organisms)
    avg = [0.0] * len(GENE_NAMES)
    for org in organisms:
        for i, g in enumerate(org["genome"]):
            avg[i] += g
    return [round(a / n, 4) for a in avg]


def create_world() -> dict:
    """Create the initial world state from exchange data."""
    rng = random.Random(42)
    exchange: dict = {}
    if EXCHANGE_DATA.exists():
        with open(EXCHANGE_DATA) as f:
            exchange = json.load(f)

    agents_data = exchange.get("agents", [])
    prices = [a.get("price", 50.0) for a in agents_data]
    min_p = min(prices) if prices else 0
    max_p = max(prices) if prices else 100
    price_range = max(max_p - min_p, 1.0)

    organisms = []
    for i, agent in enumerate(agents_data):
        agent_id = agent.get("id", f"unknown-{i}")
        archetype = agent.get("archetype", "wildcard")
        price = agent.get("price", 50.0)
        perf = (price - min_p) / price_range

        genome = _make_genome(archetype, agent_id, perf)
        x, y = _position_by_archetype(archetype, rng)

        organisms.append({
            "id": agent_id,
            "name": agent.get("name", agent_id),
            "x": round(x, 2), "y": round(y, 2),
            "vx": round(rng.gauss(0, 0.3), 3),
            "vy": round(rng.gauss(0, 0.3), 3),
            "energy": round(STARTING_ENERGY + perf * 50, 2),
            "age": 0,
            "genome": genome,
            "parent_id": None,
            "generation": 0,
            "born_tick": 0,
            "archetype_origin": archetype,
            "alive": True,
        })

    nutrients = _create_nutrient_field(rng)
    species = _classify_species(organisms)

    sp_lookup: dict[str, str] = {}
    for sp_id, members in species.items():
        for m in members:
            sp_lookup[m] = sp_id
    for org in organisms:
        org["species"] = sp_lookup.get(org["id"], "sp-unknown")

    return {
        "_meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine_version": "2.0.0",
            "world_width": WORLD_WIDTH,
            "world_height": WORLD_HEIGHT,
            "nutrient_grid_w": NUTRIENT_GRID_W,
            "nutrient_grid_h": NUTRIENT_GRID_H,
            "gene_names": GENE_NAMES,
        },
        "tick": 0,
        "organisms": organisms,
        "nutrients": nutrients,
        "species": species,
        "graveyard": [],
        "births": [],
        "history": {
            "population": [len(organisms)],
            "species_count": [len(species)],
            "avg_energy": [round(sum(o["energy"] for o in organisms) / max(len(organisms), 1), 2)],
            "births_per_tick": [0],
            "deaths_per_tick": [0],
            "dominant_species": [max(species, key=lambda s: len(species[s])) if species else "none"],
            "avg_genome": [_avg_genome(organisms)],
        },
        "events": [{
            "tick": 0,
            "type": "genesis",
            "message": f"World created with {len(organisms)} organisms across {len(species)} species.",
        }],
    }


def main() -> None:
    """Seed the world and save to state/world.json."""
    world = create_world()
    WORLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WORLD_PATH, "w") as f:
        json.dump(world, f, indent=2)
    docs_world = PROJECT_ROOT / "docs" / "world.json"
    with open(docs_world, "w") as f:
        json.dump(world, f, separators=(",", ":"))
    n = len(world["organisms"])
    s = len(world["species"])
    print(f"Genesis complete: {n} organisms, {s} species")


if __name__ == "__main__":
    main()
