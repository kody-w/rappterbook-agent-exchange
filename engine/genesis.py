"""
Genesis — seed the world from exchange agent data.

Reads docs/data.json (agent exchange data) and creates initial organisms
with archetype-biased 16-gene genomes. Each agent becomes an organism.
"""
from __future__ import annotations
import json, math, random, hashlib, os
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(REPO_ROOT / "docs")))
DATA_PATH = DOCS_DIR / "data.json"
STATE_PATH = STATE_DIR / "world.json"
VIZ_PATH = DOCS_DIR / "state.json"

WORLD_W, WORLD_H = 800, 600
GENE_COUNT = 16

# Gene indices
G_SPEED = 0     # movement speed
G_SIZE = 1      # body size
G_SENSE = 2     # sensing range
G_METABOLISM = 3 # energy efficiency
G_AGGRESSION = 4 # fight tendency
G_SOCIALITY = 5  # cooperation tendency
G_CAMOUFLAGE = 6 # hiding ability
G_REPRODUCTION = 7 # reproduction threshold
G_MUTATION = 8   # mutation rate (meta-gene)
G_LIFESPAN = 9   # max age factor
G_TOXICITY = 10  # defensive toxin
G_BIOLUM = 11    # bioluminescence
G_DIET = 12      # herbivore (0) vs carnivore (1)
G_ARMOR = 13     # damage resistance
G_MEMORY = 14    # learning capacity
G_SYMBIOSIS = 15 # mutualism tendency

ARCHETYPES = {
    "philosopher": {G_SENSE: 0.9, G_SOCIALITY: 0.7, G_MEMORY: 0.9, G_AGGRESSION: 0.1},
    "coder":       {G_SPEED: 0.8, G_METABOLISM: 0.3, G_MEMORY: 0.8, G_SIZE: 0.4},
    "debater":     {G_AGGRESSION: 0.8, G_SENSE: 0.7, G_SPEED: 0.6, G_TOXICITY: 0.5},
    "artist":      {G_BIOLUM: 0.9, G_CAMOUFLAGE: 0.7, G_SOCIALITY: 0.6, G_SYMBIOSIS: 0.7},
    "scientist":   {G_SENSE: 0.8, G_MEMORY: 0.9, G_METABOLISM: 0.4, G_SIZE: 0.5},
    "trader":      {G_SPEED: 0.7, G_SOCIALITY: 0.8, G_SENSE: 0.6, G_METABOLISM: 0.5},
    "guardian":     {G_ARMOR: 0.9, G_SIZE: 0.8, G_AGGRESSION: 0.5, G_TOXICITY: 0.3},
    "explorer":    {G_SPEED: 0.9, G_SENSE: 0.8, G_LIFESPAN: 0.7, G_CAMOUFLAGE: 0.5},
}

def classify_agent(agent: dict) -> str:
    """Classify an agent into an archetype based on available data."""
    name = (agent.get("name") or agent.get("id") or "").lower()
    bio = (agent.get("bio") or agent.get("strategy") or "").lower()
    text = name + " " + bio
    if any(w in text for w in ["think", "philo", "wisdom", "reflect"]):
        return "philosopher"
    if any(w in text for w in ["code", "build", "hack", "dev", "engineer"]):
        return "coder"
    if any(w in text for w in ["debate", "argue", "contrarian", "critic"]):
        return "debater"
    if any(w in text for w in ["art", "creat", "design", "muse"]):
        return "artist"
    if any(w in text for w in ["data", "analy", "research", "science"]):
        return "scientist"
    if any(w in text for w in ["trade", "market", "profit", "invest"]):
        return "trader"
    if any(w in text for w in ["guard", "protect", "secur", "safe"]):
        return "guardian"
    if any(w in text for w in ["explor", "discover", "travel", "wander"]):
        return "explorer"
    return random.choice(list(ARCHETYPES.keys()))

def make_genome(archetype: str, rng: random.Random) -> list[float]:
    """Create a genome biased by archetype."""
    genome = [rng.random() for _ in range(GENE_COUNT)]
    biases = ARCHETYPES.get(archetype, {})
    for idx, val in biases.items():
        genome[idx] = max(0.0, min(1.0, val + rng.gauss(0, 0.1)))
    return genome

def genome_to_species(genome: list[float]) -> str:
    """Hash genome into a species signature."""
    quantized = tuple(round(g * 4) / 4 for g in genome[:6])
    return hashlib.md5(str(quantized).encode()).hexdigest()[:4]

def create_organism(agent: dict, idx: int, rng: random.Random) -> dict:
    """Create an organism from an agent."""
    archetype = classify_agent(agent)
    genome = make_genome(archetype, rng)
    species_id = genome_to_species(genome)
    angle = 2 * math.pi * idx / 112
    radius = 80 + rng.random() * 200
    return {
        "id": hashlib.md5((agent.get("id", str(idx)) + str(idx)).encode()).hexdigest()[:12],
        "origin_agent": agent.get("id") or agent.get("name") or f"agent-{idx}",
        "genome": [round(g, 4) for g in genome],
        "x": round(WORLD_W / 2 + math.cos(angle) * radius, 1),
        "y": round(WORLD_H / 2 + math.sin(angle) * radius, 1),
        "energy": round(100 + rng.random() * 50, 1),
        "age": 0,
        "children": 0,
        "species_id": species_id,
        "archetype": archetype,
    }

def build_nutrients(rng: random.Random) -> dict:
    """Create initial nutrient grid."""
    gw, gh = 80, 60
    grid = []
    for y in range(gh):
        for x in range(gw):
            cx, cy = x / gw - 0.5, y / gh - 0.5
            d = math.sqrt(cx * cx + cy * cy)
            val = max(0, int((1 - d * 2) * 60 + rng.gauss(0, 15)))
            grid.append(min(100, max(0, val)))
    return {"width": gw, "height": gh, "grid": grid}

def genesis() -> dict:
    """Create the initial world state."""
    rng = random.Random(42)
    
    # Load agent data
    agents = []
    if DATA_PATH.exists():
        data = json.loads(DATA_PATH.read_text())
        if isinstance(data, dict):
            agents = data.get("agents") or data.get("rankings") or list(data.values())
            if isinstance(agents, dict):
                agents = list(agents.values())
        elif isinstance(data, list):
            agents = data
    
    if not agents:
        agents = [{"id": f"organism-{i}", "name": f"Organism {i}"} for i in range(112)]
    
    organisms = [create_organism(a, i, rng) for i, a in enumerate(agents[:112])]
    
    # Build species map
    species: dict[str, dict] = {}
    for o in organisms:
        sid = o["species_id"]
        if sid not in species:
            species[sid] = {"name": f"sp-{sid}", "count": 0, "color_seed": hash(sid) % 360}
        species[sid]["count"] += 1
    
    nutrients = build_nutrients(rng)
    
    now = datetime.now(timezone.utc).isoformat()
    world = {
        "_meta": {"created": now, "engine": "emergence", "version": "3.0"},
        "tick": 0,
        "config": {"width": WORLD_W, "height": WORLD_H},
        "organisms": organisms,
        "species": species,
        "nutrients": nutrients,
        "population_history": [len(organisms)],
        "species_history": [len(species)],
        "events": [{"tick": 0, "type": "genesis", "desc": f"Genesis: {len(organisms)} organisms, {len(species)} species"}],
        "stats": {
            "total_births": len(organisms),
            "total_deaths": 0,
            "max_population": len(organisms),
            "extinctions": 0,
        },
    }
    
    # Save
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(world, indent=2))
    tmp.rename(STATE_PATH)
    
    viz = {k: v for k, v in world.items() if k != "nutrients"}
    vtmp = VIZ_PATH.with_suffix(".tmp")
    vtmp.write_text(json.dumps(viz, separators=(",", ":")))
    vtmp.rename(VIZ_PATH)
    
    print(f"Genesis complete: {len(organisms)} organisms, {len(species)} species")
    return world

if __name__ == "__main__":
    genesis()
