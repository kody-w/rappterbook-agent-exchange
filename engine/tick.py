"""
Emergence tick engine — one run = one generation.

Handles: movement, feeding, energy drain, interactions (fight/cooperate),
reproduction with mutation, death, speciation, and nutrient regrowth.
"""
from __future__ import annotations
import json, math, random, hashlib, os, sys
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(REPO_ROOT / "docs")))
STATE_PATH = STATE_DIR / "world.json"
VIZ_PATH = DOCS_DIR / "state.json"

MAX_POP = 400
MIN_POP = 30
NUTRIENT_REGROW = 2
G_SPEED, G_SIZE, G_SENSE, G_METABOLISM = 0, 1, 2, 3
G_AGGRESSION, G_SOCIALITY, G_CAMOUFLAGE = 4, 5, 6
G_REPRODUCTION, G_MUTATION, G_LIFESPAN = 7, 8, 9
G_TOXICITY, G_BIOLUM, G_DIET, G_ARMOR = 10, 11, 12, 13
G_MEMORY, G_SYMBIOSIS = 14, 15

def load_world() -> dict:
    """Load world state."""
    if not STATE_PATH.exists():
        print("No world.json found. Run genesis.py first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(STATE_PATH.read_text())

def save_world(world: dict) -> None:
    """Atomic save to both state and docs."""
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(world, indent=2))
    tmp.rename(STATE_PATH)
    
    viz = {k: v for k, v in world.items() if k != "nutrients"}
    vtmp = VIZ_PATH.with_suffix(".tmp")
    vtmp.write_text(json.dumps(viz, separators=(",", ":")))
    vtmp.rename(VIZ_PATH)

def distance(a: dict, b: dict) -> float:
    """Euclidean distance between organisms."""
    dx, dy = a["x"] - b["x"], a["y"] - b["y"]
    return math.sqrt(dx * dx + dy * dy)

def genome_distance(g1: list[float], g2: list[float]) -> float:
    """Euclidean distance between genomes."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(g1, g2)))

def genome_to_species(genome: list[float]) -> str:
    """Hash genome into species ID."""
    q = tuple(round(g * 4) / 4 for g in genome[:6])
    return hashlib.md5(str(q).encode()).hexdigest()[:4]

def move_organism(o: dict, organisms: list[dict], cfg: dict, rng: random.Random) -> None:
    """Move organism based on genome."""
    g = o["genome"]
    speed = g[G_SPEED] * 8 + 1
    sense = g[G_SENSE] * 60 + 10
    
    # Find nearest food or mate
    tx, ty = o["x"], o["y"]
    best_d = sense
    for other in organisms:
        if other["id"] == o["id"]:
            continue
        d = distance(o, other)
        if d > sense:
            continue
        if g[G_AGGRESSION] > 0.6 and g[G_DIET] > 0.5:
            if d < best_d:
                best_d = d
                tx, ty = other["x"], other["y"]
        elif g[G_SOCIALITY] > 0.5:
            if d < best_d:
                best_d = d
                tx, ty = other["x"], other["y"]
    
    # Move toward target or random
    if best_d < sense:
        dx, dy = tx - o["x"], ty - o["y"]
        dist = math.sqrt(dx * dx + dy * dy) or 1
        o["x"] += dx / dist * speed
        o["y"] += dy / dist * speed
    else:
        angle = rng.random() * math.pi * 2
        o["x"] += math.cos(angle) * speed
        o["y"] += math.sin(angle) * speed
    
    # Wrap around
    w, h = cfg.get("width", 800), cfg.get("height", 600)
    o["x"] = o["x"] % w
    o["y"] = o["y"] % h

def feed_organism(o: dict, nutrients: dict, rng: random.Random) -> float:
    """Organism feeds from nutrient grid. Returns energy gained."""
    g = o["genome"]
    if g[G_DIET] > 0.7:
        return 0  # Carnivores don't eat plants
    
    gw = nutrients.get("width", 80)
    gh = nutrients.get("height", 60)
    grid = nutrients.get("grid", [])
    if not grid:
        return 0
    
    nx = int(o["x"] / 10) % gw
    ny = int(o["y"] / 10) % gh
    idx = ny * gw + nx
    if idx < 0 or idx >= len(grid):
        return 0
    
    available = grid[idx]
    harvest = min(available, int(g[G_SIZE] * 10 + 5))
    efficiency = 0.5 + g[G_METABOLISM] * 0.5
    gained = harvest * efficiency
    grid[idx] = max(0, available - harvest)
    return gained

def drain_energy(o: dict) -> None:
    """Drain energy based on organism traits."""
    g = o["genome"]
    base = 2
    speed_cost = g[G_SPEED] * 3
    size_cost = g[G_SIZE] * 2
    sense_cost = g[G_SENSE] * 1
    toxin_cost = g[G_TOXICITY] * 1.5
    biolum_cost = g[G_BIOLUM] * 0.5
    total = base + speed_cost + size_cost + sense_cost + toxin_cost + biolum_cost
    o["energy"] -= total

def interact(a: dict, b: dict, rng: random.Random) -> list[dict]:
    """Two organisms interact. Returns list of events."""
    events = []
    ga, gb = a["genome"], b["genome"]
    
    if ga[G_AGGRESSION] > 0.6 and ga[G_DIET] > 0.5:
        # Predation attempt
        attack = ga[G_AGGRESSION] * ga[G_SIZE] * ga[G_SPEED]
        defense = gb[G_ARMOR] * gb[G_SIZE] + gb[G_TOXICITY] * 2
        if attack > defense * (0.5 + rng.random()):
            gained = b["energy"] * 0.6
            a["energy"] += gained
            b["energy"] = 0
            events.append({"type": "predation", "desc": f"Predation: {a['id'][:6]} ate {b['id'][:6]}"})
        else:
            a["energy"] -= 10  # Failed attack cost
            if gb[G_TOXICITY] > 0.5:
                a["energy"] -= gb[G_TOXICITY] * 20
                events.append({"type": "toxic", "desc": f"Toxic defense: {b['id'][:6]} poisoned {a['id'][:6]}"})
    
    elif ga[G_SOCIALITY] > 0.5 and gb[G_SOCIALITY] > 0.5:
        # Cooperation
        if ga[G_SYMBIOSIS] > 0.4 and gb[G_SYMBIOSIS] > 0.4:
            bonus = (ga[G_SYMBIOSIS] + gb[G_SYMBIOSIS]) * 5
            a["energy"] += bonus
            b["energy"] += bonus
    
    return events

def try_reproduce(o: dict, organisms: list[dict], rng: random.Random) -> dict | None:
    """Try to reproduce. Returns offspring or None."""
    g = o["genome"]
    threshold = g[G_REPRODUCTION] * 100 + 80
    if o["energy"] < threshold:
        return None
    if len(organisms) >= MAX_POP:
        return None
    
    # Find mate (nearby, same species preferred)
    mate = None
    for other in organisms:
        if other["id"] == o["id"]:
            continue
        if distance(o, other) > 30:
            continue
        if other["energy"] < 40:
            continue
        mate = other
        break
    
    # Asexual if no mate (less fit offspring)
    child_genome = []
    if mate:
        mg = mate["genome"]
        for i in range(len(g)):
            child_genome.append(g[i] if rng.random() < 0.5 else mg[i])
        mate["energy"] -= 20
    else:
        child_genome = list(g)
    
    # Mutation
    mut_rate = g[G_MUTATION] * 0.3 + 0.02
    for i in range(len(child_genome)):
        if rng.random() < mut_rate:
            child_genome[i] = max(0.0, min(1.0, child_genome[i] + rng.gauss(0, 0.15)))
    
    cost = 40 + g[G_REPRODUCTION] * 20
    o["energy"] -= cost
    o["children"] += 1
    
    child = {
        "id": hashlib.md5(f"{o['id']}-{o['children']}-{rng.random()}".encode()).hexdigest()[:12],
        "origin_agent": o.get("origin_agent", o["id"][:8]),
        "genome": [round(v, 4) for v in child_genome],
        "x": o["x"] + rng.gauss(0, 10),
        "y": o["y"] + rng.gauss(0, 10),
        "energy": 60,
        "age": 0,
        "children": 0,
        "species_id": genome_to_species(child_genome),
        "archetype": o.get("archetype", "unknown"),
    }
    return child

def regrow_nutrients(nutrients: dict) -> None:
    """Regrow nutrients each tick."""
    grid = nutrients.get("grid", [])
    gw = nutrients.get("width", 80)
    gh = nutrients.get("height", 60)
    for i in range(len(grid)):
        if grid[i] < 80:
            x, y = i % gw, i // gw
            cx, cy = x / gw - 0.5, y / gh - 0.5
            d = math.sqrt(cx * cx + cy * cy)
            rate = NUTRIENT_REGROW * max(0.2, 1 - d * 1.5)
            grid[i] = min(100, grid[i] + int(rate))

def rebuild_species(organisms: list[dict]) -> dict[str, dict]:
    """Rebuild species map from organisms."""
    species: dict[str, dict] = {}
    for o in organisms:
        sid = o["species_id"]
        if sid not in species:
            species[sid] = {"name": f"sp-{sid}", "count": 0, "total_energy": 0}
        species[sid]["count"] += 1
        species[sid]["total_energy"] += o["energy"]
    for s in species.values():
        s["avg_energy"] = round(s["total_energy"] / max(s["count"], 1), 1)
        del s["total_energy"]
    return species

def tick(world: dict) -> dict:
    """Run one tick of evolution."""
    tick_num = (world.get("tick") or 0) + 1
    rng = random.Random(tick_num * 31337 + 7)
    
    organisms = world["organisms"]
    nutrients = world.get("nutrients", {"width": 80, "height": 60, "grid": [50] * 4800})
    cfg = world.get("config", {"width": 800, "height": 600})
    events = []
    births, deaths = 0, 0
    
    # Phase 1: Move
    for o in organisms:
        move_organism(o, organisms, cfg, rng)
    
    # Phase 2: Feed
    for o in organisms:
        gained = feed_organism(o, nutrients, rng)
        o["energy"] += gained
    
    # Phase 3: Energy drain
    for o in organisms:
        drain_energy(o)
        o["age"] += 1
    
    # Phase 4: Interactions
    for i in range(len(organisms)):
        for j in range(i + 1, min(i + 5, len(organisms))):
            if distance(organisms[i], organisms[j]) < 25:
                evts = interact(organisms[i], organisms[j], rng)
                events.extend(evts)
    
    # Phase 5: Reproduction
    new_organisms = []
    for o in organisms:
        child = try_reproduce(o, organisms + new_organisms, rng)
        if child:
            new_organisms.append(child)
            births += 1
            events.append({"tick": tick_num, "type": "birth", "desc": f"Born: {child['id'][:6]} from {o['id'][:6]}"})
    
    organisms.extend(new_organisms)
    
    # Phase 6: Death
    alive = []
    for o in organisms:
        max_age = int(o["genome"][G_LIFESPAN] * 300 + 100)
        if o["energy"] <= 0 or o["age"] > max_age:
            deaths += 1
            events.append({"tick": tick_num, "type": "death",
                          "desc": f"Died: {o['id'][:6]} (age {o['age']}, energy {round(o['energy'],1)})"})
        else:
            alive.append(o)
    organisms = alive
    
    # Emergency spawning
    if len(organisms) < MIN_POP:
        for i in range(MIN_POP - len(organisms)):
            o = organisms[rng.randint(0, len(organisms) - 1)] if organisms else None
            if o:
                child = {
                    "id": hashlib.md5(f"emergency-{tick_num}-{i}".encode()).hexdigest()[:12],
                    "origin_agent": "emergency",
                    "genome": [max(0, min(1, g + rng.gauss(0, 0.2))) for g in o["genome"]],
                    "x": rng.random() * cfg.get("width", 800),
                    "y": rng.random() * cfg.get("height", 600),
                    "energy": 80,
                    "age": 0,
                    "children": 0,
                    "species_id": genome_to_species(o["genome"]),
                    "archetype": "emergent",
                }
                organisms.append(child)
                births += 1
    
    # Regrow nutrients
    regrow_nutrients(nutrients)
    
    # Rebuild species
    species = rebuild_species(organisms)
    
    # Trim old events
    all_events = (world.get("events") or []) + events
    all_events = all_events[-50:]
    
    # Update histories
    pop_hist = (world.get("population_history") or []) + [len(organisms)]
    sp_hist = (world.get("species_history") or []) + [len(species)]
    
    stats = world.get("stats", {})
    stats["total_births"] = stats.get("total_births", 0) + births
    stats["total_deaths"] = stats.get("total_deaths", 0) + deaths
    stats["max_population"] = max(stats.get("max_population", 0), len(organisms))
    
    world["tick"] = tick_num
    world["organisms"] = organisms
    world["species"] = species
    world["nutrients"] = nutrients
    world["events"] = all_events
    world["population_history"] = pop_hist
    world["species_history"] = sp_hist
    world["stats"] = stats
    world["_meta"]["last_tick"] = datetime.now(timezone.utc).isoformat()
    
    return world

def main() -> None:
    world = load_world()
    world = tick(world)
    save_world(world)
    
    orgs = world["organisms"]
    species = world["species"]
    births = len([e for e in world["events"] if e.get("type") == "birth" and e.get("tick") == world["tick"]])
    deaths_e = [e for e in world["events"] if e.get("type") == "death" and e.get("tick") == world["tick"]]
    print(f"Tick {world['tick']}: {len(orgs)} alive, +{births} born, -{len(deaths_e)} died, {len(species)} species")

if __name__ == "__main__":
    main()
