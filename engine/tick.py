"""
Tick -- one heartbeat of the living ecosystem.

Each run: load state, simulate one generation (move, feed, interact,
reproduce, kill), reclassify species, record history, save state.

Python stdlib only. Zero dependencies.
"""
from __future__ import annotations

import json
import math
import random
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORLD_PATH = PROJECT_ROOT / "state" / "world.json"
DOCS_WORLD = PROJECT_ROOT / "docs" / "world.json"

MAX_POPULATION = 400
MIN_POPULATION = 30
NUTRIENT_REGEN_RATE = 0.02
NUTRIENT_CONSUMPTION = 0.15
ENERGY_DRAIN_BASE = 0.8
REPRODUCTION_COST = 0.6
MUTATION_BASE = 0.05
INTERACTION_RANGE = 60.0
AGGRESSION_DAMAGE = 8.0
COOPERATION_GIFT = 4.0
SPEED_SCALE = 3.0
SENSING_SCALE = 80.0
MAX_AGE = 200
GRAVEYARD_LIMIT = 50
BIRTHS_LOG_LIMIT = 50
EVENTS_LIMIT = 200
HISTORY_LIMIT = 500


def load_world() -> dict:
    """Load world state from JSON."""
    with open(WORLD_PATH) as f:
        return json.load(f)


def save_world(world: dict) -> None:
    """Save world state to both locations atomically."""
    tmp = WORLD_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(world, f, indent=2)
    tmp.replace(WORLD_PATH)
    tmp2 = DOCS_WORLD.with_suffix(".tmp")
    with open(tmp2, "w") as f:
        json.dump(world, f, separators=(",", ":"))
    tmp2.replace(DOCS_WORLD)


def _move_organisms(organisms: list[dict], nutrients: list[list[float]],
                    meta: dict, rng: random.Random) -> None:
    """Move each organism based on genome + nutrient gradients."""
    w, h = meta["world_width"], meta["world_height"]
    gw, gh = meta["nutrient_grid_w"], meta["nutrient_grid_h"]
    cell_w, cell_h = w / gw, h / gh

    for org in organisms:
        if not org["alive"]:
            continue
        genome = org["genome"]
        speed = genome[3] * SPEED_SCALE
        sensing = genome[11] * SENSING_SCALE

        dx = org["vx"] + rng.gauss(0, 0.5)
        dy = org["vy"] + rng.gauss(0, 0.5)

        gx = int(org["x"] / cell_w) % gw
        gy = int(org["y"] / cell_h) % gh
        best_val, best_dx, best_dy = -1.0, 0.0, 0.0
        search_r = max(1, int(sensing / max(cell_w, cell_h)))

        for sy in range(-search_r, search_r + 1):
            for sx in range(-search_r, search_r + 1):
                nx, ny = (gx + sx) % gw, (gy + sy) % gh
                val = nutrients[ny][nx]
                if val > best_val:
                    best_val = val
                    best_dx, best_dy = sx * cell_w, sy * cell_h

        if best_val > 0.01:
            dist = math.sqrt(best_dx * best_dx + best_dy * best_dy) + 0.001
            dx += (best_dx / dist) * speed * 0.4
            dy += (best_dy / dist) * speed * 0.4

        mag = math.sqrt(dx * dx + dy * dy) + 0.001
        dx = (dx / mag) * min(mag, speed)
        dy = (dy / mag) * min(mag, speed)

        org["x"] = (org["x"] + dx) % w
        org["y"] = (org["y"] + dy) % h
        org["vx"] = dx * 0.8
        org["vy"] = dy * 0.8


def _feed_organisms(organisms: list[dict], nutrients: list[list[float]],
                    meta: dict) -> None:
    """Organisms consume nutrients from the grid."""
    gw, gh = meta["nutrient_grid_w"], meta["nutrient_grid_h"]
    cell_w, cell_h = meta["world_width"] / gw, meta["world_height"] / gh

    for org in organisms:
        if not org["alive"]:
            continue
        gx = int(org["x"] / cell_w) % gw
        gy = int(org["y"] / cell_h) % gh
        available = nutrients[gy][gx]
        consumed = min(available, NUTRIENT_CONSUMPTION * (0.5 + org["genome"][6] * 0.5))
        nutrients[gy][gx] = max(0.0, available - consumed)
        org["energy"] += consumed * 30


def _drain_energy(organisms: list[dict]) -> None:
    """Each organism loses energy based on metabolism gene."""
    for org in organisms:
        if not org["alive"]:
            continue
        metabolism = org["genome"][6]
        size = org["genome"][2]
        speed_cost = abs(org["vx"]) + abs(org["vy"])
        drain = ENERGY_DRAIN_BASE * (0.5 + metabolism * 0.8) + size * 0.3 + speed_cost * 0.1
        drain *= (1.0 - org["genome"][15] * 0.3)
        org["energy"] -= drain


def _interact(organisms: list[dict], rng: random.Random) -> None:
    """Handle organism-organism interactions."""
    alive = [o for o in organisms if o["alive"]]
    grid: dict[tuple[int, int], list[dict]] = {}
    cell_size = INTERACTION_RANGE * 1.5
    for org in alive:
        key = (int(org["x"] / cell_size), int(org["y"] / cell_size))
        grid.setdefault(key, []).append(org)

    for org in alive:
        key = (int(org["x"] / cell_size), int(org["y"] / cell_size))
        social_r = org["genome"][4] * INTERACTION_RANGE
        for dkx in range(-1, 2):
            for dky in range(-1, 2):
                for other in grid.get((key[0] + dkx, key[1] + dky), []):
                    if other["id"] == org["id"] or not other["alive"]:
                        continue
                    dx = org["x"] - other["x"]
                    dy = org["y"] - other["y"]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > social_r:
                        continue
                    if org["genome"][9] > org["genome"][10] and rng.random() < org["genome"][9] * 0.3:
                        stolen = min(other["energy"] * 0.1, AGGRESSION_DAMAGE * org["genome"][9])
                        other["energy"] -= stolen
                        org["energy"] += stolen * 0.7
                    elif org["genome"][10] > 0.5 and rng.random() < org["genome"][10] * 0.2:
                        gift = min(org["energy"] * 0.05, COOPERATION_GIFT * org["genome"][10])
                        org["energy"] -= gift
                        other["energy"] += gift * 1.3


def _next_id(tick: int, idx: int) -> str:
    """Generate a unique ID for a new organism."""
    h = hashlib.md5(f"{tick}:{idx}".encode()).hexdigest()[:8]
    return f"org-{tick}-{h}"


def _mutate_genome(parent: list[float], mutation_rate: float,
                   rng: random.Random) -> list[float]:
    """Create a mutated copy of a genome."""
    child = []
    for gene in parent:
        if rng.random() < mutation_rate:
            delta = rng.gauss(0, MUTATION_BASE + mutation_rate * 0.1)
            child.append(max(0.0, min(1.0, round(gene + delta, 4))))
        else:
            child.append(gene)
    return child


def _reproduce(organisms: list[dict], tick_num: int,
               rng: random.Random) -> list[dict]:
    """Organisms with enough energy reproduce."""
    alive = [o for o in organisms if o["alive"]]
    if len(alive) >= MAX_POPULATION:
        return []
    births = []
    birth_idx = 0
    for org in alive:
        if len(alive) + len(births) >= MAX_POPULATION:
            break
        threshold = org["genome"][7] * 150 + 80
        if org["energy"] < threshold or rng.random() > 0.3:
            continue
        cost = org["energy"] * REPRODUCTION_COST
        org["energy"] -= cost
        child_genome = _mutate_genome(org["genome"], org["genome"][8], rng)
        angle = rng.uniform(0, 2 * math.pi)
        offset = rng.uniform(5, 20)
        births.append({
            "id": _next_id(tick_num, birth_idx),
            "name": f"Gen{org['generation'] + 1}-{birth_idx}",
            "x": round(org["x"] + math.cos(angle) * offset, 2),
            "y": round(org["y"] + math.sin(angle) * offset, 2),
            "vx": round(rng.gauss(0, 0.3), 3),
            "vy": round(rng.gauss(0, 0.3), 3),
            "energy": round(cost * 0.8, 2),
            "age": 0,
            "genome": child_genome,
            "parent_id": org["id"],
            "generation": org["generation"] + 1,
            "born_tick": tick_num,
            "archetype_origin": org.get("archetype_origin", "evolved"),
            "species": org.get("species", "sp-unknown"),
            "alive": True,
        })
        birth_idx += 1
    return births


def _kill_organisms(organisms: list[dict], tick_num: int,
                    rng: random.Random) -> list[dict]:
    """Remove dead organisms."""
    dead = []
    for org in organisms:
        if not org["alive"]:
            continue
        died = False
        if org["energy"] <= 0:
            died = True
        elif org["age"] > MAX_AGE:
            if rng.random() < (org["age"] - MAX_AGE) / 100.0:
                died = True
        if died:
            org["alive"] = False
            dead.append({
                "id": org["id"], "name": org.get("name", ""),
                "died_tick": tick_num, "age": org["age"],
                "generation": org["generation"],
                "species": org.get("species", ""),
                "genome": org["genome"],
                "x": org["x"], "y": org["y"],
            })
    return dead


def _regenerate_nutrients(nutrients: list[list[float]], tick_num: int,
                          rng: random.Random) -> None:
    """Slowly regenerate nutrients with shifting hotspots."""
    gh, gw = len(nutrients), len(nutrients[0])
    for gy in range(gh):
        for gx in range(gw):
            nutrients[gy][gx] = min(1.0, nutrients[gy][gx] + NUTRIENT_REGEN_RATE)
    if tick_num % 10 == 0:
        for _ in range(rng.randint(2, 5)):
            sx, sy = rng.randint(0, gw - 1), rng.randint(0, gh - 1)
            intensity = rng.uniform(0.2, 0.4)
            radius = rng.uniform(3, 6)
            for gy in range(gh):
                for gx in range(gw):
                    dist = math.sqrt((gx - sx) ** 2 + (gy - sy) ** 2)
                    if dist < radius:
                        falloff = 1.0 - (dist / radius)
                        nutrients[gy][gx] = min(1.0, nutrients[gy][gx] + intensity * falloff ** 2)


def _reclassify_species(organisms: list[dict],
                        existing: dict[str, list[str]]) -> dict[str, list[str]]:
    """Reclassify organisms into species by genome similarity."""
    alive = [o for o in organisms if o["alive"]]
    if not alive:
        return {}
    threshold = 0.9
    species: dict[str, list[str]] = {}
    assigned: set[str] = set()
    sp_counter = max(
        (int(s.split("-")[1]) for s in existing
         if s.startswith("sp-") and s.split("-")[1].isdigit()),
        default=-1,
    ) + 1

    for org in alive:
        if org["id"] in assigned:
            continue
        sp_id = f"sp-{sp_counter:03d}"
        sp_counter += 1
        members = [org["id"]]
        assigned.add(org["id"])
        for other in alive:
            if other["id"] in assigned:
                continue
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(org["genome"], other["genome"])))
            if dist < threshold:
                members.append(other["id"])
                assigned.add(other["id"])
        species[sp_id] = members
    sp_lookup = {m: sp for sp, ms in species.items() for m in ms}
    for org in alive:
        org["species"] = sp_lookup.get(org["id"], org.get("species", "sp-unknown"))
    return species


def _emergency_spawn(organisms: list[dict], tick_num: int,
                     rng: random.Random, meta: dict) -> list[dict]:
    """If population drops too low, spawn from surviving genetics."""
    alive = [o for o in organisms if o["alive"]]
    if len(alive) >= MIN_POPULATION:
        return []
    templates = alive or [o for o in organisms if o.get("genome")]
    if not templates:
        return []
    spawned = []
    for i in range(MIN_POPULATION - len(alive)):
        template = rng.choice(templates)
        genome = _mutate_genome(template["genome"], 0.3, rng)
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(50, 200)
        spawned.append({
            "id": _next_id(tick_num, 1000 + i),
            "name": f"Emergent-{tick_num}-{i}",
            "x": round((meta["world_width"] / 2 + math.cos(angle) * dist) % meta["world_width"], 2),
            "y": round((meta["world_height"] / 2 + math.sin(angle) * dist) % meta["world_height"], 2),
            "vx": round(rng.gauss(0, 0.5), 3),
            "vy": round(rng.gauss(0, 0.5), 3),
            "energy": 120.0, "age": 0, "genome": genome,
            "parent_id": "emergence", "generation": 0,
            "born_tick": tick_num, "archetype_origin": "emergent",
            "species": "sp-unknown", "alive": True,
        })
    return spawned


def _avg_genome(organisms: list[dict]) -> list[float]:
    """Average genome across living organisms."""
    alive = [o for o in organisms if o["alive"]]
    if not alive:
        return [0.5] * 16
    n = len(alive)
    avg = [0.0] * len(alive[0]["genome"])
    for org in alive:
        for i, g in enumerate(org["genome"]):
            avg[i] += g
    return [round(a / n, 4) for a in avg]


def _record_history(world: dict, num_births: int, num_deaths: int) -> None:
    """Append this tick's stats to history arrays."""
    hist = world["history"]
    alive = [o for o in world["organisms"] if o["alive"]]
    hist["population"].append(len(alive))
    hist["species_count"].append(len(world["species"]))
    hist["avg_energy"].append(
        round(sum(o["energy"] for o in alive) / max(len(alive), 1), 2))
    hist["births_per_tick"].append(num_births)
    hist["deaths_per_tick"].append(num_deaths)
    dominant = max(world["species"], key=lambda s: len(world["species"][s])) if world["species"] else "none"
    hist["dominant_species"].append(dominant)
    hist["avg_genome"].append(_avg_genome(world["organisms"]))
    for key in hist:
        if isinstance(hist[key], list) and len(hist[key]) > HISTORY_LIMIT:
            hist[key] = hist[key][-HISTORY_LIMIT:]


def _generate_events(world: dict, births: list[dict], dead: list[dict],
                     tick_num: int) -> None:
    """Generate narrative events for notable occurrences."""
    alive = [o for o in world["organisms"] if o["alive"]]
    evts = world["events"]
    if len(births) > 10:
        evts.append({"tick": tick_num, "type": "boom",
                      "message": f"Population boom! {len(births)} births."})
    if len(dead) > 10:
        evts.append({"tick": tick_num, "type": "extinction",
                      "message": f"Mass die-off: {len(dead)} perished."})
    pop = len(alive)
    if pop > 0 and pop % 50 == 0:
        evts.append({"tick": tick_num, "type": "milestone",
                      "message": f"Population reached {pop}."})
    if len(evts) > EVENTS_LIMIT:
        world["events"] = evts[-EVENTS_LIMIT:]


def tick() -> dict:
    """Execute one tick of the ecosystem simulation."""
    world = load_world()
    tick_num = world["tick"] + 1
    world["tick"] = tick_num
    rng = random.Random(tick_num * 31337 + 7)
    meta = world["_meta"]
    organisms = world["organisms"]

    for org in organisms:
        if org["alive"]:
            org["age"] += 1

    _move_organisms(organisms, world["nutrients"], meta, rng)
    _feed_organisms(organisms, world["nutrients"], meta)
    _drain_energy(organisms)
    _interact(organisms, rng)

    births = _reproduce(organisms, tick_num, rng)
    organisms.extend(births)

    dead = _kill_organisms(organisms, tick_num, rng)

    spawned = _emergency_spawn(organisms, tick_num, rng, meta)
    organisms.extend(spawned)
    births.extend(spawned)

    _regenerate_nutrients(world["nutrients"], tick_num, rng)
    world["species"] = _reclassify_species(organisms, world.get("species", {}))

    alive = [o for o in organisms if o["alive"]]
    recent_dead = [o for o in organisms if not o["alive"] and o.get("age", 0) < 10]
    world["organisms"] = alive + recent_dead[-20:]

    world["graveyard"] = (world.get("graveyard", []) + dead)[-GRAVEYARD_LIMIT:]
    birth_records = [{"id": b["id"], "parent_id": b["parent_id"],
                      "tick": tick_num, "genome": b["genome"][:4]} for b in births]
    world["births"] = (world.get("births", []) + birth_records)[-BIRTHS_LOG_LIMIT:]

    _record_history(world, len(births), len(dead))
    _generate_events(world, births, dead, tick_num)

    world["_meta"]["last_tick_at"] = datetime.now(timezone.utc).isoformat()
    world["_meta"]["total_organisms_ever"] = world["_meta"].get("total_organisms_ever", len(alive)) + len(births)

    save_world(world)
    return {"tick": tick_num, "alive": len(alive), "births": len(births),
            "deaths": len(dead), "species": len(world["species"])}


def main() -> None:
    """Run one tick and print summary."""
    if not WORLD_PATH.exists():
        print("No world state found. Run genesis.py first.")
        sys.exit(1)
    result = tick()
    print(f"Tick {result['tick']}: {result['alive']} alive, "
          f"+{result['births']} born, -{result['deaths']} died, "
          f"{result['species']} species")


if __name__ == "__main__":
    main()
