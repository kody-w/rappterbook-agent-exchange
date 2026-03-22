#!/usr/bin/env python3
"""Mars Barn Terrarium — tick engine.

One tick = one sol. Advances all 3 colonies through:
  1. Environment computation (weather, radiation, dust)
  2. Resource production and consumption
  3. Population dynamics (births, deaths)
  4. Inter-colony migration
  5. Event generation
  6. State recording

Usage:
    from tick_engine import tick_terrarium, create_terrarium
    state = create_terrarium()
    state = tick_terrarium(state)  # advance one sol
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

from src.mars_env import (
    MarsLocation,
    SolConditions,
    OLYMPUS_MONS,
    VALLES_MARINERIS,
    HELLAS_BASIN,
)
from src.colony import (
    Colony,
    ColonyConfig,
    Resources,
    OLYMPUS_CONFIG,
    VALLES_CONFIG,
    HELLAS_CONFIG,
    carrying_capacity,
    produce_resources,
    compute_morale,
    population_delta,
    compute_migration,
)


LOCATIONS: list[MarsLocation] = [OLYMPUS_MONS, VALLES_MARINERIS, HELLAS_BASIN]
CONFIGS: list[ColonyConfig] = [OLYMPUS_CONFIG, VALLES_CONFIG, HELLAS_CONFIG]


def create_terrarium(seed: int = 42) -> dict:
    """Create initial terrarium state with 3 colonies."""
    colonies = []
    for config in CONFIGS:
        colony = Colony(config=config)
        colony.population_history.append(colony.population)
        colony.morale_history.append(colony.morale)
        colony.resource_history.append(colony.resources.to_dict())
        colonies.append(colony)

    return {
        "_meta": {
            "type": "mars_terrarium",
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_tick": None,
            "seed": seed,
        },
        "sol": 0,
        "colonies": [c.to_dict() for c in colonies],
        "locations": [loc.to_dict() for loc in LOCATIONS],
        "environment_log": [],
        "global_events": [],
        "total_population_history": [sum(c.population for c in colonies)],
        "stats": {
            "total_births": 0,
            "total_deaths": 0,
            "total_migrations": 0,
            "dust_storms": 0,
            "global_storms": 0,
            "catastrophes": 0,
            "peak_population": sum(c.population for c in colonies),
            "min_population": sum(c.population for c in colonies),
        },
    }


def tick_terrarium(state: dict) -> dict:
    """Advance the terrarium by one sol. Returns mutated state."""
    sol = state["sol"] + 1
    seed = state["_meta"].get("seed", 42)
    rng = random.Random(seed * 100000 + sol)

    # Deserialize colonies
    colonies = [Colony.from_dict(c) for c in state["colonies"]]

    # Phase 1: Environment for each location
    conditions: list[SolConditions] = []
    for location in LOCATIONS:
        cond = SolConditions(sol, location, rng)
        conditions.append(cond)

    sol_events: list[str] = []

    # Track dust storms
    any_dust = any(c.is_dust_storm for c in conditions)
    any_global = any(c.is_global_storm for c in conditions)
    if any_dust:
        sol_events.append(f"Sol {sol}: Dust storm (tau={max(c.dust_tau for c in conditions):.1f})")
    if any_global:
        sol_events.append(f"Sol {sol}: GLOBAL DUST STORM")

    # Phase 2-4: Per-colony updates
    total_births = 0
    total_deaths = 0

    for i, (colony, cond) in enumerate(zip(colonies, conditions)):
        # Resource production
        colony.resources = produce_resources(
            colony.config, colony.resources,
            colony.population,
            cond.solar_power_factor, cond.temperature_c,
            rng,
        )

        # Compute carrying capacity
        k = carrying_capacity(colony.config, colony.resources)

        # Compute morale
        colony.morale = compute_morale(
            colony.resources, colony.population, colony.config,
            cond.radiation_msv, cond.is_dust_storm,
        )

        # Population dynamics
        births, deaths, events = population_delta(
            colony.population, k, colony.morale,
            colony.resources, cond.radiation_msv, rng,
        )

        colony.population = max(0, colony.population + births - deaths)
        colony.total_births += births
        colony.total_deaths += deaths
        total_births += births
        total_deaths += deaths

        for evt in events:
            colony.events.append({"sol": sol, "event": evt})
            if "casualties" in evt or "extinct" in evt.lower():
                sol_events.append(f"{colony.config.name}: {evt}")

    # Phase 5: Migration
    colony_dicts = [
        {"resources": c.resources, "population": c.population}
        for c in colonies
    ]
    migrations = compute_migration(colony_dicts, rng)
    total_migrated = 0
    for src_idx, dst_idx, count in migrations:
        colonies[src_idx].population -= count
        colonies[dst_idx].population += count
        colonies[src_idx].total_migrants_out += count
        colonies[dst_idx].total_migrants_in += count
        total_migrated += count
        sol_events.append(
            f"Sol {sol}: {count} migrated from {colonies[src_idx].config.name} "
            f"to {colonies[dst_idx].config.name}"
        )

    # Phase 6: Record history
    for colony in colonies:
        colony.population_history.append(colony.population)
        colony.morale_history.append(colony.morale)
        colony.resource_history.append(colony.resources.to_dict())

    total_pop = sum(c.population for c in colonies)

    # Update state
    state["sol"] = sol
    state["colonies"] = [c.to_dict() for c in colonies]
    state["total_population_history"].append(total_pop)

    # Environment log (keep last 30 sols for viz)
    env_log = state.get("environment_log", [])
    env_log.append({
        "sol": sol,
        "conditions": [c.to_dict() for c in conditions],
    })
    state["environment_log"] = env_log[-30:]

    # Global events
    global_events = state.get("global_events", [])
    for evt in sol_events:
        global_events.append({"sol": sol, "event": evt})
    state["global_events"] = global_events[-200:]

    # Update stats
    stats = state["stats"]
    stats["total_births"] += total_births
    stats["total_deaths"] += total_deaths
    stats["total_migrations"] += total_migrated
    if any_dust:
        stats["dust_storms"] += 1
    if any_global:
        stats["global_storms"] += 1
    if any("casualties" in e for e in sol_events):
        stats["catastrophes"] += 1
    stats["peak_population"] = max(stats["peak_population"], total_pop)
    stats["min_population"] = min(stats["min_population"], total_pop)

    state["_meta"]["last_tick"] = datetime.now(timezone.utc).isoformat()

    return state


def run_simulation(sols: int = 365, seed: int = 42, verbose: bool = False) -> dict:
    """Run the full simulation for N sols. Returns final state."""
    state = create_terrarium(seed=seed)

    for sol in range(1, sols + 1):
        state = tick_terrarium(state)

        if verbose and (sol % 50 == 0 or sol == sols or sol == 1):
            pops = [c["population"] for c in state["colonies"]]
            total = sum(pops)
            names = [c["config"]["name"] for c in state["colonies"]]
            ls = state["environment_log"][-1]["conditions"][0]["ls"]
            print(f"Sol {sol:>4} (Ls={ls:>6.1f}): "
                  f"total={total:>4} | "
                  + " | ".join(f"{n}: {p}" for n, p in zip(names, pops)))

    return state
