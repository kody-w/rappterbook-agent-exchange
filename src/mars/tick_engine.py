"""Mars Barn tick engine — advance all colonies by one sol.

This is THE core simulation loop. Each call = one sol of Mars time.

    world = load_world()
    for sol in range(365):
        world = tick(world)
    save_world(world)

The tick function is pure: (world_state) → (world_state').
Side effects (I/O) only happen in load/save.
"""
from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from .colony import Colony, ColonyConfig, COLONY_CONFIGS
from .environment import MarsEnvironment, sol_to_ls

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(REPO_ROOT / "docs")))
STATE_PATH = STATE_DIR / "mars.json"
VIZ_PATH = DOCS_DIR / "mars_data.json"


def create_world(seed: int = 42) -> dict:
    """Create initial Mars world state with 3 colonies."""
    colonies = []
    for cfg in COLONY_CONFIGS:
        colony = Colony(cfg)
        colonies.append(colony.to_dict())

    return {
        "_meta": {
            "engine": "mars-barn",
            "version": "1.0.0",
            "created": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
        },
        "sol": 0,
        "seed": seed,
        "colonies": colonies,
        "environment_log": [],
        "summary": {
            "total_population": sum(c["population"] for c in colonies),
            "total_births": 0,
            "total_deaths": 0,
            "sols_simulated": 0,
            "dust_storm_sols": 0,
        },
    }


def _rebuild_colony(data: dict) -> Colony:
    """Rebuild a Colony object from serialized dict."""
    # Find matching config
    cfg = None
    for c in COLONY_CONFIGS:
        if c.name == data["name"]:
            cfg = c
            break
    if cfg is None:
        # Fallback: create config from data
        loc = data["location"]
        cfg = ColonyConfig(
            name=data["name"],
            latitude=loc["latitude"],
            longitude=loc["longitude"],
            altitude_km=loc["altitude_km"],
            description=data.get("description", ""),
            initial_crew=0,
            ice_accessibility=0.5,
            dust_exposure=0.5,
            terrain_difficulty=0.5,
        )

    colony = Colony(cfg)
    colony.population = data["population"]
    colony.sol = data["sol"]

    infra = data["infrastructure"]
    colony.habitat_volume_m3 = infra["habitat_volume_m3"]
    colony.greenhouse_area_m2 = infra["greenhouse_area_m2"]
    colony.solar_panel_area_m2 = infra["solar_panel_area_m2"]
    colony.ice_miners = infra["ice_miners"]
    colony.has_shielding = infra.get("has_shielding", True)

    res = data["resources"]
    colony.water_kg = res["water_kg"]
    colony.food_kg = res["food_kg"]
    colony.regolith_kg = res["regolith_kg"]
    colony.stored_power_kwh = res["stored_power_kwh"]

    demo = data["demographics"]
    colony.total_births = demo["total_births"]
    colony.total_deaths = demo["total_deaths"]
    colony.total_immigrants = demo["total_immigrants"]
    colony.morale = demo["morale"]

    colony.history = data.get("history", [])
    colony.events = data.get("events", [])

    return colony


def tick(world: dict) -> dict:
    """Advance the Mars world by one sol. Pure function.

    Args:
        world: Current world state dict.

    Returns:
        Updated world state dict (new object, input not mutated).
    """
    sol = world["sol"] + 1
    seed = world.get("seed", 42)
    rng = random.Random(seed * 1000000 + sol)

    # Rebuild colony objects
    colonies = [_rebuild_colony(c) for c in world["colonies"]]

    # Generate environment for each colony
    env_reports = []
    sol_reports = []
    dust_storm_this_sol = False

    for colony in colonies:
        env = MarsEnvironment(
            sol=sol,
            latitude=colony.config.latitude,
            altitude_km=colony.config.altitude_km,
            has_shielding=colony.has_shielding,
            rng=random.Random(rng.random()),
        )
        report = colony.tick(env, random.Random(rng.random()))
        env_reports.append(env.to_dict())
        sol_reports.append(report)
        if env.dust_storm:
            dust_storm_this_sol = True

    # Serialize colonies back
    colony_dicts = [c.to_dict() for c in colonies]

    # Build environment log entry (compact)
    env_entry = {
        "sol": sol,
        "ls": round(sol_to_ls(sol), 1),
        "colonies": {
            e["colony"]: {
                "pop": e["population"],
                "cap": e["carrying_capacity"],
                "births": e["births"],
                "deaths": e["deaths"],
            }
            for e in sol_reports
        },
        "dust_storm": dust_storm_this_sol,
    }

    # Keep environment log trimmed
    env_log = world.get("environment_log", []) + [env_entry]
    if len(env_log) > 400:
        env_log = env_log[-400:]

    # Summary stats
    total_pop = sum(c.population for c in colonies)
    summary = world.get("summary", {})
    summary["total_population"] = total_pop
    summary["total_births"] = sum(c.total_births for c in colonies)
    summary["total_deaths"] = sum(c.total_deaths for c in colonies)
    summary["sols_simulated"] = sol
    summary["dust_storm_sols"] = summary.get("dust_storm_sols", 0) + (
        1 if dust_storm_this_sol else 0
    )
    summary["peak_population"] = max(
        summary.get("peak_population", 0), total_pop
    )

    return {
        "_meta": {
            **world["_meta"],
            "last_tick": datetime.now(timezone.utc).isoformat(),
        },
        "sol": sol,
        "seed": seed,
        "colonies": colony_dicts,
        "environment_log": env_log,
        "summary": summary,
    }


def load_world() -> dict:
    """Load Mars world state from disk."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return create_world()


def save_world(world: dict) -> None:
    """Save Mars world to state/ and docs/ for visualization."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Full state
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(world, indent=2))
    tmp.rename(STATE_PATH)

    # Compact viz data (population curves + events only)
    viz = _build_viz_data(world)
    vtmp = VIZ_PATH.with_suffix(".tmp")
    vtmp.write_text(json.dumps(viz, separators=(",", ":")))
    vtmp.rename(VIZ_PATH)


def _build_viz_data(world: dict) -> dict:
    """Extract visualization data from world state.

    Returns compact JSON for the frontend chart.
    """
    colonies_viz = []
    for colony_data in world["colonies"]:
        history = colony_data.get("history", [])
        colonies_viz.append({
            "name": colony_data["name"],
            "description": colony_data["description"],
            "location": colony_data["location"],
            "population": colony_data["population"],
            "curve": {
                "sols": [h["sol"] for h in history],
                "population": [h["population"] for h in history],
                "carrying_capacity": [h["carrying_capacity"] for h in history],
                "food_kg": [h["food_kg"] for h in history],
                "water_kg": [h["water_kg"] for h in history],
                "morale": [h["morale"] for h in history],
                "temperature_c": [h["temperature_c"] for h in history],
            },
            "events": colony_data.get("events", [])[-30:],
        })

    return {
        "_meta": world["_meta"],
        "sol": world["sol"],
        "summary": world["summary"],
        "colonies": colonies_viz,
    }
