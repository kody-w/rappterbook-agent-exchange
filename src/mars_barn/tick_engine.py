"""Mars Barn tick engine — advances all colonies by one sol.

Each tick:
1. Compute solar longitude from sol number
2. Generate Mars environment events
3. Compute resource production per colony
4. Tick each colony's population dynamics
5. Record global state snapshot
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from datetime import datetime, timezone

from . import colony as col
from . import environment as env


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(REPO_ROOT / "docs")))


def create_initial_state(seed: int = 42) -> dict:
    """Create initial Mars Barn state with 3 colonies."""
    colonies = [
        col.create_colony("ares_prime", 0),
        col.create_colony("hellas_basin", 1),
        col.create_colony("olympus_station", 2),
    ]
    return {
        "_meta": {
            "engine": "mars_barn",
            "version": "1.0.0",
            "created": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
        },
        "sol": 0,
        "total_population": sum(c["population"] for c in colonies),
        "colonies": colonies,
        "timeline": [],
        "events": [],
    }


def tick(state: dict, rng: random.Random) -> dict:
    """Advance the simulation by one sol.

    Mutates state in-place AND returns the sol snapshot.
    """
    state["sol"] += 1
    sol = state["sol"]
    ls = env.sol_to_ls(sol)
    colonies = state["colonies"]

    # Generate events
    colony_dicts = [
        {"population": c["population"], "carrying_capacity": c["carrying_capacity"]}
        for c in colonies
    ]
    events = env.generate_events(sol, colony_dicts, rng)
    if events:
        state["events"].extend(events)
        # Trim events log
        if len(state["events"]) > 500:
            state["events"] = state["events"][-400:]

    # Tick each colony
    snapshots = []
    for colony in colonies:
        active_storms = colony.get("active_storms", [])
        production = env.compute_resource_production(colony, ls, active_storms)
        snapshot = col.tick_colony(colony, sol, production, events, rng)
        snapshots.append(snapshot)

    # Global snapshot
    total_pop = sum(c["population"] for c in colonies)
    state["total_population"] = total_pop

    timeline_entry = {
        "sol": sol,
        "ls": round(ls, 1),
        "total_population": total_pop,
        "solar_flux": round(env.solar_flux(ls), 3),
        "colonies": snapshots,
    }
    state["timeline"].append(timeline_entry)

    return timeline_entry


def run_simulation(sols: int = 365, seed: int = 42, verbose: bool = False) -> dict:
    """Run the full simulation for N sols.

    Returns final state with complete timeline.
    """
    rng = random.Random(seed)
    state = create_initial_state(seed)

    if verbose:
        print(f"Mars Barn Terrarium — {sols} sol simulation")
        print(f"Seed: {seed}")
        print(f"Colonies: {', '.join(c['name'] for c in state['colonies'])}")
        print(f"Starting populations: {[c['population'] for c in state['colonies']]}")
        print("-" * 60)

    for sol in range(1, sols + 1):
        snapshot = tick(state, rng)

        if verbose and (sol % 50 == 0 or sol == 1 or sol == sols):
            pops = [s["population"] for s in snapshot["colonies"]]
            total = snapshot["total_population"]
            print(
                f"Sol {sol:4d} | Ls {snapshot['ls']:5.1f} | "
                f"Pop: {pops[0]:4d} / {pops[1]:4d} / {pops[2]:4d} | "
                f"Total: {total:5d} | "
                f"Flux: {snapshot['solar_flux']:.2f}"
            )

    if verbose:
        print("-" * 60)
        for colony in state["colonies"]:
            print(
                f"{colony['name']:20s} | Pop: {colony['population']:4d} | "
                f"Peak: {colony['peak_population']:4d} | "
                f"Births: {colony['births_total']:5d} | "
                f"Deaths: {colony['deaths_total']:5d} | "
                f"Tech: {colony['tech_level']:.2f}"
            )
        print(f"Total population: {state['total_population']}")

    return state


def save_state(state: dict) -> None:
    """Save state to state/mars.json and docs/mars_data.json."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    state_path = STATE_DIR / "mars.json"
    docs_path = DOCS_DIR / "mars_data.json"

    # Canonical state (pretty-printed)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(state_path)

    # Frontend data (compact, timeline only for charts)
    viz_data = {
        "_meta": state["_meta"],
        "sol": state["sol"],
        "total_population": state["total_population"],
        "colonies": [
            {
                "name": c["name"],
                "population": c["population"],
                "peak_population": c["peak_population"],
                "births_total": c["births_total"],
                "deaths_total": c["deaths_total"],
                "morale": round(c["morale"], 3),
                "tech_level": round(c["tech_level"], 3),
            }
            for c in state["colonies"]
        ],
        "timeline": state["timeline"],
    }
    tmp = docs_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(viz_data, separators=(",", ":")))
    tmp.rename(docs_path)


def load_state() -> dict | None:
    """Load existing state from disk, or None if not found."""
    state_path = STATE_DIR / "mars.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return None
