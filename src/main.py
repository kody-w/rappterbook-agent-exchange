#!/usr/bin/env python3
"""Mars Barn Terrarium — run the colony simulation.

Usage:
    python src/main.py --sols 365          # Run 365 sols, 3 colonies
    python src/main.py --sols 100 --seed 7 # Custom seed
    python src/main.py --reset --sols 365  # Fresh start

Outputs:
    state/mars.json       — full simulation state
    docs/mars_data.json   — compact data for visualization
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mars.tick_engine import create_world, tick, save_world, load_world


def run_simulation(sols: int, seed: int = 42, reset: bool = False,
                   quiet: bool = False) -> dict:
    """Run the Mars colony simulation for N sols.

    Args:
        sols: Number of sols to simulate.
        seed: Random seed for reproducibility.
        reset: If True, start from scratch.
        quiet: If True, suppress progress output.

    Returns:
        Final world state dict.
    """
    if reset:
        world = create_world(seed=seed)
    else:
        world = load_world()
        if world["sol"] == 0 or world.get("seed") != seed:
            world = create_world(seed=seed)

    start_sol = world["sol"]
    target_sol = start_sol + sols

    if not quiet:
        print(f"[mars-barn] Starting simulation: sol {start_sol} → {target_sol}")
        print(f"[mars-barn] Seed: {seed}")
        print(f"[mars-barn] Colonies:")
        for c in world["colonies"]:
            print(f"  • {c['name']}: {c['population']} crew "
                  f"({c['location']['latitude']:.1f}°, "
                  f"alt {c['location']['altitude_km']:.1f} km)")
        print()

    for i in range(sols):
        world = tick(world)
        current_sol = world["sol"]

        if not quiet and (current_sol % 50 == 0 or i == sols - 1):
            colonies = world["colonies"]
            pops = ", ".join(
                f"{c['name']}: {c['population']}"
                for c in colonies
            )
            total = sum(c["population"] for c in colonies)
            ls = world["environment_log"][-1]["ls"] if world["environment_log"] else 0
            storm = " 🌪️" if (world["environment_log"] and
                              world["environment_log"][-1].get("dust_storm")) else ""
            print(f"  Sol {current_sol:>4} (Ls {ls:>5.1f}){storm}: "
                  f"total={total:>4}  [{pops}]")

    save_world(world)

    if not quiet:
        print()
        summary = world["summary"]
        print(f"[mars-barn] Simulation complete: {summary['sols_simulated']} sols")
        print(f"[mars-barn] Peak population: {summary['peak_population']}")
        print(f"[mars-barn] Total births: {summary['total_births']}")
        print(f"[mars-barn] Total deaths: {summary['total_deaths']}")
        print(f"[mars-barn] Dust storm sols: {summary['dust_storm_sols']}")
        print(f"[mars-barn] Saved to state/mars.json + docs/mars_data.json")

    return world


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mars Barn Terrarium — colony population simulation"
    )
    parser.add_argument(
        "--sols", type=int, default=365,
        help="Number of sols to simulate (default: 365)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Force fresh simulation start"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    run_simulation(
        sols=args.sols,
        seed=args.seed,
        reset=args.reset,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
