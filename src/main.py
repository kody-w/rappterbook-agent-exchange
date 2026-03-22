#!/usr/bin/env python3
"""Mars Barn Terrarium — CLI entry point.

Usage:
    python src/main.py --sols 365
    python src/main.py --sols 365 --seed 42 --verbose
    python src/main.py --sols 668 --seed 7 --output state/terrarium.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tick_engine import run_simulation


def write_viz_data(state: dict, docs_dir: Path) -> None:
    """Write visualization data to docs/ for GitHub Pages."""
    colonies = state["colonies"]
    total_hist = state["total_population_history"]

    viz = {
        "sol": state["sol"],
        "seed": state["_meta"].get("seed", 42),
        "generated_at": state["_meta"].get("last_tick", ""),
        "stats": state["stats"],
        "curves": {
            "sols": list(range(len(total_hist))),
            "total": total_hist,
        },
        "colonies": [],
    }

    for colony in colonies:
        cfg = colony["config"]
        viz["colonies"].append({
            "name": cfg["name"],
            "strategy": cfg["strategy"],
            "population": colony["population"],
            "population_history": colony["population_history"],
            "morale_history": colony["morale_history"],
            "total_births": colony["total_births"],
            "total_deaths": colony["total_deaths"],
            "total_migrants_in": colony.get("total_migrants_in", 0),
            "total_migrants_out": colony.get("total_migrants_out", 0),
            "final_resources": colony["resources"],
        })

    # Write compact JSON for the frontend
    viz_path = docs_dir / "terrarium_data.json"
    tmp = viz_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(viz, separators=(",", ":")))
    tmp.rename(viz_path)
    print(f"Wrote visualization data to {viz_path}")


def main() -> None:
    """Run the Mars Barn Terrarium simulation."""
    parser = argparse.ArgumentParser(
        description="Mars Barn Terrarium — population dynamics simulation"
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
        "--output", type=str, default=None,
        help="Output state file path (default: state/terrarium.json)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress every 50 sols"
    )
    args = parser.parse_args()

    print(f"🔴 Mars Barn Terrarium — {args.sols} sols, seed={args.seed}")
    print(f"   3 colonies: Olympus Greenhouse | Valles Caverns | Hellas Basin Hub")
    print()

    state = run_simulation(sols=args.sols, seed=args.seed, verbose=args.verbose)

    # Save full state
    state_dir = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))
    output_path = Path(args.output) if args.output else state_dir / "terrarium.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(output_path)
    print(f"\nSaved state to {output_path}")

    # Write viz data
    docs_dir = Path(os.environ.get("DOCS_DIR", str(REPO_ROOT / "docs")))
    docs_dir.mkdir(parents=True, exist_ok=True)
    write_viz_data(state, docs_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)

    for colony in state["colonies"]:
        cfg = colony["config"]
        pop_hist = colony["population_history"]
        print(f"\n  {cfg['name']} ({cfg['strategy']})")
        print(f"    Population: {pop_hist[0]} → {pop_hist[-1]} "
              f"(peak {max(pop_hist)}, low {min(pop_hist)})")
        print(f"    Births: {colony['total_births']}  Deaths: {colony['total_deaths']}")
        print(f"    Migrants in: {colony.get('total_migrants_in', 0)}  "
              f"out: {colony.get('total_migrants_out', 0)}")
        res = colony["resources"]
        print(f"    Resources: O2={res['o2_days']:.0f}d  H2O={res['h2o_days']:.0f}d  "
              f"Food={res['food_days']:.0f}d  Power={res['power_kwh']:.0f}kWh")

    total = state["total_population_history"]
    stats = state["stats"]
    print(f"\n  TOTAL: {total[0]} → {total[-1]} (peak {stats['peak_population']})")
    print(f"  Dust storms: {stats['dust_storms']}  "
          f"Global storms: {stats['global_storms']}  "
          f"Catastrophes: {stats['catastrophes']}")
    print()


if __name__ == "__main__":
    main()
