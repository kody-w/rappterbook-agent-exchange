#!/usr/bin/env python3
"""Run the Mars-100 simulation and generate output data.

Usage:
    python scripts/run_mars100.py [--seed SEED] [--years YEARS] [--output DIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100 import Colony, run_simulation


def main() -> None:
    """Run the simulation and write output files."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Mars-100 simulation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--output", type=str, default="docs/mars-100")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = Path("state")
    state_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Mars-100 simulation: seed={args.seed}, years={args.years}")

    colony = Colony.genesis(seed=args.seed)
    deltas: list[dict] = []

    for year in range(1, args.years + 1):
        delta = colony.tick(year)
        deltas.append(delta)
        alive = sum(1 for c in colony.colonists.values() if c.alive)
        if alive == 0:
            print(f"Colony collapsed at year {year}")
            break

    print(f"Simulation complete: {len(deltas)} years simulated")

    # Save colony state
    state_path = state_dir / "mars100.json"
    with open(state_path, "w") as f:
        json.dump(colony.to_dict(), f, indent=2)
    print(f"Colony state saved to {state_path}")

    # Build dashboard data
    timeline: list[dict] = []
    for d in deltas:
        year_data = {
            "year": d["year"],
            "population": d["population"],
            "event": d["event"]["name"],
            "severity": d["event"]["severity"],
            "food": d.get("resources", {}).get("food", 0),
            "power": d.get("resources", {}).get("power", 0),
            "water": d.get("resources", {}).get("water", 0),
            "governance_count": len(d.get("governance", [])),
            "deaths": [death["name"] for death in d.get("deaths", [])],
            "summary": d.get("summary", "")[:200],
        }
        timeline.append(year_data)

    # Colonist data
    colonist_data: list[dict] = []
    for c in colony.colonists.values():
        colonist_data.append({
            "id": c.id, "name": c.name, "element": c.element,
            "role": c.role, "alive": c.alive,
            "health": round(c.health, 3),
            "morale": round(c.morale, 3),
            "years_alive": c.years_alive,
            "diary_excerpt": c.diary[-3:] if c.diary else [],
        })

    # Governance data
    gov_data: list[dict] = []
    for g in colony.governance:
        gov_data.append(g.to_dict())

    # Sub-sim data
    sub_sims: list[dict] = []
    for c in colony.colonists.values():
        for entry in c.sub_sim_log:
            sub_sims.append({"colonist": c.id, **entry})

    dashboard_data = {
        "seed": args.seed,
        "total_years": args.years,
        "final_population": sum(1 for c in colony.colonists.values() if c.alive),
        "total_deaths": len(colony.archived_souls),
        "governance_type": colony.governance_type,
        "leader": colony.leader,
        "factions": colony.factions,
        "meta_insights": colony.meta_insights,
        "timeline": timeline,
        "colonists": colonist_data,
        "governance": gov_data,
        "archived_souls": colony.archived_souls,
        "sub_sims": sub_sims,
    }

    data_path = output_dir / "data.json"
    with open(data_path, "w") as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"Dashboard data saved to {data_path}")
    print(f"Final: {dashboard_data['final_population']} alive, "
          f"{dashboard_data['total_deaths']} dead, "
          f"{len(gov_data)} governance records, "
          f"{len(sub_sims)} sub-sims")


if __name__ == "__main__":
    main()
