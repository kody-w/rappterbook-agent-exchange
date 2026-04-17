#!/usr/bin/env python3
"""
Mars-100 Engine — run the recursive colony simulation and publish results.

Usage:
    python src/mars100_engine.py --years 100
    python src/mars100_engine.py --years 50 --seed 99
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100 import Mars100Engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Mars-100 recursive colony simulation")
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--docs-dir", type=str, default=None)
    args = parser.parse_args()

    state_dir = Path(args.state_dir) if args.state_dir else REPO_ROOT / "state"
    docs_dir = Path(args.docs_dir) if args.docs_dir else REPO_ROOT / "docs" / "mars-100"
    state_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    colonists_dir = docs_dir / "colonists"
    colonists_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mars-100 v3.0 — simulating {args.years} Martian years (seed {args.seed})")

    engine = Mars100Engine(seed=args.seed, total_years=args.years)
    result = engine.run()
    data = result.to_dict()
    summary = data["summary"]

    if not args.quiet:
        for yr in data["years"]:
            year_num = yr["year"]
            pop = len(yr.get("colonist_snapshots", []))
            events = yr.get("events", [])
            event_id = events[0].get("name", events[0].get("id", "?")) if events else "none"
            if year_num % 10 == 0 or year_num <= 5:
                subsims = f" (sub-sims: {len(yr.get('subsim_log', []))})" if yr.get("subsim_log") else ""
                culture = ""
                cs = yr.get("culture_summary", {})
                if cs.get("memes_created"):
                    culture = f" 🧬{cs['memes_created']}new/{cs['active_memes']}active"
                print(f"  Year {year_num:>3}/{args.years}  Pop: {pop:>2}  Event: {event_id:<20}{subsims}{culture}")

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Years simulated:    {len(data['years'])}")
    print(f"  Final population:   {len(data['final_colonists'])}")
    print(f"  Births:             {summary['total_births']}")
    print(f"  Deaths:             {summary['total_deaths']}")
    print(f"  Sub-simulations:    {summary['total_subsims']}")
    print(f"  Governance changes: {summary['governance_changes']}")
    print(f"  Convergence:        {summary['convergence_trend']}")
    print(f"  Memes created:      {summary['total_memes_created']}")
    print(f"  Active memes:       {summary['final_active_memes']}")
    if summary.get("promoted_insights"):
        print(f"  Promoted insights:  {summary['promoted_insights']}")

    # Save canonical state
    state_path = state_dir / "mars100.json"
    with open(state_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  State -> {state_path}")

    # Save per-year deltas
    for yr in data["years"]:
        year_path = docs_dir / f"year-{yr['year']}.json"
        with open(year_path, "w") as f:
            json.dump(yr, f, indent=2, default=str)

    # Save colonist files
    for c in data["final_colonists"]:
        with open(colonists_dir / f"{c['id']}.json", "w") as f:
            json.dump(c, f, indent=2, default=str)

    # Save dashboard data
    data_path = docs_dir / "data.json"
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Dashboard -> {data_path}")
    print(f"  Files: {len(data['years'])} year deltas + {len(data['final_colonists'])} colonists")


if __name__ == "__main__":
    main()
