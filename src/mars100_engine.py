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
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100 import Mars100Engine, narrate_year


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

    print(f"Mars-100 — simulating {args.years} Martian years with seed {args.seed}")

    engine = Mars100Engine(seed=args.seed, total_years=args.years)
    result = engine.run()
    d = result.to_dict()
    summary = d["summary"]

    if not args.quiet:
        for yr in d["years"]:
            year = yr["year"]
            pop = sum(1 for c in yr["colonist_snapshots"] if c.get("alive") and not c.get("exiled"))
            events = yr["events"]
            event_name = events[0]["name"] if events else "calm"
            trials = yr.get("trials", [])
            trial_str = f" (trials: {len(trials)})" if trials else ""
            if year % 10 == 0 or year <= 5:
                print(f"  Year {year:>3}/{args.years}  Pop: {pop:>2}  Event: {event_name:<20}{trial_str}")

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Years survived:     {len(d['years'])}")
    print(f"  Final population:   {sum(1 for c in d['final_colonists'] if c.get('alive') and not c.get('exiled'))}")
    print(f"  Births:             {summary['total_births']}")
    print(f"  Deaths:             {summary['total_deaths']}")
    print(f"  Trials:             {summary['total_trials']}")
    print(f"  Trial exiles:       {summary['total_trial_exiles']}")
    print(f"  Sub-simulations:    {summary['total_subsims']}")
    print(f"  Governance changes: {summary['governance_changes']}")
    print(f"  Meta-awareness:     {summary['meta_awareness_events']} events")
    print(f"  Convergence trend:  {summary['convergence_trend']}")

    # Save state
    state_path = state_dir / "mars100.json"
    with open(state_path, "w") as f:
        json.dump(d, f, indent=2, default=str)
    print(f"\n  State -> {state_path}")

    # Save per-year deltas
    for yr in d["years"]:
        year_path = docs_dir / f"year-{yr['year']:03d}.json"
        with open(year_path, "w") as f:
            json.dump(yr, f, indent=2, default=str)

    # Save colonist files
    for c in d["final_colonists"]:
        with open(colonists_dir / f"{c['id']}.json", "w") as f:
            json.dump(c, f, indent=2, default=str)

    # Save dashboard data
    data_path = docs_dir / "data.json"
    with open(data_path, "w") as f:
        json.dump(d, f, indent=2, default=str)
    print(f"  Dashboard -> {data_path}")
    print(f"  Files: {len(d['years'])} year deltas + {len(d['final_colonists'])} colonists")


if __name__ == "__main__":
    main()
