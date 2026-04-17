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

from src.mars100 import run_simulation


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

    result = run_simulation(years=args.years, seed=args.seed)
    colony = result["colony"]
    deltas = result["deltas"]
    summary = result["summary"]

    if not args.quiet:
        for delta in deltas:
            year = delta["year"]
            pop = delta["population"]
            event_id = delta["event"]["id"] if delta["event"] else "none"
            if year % 10 == 0 or year <= 5:
                gov = " ".join(f"[{g}]" for g in delta["governance_results"])
                subsims = f" (sub-sims: {len(delta['sub_sims'])})" if delta["sub_sims"] else ""
                meta = f" * {delta['meta_awareness'][:60]}..." if delta["meta_awareness"] else ""
                print(f"  Year {year:>3}/{args.years}  Pop: {pop:>2}  Event: {event_id:<20}{subsims}{gov}{meta}")

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Years survived:     {summary['years_survived']}")
    print(f"  Final population:   {summary['final_population']}")
    print(f"  Births:             {summary['total_births']}")
    print(f"  Deaths:             {summary['total_deaths']}")
    print(f"  Sub-simulations:    {summary['total_sub_simulations']}")
    print(f"  Governance:         {summary['governance_system']}")
    if summary["meta_awareness_events"]:
        print(f"  Meta-awareness:     {len(summary['meta_awareness_events'])} events")
        for e in summary["meta_awareness_events"][:3]:
            print(f"    - {e[:100]}...")
    if summary["constitutional_amendments"]:
        print(f"  Amendments:         {len(summary['constitutional_amendments'])}")
        for a in summary["constitutional_amendments"]:
            print(f"    Year {a['year']}: {a['text']}")

    # Save state
    state_path = state_dir / "mars100.json"
    with open(state_path, "w") as f:
        json.dump(colony, f, indent=2, default=str)
    print(f"\n  State -> {state_path}")

    # Save per-year deltas
    for delta in deltas:
        year_path = docs_dir / f"year-{delta['year']}.json"
        with open(year_path, "w") as f:
            json.dump(delta, f, indent=2, default=str)

    # Save colonist files
    for c in colony["colonists"]:
        with open(colonists_dir / f"{c['id']}.json", "w") as f:
            json.dump(c, f, indent=2, default=str)
    for soul in colony["dead_souls"]:
        with open(colonists_dir / f"{soul['id']}-soul.json", "w") as f:
            json.dump(soul, f, indent=2, default=str)

    # Save dashboard data
    data_path = docs_dir / "data.json"
    dashboard_data = {
        "_meta": {
            "engine": "mars-100", "version": "1.0",
            "years": args.years, "seed": args.seed,
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "colony": colony,
        "deltas": deltas,
    }
    with open(data_path, "w") as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    print(f"  Dashboard -> {data_path}")
    print(f"  Files: {len(deltas)} year deltas + {len(colony['colonists'])} colonists")


if __name__ == "__main__":
    main()
