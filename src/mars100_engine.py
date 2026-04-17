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

from src.mars100 import (
    run_simulation, ENGINE_VERSION,
    format_amendment_proposal,
    narrate_year, generate_diary_entries, generate_final_report,
)


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

    print(f"Mars-100 v{ENGINE_VERSION} — simulating {args.years} Martian years with seed {args.seed}")

    result = run_simulation(years=args.years, seed=args.seed)
    result_dict = result.to_dict()
    summary = result_dict["summary"]

    if not args.quiet:
        for yr in result.years:
            year = yr.year
            pop = len([c for c in yr.colonist_snapshots if c.get("alive") and not c.get("exiled")])
            subsims = len(yr.subsim_log)
            births = len(yr.births)
            deaths = len(yr.deaths)
            conv = yr.convergence_score
            if year % 10 == 0 or year <= 5:
                extra = ""
                if births:
                    extra += f" +{births}born"
                if deaths:
                    extra += f" -{deaths}dead"
                if subsims:
                    extra += f" ({subsims} sub-sims)"
                if yr.meta_insights:
                    extra += f" [{len(yr.meta_insights)} insights]"
                print(f"  Year {year:>3}/{args.years}  Pop: {pop:>2}  Conv: {conv:.3f}{extra}")

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Engine version:     {ENGINE_VERSION}")
    print(f"  Years survived:     {len(result.years)}")
    print(f"  Final population:   {len([c for c in result.final_colonists if c.get('alive') and not c.get('exiled')])}")
    print(f"  Births:             {result.total_births}")
    print(f"  Deaths:             {result.total_deaths}")
    print(f"  Sub-simulations:    {result.total_subsims}")
    print(f"  Governance changes: {result.governance_changes}")
    print(f"  Meta-awareness:     {result.meta_events} events")
    print(f"  Meta-insights:      {len(result.meta_insights)}")
    print(f"  Final cohesion:     {result.final_cohesion:.2%}")
    conv = result.convergence_summary
    print(f"  Convergence:        {conv['initial']:.3f} → {conv['final']:.3f} ({conv['trend']})")
    if result.proposed_amendment:
        print(f"\n  ** PROPOSED AMENDMENT **")
        print(f"  Type: {result.proposed_amendment['type']}")
        print(f"  Strength: {result.proposed_amendment['strength']:.2f}")
        print(f"  Text: {result.proposed_amendment['proposed_amendment'][:80]}...")

    # Save state
    state_data = {
        "year": len(result.years),
        "colonists": result.final_colonists,
        "resources": result.final_resources,
        "governance": result.final_governance,
        "proposals_pending": [],
        "dead_souls": [c for c in result.final_colonists if not c.get("alive")],
        "sub_sim_log": [s for y in result.years for s in y.subsim_log],
        "event_history": [e for y in result.years for e in y.events],
        "_meta": {
            "engine": "mars-100", "version": ENGINE_VERSION,
            "seed": args.seed,
            "created": datetime.now(timezone.utc).isoformat(),
        },
    }
    state_path = state_dir / "mars100.json"
    with open(str(state_path) + ".tmp", "w") as f:
        json.dump(state_data, f, indent=2, default=str)
    Path(str(state_path) + ".tmp").rename(state_path)
    print(f"\n  State -> {state_path}")

    # Save per-year deltas
    import random as _rng_mod
    rng = _rng_mod.Random(args.seed)
    for yr in result.years:
        year_data = yr.to_dict()
        # Add diary entries
        year_data["diary_entries"] = generate_diary_entries(
            year_data, yr.colonist_snapshots, rng, count=3)
        year_path = docs_dir / f"year-{yr.year}.json"
        with open(str(year_path) + ".tmp", "w") as f:
            json.dump(year_data, f, indent=2, default=str)
        Path(str(year_path) + ".tmp").rename(year_path)

    # Save colonist files
    for c in result.final_colonists:
        cpath = colonists_dir / f"{c.get('id', 'unknown')}.json"
        with open(str(cpath) + ".tmp", "w") as f:
            json.dump(c, f, indent=2, default=str)
        Path(str(cpath) + ".tmp").rename(cpath)
    for c in result.final_colonists:
        if not c.get("alive"):
            spath = colonists_dir / f"{c.get('id', 'unknown')}-soul.json"
            with open(str(spath) + ".tmp", "w") as f:
                json.dump(c, f, indent=2, default=str)
            Path(str(spath) + ".tmp").rename(spath)

    # Save dashboard data
    data_path = docs_dir / "data.json"
    with open(str(data_path) + ".tmp", "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    Path(str(data_path) + ".tmp").rename(data_path)
    print(f"  Dashboard -> {data_path}")
    print(f"  Files: {len(result.years)} year deltas + {len(result.final_colonists)} colonists")


if __name__ == "__main__":
    main()
