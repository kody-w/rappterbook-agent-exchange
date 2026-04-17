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

from src.mars100.engine import Mars100Engine
from src.mars100.narrator import narrate_year, generate_diary_entries, generate_final_report


def _write_json(path: Path, data: object) -> None:
    """Atomic JSON write with .tmp rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)


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
    sim_result = engine.run()
    result_dict = sim_result.to_dict()
    summary = result_dict["summary"]

    if not args.quiet:
        for yr in result_dict["years"]:
            year = yr["year"]
            alive = sum(1 for c in yr["colonist_snapshots"] if c.get("alive") and not c.get("exiled"))
            events = yr.get("events", [])
            ev_name = events[0]["name"] if events else "calm"
            factions = len(yr.get("faction_state", {}).get("factions", {}))
            if year % 10 == 0 or year <= 5:
                subsims = f" (sub-sims: {len(yr.get('subsim_log', []))})" if yr.get("subsim_log") else ""
                fac_str = f" [{factions} factions]" if factions else ""
                print(f"  Year {year:>3}/{args.years}  Pop: {alive:>2}  Event: {ev_name:<20}{subsims}{fac_str}")

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Years simulated:    {len(result_dict['years'])}")
    print(f"  Final governance:   {result_dict['final_governance'].get('gov_type', '?')}")
    print(f"  Deaths:             {summary.get('total_deaths', 0)}")
    print(f"  Births:             {summary.get('total_births', 0)}")
    print(f"  Exiles:             {summary.get('total_exiles', 0)}")
    print(f"  Sub-simulations:    {summary.get('total_subsims', 0)}")
    print(f"  Governance changes: {summary.get('governance_changes', 0)}")
    print(f"  Meta-awareness:     {summary.get('meta_awareness_events', 0)} events")
    print(f"  Convergence trend:  {summary.get('convergence_trend', '?')}")
    print(f"  Final cohesion:     {summary.get('final_cohesion', 0):.0%}")
    promoted = result_dict.get("promoted_insights", [])
    if promoted:
        print(f"  Promoted insights:  {len(promoted)}")
        for ins in promoted[:3]:
            print(f"    - {ins.get('theme', '?')[:80]}")

    # Save canonical state
    state_data = {
        "_meta": {
            "engine": "mars-100", "version": "2.0",
            "seed": args.seed,
            "created": datetime.now(timezone.utc).isoformat(),
        },
        "final_colonists": result_dict["final_colonists"],
        "final_resources": result_dict["final_resources"],
        "final_governance": result_dict["final_governance"],
        "promoted_insights": promoted,
        "summary": summary,
    }
    _write_json(state_dir / "mars100.json", state_data)
    print(f"\n  State -> {state_dir / 'mars100.json'}")

    # Save per-year files
    import random as _rng_mod
    narr_rng = _rng_mod.Random(args.seed)
    for yr in result_dict["years"]:
        year_path = docs_dir / f"year-{yr['year']}.json"
        _write_json(year_path, yr)

    # Save colonist files
    for c in result_dict["final_colonists"]:
        cid = c["id"]
        _write_json(colonists_dir / f"{cid}.json", c)
        if not c.get("alive"):
            _write_json(colonists_dir / f"{cid}-soul.json", c)

    # Save dashboard data
    data_path = docs_dir / "data.json"
    dashboard_data = {
        "_meta": {
            "engine": "mars-100", "version": "2.0",
            "years": args.years, "seed": args.seed,
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "final_colonists": result_dict["final_colonists"],
        "final_resources": result_dict["final_resources"],
        "final_governance": result_dict["final_governance"],
        "promoted_insights": promoted,
        "years": {str(yr["year"]): yr for yr in result_dict["years"]},
    }
    _write_json(data_path, dashboard_data)

    # Generate final report
    report = generate_final_report(result_dict)
    report_path = docs_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    alive_count = sum(1 for c in result_dict["final_colonists"]
                      if c.get("alive") and not c.get("exiled"))
    dead_count = sum(1 for c in result_dict["final_colonists"] if not c.get("alive"))
    print(f"  Dashboard -> {data_path}")
    print(f"  Report -> {report_path}")
    print(f"  Files: {len(result_dict['years'])} year files, "
          f"{alive_count} alive + {dead_count} dead colonists")


if __name__ == "__main__":
    main()
