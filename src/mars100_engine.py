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

from src.mars100 import Mars100Engine, run_simulation, crossover_analysis
from src.mars100.narrator import generate_final_report


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically via tmp + rename."""
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

    # --- Run simulation via compatibility shim ---
    result = run_simulation(years=args.years, seed=args.seed)
    colony = result["colony"]
    deltas = result["deltas"]
    summary = result["summary"]
    full_result = result["full_result"]

    if not args.quiet:
        for delta in deltas:
            year = delta["year"]
            pop = delta["population"]
            ev = delta.get("event")
            event_id = ev.get("name", "none") if isinstance(ev, dict) else "none"
            if year % 10 == 0 or year <= 5:
                gov = " ".join(f"[{g}]" for g in delta.get("governance_results", []))
                subsims = f" (sub-sims: {len(delta.get('sub_sims', []))})" if delta.get("sub_sims") else ""
                meta = delta.get("meta_awareness", "")
                meta_str = f" * {meta[:60]}..." if meta else ""
                print(f"  Year {year:>3}/{args.years}  Pop: {pop:>2}  Event: {event_id:<20}{subsims}{gov}{meta_str}")

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total deaths:       {summary.get('total_deaths', 0)}")
    print(f"  Total exiles:       {summary.get('total_exiles', 0)}")
    print(f"  Total births:       {summary.get('total_births', 0)}")
    print(f"  Sub-simulations:    {summary.get('total_subsims', 0)}")
    print(f"  Governance changes: {summary.get('governance_changes', 0)}")
    print(f"  Meta events:        {summary.get('meta_awareness_events', 0)}")
    print(f"  Final cohesion:     {summary.get('final_cohesion', 0):.0%}")

    # --- Crossover analysis ---
    print(f"\n{'='*60}")
    print("CROSSOVER ANALYSIS — Constitutional Bridge")
    print(f"{'='*60}")

    crossover = crossover_analysis(full_result)
    cs = crossover["summary"]
    print(f"  Governance patterns: {cs['patterns_found']}")
    print(f"  Convergences found:  {cs['convergences_found']}")
    print(f"  Strongest pattern:   {cs['strongest_pattern']}")
    print(f"  Amendment title:     {cs['amendment_title']}")
    print(f"  Confidence:          {cs['amendment_confidence']:.0%}")

    amendment = crossover["amendment_proposal"]
    print(f"\n  PROPOSED AMENDMENT {amendment['number']}:")
    print(f"  \"{amendment['title']}\"")
    for line in amendment["text"].split(". "):
        print(f"    {line.strip()}.")
    print(f"\n  LisPy encoding:")
    print(f"    {amendment['lispy_expression'][:120]}...")

    # --- Save state ---
    state_path = state_dir / "mars100.json"
    _atomic_write(state_path, colony)
    print(f"\n  State -> {state_path}")

    # Save per-year deltas
    for delta in deltas:
        _atomic_write(docs_dir / f"year-{delta['year']}.json", delta)

    # Save colonist files
    for c in colony.get("colonists", []):
        cid = c.get("id", "unknown")
        _atomic_write(colonists_dir / f"{cid}.json", c)
    for soul in colony.get("dead_souls", []):
        sid = soul.get("id", "unknown")
        _atomic_write(colonists_dir / f"{sid}-soul.json", soul)

    # Save dashboard data
    dashboard_data = {
        "_meta": {
            "engine": "mars-100", "version": "2.0",
            "years": args.years, "seed": args.seed,
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "crossover": crossover["summary"],
    }
    _atomic_write(docs_dir / "data.json", dashboard_data)

    # Save crossover analysis
    _atomic_write(docs_dir / "crossover.json", crossover)
    print(f"  Crossover -> {docs_dir / 'crossover.json'}")

    # Save amendment proposal
    _atomic_write(docs_dir / "amendment_proposal.json", amendment)
    print(f"  Amendment -> {docs_dir / 'amendment_proposal.json'}")

    # Generate final report
    report = generate_final_report(full_result)
    report_path = docs_dir / "report.md"
    tmp = report_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(report)
    tmp.rename(report_path)
    print(f"  Report -> {report_path}")

    print(f"\n  Files: {len(deltas)} year deltas + "
          f"{len(colony.get('colonists', []))} colonists + "
          f"{len(colony.get('dead_souls', []))} souls")


if __name__ == "__main__":
    main()
