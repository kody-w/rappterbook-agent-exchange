"""mars100_runner.py — CLI runner for Mars-100 Recursive Colony Simulation.

Runs the 100-year simulation, writes per-year deltas, colonist state,
and summary to docs/mars-100/ for the frontend dashboard.

Usage:
    python src/mars100_runner.py --years 100
    python src/mars100_runner.py --years 10 --seed 99 --quiet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100 import Mars100Simulation, ColonyState


def main() -> None:
    """Entry point for the Mars-100 runner."""
    parser = argparse.ArgumentParser(description="Mars-100 Recursive Colony Simulation")
    parser.add_argument("--years", type=int, default=100, help="Number of Martian years")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-year output")
    parser.add_argument("--docs-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir) if args.docs_dir else REPO_ROOT / "docs" / "mars-100"
    docs_dir.mkdir(parents=True, exist_ok=True)
    colonists_dir = docs_dir / "colonists"
    colonists_dir.mkdir(parents=True, exist_ok=True)
    years_dir = docs_dir / "years"
    years_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mars-100 — simulating {args.years} Martian years (seed={args.seed})...")
    print()

    sim = Mars100Simulation(seed=args.seed, max_years=args.years)

    def on_year(year: int, state: ColonyState, delta: dict) -> None:
        """Per-year callback: print status and save year delta."""
        if not args.quiet:
            alive = len(state.alive_colonists())
            dead = len(state.dead_colonists())
            morale = state.morale
            event = delta["event"]["type"]
            subs = delta["sub_sims_this_year"]
            tf = state.terraforming_progress * 100

            status = f"  Year {year:>3}/{args.years}"
            status += f"  Alive: {alive}  Dead: {dead}"
            status += f"  Morale: {morale:.2f}"
            status += f"  TF: {tf:.2f}%"
            status += f"  Event: {event}"
            if subs > 0:
                status += f"  [sub-sims: {subs}]"
            if state.collapsed:
                status += "  *** COLLAPSED ***"
            print(status)

        # Save year delta
        year_path = years_dir / f"year-{year:03d}.json"
        _atomic_write(year_path, delta)

    results = sim.run(callback=on_year)

    print()
    print("=" * 70)
    print("MARS-100 SIMULATION COMPLETE")
    print("=" * 70)

    summary = results["summary"]
    print(f"\n  Years simulated: {summary['years_simulated']}")
    print(f"  Collapsed: {summary['collapsed']}", end="")
    if summary["collapse_reason"]:
        print(f" ({summary['collapse_reason']})")
    else:
        print()
    print(f"  Alive: {summary['alive_count']}  Dead: {summary['dead_count']}")
    print(f"  Final morale: {summary['final_morale']:.3f}")
    print(f"  Terraforming: {summary['terraforming_progress']*100:.3f}%")
    print(f"  Total proposals: {summary['total_proposals']} "
          f"(passed: {summary['passed_proposals']}, failed: {summary['failed_proposals']})")
    print(f"  Total sub-simulations: {summary['total_sub_sims']}")
    print(f"  Meta-aware colonists: {summary['meta_aware_colonists']}")

    if summary["deaths"]:
        print("\n  Deaths:")
        for d in summary["deaths"]:
            print(f"    Year {d['year']}: {d['name']} ({d['cause']})")

    if summary["governance_evolution"]:
        print("\n  Governance evolution:")
        for phase in summary["governance_evolution"]:
            print(f"    {phase['decade']}: {phase['proposals']} proposals, "
                  f"{phase['passed']} passed — dominant: {phase['dominant_type']}")

    if summary["meta_insights"]:
        print(f"\n  Meta-insights from depth-3 sub-sims: {len(summary['meta_insights'])}")
        for insight in summary["meta_insights"][:3]:
            print(f"    Year {insight['year']}: {insight['label']} (by {insight['source']})")

    print()

    # Save colonist state files
    for colonist in results["state"]["colonists"]:
        cpath = colonists_dir / f"{colonist['id']}.json"
        _atomic_write(cpath, colonist)
    print(f"Colonist files: {colonists_dir}/")

    # Save summary
    summary_path = docs_dir / "summary.json"
    _atomic_write(summary_path, results)
    print(f"Summary:        {summary_path}")

    # Save compact data for frontend
    data_path = docs_dir / "data.json"
    compact = _compact_results(results)
    _atomic_write(data_path, compact, compact=True)
    print(f"Frontend data:  {data_path}")

    print()
    print(f"View at: https://kody-w.github.io/rappterbook-agent-exchange/mars-100/")


def _compact_results(results: dict) -> dict:
    """Build compact frontend data matching docs/mars-100/index.html schema."""
    deltas = results.get("deltas", [])
    return {
        "_meta": results["_meta"],
        "summary": results["summary"],
        "timeline": [
            {
                "year": d["year"],
                "event": d["event"]["type"],
                "severity": round(d["event"].get("severity", 0.5), 3),
                "morale": d["morale"],
                "alive": d["alive_count"],
                "dead": d["dead_count"],
                "terraforming": d["terraforming"],
                "governance_form": d.get("governance_form", "anarchy"),
                "value_convergence": round(d.get("value_convergence", 0.0), 4),
                "sub_sims": d["sub_sims_this_year"],
            }
            for d in deltas
        ],
        "colonists": results["state"]["colonists"],
    }


def _atomic_write(path: Path, data: dict, compact: bool = False) -> None:
    """Atomic JSON write with fsync."""
    tmp = path.with_suffix(".tmp")
    content = json.dumps(data, separators=(",", ":")) if compact else json.dumps(data, indent=2)
    tmp.write_text(content)
    tmp.rename(path)


if __name__ == "__main__":
    main()
