"""CLI runner for Mars-100 simulation.

Usage:
    python src/run_mars100.py [--seed N] [--years N] [--output DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent dir so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100 import run_simulation


def main() -> None:
    """Run Mars-100 simulation and generate output."""
    parser = argparse.ArgumentParser(description="Mars-100 Recursive Colony Simulation")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--years", type=int, default=100, help="Simulation years (default: 100)")
    parser.add_argument("--output", type=str, default="docs/mars-100", help="Output directory")
    args = parser.parse_args()

    print(f"Running Mars-100: seed={args.seed}, years={args.years}")
    result = run_simulation(years=args.years, seed=args.seed)
    summary = result["summary"]

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "years").mkdir(exist_ok=True)
    (outdir / "colonists").mkdir(exist_ok=True)

    for delta in result["deltas"]:
        yr = delta["year"]
        with open(outdir / "years" / f"year-{yr:03d}.json", "w") as f:
            json.dump(delta, f, indent=2, default=str)

    colony = result["colony"]
    for c in colony["colonists"]:
        with open(outdir / "colonists" / f"colonist-{c['id']}.json", "w") as f:
            json.dump(c, f, indent=2, default=str)

    for soul in colony["dead_souls"]:
        sid = soul.get("id", "unknown")
        with open(outdir / "colonists" / f"soul-{sid}.json", "w") as f:
            json.dump(soul, f, indent=2, default=str)

    sub_sims = []
    for d in result["deltas"]:
        for ss in d["sub_sims"]:
            ss["year"] = d["year"]
            sub_sims.append(ss)
    with open(outdir / "subsim_log.json", "w") as f:
        json.dump(sub_sims, f, indent=2, default=str)

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(outdir / "state.json", "w") as f:
        json.dump(colony, f, indent=2, default=str)

    n_files = sum(1 for _ in outdir.rglob("*.json"))
    print(f"\nSimulation complete:")
    print(f"  Years survived: {summary['years_survived']}")
    print(f"  Final population: {summary['final_population']}")
    print(f"  Deaths: {summary['total_deaths']}, Births: {summary.get('total_births', 0)}")
    print(f"  Sub-simulations: {summary['total_sub_simulations']}")
    print(f"  Governance events: {summary['total_governance']}")
    print(f"  Constitutional amendments: {len(summary.get('constitutional_amendments', []))}")
    print(f"  Meta-awareness events: {summary['total_meta_insights']}")
    print(f"  Output: {n_files} JSON files in {outdir}/")


if __name__ == "__main__":
    main()
