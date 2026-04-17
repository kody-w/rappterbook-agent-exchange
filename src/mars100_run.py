#!/usr/bin/env python3
"""mars100_run.py — CLI runner for Mars-100 recursive colony simulation.

Usage:
    python src/mars100_run.py --years 100 --seed 42
    python src/mars100_run.py --years 10 --quiet
    python src/mars100_run.py --years 100 --output-dir docs/mars-100
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

from src.mars100_sim import Mars100Simulation


def main() -> None:
    """Run Mars-100 simulation and write outputs."""
    parser = argparse.ArgumentParser(description="Mars-100 recursive colony simulation")
    parser.add_argument("--years", type=int, default=100, help="Years to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "docs" / "mars-100"
    output_dir.mkdir(parents=True, exist_ok=True)
    colonist_dir = output_dir / "colonists"
    colonist_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Mars-100 — {args.years} years, seed {args.seed}")
        print("=" * 50)

    sim = Mars100Simulation(seed=args.seed, total_years=args.years)
    results = sim.run()

    # Write main data file
    _atomic_write(output_dir / "data.json", results)

    # Write per-colonist state
    for colonist in results["colonists"]:
        _atomic_write(colonist_dir / f"{colonist['id']}.json", colonist)

    # Write per-year deltas
    for delta in results["year_deltas"]:
        year = delta["year"]
        _atomic_write(output_dir / f"year-{year:03d}.json", delta)

    # Write archived soul files
    for soul in results["archives"]:
        soul_path = colonist_dir / f"{soul['id']}-soul.json"
        _atomic_write(soul_path, soul)

    if not args.quiet:
        s = results["summary"]
        print(f"\nSimulation complete: {s['years_survived']} years")
        print(f"  Survivors: {s['colonists_end']}/{s['colonists_start']}")
        print(f"  Deaths: {s['deaths']}")
        print(f"  Laws enacted: {s['laws_enacted']}")
        print(f"  Governance: {', '.join(s['governance_patterns'])}")
        print(f"  Sub-sims run: {s['total_sub_sims']}")
        print(f"  Depth-3 sims: {s['depth_3_sims']}")
        print(f"  Final food: {s['final_resources']['food']:.0f}")
        print(f"  Final morale: {s['final_resources']['morale']:.2f}")
        print(f"  Terraforming: {s['final_resources']['terraforming']:.1%}")
        print(f"\nOutput written to: {output_dir}")


def _atomic_write(path: Path, data: dict) -> None:
    """Atomic JSON write: tmp file → fsync → rename."""
    tmp = path.with_suffix(".json.tmp")
    content = json.dumps(data, indent=2, ensure_ascii=False)
    tmp.write_text(content)
    # fsync for durability
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.rename(path)


if __name__ == "__main__":
    main()
