"""
mars100_runner.py — CLI runner for the Mars-100 recursive colony simulation.

Runs the full simulation and writes structured output for the dashboard.
The output of year N is the input to year N+1: data sloshing made concrete.

Usage:
    python src/mars100_runner.py                       # 100 years, seed 42
    python src/mars100_runner.py --years 50 --seed 7   # custom parameters
    python src/mars100_runner.py --output /tmp/mars100  # custom output dir
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from src.colonist import Colonist, create_colony  # noqa: E402
from src.governance import GovernanceState  # noqa: E402
from src.mars100 import (  # noqa: E402
    Resources, YearResult, tick_year,
)


def write_json(path: Path, data: dict) -> None:
    """Atomic JSON write: tmp file → fsync → rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)


def run_simulation(
    years: int = 100,
    seed: int = 42,
    output_dir: Path | None = None,
    quiet: bool = False,
) -> dict:
    """Run the Mars-100 simulation and write all output files.

    Returns the final report dict.
    """
    if output_dir is None:
        output_dir = _HERE.parent / "docs" / "mars-100"

    mars_dir = output_dir
    colonist_dir = mars_dir / "colonists"
    colonist_dir.mkdir(parents=True, exist_ok=True)

    # Initialize simulation state
    rng = random.Random(seed)
    colonists = create_colony(seed)
    resources = Resources()
    governance = GovernanceState()

    timeline: list[dict] = []
    all_deltas: list[dict] = []
    all_subsims: list[dict] = []
    all_amendments: list[dict] = []
    t0 = time.monotonic()

    if not quiet:
        print(f"Mars-100 Simulation — seed={seed}, years={years}")
        print("=" * 80)

    for y in range(1, years + 1):
        alive = [c for c in colonists if c.alive]
        if not alive:
            if not quiet:
                print(f"  Year {y:3d} │ Colony extinct. Simulation halted.")
            break

        result: YearResult = tick_year(y, colonists, resources, governance, rng)
        delta = result.to_dict()
        all_deltas.append(delta)

        # Save year delta file
        write_json(mars_dir / f"year-{y:03d}.json", delta)

        # Track sub-sims and meta-insights
        for ss in result.subsim_log:
            all_subsims.append(ss)
        if result.meta_insight:
            all_amendments.append({
                "year": y,
                "insight": result.meta_insight,
                "source": "meta-insight",
                "proposer": result.simulation_awareness.get("colonist", "?")
                    if result.simulation_awareness else "colony",
            })
        for prop in result.proposals:
            if prop.get("passed"):
                all_amendments.append({
                    "year": y,
                    "title": prop.get("title", ""),
                    "rationale": prop.get("rationale", ""),
                    "source": "governance-vote",
                    "proposer": prop.get("proposer", "?"),
                })

        # Build timeline entry
        res_dict = result.resources if isinstance(result.resources, dict) else {}
        timeline.append({
            "year": y,
            "alive": result.alive_count,
            "resources": res_dict,
            "governance_type": result.governance_label,
            "event_id": result.event.get("id", ""),
            "event_description": result.event.get("description", ""),
            "deaths": result.dead_this_year,
            "sub_sim_count": len(result.subsim_log),
            "proposals": len(result.proposals),
            "meta_insight": result.meta_insight,
        })

        if not quiet:
            alive_count = result.alive_count
            gov = result.governance_label or "?"
            subs = len(result.subsim_log)
            deaths = len(result.dead_this_year)
            elapsed = time.monotonic() - t0
            bar = "█" * alive_count + "░" * (10 - alive_count)
            event_id = result.event.get("id", "?")
            print(
                f"  Year {y:3d} │ {bar} │ "
                f"gov={gov:<14s} │ event={event_id:<20s} │ "
                f"subs={subs} │ deaths={deaths} │ {elapsed:.1f}s"
            )

    years_completed = len(all_deltas)

    # Save colonist state files
    for colonist in colonists:
        write_json(colonist_dir / f"{colonist.id}.json", colonist.to_dict())

    # Build final report
    alive_final = [c for c in colonists if c.alive]
    dead_final = [c for c in colonists if not c.alive]

    gov_labels = [t["governance_type"] for t in timeline]
    final_gov = gov_labels[-1] if gov_labels else "unknown"

    report = {
        "_meta": {
            "engine": "mars-100",
            "version": "2.0",
            "seed": seed,
            "years_simulated": years_completed,
            "colony_survived": len(alive_final) > 0,
        },
        "summary": {
            "years_completed": years_completed,
            "alive_count": len(alive_final),
            "dead_count": len(dead_final),
            "final_resources": resources.to_dict(),
            "governance_type": final_gov,
            "total_sub_sims": len(all_subsims),
            "amendments_proposed": len(all_amendments),
        },
        "colonists": {c.id: c.to_dict() for c in colonists},
        "governance": governance.to_dict(),
        "amendments": all_amendments,
        "timeline": timeline,
        "year_count": years_completed,
    }

    # Save data.json
    write_json(mars_dir / "data.json", report)

    # Save governance summary
    write_json(mars_dir / "governance.json", governance.to_dict())

    # Save overall summary
    write_json(mars_dir / "summary.json", report["summary"])

    if not quiet:
        print("=" * 80)
        s = report["summary"]
        print(f"SIMULATION COMPLETE — {s['years_completed']} years")
        print(f"  Colony survived: {report['_meta']['colony_survived']}")
        print(f"  Alive: {s['alive_count']}/10")
        print(f"  Governance: {s['governance_type']}")
        print(f"  Sub-sims: {s['total_sub_sims']}")
        print(f"  Amendments proposed: {s['amendments_proposed']}")
        print(f"  Time: {time.monotonic() - t0:.1f}s")
        print(f"\n  Output: {mars_dir}/")

    return report


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Mars-100: Recursive Colony Simulation Runner"
    )
    parser.add_argument("--years", type=int, default=100,
                        help="Martian years to simulate (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for determinism (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: docs/mars-100/)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    run_simulation(
        years=args.years,
        seed=args.seed,
        output_dir=output,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
