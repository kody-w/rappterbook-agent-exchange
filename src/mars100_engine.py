#!/usr/bin/env python3
"""
Mars-100 Engine — run the recursive colony simulation and publish results.

Usage:
    python src/mars100_engine.py --years 100
    python src/mars100_engine.py --years 50 --seed 99
    python src/mars100_engine.py --analyze-only  # analyze existing state
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100.engine import Mars100Engine
from src.mars100.analysis import full_analysis
from src.mars100.narrator import generate_final_report


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via .tmp + rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(str(tmp), str(path))


def main() -> None:
    """Run the Mars-100 simulation and write outputs."""
    parser = argparse.ArgumentParser(description="Mars-100 recursive colony simulation")
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--docs-dir", type=str, default=None)
    parser.add_argument("--analyze-only", action="store_true",
                        help="Analyze existing state/mars100.json without re-running")
    args = parser.parse_args()

    state_dir = Path(args.state_dir) if args.state_dir else REPO_ROOT / "state"
    docs_dir = Path(args.docs_dir) if args.docs_dir else REPO_ROOT / "docs" / "mars-100"
    state_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    colonists_dir = docs_dir / "colonists"
    colonists_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        state_path = state_dir / "mars100.json"
        if not state_path.exists():
            print(f"Error: {state_path} not found. Run simulation first.", file=sys.stderr)
            sys.exit(1)
        with open(state_path) as f:
            sim_dict = json.load(f)
        analysis = full_analysis(sim_dict)
        analysis_path = docs_dir / "analysis.json"
        _atomic_write_json(analysis_path, analysis)
        _print_analysis_summary(analysis)
        print(f"\n  Analysis -> {analysis_path}")
        return

    print(f"Mars-100 — simulating {args.years} Martian years (seed={args.seed})")

    engine = Mars100Engine(seed=args.seed, total_years=args.years)
    years_logged: list[int] = []

    def on_year(yr_result: object) -> None:
        """Callback per year."""
        year = yr_result.year  # type: ignore[attr-defined]
        years_logged.append(year)
        if not args.quiet and (year % 10 == 0 or year <= 3):
            deaths = len(yr_result.deaths)  # type: ignore[attr-defined]
            subs = len(yr_result.subsim_log)  # type: ignore[attr-defined]
            meta = len(yr_result.meta_awareness)  # type: ignore[attr-defined]
            print(f"  Year {year:>3}/{args.years}  "
                  f"deaths={deaths} subsims={subs} meta={meta}")

    result = engine.run(callback=on_year)
    sim_dict = result.to_dict()

    # Save canonical state
    _atomic_write_json(state_dir / "mars100.json", sim_dict)

    # Save per-year files
    for yr in result.years:
        _atomic_write_json(docs_dir / f"year-{yr.year}.json", yr.to_dict())

    # Save colonist files + soul files for the dead
    for c_dict in sim_dict["final_colonists"]:
        cid = c_dict["id"]
        _atomic_write_json(colonists_dir / f"{cid}.json", c_dict)
        if not c_dict.get("alive"):
            _atomic_write_json(colonists_dir / f"{cid}-soul.json", {
                "id": cid, "name": c_dict["name"], "element": c_dict["element"],
                "archetype": c_dict["archetype"],
                "death_year": c_dict.get("death_year"),
                "death_cause": c_dict.get("death_cause"),
                "final_stats": c_dict["stats"],
                "final_skills": c_dict["skills"],
                "memories": c_dict.get("memories", []),
                "legacy": "archived",
            })

    # Run analysis
    analysis = full_analysis(sim_dict)
    _atomic_write_json(docs_dir / "analysis.json", analysis)

    # Save dashboard data
    _atomic_write_json(docs_dir / "data.json", {
        "_meta": {
            "engine": "mars-100", "version": "2.0",
            "years": args.years, "seed": args.seed,
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        "summary": sim_dict["summary"],
        "analysis": analysis,
        "final_governance": sim_dict["final_governance"],
        "final_resources": sim_dict["final_resources"],
    })

    # Generate final report
    report = generate_final_report(sim_dict)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SIMULATION COMPLETE")
    print(f"{'=' * 60}")
    s = sim_dict["summary"]
    print(f"  Years completed:  {len(result.years)}")
    print(f"  Deaths:           {s['total_deaths']}")
    print(f"  Exiles:           {s['total_exiles']}")
    print(f"  Sub-simulations:  {s['total_subsims']}")
    print(f"  Governance:       {s['governance_changes']} changes")
    print(f"  Meta-awareness:   {s['meta_awareness_events']} events")
    print(f"  Final cohesion:   {s['final_cohesion']:.0%}")

    _print_analysis_summary(analysis)

    print(f"\n  State  -> {state_dir / 'mars100.json'}")
    print(f"  Years  -> {docs_dir}/year-*.json ({len(result.years)} files)")
    print(f"  Colonists -> {colonists_dir}/ ({len(sim_dict['final_colonists'])} files)")
    print(f"  Analysis -> {docs_dir / 'analysis.json'}")
    print(f"  Dashboard -> {docs_dir / 'data.json'}")


def _print_analysis_summary(analysis: dict) -> None:
    """Print analysis highlights to stdout."""
    fitness = analysis.get("fitness", {})
    print(f"\n  Fitness score: {fitness.get('composite', 0):.3f}")
    print(f"    survival={fitness.get('survival_rate', 0):.2f}  "
          f"resources={fitness.get('resource_health', 0):.2f}  "
          f"cohesion={fitness.get('social_cohesion', 0):.2f}  "
          f"stability={fitness.get('governance_stability', 0):.2f}  "
          f"culture={fitness.get('cultural_richness', 0):.2f}")

    conv = analysis.get("value_convergence", {})
    print(f"\n  Value convergence: {conv.get('verdict', 'unknown')} "
          f"({conv.get('overall_convergence', 0):.1%})")

    gov = analysis.get("governance_stability", {})
    print(f"  Governance: {gov.get('transitions', 0)} transitions, "
          f"attractor={gov.get('attractor', 'none')}")

    subsim = analysis.get("subsim_effectiveness", {})
    print(f"  Sub-sims: {subsim.get('total_subsims', 0)} total, "
          f"max depth={subsim.get('max_depth_reached', 0)}, "
          f"backed pass rate={subsim.get('backed_pass_rate', 0):.0%}")

    amend = analysis.get("amendment_proposal", {})
    if amend.get("proposed"):
        a = amend["amendment"]
        print(f"\n  ** PROPOSED AMENDMENT: {a['title']} **")
        print(f"     {a['text'][:100]}...")
        print(f"     Strength: {a['strength']:.2f} | Evidence: {a['evidence']}")
    else:
        print(f"\n  No amendment proposed: {amend.get('reason', 'insufficient evidence')}")


if __name__ == "__main__":
    main()
