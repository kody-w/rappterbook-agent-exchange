#!/usr/bin/env python3
"""
run_mars100.py — Drive the Mars-100 recursive colony simulation.

Runs 100 Mars years (or until governance collapse / extinction).
Writes year files, colonist state, governance history, sub-sim logs.

Usage:
    python src/run_mars100.py
    python src/run_mars100.py --years 100 --seed 42
    python src/run_mars100.py --years 50 --quiet
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

from src.colonist import Colonist, create_colony
from src.governance import GovernanceState, compute_fitness, _gini
from src.mars100 import Resources, tick_year, YearResult, reset_birth_counter


def now_iso() -> str:
    """UTC ISO timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_simulation(years: int = 100, seed: int = 42,
                   output_dir: Path | None = None,
                   state_dir: Path | None = None,
                   quiet: bool = False) -> dict:
    """Run the full Mars-100 simulation.

    Returns summary dict with all results.
    """
    import random
    rng = random.Random(seed)

    if output_dir is None:
        output_dir = REPO_ROOT / "docs" / "mars-100"
    if state_dir is None:
        state_dir = REPO_ROOT / "state" / "mars100"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "colonists").mkdir(exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    # Initialize colony
    reset_birth_counter()
    colonists = create_colony(seed)
    resources = Resources()
    governance = GovernanceState()

    year_results: list[dict] = []
    all_subsim_logs: list[dict] = []
    governance_transitions: list[dict] = []
    meta_insights: list[str] = []
    prev_label = "nascent"

    if not quiet:
        print(f"Mars-100 — simulating {years} Mars years with {len(colonists)} colonists")
        print(f"Seed: {seed}")
        print()

    for year in range(1, years + 1):
        result = tick_year(year, colonists, resources, governance, rng)
        year_results.append(result.to_dict())

        # Track governance transitions
        if result.governance_label != prev_label:
            governance_transitions.append({
                "year": year,
                "from": prev_label,
                "to": result.governance_label,
            })
            prev_label = result.governance_label

        # Collect sub-sim logs
        all_subsim_logs.extend(result.subsim_log)

        # Meta insights
        if result.meta_insight:
            meta_insights.append(result.meta_insight)

        # Write year file
        year_path = output_dir / f"year-{year:03d}.json"
        year_path.write_text(json.dumps(result.to_dict(), indent=2))

        if not quiet and year % 10 == 0:
            alive = result.alive_count
            label = result.governance_label
            crisis = resources.crisis_level()
            print(f"  Year {year:>3}/{years}  alive={alive}  "
                  f"gov={label}  crisis={crisis:.2f}  "
                  f"food={resources.food:.2f}  morale={resources.morale:.2f}")

        # Check for extinction
        if result.alive_count == 0:
            if not quiet:
                print(f"\n  ☠️  EXTINCTION at year {year}")
            break

        # Check for governance collapse
        if (result.governance_label == "collapsed" or
                (resources.morale <= 0.0 and resources.crisis_level() > 0.9)):
            if not quiet:
                print(f"\n  💀 GOVERNANCE COLLAPSE at year {year}")
            break

    # Write colonist state files
    for c in colonists:
        c_path = output_dir / "colonists" / f"{c.id}.json"
        c_path.write_text(json.dumps(c.to_dict(), indent=2))

    # Compute final fitness
    active = [c for c in colonists if c.alive]
    all_trust = []
    for c in colonists:
        for other_id, trust in c.relationships.items():
            all_trust.append(trust)
    avg_trust = sum(all_trust) / max(len(all_trust), 1)
    scores = [c.leadership_score for c in active]

    colony_state = {
        "alive_count": len(active),
        "total_count": len(colonists),
        "resources": resources.to_dict(),
        "avg_trust": avg_trust,
        "power_gini": _gini(scores) if scores else 0.0,
        "total_exiles": sum(c.times_exiled for c in colonists),
    }
    fitness = compute_fitness(colony_state)

    # Build summary
    final_year = year_results[-1]["year"] if year_results else 0
    summary = {
        "_meta": {
            "engine": "mars-100",
            "version": "1.0",
            "seed": seed,
            "years_simulated": final_year,
            "generated": now_iso(),
        },
        "colony": {
            "starting_colonists": 10,
            "surviving_colonists": len(active),
            "total_deaths": sum(1 for c in colonists if not c.alive),
            "total_births": sum(1 for c in colonists if c.id.startswith("mars-")),
            "death_causes": _count_deaths(colonists),
            "fitness": round(fitness, 4),
        },
        "resources_final": resources.to_dict(),
        "governance": {
            "final_label": prev_label,
            "transitions": governance_transitions,
            "total_proposals": len(governance.proposals_history),
            "passed_proposals": sum(
                1 for p in governance.proposals_history if p.get("outcome") == "passed"),
            "amendments": governance.constitution.amendments,
            "constitution_final": governance.constitution.to_dict(),
            "leader_final": governance.leader_id,
        },
        "subsimulations": {
            "total_runs": len(all_subsim_logs),
            "max_depth_reached": max(
                (s.get("max_depth_reached", s.get("depth", 0)) for s in all_subsim_logs), default=0),
            "log": all_subsim_logs[-20:],
        },
        "meta_insights": meta_insights,
        "colonists": [c.to_dict() for c in colonists],
        "governance_full": governance.to_dict(),
    }

    # Check for promotable amendment
    promoted_amendment = None
    if meta_insights and fitness > 0.5:
        promoted_amendment = {
            "source": "Mars-100 recursive simulation",
            "year_discovered": final_year,
            "fitness": fitness,
            "insight": meta_insights[-1] if meta_insights else None,
            "proposal": (
                "Amendment: Governance systems should incorporate recursive "
                "self-modeling — decisions that affect the community should be "
                "testable in sandboxed simulations before commitment. This is "
                "the principle of simulation-informed governance."
            ),
        }
        summary["promoted_amendment"] = promoted_amendment

    # Write outputs
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    gov_path = output_dir / "governance.json"
    gov_path.write_text(json.dumps(governance.to_dict(), indent=2))

    subsim_path = state_dir / "subsim_log.json"
    subsim_path.write_text(json.dumps(all_subsim_logs, indent=2))

    # Write compact data.json for visualization
    viz_data = {
        "_meta": summary["_meta"],
        "years": [{
            "year": yr["year"],
            "event": yr["event"]["id"],
            "alive": yr["alive_count"],
            "governance": yr["governance_label"],
            "resources": yr["resources"],
        } for yr in year_results],
        "colonists": [{
            "id": c.id, "name": c.name, "element": c.element,
            "alive": c.alive, "year_died": c.year_died,
        } for c in colonists],
        "governance_transitions": governance_transitions,
        "fitness": round(fitness, 4),
    }
    viz_path = output_dir / "data.json"
    viz_path.write_text(json.dumps(viz_data, indent=2))

    if not quiet:
        print()
        print("=" * 60)
        print("MARS-100 SIMULATION COMPLETE")
        print("=" * 60)
        print(f"  Years simulated: {final_year}")
        print(f"  Survivors: {len(active)}/{len(colonists)}")
        print(f"  Final governance: {prev_label}")
        print(f"  Fitness score: {fitness:.4f}")
        print(f"  Sub-simulations run: {len(all_subsim_logs)}")
        print(f"  Meta-insights: {len(meta_insights)}")
        if promoted_amendment:
            print(f"\n  🏛️ AMENDMENT PROMOTED:")
            print(f"    {promoted_amendment['proposal']}")
        print()
        for c in colonists:
            status = "alive" if c.alive else f"died year {c.year_died} ({c.cause_of_death})"
            print(f"  {c.name:20s} [{c.element:5s}]  {status}")
        print()

    return summary


def _count_deaths(colonists: list[Colonist]) -> dict[str, int]:
    """Count death causes."""
    causes: dict[str, int] = {}
    for c in colonists:
        if c.cause_of_death:
            causes[c.cause_of_death] = causes.get(c.cause_of_death, 0) + 1
    return causes


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Mars-100 recursive colony simulation")
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--state-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    state_dir = Path(args.state_dir) if args.state_dir else None

    run_simulation(
        years=args.years,
        seed=args.seed,
        output_dir=output_dir,
        state_dir=state_dir,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
