"""
Runner for Mars-100 simulation.

Executes the simulation and produces all output artifacts:
- state/mars100.json (canonical state — single source of truth)
- docs/mars-100/data.json (dashboard data derived from state)
- docs/mars-100/colonists/{id}.json (per-colonist soul files)
- docs/mars-100/report.md (final governance report)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.mars100.engine import Mars100Engine
from src.mars100.narrator import (
    narrate_year, generate_diary_entries, generate_final_report,
)
import random


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def run_simulation(seed: int = 42, years: int = 100,
                   output_dir: Path = None) -> dict:
    """Run the Mars-100 simulation and write all artifacts."""
    if output_dir is None:
        output_dir = REPO_ROOT

    state_dir = output_dir / "state"
    docs_dir = output_dir / "docs" / "mars-100"
    colonists_dir = docs_dir / "colonists"

    for d in [state_dir, docs_dir, colonists_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Run engine
    engine = Mars100Engine(seed=seed, total_years=years)
    rng = random.Random(seed)
    result = engine.run()
    canonical = result.to_dict()

    # 1. Write canonical state
    atomic_write_json(state_dir / "mars100.json", canonical)

    # 2. Write per-colonist soul files
    for colonist in canonical["final_colonists"]:
        cid = colonist["id"]
        atomic_write_json(colonists_dir / f"{cid}.json", colonist)

    # 3. Generate diary entries and year narratives
    diary_entries: list = []
    year_narratives: list = []
    for year_data in canonical["years"]:
        narrative = narrate_year(year_data, rng)
        year_narratives.append({
            "year": year_data["year"],
            "narrative": narrative,
        })
        entries = generate_diary_entries(
            year_data, year_data.get("colonist_snapshots", []),
            rng, count=3,
        )
        diary_entries.extend(entries)

    # 4. Generate final report
    report_md = generate_final_report(canonical)
    report_path = docs_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")

    # 5. Build dashboard data (derived from canonical state)
    dashboard_data = _build_dashboard_data(canonical, diary_entries,
                                           year_narratives)
    atomic_write_json(docs_dir / "data.json", dashboard_data)

    return canonical


def _build_dashboard_data(canonical: dict, diary_entries: list,
                          year_narratives: list) -> dict:
    """Build dashboard-friendly data from canonical state."""
    meta = canonical["_meta"]
    summary = canonical["summary"]
    years = canonical["years"]

    # Resource timeline
    resource_timeline: list = []
    for y in years:
        resource_timeline.append({
            "year": y["year"],
            **y["resources_after"],
        })

    # Cohesion timeline
    cohesion_timeline: list = []
    for y in years:
        cohesion_timeline.append({
            "year": y["year"],
            "cohesion": y["social_cohesion"],
        })

    # Event timeline
    event_timeline: list = []
    for y in years:
        for ev in y["events"]:
            event_timeline.append({
                "year": y["year"],
                "name": ev["name"],
                "category": ev["category"],
                "severity": ev["severity"],
                "description": ev["description"],
            })

    # Death timeline
    death_timeline: list = []
    for y in years:
        for d in y["deaths"]:
            death_timeline.append(d)

    # Governance timeline
    gov_timeline: list = []
    for y in years:
        gov_timeline.append({
            "year": y["year"],
            "gov_type": y["governance_state"]["gov_type"],
        })

    # Sub-sim summary
    subsim_by_year: list = []
    for y in years:
        subsim_by_year.append({
            "year": y["year"],
            "count": len(y["subsim_log"]),
            "max_depth": max((s.get("depth", 1)
                              for s in y["subsim_log"]), default=0),
        })

    # Meta-awareness timeline
    meta_timeline: list = []
    for y in years:
        for m in y["meta_awareness"]:
            meta_timeline.append(m)

    return {
        "_meta": meta,
        "summary": summary,
        "colonists": canonical["final_colonists"],
        "resource_timeline": resource_timeline,
        "cohesion_timeline": cohesion_timeline,
        "event_timeline": event_timeline,
        "death_timeline": death_timeline,
        "governance_timeline": gov_timeline,
        "subsim_by_year": subsim_by_year,
        "meta_timeline": meta_timeline,
        "diary_entries": diary_entries[:100],
        "year_narratives": year_narratives,
        "final_governance": canonical["final_governance"],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run Mars-100 recursive colony simulation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output) if args.output else REPO_ROOT
    result = run_simulation(seed=args.seed, years=args.years,
                            output_dir=out)
    meta = result["_meta"]
    summary = result["summary"]
    print(f"Mars-100 simulation complete.")
    print(f"  Seed: {meta['seed']}")
    print(f"  Years: {meta['completed_years']}/{meta['requested_years']}")
    if meta["extinction_year"] > 0:
        print(f"  Extinction: year {meta['extinction_year']}")
    print(f"  Deaths: {summary['total_deaths']}")
    print(f"  Sub-sims: {summary['total_subsims']}")
    print(f"  Gov changes: {summary['governance_changes']}")
    print(f"  Meta events: {summary['meta_awareness_events']}")
