"""
Mars-100 analysis runner.

Re-runs the deterministic simulation (seed=42) to capture per-year
colonist snapshots, then runs full post-simulation analysis and
generates all output artifacts.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.mars100.engine import Mars100Engine
from src.mars100.analysis import run_full_analysis
from src.mars100.report import generate_governance_report


def run_and_analyze(seed: int = 42, total_years: int = 100) -> dict:
    """Run simulation and return analysis results."""
    engine = Mars100Engine(seed=seed, total_years=total_years)
    sim_result = engine.run()
    year_dicts = [yr.to_dict() for yr in sim_result.years]
    analysis = run_full_analysis(year_dicts)
    return {
        "sim_result": sim_result,
        "year_dicts": year_dicts,
        "analysis": analysis,
    }


def write_outputs(results: dict, docs_dir: str = "docs/mars-100",
                  state_dir: str = "state") -> None:
    """Write all output artifacts."""
    docs = Path(docs_dir)
    state = Path(state_dir)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "colonists").mkdir(exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)

    analysis = results["analysis"]
    sim_result = results["sim_result"]
    year_dicts = results["year_dicts"]

    # Write analysis.json
    _write_json(docs / "analysis.json", {
        "_meta": {
            "engine": "mars-100-analysis",
            "version": "1.0",
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        **{k: _safe_for_json(v) for k, v in analysis.items()},
    })

    # Write year files with colonist snapshots
    for yr in year_dicts:
        _write_json(docs / f"year-{yr['year']}.json", yr)

    # Write colonist files
    for colonist in sim_result.final_colonists:
        cid = colonist["id"]
        _write_json(docs / "colonists" / f"{cid}.json", colonist)
        if not colonist.get("alive"):
            _write_json(docs / "colonists" / f"{cid}-soul.json", colonist)

    # Write data.json (dashboard data)
    data = sim_result.to_dict()
    # Merge convergence data for dashboard charts
    data["convergence"] = {
        "pairwise_distance": analysis["convergence"]["pairwise_distance"],
        "per_stat_stddev": analysis["convergence"]["per_stat_stddev"],
        "stat_trends": analysis["convergence"]["stat_trends"],
        "overall_trend": analysis["convergence"]["overall_trend"],
    }
    data["governance_analysis"] = {
        "subsim_effectiveness": analysis["governance"]["subsim_effectiveness"],
        "type_breakdown": analysis["governance"]["type_breakdown"],
    }
    data["subsim_depths"] = analysis["subsim_analysis"]["depth_distribution"]
    if analysis.get("proposed_amendment"):
        data["proposed_amendment"] = analysis["proposed_amendment"]
    _write_json(docs / "data.json", data)

    # Write state/mars100.json
    _write_json(state / "mars100.json", data)

    # Write governance report
    report = generate_governance_report(
        analysis, data.get("summary", {}))
    (docs / "governance-report.md").write_text(report, encoding="utf-8")

    print(f"[mars-100] Wrote analysis to {docs}")
    print(f"[mars-100] Wrote state to {state / 'mars100.json'}")
    print(f"[mars-100] Governance report: {docs / 'governance-report.md'}")


def _write_json(path: Path, data: Any) -> None:
    """Atomic JSON write."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.rename(path)


def _safe_for_json(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _safe_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_for_json(v) for v in obj]
    if isinstance(obj, float) and (obj != obj):  # NaN check
        return 0.0
    return obj


from typing import Any


if __name__ == "__main__":
    print("[mars-100] Running simulation (seed=42, 100 years)...")
    results = run_and_analyze()
    analysis = results["analysis"]

    # Summary
    conv = analysis["convergence"]
    gov = analysis["governance"]
    subsim = analysis["subsim_analysis"]
    meta = analysis["meta_insights"]
    amendment = analysis.get("proposed_amendment")

    print(f"[mars-100] Value trend: {conv['overall_trend']}")
    print(f"[mars-100] Proposals: {gov['total_proposals']} ({gov['total_passed']} passed)")
    print(f"[mars-100] Sub-sims: {subsim['total_subsims']} (depth 3: {subsim['depth3_count']})")
    print(f"[mars-100] Meta events: {meta['total_events']}")
    if amendment:
        print(f"[mars-100] AMENDMENT PROPOSED: {amendment['text'][:80]}...")
    else:
        print("[mars-100] No amendment proposed (evidence threshold not met)")

    write_outputs(results)
    print("[mars-100] Done.")
