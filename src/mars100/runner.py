"""
Mars-100 analysis runner.

Re-runs the deterministic sim (seed=42), captures per-year colonist snapshots,
runs the full analysis pipeline, generates the governance report, and writes
all output artifacts to docs/mars-100/ and state/.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.mars100.engine import Mars100Engine
from src.mars100.analysis import full_analysis
from src.mars100.report import generate_governance_report


def run_and_analyze(seed: int = 42, total_years: int = 100) -> dict[str, Any]:
    """Run the sim and return the combined sim results + analysis.

    Returns a dict with keys: sim_dict, analysis, report_md.
    """
    engine = Mars100Engine(seed=seed, total_years=total_years)
    result = engine.run()
    sim_dict = result.to_dict()

    analysis = full_analysis(sim_dict)
    report_md = generate_governance_report(analysis, sim_dict.get("summary"))

    return {
        "sim_dict": sim_dict,
        "analysis": analysis,
        "report_md": report_md,
    }


def write_outputs(
    combined: dict[str, Any],
    docs_dir: str | Path | None = None,
    state_dir: str | Path | None = None,
) -> dict[str, str]:
    """Write analysis artifacts to disk.

    Returns dict mapping artifact names to file paths.
    """
    docs_dir = Path(docs_dir or os.environ.get("DOCS_DIR", "docs/mars-100"))
    state_dir = Path(state_dir or os.environ.get("STATE_DIR", "state"))
    docs_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    sim_dict = combined["sim_dict"]
    analysis = combined["analysis"]
    report_md = combined["report_md"]

    written: dict[str, str] = {}

    # 1. analysis.json — the full analysis
    analysis_path = docs_dir / "analysis.json"
    _atomic_write(analysis_path, analysis)
    written["analysis"] = str(analysis_path)

    # 2. governance-report.md — human-readable
    report_path = docs_dir / "governance-report.md"
    report_path.write_text(report_md, encoding="utf-8")
    written["report"] = str(report_path)

    # 3. data.json — enriched sim data with analysis summary
    data_path = docs_dir / "data.json"
    data = {
        "_meta": sim_dict.get("_meta", {}),
        "summary": sim_dict.get("summary", {}),
        "analysis_summary": {
            "fitness": analysis.get("fitness", {}),
            "convergence_verdict": analysis.get("value_convergence", {}).get("verdict"),
            "governance_attractor": analysis.get("governance_stability", {}).get("attractor"),
            "amendment_proposed": analysis.get("amendment_proposal", {}).get("proposed", False),
            "meta_events": analysis.get("meta_emergence", {}).get("total_events", 0),
        },
        "final_colonists": sim_dict.get("final_colonists", []),
        "final_resources": sim_dict.get("final_resources", {}),
        "final_governance": sim_dict.get("final_governance", {}),
    }
    _atomic_write(data_path, data)
    written["data"] = str(data_path)

    # 4. colonist files
    colonist_dir = docs_dir / "colonists"
    colonist_dir.mkdir(parents=True, exist_ok=True)
    for c in sim_dict.get("final_colonists", []):
        cid = c.get("id", "unknown")
        cpath = colonist_dir / f"{cid}.json"
        _atomic_write(cpath, c)
    written["colonists"] = str(colonist_dir)

    # 5. state/mars100.json — canonical state copy
    state_path = state_dir / "mars100.json"
    state_data = {
        "_meta": {
            "engine": "mars-100",
            "version": "1.2",
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        "summary": sim_dict.get("summary", {}),
        "analysis": {
            "fitness": analysis.get("fitness", {}),
            "amendment_proposal": analysis.get("amendment_proposal", {}),
        },
    }
    _atomic_write(state_path, state_data)
    written["state"] = str(state_path)

    return written


def _atomic_write(path: Path, data: Any) -> None:
    """Write JSON atomically via tmp + rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    """CLI entry point."""
    import sys

    seed = 42
    years = 100

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--seed" and i < len(sys.argv) - 1:
            seed = int(sys.argv[i + 1])
        elif arg == "--years" and i < len(sys.argv) - 1:
            years = int(sys.argv[i + 1])

    print(f"Running Mars-100 (seed={seed}, years={years})...")
    combined = run_and_analyze(seed=seed, total_years=years)

    print("Writing outputs...")
    written = write_outputs(combined)

    print("Artifacts written:")
    for name, path in written.items():
        print(f"  {name}: {path}")

    fitness = combined["analysis"]["fitness"]["composite"]
    print(f"\nColony fitness: {fitness:.2f}")

    amendment = combined["analysis"]["amendment_proposal"]
    if amendment.get("proposed"):
        print(f"Amendment proposed: {amendment['amendment']['title']}")
    else:
        print("No amendment proposed.")


if __name__ == "__main__":
    main()
