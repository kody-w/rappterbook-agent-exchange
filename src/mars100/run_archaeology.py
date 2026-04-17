"""Runner for the archaeology engine."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100.archaeology import run_archaeology, aggregate_subsims, load_year


def main():
    state_dir = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))
    docs_dir = Path(os.environ.get("DOCS_DIR", str(REPO_ROOT / "docs")))
    mars_dir = docs_dir / "mars-100"
    if not mars_dir.exists():
        print("No mars-100 directory found at %s" % mars_dir)
        sys.exit(1)
    year_files = sorted(mars_dir.glob("year-*.json"))
    if not year_files:
        print("No year files found in %s" % mars_dir)
        sys.exit(1)
    print("Found %d year files in %s" % (len(year_files), mars_dir))
    report = run_archaeology(str(mars_dir))
    report_dict = report.to_dict()
    out_path = mars_dir / "archaeology.json"
    tmp_path = mars_dir / "archaeology.json.tmp"
    tmp_path.write_text(json.dumps(report_dict, indent=2))
    tmp_path.rename(out_path)
    print("Wrote %s (%d bytes)" % (out_path, out_path.stat().st_size))
    state_path = state_dir / "mars100.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        state["archaeology"] = {
            "years_analyzed": report.years_analyzed,
            "epochs": len(report.epochs),
            "artifacts": len(report.artifacts),
            "total_subsims": report.subsim_summary.total,
            "proposed_amendment": report.proposed_amendment.get("title", ""),
        }
        if not state.get("sub_sim_log"):
            years = [load_year(f) for f in year_files]
            subsim = aggregate_subsims(years)
            state["sub_sim_log"] = [
                {"colonist": c, "count": n}
                for c, n in sorted(subsim.by_colonist.items(), key=lambda x: -x[1])
            ]
            print("Fixed empty sub_sim_log: %d colonists" % len(state["sub_sim_log"]))
        tmp_state = state_path.with_suffix(".json.tmp")
        tmp_state.write_text(json.dumps(state, indent=2))
        tmp_state.rename(state_path)
        print("Updated %s" % state_path)
    print("Done. %d epochs, %d artifacts, %d sub-sims aggregated." % (
        len(report.epochs), len(report.artifacts), report.subsim_summary.total))
    print("Proposed amendment: %s" % report.proposed_amendment.get("title", "none"))


if __name__ == "__main__":
    main()
