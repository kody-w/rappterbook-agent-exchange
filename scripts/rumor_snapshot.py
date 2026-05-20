#!/usr/bin/env python3
"""
rumor_snapshot.py — snapshot the colony's rumor population.

Companion to channel_health.py: runs the full Mars-100 sim, then writes a
per-rumor digest (with carrier counts, lineage, fragmentation) to
state/rumors.json + docs/mars-100/rumors.json. The rumors organ rides on top
of comm-channels — flatlined channels block information.

Usage:
  python scripts/rumor_snapshot.py
  python scripts/rumor_snapshot.py --years 50 --seed 7
  python scripts/rumor_snapshot.py --quiet --no-docs
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100 import Mars100Engine, compute_fragmentation


def build_report(engine: Mars100Engine) -> dict:
    state = engine.rumors
    lookup = {c.id: c.name for c in engine.colonists}
    active_ids = {c.id for c in engine.colonists if c.is_active()}

    rumors = []
    for rid, rumor in sorted(state.rumors.items()):
        d = rumor.to_dict()
        d["carrier_names"] = [lookup.get(cid, cid) for cid in d["carriers"]]
        d["origin_name"] = lookup.get(rumor.origin_id, rumor.origin_id)
        rumors.append(d)

    fragmentation = compute_fragmentation(state.rumors, active_ids)
    largest = max((r["carrier_count"] for r in rumors), default=0)
    mutation_chains = sum(1 for r in rumors if r["mutated_from"] is not None)
    archive_total = len(state.archive)

    return {
        "_meta": {
            "organ": "rumors",
            "engine": "mars-100",
            "version": "13.0",
            "generated": datetime.now(timezone.utc).isoformat(),
            "year_snapshot": engine.year,
        },
        "active_rumors": len(rumors),
        "largest_carrier_count": largest,
        "mutation_chains": mutation_chains,
        "fragmentation": round(fragmentation, 4),
        "archived_total": archive_total,
        "rumors": rumors,
        "archive_recent": list(state.archive[-25:]),
        "transmission_log_recent": list(state.transmission_log[-25:]),
    }


def write_report(report: dict, state_path: Path, docs_path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str))
    tmp.rename(state_path)
    if docs_path is not None:
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        dtmp = docs_path.with_suffix(".tmp")
        dtmp.write_text(json.dumps(report, separators=(",", ":"), default=str))
        dtmp.rename(docs_path)


def print_summary(report: dict) -> None:
    print(f"Rumor snapshot — year {report['_meta']['year_snapshot']}")
    print(f"  active rumors:        {report['active_rumors']}")
    print(f"  largest carrier set:  {report['largest_carrier_count']}")
    print(f"  mutation chains:      {report['mutation_chains']}")
    print(f"  fragmentation:        {report['fragmentation']}")
    print(f"  archived (dead):      {report['archived_total']}")
    if report['rumors']:
        print("\nTop 5 rumors by carrier count:")
        top = sorted(report['rumors'],
                       key=lambda r: r['carrier_count'], reverse=True)[:5]
        for r in top:
            print(f"  {r['carrier_count']:3d} carriers  | "
                  f"{r['text'][:70]}")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate state/rumors.json")
    p.add_argument("--years", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--state-dir", default=None)
    p.add_argument("--docs-dir", default=None)
    p.add_argument("--no-docs", action="store_true")
    args = p.parse_args()

    state_dir = Path(args.state_dir) if args.state_dir else REPO_ROOT / "state"
    docs_dir = (Path(args.docs_dir) if args.docs_dir
                 else REPO_ROOT / "docs" / "mars-100")

    if not args.quiet:
        print(f"Running Mars-100 for {args.years} years (seed {args.seed})...")
    engine = Mars100Engine(seed=args.seed, total_years=args.years)
    engine.run()
    report = build_report(engine)

    state_path = state_dir / "rumors.json"
    docs_path = None if args.no_docs else (docs_dir / "rumors.json")
    write_report(report, state_path, docs_path)

    if not args.quiet:
        print()
        print_summary(report)
        print(f"\nWrote -> {state_path}")
        if docs_path is not None:
            print(f"Wrote -> {docs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
