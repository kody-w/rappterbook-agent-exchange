#!/usr/bin/env python3
"""
colonist_comm_vitals.py — per-colonist comm health snapshot.

Companion to scripts/channel_health.py. Where channel_health is per-pair
and graph-level, this script answers the action chooser's node-level
question: "Which colonist is one death away from total silence? Who's a
lifeline for someone else? Who is the most urgent target for an outreach
action this year?"

Runs a Mars-100 sim for N years, then walks the engine's comm-channel
state and dumps state/colonist_comm_vitals.json. Standalone — never
touches channel_health.json.

Usage:
  python scripts/colonist_comm_vitals.py
  python scripts/colonist_comm_vitals.py --years 50 --seed 7 --quiet
  python scripts/colonist_comm_vitals.py --state-dir /tmp/foo --no-docs
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

from src.mars100 import Mars100Engine
from src.mars100.comm_vitals import compute_colonist_vitals, summarise


def build_report(engine: Mars100Engine, year: int) -> dict:
    """Compose the JSON snapshot from the engine's end-of-run state."""
    state = engine.comm_channels
    active_ids = [c.id for c in engine.colonists if c.is_active()]
    names = {c.id: c.name for c in engine.colonists}
    vitals = compute_colonist_vitals(active_ids, state.channels, names=names)

    top_urgent = [v.to_dict() for v in vitals if v.urgency >= 0.35][:10]
    lifelines = [v.to_dict() for v in vitals if v.is_lifeline]

    return {
        "_meta": {
            "organ": "comm-vitals",
            "engine": "mars-100",
            "version": "12.1",
            "generated": datetime.now(timezone.utc).isoformat(),
            "year_snapshot": year,
        },
        "summary": summarise(vitals),
        "colonists": [v.to_dict() for v in vitals],
        "top_urgent": top_urgent,
        "lifelines": lifelines,
    }


def run_simulation(seed: int, years: int) -> Mars100Engine:
    engine = Mars100Engine(seed=seed, total_years=years)
    engine.run()
    return engine


def write_report(report: dict, state_path: Path,
                  docs_path: Path | None) -> None:
    """Atomic two-file write — pretty in state/, minified in docs/."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str))
    tmp.rename(state_path)
    if docs_path is not None:
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        dtmp = docs_path.with_suffix(".tmp")
        dtmp.write_text(json.dumps(report, separators=(",", ":"),
                                    default=str))
        dtmp.rename(docs_path)


def print_summary(report: dict) -> None:
    s = report["summary"]
    meta = report["_meta"]
    print(f"Colonist comm vitals — year {meta['year_snapshot']}")
    print(f"  colonists:           {s['total_colonists']}")
    print(f"  mean urgency:        {s['mean_urgency']}")
    print(f"  healthy:             {s['healthy']}")
    print(f"  strained:            {s['strained']}")
    print(f"  isolated:            {s['isolated']}")
    print(f"  ghosted:             {s['ghosted']}")
    print(f"  lifelines:           {s['lifelines']}")
    if report["top_urgent"]:
        print("\nMost urgent outreach targets:")
        for v in report["top_urgent"][:5]:
            tag = " (LIFELINE)" if v["is_lifeline"] else ""
            print(f"  ! {v['name']:<14s} urgency={v['urgency']:.2f} "
                  f"live={v['live_channels']}/{v['channel_count']} "
                  f"[{v['classification']}]{tag}")
    if report["lifelines"]:
        print(f"\nLifelines ({len(report['lifelines'])}) — "
              f"removing these isolates someone:")
        for v in report["lifelines"][:5]:
            partners = ", ".join(v["sole_partners"])
            print(f"  * {v['name']:<14s} sole link for: {partners}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate state/colonist_comm_vitals.json")
    parser.add_argument("--years", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--no-docs", action="store_true")
    args = parser.parse_args()

    state_dir = Path(args.state_dir) if args.state_dir else REPO_ROOT / "state"
    docs_dir = (Path(args.docs_dir) if args.docs_dir
                else REPO_ROOT / "docs" / "mars-100")

    if not args.quiet:
        print(f"Running Mars-100 for {args.years} years (seed {args.seed})...")
    engine = run_simulation(seed=args.seed, years=args.years)
    report = build_report(engine, year=engine.year)

    state_path = state_dir / "colonist_comm_vitals.json"
    docs_path = (None if args.no_docs
                 else docs_dir / "colonist_comm_vitals.json")
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
