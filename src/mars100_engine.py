#!/usr/bin/env python3
"""
Mars-100 Engine — run the recursive colony simulation and publish results.

Usage:
    python src/mars100_engine.py --years 100
    python src/mars100_engine.py --years 50 --seed 99
    python src/mars100_engine.py --legacy          # use legacy monolith
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


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically via tmp + rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)


def _run_v3(args: argparse.Namespace, state_dir: Path, docs_dir: Path) -> None:
    """Run with Mars100Engine (v3 — with codex)."""
    from src.mars100 import Mars100Engine

    engine = Mars100Engine(seed=args.seed, total_years=args.years)
    print(f"Mars-100 v3 — simulating {args.years} Martian years (seed={args.seed})")

    def progress(yr):
        if not args.quiet and (yr.year <= 5 or yr.year % 10 == 0):
            cx = yr.codex_snapshot
            active = cx.get("active_entries", 0)
            print(f"  Year {yr.year:>3}/{args.years}  "
                  f"pop={len([s for s in yr.colonist_snapshots if s.get('alive', True)]):>2}  "
                  f"codex={active} entries")

    result = engine.run(callback=progress)
    d = result.to_dict()

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE (v3 — memory codex)")
    print(f"{'='*60}")
    s = d["summary"]
    print(f"  Deaths: {s['total_deaths']}  Exiles: {s['total_exiles']}  "
          f"Births: {s['total_births']}  SubSims: {s['total_subsims']}")
    print(f"  Governance changes: {s['governance_changes']}  "
          f"Meta-events: {s['meta_awareness_events']}")
    print(f"  Convergence: {s['convergence_trend']}")
    fc = d.get("final_codex", {})
    entries = fc.get("entries", [])
    active = [e for e in entries if e.get("strength", 0) >= 0.05]
    print(f"  Codex: {len(entries)} total entries, {len(active)} active")
    laws = [e for e in active if e.get("entry_type") == "law"]
    if laws:
        print(f"  Laws: {len(laws)}")
        for law in laws[:5]:
            print(f"    - {law['event_name'][:70]}")

    # Write outputs
    _atomic_write(state_dir / "mars100.json", d)
    print(f"\n  State -> {state_dir / 'mars100.json'}")

    _atomic_write(docs_dir / "data.json", d)
    print(f"  Dashboard -> {docs_dir / 'data.json'}")

    colonists_dir = docs_dir / "colonists"
    colonists_dir.mkdir(parents=True, exist_ok=True)
    for c in d["final_colonists"]:
        _atomic_write(colonists_dir / f"{c['id']}.json", c)

    for yr_data in d["years"]:
        _atomic_write(docs_dir / f"year-{yr_data['year']}.json", yr_data)

    print(f"  Files: {len(d['years'])} year deltas + "
          f"{len(d['final_colonists'])} colonists")


def _run_legacy(args: argparse.Namespace, state_dir: Path, docs_dir: Path) -> None:
    """Run with legacy monolith (src/mars100.py)."""
    from src.mars100 import run_simulation  # type: ignore[attr-defined]

    print(f"Mars-100 legacy — simulating {args.years} Martian years")
    result = run_simulation(years=args.years, seed=args.seed)
    colony = result["colony"]
    deltas = result["deltas"]
    summary = result["summary"]

    if not args.quiet:
        for delta in deltas:
            yr = delta["year"]
            if yr <= 5 or yr % 10 == 0:
                print(f"  Year {yr:>3}/{args.years}  Pop: {delta['population']:>2}")

    _atomic_write(state_dir / "mars100.json", colony)
    _atomic_write(docs_dir / "data.json", {
        "_meta": {"engine": "mars-100", "version": "1.0",
                  "years": args.years, "seed": args.seed,
                  "generated": datetime.now(timezone.utc).isoformat()},
        "summary": summary, "colony": colony, "deltas": deltas,
    })
    print(f"  State -> {state_dir / 'mars100.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mars-100 recursive colony simulation")
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--legacy", action="store_true", help="Use legacy monolith engine")
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--docs-dir", type=str, default=None)
    args = parser.parse_args()

    state_dir = Path(args.state_dir) if args.state_dir else REPO_ROOT / "state"
    docs_dir = Path(args.docs_dir) if args.docs_dir else REPO_ROOT / "docs" / "mars-100"
    state_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    if args.legacy:
        _run_legacy(args, state_dir, docs_dir)
    else:
        _run_v3(args, state_dir, docs_dir)


if __name__ == "__main__":
    main()
