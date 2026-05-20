#!/usr/bin/env python3
"""
channel_health.py — snapshot the colony's communication channels.

Mirrors the original Rappterbook channel_health task: scan every channel
for flatlines (>= 10 frames silent), emit per-channel vitals + revival
prompts, dump to state/channel_health.json. Here a "channel" is a pair
of colonists and a "frame" is a Martian year.

Usage:
  python scripts/channel_health.py                  # default 30 years, seed 42
  python scripts/channel_health.py --years 50 --seed 7
  python scripts/channel_health.py --state-dir /tmp/foo --no-docs --quiet
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

from src.mars100 import (
    Mars100Engine,
    CHANNEL_STATUS_FLATLINED, CHANNEL_STATUS_FADING, CHANNEL_STATUS_VITAL,
    CHANNEL_STATUS_REVIVED, CHANNEL_STATUS_DORMANT, CHANNEL_STATUS_INACTIVE,
    FLATLINE_SILENCE_YEARS, BRIDGE_TRUST_FLOOR, BRIDGE_COOLDOWN_YEARS,
)


def build_report(engine: Mars100Engine, year: int) -> dict:
    """Convert the engine's end-of-run state into the snapshot dict."""
    state = engine.comm_channels
    lookup = {c.id: {"name": c.name, "element": c.element,
                      "alive": c.alive, "active": c.is_active()}
              for c in engine.colonists}

    per_channel = []
    flatlined = []
    fading = []
    for key, ch in sorted(state.channels.items()):
        a, b = key
        d = ch.to_dict()
        d["name_a"] = lookup.get(a, {}).get("name", a)
        d["name_b"] = lookup.get(b, {}).get("name", b)
        d["a_alive"] = lookup.get(a, {}).get("alive", False)
        d["b_alive"] = lookup.get(b, {}).get("alive", False)
        per_channel.append(d)
        if ch.status == CHANNEL_STATUS_FLATLINED:
            flatlined.append(d)
        elif ch.status == CHANNEL_STATUS_FADING:
            fading.append(d)

    summary: dict = {}
    for ch in state.channels.values():
        summary[ch.status] = summary.get(ch.status, 0) + 1
    summary["total"] = len(state.channels)

    active_set = {c.id for c in engine.colonists if c.is_active()}
    active_vitalities = [ch.vitality for k, ch in state.channels.items()
                          if k[0] in active_set and k[1] in active_set]
    overall_health = (sum(active_vitalities) / len(active_vitalities)
                       if active_vitalities else 1.0)

    return {
        "_meta": {
            "organ": "comm-channels",
            "engine": "mars-100",
            "version": "12.1",
            "generated": datetime.now(timezone.utc).isoformat(),
            "year_snapshot": year,
            "flatline_threshold_years": FLATLINE_SILENCE_YEARS,
            "bridge_trust_floor": BRIDGE_TRUST_FLOOR,
            "bridge_cooldown_years": BRIDGE_COOLDOWN_YEARS,
        },
        "summary": summary,
        "overall_health_score": round(overall_health, 4),
        "flatlined_count": len(flatlined),
        "fading_count": len(fading),
        "flatlined_channels": flatlined,
        "fading_channels": fading,
        "channels": per_channel,
        "revival_prompts": list(state.revival_log[-25:]),
        "bridge_prompts": list(state.bridge_log[-25:]),
        "bridge_stats": {
            "total_bridge_prompts": state.total_bridge_prompts,
            "total_bridge_revivals": state.total_bridge_revivals,
            "bridge_efficacy_rate": (
                round(state.total_bridge_revivals
                      / state.total_bridge_prompts, 4)
                if state.total_bridge_prompts else 0.0),
        },
    }


def run_simulation(seed: int, years: int) -> Mars100Engine:
    """Run a full Mars-100 sim and return the engine."""
    engine = Mars100Engine(seed=seed, total_years=years)
    engine.run()
    return engine


def write_report(report: dict, state_path: Path,
                  docs_path: Path | None) -> None:
    """Atomically write the report to disk."""
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
    print(f"Channel health snapshot — year {report['_meta']['year_snapshot']}")
    print(f"  total channels:      {s.get('total', 0)}")
    for label, key in (("vital", CHANNEL_STATUS_VITAL),
                       ("fading", CHANNEL_STATUS_FADING),
                       ("flatlined", CHANNEL_STATUS_FLATLINED),
                       ("revived", CHANNEL_STATUS_REVIVED),
                       ("dormant", CHANNEL_STATUS_DORMANT),
                       ("inactive", CHANNEL_STATUS_INACTIVE)):
        print(f"  {label:20s} {s.get(key, 0)}")
    print(f"  overall_health:      {report['overall_health_score']}")
    if report["flatlined_channels"]:
        print(f"\nFlatlined channels needing revival "
              f"({len(report['flatlined_channels'])}):")
        for d in report["flatlined_channels"][:10]:
            print(f"  - {d['name_a']}<->{d['name_b']}  silent "
                  f"{d['silence_streak']}y (last seen year "
                  f"{d['last_contact_year']})")
    if report["revival_prompts"]:
        print("\nRevival prompts (most recent):")
        for p in report["revival_prompts"][-5:]:
            print(f"  ! {p['text']} [{p['suggested_action']}]")
    bs = report.get("bridge_stats", {})
    if bs.get("total_bridge_prompts", 0):
        print(f"\nBridge-builder stats: "
              f"{bs['total_bridge_prompts']} nominations, "
              f"{bs['total_bridge_revivals']} credited revivals "
              f"({bs['bridge_efficacy_rate'] * 100:.1f}% efficacy)")
    if report.get("bridge_prompts"):
        print("\nBridge prompts (most recent):")
        for p in report["bridge_prompts"][-5:]:
            print(f"  ~ {p['text']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate state/channel_health.json")
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

    state_path = state_dir / "channel_health.json"
    docs_path = None if args.no_docs else (docs_dir / "channel_health.json")
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
