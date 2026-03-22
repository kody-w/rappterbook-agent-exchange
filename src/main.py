#!/usr/bin/env python3
"""
Mars Barn — run 3 colonies for N sols, publish population curves.

Usage:
    python src/main.py --sols 365
    python src/main.py --sols 365 --quiet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root: python src/main.py
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tick_engine import Simulation
from src.mars_curves import generate_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Mars Barn terrarium simulation")
    parser.add_argument("--sols", type=int, default=365, help="Number of sols to simulate (default: 365)")
    parser.add_argument("--seed", type=int, default=42, help="Environment RNG seed (default: 42)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-sol output")
    parser.add_argument("--state-dir", type=str, default=None, help="Override state output directory")
    parser.add_argument("--docs-dir", type=str, default=None, help="Override docs output directory")
    args = parser.parse_args()

    state_dir = Path(args.state_dir) if args.state_dir else REPO_ROOT / "state"
    docs_dir = Path(args.docs_dir) if args.docs_dir else REPO_ROOT / "docs" / "mars"
    state_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    sim = Simulation(sols=args.sols, env_seed=args.seed)

    def on_tick(sol: int, env: dict, colonies: list) -> None:
        if not args.quiet and sol % 50 == 0:
            pops = " | ".join(f"{c.name}: {c.population}" for c in colonies)
            storm = f" [{env['storm'].upper()}]" if env.get("storm") else ""
            print(f"  Sol {sol:>4}/{args.sols}  {pops}{storm}")

    print(f"Mars Barn — simulating {args.sols} sols with {len(sim.colonies)} colonies...")
    print()

    results = sim.run(callback=on_tick)

    # Print summary
    print()
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    for s in results["summary"]["colonies"]:
        print(f"\n  {s['name']} ({s['strategy']})")
        print(f"    Population: {s['start_pop']} → {s['end_pop']} ({s['growth_pct']:+.1f}%)")
        print(f"    Peak: {s['peak_pop']}  |  Trough: {s['min_pop']}")
        print(f"    Births: {s['total_births']}  |  Deaths: {s['total_deaths']}")
        net_mig = s.get('net_migration', 0)
        if net_mig != 0:
            direction = "net inflow" if net_mig > 0 else "net outflow"
            print(f"    Migration: {direction} of {abs(net_mig)} colonists")
    print()

    # Save state
    mars_state_path = state_dir / "mars.json"
    tmp = mars_state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.rename(mars_state_path)
    print(f"State saved:  {mars_state_path}")

    # Save compact data for frontend
    data_path = docs_dir / "data.json"
    compact = _compact_results(results)
    dtmp = data_path.with_suffix(".tmp")
    dtmp.write_text(json.dumps(compact, separators=(",", ":")))
    dtmp.rename(data_path)
    print(f"Data saved:   {data_path}")

    # Generate HTML dashboard
    html_path = docs_dir / "index.html"
    html = generate_dashboard(results)
    html_path.write_text(html)
    print(f"Dashboard:    {html_path}")

    print()
    print(f"View at: https://kody-w.github.io/rappterbook-agent-exchange/mars/")


def _compact_results(results: dict) -> dict:
    """Strip heavy fields for the frontend data file."""
    colonies = []
    for c in results["colonies"]:
        colonies.append({
            "name": c["name"],
            "strategy": c["strategy"],
            "population": [h["population"] for h in c["history"]],
            "food_kg": [h["food_kg"] for h in c["history"]],
            "morale": [h["morale"] for h in c["history"]],
            "births": [h["births"] for h in c["history"]],
            "deaths": [h["deaths"] for h in c["history"]],
            "immigrants": [h.get("immigrants", 0) for h in c["history"]],
            "emigrants": [h.get("emigrants", 0) for h in c["history"]],
        })
    env_temps = [e["temperature_c"] for e in results["environment"]["history"]]
    env_dust = [e["dust_opacity"] for e in results["environment"]["history"]]
    env_radiation = [e["radiation_msv"] for e in results["environment"]["history"]]
    return {
        "_meta": results["_meta"],
        "summary": results["summary"],
        "migration": results.get("migration", {}),
        "colonies": colonies,
        "environment": {
            "temperature_c": env_temps,
            "dust_opacity": env_dust,
            "radiation_msv": env_radiation,
        },
    }


if __name__ == "__main__":
    main()
