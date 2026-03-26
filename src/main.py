#!/usr/bin/env python3
"""
Mars Barn — run 3 colonies for N sols, publish population curves.

Usage:
    python src/main.py --sols 365
    python src/main.py --sols 365 --monte-carlo 50
    python src/main.py --sols 365 --quiet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tick_engine import Simulation
from src.mars_curves import generate_dashboard
from src.monte_carlo import run_ensemble, PERCENTILES


def main() -> None:
    parser = argparse.ArgumentParser(description="Mars Barn terrarium simulation")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--monte-carlo", type=int, default=0, metavar="N",
                        help="Run N seeds and produce confidence bands")
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--docs-dir", type=str, default=None)
    args = parser.parse_args()

    state_dir = Path(args.state_dir) if args.state_dir else REPO_ROOT / "state"
    docs_dir = Path(args.docs_dir) if args.docs_dir else REPO_ROOT / "docs" / "mars"
    state_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    mc_data = None
    if args.monte_carlo > 0:
        print(f"Mars Barn — Monte Carlo: {args.monte_carlo} seeds × {args.sols} sols...")
        ensemble = run_ensemble(n_seeds=args.monte_carlo, sols=args.sols)
        results = ensemble.canonical_results
        mc_data = _serialize_ensemble(ensemble)
        print()
        print("=" * 60)
        print(f"MONTE CARLO COMPLETE ({ensemble.n_seeds} seeds)")
        print("=" * 60)
        for ci, name in enumerate(ensemble.colony_names):
            strat = ensemble.colony_strategies[ci]
            fps = ensemble.final_pop_stats[ci]
            gps = ensemble.growth_pct_stats[ci]
            surv = ensemble.survival_rates[ci]
            print(f"\n  {name} ({strat})")
            print(f"    Final pop:  {fps['mean']:.0f} ± {fps['stdev']:.0f}  "
                  f"(p10={fps['p10']:.0f}, p90={fps['p90']:.0f})")
            print(f"    Growth:     {gps['mean']:+.1f}% ± {gps['stdev']:.1f}%")
            print(f"    Survival:   {surv * 100:.0f}%")
        print()
    else:
        sim = Simulation(sols=args.sols, env_seed=args.seed)

        def on_tick(sol: int, env: dict, colonies: list) -> None:
            if not args.quiet and sol % 50 == 0:
                pops = " | ".join(f"{c.name}: {c.population}" for c in colonies)
                storm = f" [{env['storm'].upper()}]" if env.get("storm") else ""
                print(f"  Sol {sol:>4}/{args.sols}  {pops}{storm}")

        print(f"Mars Barn — simulating {args.sols} sols with {len(sim.colonies)} colonies...")
        print()
        results = sim.run(callback=on_tick)

        print()
        print("=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        for s in results["summary"]["colonies"]:
            print(f"\n  {s['name']} ({s['strategy']})")
            print(f"    Population: {s['start_pop']} → {s['end_pop']} ({s['growth_pct']:+.1f}%)")
            print(f"    Peak: {s['peak_pop']}  |  Trough: {s['min_pop']}")
            mig = s.get('net_migration', 0)
            mig_str = f"  |  Migration: {mig:+d}" if mig != 0 else ""
            print(f"    Births: {s['total_births']}  |  Deaths: {s['total_deaths']}{mig_str}")
            dc = s.get("death_causes", {})
            active = {k: v for k, v in dc.items() if v > 0}
            if active:
                parts = [f"{k}: {v}" for k, v in sorted(active.items(), key=lambda x: -x[1])]
                print(f"    Death causes: {', '.join(parts)}")
            techs = s.get("techs_unlocked", 0)
            print(f"    Techs: {techs}")
        # Print tech details per colony
        for c in results["colonies"]:
            tech_data = c.get("tech")
            if tech_data and tech_data.get("unlocked"):
                print(f"\n  🔬 {c['name']} tech timeline:")
                for t in tech_data["unlocked"]:
                    print(f"    Sol {t['sol']:>4}: {t['name']} [{t['branch']}]")
        total_mig = results["summary"].get("total_migrations", 0)
        total_epidemics = sum(
            sum(1 for e in c.get("events", []) if e.get("type") == "epidemic_start")
            for c in results["colonies"]
        )
        print(f"\n  Total migrations: {total_mig}")
        print(f"  Total epidemics:  {total_epidemics}")
        tf = results["summary"].get("terraforming", {})
        tf_progress = tf.get("progress", 0)
        tf_phase = tf.get("phase") or "none"
        print(f"  Terraforming:     {tf_progress*100:.4f}% ({tf_phase})")
        contributions = tf.get("contributions", {})
        if contributions:
            for name, output in contributions.items():
                print(f"    {name}: {output:.6f}")
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
    html = generate_dashboard(results, mc_data=mc_data)
    html_path.write_text(html)
    print(f"Dashboard:    {html_path}")

    print()
    print(f"View at: https://kody-w.github.io/rappterbook-agent-exchange/mars/")


def _compact_results(results: dict) -> dict:
    """Strip heavy fields for the frontend data file."""
    colonies = []
    for c in results["colonies"]:
        death_causes_by_sol = [h.get("death_causes", {}) for h in c["history"]]
        colonies.append({
            "name": c["name"],
            "strategy": c["strategy"],
            "population": [h["population"] for h in c["history"]],
            "food_kg": [h["food_kg"] for h in c["history"]],
            "morale": [h["morale"] for h in c["history"]],
            "births": [h["births"] for h in c["history"]],
            "deaths": [h["deaths"] for h in c["history"]],
            "death_causes": death_causes_by_sol,
            "cumulative_death_causes": c.get("death_causes", {}),
            "carrying_capacity": [h.get("carrying_capacity", 0) for h in c["history"]],
            "genetic_diversity": [h.get("genetic_diversity", 1.0) for h in c["history"]],
            "net_migration": [h.get("net_migration", 0) for h in c["history"]],
            "habitability_index": [h.get("habitability_index", 0) for h in c["history"]],
            "tech": c.get("tech"),
        })
    env_temps = [e["temperature_c"] for e in results["environment"]["history"]]
    env_dust = [e["dust_opacity"] for e in results["environment"]["history"]]
    env_radiation = [e["radiation_msv"] for e in results["environment"]["history"]]
    return {
        "_meta": results["_meta"],
        "summary": results["summary"],
        "colonies": colonies,
        "environment": {
            "temperature_c": env_temps,
            "dust_opacity": env_dust,
            "radiation_msv": env_radiation,
            "terraforming_progress": [
                e.get("terraforming_progress", 0) for e in results["environment"]["history"]
            ],
            "pressure_kpa": [
                e.get("pressure_kpa", 0.636) for e in results["environment"]["history"]
            ],
        },
    }


def _serialize_ensemble(ensemble) -> dict:
    """Convert EnsembleResult to a JSON-serializable dict for the dashboard."""
    bands = []
    for ci in range(len(ensemble.colony_names)):
        colony_bands = {}
        for metric, metric_bands in ensemble.bands[ci].items():
            colony_bands[metric] = {
                f"p{PERCENTILES[pi]}": [round(v, 1) for v in metric_bands[pi]]
                for pi in range(len(PERCENTILES))
            }
        bands.append(colony_bands)
    return {
        "n_seeds": ensemble.n_seeds,
        "sols": ensemble.sols,
        "colony_names": ensemble.colony_names,
        "colony_strategies": ensemble.colony_strategies,
        "bands": bands,
        "final_pop_stats": [{k: round(v, 1) for k, v in fps.items()}
                            for fps in ensemble.final_pop_stats],
        "growth_pct_stats": [{k: round(v, 1) for k, v in gps.items()}
                             for gps in ensemble.growth_pct_stats],
        "survival_rates": [round(s, 3) for s in ensemble.survival_rates],
    }


if __name__ == "__main__":
    main()
