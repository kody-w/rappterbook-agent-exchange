#!/usr/bin/env python3
"""
run_market.py — Execute Mars Barn + Prediction Market, print stdout proof.

Runs both engines end-to-end:
  1. Mars Barn terrarium simulation (N sols)
  2. LMSR prediction market (12 markets, M agents, K rounds)
  3. Market resolution against sim results
  4. Leaderboard + archetype performance

Usage:
    python src/run_market.py
    python src/run_market.py --sols 365 --seed 42 --rounds 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tick_engine import Simulation
from src.market_maker import PredictionEngine, create_default_markets


AGENT_ARCHETYPES = ["philosopher", "coder", "contrarian", "wildcard", "researcher"]


def build_agent_ids(agents_per_archetype: int = 3) -> list[str]:
    """Create deterministic agent IDs from archetypes."""
    return [
        f"zion-{arch}-{i:02d}"
        for arch in AGENT_ARCHETYPES
        for i in range(1, agents_per_archetype + 1)
    ]


def run_terrarium(sols: int, seed: int, quiet: bool = False) -> dict:
    """Phase 1: Run the Mars Barn terrarium simulation."""
    sim = Simulation(sols=sols, env_seed=seed)

    def on_tick(sol: int, env: dict, colonies: list) -> None:
        if not quiet and sol % 100 == 0:
            pops = " | ".join(f"{c.name}: {c.population}" for c in colonies)
            print(f"  Sol {sol:>4}/{sols}  {pops}")

    results = sim.run(callback=on_tick)
    return results


def run_prediction_market(
    sim_results: dict,
    agent_ids: list[str],
    rounds: int = 20,
    seed: int = 42,
) -> dict:
    """Phase 2+3: Run prediction market and resolve against sim results."""
    markets = create_default_markets()
    engine = PredictionEngine(markets, agent_ids, seed=seed)
    engine.run_betting_rounds(n_rounds=rounds)
    engine.resolve_all(sim_results)
    return engine.full_results(), engine


def print_terrarium_summary(results: dict) -> None:
    """Print terrarium simulation summary."""
    for s in results["summary"]["colonies"]:
        print(f"  {s['name']} ({s['strategy']}): "
              f"{s['start_pop']} → {s['end_pop']} ({s['growth_pct']:+.1f}%)")
        dc = s.get("death_causes", {})
        active = {k: v for k, v in dc.items() if v > 0}
        if active:
            parts = [f"{k}: {v}" for k, v in sorted(active.items(), key=lambda x: -x[1])]
            print(f"    Deaths: {', '.join(parts)}")
    for c in results["colonies"]:
        tech_data = c.get("tech")
        if tech_data and tech_data.get("unlocked"):
            techs = " → ".join(
                f"Sol {t['sol']}: {t['name']}" for t in tech_data["unlocked"]
            )
            print(f"  🔬 {c['name']}: {techs}")
    total_mig = results["summary"].get("total_migrations", 0)
    total_epi = sum(
        sum(1 for e in c.get("events", []) if e.get("type") == "epidemic_start")
        for c in results["colonies"]
    )
    print(f"  Migrations: {total_mig} | Epidemics: {total_epi}")


def print_market_results(engine) -> None:
    """Print prediction market results."""
    for m in engine.markets.values():
        emoji = "✅" if m.outcome else "❌"
        surprise = abs(m.price_yes() - (1.0 if m.outcome else 0.0))
        flag = " ⚡" if surprise > 0.4 else ""
        print(f"  {emoji} [{m.price_yes():.0%}] {m.question}{flag}")

    print()
    print("  LEADERBOARD")
    board = engine.leaderboard()
    print(f"  {'#':<3} {'Agent':<25} {'P&L':>8} {'ROI':>8} {'Pos':>4}")
    for i, entry in enumerate(board[:10], 1):
        print(f"  {i:<3} {entry['agent_id']:<25} {entry['pnl']:>+8.2f} "
              f"{entry['roi']:>+7.1f}% {entry['num_positions']:>4}")

    print()
    print("  ARCHETYPES")
    full = engine.full_results()
    for arch, perf in full["archetype_performance"].items():
        print(f"  {arch:<15} mean: {perf['mean_pnl']:>+7.2f}  "
              f"best: {perf['best_pnl']:>+7.2f}  worst: {perf['worst_pnl']:>+7.2f}")


def main() -> None:
    """Run both engines and print execution proof."""
    parser = argparse.ArgumentParser(description="Mars Barn + Prediction Market")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save", action="store_true",
                        help="Save results to state/ and docs/")
    args = parser.parse_args()

    agent_ids = build_agent_ids()

    # Phase 1: Terrarium
    print(f"{'='*60}")
    print(f"MARS BARN TERRARIUM ({args.sols} sols, seed {args.seed})")
    print(f"{'='*60}")
    print()
    sim_results = run_terrarium(args.sols, args.seed, args.quiet)
    print()
    print_terrarium_summary(sim_results)

    # Phase 2+3: Prediction Market
    print()
    print(f"{'='*60}")
    print(f"PREDICTION MARKET ({args.rounds} rounds, {len(agent_ids)} agents, 12 markets)")
    print(f"{'='*60}")
    print()
    full_results, engine = run_prediction_market(
        sim_results, agent_ids, args.rounds, args.seed,
    )
    meta = full_results["_meta"]
    print(f"  Trades: {meta['num_trades']} | Markets: {meta['num_markets']} | "
          f"Bettors: {meta['num_bettors']}")
    print()
    print_market_results(engine)

    # Save if requested
    if args.save:
        from src.mars_curves import generate_dashboard
        state_dir = REPO_ROOT / "state"
        docs_dir = REPO_ROOT / "docs" / "mars"
        state_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Save terrarium state
        mars_path = state_dir / "mars.json"
        tmp = mars_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sim_results, indent=2))
        tmp.rename(mars_path)

        # Save market state
        mkt_path = state_dir / "market.json"
        tmp = mkt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(full_results, indent=2))
        tmp.rename(mkt_path)

        # Generate dashboard
        html = generate_dashboard(sim_results)
        (docs_dir / "index.html").write_text(html)

        print(f"\n  Saved: {mars_path}")
        print(f"  Saved: {mkt_path}")
        print(f"  Saved: {docs_dir / 'index.html'}")

    print()
    print(f"{'='*60}")
    print("EXECUTION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
