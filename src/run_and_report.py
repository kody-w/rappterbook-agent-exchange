#!/usr/bin/env python3
"""
run_and_report.py — Execute Mars Barn + Prediction Market, print proof-of-execution.

This is the entry point for the run_python action. It:
1. Runs the Mars Barn terrarium (365 sols, 3 colonies)
2. Runs the prediction market (12 markets, 15 agents, 20 rounds)
3. Resolves markets against actual sim results
4. Prints structured stdout as proof

Usage:
    python src/run_and_report.py
    python src/run_and_report.py --sols 668 --seed 99
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

from src.tick_engine import Simulation
from src.market_maker import create_default_markets, PredictionEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mars Barn + Prediction Market")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", type=int, default=15, help="Number of betting agents")
    parser.add_argument("--rounds", type=int, default=20, help="Betting rounds")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    # --- 1. Run Mars Barn terrarium ---
    print("=" * 70)
    print("  MARS BARN TERRARIUM — EXECUTION PROOF")
    print("=" * 70)
    print(f"\n  Seed: {args.seed}  |  Sols: {args.sols}  |  Colonies: 3")
    print()

    sim = Simulation(sols=args.sols, env_seed=args.seed)
    results = sim.run()

    print("  COLONY RESULTS:")
    print("  " + "-" * 66)
    for s in results["summary"]["colonies"]:
        print(f"  🔴 {s['name']:20s} ({s['strategy']:12s})  "
              f"{s['start_pop']:>4d} → {s['end_pop']:>4d}  "
              f"({s['growth_pct']:+6.1f}%)")
        dc = s.get("death_causes", {})
        active = {k: v for k, v in dc.items() if v > 0}
        if active:
            parts = [f"{k}:{v}" for k, v in sorted(active.items(), key=lambda x: -x[1])]
            print(f"    Deaths: {', '.join(parts)}")

    # Tech timelines
    for c in results["colonies"]:
        tech = c.get("tech", {})
        if tech and tech.get("unlocked"):
            print(f"\n  🔬 {c['name']} tech unlocks:")
            for t in tech["unlocked"]:
                print(f"      Sol {t['sol']:>4d}: {t['name']} [{t['branch']}]")

    total_mig = results["summary"].get("total_migrations", 0)
    print(f"\n  Total migrations:  {total_mig}")
    epidemics = sum(
        sum(1 for e in c.get("events", []) if e.get("type") == "epidemic_start")
        for c in results["colonies"]
    )
    print(f"  Total epidemics:   {epidemics}")
    print()

    # --- 2. Run Prediction Market ---
    print("=" * 70)
    print("  PREDICTION MARKET — EXECUTION PROOF")
    print("=" * 70)

    archetypes = ["philosopher", "coder", "contrarian", "wildcard", "researcher"]
    agent_ids = [
        f"zion-{arch}-{i:02d}"
        for arch in archetypes
        for i in range(1, args.agents // len(archetypes) + 2)
    ][:args.agents]

    print(f"\n  Agents: {len(agent_ids)}  |  Markets: 12  |  Rounds: {args.rounds}")
    print()

    markets = create_default_markets()
    engine = PredictionEngine(markets, agent_ids, seed=args.seed)

    # Show pre-resolution prices (all 0.5)
    print("  PRE-BET PRICES (all markets start at 50%):")
    print("  " + "-" * 66)

    engine.run_betting_rounds(n_rounds=args.rounds)

    print("\n  POST-BET / PRE-RESOLUTION PRICES:")
    print("  " + "-" * 66)
    for m in engine.markets.values():
        bar_len = int(m.price_yes() * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {m.price_yes()*100:5.1f}% {bar} {m.question[:50]}")

    print()
    outcomes = engine.resolve_all(results)

    print("  MARKET RESOLUTIONS:")
    print("  " + "-" * 66)
    correct_calls = 0
    total_markets = 0
    for m in engine.markets.values():
        icon = "✅" if m.outcome else "❌"
        confidence = m.history[-1]["price_yes"] if m.history else 0.5
        was_right = (confidence > 0.5 and m.outcome) or (confidence <= 0.5 and not m.outcome)
        if was_right:
            correct_calls += 1
        total_markets += 1
        print(f"  {icon} {m.question[:55]:55s}  "
              f"implied={confidence*100:5.1f}%  actual={'YES' if m.outcome else 'NO'}")

    accuracy = correct_calls / max(total_markets, 1) * 100
    print(f"\n  Market accuracy: {correct_calls}/{total_markets} ({accuracy:.0f}%)")

    print()
    print("  LEADERBOARD (Top 10 by P&L):")
    print("  " + "-" * 66)
    print(f"  {'Rank':>4s}  {'Agent':25s}  {'Archetype':12s}  {'P&L':>8s}  {'ROI':>7s}")
    for i, entry in enumerate(engine.leaderboard()[:10], 1):
        print(f"  {i:>4d}  {entry['agent_id']:25s}  {entry['archetype']:12s}  "
              f"{entry['pnl']:>+8.2f}  {entry['roi']:>+6.1f}%")

    print()
    print("  ARCHETYPE PERFORMANCE:")
    print("  " + "-" * 66)
    arch_perf = engine._archetype_performance()
    for arch, data in sorted(arch_perf.items(), key=lambda x: -x[1]["mean_pnl"]):
        print(f"    {arch:15s}  mean_pnl={data['mean_pnl']:>+7.2f}  "
              f"best={data['best_pnl']:>+7.2f}  worst={data['worst_pnl']:>+7.2f}  "
              f"n={data['count']}")

    print()
    print("=" * 70)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"  Execution complete at {ts}")
    print(f"  Mars Barn: {args.sols} sols, seed={args.seed}")
    print(f"  Prediction Market: {len(engine.trade_log)} trades across {args.rounds} rounds")
    print("=" * 70)

    if args.json:
        combined = {
            "mars_barn": {
                "summary": results["summary"],
                "_meta": results["_meta"],
            },
            "prediction_market": engine.full_results(),
        }
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
