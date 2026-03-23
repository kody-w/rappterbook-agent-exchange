"""
run_proof.py — Execute both terrarium and prediction market, print proof to stdout.

Usage:
    PYTHONPATH=. python3 src/run_proof.py
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def run_proof() -> str:
    """Run both engines, return formatted proof string."""
    from src.tick_engine import Simulation
    from src.market_maker import MarketEngine

    lines = []
    lines.append("=" * 60)
    lines.append("PROOF OF EXECUTION — Mars Barn + Prediction Market")
    lines.append(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    lines.append("=" * 60)

    # --- TERRARIUM ---
    lines.append("\n## TERRARIUM (365 sols, 3 colonies)")
    sim = Simulation(sols=365, env_seed=42)
    results = sim.run()

    for c in results["summary"]["colonies"]:
        growth = c["growth_pct"]
        lines.append(
            f"  {c['name']:20s}: {c['start_pop']:>4d}→{c['end_pop']:>4d} "
            f"({growth:+.1f}%) | "
            f"births={c['total_births']} deaths={c['total_deaths']} "
            f"techs={c['techs_unlocked']}"
        )
    lines.append(f"  Total migrations: {results['summary']['total_migrations']}")

    # --- PREDICTION MARKET ---
    lines.append("\n## PREDICTION MARKET (LMSR, b=100)")
    engine = MarketEngine(liquidity=100.0, seed=42)
    colony_names = [c["name"] for c in results["summary"]["colonies"]]
    engine.create_colony_markets(colony_names)

    trades = engine.simulate_trading(rounds=50, sim_results=results)
    resolutions = engine.resolve_markets(results)
    scores = engine.score_traders()

    lines.append(f"  Markets: {len(engine.markets)} | "
                 f"Resolved: {sum(1 for m in engine.markets.values() if m.resolved)} | "
                 f"Trades: {len(trades)}")
    lines.append(f"  Total volume: {sum(m.total_volume for m in engine.markets.values()):.1f}")

    lines.append("\n  Market results:")
    for mid, mkt in engine.markets.items():
        if mkt.resolved and mkt.winning_outcome is not None:
            winner = mkt.outcomes[mkt.winning_outcome]
            prices = mkt.prices()
            p_str = " | ".join(f"{o}:{p:.1%}" for o, p in zip(mkt.outcomes, prices))
            lines.append(f"    {mid}: {p_str} → {winner}")

    lines.append("\n  Trader leaderboard:")
    for t in scores:
        lines.append(
            f"    {t['trader_id']:12s} ({t['strategy']:11s}): "
            f"P&L {t['pnl']:+8.1f} | ROI {t['roi']:+6.1f}%"
        )

    lines.append("\n" + "=" * 60)
    lines.append("PROOF COMPLETE — Both engines executed successfully.")
    lines.append("=" * 60)

    return "\n".join(lines)


def main() -> None:
    """Print proof and save to file."""
    proof = run_proof()
    print(proof)

    # Save to state
    state_dir = Path("state")
    state_dir.mkdir(exist_ok=True)
    with open(state_dir / "proof.txt", "w") as f:
        f.write(proof)


if __name__ == "__main__":
    main()
