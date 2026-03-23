"""
cross_sim.py — Symbiotic bridge: Terrarium × Prediction Market.

Multi-round integration: each round runs a terrarium segment, extracts
signals, then feeds those into the prediction market pipeline. The market
becomes a real-time tracking system for colony fate.

The terrarium is the WORLD. The market is the MIND reading it.

Usage:
    python3 src/cross_sim.py
    python3 src/cross_sim.py --sols 365 --rounds 5 --seed 42
    python3 src/cross_sim.py --quiet --json
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tick_engine import Simulation, DEFAULT_COLONIES
from src.market_maker import (
    Prediction, MarketState,
    generate_predictions, generate_counter_positions,
    resolve_predictions, score_predictions, compute_payouts,
    build_calibration_curve, build_agent_scores,
    brier_score, run_market,
)


# ---------------------------------------------------------------------------
# Signal extraction — what observers see between rounds
# ---------------------------------------------------------------------------

def extract_signals(terrarium_results: dict) -> dict:
    """Extract observable signals from a terrarium run.

    Returns dict of numeric signals about each colony.
    """
    signals: dict = {}
    colonies = terrarium_results.get("colonies", [])
    summary = terrarium_results.get("summary", {})

    for c in colonies:
        name = c["name"]
        history = c.get("history", [])
        if history:
            pops = [h["population"] for h in history]
            signals[f"{name}_pop_trend"] = round(
                (pops[-1] - pops[0]) / max(1, len(pops)), 4
            )
            signals[f"{name}_final_pop"] = pops[-1]
            signals[f"{name}_peak_pop"] = max(pops)
            signals[f"{name}_morale"] = round(history[-1].get("morale", 0.5), 3)
            signals[f"{name}_food_kg"] = round(history[-1].get("food_kg", 0), 1)
            signals[f"{name}_k"] = history[-1].get("carrying_capacity", 0)

        events = c.get("events", [])
        signals[f"{name}_had_epidemic"] = any(
            e.get("type") == "epidemic_start" for e in events
        )
        signals[f"{name}_epidemics"] = sum(
            1 for e in events if e.get("type") == "epidemic_start"
        )
        tech = c.get("tech", {})
        signals[f"{name}_techs"] = tech.get("unlocked_count", 0)

    signals["total_migrations"] = summary.get("total_migrations", 0)

    # Environment signals
    env_history = terrarium_results.get("environment", {}).get("history", [])
    signals["had_global_storm"] = any(
        e.get("storm") == "global" for e in env_history
    )
    signals["flare_count"] = sum(1 for e in env_history if e.get("flare", False))
    signals["max_dust"] = max(
        (e.get("dust_opacity", 0) for e in env_history), default=0
    )

    return signals


# ---------------------------------------------------------------------------
# Cross-sim prediction adjustment
# ---------------------------------------------------------------------------

def adjust_confidences(
    predictions: list[Prediction],
    signals: dict,
    adjustment_strength: float = 0.15,
) -> list[Prediction]:
    """Adjust prediction confidences based on terrarium signals.

    Agents who observe interim data update their beliefs. This is the
    bridge — the terrarium output modifies market beliefs in real time.
    """
    rng = random.Random(42)

    for pred in predictions:
        text = pred.text.lower()
        author = pred.author
        original_conf = pred.confidence

        # Signal-based adjustments
        delta = 0.0

        # Population signals
        for name in ["Ares Prime", "Olympus Station", "Red Frontier"]:
            trend = signals.get(f"{name}_pop_trend", 0)
            final_pop = signals.get(f"{name}_final_pop", 0)
            name_lower = name.lower()

            if name_lower in text:
                if "exceed" in text or "population" in text:
                    if trend > 0.1:
                        delta += adjustment_strength * 0.5
                    elif trend < -0.05:
                        delta -= adjustment_strength * 0.3

                if "survive" in text:
                    if final_pop > 30:
                        delta += adjustment_strength * 0.3
                    elif final_pop < 10:
                        delta -= adjustment_strength * 0.5

                if "morale" in text:
                    morale = signals.get(f"{name}_morale", 0.5)
                    if morale > 0.65:
                        delta += adjustment_strength * 0.3
                    elif morale < 0.4:
                        delta -= adjustment_strength * 0.3

        # Epidemic signals
        if "epidemic" in text:
            any_epidemic = any(
                signals.get(f"{n}_had_epidemic", False)
                for n in ["Ares Prime", "Olympus Station", "Red Frontier"]
            )
            if any_epidemic:
                delta += adjustment_strength * 0.4
            else:
                delta -= adjustment_strength * 0.2

        # Storm signals
        if "storm" in text or "dust" in text:
            if signals.get("had_global_storm", False):
                delta += adjustment_strength * 0.5
            elif signals.get("max_dust", 0) > 0.3:
                delta += adjustment_strength * 0.2

        # Flare signals
        if "flare" in text:
            if signals.get("flare_count", 0) > 0:
                delta += adjustment_strength * 0.4

        # Tech signals
        if "tech" in text or "unlock" in text:
            max_techs = max(
                signals.get(f"{n}_techs", 0)
                for n in ["Ares Prime", "Olympus Station", "Red Frontier"]
            )
            if max_techs >= 3:
                delta += adjustment_strength * 0.3

        # Add some noise per agent
        noise = rng.gauss(0, 0.03)
        new_conf = max(0.01, min(0.99, original_conf + delta + noise))
        pred.confidence = round(new_conf, 4)

    return predictions


# ---------------------------------------------------------------------------
# Cross-sim engine
# ---------------------------------------------------------------------------

@dataclass
class CrossSimResult:
    """Results from the cross-simulation bridge."""
    rounds: list[dict]
    final_market: dict
    consensus_accuracy: dict
    signal_history: list[dict]
    terrarium_summary: dict


def run_cross_sim(
    total_sols: int = 365,
    n_rounds: int = 5,
    n_predictions: int = 100,
    seed: int = 42,
    quiet: bool = False,
) -> dict:
    """Run the full cross-simulation bridge.

    Each round:
    1. Run terrarium for sols_per_round sols
    2. Extract signals
    3. Adjust prediction confidences based on signals
    4. After all rounds, run full sim for final resolution
    """
    sols_per_round = max(1, total_sols // n_rounds)
    colony_names = [c[0] for c in DEFAULT_COLONIES]

    if not quiet:
        print(f"Cross-Sim Bridge — {n_rounds} rounds × {sols_per_round} sols/round")
        print(f"Predictions: {n_predictions}  Colonies: {', '.join(colony_names)}")
        print()

    # Generate initial predictions
    predictions = generate_predictions(n_predictions, colony_names, seed)
    predictions = generate_counter_positions(predictions, seed)

    round_log = []
    signal_history = []

    # Multi-round: observe → update beliefs
    for round_num in range(1, n_rounds + 1):
        env_seed = seed + round_num * 7
        sim = Simulation(sols=sols_per_round, env_seed=env_seed)
        results = sim.run()
        signals = extract_signals(results)
        signal_history.append({
            "round": round_num,
            "signals": {k: v for k, v in signals.items()
                        if not isinstance(v, bool)},
        })

        # Adjust confidences based on observed signals
        strength = 0.10 + 0.02 * round_num  # strengthen with more data
        predictions = adjust_confidences(predictions, signals, strength)

        col_pops = {
            c["name"]: c.get("final_population", 0) for c in results["colonies"]
        }
        round_log.append({
            "round": round_num,
            "sols": sols_per_round,
            "env_seed": env_seed,
            "colony_pops": col_pops,
            "avg_confidence": round(
                statistics.mean(p.confidence for p in predictions), 4
            ),
        })

        if not quiet:
            pop_str = " | ".join(f"{n[:5]}: {p}" for n, p in col_pops.items())
            avg_c = round_log[-1]["avg_confidence"]
            print(f"  Round {round_num}/{n_rounds}  "
                  f"pops: [{pop_str}]  avg_conf: {avg_c:.1%}")

    # Final full simulation for resolution
    if not quiet:
        print(f"\n  Running full {total_sols}-sol sim for resolution...")
    full_sim = Simulation(sols=total_sols, env_seed=seed)
    final_results = full_sim.run()

    # Resolve, score, payout
    predictions = resolve_predictions(predictions, final_results)
    predictions = score_predictions(predictions)
    predictions = compute_payouts(predictions)
    agent_scores = build_agent_scores(predictions)
    calibration = build_calibration_curve(predictions)

    # Consensus accuracy
    resolved = [p for p in predictions if p.outcome is not None]
    correct = sum(
        1 for p in resolved
        if (p.confidence > 0.5 and p.outcome == 1.0)
        or (p.confidence <= 0.5 and p.outcome == 0.0)
    )
    consensus = {
        "total": len(resolved),
        "correct": correct,
        "accuracy": round(correct / max(1, len(resolved)), 4),
    }

    # Build results
    market = MarketState(
        predictions=predictions,
        agent_scores=agent_scores,
        calibration_curve=calibration,
        colony_names=colony_names,
    )
    market_dict = market.to_dict()

    terrarium_summary = final_results.get("summary", {})

    now = datetime.now(timezone.utc).isoformat()
    output = {
        "_meta": {
            "engine": "cross-sim-bridge",
            "version": "1.0",
            "total_sols": total_sols,
            "rounds": n_rounds,
            "sols_per_round": sols_per_round,
            "seed": seed,
            "generated": now,
        },
        "consensus_accuracy": consensus,
        "round_log": round_log,
        "signal_history": signal_history,
        "market": market_dict,
        "terrarium_summary": terrarium_summary,
    }

    if not quiet:
        _print_results(output, predictions, agent_scores, calibration)

    return output


def _print_results(
    output: dict,
    predictions: list[Prediction],
    agent_scores: dict,
    calibration: list[dict],
) -> None:
    """Print human-readable results."""
    print()
    print("=" * 64)
    print("CROSS-SIM BRIDGE RESULTS")
    print("=" * 64)

    ca = output["consensus_accuracy"]
    print(f"\n  🎯 CONSENSUS vs REALITY")
    print(f"    Correct: {ca['correct']}/{ca['total']}")
    print(f"    Accuracy: {ca['accuracy'] * 100:.1f}%")

    resolved = [p for p in predictions if p.outcome is not None]
    scored = [p for p in resolved if p.brier is not None]
    if scored:
        avg_brier = statistics.mean(p.brier for p in scored)
        print(f"    Mean Brier: {avg_brier:.4f}")

    print(f"\n  🏆 AGENT LEADERBOARD (by accuracy)")
    print("  " + "-" * 58)
    leaderboard = sorted(
        agent_scores.values(), key=lambda a: (-a.accuracy, a.mean_brier)
    )
    for a in leaderboard:
        print(f"    {a.name:<25} acc: {a.accuracy:.0%}  "
              f"brier: {a.mean_brier:.3f}  "
              f"roi: {a.roi:+.1f}%")

    print(f"\n  📈 CALIBRATION")
    print("  " + "-" * 58)
    for b in calibration:
        if b["actual_rate"] is not None:
            gap = b["actual_rate"] - b["stated_avg"]
            print(f"    {b['bucket']:<12} n={b['count']:>3}  "
                  f"stated: {b['stated_avg']:.0%}  "
                  f"actual: {b['actual_rate']:.0%}  "
                  f"gap: {gap:+.0%}")

    ts = output.get("terrarium_summary", {})
    if "colonies" in ts:
        print(f"\n  🔴🔵🟢 TERRARIUM FINAL STATE")
        for c in ts["colonies"]:
            print(f"    {c['name']:<20} {c['start_pop']} → {c['end_pop']} "
                  f"({c['growth_pct']:+.1f}%)")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Run cross-sim bridge from CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-Sim Bridge — Terrarium × Prediction Market"
    )
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--predictions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--state-dir", type=str, default=None)
    args = parser.parse_args()

    results = run_cross_sim(
        total_sols=args.sols,
        n_rounds=args.rounds,
        n_predictions=args.predictions,
        seed=args.seed,
        quiet=args.quiet or args.json,
    )

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        return

    if args.state_dir:
        state_dir = Path(args.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        out = state_dir / "cross_sim.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=2))
        tmp.rename(out)
        print(f"State saved: {out}")


if __name__ == "__main__":
    main()
