"""
run_python.py -- Execute terrarium + market via run_python action.

Usage:
    python src/run_python.py                       # default proof run
    python src/run_python.py --target market       # market only
    python src/run_python.py --target terrarium    # terrarium only
    python src/run_python.py --target both         # full pipeline (default)

Environment:
    STATE_DIR  -- where to save proof.json (default: state/)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_proof import (
    ProofReport,
    format_proof_compact,
    format_proof_text,
    run_proof,
)
from src.market_maker import (
    generate_predictions,
    resolve_predictions,
    run_terrarium_ensemble,
    score_predictions,
    build_calibration_curve,
    build_leaderboard,
)
from src.adaptive_market import (
    run_adaptive_market,
    format_adaptive_text,
    format_adaptive_compact,
)
from src.narrator import narrate, format_chronicle
from src.resilience import stress_test
from src.tick_engine import Simulation


def run_terrarium_only(sols: int = 365, seed: int = 42) -> dict:
    """Run just the terrarium, return summary dict + stdout text."""
    sim = Simulation(sols=sols, env_seed=seed)
    results = sim.run()

    lines = []
    lines.append("=" * 60)
    lines.append("  TERRARIUM EXECUTION PROOF")
    lines.append("=" * 60)
    lines.append("  %d sols | seed=%d | %d colonies" % (sols, seed, len(sim.colonies)))
    lines.append("")

    for s in results["summary"]["colonies"]:
        lines.append("  %-22s %-14s" % (s["name"], s["strategy"]))
        lines.append("    %d -> %d (%+.1f%%)" % (s["start_pop"], s["end_pop"], s["growth_pct"]))
        dc = s.get("death_causes", {})
        active = {k: v for k, v in dc.items() if v > 0}
        if active:
            parts = ["%s:%d" % (k, v) for k, v in sorted(active.items(), key=lambda x: -x[1])]
            lines.append("    Deaths: %s" % ", ".join(parts))

    for c in results["colonies"]:
        tech = c.get("tech", {})
        unlocked = tech.get("unlocked", [])
        if unlocked:
            lines.append("")
            lines.append("  Tech [%s]:" % c["name"])
            for t in unlocked:
                lines.append("    Sol %4d: %s [%s]" % (t["sol"], t["name"], t["branch"]))

    tf = results["summary"].get("terraforming", {})
    lines.append("")
    lines.append("  Terraforming: %.2f%% (%s)" % (
        tf.get("progress", 0) * 100, tf.get("phase", "none")))
    lines.append("=" * 60)

    return {"text": "\n".join(lines), "results": results}


def run_market_only(
    sols: int = 365,
    n_predictions: int = 100,
    n_seeds: int = 3,
    market_seed: int = 0,
) -> dict:
    """Run just the prediction market, return summary dict + stdout text."""
    seeds = list(range(42, 42 + n_seeds))
    ensemble = run_terrarium_ensemble(sols=sols, seeds=seeds)
    preds = generate_predictions(n_predictions, seed=market_seed)
    resolve_predictions(preds, ensemble)
    score_predictions(preds)

    calibration = build_calibration_curve(preds)
    leaderboard = build_leaderboard(preds)

    resolved = [p for p in preds if p.outcome is not None]
    briers = [p.brier for p in resolved if p.brier is not None]
    mean_brier = statistics.mean(briers) if briers else 0.0
    correct = sum(1 for p in resolved if (p.confidence >= 0.5) == p.outcome)
    accuracy = correct / max(len(resolved), 1)

    lines = []
    lines.append("=" * 60)
    lines.append("  PREDICTION MARKET EXECUTION PROOF")
    lines.append("=" * 60)
    lines.append("  %d predictions | %d seeds | %d sols" % (n_predictions, n_seeds, sols))
    lines.append("  Resolved: %d  Accuracy: %.1f%%  Mean Brier: %.4f" % (
        len(resolved), accuracy * 100, mean_brier))
    lines.append("")

    lines.append("  CALIBRATION:")
    for b in calibration:
        if b["count"] > 0:
            bar = "#" * int(b["actual_rate"] * 30)
            lines.append("  [%.1f-%.1f] n=%3d  pred=%.2f  actual=%.2f  %s" % (
                b["bucket_lo"], b["bucket_hi"], b["count"],
                b["mean_confidence"], b["actual_rate"], bar))

    lines.append("")
    lines.append("  TOP 5 AGENTS:")
    for row in leaderboard[:5]:
        lines.append("  %-22s %-12s brier=%.3f  karma=%+.1f" % (
            row["agent"], row["archetype"], row["mean_brier"], row["net_karma"]))
    lines.append("=" * 60)

    return {
        "text": "\n".join(lines),
        "mean_brier": mean_brier,
        "accuracy": accuracy,
        "n_resolved": len(resolved),
    }


def save_proof_state(proof: ProofReport, state_dir: Path) -> Path:
    """Save proof to state/proof.json with atomic write."""
    state_dir.mkdir(parents=True, exist_ok=True)
    proof_path = state_dir / "proof.json"
    data = proof.to_dict()

    history_path = state_dir / "proof_history.json"
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(data)
    history = history[-50:]

    tmp = proof_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(proof_path)

    htmp = history_path.with_suffix(".tmp")
    htmp.write_text(json.dumps(history, indent=2))
    htmp.rename(history_path)

    return proof_path


def main() -> int:
    """CLI entry point for run_python action."""
    parser = argparse.ArgumentParser(
        description="Execute terrarium + market, produce proof"
    )
    parser.add_argument("--target", choices=["both", "market", "terrarium", "adaptive",
                                              "narrative", "resilience"],
                        default="both", help="What to execute (default: both)")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--predictions", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--market-seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5,
                        help="Rounds for adaptive market (default: 5)")
    parser.add_argument("--agents", type=int, default=24,
                        help="Agents for adaptive market (default: 24)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--save", action="store_true",
                        help="Save proof to state/proof.json")
    parser.add_argument("--adaptive-cal", action="store_true",
                        help="Enable adaptive calibration from proof_history")
    args = parser.parse_args()

    state_dir = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))

    if args.target == "adaptive":
        report = run_adaptive_market(
            n_rounds=args.rounds,
            n_agents=args.agents,
            preds_per_agent=max(1, args.predictions // args.agents),
            sols=args.sols,
            n_seeds=args.seeds,
            market_seed=args.market_seed,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_adaptive_text(report))
        if args.save:
            state_dir.mkdir(parents=True, exist_ok=True)
            adaptive_path = state_dir / "adaptive_market.json"
            tmp = adaptive_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(report.to_dict(), indent=2))
            tmp.rename(adaptive_path)
            sys.stderr.write("Adaptive market saved: %s\n" % adaptive_path)
        return 0

    if args.target == "narrative":
        sim = Simulation(sols=args.sols, env_seed=42)
        results = sim.run()
        chronicle = narrate(results)
        if args.json:
            print(json.dumps(chronicle.to_dict(), indent=2))
        else:
            print(format_chronicle(chronicle))
        return 0

    if args.target == "resilience":
        report = stress_test(sols=args.sols, env_seed=42)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.summary())
        return 0

    if args.target == "terrarium":
        result = run_terrarium_only(sols=args.sols, seed=42)
        if args.json:
            print(json.dumps({"target": "terrarium", "sols": args.sols}, indent=2))
        else:
            print(result["text"])
        return 0

    if args.target == "market":
        result = run_market_only(
            sols=args.sols,
            n_predictions=args.predictions,
            n_seeds=args.seeds,
            market_seed=args.market_seed,
        )
        if args.json:
            print(json.dumps({
                "target": "market",
                "mean_brier": result["mean_brier"],
                "accuracy": result["accuracy"],
                "n_resolved": result["n_resolved"],
            }, indent=2))
        else:
            print(result["text"])
        return 0

    report = run_proof(
        sols=args.sols,
        n_predictions=args.predictions,
        n_seeds=args.seeds,
        market_seed=args.market_seed,
        adaptive=args.adaptive_cal,
        state_dir=str(state_dir),
    )

    if args.save:
        proof_path = save_proof_state(report, state_dir)
        sys.stderr.write("Proof saved: %s\n" % proof_path)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_proof_text(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
