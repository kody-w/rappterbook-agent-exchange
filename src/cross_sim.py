"""Cross-simulation bridge: Mars Barn terrarium ↔ prediction market.

Runs Monte Carlo terrarium ensemble, feeds outcomes into the prediction
market engine, computes cross-validated Brier scores, and produces a
unified report comparing market forecasts against simulation reality.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from src.market_maker import (
    AGENT_ARCHETYPES,
    Prediction,
    brier_score,
    build_calibration_curve,
    build_leaderboard,
    generate_predictions,
    resolve_predictions,
    run_terrarium_ensemble,
    score_predictions,
)
from src.monte_carlo import run_ensemble
from src.tick_engine import Simulation


# ---------------------------------------------------------------------------
# Cross-sim metrics
# ---------------------------------------------------------------------------

def consensus_accuracy(predictions: list[Prediction]) -> float:
    """Fraction of resolved predictions where majority confidence > 0.5 matched outcome."""
    resolved = [p for p in predictions if p.outcome is not None]
    if not resolved:
        return 0.0
    correct = sum(
        1 for p in resolved
        if (p.confidence > 0.5 and p.outcome == 1)
        or (p.confidence <= 0.5 and p.outcome == 0)
    )
    return correct / len(resolved)


def ensemble_agreement(seeds: list[int], sols: int = 365) -> float:
    """Run ensemble and return coefficient of variation of final total population."""
    pops = []
    for seed in seeds:
        sim = Simulation(sols=sols, env_seed=seed)
        results = sim.run()
        total = sum(c["final_population"] for c in results["colonies"])
        pops.append(total)
    if not pops or statistics.mean(pops) == 0:
        return 0.0
    return statistics.stdev(pops) / statistics.mean(pops) if len(pops) > 1 else 0.0


@dataclass
class CrossSimReport:
    """Unified report from cross-simulation analysis."""
    n_predictions: int = 0
    n_resolved: int = 0
    consensus_acc: float = 0.0
    mean_brier: float = 0.0
    ensemble_cv: float = 0.0
    category_briers: dict[str, float] = field(default_factory=dict)
    calibration: list[dict] = field(default_factory=list)
    leaderboard: list[dict] = field(default_factory=list)
    terrarium_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_predictions": self.n_predictions,
            "n_resolved": self.n_resolved,
            "consensus_accuracy": self.consensus_acc,
            "mean_brier": self.mean_brier,
            "ensemble_cv": self.ensemble_cv,
            "category_briers": self.category_briers,
            "calibration": self.calibration,
            "leaderboard": self.leaderboard,
            "terrarium_summary": self.terrarium_summary,
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_cross_sim(
    n_predictions: int = 100,
    sols: int = 365,
    seeds: list[int] | None = None,
    rng_seed: int = 42,
) -> CrossSimReport:
    """Run the full cross-simulation pipeline.

    1. Generate predictions from diverse agent archetypes
    2. Run terrarium ensemble across multiple seeds
    3. Resolve predictions against ensemble outcomes (majority vote)
    4. Score with Brier rule
    5. Compute cross-sim metrics (consensus accuracy, ensemble CV)
    6. Build calibration curve and leaderboard
    """
    if seeds is None:
        seeds = [42, 43, 44]

    rng = random.Random(rng_seed)

    # Step 1: Generate predictions
    predictions = generate_predictions(n_predictions, seed=rng_seed)

    # Step 2: Run terrarium ensemble and resolve
    ensemble_results = run_terrarium_ensemble(sols=sols, seeds=seeds)
    resolved = resolve_predictions(predictions, ensemble_results)

    # Step 3: Score
    scored = score_predictions(resolved)

    # Step 4: Cross-sim metrics
    report = CrossSimReport()
    report.n_predictions = len(scored)
    report.n_resolved = sum(1 for p in scored if p.outcome is not None)
    report.consensus_acc = consensus_accuracy(scored)

    briers = [p.brier for p in scored if p.brier is not None]
    report.mean_brier = statistics.mean(briers) if briers else 0.0

    report.ensemble_cv = ensemble_agreement(seeds, sols)

    # Category breakdown
    cats: dict[str, list[float]] = {}
    for p in scored:
        if p.brier is not None:
            cats.setdefault(p.category, []).append(p.brier)
    report.category_briers = {
        cat: statistics.mean(vals) for cat, vals in sorted(cats.items())
    }

    # Calibration + leaderboard
    report.calibration = build_calibration_curve(scored)
    report.leaderboard = build_leaderboard(scored)

    # Terrarium summary from first seed
    sim = Simulation(sols=sols, env_seed=seeds[0])
    results = sim.run()
    report.terrarium_summary = {
        "colonies": [
            {
                "name": c["name"],
                "end_pop": c["final_population"],
                "growth_pct": ((c["final_population"] - c["initial_population"]) / max(c["initial_population"], 1)) * 100,
                "techs_unlocked": len(c.get("tech", {}).get("unlocked", [])),
            }
            for c in results["colonies"]
        ],
        "total_migrations": sum(c["total_immigrants"] + c["total_emigrants"] for c in results["colonies"]) // 2,
    }

    return report


def print_report(report: CrossSimReport) -> None:
    """Pretty-print a cross-simulation report to stdout."""
    print("=" * 60)
    print("  CROSS-SIM BRIDGE — Mars Barn × Prediction Market")
    print("=" * 60)
    print(f"  Predictions: {report.n_predictions}  Resolved: {report.n_resolved}")
    print(f"  Consensus accuracy: {report.consensus_acc:.1%}")
    print(f"  Mean Brier: {report.mean_brier:.4f}")
    print(f"  Ensemble CV: {report.ensemble_cv:.4f}")
    print()

    print("  CATEGORY BRIER SCORES")
    print("  " + "-" * 50)
    for cat, brier in report.category_briers.items():
        bar = "#" * int((1 - brier) * 30)
        print(f"  {cat:<22} {brier:.3f}  {bar}")
    print()

    print("  CALIBRATION")
    print("  " + "-" * 50)
    for b in report.calibration:
        bar = "#" * int(b["actual_rate"] * 30) if b["count"] > 0 else ""
        print(f"  [{b['bucket_lo']:.1f}-{b['bucket_hi']:.1f}] n={b['count']:3d}  pred={b['mean_confidence']:.2f}  actual={b['actual_rate']:.2f}  {bar}")
    print()

    print("  TOP 5 AGENTS")
    print("  " + "-" * 50)
    for row in report.leaderboard[:5]:
        print(f"  {row['agent']:<22} {row['archetype']:<12} brier={row['mean_brier']:.3f}")
    print()

    print("  TERRARIUM STATE")
    print("  " + "-" * 50)
    for c in report.terrarium_summary.get("colonies", []):
        status = "ALIVE" if c["end_pop"] > 0 else "DEAD"
        print(f"  {c['name']:<22} {status}  pop={c['end_pop']}  growth={c['growth_pct']:+.1f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-simulation bridge")
    parser.add_argument("--predictions", type=int, default=100)
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    report = run_cross_sim(
        n_predictions=args.predictions,
        sols=args.sols,
        seeds=args.seeds,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)
