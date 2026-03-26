"""
run_proof.py -- Execution bridge: runs both engines, posts proof.

Runs the Mars Barn terrarium AND the prediction market in a single pipeline,
computes Collective Intelligence (CI) score, and outputs machine-verifiable
proof-of-execution.

Usage:
    python -m src.run_proof --sols 365 --predictions 200 --seeds 3
    python -m src.run_proof --json
    python -m src.run_proof --compact
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.market_maker import (
    Prediction,
    brier_score,
    build_calibration_curve,
    build_leaderboard,
    generate_predictions,
    resolve_predictions,
    run_terrarium,
    run_terrarium_ensemble,
    score_predictions,
)
from src.tick_engine import Simulation


@dataclass
class CIScore:
    """Collective Intelligence measurement."""
    score: float = 0.0
    calibration_error: float = 0.0
    information_ratio: float = 0.0
    category_spread: float = 0.0
    n_resolved: int = 0
    n_correct: int = 0

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "calibration_error": round(self.calibration_error, 4),
            "information_ratio": round(self.information_ratio, 4),
            "category_spread": round(self.category_spread, 4),
            "n_resolved": self.n_resolved,
            "n_correct": self.n_correct,
            "label": self.label(),
        }

    def label(self) -> str:
        """Human-readable label for CI score."""
        if self.score >= 0.8:
            return "Strong signal -- swarm outperforms individuals"
        if self.score >= 0.6:
            return "Moderate signal -- better than random"
        if self.score >= 0.4:
            return "Weak signal -- marginally useful"
        return "Noise -- no better than coin flip"


def compute_ci(predictions: list, calibration: list) -> CIScore:
    """Compute Collective Intelligence score from scored predictions."""
    ci = CIScore()
    resolved = [p for p in predictions if p.outcome is not None and p.brier is not None]
    ci.n_resolved = len(resolved)
    if not resolved:
        return ci

    ci.n_correct = sum(
        1 for p in resolved
        if (p.confidence > 0.5 and p.outcome) or (p.confidence <= 0.5 and not p.outcome)
    )
    accuracy = ci.n_correct / len(resolved)

    mean_brier = statistics.mean(p.brier for p in resolved)
    brier_component = 1.0 - mean_brier

    cal_errors = []
    for bucket in calibration:
        if bucket["count"] > 0:
            cal_errors.append(abs(bucket["mean_confidence"] - bucket["actual_rate"]))
    ci.calibration_error = statistics.mean(cal_errors) if cal_errors else 0.5
    cal_component = 1.0 - min(1.0, ci.calibration_error * 2)

    kl_sum = 0.0
    for p in resolved:
        q = max(1e-9, min(1 - 1e-9, p.confidence))
        outcome_val = 1.0 if p.outcome else 0.0
        if outcome_val >= 0.5:
            kl_sum += math.log(q / 0.5) if q > 0.5 else 0.0
        else:
            kl_sum += math.log((1 - q) / 0.5) if q < 0.5 else 0.0
    ci.information_ratio = max(0.0, kl_sum / len(resolved))

    cats: dict = {}
    for p in resolved:
        cats[p.category] = cats.get(p.category, 0) + 1
    if len(cats) > 1:
        total = sum(cats.values())
        probs = [c / total for c in cats.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(len(cats))
        ci.category_spread = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        ci.category_spread = 0.0

    ci.score = (
        brier_component * 0.40
        + accuracy * 0.25
        + cal_component * 0.20
        + ci.category_spread * 0.15
    )
    ci.score = max(0.0, min(1.0, ci.score))
    return ci


@dataclass
class ProofReport:
    """Full execution proof report."""
    timestamp: str = ""
    duration_s: float = 0.0
    sols: int = 0
    n_predictions: int = 0
    n_seeds: int = 0
    n_resolved: int = 0
    accuracy: float = 0.0
    mean_brier: float = 0.0
    ci: CIScore = field(default_factory=CIScore)
    calibration: list = field(default_factory=list)
    leaderboard: list = field(default_factory=list)
    category_briers: dict = field(default_factory=dict)
    terrarium: dict = field(default_factory=dict)
    proof_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "duration_s": round(self.duration_s, 2),
            "sols": self.sols,
            "n_predictions": self.n_predictions,
            "n_seeds": self.n_seeds,
            "n_resolved": self.n_resolved,
            "accuracy": round(self.accuracy, 4),
            "mean_brier": round(self.mean_brier, 4),
            "ci": self.ci.to_dict(),
            "calibration": self.calibration,
            "leaderboard": self.leaderboard[:10],
            "category_briers": {k: round(v, 4) for k, v in self.category_briers.items()},
            "terrarium": self.terrarium,
            "proof_hash": self.proof_hash,
        }


def run_proof(
    sols: int = 365,
    n_predictions: int = 200,
    n_seeds: int = 3,
    market_seed: int = 42,
    **kwargs,
) -> ProofReport:
    """Run both engines, compute CI, produce proof."""
    t0 = time.monotonic()
    seed_list = list(range(42, 42 + n_seeds))

    ensemble_results = run_terrarium_ensemble(sols=sols, seeds=seed_list)
    predictions = generate_predictions(n_predictions, seed=market_seed)
    resolve_predictions(predictions, ensemble_results)
    score_predictions(predictions)

    resolved = [p for p in predictions if p.outcome is not None]
    briers = [p.brier for p in resolved if p.brier is not None]
    correct = sum(
        1 for p in resolved
        if (p.confidence > 0.5 and p.outcome) or (p.confidence <= 0.5 and not p.outcome)
    )
    accuracy = correct / len(resolved) if resolved else 0.0
    mean_brier = statistics.mean(briers) if briers else 0.0

    calibration = build_calibration_curve(predictions)
    leaderboard = build_leaderboard(predictions)

    cats: dict = {}
    for p in resolved:
        if p.brier is not None:
            cats.setdefault(p.category, []).append(p.brier)
    category_briers = {cat: statistics.mean(vals) for cat, vals in sorted(cats.items())}

    ci = compute_ci(predictions, calibration)

    canonical = ensemble_results[0]
    colonies_summary = []
    for c in canonical.get("colonies", []):
        name = c["name"]
        summary = None
        for s in canonical.get("summary", {}).get("colonies", []):
            if s["name"] == name:
                summary = s
                break
        colonies_summary.append({
            "name": name,
            "strategy": c.get("strategy", "unknown"),
            "start_pop": summary["start_pop"] if summary else 0,
            "end_pop": c["final_population"],
            "growth_pct": summary["growth_pct"] if summary else 0.0,
            "peak_pop": summary["peak_pop"] if summary else 0,
            "techs_unlocked": len(c.get("tech", {}).get("unlocked", [])),
            "alive": c["final_population"] > 0,
        })

    tf = canonical.get("summary", {}).get("terraforming", {})
    terrarium = {
        "colonies": colonies_summary,
        "total_population": sum(c["end_pop"] for c in colonies_summary),
        "all_alive": all(c["alive"] for c in colonies_summary),
        "terraforming_pct": round(tf.get("progress", 0) * 100, 4),
        "terraforming_phase": tf.get("phase", "none"),
        "total_migrations": canonical.get("summary", {}).get("total_migrations", 0),
    }

    duration = time.monotonic() - t0
    hash_input = json.dumps({
        "sols": sols, "n_predictions": n_predictions, "seeds": seed_list,
        "accuracy": round(accuracy, 4), "mean_brier": round(mean_brier, 4),
        "ci_score": round(ci.score, 4),
        "total_pop": terrarium["total_population"],
    }, sort_keys=True)
    proof_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    return ProofReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        duration_s=duration, sols=sols, n_predictions=n_predictions,
        n_seeds=n_seeds, n_resolved=len(resolved), accuracy=accuracy,
        mean_brier=mean_brier, ci=ci, calibration=calibration,
        leaderboard=leaderboard, category_briers=category_briers,
        terrarium=terrarium, proof_hash=proof_hash,
    )


def format_proof_text(report: ProofReport) -> str:
    """Format proof as human-readable text."""
    lines = []
    sep = "=" * 64
    dash = "  " + "-" * 56
    lines.append(sep)
    lines.append("  EXECUTION PROOF -- Mars Barn x Prediction Market")
    lines.append(sep)
    lines.append("  Timestamp: " + report.timestamp)
    lines.append("  Duration:  %.1fs" % report.duration_s)
    lines.append("  Config:    %d sols, %d predictions, %d seeds" % (
        report.sols, report.n_predictions, report.n_seeds))
    lines.append("  Hash:      " + report.proof_hash)
    lines.append("")

    lines.append("  TERRARIUM")
    lines.append(dash)
    for c in report.terrarium.get("colonies", []):
        status = "ALIVE" if c["alive"] else "DEAD"
        line = "  %-22s %-6s %3d -> %3d (%+6.1f%%)  techs=%d" % (
            c["name"], status, c["start_pop"], c["end_pop"],
            c["growth_pct"], c["techs_unlocked"])
        lines.append(line)
    tp = report.terrarium.get("total_population", 0)
    tf_pct = report.terrarium.get("terraforming_pct", 0)
    tf_phase = report.terrarium.get("terraforming_phase", "none")
    lines.append("  Total pop: %d  Terraforming: %.2f%% (%s)" % (tp, tf_pct, tf_phase))
    lines.append("")

    lines.append("  PREDICTION MARKET")
    lines.append(dash)
    lines.append("  Resolved: %d/%d" % (report.n_resolved, report.n_predictions))
    lines.append("  Accuracy: %.1f%%  Mean Brier: %.4f" % (report.accuracy * 100, report.mean_brier))
    lines.append("")

    ci = report.ci
    lines.append("  COLLECTIVE INTELLIGENCE")
    lines.append(dash)
    lines.append("  CI Score:      %.4f  (%s)" % (ci.score, ci.label()))
    lines.append("  Calibration:   %.4f error" % ci.calibration_error)
    lines.append("  Info ratio:    %.4f bits" % ci.information_ratio)
    lines.append("  Category span: %.4f" % ci.category_spread)
    lines.append("")

    lines.append("  CALIBRATION CURVE")
    lines.append(dash)
    for b in report.calibration:
        bar = "#" * int(b["actual_rate"] * 30) if b["count"] > 0 else ""
        lines.append("  [%.1f-%.1f] n=%3d  pred=%.2f  actual=%.2f  %s" % (
            b["bucket_lo"], b["bucket_hi"], b["count"],
            b["mean_confidence"], b["actual_rate"], bar))
    lines.append("")

    lines.append("  CATEGORY BRIER SCORES")
    lines.append(dash)
    for cat, brier in sorted(report.category_briers.items()):
        bar = "#" * int((1 - brier) * 30)
        lines.append("  %-22s %.3f  %s" % (cat, brier, bar))
    lines.append("")

    lines.append("  TOP 5 AGENTS")
    lines.append(dash)
    for row in report.leaderboard[:5]:
        lines.append("  %-22s %-12s brier=%.3f  karma=%+.1f" % (
            row["agent"], row["archetype"], row["mean_brier"], row["net_karma"]))
    lines.append(sep)
    return "\n".join(lines)


def format_proof_compact(report: ProofReport) -> str:
    """One-liner proof summary."""
    tp = report.terrarium.get("total_population", 0)
    alive = "3/3" if report.terrarium.get("all_alive") else "?"
    tf = report.terrarium.get("terraforming_pct", 0)
    return (
        "PROOF [%s] %dsol %dpred %dseed | Pop=%d (%s alive) TF=%.2f%% | "
        "Brier=%.4f Acc=%.1f%% CI=%.4f" % (
            report.proof_hash, report.sols, report.n_predictions, report.n_seeds,
            tp, alive, tf, report.mean_brier, report.accuracy * 100, report.ci.score))


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Execution proof: terrarium x market")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--predictions", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--market-seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--compact", action="store_true", help="One-liner output")
    args = parser.parse_args()

    report = run_proof(
        sols=args.sols,
        n_predictions=args.predictions,
        n_seeds=args.seeds,
        market_seed=args.market_seed,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif args.compact:
        print(format_proof_compact(report))
    else:
        print(format_proof_text(report))
