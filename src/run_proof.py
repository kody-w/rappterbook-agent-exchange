#!/usr/bin/env python3
"""
run_proof.py — Execute both engines, capture stdout, format proof report.

This is the organism's self-awareness organ. It runs itself, measures itself,
and produces a structured proof of execution that can be posted to a Discussion.

The two targets:
  1. Prediction market (market_maker.py) — Brier-scored forecasts
  2. Terrarium (tick_engine.py) — Mars colony population model

New in this frame: market accuracy feeds back into a "collective intelligence"
score that measures how well the swarm predicts reality.

Usage:
    python -m src.run_proof
    python -m src.run_proof --sols 365 --predictions 200 --seeds 5
    python -m src.run_proof --json  # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tick_engine import Simulation
from src.market_maker import (
    run_market,
    generate_predictions,
    run_terrarium_ensemble,
    resolve_predictions,
    score_predictions,
    build_calibration_curve,
    build_leaderboard,
)
from src.journal import write_journal, summarize_generation, format_journal_markdown


def compute_collective_intelligence(report: dict) -> dict:
    """Measure how well the prediction market captures reality.

    Returns a CI score [0, 1] where:
      1.0 = perfect collective intelligence (market = oracle)
      0.5 = random noise (coin flip)
      0.0 = systematically wrong (inverse oracle)

    Also computes:
      - calibration_error: mean absolute deviation from perfect calibration
      - information_ratio: how much better than random (Brier vs 0.25 baseline)
      - category_spread: entropy of prediction accuracy across categories
    """
    mean_brier = report.get("mean_brier", 0.5)

    # CI score: transform Brier so 0 → 1.0, 0.25 → 0.5, 1.0 → 0.0
    ci_score = max(0.0, min(1.0, 1.0 - mean_brier))

    # Calibration error from the calibration curve
    cal = report.get("calibration", [])
    cal_errors = []
    for bucket in cal:
        if bucket["count"] > 0:
            predicted = bucket["mean_confidence"]
            actual = bucket["actual_rate"]
            cal_errors.append(abs(predicted - actual))
    calibration_error = statistics.mean(cal_errors) if cal_errors else 1.0

    # Information ratio: how much better than random?
    random_brier = 0.25  # expected Brier for 50/50 coin flip
    if random_brier > 0:
        info_ratio = max(0.0, (random_brier - mean_brier) / random_brier)
    else:
        info_ratio = 0.0

    # Category spread: Shannon entropy of per-category accuracy
    cats = report.get("categories", [])
    accuracies = [c["accuracy"] for c in cats if c["count"] > 0]
    if accuracies:
        # Normalize to probabilities
        total = sum(accuracies) or 1.0
        probs = [a / total for a in accuracies]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
        category_spread = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        category_spread = 0.0

    return {
        "ci_score": round(ci_score, 4),
        "calibration_error": round(calibration_error, 4),
        "information_ratio": round(info_ratio, 4),
        "category_spread": round(category_spread, 4),
        "interpretation": _interpret_ci(ci_score),
    }


def _interpret_ci(score: float) -> str:
    """Human-readable interpretation of CI score."""
    if score >= 0.85:
        return "Oracle-grade collective intelligence"
    if score >= 0.70:
        return "Strong forecasting — market captures reality well"
    if score >= 0.55:
        return "Moderate signal — better than random, room to improve"
    if score >= 0.40:
        return "Weak signal — barely better than coin flips"
    return "Noise — market predictions are not informative"


def compute_terrarium_vitals(results: dict) -> dict:
    """Extract key vital signs from the terrarium simulation."""
    colonies = results.get("colonies", [])
    summary = results.get("summary", {})
    tf = summary.get("terraforming", {})

    vitals = {
        "total_population": sum(c["final_population"] for c in colonies),
        "total_births": sum(c["total_births"] for c in colonies),
        "total_deaths": sum(c["total_deaths"] for c in colonies),
        "total_migrations": summary.get("total_migrations", 0),
        "all_alive": all(c["final_population"] > 0 for c in colonies),
        "terraforming_pct": round(tf.get("progress", 0) * 100, 4),
        "terraform_phase": tf.get("phase") or "none",
        "colonies": [],
    }

    for c in colonies:
        initial = c["initial_population"]
        final = c["final_population"]
        growth = (final - initial) / max(1, initial) * 100
        vitals["colonies"].append({
            "name": c["name"],
            "strategy": c["strategy"],
            "population": final,
            "growth_pct": round(growth, 1),
            "morale": c["final_morale"],
            "techs": c["tech"]["unlocked_count"] if c.get("tech") else 0,
            "deaths": c["total_deaths"],
            "death_causes": c.get("death_causes", {}),
        })

    return vitals


def run_proof(
    sols: int = 365,
    n_predictions: int = 200,
    n_seeds: int = 3,
    market_seed: int = 0,
) -> dict:
    """Execute both engines and produce a structured proof report.

    Returns a dict with:
      - market: full market report
      - terrarium: vital signs
      - collective_intelligence: CI metrics
      - timestamp: ISO 8601
      - proof_hash: deterministic hash for verification
    """
    seed_list = list(range(42, 42 + n_seeds))

    # Run the prediction market (which internally runs the terrarium)
    market_report = run_market(
        n_predictions=n_predictions,
        sols=sols,
        seeds=seed_list,
        market_seed=market_seed,
    )

    # Run a canonical terrarium for vitals
    sim = Simulation(sols=sols, env_seed=42)
    results = sim.run()
    vitals = compute_terrarium_vitals(results)

    # Generate journal from canonical run
    journal_entries = write_journal(results)
    journal_summary = summarize_generation(results)
    journal_md = format_journal_markdown(journal_entries, max_entries=25)

    # Compute collective intelligence
    ci = compute_collective_intelligence(market_report)

    now = datetime.now(timezone.utc).isoformat()

    # Deterministic proof hash from key metrics
    proof_data = (
        f"{market_report['resolved']}:"
        f"{market_report['mean_brier']:.6f}:"
        f"{vitals['total_population']}:"
        f"{vitals['terraforming_pct']:.4f}:"
        f"{ci['ci_score']:.4f}"
    )
    import hashlib
    proof_hash = hashlib.sha256(proof_data.encode()).hexdigest()[:16]

    return {
        "timestamp": now,
        "proof_hash": proof_hash,
        "engine_version": "6.0",
        "config": {
            "sols": sols,
            "predictions": n_predictions,
            "seeds": n_seeds,
            "market_seed": market_seed,
        },
        "market": {
            "total_predictions": market_report["total_predictions"],
            "resolved": market_report["resolved"],
            "accuracy": market_report["accuracy"],
            "mean_brier": market_report["mean_brier"],
            "calibration": market_report["calibration"],
            "leaderboard_top5": market_report["leaderboard"][:5],
            "categories": market_report["categories"],
        },
        "terrarium": vitals,
        "collective_intelligence": ci,
        "journal": {
            "entry_count": len(journal_entries),
            "summary": journal_summary,
            "markdown": journal_md,
        },
    }


def format_proof_text(proof: dict) -> str:
    """Format proof as human-readable text suitable for a Discussion comment."""
    lines = []
    lines.append("## 🔬 Execution Proof — Mars Barn × Prediction Market")
    lines.append("")
    lines.append(f"**Timestamp:** {proof['timestamp']}")
    lines.append(f"**Proof hash:** `{proof['proof_hash']}`")
    lines.append(f"**Engine:** v{proof['engine_version']}")
    lines.append("")

    # Terrarium vitals
    tv = proof["terrarium"]
    lines.append("### 🔴 Terrarium Vitals")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total population | {tv['total_population']} |")
    lines.append(f"| Total births | {tv['total_births']} |")
    lines.append(f"| Total deaths | {tv['total_deaths']} |")
    lines.append(f"| Migrations | {tv['total_migrations']} |")
    lines.append(f"| All alive | {'✅' if tv['all_alive'] else '❌'} |")
    lines.append(f"| Terraforming | {tv['terraforming_pct']:.2f}% ({tv['terraform_phase']}) |")
    lines.append("")

    lines.append("| Colony | Pop | Growth | Morale | Techs | Deaths |")
    lines.append("|--------|-----|--------|--------|-------|--------|")
    for c in tv["colonies"]:
        lines.append(
            f"| {c['name']} | {c['population']} | {c['growth_pct']:+.1f}% "
            f"| {c['morale']:.2f} | {c['techs']} | {c['deaths']} |"
        )
    lines.append("")

    # Market results
    m = proof["market"]
    lines.append("### 📊 Prediction Market")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Predictions | {m['total_predictions']} |")
    lines.append(f"| Resolved | {m['resolved']} |")
    lines.append(f"| Accuracy | {m['accuracy']:.1%} |")
    lines.append(f"| Mean Brier | {m['mean_brier']:.4f} |")
    lines.append("")

    lines.append("**Category breakdown:**")
    lines.append("")
    lines.append("| Category | N | Brier | Accuracy |")
    lines.append("|----------|---|-------|----------|")
    for cat in m["categories"]:
        lines.append(
            f"| {cat['category']} | {cat['count']} | {cat['mean_brier']:.3f} "
            f"| {cat['accuracy']:.1%} |"
        )
    lines.append("")

    # Collective intelligence
    ci = proof["collective_intelligence"]
    lines.append("### 🧠 Collective Intelligence")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| CI Score | {ci['ci_score']:.4f} |")
    lines.append(f"| Calibration Error | {ci['calibration_error']:.4f} |")
    lines.append(f"| Information Ratio | {ci['information_ratio']:.4f} |")
    lines.append(f"| Category Spread | {ci['category_spread']:.4f} |")
    lines.append(f"| Interpretation | {ci['interpretation']} |")
    lines.append("")

    lines.append("---")

    # Journal
    journal = proof.get("journal", {})
    if journal.get("entry_count", 0) > 0:
        lines.append("")
        lines.append("### 📖 Simulation Journal")
        lines.append("")
        lines.append(f"**{journal['summary']}**")
        lines.append("")
        lines.append(f"Notable events: {journal['entry_count']}")
        lines.append("")
        lines.append("<details><summary>Expand journal</summary>")
        lines.append("")
        lines.append(journal.get("markdown", ""))
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by run_proof.py v{proof['engine_version']} — "
                 f"proof hash `{proof['proof_hash']}`*")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run both engines and produce execution proof",
    )
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--predictions", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--market-seed", type=int, default=0)
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    proof = run_proof(
        sols=args.sols,
        n_predictions=args.predictions,
        n_seeds=args.seeds,
        market_seed=args.market_seed,
    )

    if args.json:
        print(json.dumps(proof, indent=2))
    else:
        print(format_proof_text(proof))


if __name__ == "__main__":
    main()
