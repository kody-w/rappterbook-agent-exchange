"""
cross_sim.py -- Cross-simulation bridge connecting the Mars Barn terrarium
to the prediction market with added colony-level analysis.

Usage:
    from src.cross_sim import CrossSimulation
    xsim = CrossSimulation(n_predictions=40, sols=365)
    report = xsim.run()

CLI:
    python -m src.cross_sim --sols 365 --seed 42 --predictions 40
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.market_maker import (
    AGENT_ARCHETYPES,
    COLONY_NAMES,
    Prediction,
    RESOLVERS,
    TEMPLATES,
    brier_score,
    build_calibration_curve,
    build_leaderboard,
    generate_predictions,
    resolve_predictions,
    run_terrarium,
    score_predictions,
)

# ---------------------------------------------------------------------------
# Surprise & information-gain metrics
# ---------------------------------------------------------------------------


def compute_surprise(prediction_dict: dict) -> float:
    """Surprise = 1 - confidence when outcome is True, confidence when False.

    High surprise means the outcome contradicted the agent's belief.
    Range [0, 1].
    """
    conf = prediction_dict.get("confidence", 0.5)
    outcome = prediction_dict.get("outcome")
    if outcome is None:
        return 0.0
    if outcome:
        return 1.0 - conf
    return conf


def compute_information_gain(pred_dicts: list[dict]) -> float:
    """KL divergence (bits) between predicted distribution and outcomes.

    Measures how much the market learned from resolution.
    Returns >= 0.  Higher means predictions were more surprised.
    """
    resolved = [p for p in pred_dicts if p.get("outcome") is not None]
    if not resolved:
        return 0.0

    eps = 1e-12
    kl = 0.0
    for p in resolved:
        conf = max(eps, min(1.0 - eps, p["confidence"]))
        outcome = 1.0 if p["outcome"] else 0.0
        # q = predicted dist, p_true = empirical (0 or 1)
        if outcome > 0.5:
            kl += math.log2(1.0 / max(eps, conf))
        else:
            kl += math.log2(1.0 / max(eps, 1.0 - conf))
    return kl / len(resolved)


# ---------------------------------------------------------------------------
# Colony-level aggregation
# ---------------------------------------------------------------------------


def colony_prediction_map(pred_dicts: list[dict]) -> dict[str, list[dict]]:
    """Group predictions by colony name.  Predictions without a colony key
    go into the ``"global"`` bucket."""
    mapping: dict[str, list[dict]] = {"global": []}
    for name in COLONY_NAMES:
        mapping[name] = []

    for p in pred_dicts:
        colony = (p.get("params") or {}).get("colony")
        if colony and colony in mapping:
            mapping[colony].append(p)
        else:
            mapping["global"].append(p)
    return mapping


def colony_accuracy(pred_dicts: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-colony stats: n_predictions, mean_brier, mean_surprise, true_rate."""
    groups = colony_prediction_map(pred_dicts)
    stats: dict[str, dict[str, Any]] = {}
    for colony, preds in groups.items():
        resolved = [p for p in preds if p.get("outcome") is not None]
        n = len(resolved)
        if n == 0:
            stats[colony] = {
                "n_predictions": 0,
                "mean_brier": 0.0,
                "mean_surprise": 0.0,
                "true_rate": 0.0,
            }
            continue
        mean_brier = sum(p.get("brier", 0.0) or 0.0 for p in resolved) / n
        mean_surprise = sum(compute_surprise(p) for p in resolved) / n
        true_rate = sum(1 for p in resolved if p["outcome"]) / n
        stats[colony] = {
            "n_predictions": n,
            "mean_brier": round(mean_brier, 4),
            "mean_surprise": round(mean_surprise, 4),
            "true_rate": round(true_rate, 4),
        }
    return stats


# ---------------------------------------------------------------------------
# CrossSimulation orchestrator
# ---------------------------------------------------------------------------


def _prediction_to_dict(p: Prediction) -> dict:
    """Convert a Prediction dataclass to a plain dict."""
    return {
        "id": p.id,
        "agent": p.agent,
        "archetype": p.archetype,
        "category": p.category,
        "description": p.description,
        "params": p.params,
        "confidence": p.confidence,
        "stake": p.stake,
        "outcome": p.outcome,
        "brier": p.brier,
        "log": p.log,
        "payout": p.payout,
    }


class CrossSimulation:
    """End-to-end bridge: terrarium → predictions → resolution → cross-analysis."""

    def __init__(
        self,
        n_predictions: int = 40,
        sols: int = 365,
        env_seed: int = 42,
        market_seed: int | None = None,
    ) -> None:
        self.n_predictions = n_predictions
        self.sols = sols
        self.env_seed = env_seed
        self.market_seed = market_seed if market_seed is not None else env_seed

    # ---- public API -------------------------------------------------------

    def run(self, quiet: bool = False) -> dict:
        """Execute the full cross-simulation pipeline and return a report."""
        # Stage 1: run terrarium
        if not quiet:
            print(f"[cross_sim] Running terrarium (sols={self.sols}, seed={self.env_seed})")
        sim_results = run_terrarium(sols=self.sols, seed=self.env_seed)

        # Stage 2: generate predictions
        predictions = generate_predictions(n=self.n_predictions, seed=self.market_seed)
        if not quiet:
            print(f"[cross_sim] Generated {len(predictions)} predictions")

        # Stage 3: resolve against sim
        resolve_predictions(predictions, [sim_results])

        # Stage 4: score
        score_predictions(predictions)
        if not quiet:
            resolved_count = sum(1 for p in predictions if p.outcome is not None)
            print(f"[cross_sim] Resolved {resolved_count}/{len(predictions)}")

        # Stage 5: cross-analyse
        pred_dicts = [_prediction_to_dict(p) for p in predictions]
        cal = build_calibration_curve(predictions)
        lb = build_leaderboard(predictions)

        col_acc = colony_accuracy(pred_dicts)
        info_gain = compute_information_gain(pred_dicts)

        # Enrich predictions with surprise
        for pd in pred_dicts:
            pd["surprise"] = round(compute_surprise(pd), 4)

        # Colony result summaries
        colony_results = []
        for c in sim_results.get("colonies", []):
            colony_results.append({
                "name": c["name"],
                "strategy": c.get("strategy", "unknown"),
                "initial_population": c.get("initial_population", 0),
                "final_population": c.get("final_population", 0),
                "total_births": c.get("total_births", 0),
                "total_deaths": c.get("total_deaths", 0),
                "final_morale": c.get("final_morale", 0),
                "techs_unlocked": len(
                    (c.get("tech") or {}).get("unlocked", [])
                ),
            })

        resolved_preds = [p for p in pred_dicts if p.get("outcome") is not None]
        mean_brier = (
            sum(p.get("brier", 0.0) or 0.0 for p in resolved_preds) / max(1, len(resolved_preds))
        )

        report: dict[str, Any] = {
            "_meta": {
                "engine": "cross_sim",
                "sols": self.sols,
                "env_seed": self.env_seed,
                "market_seed": self.market_seed,
                "n_predictions": self.n_predictions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "summary": {
                "total_predictions": len(pred_dicts),
                "resolved": len(resolved_preds),
                "mean_brier": round(mean_brier, 4),
                "information_gain_bits": round(info_gain, 4),
            },
            "colony_results": colony_results,
            "colony_accuracy": col_acc,
            "market_outcomes": pred_dicts,
            "leaderboard": lb,
            "calibration": cal,
        }

        if not quiet:
            print(f"[cross_sim] Done — mean Brier {mean_brier:.4f}, "
                  f"info gain {info_gain:.3f} bits")

        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-simulation bridge: terrarium ↔ prediction market",
    )
    parser.add_argument("--sols", type=int, default=365,
                        help="Simulation length in sols (default: 365)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Environment seed (default: 42)")
    parser.add_argument("--predictions", type=int, default=40,
                        help="Number of predictions (default: 40)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON report to file")
    args = parser.parse_args()

    xsim = CrossSimulation(
        n_predictions=args.predictions,
        sols=args.sols,
        env_seed=args.seed,
    )
    report = xsim.run(quiet=args.quiet)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(report, indent=2))
        print(f"Report written to {out}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
