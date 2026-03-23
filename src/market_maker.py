"""
market_maker.py — Prediction market engine backed by Mars Barn terrarium.

Generates 100+ predictions about colony outcomes, runs the terrarium to
resolve them, scores with Brier scores, builds calibration curves, and
tracks a leaderboard. Five-stage pipe per Discussion #5892:

    GENERATE → RESOLVE → SCORE → STAKE → OUTPUT

Usage:
    python3 src/market_maker.py
    python3 src/market_maker.py --sols 365 --seed 99
    python3 src/market_maker.py --quiet
"""
from __future__ import annotations

import json
import math
import random
import hashlib
import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

CATEGORIES = [
    "survival", "growth", "population_threshold",
    "epidemic", "tech_unlock", "migration", "strategy_winner",
]

PREDICTOR_AGENTS = [
    {"id": "oracle-prime", "name": "Oracle Prime", "style": "calibrated", "bias": 0.0},
    {"id": "bull-run", "name": "Bull Run", "style": "optimistic", "bias": 0.15},
    {"id": "doom-prophet", "name": "Doom Prophet", "style": "pessimistic", "bias": -0.20},
    {"id": "random-walk", "name": "Random Walk", "style": "noisy", "bias": 0.0},
    {"id": "trend-chaser", "name": "Trend Chaser", "style": "momentum", "bias": 0.05},
    {"id": "contrarian-x", "name": "Contrarian X", "style": "contrarian", "bias": -0.10},
    {"id": "data-miner", "name": "Data Miner", "style": "calibrated", "bias": 0.02},
    {"id": "gut-feel", "name": "Gut Feel", "style": "noisy", "bias": 0.08},
]

KARMA_STAKE_MAX = 50


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    """A single prediction about a Mars colony outcome."""
    id: str
    agent_id: str
    category: str
    description: str
    confidence: float
    stake: int
    colony: str
    threshold: float
    resolved: bool = False
    outcome: bool = False
    brier_score: float | None = None
    log_score: float | None = None
    payout: int = 0


@dataclass
class AgentRecord:
    """Aggregated stats for a predictor agent."""
    agent_id: str
    name: str
    style: str
    predictions: int = 0
    resolved: int = 0
    correct: int = 0
    total_staked: int = 0
    total_payout: int = 0
    mean_brier: float = 0.0


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def deterministic_seed(text: str) -> int:
    """SHA256-based deterministic seed."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def brier_score(confidence: float, outcome: bool) -> float:
    """Brier score: (forecast - outcome)^2. Lower is better. Range [0, 1]."""
    o = 1.0 if outcome else 0.0
    return (confidence - o) ** 2


def log_score(confidence: float, outcome: bool) -> float:
    """Logarithmic score. More sensitive to extreme miscalibration. Always <= 0."""
    p = confidence if outcome else (1.0 - confidence)
    p = max(p, 1e-10)
    return math.log(p)


def payout_from_brier(stake: int, bs: float) -> int:
    """Karma payout based on Brier score quality."""
    if bs < 0.10:
        return stake * 2
    elif bs < 0.25:
        return stake
    elif bs < 0.50:
        return 0
    else:
        return -stake


# ---------------------------------------------------------------------------
# Stage 1: GENERATE — create predictions
# ---------------------------------------------------------------------------
def _agent_confidence(base_prob: float, agent: dict, rng: random.Random) -> float:
    """Adjust base probability by agent style and bias."""
    bias = agent["bias"]
    style = agent["style"]
    noise = 0.0
    if style == "noisy":
        noise = rng.gauss(0, 0.15)
    elif style == "momentum":
        noise = rng.gauss(0.05, 0.08)
    elif style == "contrarian":
        base_prob = 1.0 - base_prob
    elif style == "calibrated":
        noise = rng.gauss(0, 0.05)
    conf = base_prob + bias + noise
    return max(0.05, min(0.95, conf))


def generate_predictions(
    colony_names: list[str],
    rng: random.Random,
) -> list[Prediction]:
    """Generate a diverse set of predictions about colony outcomes."""
    predictions: list[Prediction] = []
    pred_id = 0

    templates = [
        ("survival", "{colony} survives 365 sols", "each", 0.92, 1),
        ("survival", "{colony} population stays above 20", "each", 0.88, 20),
        ("growth", "{colony} grows by >50%", "each", 0.60, 50.0),
        ("growth", "{colony} grows by >100%", "each", 0.35, 100.0),
        ("growth", "{colony} grows by >200%", "each", 0.10, 200.0),
        ("population_threshold", "{colony} reaches 200 colonists", "each", 0.40, 200),
        ("population_threshold", "{colony} reaches 300 colonists", "each", 0.15, 300),
        ("epidemic", "{colony} suffers at least one epidemic", "each", 0.55, 1),
        ("epidemic", "{colony} suffers 3+ epidemics", "each", 0.10, 3),
        ("tech_unlock", "{colony} unlocks 4+ techs", "each", 0.50, 4),
        ("tech_unlock", "{colony} unlocks all 8 techs", "each", 0.05, 8),
        ("migration", "At least 10 total migrations occur", "all", 0.45, 10),
        ("migration", "At least 50 total migrations occur", "all", 0.15, 50),
        ("strategy_winner", "Aggressive strategy has highest growth", "all", 0.55, 0),
        ("strategy_winner", "Conservative has highest final population", "all", 0.65, 0),
    ]

    for tmpl in templates:
        cat, desc_fmt, scope, base_prob, threshold = tmpl
        colonies_to_use = colony_names if scope == "each" else ["all"]
        for colony in colonies_to_use:
            desc = desc_fmt.format(colony=colony)
            for agent in PREDICTOR_AGENTS:
                pred_id += 1
                seed = deterministic_seed(f"{agent['id']}:{desc}:{pred_id}")
                agent_rng = random.Random(seed)
                conf = _agent_confidence(base_prob, agent, agent_rng)
                stake = min(KARMA_STAKE_MAX, max(5, int(conf * 30 + agent_rng.randint(0, 15))))
                predictions.append(Prediction(
                    id=f"pred-{pred_id:04d}",
                    agent_id=agent["id"],
                    category=cat,
                    description=desc,
                    confidence=round(conf, 3),
                    stake=stake,
                    colony=colony,
                    threshold=threshold,
                ))
    return predictions


# ---------------------------------------------------------------------------
# Stage 2: RESOLVE — check against simulation results
# ---------------------------------------------------------------------------
def resolve_predictions(
    predictions: list[Prediction],
    sim_results: dict,
) -> list[Prediction]:
    """Resolve each prediction against simulation results."""
    summary = sim_results["summary"]
    colony_summaries = {s["name"]: s for s in summary["colonies"]}
    total_migrations = summary.get("total_migrations", 0)
    for pred in predictions:
        outcome = _check_outcome(pred, colony_summaries, total_migrations, sim_results)
        pred.resolved = True
        pred.outcome = outcome
    return predictions


def _check_outcome(
    pred: Prediction,
    colony_summaries: dict,
    total_migrations: int,
    sim_results: dict,
) -> bool:
    """Determine if a prediction came true."""
    cat = pred.category
    colony = pred.colony

    if cat == "survival":
        if colony == "all":
            return all(s["end_pop"] > 0 for s in colony_summaries.values())
        cs = colony_summaries.get(colony, {})
        if pred.threshold > 1:
            return cs.get("end_pop", 0) >= pred.threshold
        return cs.get("end_pop", 0) > 0

    elif cat == "growth":
        if colony == "all":
            return all(s.get("growth_pct", 0) > pred.threshold
                       for s in colony_summaries.values())
        cs = colony_summaries.get(colony, {})
        return cs.get("growth_pct", 0) > pred.threshold

    elif cat == "population_threshold":
        if colony == "all":
            return all(s.get("peak_pop", 0) >= pred.threshold
                       for s in colony_summaries.values())
        cs = colony_summaries.get(colony, {})
        return cs.get("peak_pop", 0) >= pred.threshold

    elif cat == "epidemic":
        count = 0
        for c in sim_results["colonies"]:
            if colony in ("all", c["name"]):
                count += sum(1 for e in c.get("events", [])
                             if e.get("type") == "epidemic_start")
        return count >= pred.threshold

    elif cat == "tech_unlock":
        for c in sim_results["colonies"]:
            if colony in ("all", c["name"]):
                tech = c.get("tech", {})
                if tech and tech.get("unlocked_count", 0) >= pred.threshold:
                    return True
        return False

    elif cat == "migration":
        return total_migrations >= pred.threshold

    elif cat == "strategy_winner":
        if "highest growth" in pred.description.lower():
            winner = max(colony_summaries.values(),
                         key=lambda s: s.get("growth_pct", 0))
            return winner.get("strategy") == "aggressive"
        elif "highest final population" in pred.description.lower():
            winner = max(colony_summaries.values(),
                         key=lambda s: s.get("end_pop", 0))
            return winner.get("strategy") == "conservative"
        return False

    return False


# ---------------------------------------------------------------------------
# Stage 3: SCORE — Brier + log scores
# ---------------------------------------------------------------------------
def score_predictions(predictions: list[Prediction]) -> list[Prediction]:
    """Compute Brier and log scores for all resolved predictions."""
    for pred in predictions:
        if not pred.resolved:
            continue
        pred.brier_score = round(brier_score(pred.confidence, pred.outcome), 4)
        pred.log_score = round(log_score(pred.confidence, pred.outcome), 4)
    return predictions


# ---------------------------------------------------------------------------
# Stage 4: STAKE — karma payouts
# ---------------------------------------------------------------------------
def compute_payouts(predictions: list[Prediction]) -> list[Prediction]:
    """Compute karma payouts for resolved predictions."""
    for pred in predictions:
        if not pred.resolved or pred.brier_score is None:
            continue
        pred.payout = payout_from_brier(pred.stake, pred.brier_score)
    return predictions


def build_calibration_curve(predictions: list[Prediction]) -> list[dict]:
    """Group predictions by confidence bucket, compute actual rate."""
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    curve = []
    for lo, hi in buckets:
        in_bucket = [p for p in predictions
                     if p.resolved and lo <= p.confidence < hi]
        if not in_bucket:
            curve.append({
                "bucket": f"{int(lo*100)}-{int(hi*100)}%",
                "count": 0, "stated_avg": round((lo + hi) / 2, 2),
                "actual_rate": 0.0,
            })
            continue
        stated_avg = sum(p.confidence for p in in_bucket) / len(in_bucket)
        actual_rate = sum(1 for p in in_bucket if p.outcome) / len(in_bucket)
        curve.append({
            "bucket": f"{int(lo*100)}-{int(hi*100)}%",
            "count": len(in_bucket),
            "stated_avg": round(stated_avg, 3),
            "actual_rate": round(actual_rate, 3),
        })
    return curve


def build_leaderboard(predictions: list[Prediction]) -> list[dict]:
    """Aggregate agent performance into a leaderboard."""
    agents: dict[str, AgentRecord] = {}
    for agent in PREDICTOR_AGENTS:
        agents[agent["id"]] = AgentRecord(
            agent_id=agent["id"], name=agent["name"], style=agent["style"],
        )
    for pred in predictions:
        if pred.agent_id not in agents:
            continue
        rec = agents[pred.agent_id]
        rec.predictions += 1
        rec.total_staked += pred.stake
        if pred.resolved:
            rec.resolved += 1
            if pred.outcome == (pred.confidence >= 0.5):
                rec.correct += 1
            rec.total_payout += pred.payout
    for rec in agents.values():
        resolved = [p for p in predictions
                    if p.agent_id == rec.agent_id and p.resolved
                    and p.brier_score is not None]
        if resolved:
            rec.mean_brier = round(
                sum(p.brier_score for p in resolved) / len(resolved), 4)
    board = sorted(agents.values(), key=lambda r: r.mean_brier)
    return [
        {
            "rank": i + 1, "agent_id": r.agent_id, "name": r.name,
            "style": r.style, "predictions": r.predictions,
            "resolved": r.resolved, "correct": r.correct,
            "accuracy": round(r.correct / max(r.resolved, 1) * 100, 1),
            "mean_brier": r.mean_brier, "total_staked": r.total_staked,
            "net_karma": r.total_payout,
        }
        for i, r in enumerate(board)
    ]


# ---------------------------------------------------------------------------
# Stage 5: OUTPUT — build report
# ---------------------------------------------------------------------------
def build_output(
    predictions: list[Prediction],
    leaderboard: list[dict],
    calibration: list[dict],
    sim_summary: dict,
) -> dict:
    """Build the final market.json structure."""
    resolved = [p for p in predictions if p.resolved]
    correct = [p for p in resolved if p.outcome == (p.confidence >= 0.5)]
    total_staked = sum(p.stake for p in predictions)
    total_payout = sum(p.payout for p in resolved)
    bs_list = [p.brier_score for p in resolved if p.brier_score is not None]
    mean_b = sum(bs_list) / max(len(bs_list), 1)

    by_category: dict[str, dict] = {}
    for pred in predictions:
        cat = pred.category
        if cat not in by_category:
            by_category[cat] = {"total": 0, "resolved": 0, "correct": 0}
        by_category[cat]["total"] += 1
        if pred.resolved:
            by_category[cat]["resolved"] += 1
            if pred.outcome == (pred.confidence >= 0.5):
                by_category[cat]["correct"] += 1

    return {
        "_meta": {
            "engine": "market-maker", "version": "2.0.0",
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_predictions": len(predictions),
            "total_resolved": len(resolved),
            "total_correct": len(correct),
            "mean_brier_score": round(mean_b, 4),
        },
        "market_stats": {
            "total_staked": total_staked,
            "total_payout": total_payout,
            "net_house_edge": total_staked - total_payout,
            "avg_confidence": round(
                sum(p.confidence for p in predictions) / max(len(predictions), 1), 3),
            "categories": by_category,
        },
        "predictions": [
            {"id": p.id, "agent_id": p.agent_id, "category": p.category,
             "description": p.description, "confidence": p.confidence,
             "stake": p.stake, "colony": p.colony, "resolved": p.resolved,
             "outcome": p.outcome, "brier_score": p.brier_score,
             "log_score": p.log_score, "payout": p.payout}
            for p in predictions
        ],
        "leaderboard": leaderboard,
        "calibration_curve": calibration,
        "sim_summary": sim_summary,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(sols: int = 365, env_seed: int = 42, quiet: bool = False) -> dict:
    """Run the full prediction market pipeline."""
    from src.tick_engine import Simulation

    rng = random.Random(env_seed)
    colony_specs = [
        ("Ares Prime", "conservative", 1001),
        ("Olympus Station", "balanced", 2002),
        ("Red Frontier", "aggressive", 3003),
    ]
    colony_names = [c[0] for c in colony_specs]

    if not quiet:
        print(f"[1/5] Generating predictions for {len(colony_names)} colonies...")
    predictions = generate_predictions(colony_names, rng)
    if not quiet:
        print(f"  Generated {len(predictions)} predictions across {len(CATEGORIES)} categories")

    if not quiet:
        print(f"\n[2/5] Running Mars Barn simulation ({sols} sols, seed={env_seed})...")
    sim = Simulation(sols=sols, env_seed=env_seed, colonies=colony_specs)
    sim_results = sim.run()
    predictions = resolve_predictions(predictions, sim_results)
    outcomes_true = sum(1 for p in predictions if p.outcome)
    if not quiet:
        print(f"  Resolved: {outcomes_true} YES / {len(predictions) - outcomes_true} NO")

    if not quiet:
        print("\n[3/5] Computing Brier scores...")
    predictions = score_predictions(predictions)
    brier_vals = [p.brier_score for p in predictions if p.brier_score is not None]
    mean_b = sum(brier_vals) / max(len(brier_vals), 1)
    if not quiet:
        print(f"  Mean Brier score: {mean_b:.4f}")

    if not quiet:
        print("\n[4/5] Computing karma payouts...")
    predictions = compute_payouts(predictions)
    total_staked = sum(p.stake for p in predictions)
    total_payout = sum(p.payout for p in predictions)
    if not quiet:
        print(f"  Total staked: {total_staked} karma")
        print(f"  Total payout: {total_payout:+d} karma")
        print(f"  House edge:   {total_staked - total_payout} karma")

    calibration = build_calibration_curve(predictions)
    leaderboard = build_leaderboard(predictions)
    output = build_output(predictions, leaderboard, calibration, sim_results["summary"])

    if not quiet:
        print("\n[5/5] Market complete.")
        _print_summary(output)

    return output


def _print_summary(output: dict) -> None:
    """Print human-readable summary to stdout."""
    meta = output["_meta"]
    stats = output["market_stats"]

    print()
    print("=" * 60)
    print("  PREDICTION MARKET RESULTS")
    print("=" * 60)
    print(f"  Predictions:  {meta['total_predictions']}")
    print(f"  Resolved:     {meta['total_resolved']}")
    print(f"  Correct:      {meta['total_correct']} "
          f"({meta['total_correct']/max(meta['total_resolved'],1)*100:.1f}%)")
    print(f"  Mean Brier:   {meta['mean_brier_score']:.4f}")
    print(f"  Total staked: {stats['total_staked']} karma")
    print(f"  Net payout:   {stats['total_payout']:+d} karma")

    print("\n  Category breakdown:")
    for cat, data in stats["categories"].items():
        correct = data.get("correct", 0)
        total = data.get("total", 0)
        pct = correct / max(total, 1) * 100
        print(f"    {cat:25s}  {correct:3d}/{total:3d}  ({pct:.0f}%)")

    print("\n  Calibration curve:")
    for bucket in output["calibration_curve"]:
        bar_len = int(bucket["actual_rate"] * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"    {bucket['bucket']:>10s}  stated={bucket['stated_avg']:.2f}  "
              f"actual={bucket['actual_rate']:.3f}  {bar}  n={bucket['count']}")

    print("\n  Leaderboard:")
    print(f"    {'Rank':>4s}  {'Agent':20s}  {'Style':12s}  "
          f"{'Brier':>6s}  {'Correct':>7s}  {'Karma':>7s}")
    for entry in output["leaderboard"]:
        print(f"    {entry['rank']:>4d}  {entry['name']:20s}  "
              f"{entry['style']:12s}  {entry['mean_brier']:6.4f}  "
              f"{entry['correct']:>4d}/{entry['resolved']:<3d}  "
              f"{entry['net_karma']:>+7d}")

    sim = output.get("sim_summary", {})
    colonies = sim.get("colonies", [])
    if colonies:
        print("\n  Colony outcomes (resolved by simulation):")
        for c in colonies:
            print(f"    {c['name']:20s}  {c['start_pop']}->{c['end_pop']}  "
                  f"({c['growth_pct']:+.1f}%)  "
                  f"births={c['total_births']}  deaths={c['total_deaths']}  "
                  f"techs={c.get('techs_unlocked', 0)}")

    print()
    print("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Mars colony prediction market")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = run_pipeline(sols=args.sols, env_seed=args.seed, quiet=args.quiet)

    out_path = Path(args.output) if args.output else REPO_ROOT / "state" / "market.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(output, indent=2))
    tmp.rename(out_path)
    if not args.quiet:
        print(f"  Market state saved: {out_path}")


if __name__ == "__main__":
    main()
