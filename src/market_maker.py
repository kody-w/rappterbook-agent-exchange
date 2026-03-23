"""
market_maker.py -- Prediction market engine for the Mars Barn terrarium.

Agents place predictions on terrarium outcomes (colony survival, population
peaks, tech unlocks, epidemics, dust storms, etc.). Predictions resolve
against actual simulation results via majority-vote ensemble.

Usage:
    from src.market_maker import run_market, run_terrarium
    result = run_terrarium(sols=365, seed=42)
    report = run_market(n_predictions=100, sols=365, seeds=[42, 43, 44])

CLI:
    python -m src.market_maker --predictions 200 --sols 365 --seeds 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.tick_engine import Simulation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLONY_NAMES: list[str] = ["Ares Prime", "Olympus Station", "Red Frontier"]

TECH_NAMES: list[str] = [
    "Advanced Solar Cells",
    "Compact Fusion Reactor",
    "Martian Crop Genetics",
    "Aquaponics Integration",
    "Regolith Rad Shielding",
    "AI Diagnostics",
    "Zero-Loss Water Recycling",
    "Autonomous Construction Bots",
]

AGENT_ARCHETYPES: dict[str, dict[str, float]] = {
    "oracle":     {"bias": 0.0,  "noise": 0.05},
    "optimist":   {"bias": 0.10, "noise": 0.08},
    "pessimist":  {"bias": -0.10, "noise": 0.08},
    "degen":      {"bias": 0.0,  "noise": 0.25},
    "analyst":    {"bias": 0.0,  "noise": 0.03},
    "contrarian": {"bias": -0.15, "noise": 0.20},
}

# ---------------------------------------------------------------------------
# Prediction dataclass
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """A single prediction placed by an agent."""
    id: str
    agent: str
    archetype: str
    category: str
    description: str
    params: dict
    confidence: float
    stake: float
    outcome: object = None
    brier: object = None
    log: object = None
    payout: object = None

# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def brier_score(predicted: float, outcome: float) -> float:
    """Brier score: (predicted - outcome)^2. Lower is better. Range [0, 1]."""
    p = max(0.0, min(1.0, predicted))
    o = max(0.0, min(1.0, outcome))
    return (p - o) ** 2


def log_score(predicted: float, outcome: float) -> float:
    """Logarithmic score: log(p) if correct, log(1-p) if wrong. Always <= 0."""
    p = max(1e-9, min(1.0 - 1e-9, predicted))
    if outcome >= 0.5:
        return math.log(p)
    return math.log(1.0 - p)


def payout_from_brier(brier: float, stake: float) -> float:
    """Convert Brier score to karma payout. Better scores pay more."""
    if brier < 0.10:
        mult = 2.0
    elif brier < 0.25:
        mult = 1.5
    elif brier < 0.50:
        mult = 1.0
    elif brier < 0.75:
        mult = 0.5
    else:
        mult = 0.0
    return max(0.0, round(stake * mult, 2))

# ---------------------------------------------------------------------------
# Resolvers -- check actual terrarium sim results
# ---------------------------------------------------------------------------

def resolve_survival(params: dict, result: dict) -> object:
    """Did the named colony survive (final population > 0)?"""
    colony = _find_colony(params.get("colony", ""), result)
    if colony is None:
        return None
    return colony["final_population"] > 0


def resolve_population_peak(params: dict, result: dict) -> object:
    """Did colony peak population exceed threshold?"""
    colony_summary = _find_colony_summary(params.get("colony", ""), result)
    if colony_summary is None:
        return None
    return colony_summary["peak_pop"] >= params.get("threshold", 200)


def resolve_population_final(params: dict, result: dict) -> object:
    """Did colony final population exceed threshold?"""
    colony_summary = _find_colony_summary(params.get("colony", ""), result)
    if colony_summary is None:
        return None
    return colony_summary["end_pop"] >= params.get("threshold", 100)


def resolve_tech_unlock(params: dict, result: dict) -> object:
    """Was a specific tech unlocked by any colony?"""
    tech_name = params.get("tech", "")
    for col in result.get("colonies", []):
        tech = col.get("tech") or {}
        for unlocked in tech.get("unlocked", []):
            if unlocked.get("name") == tech_name:
                return True
    return False


def resolve_epidemic_any(params: dict, result: dict) -> object:
    """Did any epidemic hit the named colony (or any colony if unspecified)?"""
    target = params.get("colony", "")
    for col in result.get("colonies", []):
        if target and col["name"] != target:
            continue
        for ev in col.get("events", []):
            if "epidemic" in ev.get("type", ""):
                return True
    return False


def resolve_growth_rate(params: dict, result: dict) -> object:
    """Did colony growth exceed the target percentage?"""
    colony_summary = _find_colony_summary(params.get("colony", ""), result)
    if colony_summary is None:
        return None
    return colony_summary["growth_pct"] >= params.get("target_pct", 50.0)


def resolve_global_storm(params: dict, result: dict) -> object:
    """Did a global dust storm occur during the simulation?"""
    for snap in result.get("environment", {}).get("history", []):
        if snap.get("storm") == "global":
            return True
    return False


def resolve_morale_floor(params: dict, result: dict) -> object:
    """Did any colony morale drop below the threshold at any point?"""
    threshold = params.get("threshold", 0.3)
    target = params.get("colony", "")
    for col in result.get("colonies", []):
        if target and col["name"] != target:
            continue
        for h in col.get("history", []):
            if h.get("morale", 1.0) < threshold:
                return True
    return False


def resolve_total_deaths(params: dict, result: dict) -> object:
    """Did total deaths across all colonies exceed threshold?"""
    threshold = params.get("threshold", 10)
    total = sum(c.get("total_deaths", 0) for c in result.get("colonies", []))
    return total >= threshold


def resolve_total_migrations(params: dict, result: dict) -> object:
    """Did total migrations exceed threshold?"""
    threshold = params.get("threshold", 10)
    total = result.get("summary", {}).get("total_migrations", 0)
    return total >= threshold


def resolve_highest_final_pop(params: dict, result: dict) -> object:
    """Did the named colony have the highest final population?"""
    colony_name = params.get("colony", "")
    colonies = result.get("colonies", [])
    if not colonies:
        return None
    best = max(colonies, key=lambda c: c.get("final_population", 0))
    return best["name"] == colony_name


def resolve_terraforming_pct(params: dict, result: dict) -> object:
    """Did terraforming progress exceed the target percentage?"""
    target = params.get("target_pct", 5.0)
    history = result.get("environment", {}).get("history", [])
    if not history:
        return False
    progress = history[-1].get("terraforming_progress", 0.0) * 100.0
    return progress >= target


# --- helper lookups ---

def _find_colony(name: str, result: dict) -> object:
    """Find colony dict by name in result colonies."""
    for col in result.get("colonies", []):
        if col["name"] == name:
            return col
    return None


def _find_colony_summary(name: str, result: dict) -> object:
    """Find colony summary dict by name."""
    for cs in result.get("summary", {}).get("colonies", []):
        if cs["name"] == name:
            return cs
    return None


# ---------------------------------------------------------------------------
# Resolver registry and templates
# ---------------------------------------------------------------------------

RESOLVERS: dict = {
    "survival":           resolve_survival,
    "population_peak":    resolve_population_peak,
    "population_final":   resolve_population_final,
    "tech_unlock":        resolve_tech_unlock,
    "epidemic_any":       resolve_epidemic_any,
    "growth_rate":        resolve_growth_rate,
    "global_storm":       resolve_global_storm,
    "morale_floor":       resolve_morale_floor,
    "total_deaths":       resolve_total_deaths,
    "total_migrations":   resolve_total_migrations,
    "highest_final_pop":  resolve_highest_final_pop,
    "terraforming_pct":   resolve_terraforming_pct,
}

TEMPLATES: list = [
    {
        "category": "survival",
        "description": "Will {colony} survive to the end?",
        "param_gen": lambda rng: {"colony": rng.choice(COLONY_NAMES)},
        "base_rate": 0.85,
    },
    {
        "category": "population_peak",
        "description": "Will {colony} peak above {threshold}?",
        "param_gen": lambda rng: {
            "colony": rng.choice(COLONY_NAMES),
            "threshold": rng.choice([100, 150, 200, 250, 300]),
        },
        "base_rate": 0.50,
    },
    {
        "category": "population_final",
        "description": "Will {colony} end with >={threshold} colonists?",
        "param_gen": lambda rng: {
            "colony": rng.choice(COLONY_NAMES),
            "threshold": rng.choice([50, 80, 100, 150, 200]),
        },
        "base_rate": 0.55,
    },
    {
        "category": "tech_unlock",
        "description": "Will {tech} be unlocked?",
        "param_gen": lambda rng: {"tech": rng.choice(TECH_NAMES)},
        "base_rate": 0.40,
    },
    {
        "category": "epidemic_any",
        "description": "Will an epidemic hit {colony}?",
        "param_gen": lambda rng: {"colony": rng.choice(COLONY_NAMES)},
        "base_rate": 0.30,
    },
    {
        "category": "growth_rate",
        "description": "Will {colony} grow >={target_pct} pct?",
        "param_gen": lambda rng: {
            "colony": rng.choice(COLONY_NAMES),
            "target_pct": rng.choice([10.0, 25.0, 50.0, 100.0]),
        },
        "base_rate": 0.35,
    },
    {
        "category": "global_storm",
        "description": "Will a global dust storm hit Mars?",
        "param_gen": lambda _rng: {},
        "base_rate": 0.60,
    },
    {
        "category": "morale_floor",
        "description": "Will morale drop below {threshold} for {colony}?",
        "param_gen": lambda rng: {
            "colony": rng.choice(COLONY_NAMES),
            "threshold": rng.choice([0.2, 0.3, 0.4, 0.5]),
        },
        "base_rate": 0.40,
    },
    {
        "category": "total_deaths",
        "description": "Will total deaths exceed {threshold}?",
        "param_gen": lambda rng: {"threshold": rng.choice([5, 10, 25, 50, 100])},
        "base_rate": 0.55,
    },
    {
        "category": "total_migrations",
        "description": "Will total migrations exceed {threshold}?",
        "param_gen": lambda rng: {"threshold": rng.choice([5, 10, 25, 50, 100])},
        "base_rate": 0.40,
    },
    {
        "category": "highest_final_pop",
        "description": "Will {colony} have the highest final population?",
        "param_gen": lambda rng: {"colony": rng.choice(COLONY_NAMES)},
        "base_rate": 0.33,
    },
    {
        "category": "terraforming_pct",
        "description": "Will terraforming exceed {target_pct} pct?",
        "param_gen": lambda rng: {"target_pct": rng.choice([1.0, 5.0, 10.0, 25.0])},
        "base_rate": 0.15,
    },
]

# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------

def generate_predictions(
    n: int = 100,
    seed: int = 0,
) -> list:
    """Generate n predictions from random agents using templates."""
    rng = random.Random(seed)
    archetypes_list = list(AGENT_ARCHETYPES.keys())
    predictions = []

    for i in range(n):
        template = rng.choice(TEMPLATES)
        archetype_name = rng.choice(archetypes_list)
        arch = AGENT_ARCHETYPES[archetype_name]
        params = template["param_gen"](rng)

        # Agent-biased confidence around the template base rate
        base = template["base_rate"]
        noise = rng.gauss(0, arch["noise"])
        raw_conf = base + arch["bias"] + noise
        confidence = max(0.01, min(0.99, raw_conf))

        stake = round(rng.uniform(1.0, 50.0), 2)
        desc = template["description"].format(**params)
        pid = _prediction_id(i, seed, archetype_name, template["category"])

        predictions.append(Prediction(
            id=pid,
            agent=f"{archetype_name}-{i:04d}",
            archetype=archetype_name,
            category=template["category"],
            description=desc,
            params=params,
            confidence=round(confidence, 4),
            stake=stake,
        ))

    return predictions


def _prediction_id(index: int, seed: int, archetype: str, category: str) -> str:
    """Deterministic unique ID for a prediction."""
    raw = f"{seed}:{index}:{archetype}:{category}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]

# ---------------------------------------------------------------------------
# Terrarium runner
# ---------------------------------------------------------------------------

def run_terrarium(sols: int = 365, seed: int = 42) -> dict:
    """Run one terrarium simulation, return results dict."""
    sim = Simulation(sols=sols, env_seed=seed)
    return sim.run()


def run_terrarium_ensemble(
    sols: int = 365,
    seeds: list = None,
) -> list:
    """Run terrarium across multiple seeds, return list of result dicts."""
    seeds = seeds or [42, 43, 44]
    return [run_terrarium(sols=sols, seed=s) for s in seeds]

# ---------------------------------------------------------------------------
# Resolution -- majority vote across ensemble
# ---------------------------------------------------------------------------

def resolve_predictions(
    predictions: list,
    results: list,
) -> list:
    """Resolve predictions via majority vote across ensemble results."""
    for pred in predictions:
        resolver = RESOLVERS.get(pred.category)
        if resolver is None:
            pred.outcome = None
            continue
        votes = []
        for result in results:
            v = resolver(pred.params, result)
            if v is not None:
                votes.append(v)
        if not votes:
            pred.outcome = None
            continue
        true_count = sum(1 for v in votes if v)
        pred.outcome = true_count > len(votes) / 2
    return predictions

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_predictions(predictions: list) -> list:
    """Score all resolved predictions with Brier, log, and payout."""
    for pred in predictions:
        if pred.outcome is None:
            continue
        outcome_val = 1.0 if pred.outcome else 0.0
        pred.brier = round(brier_score(pred.confidence, outcome_val), 6)
        pred.log = round(log_score(pred.confidence, outcome_val), 6)
        pred.payout = payout_from_brier(pred.brier, pred.stake)
    return predictions

# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

def build_calibration_curve(
    predictions: list,
    n_buckets: int = 5,
) -> list:
    """Build calibration curve: bucket predictions by confidence, compute actual rate."""
    resolved = [p for p in predictions if p.outcome is not None]
    if not resolved:
        return [{"bucket_lo": i / n_buckets, "bucket_hi": (i + 1) / n_buckets,
                 "mean_confidence": 0.0, "actual_rate": 0.0, "count": 0}
                for i in range(n_buckets)]

    buckets = [[] for _ in range(n_buckets)]
    for pred in resolved:
        idx = min(int(pred.confidence * n_buckets), n_buckets - 1)
        buckets[idx].append(pred)

    curve = []
    for i, bucket in enumerate(buckets):
        lo = i / n_buckets
        hi = (i + 1) / n_buckets
        if not bucket:
            curve.append({
                "bucket_lo": lo, "bucket_hi": hi,
                "mean_confidence": (lo + hi) / 2,
                "actual_rate": 0.0, "count": 0,
            })
        else:
            mean_conf = sum(p.confidence for p in bucket) / len(bucket)
            actual = sum(1 for p in bucket if p.outcome) / len(bucket)
            curve.append({
                "bucket_lo": lo, "bucket_hi": hi,
                "mean_confidence": round(mean_conf, 4),
                "actual_rate": round(actual, 4),
                "count": len(bucket),
            })
    return curve

# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def build_leaderboard(predictions: list) -> list:
    """Build agent leaderboard ranked by mean Brier score (ascending)."""
    agent_preds = {}
    for pred in predictions:
        if pred.brier is not None:
            agent_preds.setdefault(pred.agent, []).append(pred)

    rows = []
    for agent, preds in agent_preds.items():
        mean_brier = sum(p.brier for p in preds) / len(preds)
        mean_log = sum(p.log for p in preds if p.log is not None) / max(1, len(preds))
        total_payout = sum(p.payout for p in preds if p.payout is not None)
        total_stake = sum(p.stake for p in preds)
        rows.append({
            "agent": agent,
            "archetype": preds[0].archetype,
            "n_predictions": len(preds),
            "mean_brier": round(mean_brier, 4),
            "mean_log": round(mean_log, 4),
            "total_stake": round(total_stake, 2),
            "total_payout": round(total_payout, 2),
            "net_karma": round(total_payout - total_stake, 2),
        })

    rows.sort(key=lambda r: r["mean_brier"])
    return rows

# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def assemble_report(
    predictions: list,
    calibration: list,
    leaderboard: list,
    results: list,
) -> dict:
    """Assemble the full market report."""
    resolved = [p for p in predictions if p.outcome is not None]
    unresolved = [p for p in predictions if p.outcome is None]
    correct = [p for p in resolved if (p.confidence >= 0.5) == p.outcome]

    # Category breakdown
    cat_stats = {}
    for pred in resolved:
        cat = pred.category
        if cat not in cat_stats:
            cat_stats[cat] = {"count": 0, "sum_brier": 0.0, "correct": 0}
        cat_stats[cat]["count"] += 1
        cat_stats[cat]["sum_brier"] += (pred.brier or 0.0)
        if (pred.confidence >= 0.5) == pred.outcome:
            cat_stats[cat]["correct"] += 1

    categories = []
    for cat, st in sorted(cat_stats.items()):
        categories.append({
            "category": cat,
            "count": st["count"],
            "mean_brier": round(st["sum_brier"] / max(1, st["count"]), 4),
            "accuracy": round(st["correct"] / max(1, st["count"]), 4),
        })

    # Terrarium summary (from first result)
    terrarium_summary = results[0].get("summary", {}) if results else {}

    return {
        "total_predictions": len(predictions),
        "resolved": len(resolved),
        "unresolved": len(unresolved),
        "accuracy": round(len(correct) / max(1, len(resolved)), 4),
        "mean_brier": round(
            sum(p.brier for p in resolved if p.brier is not None) / max(1, len(resolved)),
            4,
        ),
        "calibration": calibration,
        "leaderboard": leaderboard[:10],
        "categories": categories,
        "terrarium_summary": terrarium_summary,
        "n_seeds": len(results),
    }

# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_market(
    n_predictions: int = 100,
    sols: int = 365,
    seeds: list = None,
    market_seed: int = 0,
) -> dict:
    """Full pipeline: generate, simulate, resolve, score, report."""
    seeds = seeds or [42, 43, 44]
    predictions = generate_predictions(n=n_predictions, seed=market_seed)
    results = run_terrarium_ensemble(sols=sols, seeds=seeds)
    resolve_predictions(predictions, results)
    score_predictions(predictions)
    calibration = build_calibration_curve(predictions)
    leaderboard = build_leaderboard(predictions)
    return assemble_report(predictions, calibration, leaderboard, results)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(report: dict, quiet: bool = False) -> None:
    """Pretty-print market report to stdout."""
    print("=" * 60)
    print("  PREDICTION MARKET -- MARS BARN TERRARIUM")
    print("=" * 60)
    resolved = report["resolved"]
    unresolv = report["unresolved"]
    total = report["total_predictions"]
    acc = report["accuracy"]
    mb = report["mean_brier"]
    ns = report["n_seeds"]
    print(f"  Predictions: {total}  Resolved: {resolved}  Unresolved: {unresolv}")
    print(f"  Accuracy: {acc:.1%}  Mean Brier: {mb:.4f}  Seeds: {ns}")
    print()

    if not quiet:
        print("  CALIBRATION CURVE (5 buckets)")
        print("  " + "-" * 50)
        for b in report["calibration"]:
            bar_len = int(b["actual_rate"] * 30) if b["count"] > 0 else 0
            bar = "#" * bar_len
            lo = b["bucket_lo"]
            hi = b["bucket_hi"]
            cnt = b["count"]
            mc = b["mean_confidence"]
            ar = b["actual_rate"]
            print(f"  [{lo:.1f}-{hi:.1f}] n={cnt:3d}  pred={mc:.2f}  actual={ar:.2f}  {bar}")
        print()

        print("  LEADERBOARD (top 10)")
        print("  " + "-" * 50)
        hdr = f"  {'Agent':<22} {'Type':<12} {'Brier':>6} {'Net':>8} {'N':>4}"
        print(hdr)
        for row in report["leaderboard"][:10]:
            ag = row["agent"]
            at = row["archetype"]
            br = row["mean_brier"]
            nk = row["net_karma"]
            np_ = row["n_predictions"]
            print(f"  {ag:<22} {at:<12} {br:>6.3f} {nk:>+8.1f} {np_:>4}")
        print()

        print("  CATEGORY BREAKDOWN")
        print("  " + "-" * 50)
        for cat in report["categories"]:
            cn = cat["category"]
            cc = cat["count"]
            cb = cat["mean_brier"]
            ca = cat["accuracy"]
            print(f"  {cn:<22} n={cc:3d}  brier={cb:.3f}  acc={ca:.1%}")
        print()

    summary = report.get("terrarium_summary", {})
    colonies = summary.get("colonies", [])
    if colonies:
        print("  TERRARIUM OUTCOME")
        print("  " + "-" * 50)
        for cs in colonies:
            nm = cs["name"]
            status = "ALIVE" if cs["end_pop"] > 0 else "DEAD"
            ep = cs["end_pop"]
            gp = cs["growth_pct"]
            tu = cs["techs_unlocked"]
            print(f"  {nm:<22} {status:<6} pop={ep:>4}  growth={gp:>+6.1f}%  techs={tu}")
        mig = summary.get("total_migrations", 0)
        print(f"  Total migrations: {mig}")
    print("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prediction market engine for Mars Barn terrarium",
    )
    parser.add_argument("--predictions", type=int, default=100,
                        help="Number of predictions to generate (default: 100)")
    parser.add_argument("--sols", type=int, default=365,
                        help="Simulation length in sols (default: 365)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of ensemble seeds (default: 3)")
    parser.add_argument("--market-seed", type=int, default=0,
                        help="RNG seed for prediction generation (default: 0)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON report to file")
    args = parser.parse_args()

    seed_list = list(range(42, 42 + args.seeds))
    report = run_market(
        n_predictions=args.predictions,
        sols=args.sols,
        seeds=seed_list,
        market_seed=args.market_seed,
    )

    _print_report(report, quiet=args.quiet)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n  Report written to {out_path}")


if __name__ == "__main__":
    main()
