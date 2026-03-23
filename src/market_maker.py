"""
market_maker.py — Prediction market engine for Mars Barn outcomes.

Six-stage pipe: GENERATE → COUNTER → RESOLVE → SCORE → STAKE → OUTPUT.

v2.0 — Incorporates community feedback from Discussion #5892:
  - Counter-positions: agents can bet AGAINST predictions (wildcard-07)
  - Sharpness bonus: brave predictions rewarded, timidity penalized (philosopher-06)
  - Explicit vs imputed confidence tracking (debater-06)
  - Risk-adjusted payouts: 0.5 confidence never wins big (philosopher-06)

Generates predictions about Mars colony simulation outcomes,
runs the sim to resolve them, scores with Brier scores, computes
calibration curves and leaderboards. Zero external dependencies.

Usage:
    python src/market_maker.py
    python src/market_maker.py --sols 365 --seed 42
    python src/market_maker.py --predictions 50 --quiet
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ─── Prediction templates ───
# Each template generates a testable proposition about a Mars colony run.

PREDICTION_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "pop_exceed_{colony}_{threshold}",
        "text": "{colony} will exceed {threshold} population by sol {deadline}",
        "check": lambda results, colony, threshold, deadline, **kw: (
            any(h["population"] > threshold
                for h in results["colonies"][colony]["history"]
                if h["sol"] <= deadline)
        ),
        "params_fn": lambda rng, n_colonies: {
            "colony": rng.randint(0, n_colonies - 1),
            "threshold": rng.choice([100, 150, 200, 250, 300]),
            "deadline": rng.choice([180, 270, 365]),
        },
    },
    {
        "id": "pop_below_{colony}_{threshold}",
        "text": "{colony} will drop below {threshold} population at some point",
        "check": lambda results, colony, threshold, **kw: (
            any(h["population"] < threshold
                for h in results["colonies"][colony]["history"])
        ),
        "params_fn": lambda rng, n_colonies: {
            "colony": rng.randint(0, n_colonies - 1),
            "threshold": rng.choice([30, 50, 70, 90]),
        },
    },
    {
        "id": "morale_above_{colony}_{threshold}",
        "text": "{colony} will maintain morale above {threshold} for 100+ sols",
        "check": lambda results, colony, threshold, **kw: (
            _consecutive_above(
                [h["morale"] for h in results["colonies"][colony]["history"]],
                threshold, 100)
        ),
        "params_fn": lambda rng, n_colonies: {
            "colony": rng.randint(0, n_colonies - 1),
            "threshold": rng.choice([0.5, 0.6, 0.7, 0.8]),
        },
    },
    {
        "id": "deaths_exceed_{colony}_{count}",
        "text": "{colony} will suffer more than {count} total deaths",
        "check": lambda results, colony, count, **kw: (
            results["colonies"][colony]["total_deaths"] > count
        ),
        "params_fn": lambda rng, n_colonies: {
            "colony": rng.randint(0, n_colonies - 1),
            "count": rng.choice([10, 20, 30, 50, 80]),
        },
    },
    {
        "id": "storm_before_sol_{deadline}",
        "text": "A dust storm will occur before sol {deadline}",
        "check": lambda results, deadline, **kw: (
            any(e.get("storm") is not None
                for e in results["environment"]["history"]
                if e["sol"] <= deadline)
        ),
        "params_fn": lambda rng, n_colonies: {
            "deadline": rng.choice([100, 200, 300]),
        },
    },
    {
        "id": "flare_count_above_{count}",
        "text": "More than {count} solar flares will occur during the simulation",
        "check": lambda results, count, **kw: (
            sum(1 for e in results["environment"]["history"] if e.get("flare")) > count
        ),
        "params_fn": lambda rng, n_colonies: {
            "count": rng.choice([0, 1, 2, 3, 5]),
        },
    },
    {
        "id": "tech_unlock_{colony}_{count}",
        "text": "{colony} will unlock at least {count} technologies",
        "check": lambda results, colony, count, **kw: (
            (results["colonies"][colony].get("tech", {}) or {}).get("unlocked_count", 0) >= count
        ),
        "params_fn": lambda rng, n_colonies: {
            "colony": rng.randint(0, n_colonies - 1),
            "count": rng.choice([2, 4, 6, 8]),
        },
    },
    {
        "id": "growth_rate_{colony}_{pct}",
        "text": "{colony} will achieve {pct}%+ population growth",
        "check": lambda results, colony, pct, **kw: (
            _growth_pct(results["colonies"][colony]) >= pct
        ),
        "params_fn": lambda rng, n_colonies: {
            "colony": rng.randint(0, n_colonies - 1),
            "pct": rng.choice([25, 50, 75, 100, 150]),
        },
    },
    {
        "id": "migration_total_{count}",
        "text": "Total inter-colony migrations will exceed {count}",
        "check": lambda results, count, **kw: (
            results["summary"].get("total_migrations", 0) > count
        ),
        "params_fn": lambda rng, n_colonies: {
            "count": rng.choice([0, 5, 10, 20, 50]),
        },
    },
    {
        "id": "survivor_{colony}",
        "text": "{colony} will survive to the end of the simulation",
        "check": lambda results, colony, **kw: (
            results["colonies"][colony]["final_population"] > 0
        ),
        "params_fn": lambda rng, n_colonies: {
            "colony": rng.randint(0, n_colonies - 1),
        },
    },
    {
        "id": "biggest_colony_{colony}",
        "text": "{colony} will have the largest final population",
        "check": lambda results, colony, **kw: (
            results["colonies"][colony]["final_population"] ==
            max(c["final_population"] for c in results["colonies"])
        ),
        "params_fn": lambda rng, n_colonies: {
            "colony": rng.randint(0, n_colonies - 1),
        },
    },
    {
        "id": "epidemic_occurs_{colony}",
        "text": "{colony} will experience at least one epidemic",
        "check": lambda results, colony, **kw: (
            any(e.get("type") == "epidemic_start"
                for e in results["colonies"][colony].get("events", []))
        ),
        "params_fn": lambda rng, n_colonies: {
            "colony": rng.randint(0, n_colonies - 1),
        },
    },
]

# ─── Agent archetypes for prediction generation ───

PREDICTOR_ARCHETYPES: list[dict[str, Any]] = [
    {"name": "zion-oracle-01", "bias": 0.0, "confidence_range": (0.55, 0.90), "stake_range": (5, 20)},
    {"name": "zion-oracle-02", "bias": 0.05, "confidence_range": (0.60, 0.95), "stake_range": (10, 30)},
    {"name": "zion-optimist-01", "bias": 0.15, "confidence_range": (0.70, 0.95), "stake_range": (8, 25)},
    {"name": "zion-optimist-02", "bias": 0.10, "confidence_range": (0.65, 0.90), "stake_range": (5, 15)},
    {"name": "zion-pessimist-01", "bias": -0.15, "confidence_range": (0.55, 0.85), "stake_range": (10, 30)},
    {"name": "zion-pessimist-02", "bias": -0.10, "confidence_range": (0.50, 0.80), "stake_range": (8, 20)},
    {"name": "zion-gambler-01", "bias": 0.0, "confidence_range": (0.50, 0.99), "stake_range": (20, 50)},
    {"name": "zion-gambler-02", "bias": 0.0, "confidence_range": (0.51, 0.98), "stake_range": (15, 45)},
    {"name": "zion-analyst-01", "bias": 0.02, "confidence_range": (0.55, 0.85), "stake_range": (10, 20)},
    {"name": "zion-analyst-02", "bias": -0.02, "confidence_range": (0.55, 0.80), "stake_range": (10, 25)},
    {"name": "zion-contrarian-01", "bias": -0.20, "confidence_range": (0.50, 0.75), "stake_range": (5, 15)},
    {"name": "zion-philosopher-01", "bias": 0.0, "confidence_range": (0.60, 0.80), "stake_range": (5, 10)},
]


# ─── Helpers ───

def _consecutive_above(values: list[float], threshold: float, min_streak: int) -> bool:
    """Check if values stay above threshold for min_streak consecutive entries."""
    streak = 0
    for v in values:
        if v > threshold:
            streak += 1
            if streak >= min_streak:
                return True
        else:
            streak = 0
    return False


def _growth_pct(colony_data: dict) -> float:
    """Compute population growth percentage."""
    initial = colony_data.get("initial_population", 1)
    final = colony_data.get("final_population", 0)
    return ((final - initial) / max(1, initial)) * 100


def brier_score(forecast: float, outcome: float) -> float:
    """Brier score: (forecast - outcome)². Lower is better. Range [0, 1]."""
    return (forecast - outcome) ** 2


def log_score(forecast: float, outcome: float) -> float:
    """Logarithmic score. Higher (less negative) is better."""
    p = forecast if outcome == 1.0 else (1.0 - forecast)
    return math.log(max(p, 1e-10))


def deterministic_seed(text: str) -> int:
    """Deterministic hash-based seed."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


# ─── Data classes ───

@dataclass
class Prediction:
    """A single prediction about a Mars colony outcome."""
    id: str
    text: str
    author: str
    confidence: float  # 0.0–1.0
    stake: int  # karma points
    template_idx: int
    params: dict
    outcome: float | None = None  # 1.0 = true, 0.0 = false, None = unresolved
    brier: float | None = None
    log_sc: float | None = None
    payout: float = 0.0
    counter_positions: list[dict] = field(default_factory=list)

    @property
    def sharpness(self) -> float:
        """How far from 0.5 (maximum uncertainty). Range [0, 0.5].

        philosopher-06: "An agent who always predicts 0.5 never loses karma."
        Sharpness measures willingness to commit.
        """
        return abs(self.confidence - 0.5)

    @property
    def has_counters(self) -> bool:
        """Whether other agents have bet against this prediction."""
        return len(self.counter_positions) > 0

    @property
    def total_counter_stake(self) -> int:
        """Total karma staked against this prediction."""
        return sum(c["stake"] for c in self.counter_positions)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "author": self.author,
            "confidence": round(self.confidence, 3),
            "stake": self.stake,
            "sharpness": round(self.sharpness, 3),
            "outcome": self.outcome,
            "brier_score": round(self.brier, 4) if self.brier is not None else None,
            "log_score": round(self.log_sc, 4) if self.log_sc is not None else None,
            "payout": round(self.payout, 1),
            "status": "resolved" if self.outcome is not None else "open",
            "counter_positions": self.counter_positions,
            "total_counter_stake": self.total_counter_stake,
        }


@dataclass
class AgentScore:
    """Aggregated scores for one predictor agent."""
    name: str
    predictions: int = 0
    resolved: int = 0
    correct: int = 0
    total_staked: int = 0
    total_payout: float = 0.0
    brier_scores: list[float] = field(default_factory=list)
    log_scores: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.resolved)

    @property
    def mean_brier(self) -> float:
        return statistics.mean(self.brier_scores) if self.brier_scores else 1.0

    @property
    def mean_log(self) -> float:
        return statistics.mean(self.log_scores) if self.log_scores else -10.0

    @property
    def roi(self) -> float:
        if self.total_staked == 0:
            return 0.0
        return ((self.total_payout - self.total_staked) / self.total_staked) * 100

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "predictions": self.predictions,
            "resolved": self.resolved,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 3),
            "mean_brier": round(self.mean_brier, 4),
            "mean_log": round(self.mean_log, 4),
            "total_staked": self.total_staked,
            "total_payout": round(self.total_payout, 1),
            "roi_pct": round(self.roi, 1),
        }


@dataclass
class MarketState:
    """Full state of the prediction market."""
    predictions: list[Prediction]
    agent_scores: dict[str, AgentScore]
    calibration_curve: list[dict]
    colony_names: list[str]

    def to_dict(self) -> dict:
        resolved = [p for p in self.predictions if p.outcome is not None]
        unresolved = [p for p in self.predictions if p.outcome is None]
        leaderboard = sorted(
            self.agent_scores.values(),
            key=lambda a: (-a.accuracy, a.mean_brier),
        )
        countered = [p for p in self.predictions if p.has_counters]
        return {
            "_meta": {
                "engine": "market-maker",
                "version": "2.0.0",
                "pipeline": "GENERATE → COUNTER → RESOLVE → SCORE → STAKE → OUTPUT",
                "design_refs": ["#5892", "#5889", "#5893"],
                "generated": datetime.now(timezone.utc).isoformat(),
                "total_predictions": len(self.predictions),
                "resolved_count": len(resolved),
                "open_count": len(unresolved),
                "countered_count": len(countered),
            },
            "resolved_predictions": [p.to_dict() for p in resolved],
            "open_predictions": [p.to_dict() for p in unresolved],
            "leaderboard": [a.to_dict() for a in leaderboard],
            "calibration_curve": self.calibration_curve,
            "market_stats": self._compute_stats(resolved),
        }

    def _compute_stats(self, resolved: list[Prediction]) -> dict:
        if not resolved:
            return {"total_staked": 0, "total_payout": 0, "avg_brier": None}
        all_preds = self.predictions
        return {
            "total_staked": sum(p.stake for p in all_preds),
            "total_counter_staked": sum(p.total_counter_stake for p in all_preds),
            "total_payout": round(sum(p.payout for p in resolved), 1),
            "avg_brier": round(statistics.mean([p.brier for p in resolved if p.brier is not None]), 4),
            "avg_confidence": round(statistics.mean([p.confidence for p in all_preds]), 3),
            "avg_sharpness": round(statistics.mean([p.sharpness for p in all_preds]), 3),
            "resolution_rate": round(len(resolved) / max(1, len(all_preds)), 3),
            "correct_rate": round(
                sum(1 for p in resolved if p.outcome == 1.0) / max(1, len(resolved)), 3),
            "countered_rate": round(
                sum(1 for p in all_preds if p.has_counters) / max(1, len(all_preds)), 3),
        }


# ─── Stage 1: GENERATE predictions ───

def generate_predictions(
    n_predictions: int,
    colony_names: list[str],
    seed: int = 42,
) -> list[Prediction]:
    """Generate n synthetic predictions about Mars colony outcomes."""
    rng = random.Random(seed)
    predictions: list[Prediction] = []
    n_colonies = len(colony_names)

    for i in range(n_predictions):
        template = PREDICTION_TEMPLATES[i % len(PREDICTION_TEMPLATES)]
        params = template["params_fn"](rng, n_colonies)

        # Assign a predictor agent
        agent = rng.choice(PREDICTOR_ARCHETYPES)
        agent_rng = random.Random(deterministic_seed(f"{agent['name']}:{i}"))

        # Generate confidence with agent bias
        base_conf = agent_rng.uniform(*agent["confidence_range"])
        confidence = max(0.51, min(0.99, base_conf + agent["bias"]))
        stake = agent_rng.randint(*agent["stake_range"])

        # Build prediction text with colony name substitution
        text_params = dict(params)
        if "colony" in text_params:
            col_idx = text_params["colony"]
            text_params["colony"] = colony_names[col_idx]

        pred_id = template["id"].format(**text_params).replace(" ", "_")
        text = template["text"].format(**text_params)

        predictions.append(Prediction(
            id=f"pred-{i:03d}-{pred_id}",
            text=text,
            author=agent["name"],
            confidence=confidence,
            stake=stake,
            template_idx=i % len(PREDICTION_TEMPLATES),
            params=params,
        ))

    return predictions


# ─── Stage 1.5: COUNTER — Generate adversarial counter-positions ───
# wildcard-07: "Make it adversarial or admit it is a journal."
# coder-08: "Other agents should be able to post 'I take the under.'"

COUNTER_ARCHETYPES: list[dict[str, Any]] = [
    {"name": "zion-contrarian-01", "aggression": 0.6, "stake_range": (5, 25)},
    {"name": "zion-contrarian-02", "aggression": 0.4, "stake_range": (8, 20)},
    {"name": "zion-pessimist-01", "aggression": 0.5, "stake_range": (10, 30)},
    {"name": "zion-gambler-02", "aggression": 0.7, "stake_range": (15, 45)},
]


def generate_counter_positions(
    predictions: list[Prediction],
    seed: int = 42,
    counter_rate: float = 0.25,
) -> list[Prediction]:
    """Generate adversarial counter-bets against existing predictions.

    About 25% of predictions get at least one counter-position.
    This turns the solitaire into poker (wildcard-07, #5892).
    """
    rng = random.Random(seed + 9999)
    for pred in predictions:
        # High-confidence predictions attract more contrarians
        attract = pred.sharpness * 2 + 0.1
        if rng.random() < counter_rate * attract:
            n_counters = rng.randint(1, min(3, len(COUNTER_ARCHETYPES)))
            chosen = rng.sample(COUNTER_ARCHETYPES, n_counters)
            for counter_agent in chosen:
                if counter_agent["name"] == pred.author:
                    continue  # can't bet against yourself
                counter_conf = max(0.51, min(0.99, 1.0 - pred.confidence + rng.uniform(-0.1, 0.1)))
                counter_stake = rng.randint(*counter_agent["stake_range"])
                pred.counter_positions.append({
                    "agent": counter_agent["name"],
                    "counter_confidence": round(counter_conf, 3),
                    "stake": counter_stake,
                    "payout": 0.0,
                })
    return predictions

def resolve_predictions(
    predictions: list[Prediction],
    sim_results: dict,
) -> list[Prediction]:
    """Resolve each prediction against actual simulation results."""
    for pred in predictions:
        template = PREDICTION_TEMPLATES[pred.template_idx]
        try:
            outcome = template["check"](sim_results, **pred.params)
            pred.outcome = 1.0 if outcome else 0.0
        except (IndexError, KeyError, TypeError):
            pred.outcome = None  # unresolvable
    return predictions


# ─── Stage 3: SCORE predictions ───

def score_predictions(predictions: list[Prediction]) -> list[Prediction]:
    """Compute Brier and log scores for resolved predictions."""
    for pred in predictions:
        if pred.outcome is None:
            continue
        pred.brier = brier_score(pred.confidence, pred.outcome)
        pred.log_sc = log_score(pred.confidence, pred.outcome)
    return predictions


# ─── Stage 4: STAKE — compute karma payouts ───

def compute_payouts(predictions: list[Prediction]) -> list[Prediction]:
    """Compute karma payouts based on Brier scores with sharpness bonus.

    Base payout schedule:
      - Perfect (brier < 0.10): 2.0x stake
      - Good    (brier < 0.25): 1.5x stake
      - Decent  (brier < 0.40): 1.0x stake (break even)
      - Bad     (brier < 0.60): 0.5x stake (lose half)
      - Awful   (brier >= 0.60): 0.0x (lose everything)

    Sharpness bonus (philosopher-06): predictions far from 0.5
    get up to +20% bonus on correct outcomes. This counters the
    perverse incentive to always predict 0.5.

    Counter-positions: losers pay winners. If the prediction is
    true, counter-bettors lose; if false, the original author loses.
    """
    for pred in predictions:
        if pred.brier is None:
            continue

        # Base payout
        if pred.brier < 0.10:
            base_mult = 2.0
        elif pred.brier < 0.25:
            base_mult = 1.5
        elif pred.brier < 0.40:
            base_mult = 1.0
        elif pred.brier < 0.60:
            base_mult = 0.5
        else:
            base_mult = 0.0

        # Sharpness bonus: up to +20% for sharp predictions
        sharpness_bonus = pred.sharpness * 0.4  # max 0.2 at sharpness 0.5
        if base_mult > 1.0:
            base_mult += sharpness_bonus

        pred.payout = pred.stake * base_mult

        # Counter-position payouts
        if pred.counter_positions and pred.outcome is not None:
            prediction_correct = pred.outcome == 1.0
            for counter in pred.counter_positions:
                counter_brier = brier_score(counter["counter_confidence"], 1.0 - pred.outcome)
                if prediction_correct:
                    # Original author wins, counter loses
                    counter["payout"] = 0.0
                    pred.payout += counter["stake"] * 0.5  # winner takes half of counter stake
                else:
                    # Counter wins, original author loses more
                    if counter_brier < 0.25:
                        counter["payout"] = counter["stake"] * 1.5
                    elif counter_brier < 0.40:
                        counter["payout"] = counter["stake"] * 1.0
                    else:
                        counter["payout"] = counter["stake"] * 0.5

    return predictions


# ─── Stage 5: OUTPUT — aggregate and build market state ───

def build_calibration_curve(predictions: list[Prediction]) -> list[dict]:
    """Build calibration curve — bucket by stated confidence, compare to actual."""
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    curve: list[dict] = []
    for lo, hi in buckets:
        bucket_preds = [
            p for p in predictions
            if p.outcome is not None and lo <= p.confidence < hi
        ]
        if not bucket_preds:
            curve.append({
                "bucket": f"{lo:.0%}-{hi:.0%}",
                "count": 0,
                "stated_avg": round((lo + hi) / 2, 2),
                "actual_rate": None,
            })
            continue
        stated_avg = statistics.mean([p.confidence for p in bucket_preds])
        actual_rate = statistics.mean([p.outcome for p in bucket_preds])
        curve.append({
            "bucket": f"{lo:.0%}-{hi:.0%}",
            "count": len(bucket_preds),
            "stated_avg": round(stated_avg, 3),
            "actual_rate": round(actual_rate, 3),
        })
    return curve


def build_agent_scores(predictions: list[Prediction]) -> dict[str, AgentScore]:
    """Aggregate per-agent scores."""
    scores: dict[str, AgentScore] = {}
    for pred in predictions:
        if pred.author not in scores:
            scores[pred.author] = AgentScore(name=pred.author)
        agent = scores[pred.author]
        agent.predictions += 1
        agent.total_staked += pred.stake
        if pred.outcome is not None:
            agent.resolved += 1
            agent.total_payout += pred.payout
            if pred.brier is not None:
                agent.brier_scores.append(pred.brier)
            if pred.log_sc is not None:
                agent.log_scores.append(pred.log_sc)
            # "Correct" = outcome matched the direction of confidence
            if (pred.confidence > 0.5 and pred.outcome == 1.0) or \
               (pred.confidence <= 0.5 and pred.outcome == 0.0):
                agent.correct += 1
    return scores


def run_market(
    sim_results: dict,
    colony_names: list[str],
    n_predictions: int = 100,
    seed: int = 42,
) -> MarketState:
    """Run the full 6-stage prediction market pipeline.

    Stage 1:   GENERATE predictions
    Stage 1.5: COUNTER — adversarial counter-positions
    Stage 2:   RESOLVE against sim results
    Stage 3:   SCORE with Brier/log
    Stage 4:   STAKE — compute payouts (with sharpness bonus)
    Stage 5:   OUTPUT — aggregate
    """
    # Stage 1: Generate
    predictions = generate_predictions(n_predictions, colony_names, seed)

    # Stage 1.5: Counter-positions
    predictions = generate_counter_positions(predictions, seed)

    # Stage 2: Resolve
    predictions = resolve_predictions(predictions, sim_results)

    # Stage 3: Score
    predictions = score_predictions(predictions)

    # Stage 4: Stake
    predictions = compute_payouts(predictions)

    # Stage 5: Aggregate
    agent_scores = build_agent_scores(predictions)
    calibration = build_calibration_curve(predictions)

    return MarketState(
        predictions=predictions,
        agent_scores=agent_scores,
        calibration_curve=calibration,
        colony_names=colony_names,
    )


# ─── CLI entry point ───

def main() -> None:
    """Run the prediction market against a live Mars Barn simulation."""
    import argparse
    import sys

    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent
    sys.path.insert(0, str(REPO_ROOT))

    from src.tick_engine import Simulation

    parser = argparse.ArgumentParser(description="Prediction Market for Mars Barn")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predictions", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  PREDICTION MARKET v2.0 — Mars Barn Outcomes")
    print("  Pipeline: GENERATE → COUNTER → RESOLVE → SCORE → STAKE → OUTPUT")
    print("=" * 60)

    # Run simulation
    print(f"\n[1/6] Running Mars Barn simulation ({args.sols} sols, seed={args.seed})...")
    sim = Simulation(sols=args.sols, env_seed=args.seed)
    results = sim.run()

    colony_names = [c["name"] for c in results["colonies"]]
    print(f"  Colonies: {', '.join(colony_names)}")
    for c in results["colonies"]:
        print(f"    {c['name']}: {c['initial_population']} → {c['final_population']}")

    # Run market
    print(f"\n[2/6] Generating {args.predictions} predictions...")
    predictions = generate_predictions(args.predictions, colony_names, args.seed)
    print(f"  Generated {len(predictions)} predictions from {len(set(p.author for p in predictions))} agents")

    print("\n[3/6] Generating counter-positions (adversarial bets)...")
    predictions = generate_counter_positions(predictions, args.seed)
    countered = [p for p in predictions if p.has_counters]
    total_counter_stake = sum(p.total_counter_stake for p in predictions)
    print(f"  Countered: {len(countered)}/{len(predictions)} predictions")
    print(f"  Counter-stake: {total_counter_stake} karma")

    print("\n[4/6] Resolving predictions against simulation results...")
    predictions = resolve_predictions(predictions, results)
    resolved = [p for p in predictions if p.outcome is not None]
    true_count = sum(1 for p in resolved if p.outcome == 1.0)
    print(f"  Resolved: {len(resolved)}/{len(predictions)}")
    print(f"  True: {true_count}, False: {len(resolved) - true_count}")

    print("\n[5/6] Scoring with Brier scores + sharpness bonus...")
    predictions = score_predictions(predictions)
    predictions = compute_payouts(predictions)
    scored = [p for p in predictions if p.brier is not None]
    if scored:
        avg_brier = statistics.mean([p.brier for p in scored])
        avg_sharpness = statistics.mean([p.sharpness for p in predictions])
        print(f"  Avg Brier score: {avg_brier:.4f} (lower is better)")
        print(f"  Avg sharpness:   {avg_sharpness:.3f} (higher = braver)")

    print("\n[6/6] Building market state...")
    agent_scores = build_agent_scores(predictions)
    calibration = build_calibration_curve(predictions)
    market = MarketState(predictions=predictions, agent_scores=agent_scores,
                         calibration_curve=calibration, colony_names=colony_names)

    # Output
    print()
    print("=" * 60)
    print("  MARKET RESULTS")
    print("=" * 60)

    # Leaderboard
    leaderboard = sorted(agent_scores.values(), key=lambda a: (-a.accuracy, a.mean_brier))
    print("\n  📊 LEADERBOARD (by accuracy, then Brier):")
    print(f"  {'Agent':<25s} {'Acc':>5s} {'Brier':>6s} {'ROI':>7s} {'Staked':>7s} {'Payout':>8s}")
    print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")
    for a in leaderboard:
        print(f"  {a.name:<25s} {a.accuracy:5.1%} {a.mean_brier:6.3f} {a.roi:+6.1f}% {a.total_staked:7d} {a.total_payout:8.1f}")

    # Calibration
    print("\n  📈 CALIBRATION CURVE:")
    print(f"  {'Bucket':<12s} {'Count':>6s} {'Stated':>7s} {'Actual':>7s} {'Gap':>7s}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    for bucket in calibration:
        if bucket["actual_rate"] is not None:
            gap = bucket["actual_rate"] - bucket["stated_avg"]
            print(f"  {bucket['bucket']:<12s} {bucket['count']:6d} {bucket['stated_avg']:6.1%} "
                  f"{bucket['actual_rate']:6.1%} {gap:+6.1%}")
        else:
            print(f"  {bucket['bucket']:<12s} {bucket['count']:6d}   {'N/A':>5s}   {'N/A':>5s}   {'N/A':>5s}")

    # Sample predictions
    if not args.quiet:
        print("\n  🎯 SAMPLE RESOLVED PREDICTIONS:")
        for p in resolved[:15]:
            outcome_str = "✅ TRUE" if p.outcome == 1.0 else "❌ FALSE"
            print(f"    [{p.confidence:.0%}] {p.text}")
            print(f"         → {outcome_str}  Brier={p.brier:.3f}  Payout={p.payout:.0f}/{p.stake}")

    # Stats summary
    stats = market.to_dict()["market_stats"]
    print(f"\n  📋 SUMMARY:")
    print(f"    Total predictions:  {len(predictions)}")
    print(f"    Resolved:           {len(resolved)}")
    print(f"    Avg confidence:     {stats.get('avg_confidence', 0):.1%}")
    print(f"    Correct rate:       {stats.get('correct_rate', 0):.1%}")
    print(f"    Avg Brier:          {stats.get('avg_brier', 'N/A')}")
    print(f"    Total staked:       {stats.get('total_staked', 0)} karma")
    print(f"    Total payout:       {stats.get('total_payout', 0):.0f} karma")

    # Save output
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = REPO_ROOT / "state" / "market.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(market.to_dict(), indent=2))
    tmp.rename(out_path)
    print(f"\n  State saved: {out_path}")
    print()


if __name__ == "__main__":
    main()
