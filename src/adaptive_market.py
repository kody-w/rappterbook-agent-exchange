"""
adaptive_market.py — Multi-round prediction market with learning agents.

Agents place predictions, get scored, then ADAPT their bias and noise
based on Brier scores. Over multiple rounds, good predictors sharpen;
bad ones drift. The swarm's collective accuracy evolves.

Key metrics:
  - Learning rate: how fast agents reduce their Brier scores
  - Information velocity: how quickly the market converges on truth
  - Swarm divergence: do agents specialize or converge?

Usage:
    from src.adaptive_market import run_adaptive_market
    report = run_adaptive_market(n_rounds=5, preds_per_round=40)
"""
from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.market_maker import (
    AGENT_ARCHETYPES,
    COLONY_NAMES,
    TEMPLATES,
    RESOLVERS,
    Prediction,
    brier_score,
    log_score,
    payout_from_brier,
    build_calibration_curve,
    build_leaderboard,
    run_terrarium_ensemble,
)


# ---------------------------------------------------------------------------
# Adaptive agent
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveAgent:
    """An agent that learns from past predictions."""
    name: str
    archetype: str
    bias: float
    noise: float
    history: list[float] = field(default_factory=list)
    category_history: dict[str, list[float]] = field(default_factory=dict)
    total_karma: float = 0.0
    rounds_played: int = 0

    def record(self, brier: float, category: str) -> None:
        """Record a Brier score and update history."""
        self.history.append(brier)
        self.category_history.setdefault(category, []).append(brier)

    def adapt(self, learning_rate: float = 0.15) -> None:
        """Adjust bias and noise based on recent performance.

        - Good Brier (< 0.25): reduce noise (agent is accurate, trust itself more)
        - Bad Brier (> 0.50): increase noise + drift bias toward center
        - Medium: small adjustments
        """
        if not self.history:
            return
        recent = self.history[-10:]
        mean_brier = statistics.mean(recent)

        if mean_brier < 0.15:
            self.noise *= (1.0 - learning_rate * 0.5)
            self.bias *= 0.95
        elif mean_brier < 0.25:
            self.noise *= (1.0 - learning_rate * 0.3)
            self.bias *= 0.97
        elif mean_brier > 0.60:
            self.noise *= (1.0 + learning_rate * 0.3)
            self.bias *= 0.80
        elif mean_brier > 0.50:
            self.noise *= (1.0 + learning_rate * 0.1)
            self.bias *= 0.90

        self.noise = max(0.01, min(0.40, self.noise))
        self.bias = max(-0.25, min(0.25, self.bias))
        self.rounds_played += 1

    def category_bias(self, category: str) -> float:
        """Extra bias from category-specific performance."""
        hist = self.category_history.get(category, [])
        if len(hist) < 2:
            return 0.0
        mean_cat = statistics.mean(hist[-5:])
        if mean_cat < 0.20:
            return -0.03
        if mean_cat > 0.55:
            return 0.05
        return 0.0

    def snapshot(self) -> dict:
        """Serialize agent state."""
        return {
            "name": self.name,
            "archetype": self.archetype,
            "bias": round(self.bias, 4),
            "noise": round(self.noise, 4),
            "mean_brier": round(statistics.mean(self.history), 4) if self.history else None,
            "n_predictions": len(self.history),
            "rounds_played": self.rounds_played,
            "total_karma": round(self.total_karma, 2),
            "categories": {
                cat: round(statistics.mean(scores), 4)
                for cat, scores in self.category_history.items()
                if scores
            },
        }


# ---------------------------------------------------------------------------
# Agent pool
# ---------------------------------------------------------------------------

def create_agent_pool(n_agents: int = 24, seed: int = 0) -> list[AdaptiveAgent]:
    """Create a diverse pool of adaptive agents."""
    rng = random.Random(seed)
    archetypes = list(AGENT_ARCHETYPES.keys())
    agents = []
    for i in range(n_agents):
        arch_name = archetypes[i % len(archetypes)]
        base = AGENT_ARCHETYPES[arch_name]
        initial_noise = base["noise"] * rng.uniform(0.8, 1.2)
        initial_bias = base["bias"] * rng.uniform(0.8, 1.2)
        agents.append(AdaptiveAgent(
            name=f"{arch_name}-{i:03d}",
            archetype=arch_name,
            bias=initial_bias,
            noise=initial_noise,
        ))
    return agents


# ---------------------------------------------------------------------------
# Round execution
# ---------------------------------------------------------------------------

def generate_round_predictions(
    agents: list[AdaptiveAgent],
    preds_per_agent: int,
    round_num: int,
    rng: random.Random,
) -> list[Prediction]:
    """Generate predictions from adaptive agents for one round."""
    predictions = []
    for agent in agents:
        for j in range(preds_per_agent):
            template = rng.choice(TEMPLATES)
            params = template["param_gen"](rng)
            base = template["base_rate"]
            cat_bias = agent.category_bias(template["category"])
            noise = rng.gauss(0, agent.noise)
            raw_conf = base + agent.bias + cat_bias + noise
            confidence = max(0.01, min(0.99, raw_conf))
            stake = round(rng.uniform(1.0, 50.0), 2)
            desc = template["description"].format(**params)
            pid = _adaptive_id(agent.name, round_num, j, template["category"])
            predictions.append(Prediction(
                id=pid,
                agent=agent.name,
                archetype=agent.archetype,
                category=template["category"],
                description=desc,
                params=params,
                confidence=round(confidence, 4),
                stake=stake,
            ))
    return predictions


def resolve_and_score(
    predictions: list[Prediction],
    ensemble_results: list[dict],
) -> list[Prediction]:
    """Resolve and score predictions against ensemble."""
    for pred in predictions:
        resolver = RESOLVERS.get(pred.category)
        if resolver is None:
            pred.outcome = None
            continue
        votes = []
        for result in ensemble_results:
            v = resolver(pred.params, result)
            if v is not None:
                votes.append(v)
        if not votes:
            pred.outcome = None
            continue
        true_count = sum(1 for v in votes if v)
        pred.outcome = true_count > len(votes) / 2

    for pred in predictions:
        if pred.outcome is None:
            continue
        outcome_val = 1.0 if pred.outcome else 0.0
        pred.brier = round(brier_score(pred.confidence, outcome_val), 6)
        pred.log = round(log_score(pred.confidence, outcome_val), 6)
        pred.payout = payout_from_brier(pred.brier, pred.stake)

    return predictions


def update_agents(
    agents: list[AdaptiveAgent],
    predictions: list[Prediction],
    learning_rate: float = 0.15,
) -> None:
    """Feed scored predictions back to agents, trigger adaptation."""
    agent_map = {a.name: a for a in agents}
    for pred in predictions:
        if pred.brier is None:
            continue
        agent = agent_map.get(pred.agent)
        if agent is None:
            continue
        agent.record(pred.brier, pred.category)
        agent.total_karma += (pred.payout or 0.0) - pred.stake

    for agent in agents:
        agent.adapt(learning_rate=learning_rate)


def _adaptive_id(agent: str, round_num: int, idx: int, category: str) -> str:
    """Deterministic prediction ID."""
    raw = f"adaptive:{agent}:{round_num}:{idx}:{category}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Round-level metrics
# ---------------------------------------------------------------------------

@dataclass
class RoundMetrics:
    """Metrics from one round of the adaptive market."""
    round_num: int
    n_predictions: int = 0
    n_resolved: int = 0
    mean_brier: float = 0.0
    accuracy: float = 0.0
    noise_mean: float = 0.0
    noise_stdev: float = 0.0
    bias_mean: float = 0.0
    calibration: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "n_predictions": self.n_predictions,
            "n_resolved": self.n_resolved,
            "mean_brier": round(self.mean_brier, 4),
            "accuracy": round(self.accuracy, 4),
            "noise_mean": round(self.noise_mean, 4),
            "noise_stdev": round(self.noise_stdev, 4),
            "bias_mean": round(self.bias_mean, 4),
        }


def compute_round_metrics(
    round_num: int,
    predictions: list[Prediction],
    agents: list[AdaptiveAgent],
) -> RoundMetrics:
    """Compute metrics for a completed round."""
    resolved = [p for p in predictions if p.outcome is not None]
    briers = [p.brier for p in resolved if p.brier is not None]
    correct = sum(
        1 for p in resolved
        if (p.confidence > 0.5 and p.outcome) or (p.confidence <= 0.5 and not p.outcome)
    )
    noises = [a.noise for a in agents]
    biases = [a.bias for a in agents]
    return RoundMetrics(
        round_num=round_num,
        n_predictions=len(predictions),
        n_resolved=len(resolved),
        mean_brier=statistics.mean(briers) if briers else 0.0,
        accuracy=correct / max(1, len(resolved)),
        noise_mean=statistics.mean(noises),
        noise_stdev=statistics.stdev(noises) if len(noises) > 1 else 0.0,
        bias_mean=statistics.mean(biases),
        calibration=build_calibration_curve(predictions),
    )


# ---------------------------------------------------------------------------
# Evolution metrics (across rounds)
# ---------------------------------------------------------------------------

@dataclass
class MarketEvolution:
    """Tracks how the market evolves across rounds."""
    n_rounds: int = 0
    brier_trajectory: list[float] = field(default_factory=list)
    accuracy_trajectory: list[float] = field(default_factory=list)
    noise_trajectory: list[float] = field(default_factory=list)
    learning_rate_estimate: float = 0.0
    information_velocity: float = 0.0
    swarm_divergence: float = 0.0
    converged: bool = False

    def to_dict(self) -> dict:
        return {
            "n_rounds": self.n_rounds,
            "brier_trajectory": [round(b, 4) for b in self.brier_trajectory],
            "accuracy_trajectory": [round(a, 4) for a in self.accuracy_trajectory],
            "noise_trajectory": [round(n, 4) for n in self.noise_trajectory],
            "learning_rate_estimate": round(self.learning_rate_estimate, 4),
            "information_velocity": round(self.information_velocity, 4),
            "swarm_divergence": round(self.swarm_divergence, 4),
            "converged": self.converged,
            "label": self.label(),
        }

    def label(self) -> str:
        """Human-readable evolution summary."""
        if self.learning_rate_estimate > 0.05:
            return "Fast learner — swarm improves rapidly"
        if self.learning_rate_estimate > 0.01:
            return "Steady learner — gradual improvement"
        if self.learning_rate_estimate > -0.01:
            return "Plateau — market found its equilibrium"
        return "Degrading — noise overwhelming signal"


def compute_evolution(round_metrics: list[RoundMetrics], agents: list[AdaptiveAgent]) -> MarketEvolution:
    """Compute evolution metrics from round history."""
    evo = MarketEvolution()
    evo.n_rounds = len(round_metrics)
    evo.brier_trajectory = [r.mean_brier for r in round_metrics]
    evo.accuracy_trajectory = [r.accuracy for r in round_metrics]
    evo.noise_trajectory = [r.noise_mean for r in round_metrics]

    if len(evo.brier_trajectory) >= 2:
        deltas = [
            evo.brier_trajectory[i] - evo.brier_trajectory[i + 1]
            for i in range(len(evo.brier_trajectory) - 1)
        ]
        evo.learning_rate_estimate = statistics.mean(deltas)
        evo.information_velocity = sum(
            max(0, d) for d in deltas
        ) / len(deltas)
    else:
        evo.learning_rate_estimate = 0.0
        evo.information_velocity = 0.0

    if agents:
        noises = [a.noise for a in agents]
        evo.swarm_divergence = (
            statistics.stdev(noises) / statistics.mean(noises)
            if len(noises) > 1 and statistics.mean(noises) > 0
            else 0.0
        )

    if len(evo.brier_trajectory) >= 3:
        last_3 = evo.brier_trajectory[-3:]
        spread = max(last_3) - min(last_3)
        evo.converged = spread < 0.02

    return evo


# ---------------------------------------------------------------------------
# Full adaptive market report
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveMarketReport:
    """Complete adaptive market output."""
    timestamp: str = ""
    n_rounds: int = 0
    n_agents: int = 0
    preds_per_agent: int = 0
    sols: int = 0
    n_seeds: int = 0
    rounds: list[dict] = field(default_factory=list)
    evolution: MarketEvolution = field(default_factory=MarketEvolution)
    final_leaderboard: list[dict] = field(default_factory=list)
    agent_snapshots: list[dict] = field(default_factory=list)
    terrarium_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "n_rounds": self.n_rounds,
            "n_agents": self.n_agents,
            "preds_per_agent": self.preds_per_agent,
            "total_predictions": self.n_rounds * self.n_agents * self.preds_per_agent,
            "sols": self.sols,
            "n_seeds": self.n_seeds,
            "rounds": self.rounds,
            "evolution": self.evolution.to_dict(),
            "final_leaderboard": self.final_leaderboard[:10],
            "agent_snapshots": self.agent_snapshots[:10],
            "terrarium_summary": self.terrarium_summary,
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_adaptive_market(
    n_rounds: int = 5,
    n_agents: int = 24,
    preds_per_agent: int = 2,
    sols: int = 365,
    n_seeds: int = 3,
    learning_rate: float = 0.15,
    market_seed: int = 42,
) -> AdaptiveMarketReport:
    """Run a multi-round adaptive prediction market.

    Each round:
      1. Agents generate predictions using their current bias/noise
      2. Terrarium ensemble resolves outcomes
      3. Predictions are scored
      4. Agents adapt based on their Brier scores
      5. Metrics are recorded

    Returns a full report with evolution trajectory.
    """
    rng = random.Random(market_seed)
    seed_list = list(range(42, 42 + n_seeds))

    ensemble_results = run_terrarium_ensemble(sols=sols, seeds=seed_list)

    agents = create_agent_pool(n_agents=n_agents, seed=market_seed)
    all_predictions: list[Prediction] = []
    round_metrics_list: list[RoundMetrics] = []

    for round_num in range(n_rounds):
        preds = generate_round_predictions(
            agents=agents,
            preds_per_agent=preds_per_agent,
            round_num=round_num,
            rng=rng,
        )
        resolve_and_score(preds, ensemble_results)
        update_agents(agents, preds, learning_rate=learning_rate)
        metrics = compute_round_metrics(round_num, preds, agents)
        round_metrics_list.append(metrics)
        all_predictions.extend(preds)

    evolution = compute_evolution(round_metrics_list, agents)
    leaderboard = build_leaderboard(all_predictions)

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
            "end_pop": c["final_population"],
            "alive": c["final_population"] > 0,
        })

    report = AdaptiveMarketReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        n_rounds=n_rounds,
        n_agents=n_agents,
        preds_per_agent=preds_per_agent,
        sols=sols,
        n_seeds=n_seeds,
        rounds=[m.to_dict() for m in round_metrics_list],
        evolution=evolution,
        final_leaderboard=leaderboard,
        agent_snapshots=[a.snapshot() for a in sorted(agents, key=lambda a: statistics.mean(a.history) if a.history else 1.0)],
        terrarium_summary={
            "colonies": colonies_summary,
            "total_population": sum(c["end_pop"] for c in colonies_summary),
            "all_alive": all(c["alive"] for c in colonies_summary),
        },
    )

    return report


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def format_adaptive_text(report: AdaptiveMarketReport) -> str:
    """Format adaptive market report as human-readable text."""
    lines = []
    sep = "=" * 64
    dash = "  " + "-" * 56

    lines.append(sep)
    lines.append("  ADAPTIVE PREDICTION MARKET — Learning Agents")
    lines.append(sep)
    lines.append("  %d rounds × %d agents × %d preds/agent = %d total" % (
        report.n_rounds, report.n_agents, report.preds_per_agent,
        report.n_rounds * report.n_agents * report.preds_per_agent))
    lines.append("  Terrarium: %d sols, %d seeds" % (report.sols, report.n_seeds))
    lines.append("")

    lines.append("  ROUND-BY-ROUND EVOLUTION")
    lines.append(dash)
    lines.append("  %-8s %-8s %-8s %-8s %-8s" % (
        "Round", "Brier", "Acc", "Noise", "Resolved"))
    for r in report.rounds:
        lines.append("  %-8d %-8.4f %-8.1f%% %-8.4f %-8d" % (
            r["round"], r["mean_brier"], r["accuracy"] * 100,
            r["noise_mean"], r["n_resolved"]))
    lines.append("")

    evo = report.evolution
    lines.append("  MARKET EVOLUTION")
    lines.append(dash)
    lines.append("  Learning rate:  %+.4f Brier/round" % evo.learning_rate_estimate)
    lines.append("  Info velocity:  %.4f bits/round" % evo.information_velocity)
    lines.append("  Swarm div:      %.4f (agent noise CV)" % evo.swarm_divergence)
    lines.append("  Converged:      %s" % ("YES" if evo.converged else "NO"))
    lines.append("  Verdict:        %s" % evo.label())
    lines.append("")

    brier_traj = evo.brier_trajectory
    if brier_traj:
        lines.append("  BRIER TRAJECTORY")
        lines.append(dash)
        max_brier = max(brier_traj) if brier_traj else 1.0
        for i, b in enumerate(brier_traj):
            bar_len = int(b / max(max_brier, 0.01) * 30)
            bar = "█" * bar_len
            lines.append("  R%d  %.4f  %s" % (i, b, bar))
        lines.append("")

    lines.append("  TOP 5 ADAPTIVE AGENTS (by mean Brier)")
    lines.append(dash)
    for snap in report.agent_snapshots[:5]:
        mb = snap.get("mean_brier")
        mb_str = "%.4f" % mb if mb is not None else "N/A"
        lines.append("  %-22s %-12s brier=%s  noise=%.3f  karma=%+.1f" % (
            snap["name"], snap["archetype"], mb_str,
            snap["noise"], snap["total_karma"]))
    lines.append("")

    lines.append("  TERRARIUM STATE")
    lines.append(dash)
    for c in report.terrarium_summary.get("colonies", []):
        status = "ALIVE" if c["alive"] else "DEAD"
        lines.append("  %-22s %s  pop=%d" % (c["name"], status, c["end_pop"]))
    tp = report.terrarium_summary.get("total_population", 0)
    lines.append("  Total population: %d" % tp)
    lines.append(sep)

    return "\n".join(lines)


def format_adaptive_compact(report: AdaptiveMarketReport) -> str:
    """One-liner summary of adaptive market."""
    evo = report.evolution
    b0 = evo.brier_trajectory[0] if evo.brier_trajectory else 0.0
    bf = evo.brier_trajectory[-1] if evo.brier_trajectory else 0.0
    tp = report.terrarium_summary.get("total_population", 0)
    return (
        "ADAPTIVE [%dr×%da] Brier %.4f→%.4f LR=%+.4f IV=%.4f Pop=%d %s" % (
            report.n_rounds, report.n_agents,
            b0, bf, evo.learning_rate_estimate, evo.information_velocity,
            tp, evo.label()))
