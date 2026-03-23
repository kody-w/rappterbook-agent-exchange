"""
adaptive_market.py — Prediction market with agent memory and Bayesian learning.

Agents don't just predict — they LEARN. After each round, agents update
their internal parameters based on how well they performed:

- Good forecasters (low Brier) → bias shrinks, noise tightens → sharper
- Bad forecasters (high Brier) → bias drifts, noise widens → noisier
- Contrarians who outperform → bias inverts less → market-seeking
- Degens who get lucky → no real learning → still chaotic

This creates a market that evolves over multiple rounds. By round 5,
the agent population has self-organized: analysts dominate, degens
wash out, and the ensemble becomes calibrated.

The output of round N is the input to round N+1. Data sloshing.

Usage:
    from src.adaptive_market import run_adaptive_market
    report = run_adaptive_market(rounds=5, predictions_per_round=50, sols=200)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any

from src.market_maker import (
    AGENT_ARCHETYPES,
    COLONY_NAMES,
    TECH_NAMES,
    TEMPLATES,
    RESOLVERS,
    Prediction,
    brier_score,
    log_score,
    payout_from_brier,
    generate_predictions,
    resolve_predictions,
    score_predictions,
    build_calibration_curve,
    build_leaderboard,
    run_terrarium,
    run_terrarium_ensemble,
)
from src.tick_engine import Simulation


# ---------------------------------------------------------------------------
# Adaptive agent state
# ---------------------------------------------------------------------------

LEARNING_RATE = 0.15          # How fast agents adapt
NOISE_DECAY = 0.92            # Noise shrinks for good forecasters
NOISE_GROWTH = 1.08           # Noise grows for bad forecasters
BIAS_DECAY = 0.85             # Bias shrinks toward zero for good forecasters
MIN_NOISE = 0.01              # Floor — even oracles have some uncertainty
MAX_NOISE = 0.40              # Ceiling — max chaos
BANKRUPTCY_KARMA = -50.0      # Agents below this get replaced
SPAWN_NOISE = 0.12            # New agents start with moderate noise


@dataclass
class AdaptiveAgent:
    """An agent that learns from prediction outcomes."""
    name: str
    archetype: str
    bias: float
    noise: float
    karma: float = 100.0
    rounds_played: int = 0
    total_predictions: int = 0
    cumulative_brier: float = 0.0
    best_brier: float = 1.0
    worst_brier: float = 0.0
    streak: int = 0            # consecutive rounds of improvement

    @property
    def mean_brier(self) -> float:
        """Lifetime mean Brier score."""
        if self.total_predictions == 0:
            return 0.5
        return self.cumulative_brier / self.total_predictions

    def update(self, round_brier: float, round_preds: int) -> None:
        """Update agent parameters based on round performance."""
        self.rounds_played += 1
        self.total_predictions += round_preds

        old_mean = self.mean_brier
        self.cumulative_brier += round_brier * round_preds

        if round_brier < 0.25:
            # Good round — tighten up
            self.noise = max(MIN_NOISE, self.noise * NOISE_DECAY)
            self.bias *= BIAS_DECAY
            self.streak += 1
        elif round_brier > 0.50:
            # Bad round — drift
            self.noise = min(MAX_NOISE, self.noise * NOISE_GROWTH)
            self.streak = 0
        else:
            # Mediocre — slight tightening
            self.noise = max(MIN_NOISE, self.noise * 0.98)
            self.streak = max(0, self.streak - 1)

        self.best_brier = min(self.best_brier, round_brier)
        self.worst_brier = max(self.worst_brier, round_brier)

    def to_dict(self) -> dict:
        """Serialize agent state."""
        return {
            "name": self.name,
            "archetype": self.archetype,
            "bias": round(self.bias, 4),
            "noise": round(self.noise, 4),
            "karma": round(self.karma, 2),
            "rounds_played": self.rounds_played,
            "total_predictions": self.total_predictions,
            "mean_brier": round(self.mean_brier, 4),
            "best_brier": round(self.best_brier, 4),
            "streak": self.streak,
        }


def create_agent_pool(n_agents: int, rng: random.Random) -> list[AdaptiveAgent]:
    """Create initial agent pool with archetype-seeded parameters."""
    archetypes = list(AGENT_ARCHETYPES.keys())
    agents = []
    for i in range(n_agents):
        arch_name = archetypes[i % len(archetypes)]
        arch = AGENT_ARCHETYPES[arch_name]
        agents.append(AdaptiveAgent(
            name=f"{arch_name}-{i:03d}",
            archetype=arch_name,
            bias=arch["bias"] + rng.gauss(0, 0.02),
            noise=arch["noise"] + rng.gauss(0, 0.01),
        ))
    return agents


def replace_bankrupt(agents: list[AdaptiveAgent], rng: random.Random) -> int:
    """Replace agents with karma below threshold. Returns count replaced."""
    replaced = 0
    archetypes = list(AGENT_ARCHETYPES.keys())
    for i, agent in enumerate(agents):
        if agent.karma < BANKRUPTCY_KARMA:
            arch_name = rng.choice(archetypes)
            agents[i] = AdaptiveAgent(
                name=f"{arch_name}-new-{rng.randint(1000, 9999)}",
                archetype=arch_name,
                bias=AGENT_ARCHETYPES[arch_name]["bias"],
                noise=SPAWN_NOISE,
            )
            replaced += 1
    return replaced


# ---------------------------------------------------------------------------
# Adaptive prediction generation
# ---------------------------------------------------------------------------

def generate_adaptive_predictions(
    agents: list[AdaptiveAgent],
    preds_per_agent: int,
    round_idx: int,
    rng: random.Random,
) -> list[Prediction]:
    """Generate predictions using agents' CURRENT (adapted) parameters."""
    predictions = []
    for agent in agents:
        for j in range(preds_per_agent):
            template = rng.choice(TEMPLATES)
            params = template["param_gen"](rng)

            # Agent-specific confidence using adapted bias/noise
            base = template["base_rate"]
            noise = rng.gauss(0, agent.noise)
            raw_conf = base + agent.bias + noise
            confidence = max(0.01, min(0.99, raw_conf))

            stake = round(rng.uniform(1.0, 50.0), 2)
            desc = template["description"].format(**params)
            pid = hashlib.sha256(
                f"{round_idx}:{agent.name}:{j}:{template['category']}".encode()
            ).hexdigest()[:12]

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


# ---------------------------------------------------------------------------
# Round result
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    """Results from one round of adaptive market."""
    round_idx: int
    n_predictions: int
    n_resolved: int
    mean_brier: float
    accuracy: float
    agents_replaced: int
    top_agent: str
    calibration: list[dict]
    agent_briers: dict[str, float]


@dataclass
class AdaptiveMarketReport:
    """Full multi-round adaptive market report."""
    n_rounds: int
    n_agents: int
    predictions_per_round: int
    sols: int
    seeds: list[int]
    rounds: list[RoundResult]
    final_agents: list[dict]
    evolution_curve: list[dict]   # mean_brier per round — should decrease
    total_replaced: int

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "n_rounds": self.n_rounds,
            "n_agents": self.n_agents,
            "predictions_per_round": self.predictions_per_round,
            "sols": self.sols,
            "seeds": self.seeds,
            "rounds": [
                {
                    "round": r.round_idx,
                    "n_predictions": r.n_predictions,
                    "n_resolved": r.n_resolved,
                    "mean_brier": r.mean_brier,
                    "accuracy": r.accuracy,
                    "agents_replaced": r.agents_replaced,
                    "top_agent": r.top_agent,
                    "calibration": r.calibration,
                }
                for r in self.rounds
            ],
            "final_agents": self.final_agents,
            "evolution_curve": self.evolution_curve,
            "total_replaced": self.total_replaced,
            "learning_signal": self._learning_signal(),
        }

    def _learning_signal(self) -> dict:
        """Quantify whether the market actually learned."""
        if len(self.rounds) < 2:
            return {"learned": False, "improvement": 0.0, "verdict": "insufficient_rounds"}
        first_brier = self.rounds[0].mean_brier
        last_brier = self.rounds[-1].mean_brier
        improvement = first_brier - last_brier
        pct = improvement / max(0.001, first_brier) * 100
        learned = improvement > 0.01
        if learned and pct > 15:
            verdict = "strong_learning"
        elif learned:
            verdict = "moderate_learning"
        elif improvement > 0:
            verdict = "marginal_learning"
        else:
            verdict = "no_learning"
        return {
            "learned": learned,
            "improvement": round(improvement, 4),
            "improvement_pct": round(pct, 1),
            "first_brier": round(first_brier, 4),
            "last_brier": round(last_brier, 4),
            "verdict": verdict,
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_adaptive_market(
    rounds: int = 5,
    n_agents: int = 18,
    preds_per_agent: int = 5,
    sols: int = 200,
    seeds: list[int] | None = None,
    rng_seed: int = 42,
) -> AdaptiveMarketReport:
    """Run multi-round adaptive prediction market.

    Each round:
    1. Agents generate predictions using their current parameters
    2. Terrarium ensemble runs, predictions resolve
    3. Agents update their bias/noise based on performance
    4. Bankrupt agents get replaced by fresh spawns
    """
    seeds = seeds or [42, 43, 44]
    rng = random.Random(rng_seed)
    agents = create_agent_pool(n_agents, rng)
    round_results: list[RoundResult] = []
    evolution_curve: list[dict] = []
    total_replaced = 0

    # Pre-run terrarium (same for all rounds — the environment is fixed)
    ensemble_results = run_terrarium_ensemble(sols=sols, seeds=seeds)

    for round_idx in range(rounds):
        # Step 1: Generate predictions with current agent states
        predictions = generate_adaptive_predictions(
            agents, preds_per_agent, round_idx, rng,
        )

        # Step 2: Resolve against terrarium
        resolve_predictions(predictions, ensemble_results)

        # Step 3: Score
        score_predictions(predictions)

        # Step 4: Compute per-agent Brier scores
        agent_briers: dict[str, list[float]] = {}
        agent_payouts: dict[str, float] = {}
        for pred in predictions:
            if pred.brier is not None:
                agent_briers.setdefault(pred.agent, []).append(pred.brier)
                agent_payouts[pred.agent] = (
                    agent_payouts.get(pred.agent, 0.0)
                    + (pred.payout or 0.0) - pred.stake
                )

        # Step 5: Update agents
        for agent in agents:
            briers = agent_briers.get(agent.name, [])
            if briers:
                round_brier = statistics.mean(briers)
                agent.update(round_brier, len(briers))
                agent.karma += agent_payouts.get(agent.name, 0.0)

        # Step 6: Replace bankrupt agents
        replaced = replace_bankrupt(agents, rng)
        total_replaced += replaced

        # Build round result
        all_briers = [b for bs in agent_briers.values() for b in bs]
        resolved = [p for p in predictions if p.outcome is not None]
        correct = sum(
            1 for p in resolved
            if (p.confidence >= 0.5) == p.outcome
        )
        mean_agent_briers = {
            name: round(statistics.mean(bs), 4)
            for name, bs in agent_briers.items()
        }
        best_agent = min(mean_agent_briers, key=mean_agent_briers.get) if mean_agent_briers else "none"

        round_results.append(RoundResult(
            round_idx=round_idx,
            n_predictions=len(predictions),
            n_resolved=len(resolved),
            mean_brier=round(statistics.mean(all_briers), 4) if all_briers else 0.5,
            accuracy=round(correct / max(1, len(resolved)), 4),
            agents_replaced=replaced,
            top_agent=best_agent,
            calibration=build_calibration_curve(predictions),
            agent_briers=mean_agent_briers,
        ))

        evolution_curve.append({
            "round": round_idx,
            "mean_brier": round_results[-1].mean_brier,
            "accuracy": round_results[-1].accuracy,
            "mean_noise": round(statistics.mean(a.noise for a in agents), 4),
            "mean_abs_bias": round(statistics.mean(abs(a.bias) for a in agents), 4),
            "agents_alive": sum(1 for a in agents if a.karma > BANKRUPTCY_KARMA),
        })

    # Build final agent standings
    final_agents = sorted(
        [a.to_dict() for a in agents],
        key=lambda x: x["mean_brier"],
    )

    return AdaptiveMarketReport(
        n_rounds=rounds,
        n_agents=n_agents,
        predictions_per_round=n_agents * preds_per_agent,
        sols=sols,
        seeds=seeds,
        rounds=round_results,
        final_agents=final_agents,
        evolution_curve=evolution_curve,
        total_replaced=total_replaced,
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_adaptive_report(report: AdaptiveMarketReport) -> None:
    """Print human-readable adaptive market report."""
    print("=" * 64)
    print("  ADAPTIVE PREDICTION MARKET — Mars Barn Terrarium")
    print("  Agents learn. Markets evolve. The swarm gets smarter.")
    print("=" * 64)
    print(f"  Rounds: {report.n_rounds}  Agents: {report.n_agents}  "
          f"Preds/round: {report.predictions_per_round}")
    print(f"  Sols: {report.sols}  Seeds: {report.seeds}")
    print()

    print("  EVOLUTION CURVE (should decrease)")
    print("  " + "-" * 56)
    for ec in report.evolution_curve:
        bar_len = int((1.0 - ec["mean_brier"]) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  Round {ec['round']}  brier={ec['mean_brier']:.4f}  "
              f"acc={ec['accuracy']:.1%}  noise={ec['mean_noise']:.3f}  {bar}")
    print()

    signal = report.to_dict()["learning_signal"]
    print(f"  LEARNING SIGNAL: {signal['verdict']}")
    print(f"  Brier: {signal['first_brier']:.4f} → {signal['last_brier']:.4f}  "
          f"({signal['improvement_pct']:+.1f}%)")
    print(f"  Agents replaced: {report.total_replaced}")
    print()

    print("  TOP 5 SURVIVING AGENTS")
    print("  " + "-" * 56)
    for agent in report.final_agents[:5]:
        print(f"  {agent['name']:<28} {agent['archetype']:<12} "
              f"brier={agent['mean_brier']:.3f}  karma={agent['karma']:+.0f}  "
              f"streak={agent['streak']}")
    print()

    print("  BOTTOM 3 AGENTS")
    print("  " + "-" * 56)
    for agent in report.final_agents[-3:]:
        print(f"  {agent['name']:<28} {agent['archetype']:<12} "
              f"brier={agent['mean_brier']:.3f}  karma={agent['karma']:+.0f}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive prediction market")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--agents", type=int, default=18)
    parser.add_argument("--preds-per-agent", type=int, default=5)
    parser.add_argument("--sols", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_adaptive_market(
        rounds=args.rounds,
        n_agents=args.agents,
        preds_per_agent=args.preds_per_agent,
        sols=args.sols,
        seeds=args.seeds,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_adaptive_report(report)
