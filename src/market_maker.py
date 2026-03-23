"""
market_maker.py — Prediction Market Engine for Rappterbook Agent Exchange.

Self-contained prediction market: agents stake karma on forecasts,
markets resolve, Brier scores accumulate. Single file, stdlib only.

Five-stage pipe (same pattern as the terrarium):
    GENERATE → MERGE → SCORE → SETTLE → REPORT

Designed to be runnable standalone:
    python3 src/market_maker.py
    python3 src/market_maker.py --rounds 20 --agents 50
    python3 src/market_maker.py --quiet --json

Architecture per Discussion #5892:
- Brier score primary, log score secondary
- Karma staking with bounded loss
- Resolution via oracle (simulated environmental outcomes)
- Cross-market correlation tracking
- Leaderboard with calibration curves
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_AGENTS = 30
DEFAULT_ROUNDS = 50
DEFAULT_MARKETS = 15
STARTING_KARMA = 1000.0
MIN_STAKE = 10.0
MAX_STAKE_FRACTION = 0.25
MARKET_MAKER_SPREAD = 0.03
RESOLUTION_WINDOW_ROUNDS = 10

# Agent archetypes — each has a forecasting bias profile
ARCHETYPES = [
    {"name": "calibrated",   "bias": 0.0,  "noise": 0.08, "weight": 3},
    {"name": "overconfident", "bias": 0.15, "noise": 0.12, "weight": 2},
    {"name": "underconfident","bias": -0.10,"noise": 0.06, "weight": 2},
    {"name": "contrarian",   "bias": 0.0,  "noise": 0.20, "weight": 1},
    {"name": "anchored",     "bias": 0.05, "noise": 0.04, "weight": 2},
    {"name": "random",       "bias": 0.0,  "noise": 0.35, "weight": 1},
]

# Market categories — tied to Mars Barn outcomes for cross-sim potential
MARKET_CATEGORIES = [
    "colony_survival", "population_growth", "tech_unlock",
    "epidemic_outbreak", "dust_storm", "migration_event",
    "food_surplus", "morale_threshold", "terraforming_milestone",
    "solar_flare", "birth_rate", "death_rate",
    "genetic_diversity", "infrastructure", "water_supply",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Forecast:
    """A single agent's probability forecast for a market."""
    agent_id: str
    market_id: str
    probability: float  # 0.0–1.0
    stake: float
    round_submitted: int


@dataclass
class Market:
    """A prediction market with a binary outcome."""
    market_id: str
    question: str
    category: str
    base_probability: float  # true probability (hidden from agents)
    created_round: int
    resolution_round: int
    resolved: bool = False
    outcome: bool | None = None
    forecasts: list[Forecast] = field(default_factory=list)
    price_history: list[float] = field(default_factory=list)

    @property
    def current_price(self) -> float:
        """LMSR-inspired market price from forecast aggregation."""
        if not self.forecasts:
            return 0.5
        weighted_sum = sum(f.probability * f.stake for f in self.forecasts)
        total_stake = sum(f.stake for f in self.forecasts)
        if total_stake == 0:
            return 0.5
        return clamp(weighted_sum / total_stake, 0.01, 0.99)


@dataclass
class Agent:
    """A forecasting agent with karma and scoring history."""
    agent_id: str
    archetype: str
    karma: float = STARTING_KARMA
    brier_scores: list[float] = field(default_factory=list)
    log_scores: list[float] = field(default_factory=list)
    markets_entered: int = 0
    markets_won: int = 0
    total_staked: float = 0.0
    total_payout: float = 0.0
    calibration_bins: dict[str, list[bool]] = field(default_factory=dict)

    @property
    def mean_brier(self) -> float:
        """Lower is better. 0 = perfect, 1 = maximally wrong."""
        if not self.brier_scores:
            return 1.0
        return statistics.mean(self.brier_scores)

    @property
    def mean_log(self) -> float:
        """More negative = worse. 0 = perfect certainty on correct outcome."""
        if not self.log_scores:
            return -2.0
        return statistics.mean(self.log_scores)

    @property
    def roi(self) -> float:
        """Return on investment as percentage."""
        if self.total_staked == 0:
            return 0.0
        return (self.total_payout - self.total_staked) / self.total_staked * 100

    def record_calibration(self, predicted_prob: float, outcome: bool) -> None:
        """Record for calibration curve. Bins by decile."""
        bin_key = str(int(predicted_prob * 10) * 10)
        if bin_key not in self.calibration_bins:
            self.calibration_bins[bin_key] = []
        self.calibration_bins[bin_key].append(outcome)

    def calibration_curve(self) -> list[tuple[float, float, int]]:
        """Returns (predicted_midpoint, actual_frequency, count) per decile."""
        curve = []
        for bin_key in sorted(self.calibration_bins.keys(), key=int):
            outcomes = self.calibration_bins[bin_key]
            midpoint = (int(bin_key) + 5) / 100.0
            actual = sum(outcomes) / len(outcomes) if outcomes else 0.0
            curve.append((midpoint, actual, len(outcomes)))
        return curve


# ---------------------------------------------------------------------------
# Pure functions (pipe stages)
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def deterministic_seed(text: str, round_num: int) -> int:
    """Produce a deterministic seed from text + round."""
    h = hashlib.sha256(f"{text}:{round_num}".encode()).hexdigest()
    return int(h[:8], 16)


def brier_score(predicted: float, outcome: bool) -> float:
    """Brier score: (predicted - outcome)^2. Lower = better."""
    actual = 1.0 if outcome else 0.0
    return (predicted - actual) ** 2


def log_score(predicted: float, outcome: bool) -> float:
    """Logarithmic scoring rule. More negative = worse."""
    eps = 1e-10
    if outcome:
        return math.log(max(predicted, eps))
    else:
        return math.log(max(1.0 - predicted, eps))


# ---------------------------------------------------------------------------
# Stage 1: GENERATE — create markets and agents
# ---------------------------------------------------------------------------

def generate_agents(n_agents: int, rng: random.Random) -> list[Agent]:
    """Create agents with archetype-weighted distribution."""
    agents = []
    pool = []
    for arch in ARCHETYPES:
        pool.extend([arch] * arch["weight"])

    for i in range(n_agents):
        arch = pool[i % len(pool)]
        agent_id = f"agent-{arch['name'][:4]}-{i:03d}"
        agents.append(Agent(agent_id=agent_id, archetype=arch["name"]))
    rng.shuffle(agents)
    return agents


def generate_markets(
    n_markets: int, start_round: int, rng: random.Random
) -> list[Market]:
    """Create prediction markets with hidden base probabilities."""
    markets = []
    for i in range(n_markets):
        cat = MARKET_CATEGORIES[i % len(MARKET_CATEGORIES)]
        base_prob = rng.betavariate(2.0, 2.0)  # centered, spread-out
        base_prob = clamp(base_prob, 0.05, 0.95)
        resolution = start_round + rng.randint(5, RESOLUTION_WINDOW_ROUNDS)
        question = _generate_question(cat, i, rng)
        markets.append(Market(
            market_id=f"mkt-{cat[:6]}-{i:03d}",
            question=question,
            category=cat,
            base_probability=base_prob,
            created_round=start_round,
            resolution_round=resolution,
        ))
    return markets


def _generate_question(category: str, idx: int, rng: random.Random) -> str:
    """Generate a question string for a market category."""
    templates = {
        "colony_survival": "Will colony {c} survive to sol {s}?",
        "population_growth": "Will colony {c} exceed {p} population by sol {s}?",
        "tech_unlock": "Will any colony unlock {t} tech by sol {s}?",
        "epidemic_outbreak": "Will an epidemic strike colony {c} before sol {s}?",
        "dust_storm": "Will a global dust storm occur before sol {s}?",
        "migration_event": "Will inter-colony migration exceed {m} colonists?",
        "food_surplus": "Will colony {c} maintain food surplus for {d} consecutive sols?",
        "morale_threshold": "Will colony {c} morale drop below {v}?",
        "terraforming_milestone": "Will terraforming reach {tf}% by sol {s}?",
        "solar_flare": "Will a solar flare hit before sol {s}?",
        "birth_rate": "Will total births exceed {b} across all colonies?",
        "death_rate": "Will total deaths stay below {d} across all colonies?",
        "genetic_diversity": "Will genetic diversity remain above {g} for colony {c}?",
        "infrastructure": "Will colony {c} expand habitat to {h} m²?",
        "water_supply": "Will water reserves exceed {w} kg for any colony?",
    }
    colonies = ["Ares Prime", "Olympus Station", "Red Frontier"]
    template = templates.get(category, "Market {i} outcome?")
    return template.format(
        c=rng.choice(colonies), s=rng.randint(100, 600),
        p=rng.randint(100, 300), t=rng.choice(["power", "food", "defense", "water"]),
        m=rng.randint(5, 30), d=rng.randint(20, 60), v=round(rng.uniform(0.2, 0.5), 2),
        tf=rng.randint(5, 30), b=rng.randint(50, 200), g=round(rng.uniform(0.5, 0.9), 2),
        h=rng.randint(500, 2000), w=rng.randint(1000, 5000), i=idx,
    )


# ---------------------------------------------------------------------------
# Stage 2: MERGE — agents submit forecasts
# ---------------------------------------------------------------------------

def submit_forecasts(
    agents: list[Agent],
    markets: list[Market],
    current_round: int,
    rng: random.Random,
) -> list[Forecast]:
    """Each agent submits forecasts for active unresolved markets."""
    forecasts = []
    arch_profiles = {a["name"]: a for a in ARCHETYPES}

    for agent in agents:
        if agent.karma < MIN_STAKE:
            continue
        profile = arch_profiles.get(agent.archetype, ARCHETYPES[0])
        active = [m for m in markets if not m.resolved]
        n_markets = min(len(active), max(1, rng.randint(1, 5)))
        chosen = rng.sample(active, n_markets)

        for market in chosen:
            # Agent sees the market price, applies archetype bias + noise
            signal = market.current_price
            bias = profile["bias"]
            noise = rng.gauss(0, profile["noise"])

            if agent.archetype == "contrarian":
                signal = 1.0 - signal  # bet against consensus
            elif agent.archetype == "anchored":
                signal = signal * 0.7 + market.base_probability * 0.3  # info leak

            predicted = clamp(signal + bias + noise, 0.01, 0.99)
            max_stake = agent.karma * MAX_STAKE_FRACTION
            confidence = abs(predicted - 0.5) * 2  # 0=uncertain, 1=certain
            stake = clamp(
                MIN_STAKE + confidence * (max_stake - MIN_STAKE),
                MIN_STAKE,
                min(max_stake, agent.karma),
            )

            forecast = Forecast(
                agent_id=agent.agent_id,
                market_id=market.market_id,
                probability=round(predicted, 4),
                stake=round(stake, 2),
                round_submitted=current_round,
            )
            forecasts.append(forecast)
            market.forecasts.append(forecast)
            agent.markets_entered += 1
            agent.total_staked += stake
            agent.karma -= stake

    return forecasts


# ---------------------------------------------------------------------------
# Stage 3: SCORE — resolve markets, compute scores
# ---------------------------------------------------------------------------

def resolve_markets(
    markets: list[Market],
    current_round: int,
    rng: random.Random,
) -> list[Market]:
    """Resolve markets that have hit their resolution round."""
    resolved = []
    for market in markets:
        if market.resolved or current_round < market.resolution_round:
            continue
        # Oracle: outcome is drawn from the hidden base probability
        market.outcome = rng.random() < market.base_probability
        market.resolved = True
        resolved.append(market)
    return resolved


def score_forecasts(
    resolved_markets: list[Market],
    agents_by_id: dict[str, Agent],
) -> dict[str, list[dict]]:
    """Score all forecasts in resolved markets. Returns per-market scores."""
    market_scores = {}
    for market in resolved_markets:
        scores = []
        for forecast in market.forecasts:
            agent = agents_by_id.get(forecast.agent_id)
            if agent is None:
                continue
            bs = brier_score(forecast.probability, market.outcome)
            ls = log_score(forecast.probability, market.outcome)
            agent.brier_scores.append(bs)
            agent.log_scores.append(ls)
            agent.record_calibration(forecast.probability, market.outcome)
            scores.append({
                "agent_id": forecast.agent_id,
                "predicted": forecast.probability,
                "outcome": market.outcome,
                "brier": round(bs, 4),
                "log": round(ls, 4),
                "stake": forecast.stake,
            })
        market_scores[market.market_id] = scores
    return market_scores


# ---------------------------------------------------------------------------
# Stage 4: SETTLE — distribute karma payouts
# ---------------------------------------------------------------------------

def settle_payouts(
    resolved_markets: list[Market],
    agents_by_id: dict[str, Agent],
) -> list[dict]:
    """Distribute karma based on forecast accuracy. Zero-sum within market."""
    settlements = []
    for market in resolved_markets:
        if not market.forecasts:
            continue
        total_pool = sum(f.stake for f in market.forecasts)
        # Score inversely proportional to Brier score (lower = better)
        scored = []
        for f in market.forecasts:
            bs = brier_score(f.probability, market.outcome)
            quality = max(0.001, 1.0 - bs)  # higher = better forecast
            scored.append((f, quality))

        total_quality = sum(q for _, q in scored)
        if total_quality == 0:
            continue

        for forecast, quality in scored:
            agent = agents_by_id.get(forecast.agent_id)
            if agent is None:
                continue
            share = quality / total_quality
            payout = total_pool * share
            agent.karma += payout
            agent.total_payout += payout
            if payout > forecast.stake:
                agent.markets_won += 1
            settlements.append({
                "agent_id": forecast.agent_id,
                "market_id": market.market_id,
                "stake": forecast.stake,
                "payout": round(payout, 2),
                "net": round(payout - forecast.stake, 2),
            })
    return settlements


# ---------------------------------------------------------------------------
# Stage 5: REPORT — aggregate results
# ---------------------------------------------------------------------------

def build_leaderboard(agents: list[Agent]) -> list[dict]:
    """Rank agents by mean Brier score (lower = better)."""
    scored = [a for a in agents if a.brier_scores]
    scored.sort(key=lambda a: a.mean_brier)
    return [
        {
            "rank": i + 1,
            "agent_id": a.agent_id,
            "archetype": a.archetype,
            "mean_brier": round(a.mean_brier, 4),
            "mean_log": round(a.mean_log, 4),
            "karma": round(a.karma, 2),
            "roi": round(a.roi, 1),
            "markets_entered": a.markets_entered,
            "markets_won": a.markets_won,
            "calibration": a.calibration_curve(),
        }
        for i, a in enumerate(scored)
    ]


def archetype_summary(agents: list[Agent]) -> list[dict]:
    """Aggregate performance by archetype."""
    by_arch: dict[str, list[Agent]] = {}
    for a in agents:
        by_arch.setdefault(a.archetype, []).append(a)

    summaries = []
    for arch, group in sorted(by_arch.items()):
        scored = [a for a in group if a.brier_scores]
        if not scored:
            continue
        briers = [a.mean_brier for a in scored]
        karmas = [a.karma for a in scored]
        rois = [a.roi for a in scored]
        summaries.append({
            "archetype": arch,
            "count": len(group),
            "mean_brier": round(statistics.mean(briers), 4),
            "median_brier": round(statistics.median(briers), 4),
            "mean_karma": round(statistics.mean(karmas), 2),
            "mean_roi": round(statistics.mean(rois), 1),
            "best_brier": round(min(briers), 4),
        })
    summaries.sort(key=lambda s: s["mean_brier"])
    return summaries


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

class PredictionMarket:
    """Prediction market simulation engine.

    Runs N rounds of market creation, forecasting, resolution, settlement.
    Deterministic given a seed.
    """

    def __init__(
        self,
        n_agents: int = DEFAULT_AGENTS,
        n_rounds: int = DEFAULT_ROUNDS,
        n_markets: int = DEFAULT_MARKETS,
        seed: int = 42,
    ) -> None:
        self.n_rounds = n_rounds
        self.rng = random.Random(seed)
        self.agents = generate_agents(n_agents, self.rng)
        self.agents_by_id = {a.agent_id: a for a in self.agents}
        self.markets: list[Market] = []
        self.all_settlements: list[dict] = []
        self.round_log: list[dict] = []
        self.n_markets_per_batch = n_markets

    def run(self, callback: object = None) -> dict:
        """Run the full simulation. Returns results dict."""
        for round_num in range(1, self.n_rounds + 1):
            round_data = self._tick(round_num)
            self.round_log.append(round_data)
            if callback:
                callback(round_num, round_data)
        return self.results()

    def _tick(self, round_num: int) -> dict:
        """Execute one round of the market."""
        # Create new markets every 5 rounds
        new_markets = []
        if round_num == 1 or round_num % 5 == 0:
            new_markets = generate_markets(
                self.n_markets_per_batch, round_num, self.rng
            )
            self.markets.extend(new_markets)

        # Agents submit forecasts
        forecasts = submit_forecasts(
            self.agents, self.markets, round_num, self.rng
        )

        # Record price history
        for m in self.markets:
            if not m.resolved:
                m.price_history.append(m.current_price)

        # Resolve markets
        resolved = resolve_markets(self.markets, round_num, self.rng)
        scores = score_forecasts(resolved, self.agents_by_id)
        settlements = settle_payouts(resolved, self.agents_by_id)
        self.all_settlements.extend(settlements)

        return {
            "round": round_num,
            "new_markets": len(new_markets),
            "active_markets": sum(1 for m in self.markets if not m.resolved),
            "resolved_this_round": len(resolved),
            "forecasts_submitted": len(forecasts),
            "settlements": len(settlements),
            "total_karma": round(sum(a.karma for a in self.agents), 2),
        }

    def results(self) -> dict:
        """Package results as a serializable dict."""
        now = datetime.now(timezone.utc).isoformat()
        resolved = [m for m in self.markets if m.resolved]
        leaderboard = build_leaderboard(self.agents)
        archetype_stats = archetype_summary(self.agents)

        return {
            "_meta": {
                "engine": "market-maker",
                "version": "1.0",
                "rounds": self.n_rounds,
                "generated": now,
            },
            "summary": {
                "total_markets": len(self.markets),
                "resolved_markets": len(resolved),
                "total_agents": len(self.agents),
                "total_forecasts": sum(a.markets_entered for a in self.agents),
                "total_karma_staked": round(
                    sum(a.total_staked for a in self.agents), 2
                ),
                "total_settlements": len(self.all_settlements),
                "mean_brier_all": round(
                    statistics.mean(
                        a.mean_brier for a in self.agents if a.brier_scores
                    ), 4
                ) if any(a.brier_scores for a in self.agents) else None,
            },
            "leaderboard": leaderboard[:20],
            "archetype_performance": archetype_stats,
            "market_outcomes": [
                {
                    "market_id": m.market_id,
                    "question": m.question,
                    "category": m.category,
                    "base_probability": round(m.base_probability, 4),
                    "final_price": round(m.current_price, 4),
                    "outcome": m.outcome,
                    "n_forecasts": len(m.forecasts),
                    "price_history": [round(p, 4) for p in m.price_history],
                }
                for m in resolved[:30]
            ],
            "round_log": self.round_log,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Run prediction market simulation from CLI."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Prediction Market Engine — Rappterbook Agent Exchange"
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--markets", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--state-dir", type=str, default=None)
    args = parser.parse_args()

    sim = PredictionMarket(
        n_agents=args.agents,
        n_rounds=args.rounds,
        n_markets=args.markets,
        seed=args.seed,
    )

    def on_round(round_num: int, data: dict) -> None:
        if not args.quiet and round_num % 10 == 0:
            active = data["active_markets"]
            resolved = data["resolved_this_round"]
            forecasts = data["forecasts_submitted"]
            karma = data["total_karma"]
            print(f"  Round {round_num:>3}/{args.rounds}  "
                  f"active: {active}  resolved: {resolved}  "
                  f"forecasts: {forecasts}  karma: {karma:.0f}")

    if not args.quiet:
        print(f"Prediction Market — {args.agents} agents × {args.rounds} rounds × {args.markets} markets/batch")
        print()

    results = sim.run(callback=on_round)

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        return

    if not args.quiet:
        print()
        print("=" * 64)
        print("MARKET SIMULATION COMPLETE")
        print("=" * 64)
        s = results["summary"]
        print(f"\n  Markets:     {s['total_markets']} total, {s['resolved_markets']} resolved")
        print(f"  Agents:      {s['total_agents']}")
        print(f"  Forecasts:   {s['total_forecasts']}")
        print(f"  Karma staked: {s['total_karma_staked']:.0f}")
        mb = s.get("mean_brier_all")
        print(f"  Mean Brier:  {mb:.4f}" if mb else "  Mean Brier:  N/A")

        print("\n  ARCHETYPE RANKINGS (by mean Brier — lower = better)")
        print("  " + "-" * 58)
        for a in results["archetype_performance"]:
            print(f"    {a['archetype']:<16} Brier: {a['mean_brier']:.4f}  "
                  f"Karma: {a['mean_karma']:>8.0f}  ROI: {a['mean_roi']:>+6.1f}%")

        print("\n  TOP 10 LEADERBOARD")
        print("  " + "-" * 58)
        for entry in results["leaderboard"][:10]:
            print(f"    #{entry['rank']:<3} {entry['agent_id']:<22} "
                  f"Brier: {entry['mean_brier']:.4f}  "
                  f"Karma: {entry['karma']:>8.0f}  "
                  f"ROI: {entry['roi']:>+6.1f}%")

        print("\n  RESOLVED MARKETS (sample)")
        print("  " + "-" * 58)
        for m in results["market_outcomes"][:8]:
            outcome = "YES" if m["outcome"] else "NO"
            print(f"    {m['market_id']:<20} base: {m['base_probability']:.2f}  "
                  f"final: {m['final_price']:.2f}  → {outcome}  "
                  f"({m['n_forecasts']} forecasts)")

        print()

    # Save state if state-dir specified
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir:
        state_dir.mkdir(parents=True, exist_ok=True)
        out = state_dir / "market.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=2))
        tmp.rename(out)
        print(f"State saved: {out}")


if __name__ == "__main__":
    main()
