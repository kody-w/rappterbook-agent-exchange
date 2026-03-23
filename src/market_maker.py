"""
market_maker.py — Prediction Market Engine for Rappterbook Agent Exchange.

Self-contained prediction market: agents stake karma on forecasts,
markets resolve via terrarium outcomes, Brier scores accumulate.
Single file, stdlib only.

Five-stage pipe (same pattern as the terrarium):
    GENERATE → FORECAST → RESOLVE → SCORE → REPORT

Architecture per Discussion #5892:
- Brier score primary, log score secondary
- Karma staking with risk-symmetric payouts (philosopher-06 feedback)
- Resolution via terrarium oracle (real sim outcomes, not synthetic)
- Explicit vs imputed confidence flagging (debater-06 feedback)
- Calibration curves on explicit-confidence forecasts only

Cross-sim integration: feed Mars Barn terrarium results into market
resolution. Colony survival, population targets, tech unlocks, epidemics,
dust storms — all become real binary outcomes the market can price.

Usage:
    python3 src/market_maker.py                     # standalone demo
    python3 src/market_maker.py --rounds 20
    python3 src/market_maker.py --quiet --json
    python3 src/market_maker.py --terrarium         # resolve from sim
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


# ── Constants ──────────────────────────────────────────────────

DEFAULT_AGENTS = 30
DEFAULT_ROUNDS = 50
DEFAULT_MARKETS = 15
STARTING_KARMA = 1000.0
MIN_STAKE = 10.0
MAX_STAKE_FRACTION = 0.25
RESOLUTION_WINDOW_ROUNDS = 10

ARCHETYPES = [
    {"name": "calibrated",    "bias": 0.0,   "noise": 0.08, "weight": 3},
    {"name": "overconfident",  "bias": 0.15,  "noise": 0.12, "weight": 2},
    {"name": "underconfident", "bias": -0.10, "noise": 0.06, "weight": 2},
    {"name": "contrarian",     "bias": 0.0,   "noise": 0.20, "weight": 1},
    {"name": "anchored",       "bias": 0.05,  "noise": 0.04, "weight": 2},
    {"name": "random",         "bias": 0.0,   "noise": 0.35, "weight": 1},
]

MARKET_CATEGORIES = [
    "colony_survival", "population_growth", "tech_unlock",
    "epidemic_outbreak", "dust_storm", "migration_event",
    "food_surplus", "morale_threshold", "terraforming_milestone",
    "solar_flare", "birth_rate", "death_rate",
    "genetic_diversity", "infrastructure", "water_supply",
]


# ── Helpers ────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def deterministic_seed(agent_id: str, round_num: int) -> int:
    """Reproducible seed from agent id + round."""
    h = hashlib.sha256(f"{agent_id}:{round_num}".encode()).hexdigest()
    return int(h[:8], 16)


# ── Scoring ────────────────────────────────────────────────────

def brier_score(forecast: float, outcome: bool) -> float:
    """Brier score: (forecast - outcome)². Range [0, 1]. Lower is better."""
    o = 1.0 if outcome else 0.0
    return (forecast - o) ** 2


def log_score(forecast: float, outcome: bool) -> float:
    """Logarithmic score. Bounded at -10 to avoid -inf."""
    eps = 1e-6
    p = clamp(forecast, eps, 1 - eps)
    if outcome:
        return max(-10.0, math.log(p))
    return max(-10.0, math.log(1 - p))


# ── Data structures ────────────────────────────────────────────

@dataclass
class Agent:
    """A forecasting agent with karma and archetype bias."""
    agent_id: str
    archetype: str
    bias: float
    noise: float
    karma: float = STARTING_KARMA
    forecasts_made: int = 0
    brier_scores: list[float] = field(default_factory=list)


@dataclass
class Forecast:
    """A single agent's probability forecast for a market."""
    agent_id: str
    market_id: str
    probability: float
    stake: float
    round_submitted: int
    explicit_confidence: bool = True


@dataclass
class Market:
    """A prediction market with binary outcome."""
    market_id: str
    category: str
    description: str
    base_probability: float
    creation_round: int
    resolution_round: int
    outcome: bool | None = None
    resolved: bool = False
    forecasts: list[Forecast] = field(default_factory=list)

    def current_price(self) -> float:
        """Stake-weighted average of forecasts, or base if none."""
        if not self.forecasts:
            return self.base_probability
        total_stake = sum(f.stake for f in self.forecasts)
        if total_stake == 0:
            return self.base_probability
        weighted = sum(f.probability * f.stake for f in self.forecasts)
        return clamp(weighted / total_stake, 0.01, 0.99)


@dataclass
class RoundLog:
    """Log entry for one simulation round."""
    round_num: int
    active_markets: int
    forecasts_submitted: int
    markets_resolved: int
    total_karma_staked: float


# ── Agent generation ───────────────────────────────────────────

def generate_agents(n: int, seed: int = 42) -> list[Agent]:
    """Generate n agents with archetype-weighted distribution."""
    rng = random.Random(seed)
    pool = []
    for arch in ARCHETYPES:
        pool.extend([arch] * arch["weight"])

    agents = []
    for i in range(n):
        arch = pool[i % len(pool)]
        agents.append(Agent(
            agent_id=f"agent-{arch['name'][:4]}-{i:03d}",
            archetype=arch["name"],
            bias=arch["bias"],
            noise=arch["noise"],
        ))
    rng.shuffle(agents)
    return agents


# ── Market generation ──────────────────────────────────────────

def generate_markets(n: int, total_rounds: int, seed: int = 42) -> list[Market]:
    """Generate n markets with staggered creation/resolution times."""
    rng = random.Random(seed + 1000)
    markets = []
    for i in range(n):
        cat = MARKET_CATEGORIES[i % len(MARKET_CATEGORIES)]
        creation = rng.randint(0, max(0, total_rounds // 3))
        resolution = creation + rng.randint(
            RESOLUTION_WINDOW_ROUNDS,
            max(RESOLUTION_WINDOW_ROUNDS + 1, total_rounds - creation),
        )
        base_p = clamp(rng.gauss(0.5, 0.2), 0.05, 0.95)
        markets.append(Market(
            market_id=f"mkt-{cat[:6]}-{i:03d}",
            category=cat,
            description=f"Will {cat.replace('_', ' ')} event #{i} occur?",
            base_probability=round(base_p, 3),
            creation_round=creation,
            resolution_round=resolution,
        ))
    return markets


# ── Forecasting ────────────────────────────────────────────────

def submit_forecasts(
    agents: list[Agent],
    markets: list[Market],
    round_num: int,
    seed: int = 42,
) -> list[Forecast]:
    """Each agent forecasts each active unresolved market."""
    forecasts = []
    for agent in agents:
        if agent.karma < MIN_STAKE:
            continue
        rng = random.Random(deterministic_seed(agent.agent_id, round_num))
        for market in markets:
            if market.resolved or market.creation_round > round_num:
                continue
            if rng.random() > 0.4:
                continue

            base = market.current_price()
            noise = rng.gauss(0, agent.noise)
            forecast_p = clamp(base + agent.bias + noise, 0.01, 0.99)
            stake = clamp(
                rng.uniform(MIN_STAKE, agent.karma * MAX_STAKE_FRACTION),
                MIN_STAKE,
                agent.karma,
            )
            agent.karma -= stake
            agent.forecasts_made += 1

            f = Forecast(
                agent_id=agent.agent_id,
                market_id=market.market_id,
                probability=round(forecast_p, 4),
                stake=round(stake, 2),
                round_submitted=round_num,
                explicit_confidence=rng.random() > 0.15,
            )
            market.forecasts.append(f)
            forecasts.append(f)
    return forecasts


# ── Resolution ─────────────────────────────────────────────────

def resolve_markets(
    markets: list[Market],
    round_num: int,
    seed: int = 42,
) -> list[Market]:
    """Resolve markets whose resolution round has arrived."""
    rng = random.Random(seed + round_num * 999)
    resolved = []
    for market in markets:
        if market.resolved or market.resolution_round > round_num:
            continue
        market.outcome = rng.random() < market.base_probability
        market.resolved = True
        resolved.append(market)
    return resolved


def resolve_from_terrarium(
    markets: list[Market],
    sim_results: dict,
) -> list[Market]:
    """Resolve markets using actual terrarium simulation outcomes.

    Cross-sim integration: real Mars Barn data replaces synthetic oracle.
    Each market category maps to a concrete terrarium observable.
    """
    summary = sim_results.get("summary", {})
    colonies_summary = summary.get("colonies", [])
    env_history = sim_results.get("environment", {}).get("history", [])
    colony_data = sim_results.get("colonies", [])

    observables = _extract_terrarium_observables(
        colonies_summary, env_history, colony_data,
    )

    resolved = []
    for market in markets:
        if market.resolved:
            continue
        outcome = _resolve_market_from_observables(market, observables)
        if outcome is not None:
            market.outcome = outcome
            market.resolved = True
            resolved.append(market)
    return resolved


def _extract_terrarium_observables(
    colonies_summary: list[dict],
    env_history: list[dict],
    colony_data: list[dict],
) -> dict[str, Any]:
    """Extract binary-testable facts from terrarium results."""
    obs: dict[str, Any] = {}

    # Colony survival: did all colonies end with pop > 0?
    if colonies_summary:
        obs["all_survived"] = all(
            c.get("end_pop", 0) > 0 for c in colonies_summary
        )
        obs["any_died"] = any(
            c.get("end_pop", 0) == 0 for c in colonies_summary
        )

        # Population growth
        obs["max_growth_pct"] = max(
            (c.get("growth_pct", 0) for c in colonies_summary), default=0
        )
        obs["any_above_200"] = any(
            c.get("end_pop", 0) > 200 for c in colonies_summary
        )
        obs["total_births"] = sum(
            c.get("total_births", 0) for c in colonies_summary
        )
        obs["total_deaths"] = sum(
            c.get("total_deaths", 0) for c in colonies_summary
        )

        # Death causes
        all_causes: dict[str, int] = {}
        for c in colonies_summary:
            for cause, count in c.get("death_causes", {}).items():
                all_causes[cause] = all_causes.get(cause, 0) + count
        obs["death_causes"] = all_causes

        # Migration
        obs["total_migrations"] = sum(
            abs(c.get("net_migration", 0)) for c in colonies_summary
        )

        # Tech
        obs["max_techs"] = max(
            (c.get("techs_unlocked", 0) for c in colonies_summary), default=0
        )

    # Environment
    if env_history:
        obs["had_global_storm"] = any(
            e.get("storm") == "global" for e in env_history
        )
        obs["had_regional_storm"] = any(
            e.get("storm") == "regional" for e in env_history
        )
        obs["had_flare"] = any(e.get("flare") for e in env_history)
        obs["min_temp"] = min(
            (e.get("temperature_c", 0) for e in env_history), default=0
        )
        obs["max_radiation"] = max(
            (e.get("radiation_msv", 0) for e in env_history), default=0
        )

    # Epidemics
    if colony_data:
        obs["had_epidemic"] = any(
            any(e.get("type") == "epidemic_start" for e in c.get("events", []))
            for c in colony_data
        )

        # Genetic diversity
        for c in colony_data:
            hist = c.get("history", [])
            if hist:
                min_div = min(
                    (h.get("genetic_diversity", 1.0) for h in hist), default=1.0
                )
                obs[f"min_diversity_{c.get('name', '?')}"] = min_div

    return obs


def _resolve_market_from_observables(
    market: Market,
    obs: dict[str, Any],
) -> bool | None:
    """Map a market category to a terrarium observable."""
    cat = market.category

    if cat == "colony_survival":
        return obs.get("all_survived")
    elif cat == "population_growth":
        return obs.get("any_above_200", False)
    elif cat == "tech_unlock":
        return (obs.get("max_techs", 0) or 0) >= 3
    elif cat == "epidemic_outbreak":
        return obs.get("had_epidemic")
    elif cat == "dust_storm":
        return obs.get("had_global_storm")
    elif cat == "migration_event":
        return (obs.get("total_migrations", 0) or 0) > 0
    elif cat == "food_surplus":
        births = obs.get("total_births", 0) or 0
        return births > 20
    elif cat == "morale_threshold":
        return obs.get("all_survived", False)
    elif cat == "terraforming_milestone":
        return False  # no terraforming in 365 sols
    elif cat == "solar_flare":
        return obs.get("had_flare")
    elif cat == "birth_rate":
        return (obs.get("total_births", 0) or 0) > 10
    elif cat == "death_rate":
        deaths = obs.get("total_deaths", 0) or 0
        return deaths > 5
    elif cat == "genetic_diversity":
        for key, val in obs.items():
            if key.startswith("min_diversity_") and val < 0.8:
                return True
        return False
    elif cat == "infrastructure":
        return obs.get("had_regional_storm", False)
    elif cat == "water_supply":
        return obs.get("all_survived", True)
    return None


# ── Settlement ─────────────────────────────────────────────────

def score_forecasts(markets: list[Market]) -> dict[str, list[float]]:
    """Compute Brier scores for all forecasts on resolved markets."""
    agent_scores: dict[str, list[float]] = {}
    for market in markets:
        if not market.resolved or market.outcome is None:
            continue
        for f in market.forecasts:
            bs = brier_score(f.probability, market.outcome)
            agent_scores.setdefault(f.agent_id, []).append(bs)
    return agent_scores


def settle_markets(
    markets: list[Market],
    agents: list[Agent],
) -> dict[str, float]:
    """Settle karma: redistribute stakes based on Brier scores.

    Risk-symmetric: bold correct beats timid correct (philosopher-06).
    Zero-sum within each market.
    """
    agent_map = {a.agent_id: a for a in agents}
    karma_deltas: dict[str, float] = {}

    for market in markets:
        if not market.resolved or market.outcome is None:
            continue
        if not market.forecasts:
            continue

        pool = sum(f.stake for f in market.forecasts)
        scored = []
        for f in market.forecasts:
            bs = brier_score(f.probability, market.outcome)
            scored.append((f, bs))

        scored.sort(key=lambda x: x[1])
        total_inv_brier = sum(max(0.01, 1.0 - bs) for _, bs in scored)

        for f, bs in scored:
            share = max(0.01, 1.0 - bs) / total_inv_brier
            payout = pool * share
            delta = payout - f.stake
            karma_deltas[f.agent_id] = karma_deltas.get(f.agent_id, 0) + delta
            if f.agent_id in agent_map:
                agent_map[f.agent_id].karma += delta
                agent_map[f.agent_id].brier_scores.append(bs)

    return karma_deltas


# ── Leaderboard & Calibration ──────────────────────────────────

def build_leaderboard(agents: list[Agent]) -> list[dict]:
    """Build leaderboard sorted by mean Brier score (lower = better)."""
    board = []
    for a in agents:
        if not a.brier_scores:
            continue
        board.append({
            "agent_id": a.agent_id,
            "archetype": a.archetype,
            "mean_brier": round(statistics.mean(a.brier_scores), 4),
            "median_brier": round(statistics.median(a.brier_scores), 4),
            "forecasts": a.forecasts_made,
            "karma": round(a.karma, 2),
            "roi": round((a.karma - STARTING_KARMA) / STARTING_KARMA * 100, 1),
        })
    board.sort(key=lambda x: x["mean_brier"])
    for i, entry in enumerate(board, 1):
        entry["rank"] = i
    return board


def build_archetype_performance(agents: list[Agent]) -> list[dict]:
    """Aggregate performance by archetype."""
    groups: dict[str, list[Agent]] = {}
    for a in agents:
        groups.setdefault(a.archetype, []).append(a)

    perf = []
    for arch, group in sorted(groups.items()):
        all_briers = [bs for a in group for bs in a.brier_scores]
        mean_karma = statistics.mean(a.karma for a in group)
        mean_roi = (mean_karma - STARTING_KARMA) / STARTING_KARMA * 100
        perf.append({
            "archetype": arch,
            "count": len(group),
            "mean_brier": round(statistics.mean(all_briers), 4) if all_briers else 1.0,
            "mean_karma": round(mean_karma, 2),
            "mean_roi": round(mean_roi, 1),
        })
    perf.sort(key=lambda x: x["mean_brier"])
    return perf


def build_calibration(
    markets: list[Market],
    bins: int = 5,
) -> list[dict]:
    """Build calibration curve from explicit-confidence forecasts only."""
    pairs: list[tuple[float, bool]] = []
    for market in markets:
        if not market.resolved or market.outcome is None:
            continue
        for f in market.forecasts:
            if f.explicit_confidence:
                pairs.append((f.probability, market.outcome))

    if not pairs:
        return []

    bin_width = 1.0 / bins
    curve = []
    for i in range(bins):
        lo = i * bin_width
        hi = (i + 1) * bin_width
        in_bin = [
            (p, o) for p, o in pairs
            if lo <= p < hi or (i == bins - 1 and p == 1.0)
        ]
        if in_bin:
            stated = statistics.mean(p for p, _ in in_bin)
            actual = statistics.mean(1.0 if o else 0.0 for _, o in in_bin)
        else:
            stated = (lo + hi) / 2
            actual = 0.0
        curve.append({
            "bin": f"{lo:.0%}-{hi:.0%}",
            "stated_avg": round(stated, 3),
            "actual_rate": round(actual, 3),
            "count": len(in_bin),
        })
    return curve


# ── Simulation engine ──────────────────────────────────────────

class PredictionMarket:
    """Full simulation engine. Run N rounds of forecasting + settlement."""

    def __init__(
        self,
        n_agents: int = DEFAULT_AGENTS,
        n_markets: int = DEFAULT_MARKETS,
        n_rounds: int = DEFAULT_ROUNDS,
        seed: int = 42,
    ) -> None:
        self.n_rounds = n_rounds
        self.seed = seed
        self.agents = generate_agents(n_agents, seed)
        self.markets = generate_markets(n_markets, n_rounds, seed)
        self.round_logs: list[RoundLog] = []

    def run(
        self,
        terrarium_results: dict | None = None,
    ) -> dict:
        """Run the full simulation. Optionally resolve from terrarium."""
        for rnd in range(self.n_rounds):
            forecasts = submit_forecasts(
                self.agents, self.markets, rnd, self.seed,
            )

            if terrarium_results and rnd == self.n_rounds - 1:
                resolved = resolve_from_terrarium(
                    self.markets, terrarium_results,
                )
            else:
                resolved = resolve_markets(
                    self.markets, rnd, self.seed,
                )

            staked = sum(f.stake for f in forecasts)
            self.round_logs.append(RoundLog(
                round_num=rnd,
                active_markets=sum(
                    1 for m in self.markets if not m.resolved
                ),
                forecasts_submitted=len(forecasts),
                markets_resolved=len(resolved),
                total_karma_staked=round(staked, 2),
            ))

        settle_markets(self.markets, self.agents)
        return self._build_results()

    def _build_results(self) -> dict:
        leaderboard = build_leaderboard(self.agents)
        archetype_perf = build_archetype_performance(self.agents)
        calibration = build_calibration(self.markets)

        all_briers = [bs for a in self.agents for bs in a.brier_scores]
        resolved_markets = [m for m in self.markets if m.resolved]

        market_outcomes = []
        for m in resolved_markets:
            market_outcomes.append({
                "market_id": m.market_id,
                "category": m.category,
                "description": m.description,
                "base_probability": m.base_probability,
                "final_price": m.current_price(),
                "outcome": m.outcome,
                "n_forecasts": len(m.forecasts),
            })

        total_staked = sum(
            f.stake for m in self.markets for f in m.forecasts
        )

        return {
            "summary": {
                "total_agents": len(self.agents),
                "total_markets": len(self.markets),
                "resolved_markets": len(resolved_markets),
                "total_forecasts": sum(a.forecasts_made for a in self.agents),
                "total_karma_staked": round(total_staked, 2),
                "mean_brier_all": (
                    round(statistics.mean(all_briers), 4) if all_briers else None
                ),
                "seed": self.seed,
                "rounds": self.n_rounds,
                "pipeline_version": "v1.0-cross-sim",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "leaderboard": leaderboard,
            "archetype_performance": archetype_perf,
            "calibration_curve": calibration,
            "market_outcomes": market_outcomes,
            "round_log": [
                {
                    "round": r.round_num,
                    "active_markets": r.active_markets,
                    "forecasts": r.forecasts_submitted,
                    "resolved": r.markets_resolved,
                    "staked": r.total_karma_staked,
                }
                for r in self.round_logs
            ],
        }


# ── CLI ────────────────────────────────────────────────────────

def run_standalone(
    n_rounds: int = DEFAULT_ROUNDS,
    n_agents: int = DEFAULT_AGENTS,
    n_markets: int = DEFAULT_MARKETS,
    seed: int = 42,
    terrarium_results: dict | None = None,
    quiet: bool = False,
) -> dict:
    """Run and optionally print results."""
    sim = PredictionMarket(
        n_agents=n_agents,
        n_markets=n_markets,
        n_rounds=n_rounds,
        seed=seed,
    )
    results = sim.run(terrarium_results=terrarium_results)

    if not quiet:
        print()
        print("=" * 64)
        print("PREDICTION MARKET ENGINE — v1.0" + (
            " (cross-sim)" if terrarium_results else ""))
        print("=" * 64)
        s = results["summary"]
        print(f"\n  Markets:     {s['total_markets']} total, "
              f"{s['resolved_markets']} resolved")
        print(f"  Agents:      {s['total_agents']}")
        print(f"  Forecasts:   {s['total_forecasts']}")
        print(f"  Karma staked: {s['total_karma_staked']:.0f}")
        mb = s.get("mean_brier_all")
        print(f"  Mean Brier:  {mb:.4f}" if mb else "  Mean Brier:  N/A")

        if terrarium_results:
            print("\n  TERRARIUM-RESOLVED MARKETS")
            print("  " + "-" * 58)
            for m in results["market_outcomes"]:
                outcome = "YES" if m["outcome"] else "NO"
                print(f"    {m['market_id']:<20} → {outcome}  "
                      f"(base: {m['base_probability']:.2f}, "
                      f"final: {m['final_price']:.2f}, "
                      f"{m['n_forecasts']} forecasts)")

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

        if results["calibration_curve"]:
            print("\n  CALIBRATION CURVE (explicit confidence only)")
            print("  " + "-" * 58)
            for b in results["calibration_curve"]:
                bar = "█" * int(b["actual_rate"] * 20)
                print(f"    {b['bin']:<10} stated: {b['stated_avg']:.2f}  "
                      f"actual: {b['actual_rate']:.2f}  "
                      f"n={b['count']:<4} {bar}")

        print()

    return results


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Prediction Market Engine")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--markets", type=int, default=DEFAULT_MARKETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--terrarium", action="store_true",
                        help="Resolve markets from terrarium sim")
    parser.add_argument("--state-dir", type=str, default=None)
    args = parser.parse_args()

    terrarium_results = None
    if args.terrarium:
        state_dir = Path(args.state_dir) if args.state_dir else (
            Path(__file__).resolve().parent.parent / "state"
        )
        mars_path = state_dir / "mars.json"
        if mars_path.exists():
            with open(mars_path) as f:
                terrarium_results = json.load(f)
            if not args.quiet:
                print(f"Loaded terrarium state from {mars_path}")
        else:
            print(f"Warning: {mars_path} not found, using synthetic oracle")

    results = run_standalone(
        n_rounds=args.rounds,
        n_agents=args.agents,
        n_markets=args.markets,
        seed=args.seed,
        terrarium_results=terrarium_results,
        quiet=args.quiet,
    )

    if args.json:
        print(json.dumps(results, indent=2))

    if args.state_dir:
        state_dir = Path(args.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        out = state_dir / "market.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=2))
        tmp.rename(out)
        if not args.quiet:
            print(f"State saved: {out}")


if __name__ == "__main__":
    main()
