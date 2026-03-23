"""
market_maker.py — Prediction market engine for Mars colony outcomes.

Agents bet on colony simulation outcomes using a Logarithmic Market
Scoring Rule (LMSR). The market resolves by running the Mars Barn sim.
Zero external dependencies. Deterministic with seed control.

Five-stage pipe:
    SEED → AGENTS_BET → SIMULATE → RESOLVE → REPORT

Usage:
    python src/market_maker.py
    python src/market_maker.py --seeds 10 --sols 365 --agents 20
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- LMSR Parameters ---
LMSR_LIQUIDITY = 100.0  # b parameter — higher = more stable prices
STARTING_KARMA = 100
MIN_BET = 1
MAX_BET = 25

# --- Market definitions ---
COLONY_NAMES = ["Ares Prime", "Olympus Station", "Red Frontier"]
STRATEGIES = ["conservative", "balanced", "aggressive"]

MARKETS: list[dict[str, Any]] = [
    {
        "id": "highest_pop",
        "question": "Which colony has the highest population at sol 365?",
        "outcomes": COLONY_NAMES,
        "type": "categorical",
    },
    {
        "id": "any_death",
        "question": "Does any colony population drop below 10?",
        "outcomes": ["yes", "no"],
        "type": "binary",
    },
    {
        "id": "most_techs",
        "question": "Which colony unlocks the most technologies?",
        "outcomes": COLONY_NAMES,
        "type": "categorical",
    },
    {
        "id": "total_pop_over_400",
        "question": "Is total population across all colonies > 400 at sol 365?",
        "outcomes": ["yes", "no"],
        "type": "binary",
    },
    {
        "id": "first_epidemic",
        "question": "Which colony suffers the first epidemic?",
        "outcomes": COLONY_NAMES + ["none"],
        "type": "categorical",
    },
]

# --- Agent archetypes and betting strategies ---
ARCHETYPES = [
    "philosopher", "coder", "debater", "welcomer", "curator",
    "storyteller", "researcher", "contrarian", "archivist", "wildcard",
]

ARCHETYPE_BIASES: dict[str, dict[str, float]] = {
    "philosopher": {"conservative_lean": 0.3, "risk_aversion": 0.7, "contrarian": 0.1},
    "coder": {"conservative_lean": 0.1, "risk_aversion": 0.3, "contrarian": 0.2},
    "debater": {"conservative_lean": 0.0, "risk_aversion": 0.2, "contrarian": 0.6},
    "welcomer": {"conservative_lean": 0.5, "risk_aversion": 0.8, "contrarian": 0.0},
    "curator": {"conservative_lean": 0.4, "risk_aversion": 0.5, "contrarian": 0.1},
    "storyteller": {"conservative_lean": 0.2, "risk_aversion": 0.4, "contrarian": 0.3},
    "researcher": {"conservative_lean": 0.2, "risk_aversion": 0.4, "contrarian": 0.1},
    "contrarian": {"conservative_lean": 0.0, "risk_aversion": 0.1, "contrarian": 0.9},
    "archivist": {"conservative_lean": 0.6, "risk_aversion": 0.6, "contrarian": 0.05},
    "wildcard": {"conservative_lean": 0.0, "risk_aversion": 0.0, "contrarian": 0.5},
}


# --- LMSR Market Scoring ---

@dataclass
class LMSRMarket:
    """Logarithmic Market Scoring Rule for a single question.

    Hanson's LMSR: cost = b * ln(sum(exp(q_i/b)))
    Price of outcome i = exp(q_i/b) / sum(exp(q_j/b))
    """
    market_id: str
    question: str
    outcomes: list[str]
    liquidity: float = LMSR_LIQUIDITY
    quantities: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.quantities:
            self.quantities = [0.0] * len(self.outcomes)

    def _cost_function(self, q: list[float]) -> float:
        """LMSR cost: b * ln(sum(exp(q_i / b)))."""
        max_q = max(q)
        shifted = [qi - max_q for qi in q]
        return self.liquidity * (max_q / self.liquidity + math.log(
            sum(math.exp(s / self.liquidity) for s in shifted)
        ))

    def prices(self) -> list[float]:
        """Current price for each outcome (sums to 1.0)."""
        max_q = max(self.quantities)
        shifted = [qi - max_q for qi in self.quantities]
        exps = [math.exp(s / self.liquidity) for s in shifted]
        total = sum(exps)
        return [e / total for e in exps]

    def price_of(self, outcome: str) -> float:
        """Current price for a specific outcome."""
        idx = self.outcomes.index(outcome)
        return self.prices()[idx]

    def cost_to_buy(self, outcome: str, shares: float) -> float:
        """Cost to buy `shares` of `outcome`."""
        idx = self.outcomes.index(outcome)
        old_cost = self._cost_function(self.quantities)
        new_q = list(self.quantities)
        new_q[idx] += shares
        new_cost = self._cost_function(new_q)
        return new_cost - old_cost

    def buy(self, outcome: str, shares: float) -> float:
        """Execute a buy. Returns cost paid."""
        cost = self.cost_to_buy(outcome, shares)
        idx = self.outcomes.index(outcome)
        self.quantities[idx] += shares
        return cost

    def resolve(self, winning_outcome: str) -> None:
        """Mark the market as resolved with the winning outcome."""
        if winning_outcome not in self.outcomes:
            raise ValueError(f"Unknown outcome: {winning_outcome}")
        self.resolved_outcome = winning_outcome

    def snapshot(self) -> dict:
        """Serializable snapshot."""
        ps = self.prices()
        return {
            "market_id": self.market_id,
            "question": self.question,
            "outcomes": {
                self.outcomes[i]: {
                    "price": round(ps[i], 4),
                    "implied_prob": f"{ps[i]*100:.1f}%",
                    "quantity": round(self.quantities[i], 2),
                }
                for i in range(len(self.outcomes))
            },
            "resolved": getattr(self, "resolved_outcome", None),
        }


# --- Bet tracking ---

@dataclass
class Bet:
    """A single bet placed by an agent."""
    agent_id: str
    market_id: str
    outcome: str
    shares: float
    cost: float
    price_at_bet: float


@dataclass
class AgentAccount:
    """Agent's karma balance and bet history."""
    agent_id: str
    archetype: str
    karma: float = STARTING_KARMA
    bets: list[Bet] = field(default_factory=list)
    payout: float = 0.0


# --- Betting strategies ---

def _deterministic_seed(agent_id: str, market_id: str) -> int:
    """Reproducible seed from agent + market."""
    h = hashlib.sha256(f"{agent_id}:{market_id}".encode()).hexdigest()
    return int(h[:8], 16)


def pick_outcome_and_size(
    agent: AgentAccount,
    market: LMSRMarket,
    rng: random.Random,
) -> tuple[str, float] | None:
    """Choose which outcome to bet on and how much, based on archetype."""
    bias = ARCHETYPE_BIASES.get(agent.archetype, ARCHETYPE_BIASES["wildcard"])
    prices = market.prices()

    if agent.karma < MIN_BET:
        return None

    # Contrarians bet on the least-likely outcome
    if rng.random() < bias["contrarian"]:
        min_idx = prices.index(min(prices))
        outcome = market.outcomes[min_idx]
    else:
        # Colony preference based on archetype
        if market.market_id in ("highest_pop", "most_techs", "first_epidemic"):
            if rng.random() < bias["conservative_lean"] and "Ares Prime" in market.outcomes:
                outcome = "Ares Prime"
            elif rng.random() < 0.5 and "Red Frontier" in market.outcomes:
                outcome = "Red Frontier"
            else:
                outcome = rng.choice(market.outcomes)
        elif market.market_id in ("any_death", "total_pop_over_400"):
            if bias["risk_aversion"] > 0.5:
                outcome = "no" if market.market_id == "any_death" else "yes"
            else:
                outcome = rng.choice(market.outcomes)
        else:
            outcome = rng.choice(market.outcomes)

    # Bet size: risk-averse agents bet less
    max_bet = min(MAX_BET, agent.karma * 0.3)
    risk_factor = 1.0 - bias["risk_aversion"] * 0.5
    bet_size = max(MIN_BET, rng.uniform(MIN_BET, max_bet) * risk_factor)
    bet_size = min(bet_size, agent.karma)

    if bet_size < MIN_BET:
        return None

    return outcome, bet_size


# --- Resolution logic ---

def resolve_markets(
    markets: dict[str, LMSRMarket],
    sim_results: dict,
) -> dict[str, str]:
    """Determine winning outcomes from simulation results."""
    resolutions: dict[str, str] = {}
    colonies = sim_results["summary"]["colonies"]

    # highest_pop: which colony has the most people
    pops = {c["name"]: c["end_pop"] for c in colonies}
    winner = max(pops, key=pops.get)
    resolutions["highest_pop"] = winner

    # any_death: did any colony drop below 10?
    any_low = any(c["min_pop"] < 10 for c in colonies)
    resolutions["any_death"] = "yes" if any_low else "no"

    # most_techs: which colony unlocked the most techs?
    techs = {c["name"]: c.get("techs_unlocked", 0) for c in colonies}
    tech_winner = max(techs, key=techs.get)
    resolutions["most_techs"] = tech_winner

    # total_pop_over_400
    total_pop = sum(c["end_pop"] for c in colonies)
    resolutions["total_pop_over_400"] = "yes" if total_pop > 400 else "no"

    # first_epidemic: which colony had the earliest epidemic?
    first_sol = float("inf")
    first_colony = "none"
    for col_data in sim_results["colonies"]:
        for event in col_data.get("events", []):
            if event.get("type") == "epidemic_start":
                if event["sol"] < first_sol:
                    first_sol = event["sol"]
                    first_colony = col_data["name"]
                break
    resolutions["first_epidemic"] = first_colony

    return resolutions


def compute_brier_score(predicted_prob: float, outcome: bool) -> float:
    """Brier score: (forecast - actual)^2. Lower = better."""
    actual = 1.0 if outcome else 0.0
    return (predicted_prob - actual) ** 2


def compute_payouts(
    agents: dict[str, AgentAccount],
    markets: dict[str, LMSRMarket],
    resolutions: dict[str, str],
) -> dict[str, dict]:
    """Calculate payouts for all agents based on resolved markets."""
    agent_results: dict[str, dict] = {}

    for aid, account in agents.items():
        total_wagered = 0.0
        total_payout = 0.0
        bet_results: list[dict] = []

        for bet in account.bets:
            market = markets[bet.market_id]
            winning = resolutions.get(bet.market_id)
            won = bet.outcome == winning

            # Payout: shares pay 1.0 if correct, 0.0 if wrong
            payout = bet.shares if won else 0.0
            profit = payout - bet.cost
            brier = compute_brier_score(
                bet.price_at_bet,
                won,
            )

            total_wagered += bet.cost
            total_payout += payout

            bet_results.append({
                "market": bet.market_id,
                "outcome": bet.outcome,
                "shares": round(bet.shares, 2),
                "cost": round(bet.cost, 2),
                "won": won,
                "payout": round(payout, 2),
                "profit": round(profit, 2),
                "brier": round(brier, 4),
            })

        account.payout = total_payout
        net_profit = total_payout - total_wagered
        agent_results[aid] = {
            "agent_id": aid,
            "archetype": account.archetype,
            "starting_karma": STARTING_KARMA,
            "total_wagered": round(total_wagered, 2),
            "total_payout": round(total_payout, 2),
            "net_profit": round(net_profit, 2),
            "final_karma": round(account.karma + net_profit, 2),
            "bets": bet_results,
            "avg_brier": round(
                sum(b["brier"] for b in bet_results) / max(1, len(bet_results)), 4
            ),
        }

    return agent_results


# --- Main pipeline ---

def generate_agents(n_agents: int, seed: int = 42) -> dict[str, AgentAccount]:
    """Create synthetic agent accounts with varied archetypes."""
    rng = random.Random(seed)
    agents: dict[str, AgentAccount] = {}
    for i in range(n_agents):
        archetype = ARCHETYPES[i % len(ARCHETYPES)]
        agent_id = f"zion-{archetype}-{i+1:02d}"
        agents[agent_id] = AgentAccount(agent_id=agent_id, archetype=archetype)
    return agents


def run_prediction_market(
    n_agents: int = 20,
    sim_sols: int = 365,
    sim_seed: int = 42,
    agent_seed: int = 42,
) -> dict:
    """Full pipeline: create markets → agents bet → sim runs → resolve → report.

    Returns a complete results dict with market data, agent payouts,
    simulation summary, and calibration analysis.
    """
    # Stage 1: Create markets
    markets: dict[str, LMSRMarket] = {}
    for mdef in MARKETS:
        markets[mdef["id"]] = LMSRMarket(
            market_id=mdef["id"],
            question=mdef["question"],
            outcomes=list(mdef["outcomes"]),
        )

    # Stage 2: Agents place bets
    agents = generate_agents(n_agents, seed=agent_seed)
    all_bets: list[dict] = []

    for aid, account in agents.items():
        for mid, market in markets.items():
            rng = random.Random(_deterministic_seed(aid, mid))
            result = pick_outcome_and_size(account, market, rng)
            if result is None:
                continue
            outcome, shares = result
            cost = market.cost_to_buy(outcome, shares)
            if cost > account.karma:
                shares = shares * (account.karma / cost) * 0.9
                cost = market.cost_to_buy(outcome, shares)
            if cost > account.karma or cost < 0.01:
                continue
            price_before = market.price_of(outcome)
            actual_cost = market.buy(outcome, shares)
            account.karma -= actual_cost
            bet = Bet(aid, mid, outcome, shares, actual_cost, price_before)
            account.bets.append(bet)
            all_bets.append({
                "agent": aid,
                "market": mid,
                "outcome": outcome,
                "shares": round(shares, 2),
                "cost": round(actual_cost, 2),
                "price_moved": f"{price_before:.3f} → {market.price_of(outcome):.3f}",
            })

    # Capture pre-resolution market state
    pre_resolution = {mid: m.snapshot() for mid, m in markets.items()}

    # Stage 3: Run the simulation
    from src.tick_engine import Simulation
    sim = Simulation(sols=sim_sols, env_seed=sim_seed)
    sim_results = sim.run()

    # Stage 4: Resolve markets
    resolutions = resolve_markets(markets, sim_results)
    for mid, winning in resolutions.items():
        markets[mid].resolve(winning)

    # Stage 5: Compute payouts and report
    agent_results = compute_payouts(agents, markets, resolutions)

    # Calibration analysis: for each market, compare pre-bet implied prob
    # of winning outcome vs actual resolution
    calibration: list[dict] = []
    for mid, market in markets.items():
        pre = pre_resolution[mid]
        winning = resolutions[mid]
        # Get final implied probability
        final_prices = market.prices()
        winning_idx = market.outcomes.index(winning)
        calibration.append({
            "market": mid,
            "question": market.question,
            "winning_outcome": winning,
            "pre_bet_prob": round(1.0 / len(market.outcomes), 4),
            "final_implied_prob": round(final_prices[winning_idx], 4),
            "total_shares_on_winner": round(market.quantities[winning_idx], 2),
        })

    # Leaderboard
    leaderboard = sorted(
        agent_results.values(),
        key=lambda x: -x["net_profit"],
    )

    # Archetype performance
    archetype_perf: dict[str, list[float]] = {}
    for ar in agent_results.values():
        arch = ar["archetype"]
        archetype_perf.setdefault(arch, []).append(ar["net_profit"])
    archetype_summary = {
        arch: {
            "mean_profit": round(sum(vs) / len(vs), 2),
            "best": round(max(vs), 2),
            "worst": round(min(vs), 2),
            "count": len(vs),
        }
        for arch, vs in archetype_perf.items()
    }

    # Simulation summary for output
    colony_summary = []
    for c in sim_results["summary"]["colonies"]:
        colony_summary.append({
            "name": c["name"],
            "strategy": c["strategy"],
            "start_pop": c["start_pop"],
            "end_pop": c["end_pop"],
            "growth_pct": c["growth_pct"],
            "techs_unlocked": c.get("techs_unlocked", 0),
        })

    return {
        "_meta": {
            "engine": "market-maker",
            "version": "1.0.0",
            "generated": datetime.now(timezone.utc).isoformat(),
            "n_agents": n_agents,
            "sim_sols": sim_sols,
            "sim_seed": sim_seed,
            "total_bets": len(all_bets),
            "total_markets": len(markets),
        },
        "markets": {mid: m.snapshot() for mid, m in markets.items()},
        "resolutions": resolutions,
        "calibration": calibration,
        "bets": all_bets[-50:],
        "agent_results": {
            aid: {k: v for k, v in ar.items() if k != "bets"}
            for aid, ar in agent_results.items()
        },
        "leaderboard": [
            {
                "rank": i + 1,
                "agent": lb["agent_id"],
                "archetype": lb["archetype"],
                "net_profit": lb["net_profit"],
                "avg_brier": lb["avg_brier"],
            }
            for i, lb in enumerate(leaderboard[:20])
        ],
        "archetype_summary": archetype_summary,
        "colony_results": colony_summary,
    }


def format_report(results: dict) -> str:
    """Format results as human-readable stdout report."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("  PREDICTION MARKET — Mars Colony Outcomes")
    lines.append("=" * 64)
    meta = results["_meta"]
    lines.append(f"  Agents: {meta['n_agents']}  |  Markets: {meta['total_markets']}"
                 f"  |  Bets: {meta['total_bets']}  |  Sols: {meta['sim_sols']}")
    lines.append("")

    # Colony results
    lines.append("  ── SIMULATION RESULTS ──")
    for c in results["colony_results"]:
        lines.append(f"    {c['name']:20s} ({c['strategy']:12s})"
                     f"  pop: {c['start_pop']} → {c['end_pop']} ({c['growth_pct']:+.1f}%)"
                     f"  techs: {c['techs_unlocked']}")
    lines.append("")

    # Market resolutions
    lines.append("  ── MARKET RESOLUTIONS ──")
    for cal in results["calibration"]:
        lines.append(f"    Q: {cal['question']}")
        lines.append(f"    A: {cal['winning_outcome']}"
                     f"  (implied prob: {cal['final_implied_prob']:.1%},"
                     f" shares: {cal['total_shares_on_winner']})")
        lines.append("")

    # Leaderboard
    lines.append("  ── LEADERBOARD (Top 10) ──")
    lines.append(f"    {'Rank':>4s}  {'Agent':25s}  {'Archetype':12s}"
                 f"  {'Profit':>8s}  {'Brier':>6s}")
    lines.append("    " + "-" * 65)
    for lb in results["leaderboard"][:10]:
        lines.append(f"    {lb['rank']:4d}  {lb['agent']:25s}  {lb['archetype']:12s}"
                     f"  {lb['net_profit']:+8.2f}  {lb['avg_brier']:6.4f}")
    lines.append("")

    # Archetype summary
    lines.append("  ── ARCHETYPE PERFORMANCE ──")
    sorted_archs = sorted(
        results["archetype_summary"].items(),
        key=lambda x: -x[1]["mean_profit"],
    )
    for arch, stats in sorted_archs:
        lines.append(f"    {arch:15s}  mean: {stats['mean_profit']:+7.2f}"
                     f"  best: {stats['best']:+7.2f}"
                     f"  worst: {stats['worst']:+7.2f}"
                     f"  (n={stats['count']})")
    lines.append("")
    lines.append("=" * 64)

    return "\n".join(lines)


def main() -> None:
    """Run the prediction market and print results."""
    results = run_prediction_market(n_agents=20, sim_sols=365, sim_seed=42)
    report = format_report(results)
    print(report)

    # Save state
    out_dir = Path(__file__).resolve().parent.parent / "state"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "market.json"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.rename(out_path)
    print(f"\nState saved: {out_path}")


if __name__ == "__main__":
    main()
