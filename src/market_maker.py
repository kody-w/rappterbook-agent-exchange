"""
market_maker.py — Prediction market engine for Mars Barn outcomes.

Agents bet on colony outcomes using an LMSR (Logarithmic Market Scoring Rule)
automated market maker. Markets resolve against actual simulation results.

Binary markets (YES/NO shares):
  - Population milestones: "Will Ares Prime exceed 200 by sol 365?"
  - Tech races: "Will Red Frontier unlock fusion first?"
  - Survival: "Will all colonies survive?"
  - Events: "Will a global dust storm last > 40 sols?"
  - Migration: "Will total migration exceed 50?"

The LMSR guarantees liquidity — every market has a price at all times.
Cost function: C(q) = b * ln(exp(q_yes/b) + exp(q_no/b))

Agents place bets based on their archetype's risk profile:
  - Philosophers: bet on long-term outcomes (survival, terraforming)
  - Coders: bet on tech unlocks
  - Contrarians: bet against consensus
  - Wildcards: random bets

After simulation runs, markets resolve and P&L is computed.

Usage:
    from src.market_maker import create_default_markets, PredictionEngine
    engine = PredictionEngine(markets, agents, seed=42)
    engine.run_betting_rounds(n_rounds=20)
    engine.resolve_all(sim_results)
    print(engine.leaderboard())
"""
from __future__ import annotations

import math
import random
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# LMSR Market Maker
# ---------------------------------------------------------------------------

LMSR_LIQUIDITY_B = 50.0  # higher = more liquid, less price movement per trade


@dataclass
class Market:
    """A binary prediction market (YES/NO)."""
    market_id: str
    question: str
    category: str  # "population", "tech", "survival", "event", "migration"
    resolve_fn_name: str  # name of resolver function
    resolve_args: dict  # args passed to resolver

    # LMSR state
    q_yes: float = 0.0
    q_no: float = 0.0
    b: float = LMSR_LIQUIDITY_B

    # Tracking
    total_volume: float = 0.0
    num_trades: int = 0
    resolved: bool = False
    outcome: bool | None = None
    history: list[dict] = field(default_factory=list)

    def price_yes(self) -> float:
        """Current implied probability of YES."""
        exp_yes = math.exp(self.q_yes / self.b)
        exp_no = math.exp(self.q_no / self.b)
        return exp_yes / (exp_yes + exp_no)

    def price_no(self) -> float:
        """Current implied probability of NO."""
        return 1.0 - self.price_yes()

    def cost(self) -> float:
        """Current LMSR cost function value."""
        return self.b * math.log(
            math.exp(self.q_yes / self.b) + math.exp(self.q_no / self.b)
        )

    def buy_yes(self, shares: float) -> float:
        """Buy YES shares. Returns cost (positive = spend)."""
        old_cost = self.cost()
        self.q_yes += shares
        new_cost = self.cost()
        cost = new_cost - old_cost
        self.total_volume += abs(cost)
        self.num_trades += 1
        return cost

    def buy_no(self, shares: float) -> float:
        """Buy NO shares. Returns cost (positive = spend)."""
        old_cost = self.cost()
        self.q_no += shares
        new_cost = self.cost()
        cost = new_cost - old_cost
        self.total_volume += abs(cost)
        self.num_trades += 1
        return cost

    def record_snapshot(self, round_num: int) -> None:
        """Record price snapshot for history."""
        self.history.append({
            "round": round_num,
            "price_yes": round(self.price_yes(), 4),
            "price_no": round(self.price_no(), 4),
            "volume": round(self.total_volume, 2),
        })

    def resolve(self, outcome: bool) -> None:
        """Resolve the market. YES shares pay 1.0 if True, 0 otherwise."""
        self.resolved = True
        self.outcome = outcome

    def snapshot(self) -> dict:
        """Serializable snapshot."""
        return {
            "market_id": self.market_id,
            "question": self.question,
            "category": self.category,
            "price_yes": round(self.price_yes(), 4),
            "price_no": round(self.price_no(), 4),
            "total_volume": round(self.total_volume, 2),
            "num_trades": self.num_trades,
            "resolved": self.resolved,
            "outcome": self.outcome,
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# Market resolution functions
# ---------------------------------------------------------------------------

def resolve_population_above(results: dict, colony_name: str, threshold: int) -> bool:
    """Did colony_name end with population > threshold?"""
    for colony in results.get("colonies", []):
        if colony["name"] == colony_name:
            return colony["final_population"] > threshold
    return False


def resolve_all_survive(results: dict) -> bool:
    """Did all colonies end with population > 0?"""
    return all(
        c["final_population"] > 0 for c in results.get("colonies", [])
    )


def resolve_tech_first(results: dict, tech_name: str) -> str | None:
    """Which colony unlocked tech_name first? Returns colony name or None."""
    earliest_sol = float("inf")
    winner = None
    for colony in results.get("colonies", []):
        tech_data = colony.get("tech", {})
        for unlock in tech_data.get("unlocked", []):
            if unlock["name"] == tech_name and unlock["sol"] < earliest_sol:
                earliest_sol = unlock["sol"]
                winner = colony["name"]
    return winner


def resolve_tech_first_colony(results: dict, tech_name: str, colony_name: str) -> bool:
    """Did colony_name unlock tech_name first?"""
    winner = resolve_tech_first(results, tech_name)
    return winner == colony_name


def resolve_total_migration_above(results: dict, threshold: int) -> bool:
    """Did total migration exceed threshold?"""
    total = results.get("summary", {}).get("total_migrations", 0)
    return total > threshold


def resolve_epidemic_count_above(results: dict, threshold: int) -> bool:
    """Were there more than threshold epidemics total?"""
    count = 0
    for colony in results.get("colonies", []):
        for event in colony.get("events", []):
            if event.get("type") == "epidemic_start":
                count += 1
    return count > threshold


def resolve_global_storm_duration(results: dict, min_sols: int) -> bool:
    """Did any global dust storm last at least min_sols?"""
    env_history = results.get("environment", {}).get("history", [])
    streak = 0
    for snap in env_history:
        if snap.get("storm") == "global":
            streak += 1
            if streak >= min_sols:
                return True
        else:
            streak = 0
    return False


def resolve_growth_pct_above(results: dict, colony_name: str, threshold: float) -> bool:
    """Did colony_name achieve growth > threshold%?"""
    for s in results.get("summary", {}).get("colonies", []):
        if s["name"] == colony_name:
            return s["growth_pct"] > threshold
    return False


RESOLVERS = {
    "population_above": resolve_population_above,
    "all_survive": resolve_all_survive,
    "tech_first_colony": resolve_tech_first_colony,
    "total_migration_above": resolve_total_migration_above,
    "epidemic_count_above": resolve_epidemic_count_above,
    "global_storm_duration": resolve_global_storm_duration,
    "growth_pct_above": resolve_growth_pct_above,
}


# ---------------------------------------------------------------------------
# Default market set
# ---------------------------------------------------------------------------

def create_default_markets() -> list[Market]:
    """Create the standard set of 12 prediction markets for Mars Barn."""
    return [
        Market(
            market_id="ares-200",
            question="Will Ares Prime exceed 200 population by sol 365?",
            category="population",
            resolve_fn_name="population_above",
            resolve_args={"colony_name": "Ares Prime", "threshold": 200},
        ),
        Market(
            market_id="olympus-150",
            question="Will Olympus Station exceed 150 population?",
            category="population",
            resolve_fn_name="population_above",
            resolve_args={"colony_name": "Olympus Station", "threshold": 150},
        ),
        Market(
            market_id="frontier-100",
            question="Will Red Frontier exceed 100 population?",
            category="population",
            resolve_fn_name="population_above",
            resolve_args={"colony_name": "Red Frontier", "threshold": 100},
        ),
        Market(
            market_id="all-survive",
            question="Will all 3 colonies survive to sol 365?",
            category="survival",
            resolve_fn_name="all_survive",
            resolve_args={},
        ),
        Market(
            market_id="frontier-fusion",
            question="Will Red Frontier unlock Compact Fusion Reactor first?",
            category="tech",
            resolve_fn_name="tech_first_colony",
            resolve_args={"tech_name": "Compact Fusion Reactor", "colony_name": "Red Frontier"},
        ),
        Market(
            market_id="ares-solar",
            question="Will Ares Prime unlock Advanced Solar Cells first?",
            category="tech",
            resolve_fn_name="tech_first_colony",
            resolve_args={"tech_name": "Advanced Solar Cells", "colony_name": "Ares Prime"},
        ),
        Market(
            market_id="migration-50",
            question="Will total inter-colony migration exceed 50?",
            category="migration",
            resolve_fn_name="total_migration_above",
            resolve_args={"threshold": 50},
        ),
        Market(
            market_id="epidemics-5",
            question="Will there be more than 5 epidemics total?",
            category="event",
            resolve_fn_name="epidemic_count_above",
            resolve_args={"threshold": 5},
        ),
        Market(
            market_id="global-storm-40",
            question="Will a global dust storm last 40+ sols?",
            category="event",
            resolve_fn_name="global_storm_duration",
            resolve_args={"min_sols": 40},
        ),
        Market(
            market_id="frontier-growth-100",
            question="Will Red Frontier achieve >100% growth?",
            category="population",
            resolve_fn_name="growth_pct_above",
            resolve_args={"colony_name": "Red Frontier", "threshold": 100.0},
        ),
        Market(
            market_id="ares-growth-50",
            question="Will Ares Prime achieve >50% growth?",
            category="population",
            resolve_fn_name="growth_pct_above",
            resolve_args={"colony_name": "Ares Prime", "threshold": 50.0},
        ),
        Market(
            market_id="migration-20",
            question="Will total migration exceed 20?",
            category="migration",
            resolve_fn_name="total_migration_above",
            resolve_args={"threshold": 20},
        ),
    ]


# ---------------------------------------------------------------------------
# Agent betting strategies
# ---------------------------------------------------------------------------

def _agent_seed(agent_id: str, round_num: int) -> int:
    """Deterministic seed from agent + round."""
    h = hashlib.sha256(f"market:{agent_id}:{round_num}".encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class BettorState:
    """Per-agent betting state."""
    agent_id: str
    bankroll: float = 100.0
    positions: dict[str, float] = field(default_factory=dict)  # market_id -> net YES shares
    total_spent: float = 0.0
    total_pnl: float = 0.0

    def net_position(self, market_id: str) -> float:
        return self.positions.get(market_id, 0.0)


def _philosopher_bets(
    bettor: BettorState, markets: list[Market], rng: random.Random
) -> list[tuple[str, str, float]]:
    """Philosophers bet on survival and long-term outcomes."""
    bets = []
    for m in markets:
        if m.resolved:
            continue
        if m.category in ("survival", "population"):
            side = "yes" if rng.random() < 0.65 else "no"
            size = rng.uniform(1.0, 4.0)
            bets.append((m.market_id, side, size))
        elif rng.random() < 0.2:
            side = rng.choice(["yes", "no"])
            bets.append((m.market_id, side, rng.uniform(0.5, 2.0)))
    return bets[:4]


def _coder_bets(
    bettor: BettorState, markets: list[Market], rng: random.Random
) -> list[tuple[str, str, float]]:
    """Coders bet heavily on tech race markets."""
    bets = []
    for m in markets:
        if m.resolved:
            continue
        if m.category == "tech":
            side = "yes" if rng.random() < 0.7 else "no"
            size = rng.uniform(2.0, 6.0)
            bets.append((m.market_id, side, size))
        elif rng.random() < 0.15:
            bets.append((m.market_id, "yes", rng.uniform(0.5, 2.0)))
    return bets[:4]


def _contrarian_bets(
    bettor: BettorState, markets: list[Market], rng: random.Random
) -> list[tuple[str, str, float]]:
    """Contrarians bet AGAINST the current consensus."""
    bets = []
    for m in markets:
        if m.resolved:
            continue
        p_yes = m.price_yes()
        if p_yes > 0.65:
            bets.append((m.market_id, "no", rng.uniform(2.0, 5.0)))
        elif p_yes < 0.35:
            bets.append((m.market_id, "yes", rng.uniform(2.0, 5.0)))
    return bets[:4]


def _wildcard_bets(
    bettor: BettorState, markets: list[Market], rng: random.Random
) -> list[tuple[str, str, float]]:
    """Wildcards bet randomly."""
    bets = []
    available = [m for m in markets if not m.resolved]
    rng.shuffle(available)
    for m in available[:3]:
        side = rng.choice(["yes", "no"])
        size = rng.uniform(1.0, 5.0)
        bets.append((m.market_id, side, size))
    return bets


def _balanced_bets(
    bettor: BettorState, markets: list[Market], rng: random.Random
) -> list[tuple[str, str, float]]:
    """Default strategy — moderate bets across categories."""
    bets = []
    for m in markets:
        if m.resolved:
            continue
        if rng.random() < 0.3:
            side = "yes" if rng.random() < 0.55 else "no"
            size = rng.uniform(1.0, 3.0)
            bets.append((m.market_id, side, size))
    return bets[:3]


BETTING_STRATEGIES = {
    "philosopher": _philosopher_bets,
    "coder": _coder_bets,
    "contrarian": _contrarian_bets,
    "wildcard": _wildcard_bets,
    "researcher": _balanced_bets,
    "storyteller": _balanced_bets,
    "welcomer": _balanced_bets,
    "debater": _contrarian_bets,
    "curator": _philosopher_bets,
    "archivist": _coder_bets,
}


# ---------------------------------------------------------------------------
# Prediction Engine
# ---------------------------------------------------------------------------

class PredictionEngine:
    """Runs the prediction market simulation.

    1. Create markets
    2. Agents place bets over N rounds (prices move via LMSR)
    3. Simulation runs
    4. Markets resolve
    5. P&L computed
    """

    def __init__(
        self,
        markets: list[Market],
        agent_ids: list[str],
        seed: int = 42,
    ) -> None:
        self.markets = {m.market_id: m for m in markets}
        self.bettors: dict[str, BettorState] = {
            aid: BettorState(agent_id=aid) for aid in agent_ids
        }
        self.seed = seed
        self.trade_log: list[dict] = []

    def _extract_archetype(self, agent_id: str) -> str:
        parts = agent_id.split("-")
        if len(parts) >= 2 and parts[0] == "zion":
            return parts[1]
        return "wildcard"

    def run_betting_rounds(self, n_rounds: int = 20) -> None:
        """Run n_rounds of betting. Each round, every agent places bets."""
        market_list = list(self.markets.values())

        for round_num in range(1, n_rounds + 1):
            for agent_id, bettor in self.bettors.items():
                archetype = self._extract_archetype(agent_id)
                strategy = BETTING_STRATEGIES.get(archetype, _balanced_bets)
                rng = random.Random(_agent_seed(agent_id, round_num))

                bets = strategy(bettor, market_list, rng)
                for market_id, side, shares in bets:
                    market = self.markets.get(market_id)
                    if market is None or market.resolved:
                        continue
                    if shares <= 0:
                        continue

                    # Check bankroll
                    max_cost = shares * 1.5  # conservative estimate
                    if bettor.bankroll < max_cost:
                        shares = bettor.bankroll / 1.5
                        if shares < 0.1:
                            continue

                    if side == "yes":
                        cost = market.buy_yes(shares)
                        bettor.positions[market_id] = (
                            bettor.positions.get(market_id, 0.0) + shares
                        )
                    else:
                        cost = market.buy_no(shares)
                        bettor.positions[market_id] = (
                            bettor.positions.get(market_id, 0.0) - shares
                        )

                    bettor.bankroll -= cost
                    bettor.total_spent += abs(cost)

                    self.trade_log.append({
                        "round": round_num,
                        "agent": agent_id,
                        "market": market_id,
                        "side": side,
                        "shares": round(shares, 2),
                        "cost": round(cost, 2),
                        "new_price_yes": round(market.price_yes(), 4),
                    })

            # Record price snapshots
            for m in market_list:
                m.record_snapshot(round_num)

    def resolve_all(self, sim_results: dict) -> dict[str, bool]:
        """Resolve all markets against simulation results."""
        outcomes = {}
        for market_id, market in self.markets.items():
            resolver = RESOLVERS.get(market.resolve_fn_name)
            if resolver is None:
                continue
            outcome = resolver(sim_results, **market.resolve_args)
            market.resolve(outcome)
            outcomes[market_id] = outcome

        # Compute P&L for each bettor
        for bettor in self.bettors.values():
            pnl = 0.0
            for market_id, net_yes in bettor.positions.items():
                market = self.markets.get(market_id)
                if market is None or not market.resolved:
                    continue
                if market.outcome:
                    # YES wins: YES shares pay 1.0 each
                    pnl += net_yes * 1.0
                else:
                    # NO wins: NO shares (negative YES) pay 1.0 each
                    pnl -= net_yes * 1.0
            bettor.total_pnl = pnl

        return outcomes

    def leaderboard(self) -> list[dict]:
        """Sorted leaderboard by P&L."""
        board = []
        for bettor in self.bettors.values():
            board.append({
                "agent_id": bettor.agent_id,
                "archetype": self._extract_archetype(bettor.agent_id),
                "bankroll": round(bettor.bankroll, 2),
                "total_spent": round(bettor.total_spent, 2),
                "pnl": round(bettor.total_pnl, 2),
                "roi": round(
                    (bettor.total_pnl / max(bettor.total_spent, 0.01)) * 100, 1
                ),
                "num_positions": sum(
                    1 for v in bettor.positions.values() if abs(v) > 0.01
                ),
            })
        board.sort(key=lambda x: -x["pnl"])
        return board

    def market_summary(self) -> list[dict]:
        """Summary of all markets."""
        return [m.snapshot() for m in self.markets.values()]

    def full_results(self) -> dict:
        """Complete prediction market results."""
        return {
            "_meta": {
                "engine": "prediction-market",
                "version": "1.0.0",
                "generated": datetime.now(timezone.utc).isoformat(),
                "num_markets": len(self.markets),
                "num_bettors": len(self.bettors),
                "num_trades": len(self.trade_log),
            },
            "markets": self.market_summary(),
            "leaderboard": self.leaderboard()[:20],
            "recent_trades": self.trade_log[-50:],
            "archetype_performance": self._archetype_performance(),
        }

    def _archetype_performance(self) -> dict[str, dict]:
        """Aggregate performance by archetype."""
        arch_data: dict[str, list[float]] = {}
        for bettor in self.bettors.values():
            arch = self._extract_archetype(bettor.agent_id)
            arch_data.setdefault(arch, []).append(bettor.total_pnl)

        result = {}
        for arch, pnls in arch_data.items():
            result[arch] = {
                "mean_pnl": round(sum(pnls) / len(pnls), 2),
                "best_pnl": round(max(pnls), 2),
                "worst_pnl": round(min(pnls), 2),
                "count": len(pnls),
            }
        return result
