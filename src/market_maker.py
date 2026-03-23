"""
market_maker.py — Prediction market engine for Mars colony outcomes.

Binary outcome markets using LMSR (Logarithmic Market Scoring Rule).
Agents bet on colony survival, population thresholds, epidemics, and
tech races. Markets resolve by running the actual Mars Barn terrarium.

LMSR math (Hanson 2003, same as Polymarket/Metaculus):
  Cost: C(q) = b * ln(exp(q_yes/b) + exp(q_no/b))
  Price: p_yes = exp(q_yes/b) / (exp(q_yes/b) + exp(q_no/b))
  => p_yes + p_no = 1.0 always

Python stdlib only.

Usage:
    python src/market_maker.py                     # run with defaults
    python src/market_maker.py --agents 30         # synthetic agents
    python src/market_maker.py --rounds 100        # more trading rounds
    python src/market_maker.py --resolve           # resolve via sim
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# LMSR liquidity parameter — higher = more liquid, less price impact per trade
LMSR_B = 100.0

# Trading config
DEFAULT_ROUNDS = 50
DEFAULT_AGENTS = 20
AGENT_STARTING_CASH = 500.0
MAX_TRADE_SIZE = 15


# ---------------------------------------------------------------------------
# LMSR Automated Market Maker
# ---------------------------------------------------------------------------

def lmsr_cost(q_yes: float, q_no: float, b: float) -> float:
    """LMSR cost function C(q) = b * ln(exp(q_yes/b) + exp(q_no/b)).

    Uses log-sum-exp trick for numerical stability.
    """
    m = max(q_yes / b, q_no / b)
    return b * (m + math.log(math.exp(q_yes / b - m) + math.exp(q_no / b - m)))


def lmsr_price_yes(q_yes: float, q_no: float, b: float) -> float:
    """Price of YES share: exp(q_yes/b) / (exp(q_yes/b) + exp(q_no/b))."""
    diff = (q_yes - q_no) / b
    if diff > 500:
        return 1.0
    if diff < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-diff))


def lmsr_trade_cost(
    q_yes: float, q_no: float, b: float,
    outcome: str, shares: float,
) -> float:
    """Cost to buy `shares` of `outcome` ("yes" or "no").

    Returns the dollar cost (positive = agent pays, negative = agent receives).
    """
    old_cost = lmsr_cost(q_yes, q_no, b)
    if outcome == "yes":
        new_cost = lmsr_cost(q_yes + shares, q_no, b)
    else:
        new_cost = lmsr_cost(q_yes, q_no + shares, b)
    return new_cost - old_cost


# ---------------------------------------------------------------------------
# Market definition
# ---------------------------------------------------------------------------

@dataclass
class MarketCondition:
    """Resolution condition for a binary market."""
    kind: str           # "survival", "pop_threshold", "epidemic", "tech_race", "growth"
    colony: str | None  # colony name or None for cross-colony conditions
    threshold: float    # population threshold, growth %, etc.
    comparator: str     # ">=", "<=", ">", "<", "=="
    sol: int            # sol at which to evaluate (0 = end of sim)

    def evaluate(self, results: dict) -> bool:
        """Check condition against simulation results."""
        colonies = {c["name"]: c for c in results["colonies"]}
        summary = {c["name"]: c for c in results["summary"]["colonies"]}

        if self.kind == "survival":
            col = colonies.get(self.colony, {})
            final_pop = col.get("final_population", 0)
            return final_pop > 0

        if self.kind == "pop_threshold":
            col = colonies.get(self.colony, {})
            history = col.get("history", [])
            sol_idx = min(self.sol - 1, len(history) - 1) if self.sol > 0 else len(history) - 1
            pop = history[sol_idx]["population"] if sol_idx >= 0 and history else 0
            return self._compare(pop)

        if self.kind == "growth":
            col_summary = summary.get(self.colony, {})
            growth = col_summary.get("growth_pct", 0)
            return self._compare(growth)

        if self.kind == "epidemic":
            for col in results["colonies"]:
                events = col.get("events", [])
                epidemic_deaths = sum(
                    1 for e in events if e.get("type") == "epidemic_start"
                )
                if epidemic_deaths > 0:
                    return True
            return False

        if self.kind == "tech_race":
            col = colonies.get(self.colony, {})
            tech = col.get("tech", {})
            unlocked = len(tech.get("unlocked", []))
            return self._compare(unlocked)

        return False

    def _compare(self, value: float) -> bool:
        """Apply comparator to value vs threshold."""
        if self.comparator == ">=":
            return value >= self.threshold
        if self.comparator == "<=":
            return value <= self.threshold
        if self.comparator == ">":
            return value > self.threshold
        if self.comparator == "<":
            return value < self.threshold
        if self.comparator == "==":
            return abs(value - self.threshold) < 0.001
        return False

    def describe(self) -> str:
        """Human-readable market question."""
        if self.kind == "survival":
            return f"Will {self.colony} survive {self.sol} sols?"
        if self.kind == "pop_threshold":
            return f"Will {self.colony} reach {self.threshold:.0f}+ population by sol {self.sol}?"
        if self.kind == "growth":
            return f"Will {self.colony} grow {self.comparator} {self.threshold:.0f}%?"
        if self.kind == "epidemic":
            return "Will at least one epidemic occur?"
        if self.kind == "tech_race":
            return f"Will {self.colony} unlock {self.comparator} {self.threshold:.0f} techs?"
        return f"[{self.kind}] {self.colony} {self.comparator} {self.threshold}"


@dataclass
class Market:
    """A single binary prediction market."""
    market_id: str
    question: str
    condition: MarketCondition
    b: float = LMSR_B
    q_yes: float = 0.0
    q_no: float = 0.0
    total_volume: float = 0.0
    trade_count: int = 0
    resolved: bool = False
    outcome: bool | None = None
    price_history: list[float] = field(default_factory=list)

    @property
    def price_yes(self) -> float:
        return lmsr_price_yes(self.q_yes, self.q_no, self.b)

    @property
    def price_no(self) -> float:
        return 1.0 - self.price_yes

    def buy(self, outcome: str, shares: float) -> float:
        """Buy shares. Returns cost (positive = paid)."""
        cost = lmsr_trade_cost(self.q_yes, self.q_no, self.b, outcome, shares)
        if outcome == "yes":
            self.q_yes += shares
        else:
            self.q_no += shares
        self.total_volume += abs(cost)
        self.trade_count += 1
        self.price_history.append(round(self.price_yes, 4))
        return cost

    def sell(self, outcome: str, shares: float) -> float:
        """Sell shares. Returns proceeds (negative cost = received)."""
        return self.buy(outcome, -shares)

    def resolve(self, results: dict) -> bool:
        """Resolve market against simulation results. Returns outcome."""
        self.outcome = self.condition.evaluate(results)
        self.resolved = True
        return self.outcome

    def snapshot(self) -> dict:
        """Serializable snapshot."""
        return {
            "market_id": self.market_id,
            "question": self.question,
            "price_yes": round(self.price_yes, 4),
            "price_no": round(self.price_no, 4),
            "volume": round(self.total_volume, 2),
            "trades": self.trade_count,
            "resolved": self.resolved,
            "outcome": self.outcome,
            "q_yes": round(self.q_yes, 2),
            "q_no": round(self.q_no, 2),
            "price_history": self.price_history[-200:],
        }


# ---------------------------------------------------------------------------
# Market catalog — the questions agents bet on
# ---------------------------------------------------------------------------

def create_default_markets(sols: int = 365) -> list[Market]:
    """Create the default set of prediction markets for a Mars Barn run."""
    markets = [
        Market(
            market_id="ares-survive",
            question="Will Ares Prime survive 365 sols?",
            condition=MarketCondition("survival", "Ares Prime", 0, ">", sols),
        ),
        Market(
            market_id="olympus-survive",
            question="Will Olympus Station survive 365 sols?",
            condition=MarketCondition("survival", "Olympus Station", 0, ">", sols),
        ),
        Market(
            market_id="frontier-survive",
            question="Will Red Frontier survive 365 sols?",
            condition=MarketCondition("survival", "Red Frontier", 0, ">", sols),
        ),
        Market(
            market_id="ares-200",
            question="Will Ares Prime reach 200+ population?",
            condition=MarketCondition("pop_threshold", "Ares Prime", 200, ">=", sols),
        ),
        Market(
            market_id="frontier-150",
            question="Will Red Frontier reach 150+ population?",
            condition=MarketCondition("pop_threshold", "Red Frontier", 150, ">=", sols),
        ),
        Market(
            market_id="frontier-outgrows",
            question="Will Red Frontier grow >= 100%?",
            condition=MarketCondition("growth", "Red Frontier", 100, ">=", 0),
        ),
        Market(
            market_id="epidemic-any",
            question="Will at least one epidemic occur?",
            condition=MarketCondition("epidemic", None, 0, ">", 0),
        ),
        Market(
            market_id="ares-tech-5",
            question="Will Ares Prime unlock >= 5 techs?",
            condition=MarketCondition("tech_race", "Ares Prime", 5, ">=", 0),
        ),
        Market(
            market_id="olympus-100",
            question="Will Olympus Station reach 100+ population?",
            condition=MarketCondition("pop_threshold", "Olympus Station", 100, ">=", sols),
        ),
        Market(
            market_id="frontier-tech-6",
            question="Will Red Frontier unlock >= 6 techs?",
            condition=MarketCondition("tech_race", "Red Frontier", 6, ">=", 0),
        ),
    ]
    return markets


# ---------------------------------------------------------------------------
# Synthetic agents (trade participants)
# ---------------------------------------------------------------------------

@dataclass
class Trader:
    """A prediction market participant."""
    trader_id: str
    personality: str   # "optimist", "pessimist", "contrarian", "random", "informed"
    cash: float = AGENT_STARTING_CASH
    positions: dict[str, float] = field(default_factory=dict)  # market_id -> net YES shares
    pnl: float = 0.0

    def can_afford(self, cost: float) -> bool:
        return self.cash >= cost

    def snapshot(self) -> dict:
        return {
            "trader_id": self.trader_id,
            "personality": self.personality,
            "cash": round(self.cash, 2),
            "positions": {k: round(v, 2) for k, v in self.positions.items()},
            "pnl": round(self.pnl, 2),
        }


PERSONALITIES = ["optimist", "pessimist", "contrarian", "random", "informed"]


def create_traders(n: int, rng: random.Random) -> list[Trader]:
    """Create n synthetic traders with distributed personalities."""
    traders = []
    for i in range(n):
        personality = PERSONALITIES[i % len(PERSONALITIES)]
        traders.append(Trader(
            trader_id=f"trader-{i:03d}",
            personality=personality,
            cash=AGENT_STARTING_CASH,
        ))
    return traders


# ---------------------------------------------------------------------------
# Trading strategies
# ---------------------------------------------------------------------------

def _det_seed(trader_id: str, market_id: str, round_num: int) -> int:
    """Deterministic seed for reproducible trading decisions."""
    h = hashlib.sha256(f"{trader_id}:{market_id}:{round_num}".encode()).hexdigest()
    return int(h[:8], 16)


def decide_trade(
    trader: Trader,
    market: Market,
    round_num: int,
) -> tuple[str, float] | None:
    """Decide whether to trade and how much.

    Returns (outcome, shares) or None if no trade.
    outcome is "yes" or "no", shares is positive for buy.
    """
    rng = random.Random(_det_seed(trader.trader_id, market.market_id, round_num))
    price = market.price_yes

    # Base belief depends on personality
    if trader.personality == "optimist":
        belief = 0.55 + rng.gauss(0, 0.1)
    elif trader.personality == "pessimist":
        belief = 0.35 + rng.gauss(0, 0.1)
    elif trader.personality == "contrarian":
        belief = 1.0 - price + rng.gauss(0, 0.08)
    elif trader.personality == "informed":
        # Informed traders have a noisy signal closer to truth
        # (they "know" survival is likely, epidemics happen ~50%)
        base = 0.75 if "survive" in market.market_id else 0.5
        belief = base + rng.gauss(0, 0.05)
    else:  # random
        belief = rng.random()

    belief = max(0.01, min(0.99, belief))

    # Only trade if belief diverges from market price by > 5%
    edge = belief - price
    if abs(edge) < 0.05:
        return None

    # Size proportional to edge, capped by cash
    shares = min(MAX_TRADE_SIZE, max(1, int(abs(edge) * 50)))

    if edge > 0:
        # Belief higher than price → buy YES
        cost = lmsr_trade_cost(market.q_yes, market.q_no, market.b, "yes", shares)
        if cost > 0 and trader.can_afford(cost):
            return ("yes", shares)
    else:
        # Belief lower than price → buy NO
        cost = lmsr_trade_cost(market.q_yes, market.q_no, market.b, "no", shares)
        if cost > 0 and trader.can_afford(cost):
            return ("no", shares)

    return None


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

@dataclass
class PredictionMarketSim:
    """Full prediction market simulation."""
    markets: list[Market]
    traders: list[Trader]
    rounds: int = DEFAULT_ROUNDS
    seed: int = 42
    trade_log: list[dict] = field(default_factory=list)

    def run(self, quiet: bool = False) -> dict:
        """Run the trading simulation. Returns results dict."""
        rng = random.Random(self.seed)

        for round_num in range(1, self.rounds + 1):
            # Shuffle trader order each round
            order = list(range(len(self.traders)))
            rng.shuffle(order)

            round_trades = 0
            for ti in order:
                trader = self.traders[ti]
                for market in self.markets:
                    if market.resolved:
                        continue
                    decision = decide_trade(trader, market, round_num)
                    if decision is None:
                        continue
                    outcome, shares = decision
                    cost = market.buy(outcome, shares)
                    if cost > trader.cash and cost > 0:
                        # Rollback — shouldn't happen due to check, but safety
                        market.buy(outcome, -shares)
                        continue
                    trader.cash -= cost
                    pos_key = market.market_id
                    current = trader.positions.get(pos_key, 0.0)
                    if outcome == "yes":
                        trader.positions[pos_key] = current + shares
                    else:
                        trader.positions[pos_key] = current - shares
                    round_trades += 1
                    self.trade_log.append({
                        "round": round_num,
                        "trader": trader.trader_id,
                        "market": market.market_id,
                        "outcome": outcome,
                        "shares": round(shares, 2),
                        "cost": round(cost, 4),
                        "price_after": round(market.price_yes, 4),
                    })

            if not quiet and round_num % 10 == 0:
                prices = " | ".join(
                    f"{m.market_id}: {m.price_yes:.2f}" for m in self.markets[:5]
                )
                print(f"  Round {round_num:>3}/{self.rounds}  trades={round_trades}  {prices}")

        return self._build_results()

    def resolve(self, sim_results: dict) -> dict:
        """Resolve all markets against simulation results."""
        resolutions = {}
        for market in self.markets:
            outcome = market.resolve(sim_results)
            resolutions[market.market_id] = {
                "question": market.question,
                "outcome": outcome,
                "final_price": round(market.price_yes, 4),
                "accuracy": round(1.0 - abs(market.price_yes - (1.0 if outcome else 0.0)), 4),
            }

        # Settle PnL
        for trader in self.traders:
            pnl = 0.0
            for market in self.markets:
                pos = trader.positions.get(market.market_id, 0.0)
                if market.outcome:
                    pnl += pos * 1.0  # YES pays $1/share
                else:
                    pnl -= pos * 1.0  # NO pays (negative pos = short YES)
            trader.pnl = pnl

        return resolutions

    def _build_results(self) -> dict:
        """Package results as serializable dict."""
        now = datetime.now(timezone.utc).isoformat()
        return {
            "_meta": {
                "engine": "prediction-market",
                "version": "1.0.0",
                "rounds": self.rounds,
                "num_markets": len(self.markets),
                "num_traders": len(self.traders),
                "total_trades": len(self.trade_log),
                "generated": now,
            },
            "markets": [m.snapshot() for m in self.markets],
            "traders": [t.snapshot() for t in self.traders],
            "trade_log": self.trade_log[-200:],
        }


# ---------------------------------------------------------------------------
# Main — run standalone
# ---------------------------------------------------------------------------

def run_prediction_market(
    n_agents: int = DEFAULT_AGENTS,
    rounds: int = DEFAULT_ROUNDS,
    seed: int = 42,
    resolve: bool = False,
    sols: int = 365,
    quiet: bool = False,
) -> dict:
    """Run the prediction market, optionally resolving via terrarium sim."""
    rng = random.Random(seed)
    markets = create_default_markets(sols=sols)
    traders = create_traders(n_agents, rng)

    sim = PredictionMarketSim(
        markets=markets,
        traders=traders,
        rounds=rounds,
        seed=seed,
    )

    if not quiet:
        print("=" * 64)
        print("  PREDICTION MARKET — Mars Colony Outcomes")
        print("=" * 64)
        print(f"\n  Markets:  {len(markets)}")
        print(f"  Traders:  {len(traders)}")
        print(f"  Rounds:   {rounds}")
        print(f"  LMSR b:   {LMSR_B}")
        print()

    results = sim.run(quiet=quiet)

    if not quiet:
        print()
        print("=" * 64)
        print("  MARKET RESULTS (pre-resolution)")
        print("=" * 64)
        for m in sim.markets:
            vol = f"${m.total_volume:,.0f}" if m.total_volume > 0 else "$0"
            print(f"\n  {m.question}")
            print(f"    YES: {m.price_yes:.1%}  |  NO: {m.price_no:.1%}  |  Vol: {vol}  |  Trades: {m.trade_count}")

    if resolve:
        if not quiet:
            print()
            print("-" * 64)
            print("  Resolving markets via Mars Barn terrarium...")
            print("-" * 64)

        from src.tick_engine import Simulation
        terrarium = Simulation(sols=sols, env_seed=seed)
        sim_results = terrarium.run()
        resolutions = sim.resolve(sim_results)
        results["resolutions"] = resolutions
        results["terrarium_summary"] = sim_results["summary"]
        # Refresh trader snapshots with settled PnL
        results["traders"] = [t.snapshot() for t in sim.traders]

        if not quiet:
            print()
            correct = 0
            total = len(resolutions)
            for mid, res in resolutions.items():
                icon = "✅" if res["outcome"] else "❌"
                accuracy = res["accuracy"]
                print(f"  {icon} {res['question']}")
                print(f"     Price: {res['final_price']:.1%}  |  Accuracy: {accuracy:.1%}")
                if res["accuracy"] > 0.5:
                    correct += 1

            print(f"\n  Market calibration: {correct}/{total} markets priced > 50% correctly")

            # Trader leaderboard
            ranked = sorted(sim.traders, key=lambda t: t.pnl, reverse=True)
            print(f"\n  Top traders (by PnL):")
            for t in ranked[:5]:
                print(f"    {t.trader_id} ({t.personality:>10s}): ${t.pnl:+.2f}")
            print(f"\n  Bottom traders:")
            for t in ranked[-3:]:
                print(f"    {t.trader_id} ({t.personality:>10s}): ${t.pnl:+.2f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction market for Mars colony outcomes")
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolve", action="store_true", help="Resolve by running terrarium")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    results = run_prediction_market(
        n_agents=args.agents,
        rounds=args.rounds,
        seed=args.seed,
        resolve=args.resolve,
        sols=args.sols,
        quiet=args.quiet,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n  Output saved: {out_path}")

    print()
    print("=" * 64)
    meta = results["_meta"]
    print(f"  Total trades: {meta['total_trades']}")
    print(f"  Markets:      {meta['num_markets']}")
    print(f"  Traders:      {meta['num_traders']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
