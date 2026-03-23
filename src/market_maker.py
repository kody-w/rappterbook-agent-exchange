"""
market_maker.py — LMSR prediction market engine for Mars Barn terrarium.

Logarithmic Market Scoring Rule (Hanson 2003/2007) applied to colony outcomes.
Agents stake karma on predictions about terrarium simulation results:
  - Colony survival (binary)
  - Population thresholds (binary per colony)
  - Comparative outcomes (multi-outcome: who grows fastest, biggest, etc.)
  - Epidemic occurrence (binary)
  - Tech leadership (multi-outcome)

The market uses LMSR for automated market making:
  Cost function: C(q) = b * ln(sum(exp(q_i / b)))
  Price of outcome i: p_i = exp(q_i / b) / sum(exp(q_j / b))
  Prices always sum to 1.0 (probability constraint).

Usage:
    # Standalone (runs internal sim)
    PYTHONPATH=. python3 src/market_maker.py

    # With live sim data
    PYTHONPATH=. python3 src/market_maker.py --live --sols 365 --seeds 10

    # As library
    from src.market_maker import MarketEngine, run_prediction_market
    results = run_prediction_market(sols=365, seeds=10)
"""
from __future__ import annotations

import json
import math
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# LMSR math (pure functions, no side effects)
# ---------------------------------------------------------------------------

def lmsr_cost(quantities: list[float], b: float) -> float:
    """LMSR cost function: C(q) = b * ln(sum(exp(q_i / b))).

    Uses log-sum-exp trick for numerical stability.
    """
    if b <= 0:
        raise ValueError("Liquidity parameter b must be positive")
    max_q = max(quantities)
    shifted = [q - max_q for q in quantities]
    return b * (max_q / b + math.log(sum(math.exp(s / b) for s in shifted)))


def lmsr_prices(quantities: list[float], b: float) -> list[float]:
    """Compute LMSR prices (probabilities) for each outcome.

    Price of outcome i = exp(q_i / b) / sum(exp(q_j / b)).
    Always sums to 1.0.
    """
    if b <= 0:
        raise ValueError("Liquidity parameter b must be positive")
    max_q = max(quantities)
    exps = [math.exp((q - max_q) / b) for q in quantities]
    total = sum(exps)
    return [e / total for e in exps]


def lmsr_trade_cost(
    quantities: list[float], outcome_idx: int, shares: float, b: float
) -> float:
    """Cost to buy `shares` of outcome `outcome_idx`.

    Returns positive number = cost to buyer, negative = payout to seller.
    """
    old_cost = lmsr_cost(quantities, b)
    new_q = list(quantities)
    new_q[outcome_idx] += shares
    new_cost = lmsr_cost(new_q, b)
    return new_cost - old_cost


# ---------------------------------------------------------------------------
# Market data structures
# ---------------------------------------------------------------------------

@dataclass
class Market:
    """A single prediction market with N outcomes."""
    market_id: str
    question: str
    outcomes: list[str]
    quantities: list[float]
    b: float  # liquidity parameter
    resolved: bool = False
    winning_outcome: int | None = None
    total_volume: float = 0.0
    trade_count: int = 0

    def prices(self) -> list[float]:
        """Current market prices (implied probabilities)."""
        return lmsr_prices(self.quantities, self.b)

    def buy(self, outcome_idx: int, shares: float) -> float:
        """Execute a buy trade. Returns cost."""
        cost = lmsr_trade_cost(self.quantities, outcome_idx, shares, self.b)
        self.quantities[outcome_idx] += shares
        self.total_volume += abs(cost)
        self.trade_count += 1
        return cost

    def resolve(self, winning_outcome: int) -> None:
        """Resolve the market with the winning outcome index."""
        self.resolved = True
        self.winning_outcome = winning_outcome

    def snapshot(self) -> dict:
        """Serialize market state."""
        return {
            "market_id": self.market_id,
            "question": self.question,
            "outcomes": self.outcomes,
            "prices": [round(p, 4) for p in self.prices()],
            "resolved": self.resolved,
            "winning_outcome": self.winning_outcome,
            "total_volume": round(self.total_volume, 2),
            "trade_count": self.trade_count,
        }


@dataclass
class TraderPosition:
    """Tracks a trader's position in a single market."""
    shares: dict[int, float] = field(default_factory=dict)  # outcome_idx → shares
    total_cost: float = 0.0


@dataclass
class Trader:
    """A market participant with a trading strategy."""
    trader_id: str
    strategy: str  # oracle, bull, bear, contrarian, techie, random
    karma: float = 1000.0
    positions: dict[str, TraderPosition] = field(default_factory=dict)

    def pnl(self, markets: dict[str, Market]) -> float:
        """Calculate P&L across all resolved markets."""
        total_pnl = 0.0
        for mid, pos in self.positions.items():
            if mid not in markets:
                continue
            mkt = markets[mid]
            if not mkt.resolved or mkt.winning_outcome is None:
                continue
            winnings = pos.shares.get(mkt.winning_outcome, 0.0)
            total_pnl += winnings - pos.total_cost
        return total_pnl

    def roi(self, markets: dict[str, Market]) -> float:
        """Return on investment as percentage."""
        total_cost = sum(p.total_cost for p in self.positions.values())
        if total_cost <= 0:
            return 0.0
        return (self.pnl(markets) / total_cost) * 100.0


# ---------------------------------------------------------------------------
# Market engine (create, trade, resolve)
# ---------------------------------------------------------------------------

class MarketEngine:
    """Creates and manages prediction markets for terrarium outcomes."""

    def __init__(self, liquidity: float = 100.0, seed: int = 42) -> None:
        self.liquidity = liquidity
        self.markets: dict[str, Market] = {}
        self.traders: dict[str, Trader] = {}
        self.rng = random.Random(seed)

    def create_market(
        self, market_id: str, question: str, outcomes: list[str]
    ) -> Market:
        """Create a new market with uniform initial prices."""
        n = len(outcomes)
        quantities = [0.0] * n
        mkt = Market(
            market_id=market_id,
            question=question,
            outcomes=outcomes,
            quantities=quantities,
            b=self.liquidity,
        )
        self.markets[market_id] = mkt
        return mkt

    def create_colony_markets(self, colony_names: list[str]) -> list[Market]:
        """Create the standard set of terrarium prediction markets."""
        created = []

        # 1. Survival markets (binary per colony)
        for name in colony_names:
            slug = name.lower().replace(" ", "-")
            mkt = self.create_market(
                f"survival-{slug}",
                f"Will {name} survive 365 sols?",
                ["Survives", "Extinct"],
            )
            created.append(mkt)

        # 2. Population threshold markets (3 thresholds per colony)
        for name in colony_names:
            slug = name.lower().replace(" ", "-")
            for threshold in [100, 150, 200]:
                mkt = self.create_market(
                    f"pop-{slug}-{threshold}",
                    f"Will {name} exceed {threshold} by sol 365?",
                    ["Yes", "No"],
                )
                created.append(mkt)

        # 3. Comparative markets
        mkt = self.create_market(
            "fastest-grower",
            "Which colony grows fastest (% increase)?",
            colony_names,
        )
        created.append(mkt)

        mkt = self.create_market(
            "largest-pop",
            "Which colony has largest population at sol 365?",
            colony_names,
        )
        created.append(mkt)

        # 4. Epidemic market (binary)
        mkt = self.create_market(
            "any-epidemic",
            "Will any colony experience an epidemic in 365 sols?",
            ["Yes", "No"],
        )
        created.append(mkt)

        # 5. Tech leadership (multi-outcome)
        mkt = self.create_market(
            "tech-leader",
            "Which colony unlocks the most technologies?",
            colony_names,
        )
        created.append(mkt)

        return created

    def add_trader(self, trader_id: str, strategy: str, karma: float = 1000.0) -> Trader:
        """Add a trader to the engine."""
        trader = Trader(trader_id=trader_id, strategy=strategy, karma=karma)
        self.traders[trader_id] = trader
        return trader

    def execute_trade(
        self, trader: Trader, market: Market, outcome_idx: int, shares: float
    ) -> float:
        """Execute a trade for a trader. Returns cost."""
        cost = market.buy(outcome_idx, shares)
        if market.market_id not in trader.positions:
            trader.positions[market.market_id] = TraderPosition()
        pos = trader.positions[market.market_id]
        pos.shares[outcome_idx] = pos.shares.get(outcome_idx, 0.0) + shares
        pos.total_cost += cost
        return cost

    def simulate_trading(
        self, rounds: int = 50, sim_results: dict | None = None
    ) -> list[dict]:
        """Simulate trading rounds with different strategies."""
        if not self.traders:
            self._create_default_traders()

        trades = []
        market_list = list(self.markets.values())

        for round_num in range(rounds):
            for trader in self.traders.values():
                market = self.rng.choice(market_list)
                if market.resolved:
                    continue

                outcome_idx, shares = self._trader_decision(
                    trader, market, sim_results
                )
                cost = self.execute_trade(trader, market, outcome_idx, shares)
                trades.append({
                    "round": round_num,
                    "trader": trader.trader_id,
                    "market": market.market_id,
                    "outcome": market.outcomes[outcome_idx],
                    "shares": round(shares, 2),
                    "cost": round(cost, 4),
                })

        return trades

    def resolve_markets(self, sim_results: dict) -> dict[str, int]:
        """Resolve all markets using terrarium simulation results."""
        resolutions = {}
        colonies = sim_results.get("summary", {}).get("colonies", [])
        colony_names = [c["name"] for c in colonies]

        for mid, market in self.markets.items():
            winner = self._resolve_single(mid, market, colonies, colony_names)
            if winner is not None:
                market.resolve(winner)
                resolutions[mid] = winner

        return resolutions

    def score_traders(self) -> list[dict]:
        """Score all traders by P&L and ROI."""
        scores = []
        for trader in sorted(
            self.traders.values(),
            key=lambda t: t.pnl(self.markets),
            reverse=True,
        ):
            scores.append({
                "trader_id": trader.trader_id,
                "strategy": trader.strategy,
                "pnl": round(trader.pnl(self.markets), 2),
                "roi": round(trader.roi(self.markets), 1),
                "markets_traded": len(trader.positions),
            })
        return scores

    def snapshot(self) -> dict:
        """Full engine state snapshot."""
        return {
            "markets": {
                mid: mkt.snapshot() for mid, mkt in self.markets.items()
            },
            "traders": self.score_traders(),
            "total_markets": len(self.markets),
            "resolved_markets": sum(
                1 for m in self.markets.values() if m.resolved
            ),
            "total_volume": round(
                sum(m.total_volume for m in self.markets.values()), 2
            ),
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _create_default_traders(self) -> None:
        """Create the 6 default trader archetypes."""
        archetypes = [
            ("oracle", "oracle", 1500),
            ("bull", "bull", 1000),
            ("bear", "bear", 1000),
            ("contrarian", "contrarian", 800),
            ("techie", "techie", 1200),
            ("noise", "random", 500),
        ]
        for tid, strat, karma in archetypes:
            self.add_trader(tid, strat, karma)

    def _trader_decision(
        self,
        trader: Trader,
        market: Market,
        sim_results: dict | None,
    ) -> tuple[int, float]:
        """Decide which outcome to buy and how many shares.

        Returns (outcome_idx, shares).
        """
        prices = market.prices()
        n = len(prices)

        if trader.strategy == "oracle" and sim_results is not None:
            # Oracle knows the true outcome — bets on the winner
            winner = self._peek_resolution(market, sim_results)
            if winner is not None:
                return winner, self.rng.uniform(5, 15)
            return self.rng.randrange(n), self.rng.uniform(2, 8)

        if trader.strategy == "bull":
            # Always bets on the first (positive) outcome
            return 0, self.rng.uniform(3, 10)

        if trader.strategy == "bear":
            # Always bets on the last (negative) outcome
            return n - 1, self.rng.uniform(3, 10)

        if trader.strategy == "contrarian":
            # Bets on the cheapest outcome
            cheapest = min(range(n), key=lambda i: prices[i])
            return cheapest, self.rng.uniform(2, 8)

        if trader.strategy == "techie":
            # Bets on tech-related markets or positive outcomes
            if "tech" in market.market_id:
                return self.rng.randrange(n), self.rng.uniform(5, 12)
            return 0, self.rng.uniform(2, 6)

        # random / fallback
        return self.rng.randrange(n), self.rng.uniform(1, 5)

    def _peek_resolution(self, market: Market, sim_results: dict) -> int | None:
        """Oracle peeks at simulation results to know the winner."""
        try:
            return self._resolve_single(
                market.market_id,
                market,
                sim_results.get("summary", {}).get("colonies", []),
                [c["name"] for c in sim_results.get("summary", {}).get("colonies", [])],
            )
        except Exception:
            return None

    def _resolve_single(
        self,
        market_id: str,
        market: Market,
        colonies: list[dict],
        colony_names: list[str],
    ) -> int | None:
        """Resolve a single market given colony summary data."""
        if market_id.startswith("survival-"):
            slug = market_id.replace("survival-", "")
            for c in colonies:
                if c["name"].lower().replace(" ", "-") == slug:
                    return 0 if c["end_pop"] > 0 else 1
            return None

        if market_id.startswith("pop-"):
            parts = market_id.split("-")
            threshold = int(parts[-1])
            slug = "-".join(parts[1:-1])
            for c in colonies:
                if c["name"].lower().replace(" ", "-") == slug:
                    return 0 if c["end_pop"] >= threshold else 1
            return None

        if market_id == "fastest-grower":
            if not colonies:
                return None
            best_idx = max(
                range(len(colonies)),
                key=lambda i: colonies[i].get("growth_pct", 0),
            )
            name = colonies[best_idx]["name"]
            if name in market.outcomes:
                return market.outcomes.index(name)
            return None

        if market_id == "largest-pop":
            if not colonies:
                return None
            best_idx = max(
                range(len(colonies)),
                key=lambda i: colonies[i].get("end_pop", 0),
            )
            name = colonies[best_idx]["name"]
            if name in market.outcomes:
                return market.outcomes.index(name)
            return None

        if market_id == "any-epidemic":
            for c in colonies:
                causes = c.get("death_causes", {})
                if causes.get("epidemic", 0) > 0:
                    return 0  # Yes
            return 1  # No

        if market_id == "tech-leader":
            if not colonies:
                return None
            best_idx = max(
                range(len(colonies)),
                key=lambda i: colonies[i].get("techs_unlocked", 0),
            )
            name = colonies[best_idx]["name"]
            if name in market.outcomes:
                return market.outcomes.index(name)
            return None

        return None


# ---------------------------------------------------------------------------
# Pipeline: create markets → trade → resolve → score
# ---------------------------------------------------------------------------

def run_prediction_market(
    sols: int = 365,
    seeds: int = 1,
    liquidity: float = 100.0,
    trading_rounds: int = 50,
    engine_seed: int = 42,
) -> dict:
    """Full pipeline: run sim, create markets, trade, resolve, score."""
    from src.tick_engine import Simulation

    # Run terrarium simulation
    sim = Simulation(sols=sols, env_seed=engine_seed)
    sim_results = sim.run()

    # Build market engine
    engine = MarketEngine(liquidity=liquidity, seed=engine_seed)
    colony_names = [c["name"] for c in sim_results["summary"]["colonies"]]
    engine.create_colony_markets(colony_names)

    # Simulate trading
    trades = engine.simulate_trading(
        rounds=trading_rounds, sim_results=sim_results
    )

    # Resolve markets
    resolutions = engine.resolve_markets(sim_results)

    # Score traders
    scores = engine.score_traders()

    now = datetime.now(timezone.utc).isoformat()
    return {
        "_meta": {
            "engine": "prediction-market",
            "version": "1.0",
            "model": "LMSR",
            "liquidity": liquidity,
            "generated": now,
        },
        "sim_summary": sim_results["summary"],
        "markets": engine.snapshot()["markets"],
        "trader_scores": scores,
        "trade_log_sample": trades[:20],
        "stats": {
            "total_markets": len(engine.markets),
            "resolved_markets": sum(
                1 for m in engine.markets.values() if m.resolved
            ),
            "total_trades": len(trades),
            "total_volume": round(
                sum(m.total_volume for m in engine.markets.values()), 2
            ),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="LMSR Prediction Market Engine")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--liquidity", type=float, default=100.0)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--live", action="store_true", help="Run with live sim")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_prediction_market(
        sols=args.sols,
        seeds=args.seeds,
        liquidity=args.liquidity,
        trading_rounds=args.rounds,
    )

    # Print summary
    print("=" * 60)
    print("LMSR PREDICTION MARKET — RESULTS")
    print("=" * 60)
    stats = results["stats"]
    print(f"Markets: {stats['total_markets']} | "
          f"Resolved: {stats['resolved_markets']} | "
          f"Trades: {stats['total_trades']} | "
          f"Volume: {stats['total_volume']:.1f}")
    print()

    print("Colony Outcomes:")
    for c in results["sim_summary"]["colonies"]:
        print(f"  {c['name']}: {c['start_pop']}→{c['end_pop']} "
              f"({c['growth_pct']:+.1f}%) | "
              f"Techs: {c['techs_unlocked']}")
    print()

    print("Market Prices (final):")
    for mid, mdata in results["markets"].items():
        winner = ""
        if mdata["resolved"] and mdata["winning_outcome"] is not None:
            winner = f" ✓ {mdata['outcomes'][mdata['winning_outcome']]}"
        prices_str = " | ".join(
            f"{o}: {p:.1%}" for o, p in zip(mdata["outcomes"], mdata["prices"])
        )
        print(f"  {mid}: {prices_str}{winner}")
    print()

    print("Trader Leaderboard:")
    for t in results["trader_scores"]:
        print(f"  {t['trader_id']:12s} ({t['strategy']:11s}): "
              f"P&L {t['pnl']:+8.1f} | ROI {t['roi']:+6.1f}%")

    # Save output
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("state/prediction_market.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
