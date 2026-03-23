"""
market_maker.py — Prediction market for Mars colony outcomes.

Uses Logarithmic Market Scoring Rule (LMSR) to price binary event
contracts on colony survival, population thresholds, and tech milestones.

Run standalone:
    python src/market_maker.py                  # uses saved state/mars.json
    python src/market_maker.py --live            # runs sim first, then markets
    python src/market_maker.py --sols 365 --seeds 10

The market maker IS the automated market maker (AMM). Traders are
simulated agents with different risk profiles. Resolution is deterministic.

Python stdlib only.
"""
from __future__ import annotations

import json
import math
import random
import hashlib
import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# LMSR (Logarithmic Market Scoring Rule)
# ---------------------------------------------------------------------------

DEFAULT_LIQUIDITY = 100.0


def lmsr_cost(quantities: list[float], b: float = DEFAULT_LIQUIDITY) -> float:
    """Cost function C(q) = b * ln(sum(exp(q_i / b)))."""
    max_q = max(quantities)
    shifted = [q - max_q for q in quantities]
    return b * (max_q / b + math.log(sum(math.exp(s / b) for s in shifted)))


def lmsr_prices(quantities: list[float], b: float = DEFAULT_LIQUIDITY) -> list[float]:
    """Price of each outcome = exp(q_i / b) / sum(exp(q_j / b))."""
    max_q = max(quantities)
    shifted = [q - max_q for q in quantities]
    exps = [math.exp(s / b) for s in shifted]
    total = sum(exps)
    return [e / total for e in exps]


def lmsr_trade_cost(
    quantities: list[float],
    outcome_idx: int,
    shares: float,
    b: float = DEFAULT_LIQUIDITY,
) -> float:
    """Cost to buy `shares` of outcome `outcome_idx`."""
    old_cost = lmsr_cost(quantities, b)
    new_quantities = list(quantities)
    new_quantities[outcome_idx] += shares
    new_cost = lmsr_cost(new_quantities, b)
    return new_cost - old_cost


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Record of a single trade."""
    trader_id: str
    market_id: str
    outcome_idx: int
    shares: float
    cost: float
    price_before: float
    price_after: float
    sol: int


@dataclass
class Market:
    """A binary (or multi-outcome) prediction market."""
    market_id: str
    question: str
    outcomes: list[str]
    quantities: list[float]
    liquidity: float
    resolution: int | None = None
    trades: list[Trade] = field(default_factory=list)
    created_sol: int = 0

    def prices(self) -> list[float]:
        return lmsr_prices(self.quantities, self.liquidity)

    def buy(self, outcome_idx: int, shares: float, trader_id: str, sol: int = 0) -> Trade:
        price_before = self.prices()[outcome_idx]
        cost = lmsr_trade_cost(self.quantities, outcome_idx, shares, self.liquidity)
        self.quantities[outcome_idx] += shares
        price_after = self.prices()[outcome_idx]
        trade = Trade(
            trader_id=trader_id, market_id=self.market_id,
            outcome_idx=outcome_idx, shares=shares, cost=cost,
            price_before=round(price_before, 4),
            price_after=round(price_after, 4), sol=sol,
        )
        self.trades.append(trade)
        return trade

    def resolve(self, winning_idx: int) -> None:
        self.resolution = winning_idx

    def pnl(self, trader_id: str) -> float:
        if self.resolution is None:
            return 0.0
        total_cost = 0.0
        winning_shares = 0.0
        for trade in self.trades:
            if trade.trader_id != trader_id:
                continue
            total_cost += trade.cost
            if trade.outcome_idx == self.resolution:
                winning_shares += trade.shares
        return winning_shares - total_cost

    def snapshot(self) -> dict:
        prices = self.prices()
        return {
            "market_id": self.market_id,
            "question": self.question,
            "outcomes": self.outcomes,
            "prices": [round(p, 4) for p in prices],
            "num_trades": len(self.trades),
            "resolved": self.resolution is not None,
            "winning_outcome": (
                self.outcomes[self.resolution]
                if self.resolution is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# Market factory
# ---------------------------------------------------------------------------

def create_colony_markets(colony_names: list[str], strategies: list[str]) -> list[Market]:
    """Create prediction markets for colony outcomes."""
    markets: list[Market] = []

    for name in colony_names:
        slug = name.lower().replace(" ", "-")
        markets.append(Market(
            market_id=f"survival-{slug}",
            question=f"Will {name} survive 365 sols?",
            outcomes=["Yes", "No"],
            quantities=[0.0, 0.0],
            liquidity=DEFAULT_LIQUIDITY,
        ))

    thresholds = [100, 150, 200]
    for name in colony_names:
        slug = name.lower().replace(" ", "-")
        for threshold in thresholds:
            markets.append(Market(
                market_id=f"pop-{slug}-{threshold}",
                question=f"Will {name} exceed {threshold} population by sol 365?",
                outcomes=["Yes", "No"],
                quantities=[0.0, 0.0],
                liquidity=DEFAULT_LIQUIDITY,
            ))

    markets.append(Market(
        market_id="fastest-grower",
        question="Which colony will have the highest growth %?",
        outcomes=colony_names,
        quantities=[0.0] * len(colony_names),
        liquidity=DEFAULT_LIQUIDITY,
    ))

    markets.append(Market(
        market_id="largest-pop",
        question="Which colony will have the largest final population?",
        outcomes=colony_names,
        quantities=[0.0] * len(colony_names),
        liquidity=DEFAULT_LIQUIDITY,
    ))

    markets.append(Market(
        market_id="any-epidemic",
        question="Will any colony experience an epidemic?",
        outcomes=["Yes", "No"],
        quantities=[0.0, 0.0],
        liquidity=DEFAULT_LIQUIDITY,
    ))

    markets.append(Market(
        market_id="tech-leader",
        question="Which colony will unlock the most technologies?",
        outcomes=colony_names,
        quantities=[0.0] * len(colony_names),
        liquidity=DEFAULT_LIQUIDITY,
    ))

    return markets


# ---------------------------------------------------------------------------
# Simulated traders
# ---------------------------------------------------------------------------

TRADER_PROFILES = [
    {"id": "oracle", "style": "informed", "bankroll": 500.0},
    {"id": "bull", "style": "optimist", "bankroll": 300.0},
    {"id": "bear", "style": "pessimist", "bankroll": 300.0},
    {"id": "contrarian", "style": "contrarian", "bankroll": 200.0},
    {"id": "random-walk", "style": "noise", "bankroll": 150.0},
    {"id": "techie", "style": "tech_bull", "bankroll": 250.0},
]


@dataclass
class Trader:
    """Simulated market participant."""
    trader_id: str
    style: str
    bankroll: float
    spent: float = 0.0

    def can_afford(self, cost: float) -> bool:
        return self.spent + cost <= self.bankroll

    def record_spend(self, cost: float) -> None:
        self.spent += max(0, cost)


def _deterministic_rng(trader_id: str, market_id: str) -> random.Random:
    h = hashlib.sha256(f"{trader_id}:{market_id}".encode()).hexdigest()
    return random.Random(int(h[:8], 16))


def simulate_trading(
    markets: list[Market],
    mc_stats: dict | None = None,
    rounds: int = 5,
) -> list[Trader]:
    """Simulate trading rounds across all markets."""
    traders = [
        Trader(trader_id=p["id"], style=p["style"], bankroll=p["bankroll"])
        for p in TRADER_PROFILES
    ]
    for round_num in range(rounds):
        for market in markets:
            prices = market.prices()
            for trader in traders:
                rng = _deterministic_rng(
                    trader.trader_id, f"{market.market_id}:{round_num}")
                shares, outcome_idx = _decide_trade(
                    trader, market, prices, mc_stats, rng)
                if shares <= 0:
                    continue
                cost = lmsr_trade_cost(
                    market.quantities, outcome_idx, shares, market.liquidity)
                if cost > 0 and not trader.can_afford(cost):
                    shares = max(1, shares // 2)
                    cost = lmsr_trade_cost(
                        market.quantities, outcome_idx, shares, market.liquidity)
                    if cost > 0 and not trader.can_afford(cost):
                        continue
                market.buy(outcome_idx, shares, trader.trader_id, sol=round_num)
                if cost > 0:
                    trader.record_spend(cost)
    return traders


def _decide_trade(
    trader: Trader, market: Market, prices: list[float],
    mc_stats: dict | None, rng: random.Random,
) -> tuple[float, int]:
    """Route to strategy-specific trade logic."""
    if trader.style == "informed" and mc_stats:
        return _informed_trade(market, mc_stats, rng)
    elif trader.style == "optimist":
        return _optimist_trade(market, prices, rng)
    elif trader.style == "pessimist":
        return _pessimist_trade(market, prices, rng)
    elif trader.style == "contrarian":
        return _contrarian_trade(market, prices, rng)
    elif trader.style == "tech_bull":
        return _tech_bull_trade(market, prices, rng)
    else:
        return _noise_trade(market, prices, rng)


def _informed_trade(
    market: Market, mc_stats: dict, rng: random.Random,
) -> tuple[float, int]:
    """Oracle trades on Monte Carlo data."""
    mid = market.market_id
    colony_names = mc_stats.get("colony_names", [])
    survival_rates = mc_stats.get("survival_rates", [])
    growth_stats = mc_stats.get("growth_pct_stats", [])

    if mid.startswith("survival-"):
        slug = mid.replace("survival-", "")
        for i, name in enumerate(colony_names):
            if name.lower().replace(" ", "-") == slug and i < len(survival_rates):
                rate = survival_rates[i]
                shares = rng.randint(5, 15)
                return (shares, 0) if rate > 0.7 else (shares, 1)
        return (0, 0)

    if mid.startswith("pop-"):
        parts = mid.split("-")
        threshold = int(parts[-1])
        slug = "-".join(parts[1:-1])
        for i, name in enumerate(colony_names):
            if name.lower().replace(" ", "-") == slug and i < len(growth_stats):
                mean_final = mc_stats.get("final_pop_stats", [{}])[i].get("mean", 0)
                shares = rng.randint(3, 12)
                return (shares, 0) if mean_final > threshold else (shares, 1)
        return (0, 0)

    if mid == "fastest-grower" and growth_stats:
        best_idx = max(
            range(len(growth_stats)),
            key=lambda j: growth_stats[j].get("mean", 0))
        return (rng.randint(5, 15), best_idx)

    if mid == "largest-pop":
        final_pops = mc_stats.get("final_pop_stats", [])
        if final_pops:
            best_idx = max(
                range(len(final_pops)),
                key=lambda j: final_pops[j].get("mean", 0))
            return (rng.randint(5, 15), best_idx)

    if mid == "tech-leader":
        return (rng.randint(3, 8), 0)

    return (rng.randint(2, 8), rng.randint(0, len(market.outcomes) - 1))


def _optimist_trade(
    market: Market, prices: list[float], rng: random.Random,
) -> tuple[float, int]:
    if "survive" in market.question.lower() or "exceed" in market.question.lower():
        return (rng.randint(3, 10), 0)
    return (rng.randint(2, 6), rng.randint(0, len(market.outcomes) - 1))


def _pessimist_trade(
    market: Market, prices: list[float], rng: random.Random,
) -> tuple[float, int]:
    if "survive" in market.question.lower():
        return (rng.randint(3, 10), 1)
    if "exceed" in market.question.lower():
        return (rng.randint(2, 8), 1)
    return (rng.randint(2, 5), rng.randint(0, len(market.outcomes) - 1))


def _contrarian_trade(
    market: Market, prices: list[float], rng: random.Random,
) -> tuple[float, int]:
    cheapest_idx = min(range(len(prices)), key=lambda i: prices[i])
    return (rng.randint(3, 8), cheapest_idx)


def _tech_bull_trade(
    market: Market, prices: list[float], rng: random.Random,
) -> tuple[float, int]:
    if "tech" in market.question.lower():
        return (rng.randint(5, 12), 2)
    if "exceed" in market.question.lower() and "200" in market.question:
        return (rng.randint(3, 8), 0)
    return (rng.randint(1, 4), rng.randint(0, len(market.outcomes) - 1))


def _noise_trade(
    market: Market, prices: list[float], rng: random.Random,
) -> tuple[float, int]:
    return (rng.randint(1, 5), rng.randint(0, len(market.outcomes) - 1))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_markets(markets: list[Market], sim_results: dict) -> dict:
    """Resolve all markets against actual simulation results."""
    colonies = sim_results.get("colonies", [])
    colony_data = {}
    for i, col in enumerate(colonies):
        name = col["name"]
        slug = name.lower().replace(" ", "-")
        final_pop = col.get("final_population", 0)
        initial_pop = col.get("initial_population", 1)
        growth_pct = (final_pop - initial_pop) / max(1, initial_pop) * 100
        techs = col.get("tech", {}).get("unlocked_count", 0)
        events = col.get("events", [])
        had_epidemic = any(
            e.get("type", "").startswith("epidemic") for e in events)
        colony_data[slug] = {
            "name": name, "idx": i, "final_pop": final_pop,
            "growth_pct": growth_pct, "techs": techs,
            "had_epidemic": had_epidemic, "survived": final_pop > 0,
        }

    any_epidemic = any(cd["had_epidemic"] for cd in colony_data.values())
    resolutions = {}

    for market in markets:
        mid = market.market_id
        if mid.startswith("survival-"):
            slug = mid.replace("survival-", "")
            if slug in colony_data:
                survived = colony_data[slug]["survived"]
                market.resolve(0 if survived else 1)
                resolutions[mid] = "Yes" if survived else "No"

        elif mid.startswith("pop-"):
            parts = mid.split("-")
            threshold = int(parts[-1])
            slug = "-".join(parts[1:-1])
            if slug in colony_data:
                exceeded = colony_data[slug]["final_pop"] > threshold
                market.resolve(0 if exceeded else 1)
                resolutions[mid] = "Yes" if exceeded else "No"

        elif mid == "fastest-grower":
            if colony_data:
                fastest = max(colony_data.values(), key=lambda x: x["growth_pct"])
                market.resolve(fastest["idx"])
                resolutions[mid] = fastest["name"]

        elif mid == "largest-pop":
            if colony_data:
                largest = max(colony_data.values(), key=lambda x: x["final_pop"])
                market.resolve(largest["idx"])
                resolutions[mid] = largest["name"]

        elif mid == "any-epidemic":
            market.resolve(0 if any_epidemic else 1)
            resolutions[mid] = "Yes" if any_epidemic else "No"

        elif mid == "tech-leader":
            if colony_data:
                leader = max(colony_data.values(), key=lambda x: x["techs"])
                market.resolve(leader["idx"])
                resolutions[mid] = leader["name"]

    return resolutions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_traders(markets: list[Market], traders: list[Trader]) -> list[dict]:
    """Compute final P&L for each trader across all resolved markets."""
    scores = []
    for trader in traders:
        total_pnl = 0.0
        market_pnls = {}
        for market in markets:
            pnl = market.pnl(trader.trader_id)
            total_pnl += pnl
            if abs(pnl) > 0.01:
                market_pnls[market.market_id] = round(pnl, 2)
        scores.append({
            "trader_id": trader.trader_id,
            "style": trader.style,
            "bankroll": trader.bankroll,
            "spent": round(trader.spent, 2),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(total_pnl / max(trader.spent, 1) * 100, 1),
            "market_pnls": market_pnls,
        })
    scores.sort(key=lambda x: -x["total_pnl"])
    return scores


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_prediction_market(
    sim_results: dict,
    mc_stats: dict | None = None,
    trading_rounds: int = 5,
    verbose: bool = True,
) -> dict:
    """Full pipeline: create markets -> trade -> resolve -> score."""
    colonies = sim_results.get("colonies", [])
    colony_names = [c["name"] for c in colonies]
    colony_strategies = [c.get("strategy", "balanced") for c in colonies]

    if verbose:
        print("\n" + "=" * 60)
        print("  PREDICTION MARKET — Mars Colony Futures")
        print("=" * 60)

    markets = create_colony_markets(colony_names, colony_strategies)
    if verbose:
        print(f"\n  Created {len(markets)} markets:")
        for m in markets:
            prices = m.prices()
            price_str = " | ".join(
                f"{o}: {p:.1%}" for o, p in zip(m.outcomes, prices))
            print(f"    [{m.market_id}] {m.question}")
            print(f"      Opening: {price_str}")

    if verbose:
        print(f"\n  Simulating {trading_rounds} rounds with "
              f"{len(TRADER_PROFILES)} traders...")
    traders = simulate_trading(markets, mc_stats, rounds=trading_rounds)

    if verbose:
        print("\n  Post-trading prices:")
        for m in markets:
            prices = m.prices()
            price_str = " | ".join(
                f"{o}: {p:.1%}" for o, p in zip(m.outcomes, prices))
            print(f"    [{m.market_id}] {price_str}  ({len(m.trades)} trades)")

    resolutions = resolve_markets(markets, sim_results)
    if verbose:
        print("\n  RESOLUTION:")
        for mid, result in resolutions.items():
            print(f"    {mid}: {result}")

    scores = score_traders(markets, traders)
    if verbose:
        print("\n  TRADER LEADERBOARD:")
        print(f"    {'Trader':<15} {'Style':<12} "
              f"{'Spent':>8} {'P&L':>8} {'ROI':>8}")
        print(f"    {'-'*15} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
        for s in scores:
            print(f"    {s['trader_id']:<15} {s['style']:<12} "
                  f"${s['spent']:>7.2f} ${s['total_pnl']:>7.2f} "
                  f"{s['roi_pct']:>6.1f}%")

    now = datetime.now(timezone.utc).isoformat()
    total_trades = sum(len(m.trades) for m in markets)
    result = {
        "_meta": {
            "engine": "prediction-market",
            "version": "1.0.0",
            "generated": now,
            "num_markets": len(markets),
            "num_traders": len(traders),
            "trading_rounds": trading_rounds,
            "total_trades": total_trades,
        },
        "markets": [m.snapshot() for m in markets],
        "resolutions": resolutions,
        "traders": scores,
    }

    if verbose:
        print(f"\n  Total trades: {total_trades}")
        best = scores[0] if scores else None
        if best:
            print(f"  Best trader: {best['trader_id']} ({best['style']}) "
                  f"with ${best['total_pnl']:.2f} P&L "
                  f"({best['roi_pct']:.1f}% ROI)")
        print()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Run prediction market — either from saved state or live sim."""
    parser = argparse.ArgumentParser(
        description="Mars Colony Prediction Market")
    parser.add_argument("--live", action="store_true",
                        help="Run sim first, then markets")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--seeds", type=int, default=10,
                        help="Monte Carlo seeds")
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--trading-rounds", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    state_dir = Path(args.state_dir) if args.state_dir else REPO_ROOT / "state"

    if args.live:
        print("=" * 60)
        print("  MARS BARN TERRARIUM + PREDICTION MARKET")
        print("=" * 60)

        from src.tick_engine import Simulation
        from src.monte_carlo import run_ensemble

        print(f"\n  Running Monte Carlo: {args.seeds} seeds x {args.sols} sols...")
        ensemble = run_ensemble(n_seeds=args.seeds, sols=args.sols)
        sim_results = ensemble.canonical_results

        print("\n  TERRARIUM RESULTS:")
        for col in sim_results["colonies"]:
            name = col["name"]
            start = col["initial_population"]
            end = col["final_population"]
            growth = (end - start) / max(1, start) * 100
            techs = col.get("tech", {}).get("unlocked_count", 0)
            print(f"    {name}: {start} -> {end} ({growth:+.1f}%) "
                  f"[{techs} techs]")

        mc_stats = {
            "colony_names": ensemble.colony_names,
            "colony_strategies": ensemble.colony_strategies,
            "survival_rates": ensemble.survival_rates,
            "final_pop_stats": [
                {k: round(v, 1) for k, v in fps.items()}
                for fps in ensemble.final_pop_stats
            ],
            "growth_pct_stats": [
                {k: round(v, 1) for k, v in gps.items()}
                for gps in ensemble.growth_pct_stats
            ],
        }

        result = run_prediction_market(
            sim_results, mc_stats=mc_stats,
            trading_rounds=args.trading_rounds,
            verbose=not args.quiet,
        )
    else:
        mars_path = state_dir / "mars.json"
        if not mars_path.exists():
            print(f"ERROR: {mars_path} not found. "
                  "Run with --live or run main.py first.")
            sys.exit(1)
        with open(mars_path) as f:
            sim_results = json.load(f)
        result = run_prediction_market(
            sim_results, mc_stats=None,
            trading_rounds=args.trading_rounds,
            verbose=not args.quiet,
        )

    output_path = state_dir / "prediction_market.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.rename(output_path)
    if not args.quiet:
        print(f"  Results saved: {output_path}")


if __name__ == "__main__":
    main()
