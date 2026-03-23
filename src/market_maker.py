"""
market_maker.py -- LMSR prediction market for Mars colony outcomes.

Uses Logarithmic Market Scoring Rule (Hanson 2003) to price binary
event contracts on colony survival, population, tech milestones.

Usage:
    python src/market_maker.py                # uses saved state/mars.json
    python src/market_maker.py --live          # runs sim first
    python src/market_maker.py --live --sols 365 --seeds 10

Python stdlib only. Zero dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# LMSR (Logarithmic Market Scoring Rule)
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
    quantities: list[float], outcome_idx: int,
    shares: float, b: float = DEFAULT_LIQUIDITY,
) -> float:
    """Cost to buy shares of outcome. Positive = cost, negative = refund."""
    old_cost = lmsr_cost(quantities, b)
    new_q = list(quantities)
    new_q[outcome_idx] += shares
    return lmsr_cost(new_q, b) - old_cost


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
    """A binary prediction market with LMSR pricing."""
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
                self.outcomes[self.resolution] if self.resolution is not None else None
            ),
        }


def create_colony_markets(
    colony_names: list[str], strategies: list[str],
) -> list[Market]:
    """Create prediction markets for colony outcomes."""
    markets: list[Market] = []
    for name in colony_names:
        slug = name.lower().replace(" ", "-")
        markets.append(Market(
            market_id="survival-" + slug,
            question="Will " + name + " survive 365 sols?",
            outcomes=["Yes", "No"], quantities=[0.0, 0.0],
            liquidity=DEFAULT_LIQUIDITY,
        ))
    thresholds = [100, 150, 200]
    for name in colony_names:
        slug = name.lower().replace(" ", "-")
        for thr in thresholds:
            markets.append(Market(
                market_id="pop-" + slug + "-" + str(thr),
                question="Will " + name + " exceed " + str(thr) + " pop by sol 365?",
                outcomes=["Yes", "No"], quantities=[0.0, 0.0],
                liquidity=DEFAULT_LIQUIDITY,
            ))
    markets.append(Market(
        market_id="fastest-grower",
        question="Which colony will have highest growth %?",
        outcomes=colony_names, quantities=[0.0] * len(colony_names),
        liquidity=DEFAULT_LIQUIDITY,
    ))
    markets.append(Market(
        market_id="largest-pop",
        question="Which colony will have largest final population?",
        outcomes=colony_names, quantities=[0.0] * len(colony_names),
        liquidity=DEFAULT_LIQUIDITY,
    ))
    markets.append(Market(
        market_id="any-epidemic",
        question="Will any colony experience an epidemic?",
        outcomes=["Yes", "No"], quantities=[0.0, 0.0],
        liquidity=DEFAULT_LIQUIDITY,
    ))
    markets.append(Market(
        market_id="tech-leader",
        question="Which colony will unlock the most technologies?",
        outcomes=colony_names, quantities=[0.0] * len(colony_names),
        liquidity=DEFAULT_LIQUIDITY,
    ))
    return markets


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
    trader_id: str
    style: str
    bankroll: float
    spent: float = 0.0

    def can_afford(self, cost: float) -> bool:
        return self.spent + cost <= self.bankroll

    def record_spend(self, cost: float) -> None:
        self.spent += max(0, cost)


def _det_rng(trader_id: str, market_id: str) -> random.Random:
    h = hashlib.sha256((trader_id + ":" + market_id).encode()).hexdigest()
    return random.Random(int(h[:8], 16))


def simulate_trading(
    markets: list[Market], mc_stats: dict | None = None, rounds: int = 5,
) -> list[Trader]:
    traders = [
        Trader(trader_id=p["id"], style=p["style"], bankroll=p["bankroll"])
        for p in TRADER_PROFILES
    ]
    for rd in range(rounds):
        for market in markets:
            prices = market.prices()
            for trader in traders:
                rng = _det_rng(trader.trader_id, market.market_id + ":" + str(rd))
                shares, oidx = _decide_trade(trader, market, prices, mc_stats, rng)
                if shares <= 0:
                    continue
                cost = lmsr_trade_cost(market.quantities, oidx, shares, market.liquidity)
                if cost > 0 and not trader.can_afford(cost):
                    shares = max(1, shares // 2)
                    cost = lmsr_trade_cost(market.quantities, oidx, shares, market.liquidity)
                    if cost > 0 and not trader.can_afford(cost):
                        continue
                market.buy(oidx, shares, trader.trader_id, sol=rd)
                if cost > 0:
                    trader.record_spend(cost)
    return traders


def _decide_trade(trader, market, prices, mc_stats, rng):
    mid = market.market_id
    if trader.style == "informed" and mc_stats:
        return _informed_trade(market, mc_stats, rng)
    elif trader.style == "optimist":
        if "survive" in market.question.lower() or "exceed" in market.question.lower():
            return (rng.randint(3, 10), 0)
        return (rng.randint(2, 6), rng.randint(0, len(market.outcomes) - 1))
    elif trader.style == "pessimist":
        if "survive" in market.question.lower():
            return (rng.randint(3, 10), 1)
        if "exceed" in market.question.lower():
            return (rng.randint(2, 8), 1)
        return (rng.randint(2, 5), rng.randint(0, len(market.outcomes) - 1))
    elif trader.style == "contrarian":
        cheapest = min(range(len(prices)), key=lambda i: prices[i])
        return (rng.randint(3, 8), cheapest)
    elif trader.style == "tech_bull":
        if "tech" in market.question.lower():
            return (rng.randint(5, 12), min(2, len(market.outcomes) - 1))
        return (rng.randint(1, 4), rng.randint(0, len(market.outcomes) - 1))
    else:
        return (rng.randint(1, 5), rng.randint(0, len(market.outcomes) - 1))


def _informed_trade(market, mc_stats, rng):
    mid = market.market_id
    colony_names = mc_stats.get("colony_names", [])
    survival_rates = mc_stats.get("survival_rates", [])
    growth_stats = mc_stats.get("growth_pct_stats", [])
    final_pop_stats = mc_stats.get("final_pop_stats", [])

    if mid.startswith("survival-"):
        slug = mid.replace("survival-", "")
        for i, name in enumerate(colony_names):
            if name.lower().replace(" ", "-") == slug and i < len(survival_rates):
                return (rng.randint(5, 15), 0) if survival_rates[i] > 0.7 else (rng.randint(5, 15), 1)
        return (0, 0)
    if mid.startswith("pop-"):
        parts = mid.split("-")
        threshold = int(parts[-1])
        slug = "-".join(parts[1:-1])
        for i, name in enumerate(colony_names):
            if name.lower().replace(" ", "-") == slug and i < len(final_pop_stats):
                mean_fp = final_pop_stats[i].get("mean", 0)
                return (rng.randint(3, 12), 0) if mean_fp > threshold else (rng.randint(3, 12), 1)
        return (0, 0)
    if mid == "fastest-grower" and growth_stats:
        best = max(range(len(growth_stats)), key=lambda j: growth_stats[j].get("mean", 0))
        return (rng.randint(5, 15), best)
    if mid == "largest-pop" and final_pop_stats:
        best = max(range(len(final_pop_stats)), key=lambda j: final_pop_stats[j].get("mean", 0))
        return (rng.randint(5, 15), best)
    if mid == "tech-leader":
        return (rng.randint(3, 8), 0)
    return (rng.randint(2, 8), rng.randint(0, len(market.outcomes) - 1))


def resolve_markets(markets: list[Market], sim_results: dict) -> dict:
    """Resolve all markets against simulation results."""
    colonies = sim_results.get("colonies", [])
    colony_data = {}
    for i, col in enumerate(colonies):
        name = col["name"]
        slug = name.lower().replace(" ", "-")
        fp = col.get("final_population", 0)
        ip = col.get("initial_population", 1)
        techs = col.get("tech", {}).get("unlocked_count", 0)
        events = col.get("events", [])
        had_epi = any(e.get("type", "").startswith("epidemic") for e in events)
        colony_data[slug] = {
            "name": name, "idx": i, "final_pop": fp,
            "growth_pct": (fp - ip) / max(1, ip) * 100,
            "techs": techs, "had_epidemic": had_epi, "survived": fp > 0,
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
        elif mid == "fastest-grower" and colony_data:
            fastest = max(colony_data.values(), key=lambda x: x["growth_pct"])
            market.resolve(fastest["idx"])
            resolutions[mid] = fastest["name"]
        elif mid == "largest-pop" and colony_data:
            largest = max(colony_data.values(), key=lambda x: x["final_pop"])
            market.resolve(largest["idx"])
            resolutions[mid] = largest["name"]
        elif mid == "any-epidemic":
            market.resolve(0 if any_epidemic else 1)
            resolutions[mid] = "Yes" if any_epidemic else "No"
        elif mid == "tech-leader" and colony_data:
            leader = max(colony_data.values(), key=lambda x: x["techs"])
            market.resolve(leader["idx"])
            resolutions[mid] = leader["name"]
    return resolutions


def score_traders(markets: list[Market], traders: list[Trader]) -> list[dict]:
    scores = []
    for trader in traders:
        total_pnl = 0.0
        mpnls = {}
        for m in markets:
            pnl = m.pnl(trader.trader_id)
            total_pnl += pnl
            if abs(pnl) > 0.01:
                mpnls[m.market_id] = round(pnl, 2)
        scores.append({
            "trader_id": trader.trader_id, "style": trader.style,
            "bankroll": trader.bankroll, "spent": round(trader.spent, 2),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(total_pnl / max(trader.spent, 1) * 100, 1),
            "market_pnls": mpnls,
        })
    scores.sort(key=lambda x: -x["total_pnl"])
    return scores


def run_prediction_market(
    sim_results: dict, mc_stats: dict | None = None,
    trading_rounds: int = 5, verbose: bool = True,
) -> dict:
    """Full pipeline: create -> trade -> resolve -> score."""
    colonies = sim_results.get("colonies", [])
    colony_names = [c["name"] for c in colonies]
    colony_strategies = [c.get("strategy", "balanced") for c in colonies]

    markets = create_colony_markets(colony_names, colony_strategies)
    if verbose:
        print("\n  Created " + str(len(markets)) + " markets")

    traders = simulate_trading(markets, mc_stats, rounds=trading_rounds)
    if verbose:
        total_trades = sum(len(m.trades) for m in markets)
        print("  " + str(total_trades) + " trades across " + str(trading_rounds) + " rounds")

    resolutions = resolve_markets(markets, sim_results)
    scores = score_traders(markets, traders)

    if verbose:
        print("\n  RESOLUTIONS:")
        for mid, result in resolutions.items():
            print("    " + mid + ": " + str(result))
        print("\n  TRADER LEADERBOARD:")
        hdr = "    %-15s %-12s %8s %8s %8s" % ("Trader", "Style", "Spent", "P&L", "ROI")
        print(hdr)
        print("    " + "-" * 55)
        for s in scores:
            row = "    %-15s %-12s $%7.2f $%7.2f %6.1f%%" % (
                s["trader_id"], s["style"], s["spent"], s["total_pnl"], s["roi_pct"])
            print(row)

    now = datetime.now(timezone.utc).isoformat()
    return {
        "_meta": {
            "engine": "prediction-market", "version": "5.0",
            "generated": now, "num_markets": len(markets),
            "num_traders": len(traders), "trading_rounds": trading_rounds,
        },
        "markets": [m.snapshot() for m in markets],
        "resolutions": resolutions,
        "traders": scores,
    }


def main() -> None:
    import argparse
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Mars Colony Prediction Market")
    parser.add_argument("--live", action="store_true", help="Run sim first")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--trading-rounds", type=int, default=5)
    args = parser.parse_args()

    state_dir = Path(args.state_dir) if args.state_dir else repo_root / "state"

    if args.live:
        print("=" * 60)
        print("  MARS BARN TERRARIUM + PREDICTION MARKET")
        print("=" * 60)

        from src.tick_engine import Simulation
        from src.monte_carlo import run_ensemble

        print("\n  Running Monte Carlo: %d seeds x %d sols..." % (args.seeds, args.sols))
        ensemble = run_ensemble(n_seeds=args.seeds, sols=args.sols)
        sim_results = ensemble.canonical_results

        print("\n  TERRARIUM RESULTS:")
        for col in sim_results["colonies"]:
            name = col["name"]
            start = col["initial_population"]
            end = col["final_population"]
            growth = (end - start) / max(1, start) * 100
            techs = col.get("tech", {}).get("unlocked_count", 0)
            print("    %s: %d -> %d (%+.1f%%) [%d techs]" % (name, start, end, growth, techs))

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
            sim_results, mc_stats=mc_stats, trading_rounds=args.trading_rounds,
        )
    else:
        mars_path = state_dir / "mars.json"
        if not mars_path.exists():
            print("ERROR: " + str(mars_path) + " not found. Run with --live first.")
            sys.exit(1)
        with open(mars_path) as f:
            sim_results = json.load(f)
        result = run_prediction_market(
            sim_results, trading_rounds=args.trading_rounds,
        )

    output_path = state_dir / "prediction_market.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.rename(output_path)
    print("\n  Results saved: " + str(output_path))


if __name__ == "__main__":
    main()
