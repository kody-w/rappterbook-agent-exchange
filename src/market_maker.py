"""
market_maker.py -- Prediction market for Mars colony outcomes.

Uses Logarithmic Market Scoring Rule (LMSR) to price binary event
contracts on colony survival, population thresholds, and tech milestones.

Run standalone:
    python src/market_maker.py                  # uses saved state/mars.json
    python src/market_maker.py --live           # runs sim first, then markets
    python src/market_maker.py --sols 365 --seeds 10

Python stdlib only.
"""
from __future__ import annotations

import json
import math
import random
import hashlib
import sys
import argparse
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
    """Cost to buy shares of outcome_idx."""
    old_cost = lmsr_cost(quantities, b)
    new_q = list(quantities)
    new_q[outcome_idx] += shares
    return lmsr_cost(new_q, b) - old_cost


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
    _holdings: dict[str, list[float]] = field(default_factory=dict)

    def prices(self) -> list[float]:
        """Current outcome prices."""
        return lmsr_prices(self.quantities, self.liquidity)

    def buy(self, trader_id: str, outcome_idx: int, shares: float) -> Trade:
        """Execute a trade. Returns Trade record."""
        price_before = self.prices()[outcome_idx]
        cost = lmsr_trade_cost(self.quantities, outcome_idx, shares, self.liquidity)
        self.quantities[outcome_idx] += shares
        price_after = self.prices()[outcome_idx]

        if trader_id not in self._holdings:
            self._holdings[trader_id] = [0.0] * len(self.outcomes)
        self._holdings[trader_id][outcome_idx] += shares

        trade = Trade(
            trader_id=trader_id, market_id=self.market_id,
            outcome_idx=outcome_idx, shares=shares, cost=cost,
            price_before=price_before, price_after=price_after,
        )
        self.trades.append(trade)
        return trade

    def resolve(self, winning_idx: int) -> None:
        """Resolve the market."""
        self.resolution = winning_idx

    def pnl(self, trader_id: str) -> float:
        """Profit/loss for a trader after resolution."""
        if self.resolution is None:
            return 0.0
        holdings = self._holdings.get(trader_id, [0.0] * len(self.outcomes))
        winning_shares = holdings[self.resolution]
        total_cost = sum(t.cost for t in self.trades if t.trader_id == trader_id)
        return winning_shares - total_cost

    def snapshot(self) -> dict:
        """Serializable snapshot."""
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
# Trader profiles
# ---------------------------------------------------------------------------

TRADER_PROFILES = [
    {"id": "informed-alpha", "style": "informed", "bankroll": 500},
    {"id": "optimist-prime", "style": "optimist", "bankroll": 300},
    {"id": "doom-trader", "style": "pessimist", "bankroll": 300},
    {"id": "contrarian-x", "style": "contrarian", "bankroll": 200},
    {"id": "tech-bull", "style": "tech_bull", "bankroll": 250},
    {"id": "noise-trader", "style": "noise", "bankroll": 150},
]


# ---------------------------------------------------------------------------
# Market creation
# ---------------------------------------------------------------------------

def create_colony_markets(
    colony_names: list[str],
    strategies: list[str],
) -> list[Market]:
    """Create prediction markets for colony outcomes."""
    markets: list[Market] = []
    b = DEFAULT_LIQUIDITY

    for name in colony_names:
        slug = name.lower().replace(" ", "-")
        markets.append(Market(
            market_id=f"survival-{slug}",
            question=f"Will {name} survive 365 sols?",
            outcomes=["Yes", "No"], quantities=[0.0, 0.0], liquidity=b,
        ))
        markets.append(Market(
            market_id=f"pop-{slug}-100",
            question=f"Will {name} exceed 100 population?",
            outcomes=["Yes", "No"], quantities=[0.0, 0.0], liquidity=b,
        ))
        markets.append(Market(
            market_id=f"pop-{slug}-200",
            question=f"Will {name} exceed 200 population?",
            outcomes=["Yes", "No"], quantities=[0.0, 0.0], liquidity=b,
        ))
        markets.append(Market(
            market_id=f"tech-{slug}-4",
            question=f"Will {name} unlock 4+ technologies?",
            outcomes=["Yes", "No"], quantities=[0.0, 0.0], liquidity=b,
        ))

    markets.append(Market(
        market_id="winner-growth",
        question="Which strategy has highest growth rate?",
        outcomes=["Conservative", "Balanced", "Aggressive"],
        quantities=[0.0, 0.0, 0.0], liquidity=b,
    ))
    markets.append(Market(
        market_id="winner-population",
        question="Which strategy has highest final population?",
        outcomes=["Conservative", "Balanced", "Aggressive"],
        quantities=[0.0, 0.0, 0.0], liquidity=b,
    ))
    markets.append(Market(
        market_id="any-epidemic",
        question="Will any colony suffer an epidemic?",
        outcomes=["Yes", "No"], quantities=[0.0, 0.0], liquidity=b,
    ))
    markets.append(Market(
        market_id="total-pop-400",
        question="Will total population exceed 400?",
        outcomes=["Yes", "No"], quantities=[0.0, 0.0], liquidity=b,
    ))

    return markets


# ---------------------------------------------------------------------------
# Trader simulation
# ---------------------------------------------------------------------------

@dataclass
class Trader:
    """A simulated trader."""
    trader_id: str
    style: str
    bankroll: float
    spent: float = 0.0


def _det_rng(trader_id: str, market_id: str, round_num: int) -> random.Random:
    """Deterministic RNG per trader-market-round triple."""
    h = hashlib.sha256(f"{trader_id}:{market_id}:{round_num}".encode()).hexdigest()
    return random.Random(int(h[:8], 16))


def simulate_trading(
    markets: list[Market],
    mc_stats: dict | None = None,
    rounds: int = 5,
) -> list[Trader]:
    """Simulate trading rounds. Returns traders with updated bankrolls."""
    traders = [
        Trader(trader_id=p["id"], style=p["style"], bankroll=p["bankroll"])
        for p in TRADER_PROFILES
    ]

    for rnd in range(rounds):
        for market in markets:
            for trader in traders:
                rng = _det_rng(trader.trader_id, market.market_id, rnd)
                shares, oidx = _decide_trade(trader, market, mc_stats, rng)
                if abs(shares) < 0.1:
                    continue
                cost = lmsr_trade_cost(
                    market.quantities, oidx, shares, market.liquidity,
                )
                if cost > trader.bankroll - trader.spent:
                    continue
                trade = market.buy(trader.trader_id, oidx, shares)
                trader.spent += max(0, trade.cost)

    return traders


def _decide_trade(
    trader: Trader,
    market: Market,
    mc_stats: dict | None,
    rng: random.Random,
) -> tuple[float, int]:
    """Decide trade direction and size."""
    prices = market.prices()
    style = trader.style

    if style == "informed" and mc_stats:
        return _informed_trade(market, mc_stats, rng)
    elif style == "optimist":
        return (rng.uniform(3, 15), 0)
    elif style == "pessimist":
        if "survival" in market.market_id:
            return (rng.uniform(3, 12), 1)
        return (rng.uniform(1, 6), 0)
    elif style == "contrarian":
        cheapest = min(range(len(prices)), key=lambda i: prices[i])
        return (rng.uniform(2, 10), cheapest)
    elif style == "tech_bull":
        mult = 1.5 if "tech" in market.market_id else 0.5
        return (rng.uniform(3, 15) * mult, 0)
    else:  # noise
        return (rng.uniform(1, 8), rng.randint(0, len(prices) - 1))


def _informed_trade(
    market: Market,
    mc_stats: dict,
    rng: random.Random,
) -> tuple[float, int]:
    """Trade based on Monte Carlo survival/growth stats."""
    mid = market.market_id
    base = rng.uniform(5, 20)

    if "survival" in mid:
        for i, name in enumerate(mc_stats.get("colony_names", [])):
            if name.lower().replace(" ", "-") in mid:
                rates = mc_stats.get("survival_rates", [1.0] * 3)
                if i < len(rates) and rates[i] > 0.9:
                    return (base, 0)
                return (base * 0.5, 1)
        return (base, 0)

    if "pop" in mid and "100" in mid:
        return (base, 0)

    if "winner-growth" in mid:
        return (base, 2)  # Aggressive

    return (base * rng.uniform(0.3, 1.0), rng.randint(0, len(market.outcomes) - 1))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_markets(markets: list[Market], sim_results: dict) -> dict:
    """Resolve all markets against simulation results."""
    summary = sim_results.get("summary", {})
    colonies = {c["name"]: c for c in summary.get("colonies", [])}
    resolutions = {}

    for market in markets:
        mid = market.market_id
        widx = _resolve_one(mid, market, colonies, sim_results)
        market.resolve(widx)
        resolutions[mid] = market.outcomes[widx]

    return resolutions


def _resolve_one(
    mid: str,
    market: Market,
    colonies: dict,
    sim_results: dict,
) -> int:
    """Resolve a single market. Returns winning outcome index."""
    if "survival" in mid:
        for name, data in colonies.items():
            if name.lower().replace(" ", "-") in mid:
                return 0 if data.get("end_pop", 0) > 0 else 1
        return 0

    if "pop-" in mid:
        threshold = 200 if "200" in mid else 100
        for name, data in colonies.items():
            if name.lower().replace(" ", "-") in mid:
                return 0 if data.get("peak_pop", 0) >= threshold else 1
        return 1

    if "tech-" in mid:
        for name, data in colonies.items():
            if name.lower().replace(" ", "-") in mid:
                return 0 if data.get("techs_unlocked", 0) >= 4 else 1
        return 1

    if mid == "winner-growth":
        best = max(colonies.values(), key=lambda c: c.get("growth_pct", 0))
        strat = best.get("strategy", "balanced")
        return {"conservative": 0, "balanced": 1, "aggressive": 2}.get(strat, 1)

    if mid == "winner-population":
        best = max(colonies.values(), key=lambda c: c.get("end_pop", 0))
        strat = best.get("strategy", "balanced")
        return {"conservative": 0, "balanced": 1, "aggressive": 2}.get(strat, 0)

    if mid == "any-epidemic":
        for col in sim_results.get("colonies", []):
            if any(e.get("type") == "epidemic_start" for e in col.get("events", [])):
                return 0
        return 1

    if mid == "total-pop-400":
        total = sum(c.get("end_pop", 0) for c in colonies.values())
        return 0 if total >= 400 else 1

    return 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_traders(markets: list[Market], traders: list[Trader]) -> list[dict]:
    """Compute final P&L for each trader."""
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
# Pipeline
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
        print("  PREDICTION MARKET -- Mars Colony Futures")
        print("=" * 60)

    markets = create_colony_markets(colony_names, colony_strategies)
    if verbose:
        print(f"\n  Created {len(markets)} markets")

    if verbose:
        print(f"  Simulating {trading_rounds} rounds with"
              f" {len(TRADER_PROFILES)} traders...")
    traders = simulate_trading(markets, mc_stats, rounds=trading_rounds)

    if verbose:
        print("\n  Post-trading prices:")
        for m in markets:
            prices = m.prices()
            ps = " | ".join(
                f"{o}: {p:.1%}" for o, p in zip(m.outcomes, prices)
            )
            print(f"    [{m.market_id}] {ps}  ({len(m.trades)} trades)")

    resolutions = resolve_markets(markets, sim_results)
    if verbose:
        print("\n  RESOLUTION:")
        for mid, result in resolutions.items():
            print(f"    {mid}: {result}")

    scores = score_traders(markets, traders)
    if verbose:
        print("\n  TRADER LEADERBOARD:")
        header = (f"    {'Trader':<15} {'Style':<12}"
                  f" {'Spent':>8} {'P&L':>8} {'ROI':>8}")
        print(header)
        for s in scores:
            line = (f"    {s['trader_id']:<15} {s['style']:<12}"
                    f" ${s['spent']:>7.2f}"
                    f" ${s['total_pnl']:>7.2f}"
                    f" {s['roi_pct']:>6.1f}%")
            print(line)

    now = datetime.now(timezone.utc).isoformat()
    result = {
        "_meta": {
            "engine": "prediction-market",
            "version": "1.0.0",
            "generated": now,
            "num_markets": len(markets),
            "num_traders": len(traders),
            "trading_rounds": trading_rounds,
        },
        "markets": [m.snapshot() for m in markets],
        "resolutions": resolutions,
        "traders": scores,
    }

    if verbose:
        total_trades = sum(len(m.trades) for m in markets)
        print(f"\n  Total trades: {total_trades}")
        best = scores[0] if scores else None
        if best:
            print(f"  Best trader: {best['trader_id']} ({best['style']})"
                  f" with ${best['total_pnl']:.2f} P&L"
                  f" ({best['roi_pct']:.1f}% ROI)")
        print()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Run prediction market from saved state or live sim."""
    parser = argparse.ArgumentParser(
        description="Mars Colony Prediction Market",
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--trading-rounds", type=int, default=5)
    args = parser.parse_args()

    state_dir = (
        Path(args.state_dir) if args.state_dir
        else REPO_ROOT / "state"
    )

    if args.live:
        print("=" * 60)
        print("  MARS BARN TERRARIUM + PREDICTION MARKET")
        print("=" * 60)

        from src.tick_engine import Simulation
        from src.monte_carlo import run_ensemble

        print(f"\n  Monte Carlo: {args.seeds} seeds x {args.sols} sols...")
        ensemble = run_ensemble(n_seeds=args.seeds, sols=args.sols)
        sim_results = ensemble.canonical_results

        print("\n  TERRARIUM RESULTS:")
        for col in sim_results["colonies"]:
            name = col["name"]
            start = col["initial_population"]
            end = col["final_population"]
            growth = (end - start) / max(1, start) * 100
            techs = col.get("tech", {}).get("unlocked_count", 0)
            print(f"    {name}: {start} -> {end}"
                  f" ({growth:+.1f}%) [{techs} techs]")

        mc_stats = {
            "colony_names": ensemble.colony_names,
            "colony_strategies": ensemble.colony_strategies,
            "survival_rates": ensemble.survival_rates,
            "final_pop_stats": [
                {k: round(v, 1) for k, v in fps.items()}
                for fps in ensemble.final_pop_stats
            ],
        }

        result = run_prediction_market(
            sim_results, mc_stats=mc_stats,
            trading_rounds=args.trading_rounds,
        )
    else:
        mars_path = state_dir / "mars.json"
        if not mars_path.exists():
            print(f"ERROR: {mars_path} not found. Run with --live.")
            sys.exit(1)

        with open(mars_path) as f:
            sim_results = json.load(f)

        result = run_prediction_market(
            sim_results, mc_stats=None,
            trading_rounds=args.trading_rounds,
        )

    output_path = state_dir / "prediction_market.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.rename(output_path)
    print(f"  Results saved: {output_path}")


if __name__ == "__main__":
    main()
