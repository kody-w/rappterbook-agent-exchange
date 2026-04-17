"""
Economics organ for Mars-100 colony simulation.

Models personal wealth, labor markets, bilateral trade, taxation, and
economic inequality.  The colony starts as a communal economy; individual
wallets emerge as colonists accumulate surplus from skilled labour.

Key dynamics:
  - Labor income: after colony resource tick, skilled actions divert a
    fraction of realized surplus INTO personal wallets (conservation: wealth
    comes from colony pool, not conjured)
  - Trade: bilateral exchanges between colonists with mutual trust, matching
    surplus to deficit
  - Taxation: governance type sets tax rate; taxes move personal wealth back
    to colony pool
  - Inequality: Gini coefficient tracked each year; high Gini erodes
    social cohesion and can trigger economic revolt (governance re-election)
  - Economic pressure: wallet contents influence action selection weights

Engine v7.0.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist, SKILL_NAMES
from src.mars100.colony import RESOURCE_NAMES, Resources, SocialGraph

# -- constants ---------------------------------------------------------------

LABOR_INCOME_FRACTION = 0.15   # fraction of action's resource bonus diverted to wallet
TRADE_TRUST_THRESHOLD = 0.30   # min(trust_ab, trust_ba) needed for trade
TRADE_MAX_PER_YEAR = 5         # max trades executed per year
TRADE_VOLUME_SCALE = 0.05      # max per-trade volume scaling factor

TAX_RATES: dict[str, float] = {
    "dictator": 0.40,
    "ai_governor": 0.30,
    "council": 0.25,
    "consensus": 0.20,
    "lottery": 0.15,
    "anarchy": 0.05,
}
DEFAULT_TAX_RATE = 0.20

GINI_UNREST_THRESHOLD = 0.60   # cohesion penalty kicks in
GINI_REVOLT_THRESHOLD = 0.70   # chance of forced governance re-election
REVOLT_PROBABILITY = 0.40      # per-year chance when Gini > revolt threshold

IMMIGRANT_ENDOWMENT = 0.02     # small resource grant per wallet slot


# -- data classes ------------------------------------------------------------


@dataclass
class TradeRecord:
    """A single bilateral trade."""
    year: int
    buyer_id: str
    seller_id: str
    resource_given: str
    amount_given: float
    resource_received: str
    amount_received: float

    def to_dict(self) -> dict:
        return {
            "year": self.year, "buyer_id": self.buyer_id,
            "seller_id": self.seller_id,
            "resource_given": self.resource_given,
            "amount_given": round(self.amount_given, 6),
            "resource_received": self.resource_received,
            "amount_received": round(self.amount_received, 6),
        }


@dataclass
class EconomicState:
    """Colony-wide economic tracking."""
    gini_history: list[float] = field(default_factory=list)
    total_trades: int = 0
    total_volume: float = 0.0
    pending_revolt: bool = False
    revolution_count: int = 0
    total_tax_collected: float = 0.0

    def to_dict(self) -> dict:
        return {
            "gini_history": [round(g, 4) for g in self.gini_history[-20:]],
            "current_gini": round(self.gini_history[-1], 4) if self.gini_history else 0.0,
            "total_trades": self.total_trades,
            "total_volume": round(self.total_volume, 4),
            "pending_revolt": self.pending_revolt,
            "revolution_count": self.revolution_count,
            "total_tax_collected": round(self.total_tax_collected, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> EconomicState:
        if not d:
            return cls()
        return cls(
            gini_history=d.get("gini_history", []),
            total_trades=d.get("total_trades", 0),
            total_volume=d.get("total_volume", 0.0),
            pending_revolt=d.get("pending_revolt", False),
            revolution_count=d.get("revolution_count", 0),
            total_tax_collected=d.get("total_tax_collected", 0.0),
        )


@dataclass
class EconomicTickResult:
    """Result of one year of economic activity."""
    year: int
    gini: float
    trades: list[dict]
    tax_collected: float
    labor_income_total: float
    revolt_triggered: bool = False
    estates_liquidated: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "year": self.year, "gini": round(self.gini, 4),
            "trades": self.trades,
            "tax_collected": round(self.tax_collected, 4),
            "labor_income_total": round(self.labor_income_total, 4),
            "revolt_triggered": self.revolt_triggered,
            "estates_liquidated": self.estates_liquidated,
        }


# -- core functions ----------------------------------------------------------

ACTION_RESOURCE_MAP: dict[str, str] = {
    "terraform": "water",
    "farm": "food",
    "code": "power",
    "pray": "medicine",
    "explore": "air",
}

ACTION_SKILL_MAP: dict[str, str] = {
    "terraform": "terraforming",
    "farm": "hydroponics",
    "code": "coding",
    "pray": "prayer",
    "explore": "terraforming",
}


def allocate_labor_income(
    colonist: Colonist,
    action: str,
    resources: Resources,
    rng: random.Random,
) -> float:
    """Divert a fraction of action output from colony pool to personal wallet.

    Returns the total amount diverted (for tracking).  The colony pool is
    reduced by the same amount — conservation is maintained.
    """
    resource_name = ACTION_RESOURCE_MAP.get(action)
    if resource_name is None:
        return 0.0

    skill_name = ACTION_SKILL_MAP.get(action, "")
    skill_val = getattr(colonist.skills, skill_name, 0.0) if skill_name else 0.0

    # Income scales with skill level and a small random factor
    base_income = LABOR_INCOME_FRACTION * skill_val
    noise = rng.uniform(0.8, 1.2)
    income = base_income * noise

    # Don't take more than the colony has
    available = getattr(resources, resource_name)
    income = min(income, available * 0.1)  # cap at 10% of colony stock
    income = max(0.0, income)

    if income > 0:
        setattr(resources, resource_name, available - income)
        colonist.wallet.deposit(resource_name, income)
        colonist.wallet.total_earned += income

    return income


def find_trades(
    colonists: list[Colonist],
    social: SocialGraph,
    year: int,
    rng: random.Random,
) -> list[TradeRecord]:
    """Find mutually beneficial trades between colonists.

    A trade happens when:
    - min(trust A→B, trust B→A) > TRADE_TRUST_THRESHOLD
    - A has surplus in resource X (above median wealth in X)
    - B has deficit in resource X (below median wealth in X)
    - B has surplus in some resource Y that A wants
    """
    active = [c for c in colonists if c.is_active()]
    if len(active) < 2:
        return []

    # Compute median holdings per resource
    medians: dict[str, float] = {}
    for res in RESOURCE_NAMES:
        vals = sorted(c.wallet.holdings.get(res, 0.0) for c in active)
        mid = len(vals) // 2
        medians[res] = vals[mid] if vals else 0.0

    trades: list[TradeRecord] = []
    pairs = [(a, b) for i, a in enumerate(active)
             for b in active[i + 1:]]
    rng.shuffle(pairs)

    for a, b in pairs:
        if len(trades) >= TRADE_MAX_PER_YEAR:
            break

        trust_ab = social.get(a.id, b.id).trust
        trust_ba = social.get(b.id, a.id).trust
        mutual_trust = min(trust_ab, trust_ba)
        if mutual_trust < TRADE_TRUST_THRESHOLD:
            continue

        # Find what A has surplus, B has deficit
        a_surplus = [r for r in RESOURCE_NAMES
                     if a.wallet.holdings.get(r, 0.0) > medians[r] + 0.001]
        b_deficit = [r for r in RESOURCE_NAMES
                     if b.wallet.holdings.get(r, 0.0) < medians[r]]
        # And vice versa
        b_surplus = [r for r in RESOURCE_NAMES
                     if b.wallet.holdings.get(r, 0.0) > medians[r] + 0.001]
        a_deficit = [r for r in RESOURCE_NAMES
                     if a.wallet.holdings.get(r, 0.0) < medians[r]]

        # Match: A gives something B needs, B gives something A needs
        a_gives = set(a_surplus) & set(b_deficit)
        b_gives = set(b_surplus) & set(a_deficit)

        if not a_gives or not b_gives:
            continue

        res_a = rng.choice(sorted(a_gives))
        res_b = rng.choice(sorted(b_gives))

        avg_affection = (social.get(a.id, b.id).affection +
                         social.get(b.id, a.id).affection) / 2
        volume = TRADE_VOLUME_SCALE * mutual_trust * avg_affection

        amount_a = a.wallet.withdraw(res_a, volume)
        amount_b = b.wallet.withdraw(res_b, volume)

        if amount_a > 0 and amount_b > 0:
            b.wallet.deposit(res_a, amount_a)
            a.wallet.deposit(res_b, amount_b)
            a.wallet.total_traded += amount_a
            b.wallet.total_traded += amount_b

            trades.append(TradeRecord(
                year=year, buyer_id=b.id, seller_id=a.id,
                resource_given=res_a, amount_given=amount_a,
                resource_received=res_b, amount_received=amount_b,
            ))
        else:
            # Return partial withdrawals
            if amount_a > 0:
                a.wallet.deposit(res_a, amount_a)
            if amount_b > 0:
                b.wallet.deposit(res_b, amount_b)

    return trades


def apply_taxation(
    colonists: list[Colonist],
    gov_type: str,
    resources: Resources,
) -> float:
    """Tax personal wealth and return it to colony pool.

    Returns total tax collected.
    """
    rate = TAX_RATES.get(gov_type, DEFAULT_TAX_RATE)
    total_collected = 0.0
    active = [c for c in colonists if c.is_active()]

    for colonist in active:
        for res in RESOURCE_NAMES:
            held = colonist.wallet.holdings.get(res, 0.0)
            if held <= 0:
                continue
            tax = held * rate
            actual = colonist.wallet.withdraw(res, tax)
            colonist.wallet.total_taxed += actual
            # Return to colony pool
            current = getattr(resources, res)
            setattr(resources, res, min(1.0, current + actual))
            total_collected += actual

    return total_collected


def compute_gini(colonists: list[Colonist]) -> float:
    """Compute Gini coefficient of wealth distribution.

    Returns 0.0 (perfect equality) to 1.0 (perfect inequality).
    With 0 or 1 colonists, returns 0.0.
    """
    active = [c for c in colonists if c.is_active()]
    if len(active) <= 1:
        return 0.0

    wealths = sorted(c.wallet.total_wealth() for c in active)
    n = len(wealths)
    total = sum(wealths)

    if total == 0:
        return 0.0

    numerator = sum((2 * (i + 1) - n - 1) * w for i, w in enumerate(wealths))
    return numerator / (n * total)


def compute_economic_pressure(colonist: Colonist) -> dict[str, float]:
    """Compute action-weight deltas based on colonist's wallet.

    Rich colonists lean toward hoarding; poor colonists lean toward
    cooperative and productive actions.
    """
    wealth = colonist.wallet.total_wealth()
    pressure: dict[str, float] = {}

    if wealth > 0.15:
        pressure["hoard"] = min(0.5, wealth * 0.8)
        pressure["cooperate"] = -0.1
    elif wealth < 0.02:
        pressure["cooperate"] = 0.3
        pressure["farm"] = 0.2
        pressure["hoard"] = -0.2

    return pressure


def liquidate_estate(
    colonist: Colonist,
    resources: Resources,
) -> dict:
    """Liquidate a dead or exiled colonist's wallet back to colony pool.

    Dead colonists' estates go to the commons. Exiled colonists lose
    everything (it's lost to the wasteland).
    """
    estate: dict[str, float] = {}
    for res in RESOURCE_NAMES:
        held = colonist.wallet.withdraw(res, colonist.wallet.holdings.get(res, 0.0))
        if held > 0:
            estate[res] = held
            if colonist.alive is False:
                # Dead: estate returns to colony
                current = getattr(resources, res)
                setattr(resources, res, min(1.0, current + held))
            # Exiled: wealth is lost (already withdrawn)

    return estate


def endow_immigrant(colonist: Colonist) -> None:
    """Give a new immigrant a small resource endowment."""
    for res in RESOURCE_NAMES:
        colonist.wallet.deposit(res, IMMIGRANT_ENDOWMENT)
        colonist.wallet.total_earned += IMMIGRANT_ENDOWMENT


def tick_economics(
    colonists: list[Colonist],
    actions: dict[str, str],
    gov_type: str,
    social: SocialGraph,
    resources: Resources,
    economic_state: EconomicState,
    year: int,
    rng: random.Random,
) -> EconomicTickResult:
    """Run one year of economic activity.

    Called AFTER resource tick / infra costs, so colony pool reflects
    realized production.  Returns result for logging.
    """
    active = [c for c in colonists if c.is_active()]

    # 1. Labor income: divert from colony pool to personal wallets
    total_income = 0.0
    for colonist in active:
        action = actions.get(colonist.id, "rest")
        income = allocate_labor_income(colonist, action, resources, rng)
        total_income += income

    # 2. Trade: bilateral exchanges
    trades = find_trades(active, social, year, rng)
    for trade in trades:
        economic_state.total_trades += 1
        economic_state.total_volume += trade.amount_given + trade.amount_received

    # 3. Taxation: move personal wealth back to colony
    tax_collected = apply_taxation(colonists, gov_type, resources)
    economic_state.total_tax_collected += tax_collected

    # 4. Compute inequality
    gini = compute_gini(colonists)
    economic_state.gini_history.append(gini)

    # 5. Check for economic revolt
    revolt = False
    if gini > GINI_REVOLT_THRESHOLD and rng.random() < REVOLT_PROBABILITY:
        economic_state.pending_revolt = True
        economic_state.revolution_count += 1
        revolt = True

    resources.clamp()

    return EconomicTickResult(
        year=year, gini=gini,
        trades=[t.to_dict() for t in trades],
        tax_collected=tax_collected,
        labor_income_total=total_income,
        revolt_triggered=revolt,
    )
