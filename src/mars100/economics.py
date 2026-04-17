"""
Economics organ for Mars-100 colony simulation.

Models individual colonist wealth, trade, taxation, and inequality.
The colony has shared Resources, but colonists also accumulate personal
stockpiles through labour retention.  Economic tension between individual
hoarding and collective welfare drives governance emergence.

Key invariant: labour retention is *deducted* from colony production,
so total wealth is never double-minted.  Transfers between personal
accounts and the colony pool are zero-sum.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist

RESOURCE_KINDS = ("food", "water", "power", "medicine")
MAX_PERSONAL_TOTAL = 0.20
TRADE_TRUST_THRESHOLD = 0.55
TRADE_FRACTION = 0.03
BUREAUCRACY_LOSS = 0.05
BLACK_MARKET_TAX_THRESHOLD = 0.25
BLACK_MARKET_PARANOIA_THRESHOLD = 0.50

TAX_RATES: dict[str, float] = {
    "anarchy": 0.0,
    "council": 0.15,
    "dictator": 0.30,
    "lottery": 0.10,
    "consensus": 0.20,
    "ai_governor": 0.15,
}


@dataclass
class PersonalAccount:
    """A colonist's private resource stores."""
    food: float = 0.0
    water: float = 0.0
    power: float = 0.0
    medicine: float = 0.0

    def total(self) -> float:
        """Sum of all personal stores."""
        return sum(getattr(self, r) for r in RESOURCE_KINDS)

    def to_dict(self) -> dict[str, float]:
        """Serialize to dict."""
        return {r: round(getattr(self, r), 6) for r in RESOURCE_KINDS}

    def clamp(self) -> None:
        """Enforce non-negative values and cap total."""
        for r in RESOURCE_KINDS:
            setattr(self, r, max(0.0, getattr(self, r)))
        total = self.total()
        if total > MAX_PERSONAL_TOTAL:
            scale = MAX_PERSONAL_TOTAL / total
            for r in RESOURCE_KINDS:
                setattr(self, r, getattr(self, r) * scale)


@dataclass
class TradeRecord:
    """Log entry for a bilateral trade."""
    year: int
    giver_id: str
    receiver_id: str
    resource: str
    amount: float
    black_market: bool = False

    def to_dict(self) -> dict:
        """Serialize to dict."""
        d: dict[str, Any] = {
            "year": self.year, "giver": self.giver_id,
            "receiver": self.receiver_id, "resource": self.resource,
            "amount": round(self.amount, 6),
        }
        if self.black_market:
            d["black_market"] = True
        return d


@dataclass
class EconomyState:
    """Colony-wide economic state."""
    accounts: dict[str, PersonalAccount] = field(default_factory=dict)
    trade_log: list[TradeRecord] = field(default_factory=list)
    gini_history: list[float] = field(default_factory=list)
    tax_collected: float = 0.0
    tax_redistributed: float = 0.0
    black_market_active: bool = False

    def get_account(self, colonist_id: str) -> PersonalAccount:
        """Get or create a colonist's account."""
        if colonist_id not in self.accounts:
            self.accounts[colonist_id] = PersonalAccount()
        return self.accounts[colonist_id]

    def remove_account(self, colonist_id: str) -> PersonalAccount | None:
        """Remove and return a colonist's account (death/exile)."""
        return self.accounts.pop(colonist_id, None)

    def to_dict(self) -> dict:
        """Full serialization."""
        return {
            "accounts": {cid: a.to_dict() for cid, a in self.accounts.items()},
            "gini": round(self.gini_history[-1], 4) if self.gini_history else 0.0,
            "gini_history": [round(g, 4) for g in self.gini_history[-20:]],
            "tax_collected": round(self.tax_collected, 4),
            "tax_redistributed": round(self.tax_redistributed, 4),
            "black_market_active": self.black_market_active,
            "recent_trades": [t.to_dict() for t in self.trade_log[-10:]],
        }

    def summary(self) -> dict:
        """Short summary for YearResult."""
        return {
            "gini": round(self.gini_history[-1], 4) if self.gini_history else 0.0,
            "total_wealth": round(
                sum(a.total() for a in self.accounts.values()), 4),
            "black_market": self.black_market_active,
            "trades_this_year": len(self.trade_log),
            "tax_collected": round(self.tax_collected, 4),
        }


# ---------------------------------------------------------------------------
# Labour income — deducted from colony production
# ---------------------------------------------------------------------------

ACTION_INCOME: dict[str, tuple[str, float]] = {
    "farm": ("food", 0.04),
    "terraform": ("water", 0.03),
    "code": ("power", 0.03),
    "pray": ("medicine", 0.01),
    "research": ("power", 0.02),
}


def allocate_labor_income(
    economy: EconomyState,
    colonist: Colonist,
    action: str,
    resources: Any,
    rng: random.Random,
) -> float:
    """Credit a colonist's personal account from their productive action.

    The credited amount is DEDUCTED from colony Resources so no wealth
    is double-minted.  Returns the total amount retained.
    """
    entry = ACTION_INCOME.get(action)
    if entry is None:
        return 0.0
    resource_kind, base_rate = entry
    hoarding_bonus = colonist.stats.hoarding * 0.02
    retention = base_rate + hoarding_bonus + rng.gauss(0, 0.005)
    retention = max(0.0, min(0.08, retention))
    colony_val = getattr(resources, resource_kind, 0.0)
    actual = min(retention, colony_val)
    if actual <= 0:
        return 0.0
    setattr(resources, resource_kind, colony_val - actual)
    account = economy.get_account(colonist.id)
    setattr(account, resource_kind,
            getattr(account, resource_kind) + actual)
    account.clamp()
    return actual


def allocate_hoard_action(
    economy: EconomyState,
    colonist: Colonist,
    resources: Any,
    rng: random.Random,
) -> float:
    """The 'hoard' action: aggressively move colony resources to personal."""
    total_taken = 0.0
    rate = 0.02 + colonist.stats.hoarding * 0.03 + rng.gauss(0, 0.005)
    rate = max(0.0, min(0.06, rate))
    for r in RESOURCE_KINDS:
        colony_val = getattr(resources, r, 0.0)
        take = min(rate, colony_val)
        if take > 0:
            setattr(resources, r, colony_val - take)
            account = economy.get_account(colonist.id)
            setattr(account, r, getattr(account, r) + take)
            total_taken += take
    economy.get_account(colonist.id).clamp()
    return total_taken


# ---------------------------------------------------------------------------
# Taxation
# ---------------------------------------------------------------------------

def collect_taxes(
    economy: EconomyState,
    gov_type: str,
    active_colonists: list[Colonist],
    rng: random.Random,
) -> float:
    """Collect taxes from personal accounts.  Returns total collected."""
    rate = TAX_RATES.get(gov_type, 0.0)
    if rate <= 0:
        economy.tax_collected = 0.0
        return 0.0
    total = 0.0
    for colonist in active_colonists:
        account = economy.get_account(colonist.id)
        for r in RESOURCE_KINDS:
            val = getattr(account, r)
            tax = val * rate
            if economy.black_market_active:
                evasion = colonist.stats.paranoia * 0.4
                tax *= (1.0 - evasion)
            tax = max(0.0, min(val, tax))
            setattr(account, r, val - tax)
            total += tax
    lost = total * BUREAUCRACY_LOSS
    economy.tax_collected = total - lost
    return economy.tax_collected


def redistribute(
    economy: EconomyState,
    gov_type: str,
    active_colonists: list[Colonist],
    leader_id: str | None,
) -> float:
    """Redistribute collected tax revenue.  Returns amount distributed."""
    pool = economy.tax_collected
    if pool <= 0 or not active_colonists:
        economy.tax_redistributed = 0.0
        return 0.0
    distributed = 0.0
    n = len(active_colonists)
    if gov_type == "dictator" and leader_id:
        leader_share = pool * 0.6
        remainder = pool - leader_share
        for c in active_colonists:
            account = economy.get_account(c.id)
            share = (leader_share + remainder / n) if c.id == leader_id else (remainder / n)
            per_kind = share / max(1, len(RESOURCE_KINDS))
            for r in RESOURCE_KINDS:
                setattr(account, r, getattr(account, r) + per_kind)
            account.clamp()
            distributed += share
    else:
        share = pool / n
        for c in active_colonists:
            account = economy.get_account(c.id)
            per_kind = share / max(1, len(RESOURCE_KINDS))
            for r in RESOURCE_KINDS:
                setattr(account, r, getattr(account, r) + per_kind)
            account.clamp()
            distributed += share
    economy.tax_redistributed = distributed
    return distributed


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------

def execute_trades(
    economy: EconomyState,
    social_graph: Any,
    active_colonists: list[Colonist],
    year: int,
    rng: random.Random,
) -> list[TradeRecord]:
    """Execute bilateral trades between colonists.

    Trade is driven by trust (must exceed threshold) and need.
    Trades are zero-sum between accounts.
    """
    trades: list[TradeRecord] = []
    if len(active_colonists) < 2:
        return trades
    for giver in active_colonists:
        for receiver in active_colonists:
            if giver.id == receiver.id:
                continue
            rel = social_graph.get(giver.id, receiver.id)
            trust = rel.trust
            threshold = TRADE_TRUST_THRESHOLD
            if economy.black_market_active:
                threshold *= 0.7
            if trust < threshold:
                continue
            if rng.random() > 0.3:
                continue
            g_acc = economy.get_account(giver.id)
            r_acc = economy.get_account(receiver.id)
            for res in RESOURCE_KINDS:
                g_val = getattr(g_acc, res)
                r_val = getattr(r_acc, res)
                if g_val > 0.02 and r_val < g_val - 0.01:
                    amount = min(TRADE_FRACTION, g_val * 0.5)
                    amount = max(0.0, amount + rng.gauss(0, 0.003))
                    amount = min(amount, g_val)
                    if amount <= 0:
                        continue
                    setattr(g_acc, res, g_val - amount)
                    setattr(r_acc, res, r_val + amount)
                    trade = TradeRecord(
                        year=year, giver_id=giver.id,
                        receiver_id=receiver.id, resource=res,
                        amount=amount,
                        black_market=economy.black_market_active,
                    )
                    trades.append(trade)
                    break
    economy.trade_log = trades
    return trades


# ---------------------------------------------------------------------------
# Inequality
# ---------------------------------------------------------------------------

def compute_gini(accounts: dict[str, PersonalAccount]) -> float:
    """Compute the Gini coefficient of wealth distribution.

    Returns 0.0 (perfect equality) to 1.0 (perfect inequality).
    """
    values = sorted(a.total() for a in accounts.values())
    n = len(values)
    if n < 2:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    numerator = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(values))
    return numerator / (n * total)


# ---------------------------------------------------------------------------
# Black market detection
# ---------------------------------------------------------------------------

def check_black_market(
    economy: EconomyState,
    gov_type: str,
    active_colonists: list[Colonist],
) -> bool:
    """Check if a black market emerges."""
    rate = TAX_RATES.get(gov_type, 0.0)
    if rate < BLACK_MARKET_TAX_THRESHOLD:
        economy.black_market_active = False
        return False
    if not active_colonists:
        economy.black_market_active = False
        return False
    avg_paranoia = sum(c.stats.paranoia for c in active_colonists) / len(active_colonists)
    economy.black_market_active = avg_paranoia > BLACK_MARKET_PARANOIA_THRESHOLD
    return economy.black_market_active


# ---------------------------------------------------------------------------
# Wealth-based survival
# ---------------------------------------------------------------------------

def spend_personal_reserves(
    economy: EconomyState,
    colonist_id: str,
    crisis_resource: str,
) -> float:
    """Spend personal reserves to avert a resource-scarcity death.

    Returns the amount spent (transferred back to colony pool).
    """
    if crisis_resource not in RESOURCE_KINDS:
        return 0.0
    account = economy.get_account(colonist_id)
    available = getattr(account, crisis_resource)
    spend = min(available, 0.05)
    if spend <= 0:
        return 0.0
    setattr(account, crisis_resource, available - spend)
    return spend


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def handle_death(economy: EconomyState, colonist_id: str,
                 resources: Any) -> float:
    """Transfer dead colonist's assets back to colony pool."""
    account = economy.remove_account(colonist_id)
    if account is None:
        return 0.0
    total = 0.0
    for r in RESOURCE_KINDS:
        val = getattr(account, r)
        if val > 0 and hasattr(resources, r):
            setattr(resources, r, min(1.0, getattr(resources, r) + val))
            total += val
    return total


def handle_exile(economy: EconomyState, colonist_id: str,
                 resources: Any) -> float:
    """Confiscate exiled colonist's assets to colony pool."""
    return handle_death(economy, colonist_id, resources)


def handle_birth(economy: EconomyState, child_id: str) -> None:
    """Create empty account for newborn."""
    economy.get_account(child_id)


def handle_immigrant(economy: EconomyState, immigrant_id: str,
                     rng: random.Random) -> None:
    """Create account for immigrant with small endowment."""
    account = economy.get_account(immigrant_id)
    for r in RESOURCE_KINDS:
        setattr(account, r, 0.01 + rng.gauss(0, 0.005))
    account.clamp()


# ---------------------------------------------------------------------------
# Main tick
# ---------------------------------------------------------------------------

def tick_economy(
    economy: EconomyState,
    active_colonists: list[Colonist],
    actions: dict[str, str],
    gov_type: str,
    leader_id: str | None,
    social_graph: Any,
    resources: Any,
    year: int,
    rng: random.Random,
) -> dict:
    """Advance the economy by one year.

    Called from engine after colonist actions, before death checks.
    Returns a summary dict for YearResult.
    """
    # 1. Labour income
    for colonist in active_colonists:
        action = actions.get(colonist.id, "rest")
        if action == "hoard":
            allocate_hoard_action(economy, colonist, resources, rng)
        else:
            allocate_labor_income(economy, colonist, action, resources, rng)

    # 2. Black market check
    check_black_market(economy, gov_type, active_colonists)

    # 3. Taxation
    collect_taxes(economy, gov_type, active_colonists, rng)

    # 4. Redistribution
    redistribute(economy, gov_type, active_colonists, leader_id)

    # 5. Trade
    execute_trades(economy, social_graph, active_colonists, year, rng)

    # 6. Gini coefficient
    gini = compute_gini(economy.accounts)
    economy.gini_history.append(gini)

    return economy.summary()
