"""Personal economy and trade system for Mars-100 colony.

Colonists accumulate small personal reserves from their activities.
Trade happens based on social trust.  Inequality (Gini coefficient)
creates pressure on governance and action selection.  The hoarding
stat affects retention — selfish allocators, not efficient workers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

TRADEABLE = ("food", "water", "power", "medicine")
MAX_RESERVE = 1.0
MAX_TRADE_HISTORY = 200
MAX_GINI_HISTORY = 100

# Base personal earnings per action (small — separate from colony pool).
EARNING_RATES: dict[str, dict[str, float]] = {
    "farm":      {"food": 0.06},
    "terraform": {"water": 0.04},
    "code":      {"power": 0.04},
    "pray":      {"medicine": 0.02},
    "cooperate": {"food": 0.02, "water": 0.02},
    "mediate":   {"medicine": 0.02},
    "explore":   {"water": 0.02, "food": 0.02},
    "research":  {"power": 0.03},
    "hoard":     {},  # hoarders don't produce; they retain more
    "sabotage":  {},
    "rest":      {},
}

# Retention = fraction of earnings a colonist keeps (before hoarding bonus).
RETENTION_BY_SYSTEM: dict[str, float] = {
    "communal": 0.0,   # commune provides; no personal accumulation
    "barter":   1.0,   # keep everything
    "market":   0.7,   # 30% tax
    "planned":  0.5,   # 50% centrally allocated
}

# Hoarding bonus: up to +40% extra retention on top of base retention.
MAX_HOARD_BONUS = 0.4

# Trade parameters
TRADE_ATTEMPTS = 3          # how many potential partners to try
TRADE_MIN_SURPLUS = 0.08    # minimum surplus to offer a trade
TRADE_AMOUNT_FRAC = 0.25    # fraction of surplus offered per trade

# Inequality thresholds
INEQUALITY_GOVERNANCE_THRESHOLD = 0.35  # Gini above this biases governance


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PersonalInventory:
    """A colonist's personal resource reserves (0.0 to MAX_RESERVE)."""
    food: float = 0.0
    water: float = 0.0
    power: float = 0.0
    medicine: float = 0.0
    total_earned: float = 0.0
    total_traded: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {r: round(getattr(self, r), 4) for r in TRADEABLE} | {
            "total_earned": round(self.total_earned, 4),
            "total_traded": round(self.total_traded, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PersonalInventory:
        return cls(**{k: d.get(k, 0.0)
                      for k in (*TRADEABLE, "total_earned", "total_traded")})

    def wealth(self) -> float:
        """Total material holdings (used for Gini)."""
        return sum(getattr(self, r) for r in TRADEABLE)

    def clamp(self) -> None:
        for r in TRADEABLE:
            setattr(self, r, max(0.0, min(MAX_RESERVE, getattr(self, r))))

    def most_needed(self) -> str:
        return min(TRADEABLE, key=lambda r: getattr(self, r))

    def most_surplus(self) -> str:
        return max(TRADEABLE, key=lambda r: getattr(self, r))


@dataclass
class Trade:
    """Record of one exchange between two colonists."""
    year: int
    seller_id: str
    buyer_id: str
    given_resource: str
    given_amount: float
    received_resource: str
    received_amount: float

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "seller": self.seller_id, "buyer": self.buyer_id,
            "gave": f"{self.given_amount:.3f} {self.given_resource}",
            "got": f"{self.received_amount:.3f} {self.received_resource}",
        }


@dataclass
class EconomicState:
    """Colony-wide economic tracking."""
    inventories: dict[str, PersonalInventory] = field(default_factory=dict)
    trade_log: list[Trade] = field(default_factory=list)
    gini_history: list[float] = field(default_factory=list)
    system: str = "communal"

    def gini(self) -> float:
        """Current Gini coefficient from material holdings."""
        return compute_gini([inv.wealth() for inv in self.inventories.values()])

    def add_colonist(self, colonist_id: str) -> None:
        if colonist_id not in self.inventories:
            self.inventories[colonist_id] = PersonalInventory()

    def remove_colonist(self, colonist_id: str) -> dict[str, float]:
        """Remove a colonist, returning their holdings for redistribution."""
        inv = self.inventories.pop(colonist_id, PersonalInventory())
        return {r: getattr(inv, r) * 0.5 for r in TRADEABLE}

    def redistribute(self, pool: dict[str, float], active_ids: list[str]) -> None:
        """Distribute a resource pool equally among active colonists."""
        if not active_ids:
            return
        share = 1.0 / len(active_ids)
        for cid in active_ids:
            inv = self.inventories.get(cid)
            if inv is None:
                continue
            for r in TRADEABLE:
                setattr(inv, r, getattr(inv, r) + pool.get(r, 0.0) * share)
            inv.clamp()

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "gini": round(self.gini(), 4),
            "gini_history": [round(g, 4) for g in self.gini_history[-MAX_GINI_HISTORY:]],
            "inventories": {k: v.to_dict() for k, v in self.inventories.items()},
            "total_trades": len(self.trade_log),
            "recent_trades": [t.to_dict() for t in self.trade_log[-10:]],
        }

    def summary(self) -> dict:
        n = max(1, len(self.inventories))
        return {
            "system": self.system,
            "gini": round(self.gini(), 4),
            "total_trades": len(self.trade_log),
            "avg_wealth": round(
                sum(inv.wealth() for inv in self.inventories.values()) / n, 4),
        }


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def compute_gini(values: list[float]) -> float:
    """Gini coefficient.  0 = perfect equality, 1 = maximum inequality."""
    if len(values) < 2:
        return 0.0
    s = sorted(values)
    n = len(s)
    total = sum(s)
    if total <= 0:
        return 0.0
    weighted = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(s))
    return max(0.0, min(1.0, weighted / (n * total)))


def map_economic_system(gov_type: str) -> str:
    """Map governance type to economic system."""
    return {
        "anarchy": "barter",
        "council": "market",
        "dictator": "planned",
        "lottery": "barter",
        "consensus": "communal",
        "ai_governor": "planned",
    }.get(gov_type, "communal")


def tick_economics(
    econ: EconomicState,
    actions: dict[str, str],
    colonist_hoarding: dict[str, float],
    social,
    active_ids: list[str],
    year: int,
    gov_type: str,
    rng: random.Random,
) -> dict:
    """Advance personal economics by one year.  Returns year summary."""
    econ.system = map_economic_system(gov_type)
    base_retention = RETENTION_BY_SYSTEM.get(econ.system, 0.5)

    for cid in active_ids:
        econ.add_colonist(cid)

    # --- Phase 1: earning ---
    for cid in active_ids:
        action = actions.get(cid, "rest")
        inv = econ.inventories[cid]
        rates = EARNING_RATES.get(action, {})
        hoard_stat = colonist_hoarding.get(cid, 0.5)
        retention = min(1.0, base_retention + hoard_stat * MAX_HOARD_BONUS)
        for resource, rate in rates.items():
            if resource in TRADEABLE:
                amount = rate * retention
                setattr(inv, resource, getattr(inv, resource) + amount)
                inv.total_earned += amount
        inv.clamp()

    # --- Phase 2: trading ---
    trades_this_year: list[Trade] = []
    traded_this_year: set[str] = set()
    shuffled = list(active_ids)
    rng.shuffle(shuffled)

    for seller_id in shuffled:
        if seller_id in traded_this_year:
            continue
        seller_inv = econ.inventories.get(seller_id)
        if seller_inv is None:
            continue
        surplus_res = seller_inv.most_surplus()
        surplus_val = getattr(seller_inv, surplus_res)
        if surplus_val < TRADE_MIN_SURPLUS:
            continue

        # Try a few potential partners
        candidates = [c for c in shuffled
                       if c != seller_id and c not in traded_this_year]
        for buyer_id in candidates[:TRADE_ATTEMPTS]:
            buyer_inv = econ.inventories.get(buyer_id)
            if buyer_inv is None:
                continue
            trust = social.get(seller_id, buyer_id).trust
            # Trust modulates trade probability (not a binary gate)
            if rng.random() > trust:
                continue
            needed = buyer_inv.most_needed()
            if needed != surplus_res:
                continue
            buyer_surplus_res = buyer_inv.most_surplus()
            if buyer_surplus_res == needed:
                continue
            buyer_surplus_val = getattr(buyer_inv, buyer_surplus_res)
            if buyer_surplus_val < 0.04:
                continue

            # Execute trade
            give_amount = min(surplus_val * TRADE_AMOUNT_FRAC, 0.06)
            # Better trust → more favorable exchange rate for seller
            exchange_rate = 0.7 + trust * 0.6
            recv_amount = min(give_amount * exchange_rate, buyer_surplus_val * 0.5)

            setattr(seller_inv, surplus_res,
                    getattr(seller_inv, surplus_res) - give_amount)
            setattr(buyer_inv, surplus_res,
                    getattr(buyer_inv, surplus_res) + give_amount)
            setattr(buyer_inv, buyer_surplus_res,
                    getattr(buyer_inv, buyer_surplus_res) - recv_amount)
            setattr(seller_inv, buyer_surplus_res,
                    getattr(seller_inv, buyer_surplus_res) + recv_amount)

            seller_inv.total_traded += give_amount
            buyer_inv.total_traded += recv_amount
            seller_inv.clamp()
            buyer_inv.clamp()

            trade = Trade(year=year, seller_id=seller_id, buyer_id=buyer_id,
                          given_resource=surplus_res, given_amount=round(give_amount, 4),
                          received_resource=buyer_surplus_res,
                          received_amount=round(recv_amount, 4))
            trades_this_year.append(trade)
            traded_this_year.add(seller_id)
            traded_this_year.add(buyer_id)
            break  # max 1 successful trade per seller

    econ.trade_log.extend(trades_this_year)
    # Prune old trades to bound memory
    if len(econ.trade_log) > MAX_TRADE_HISTORY:
        econ.trade_log = econ.trade_log[-MAX_TRADE_HISTORY:]

    # --- Phase 3: compute Gini ---
    gini = econ.gini()
    econ.gini_history.append(gini)
    if len(econ.gini_history) > MAX_GINI_HISTORY:
        econ.gini_history = econ.gini_history[-MAX_GINI_HISTORY:]

    return {
        "year": year,
        "system": econ.system,
        "gini": round(gini, 4),
        "trades": len(trades_this_year),
    }


def compute_economic_pressure(econ: EconomicState) -> dict[str, float]:
    """Action weight adjustments driven by inequality.

    High Gini → more pressure to cooperate/mediate, less to hoard.
    """
    gini = econ.gini()
    if gini < 0.15:
        return {}
    pressure: dict[str, float] = {
        "cooperate": 0.25 * gini,
        "mediate": 0.15 * gini,
        "hoard": -0.3 * gini,
    }
    if gini > 0.5:
        pressure["sabotage"] = 0.1 * gini
    return pressure


def inequality_vote_bias(econ: EconomicState, colonist_id: str,
                         proposed_gov: str) -> float:
    """Small vote bias from inequality.  Poor colonists favour redistribution."""
    gini = econ.gini()
    if gini < INEQUALITY_GOVERNANCE_THRESHOLD:
        return 0.0
    inv = econ.inventories.get(colonist_id)
    if inv is None:
        return 0.0
    avg = sum(i.wealth() for i in econ.inventories.values()) / max(1, len(econ.inventories))
    relative = inv.wealth() - avg  # negative = poorer than average
    redistributive = {"consensus", "council", "ai_governor"}
    if proposed_gov in redistributive and relative < 0:
        return 0.15 * gini  # poor colonists favour redistribution
    if proposed_gov in redistributive and relative > 0:
        return -0.1 * gini  # rich colonists resist redistribution
    return 0.0
