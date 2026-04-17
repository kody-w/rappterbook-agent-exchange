"""Economics organ for Mars-100 colony simulation.

Models emergent trade, labor markets, wealth inequality, and
economic governance interaction.  Credits are IOUs from barter —
no private resource stockpiles, no double-counting with the
communal resource pool.

Key dynamics:
  - Colonists produce labor_value each year from their action + skills
  - Complementary-skill pairs trade via barter, generating IOU credits
  - Trust from the social graph gates willingness to trade
  - Gini coefficient (smoothed over 3 years) measures inequality
  - High sustained inequality feeds governance proposals + cultural pressure
  - Different governance types apply redistribution policies
  - Independence from Earth disrupts trade patterns
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# -- constants ---------------------------------------------------------------

TRADE_TRUST_THRESHOLD = 0.35
MAX_TRADES_PER_YEAR = 20
MAX_TRADE_LOG = 50
GINI_WINDOW = 3
GINI_GOVERNANCE_THRESHOLD = 0.45
GINI_GOVERNANCE_SUSTAIN = 3

REDISTRIBUTION_RATES = {
    "anarchy": 0.0,
    "council": 0.15,
    "consensus": 0.20,
    "dictator": 0.25,
    "lottery": 0.10,
    "ai_governor": 0.18,
}

LABOR_VALUE_BASE = 0.1
SKILL_LABOR_MAP = {
    "terraform": "terraforming",
    "farm": "hydroponics",
    "code": "coding",
    "mediate": "mediation",
    "pray": "prayer",
    "research": "coding",
    "cooperate": "mediation",
    "explore": "improvisation",
    "hoard": "hoarding",
    "sabotage": "sabotage",
    "rest": None,
}

COMPLEMENTARY_SKILLS = {
    "terraforming": ["hydroponics", "coding"],
    "hydroponics": ["terraforming", "mediation"],
    "coding": ["terraforming", "prayer"],
    "mediation": ["hydroponics", "coding"],
    "prayer": ["mediation", "coding"],
    "sabotage": [],
}


# -- data classes ------------------------------------------------------------

@dataclass
class Wallet:
    """A colonist's IOU credit balance and trade history."""
    credits: float = 0.0
    lifetime_earned: float = 0.0
    lifetime_spent: float = 0.0
    trades_completed: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "credits": round(self.credits, 4),
            "lifetime_earned": round(self.lifetime_earned, 4),
            "lifetime_spent": round(self.lifetime_spent, 4),
            "trades_completed": self.trades_completed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Wallet:
        return cls(
            credits=d.get("credits", 0.0),
            lifetime_earned=d.get("lifetime_earned", 0.0),
            lifetime_spent=d.get("lifetime_spent", 0.0),
            trades_completed=int(d.get("trades_completed", 0)),
        )


@dataclass
class TradeRecord:
    """A single barter trade between two colonists."""
    year: int
    seller_id: str
    buyer_id: str
    seller_skill: str
    buyer_skill: str
    value: float

    def to_dict(self) -> dict:
        return {
            "year": self.year, "seller_id": self.seller_id,
            "buyer_id": self.buyer_id, "seller_skill": self.seller_skill,
            "buyer_skill": self.buyer_skill,
            "value": round(self.value, 4),
        }


@dataclass
class EconomicState:
    """Colony-wide economic state across simulation years."""
    wallets: dict[str, Wallet] = field(default_factory=dict)
    trade_log: list[TradeRecord] = field(default_factory=list)
    gini_history: list[float] = field(default_factory=list)
    total_trades: int = 0
    currency_velocity: float = 0.0
    economic_policy: str = "communal"
    years_above_gini_threshold: int = 0

    def to_dict(self) -> dict:
        return {
            "wallets": {k: v.to_dict() for k, v in self.wallets.items()},
            "trade_log": [t.to_dict() for t in self.trade_log[-MAX_TRADE_LOG:]],
            "gini_history": [round(g, 4) for g in self.gini_history[-10:]],
            "gini_current": round(self.gini_history[-1], 4) if self.gini_history else 0.0,
            "gini_smoothed": round(self.smoothed_gini(), 4),
            "total_trades": self.total_trades,
            "currency_velocity": round(self.currency_velocity, 4),
            "economic_policy": self.economic_policy,
        }

    def smoothed_gini(self) -> float:
        """Rolling average Gini over the last GINI_WINDOW years."""
        recent = self.gini_history[-GINI_WINDOW:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def ensure_account(self, colonist_id: str) -> Wallet:
        """Create wallet for a new colonist if it doesn't exist."""
        if colonist_id not in self.wallets:
            self.wallets[colonist_id] = Wallet()
        return self.wallets[colonist_id]


@dataclass
class EconomicTickResult:
    """Result of one year's economic activity."""
    year: int
    trades: list[dict] = field(default_factory=list)
    gini: float = 0.0
    gini_smoothed: float = 0.0
    redistribution: float = 0.0
    policy: str = "communal"
    velocity: float = 0.0
    inequality_alert: bool = False

    def to_dict(self) -> dict:
        return {
            "year": self.year, "trades": self.trades,
            "gini": round(self.gini, 4),
            "gini_smoothed": round(self.gini_smoothed, 4),
            "redistribution": round(self.redistribution, 4),
            "policy": self.policy,
            "velocity": round(self.velocity, 4),
            "inequality_alert": self.inequality_alert,
        }


# -- labor value computation ------------------------------------------------

def compute_labor_value(action: str, colonist_skills: dict[str, float],
                        colonist_stats: dict[str, float]) -> float:
    """Compute the labor value produced by a colonist's action.

    Returns a float >= 0. Higher skill in the relevant domain
    produces more labor value.
    """
    skill_name = SKILL_LABOR_MAP.get(action)
    if skill_name is None:
        return LABOR_VALUE_BASE * 0.5

    if skill_name in colonist_skills:
        skill_level = colonist_skills[skill_name]
    elif skill_name in colonist_stats:
        skill_level = colonist_stats[skill_name]
    else:
        skill_level = 0.3

    return LABOR_VALUE_BASE * (1.0 + skill_level)


# -- trade execution ---------------------------------------------------------

def find_trade_pairs(active_ids: list[str], actions: dict[str, str],
                     social_trust: dict[tuple[str, str], float],
                     rng: random.Random) -> list[tuple[str, str]]:
    """Find pairs of colonists whose skills are complementary and who trust each other.

    Returns a list of (seller_id, buyer_id) pairs.
    """
    pairs: list[tuple[str, str, float]] = []
    for i, a_id in enumerate(active_ids):
        a_action = actions.get(a_id, "rest")
        a_skill = SKILL_LABOR_MAP.get(a_action)
        if a_skill is None or a_skill == "sabotage":
            continue
        complements = COMPLEMENTARY_SKILLS.get(a_skill, [])
        for b_id in active_ids[i + 1:]:
            b_action = actions.get(b_id, "rest")
            b_skill = SKILL_LABOR_MAP.get(b_action)
            if b_skill is None:
                continue
            if b_skill in complements or a_skill in COMPLEMENTARY_SKILLS.get(b_skill, []):
                trust = social_trust.get((a_id, b_id), 0.3)
                if trust >= TRADE_TRUST_THRESHOLD:
                    pairs.append((a_id, b_id, trust))
    rng.shuffle(pairs)
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for a, b, _ in pairs[:MAX_TRADES_PER_YEAR]]


def execute_trade(state: EconomicState, seller_id: str, buyer_id: str,
                  seller_action: str, buyer_action: str,
                  seller_skills: dict[str, float],
                  buyer_skills: dict[str, float],
                  seller_stats: dict[str, float],
                  buyer_stats: dict[str, float],
                  year: int, rng: random.Random) -> TradeRecord | None:
    """Execute a barter trade between two colonists.

    Both parties gain credits proportional to the labor value they
    contribute. This is an IOU system — credits represent deferred
    exchange value, not minted currency.
    """
    seller_wallet = state.ensure_account(seller_id)
    buyer_wallet = state.ensure_account(buyer_id)

    seller_value = compute_labor_value(seller_action, seller_skills, seller_stats)
    buyer_value = compute_labor_value(buyer_action, buyer_skills, buyer_stats)

    trade_value = (seller_value + buyer_value) / 2
    noise = rng.gauss(0, 0.01)
    trade_value = max(0.01, trade_value + noise)

    seller_wallet.credits += trade_value * 0.5
    seller_wallet.lifetime_earned += trade_value * 0.5
    seller_wallet.trades_completed += 1

    buyer_wallet.credits += trade_value * 0.5
    buyer_wallet.lifetime_earned += trade_value * 0.5
    buyer_wallet.trades_completed += 1

    seller_skill = SKILL_LABOR_MAP.get(seller_action, "unknown")
    buyer_skill = SKILL_LABOR_MAP.get(buyer_action, "unknown")

    record = TradeRecord(
        year=year, seller_id=seller_id, buyer_id=buyer_id,
        seller_skill=seller_skill or "general",
        buyer_skill=buyer_skill or "general",
        value=trade_value,
    )
    state.trade_log.append(record)
    if len(state.trade_log) > MAX_TRADE_LOG:
        state.trade_log = state.trade_log[-MAX_TRADE_LOG:]
    state.total_trades += 1
    return record


# -- inequality measurement -------------------------------------------------

def compute_gini(values: list[float]) -> float:
    """Compute the Gini coefficient for a list of values.

    Returns 0.0 for perfect equality, approaches 1.0 for total inequality.
    """
    n = len(values)
    if n < 2:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    sorted_vals = sorted(values)
    numerator = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
    return numerator / (n * total)


# -- redistribution ----------------------------------------------------------

def apply_redistribution(state: EconomicState, gov_type: str,
                         active_ids: list[str],
                         leader_id: str | None = None) -> float:
    """Apply redistribution based on governance type.

    Returns total credits redistributed.
    """
    rate = REDISTRIBUTION_RATES.get(gov_type, 0.0)
    if rate <= 0 or not active_ids:
        return 0.0

    total_tax = 0.0
    for cid in active_ids:
        wallet = state.ensure_account(cid)
        if wallet.credits > 0:
            tax = wallet.credits * rate
            wallet.credits -= tax
            wallet.lifetime_spent += tax
            total_tax += tax

    if total_tax <= 0:
        return 0.0

    if gov_type == "dictator" and leader_id and leader_id in active_ids:
        leader_share = total_tax * 0.3
        pool = total_tax - leader_share
        leader_wallet = state.ensure_account(leader_id)
        leader_wallet.credits += leader_share
        leader_wallet.lifetime_earned += leader_share
        others = [cid for cid in active_ids if cid != leader_id]
        if others:
            per_capita = pool / len(others)
            for cid in others:
                w = state.ensure_account(cid)
                w.credits += per_capita
                w.lifetime_earned += per_capita
    else:
        per_capita = total_tax / len(active_ids)
        for cid in active_ids:
            w = state.ensure_account(cid)
            w.credits += per_capita
            w.lifetime_earned += per_capita

    return total_tax


# -- economic policy update --------------------------------------------------

def update_economic_policy(gini_smoothed: float, gov_type: str) -> str:
    """Derive economic policy from governance type and inequality level."""
    if gov_type == "anarchy":
        return "market" if gini_smoothed > 0.3 else "communal"
    elif gov_type == "dictator":
        return "planned"
    elif gov_type in ("council", "consensus"):
        return "mixed"
    elif gov_type == "ai_governor":
        return "algorithmic"
    elif gov_type == "lottery":
        return "mixed"
    return "communal"


# -- estate handling ---------------------------------------------------------

def handle_death(state: EconomicState, colonist_id: str,
                 active_ids: list[str]) -> float:
    """Redistribute a dead colonist's credits to surviving colonists."""
    wallet = state.wallets.get(colonist_id)
    if wallet is None or wallet.credits <= 0:
        return 0.0
    estate = wallet.credits
    wallet.credits = 0.0
    survivors = [cid for cid in active_ids if cid != colonist_id]
    if not survivors:
        return estate
    per_capita = estate / len(survivors)
    for cid in survivors:
        w = state.ensure_account(cid)
        w.credits += per_capita
        w.lifetime_earned += per_capita
    return estate


def handle_exile(state: EconomicState, colonist_id: str) -> float:
    """Confiscate an exiled colonist's credits (they lose everything)."""
    wallet = state.wallets.get(colonist_id)
    if wallet is None:
        return 0.0
    confiscated = wallet.credits
    wallet.credits = 0.0
    return confiscated


# -- main tick ---------------------------------------------------------------

def tick_economics(
    state: EconomicState,
    year: int,
    active_ids: list[str],
    actions: dict[str, str],
    social_trust: dict[tuple[str, str], float],
    colonist_skills: dict[str, dict[str, float]],
    colonist_stats: dict[str, dict[str, float]],
    gov_type: str,
    leader_id: str | None,
    rng: random.Random,
) -> EconomicTickResult:
    """Advance the colony economy by one Martian year.

    Called after social graph updates and before governance proposals.
    """
    result = EconomicTickResult(year=year)

    for cid in active_ids:
        state.ensure_account(cid)

    trade_pairs = find_trade_pairs(active_ids, actions, social_trust, rng)
    year_trades: list[dict] = []
    for seller_id, buyer_id in trade_pairs:
        record = execute_trade(
            state, seller_id, buyer_id,
            actions.get(seller_id, "rest"), actions.get(buyer_id, "rest"),
            colonist_skills.get(seller_id, {}),
            colonist_skills.get(buyer_id, {}),
            colonist_stats.get(seller_id, {}),
            colonist_stats.get(buyer_id, {}),
            year, rng,
        )
        if record:
            year_trades.append(record.to_dict())
    result.trades = year_trades

    credit_values = [state.wallets[cid].credits for cid in active_ids
                     if cid in state.wallets]
    gini = compute_gini(credit_values)
    state.gini_history.append(gini)
    result.gini = gini

    gini_smoothed = state.smoothed_gini()
    result.gini_smoothed = gini_smoothed

    if gini_smoothed > GINI_GOVERNANCE_THRESHOLD:
        state.years_above_gini_threshold += 1
    else:
        state.years_above_gini_threshold = max(0, state.years_above_gini_threshold - 1)
    result.inequality_alert = (
        state.years_above_gini_threshold >= GINI_GOVERNANCE_SUSTAIN
    )

    redistribution = apply_redistribution(state, gov_type, active_ids, leader_id)
    result.redistribution = redistribution

    policy = update_economic_policy(gini_smoothed, gov_type)
    state.economic_policy = policy
    result.policy = policy

    if active_ids:
        state.currency_velocity = len(year_trades) / len(active_ids)
    else:
        state.currency_velocity = 0.0
    result.velocity = state.currency_velocity

    return result
