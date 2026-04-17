"""
Economy organ for Mars-100 colony simulation.

Models wealth distribution, labour valuation, taxation, redistribution,
and inequality tracking.  Wealth is represented as *shares* of economic
output — they sum to ~1.0 across all active colonists, so the economy
is a zero-sum internal distribution rather than an external currency.

Key dynamics:
  - Labour value: actions addressing scarce resources pay more
  - Taxation: governance type sets the colony tax rate
  - Redistribution: tax revenue is redistributed (equally, to leader, etc.)
  - Trust-based transfers: high-trust pairs with empathy share wealth
  - Inequality (Gini): high inequality drives governance proposals and conflict
  - Lifecycle: births get a stipend, deaths' estates are inherited,
    exiles' wealth is confiscated and redistributed

Engine v7.0
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# -- constants ---------------------------------------------------------------

INITIAL_SHARE = 0.1  # founding colonists each start with equal share
MIN_SHARE = 0.001    # nobody's share drops below this floor

# Tax rates by governance type
TAX_RATES: dict[str, float] = {
    "anarchy": 0.0,
    "council": 0.08,
    "dictator": 0.15,
    "lottery": 0.05,
    "consensus": 0.10,
    "ai_governor": 0.12,
}

# Labour value multipliers when a resource is critical (< 0.15)
SCARCITY_BONUS: dict[str, list[str]] = {
    "food": ["farm", "cooperate"],
    "water": ["terraform"],
    "power": ["code", "research"],
    "air": ["terraform"],
    "medicine": ["mediate", "pray"],
}

# Base labour value per action (before scarcity adjustment)
BASE_LABOUR_VALUE: dict[str, float] = {
    "terraform": 0.12,
    "farm": 0.12,
    "mediate": 0.08,
    "code": 0.10,
    "pray": 0.04,
    "sabotage": -0.05,
    "cooperate": 0.09,
    "hoard": 0.02,
    "explore": 0.06,
    "rest": 0.01,
    "research": 0.10,
}

# Gini thresholds for emergent effects
GINI_PROPOSAL_THRESHOLD = 0.55
GINI_CONFLICT_THRESHOLD = 0.65
GINI_CRISIS_THRESHOLD = 0.75


# -- data classes ------------------------------------------------------------

@dataclass
class Trade:
    """A single wealth transfer between two colonists."""
    giver_id: str
    receiver_id: str
    amount: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "giver": self.giver_id, "receiver": self.receiver_id,
            "amount": round(self.amount, 6), "reason": self.reason,
        }


@dataclass
class EconomicState:
    """Persistent colony economic state across simulation years."""
    shares: dict[str, float] = field(default_factory=dict)
    tax_rate: float = 0.0
    treasury: float = 0.0
    gini: float = 0.0
    total_trades: int = 0
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "shares": {k: round(v, 6) for k, v in self.shares.items()},
            "tax_rate": round(self.tax_rate, 4),
            "treasury": round(self.treasury, 6),
            "gini": round(self.gini, 4),
            "total_trades": self.total_trades,
        }

    def summary(self) -> dict:
        """Brief summary for year result."""
        active = {k: v for k, v in self.shares.items() if v > 0}
        if not active:
            return {"gini": 0.0, "richest": None, "poorest": None,
                    "tax_rate": self.tax_rate, "treasury": round(self.treasury, 4)}
        richest = max(active, key=active.get)
        poorest = min(active, key=active.get)
        return {
            "gini": round(self.gini, 4),
            "richest": richest,
            "richest_share": round(active[richest], 4),
            "poorest": poorest,
            "poorest_share": round(active[poorest], 4),
            "tax_rate": round(self.tax_rate, 4),
            "treasury": round(self.treasury, 4),
            "active_participants": len(active),
        }


@dataclass
class EconomicTickResult:
    """Result of one year's economic activity."""
    year: int
    incomes: dict[str, float] = field(default_factory=dict)
    taxes_collected: float = 0.0
    redistributed: float = 0.0
    trades: list[dict] = field(default_factory=list)
    gini_before: float = 0.0
    gini_after: float = 0.0
    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "incomes": {k: round(v, 6) for k, v in self.incomes.items()},
            "taxes_collected": round(self.taxes_collected, 6),
            "redistributed": round(self.redistributed, 6),
            "trades": self.trades,
            "gini_before": round(self.gini_before, 4),
            "gini_after": round(self.gini_after, 4),
            "events": self.events,
        }


# -- core functions ----------------------------------------------------------

def initialize_shares(colonist_ids: list[str]) -> dict[str, float]:
    """Create equal starting shares for a list of colonists."""
    if not colonist_ids:
        return {}
    share = 1.0 / len(colonist_ids)
    return {cid: share for cid in colonist_ids}


def compute_gini(shares: dict[str, float]) -> float:
    """Compute the Gini coefficient of a wealth distribution.

    Returns 0.0 (perfect equality) to approaching 1.0 (maximum inequality).
    With fewer than 2 participants, returns 0.0.
    """
    values = sorted(v for v in shares.values() if v > 0)
    n = len(values)
    if n < 2:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    cumulative = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(values):
        cumulative += v
        weighted_sum += (i + 1) * v
    return (2.0 * weighted_sum) / (n * total) - (n + 1) / n


def compute_labour_value(action: str, resources_dict: dict[str, float],
                         skill_level: float) -> float:
    """Compute the economic value of an action given current scarcity.

    Higher value when the action addresses a scarce resource.
    Skill amplifies the value.
    """
    base = BASE_LABOUR_VALUE.get(action, 0.05)
    scarcity_mult = 1.0
    for resource, actions in SCARCITY_BONUS.items():
        if action in actions:
            level = resources_dict.get(resource, 0.5)
            if level < 0.15:
                scarcity_mult += 2.0 * (0.15 - level) / 0.15
            elif level < 0.3:
                scarcity_mult += 0.5 * (0.3 - level) / 0.3
    return max(0.0, base * scarcity_mult * (0.7 + 0.6 * skill_level))


def _get_action_skill(colonist_dict: dict, action: str) -> float:
    """Look up the relevant skill level for an action."""
    skill_map = {
        "terraform": "terraforming", "farm": "hydroponics",
        "mediate": "mediation", "code": "coding",
        "pray": "prayer", "sabotage": "sabotage",
        "research": "coding",
    }
    skill_name = skill_map.get(action)
    if skill_name and "skills" in colonist_dict:
        return colonist_dict["skills"].get(skill_name, 0.0)
    return 0.0


def execute_trades(shares: dict[str, float],
                   colonist_map: dict[str, dict],
                   social_edges: dict,
                   active_ids: list[str],
                   rng: random.Random) -> list[Trade]:
    """Execute trust-based voluntary wealth transfers.

    High-empathy colonists with high-trust relationships transfer
    small amounts from wealthier to poorer.
    """
    trades: list[Trade] = []
    for cid in active_ids:
        c = colonist_map.get(cid, {})
        empathy = c.get("stats", {}).get("empathy", 0.5)
        hoarding = c.get("stats", {}).get("hoarding", 0.5)
        if empathy < 0.5 or hoarding > 0.6:
            continue
        my_share = shares.get(cid, 0.0)
        if my_share < 0.05:
            continue
        edges = social_edges.get(cid, {})
        for other_id in active_ids:
            if other_id == cid:
                continue
            rel = edges.get(other_id, {})
            trust = rel.get("trust", 0.5) if isinstance(rel, dict) else 0.5
            if trust < 0.6:
                continue
            other_share = shares.get(other_id, 0.0)
            if other_share >= my_share:
                continue
            transfer_prob = empathy * 0.3 * (trust - 0.5)
            if rng.random() > transfer_prob:
                continue
            gap = my_share - other_share
            amount = min(gap * 0.1, my_share * 0.05)
            amount = max(0.0, amount)
            if amount < 0.001:
                continue
            shares[cid] = max(MIN_SHARE, shares[cid] - amount)
            shares[other_id] = shares.get(other_id, 0.0) + amount
            trades.append(Trade(
                giver_id=cid, receiver_id=other_id,
                amount=amount, reason="empathy_transfer",
            ))
    return trades


def apply_tax(shares: dict[str, float], active_ids: list[str],
              tax_rate: float) -> float:
    """Collect tax from all active colonists. Returns total collected."""
    if tax_rate <= 0:
        return 0.0
    collected = 0.0
    for cid in active_ids:
        share = shares.get(cid, 0.0)
        tax = share * tax_rate
        shares[cid] = max(MIN_SHARE, share - tax)
        collected += tax
    return collected


def redistribute(shares: dict[str, float], active_ids: list[str],
                 amount: float, gov_type: str,
                 leader_id: str | None) -> float:
    """Redistribute treasury to colonists based on governance type.

    Returns amount actually distributed.
    """
    if amount <= 0 or not active_ids:
        return 0.0

    if gov_type == "dictator" and leader_id and leader_id in active_ids:
        # Dictator keeps 60%, distributes 40%
        leader_take = amount * 0.6
        shares[leader_id] = shares.get(leader_id, 0.0) + leader_take
        remainder = amount * 0.4
        per_capita = remainder / len(active_ids)
        for cid in active_ids:
            shares[cid] = shares.get(cid, 0.0) + per_capita
    elif gov_type == "anarchy":
        return 0.0  # no redistribution under anarchy
    else:
        per_capita = amount / len(active_ids)
        for cid in active_ids:
            shares[cid] = shares.get(cid, 0.0) + per_capita

    return amount


def handle_death(shares: dict[str, float], dead_id: str,
                 social_edges: dict, active_ids: list[str]) -> list[Trade]:
    """Handle wealth of a deceased colonist — estate goes to most trusted."""
    estate = shares.pop(dead_id, 0.0)
    if estate <= 0 or not active_ids:
        return []

    edges = social_edges.get(dead_id, {})
    trust_scores = []
    for cid in active_ids:
        if cid == dead_id:
            continue
        rel = edges.get(cid, {})
        trust = rel.get("trust", 0.5) if isinstance(rel, dict) else 0.5
        trust_scores.append((cid, trust))

    if not trust_scores:
        return []

    trust_scores.sort(key=lambda x: x[1], reverse=True)
    total_trust = sum(t for _, t in trust_scores)
    if total_trust <= 0:
        per_capita = estate / len(trust_scores)
        for cid, _ in trust_scores:
            shares[cid] = shares.get(cid, 0.0) + per_capita
        return [Trade(giver_id=dead_id, receiver_id="colony",
                      amount=estate, reason="estate_equal")]

    trades = []
    for cid, trust in trust_scores:
        portion = estate * (trust / total_trust)
        shares[cid] = shares.get(cid, 0.0) + portion
        if portion > 0.001:
            trades.append(Trade(giver_id=dead_id, receiver_id=cid,
                                amount=portion, reason="inheritance"))
    return trades


def handle_exile(shares: dict[str, float], exiled_id: str,
                 active_ids: list[str]) -> float:
    """Confiscate exiled colonist's wealth, redistribute equally."""
    confiscated = shares.pop(exiled_id, 0.0)
    remaining = [cid for cid in active_ids if cid != exiled_id]
    if confiscated > 0 and remaining:
        per_capita = confiscated / len(remaining)
        for cid in remaining:
            shares[cid] = shares.get(cid, 0.0) + per_capita
    return confiscated


def handle_birth(shares: dict[str, float], child_id: str,
                 active_ids: list[str]) -> None:
    """Give a newborn a small stipend, slightly diluting everyone else."""
    n = len(active_ids) + 1
    stipend = 1.0 / n * 0.5  # half of a fair share
    # Dilute existing shares proportionally
    total_before = sum(shares.get(cid, 0.0) for cid in active_ids)
    if total_before > 0:
        dilution = stipend / total_before
        for cid in active_ids:
            shares[cid] = max(MIN_SHARE, shares[cid] * (1.0 - dilution))
    shares[child_id] = stipend


def handle_immigrant(shares: dict[str, float], immigrant_id: str,
                     active_ids: list[str]) -> None:
    """Give an immigrant a baseline share (slightly more than a child)."""
    n = len(active_ids) + 1
    stipend = 1.0 / n * 0.7  # 70% of a fair share
    total_before = sum(shares.get(cid, 0.0) for cid in active_ids)
    if total_before > 0:
        dilution = stipend / total_before
        for cid in active_ids:
            shares[cid] = max(MIN_SHARE, shares[cid] * (1.0 - dilution))
    shares[immigrant_id] = stipend


def normalize_shares(shares: dict[str, float]) -> None:
    """Normalize shares so they sum to 1.0, maintaining relative proportions."""
    total = sum(v for v in shares.values() if v > 0)
    if total <= 0:
        return
    factor = 1.0 / total
    for cid in shares:
        shares[cid] = max(0.0, shares[cid] * factor)


def update_tax_rate(state: EconomicState, gov_type: str) -> None:
    """Update tax rate based on current governance type."""
    state.tax_rate = TAX_RATES.get(gov_type, 0.05)


# -- main tick ---------------------------------------------------------------

def tick_economy(
    state: EconomicState,
    year: int,
    colonist_snapshots: list[dict],
    actions: dict[str, str],
    resources_dict: dict[str, float],
    gov_type: str,
    leader_id: str | None,
    social_edges: dict,
    deaths: list[dict],
    exiles: list[dict],
    births: list[dict],
    immigrants: list[dict],
    rng: random.Random,
) -> EconomicTickResult:
    """Advance the colony economy by one Martian year.

    Called AFTER resource tick and deaths/exiles/births in the engine,
    so we have realized outcomes for labour valuation.
    """
    result = EconomicTickResult(year=year)

    active_ids = [c["id"] for c in colonist_snapshots
                  if c.get("alive", True) and not c.get("exiled", False)]
    colonist_map = {c["id"]: c for c in colonist_snapshots}

    # Initialize shares for new colonists not yet in the economy
    for cid in active_ids:
        if cid not in state.shares:
            state.shares[cid] = INITIAL_SHARE

    result.gini_before = compute_gini(
        {k: v for k, v in state.shares.items() if k in active_ids})

    # 1. Compute labour income
    total_income = 0.0
    incomes: dict[str, float] = {}
    for cid in active_ids:
        action = actions.get(cid, "rest")
        skill = _get_action_skill(colonist_map.get(cid, {}), action)
        value = compute_labour_value(action, resources_dict, skill)
        incomes[cid] = value
        total_income += value

    # Normalize incomes to a small share redistribution (max 20% of total shares)
    if total_income > 0:
        income_pool = 0.15  # 15% of share value redistributed by income
        for cid in incomes:
            income_fraction = incomes[cid] / total_income
            income_delta = income_pool * income_fraction - income_pool / len(active_ids)
            state.shares[cid] = max(MIN_SHARE,
                                    state.shares.get(cid, 0.0) + income_delta)
    result.incomes = incomes

    # 2. Update tax rate from governance
    update_tax_rate(state, gov_type)

    # 3. Collect taxes
    taxes = apply_tax(state.shares, active_ids, state.tax_rate)
    state.treasury += taxes
    result.taxes_collected = taxes

    # 4. Redistribute treasury
    if state.treasury > 0.01:
        distributed = redistribute(
            state.shares, active_ids, state.treasury,
            gov_type, leader_id)
        state.treasury = max(0.0, state.treasury - distributed)
        result.redistributed = distributed

    # 5. Trust-based voluntary transfers
    trades = execute_trades(
        state.shares, colonist_map, social_edges, active_ids, rng)
    result.trades = [t.to_dict() for t in trades]
    state.total_trades += len(trades)

    # 6. Handle lifecycle events
    for death in deaths:
        dead_id = death.get("id", "")
        if dead_id in state.shares:
            handle_death(state.shares, dead_id, social_edges, active_ids)
            result.events.append(f"estate_distributed:{dead_id}")

    for exile in exiles:
        exiled_id = exile.get("id", "")
        if exiled_id in state.shares:
            handle_exile(state.shares, exiled_id, active_ids)
            result.events.append(f"wealth_confiscated:{exiled_id}")

    for birth in births:
        child_id = birth.get("id", "")
        handle_birth(state.shares, child_id, active_ids)
        result.events.append(f"stipend_granted:{child_id}")

    for imm in immigrants:
        imm_id = imm.get("id", "")
        handle_immigrant(state.shares, imm_id, active_ids)
        result.events.append(f"immigrant_stipend:{imm_id}")

    # 7. Normalize to prevent drift
    normalize_shares(state.shares)

    # 8. Compute final Gini
    state.gini = compute_gini(
        {k: v for k, v in state.shares.items() if k in active_ids})
    result.gini_after = state.gini

    # 9. Record history
    if year % 10 == 0 or abs(result.gini_after - result.gini_before) > 0.1:
        state.history.append({
            "year": year, "gini": round(state.gini, 4),
            "tax_rate": round(state.tax_rate, 4),
            "trades": len(trades),
        })

    return result


# -- economic pressure on action choice --------------------------------------

def compute_economic_pressure(shares: dict[str, float],
                              colonist_id: str,
                              active_ids: list[str]) -> dict[str, float]:
    """Compute action-weight modifiers based on a colonist's wealth.

    Wealthy colonists lean toward rest/explore/research.
    Poor colonists lean toward farm/hoard/cooperate.
    """
    if not active_ids or colonist_id not in shares:
        return {}

    my_share = shares.get(colonist_id, 0.0)
    avg_share = 1.0 / max(1, len(active_ids))
    relative = my_share / max(0.001, avg_share)

    pressure: dict[str, float] = {}
    if relative > 1.3:
        # Wealthy: lean toward leisure and research
        pressure["rest"] = 0.3
        pressure["explore"] = 0.2
        pressure["research"] = 0.2
        pressure["farm"] = -0.1
    elif relative < 0.7:
        # Poor: lean toward productive work
        pressure["farm"] = 0.3
        pressure["cooperate"] = 0.2
        pressure["hoard"] = 0.15
        pressure["rest"] = -0.2
        pressure["explore"] = -0.1

    return pressure
