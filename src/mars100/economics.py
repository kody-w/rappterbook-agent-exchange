"""
Economics organ for Mars-100.

Models labor credit accumulation, inter-colonist trade, inequality
(Gini coefficient), labor diversity, and economic policy.  The colony
operates communally at the resource level; the economics organ tracks
*individual contributions and entitlements* layered on top.

Key dynamics:
  - Colonists earn labor credits from productive actions each year
  - Credits decay 10 % annually (no infinite accumulation)
  - Trusted pairs exchange favors (credit transfers), improving efficiency
  - Governance type determines taxation / redistribution rate
  - Gini coefficient and labor diversity drive an efficiency modifier
    that scales existing resource production — the economy never creates
    resources directly, only modulates how well the colony converts
    labor into output
  - High inequality → social friction → efficiency penalty
  - High labor diversity → specialization bonus → efficiency bonus
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

# -- constants ---------------------------------------------------------------

CREDIT_DECAY_RATE = 0.10         # 10 % annual decay
TRADE_TRUST_THRESHOLD = 0.45     # minimum trust to trade
TRADE_FRACTION = 0.15            # fraction of surplus transferred per trade
MAX_TRADES_PER_YEAR = 5          # cap on bilateral trades per tick
STARTING_CREDITS = 1.0           # credits given to newborns / immigrants

ACTION_CREDIT_VALUES: dict[str, float] = {
    "terraform": 1.2,
    "farm": 1.3,
    "code": 1.1,
    "mediate": 0.9,
    "cooperate": 0.8,
    "explore": 0.7,
    "research": 1.0,
    "pray": 0.4,
    "rest": 0.2,
    "hoard": 0.3,
    "sabotage": 0.0,
}

POLICY_TAX_RATES: dict[str, float] = {
    "communal": 0.40,
    "mixed": 0.20,
    "free_market": 0.05,
}

GOV_TO_POLICY: dict[str, str] = {
    "council": "communal",
    "consensus": "communal",
    "dictator": "mixed",
    "lottery": "mixed",
    "ai_governor": "mixed",
    "anarchy": "free_market",
}


# -- data classes ------------------------------------------------------------

@dataclass
class EconomicState:
    """Colony economic state, tracked across years."""
    credits: dict[str, float] = field(default_factory=dict)
    policy: str = "communal"
    total_trades: int = 0
    gini_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "credits": dict(self.credits),
            "policy": self.policy,
            "total_trades": self.total_trades,
            "gini_history": list(self.gini_history[-20:]),
        }

    def summary(self) -> dict[str, Any]:
        vals = [v for v in self.credits.values() if v > 0]
        return {
            "policy": self.policy,
            "total_trades": self.total_trades,
            "active_accounts": len(vals),
            "mean_credits": sum(vals) / max(1, len(vals)),
            "gini": self.gini_history[-1] if self.gini_history else 0.0,
        }


@dataclass
class EconomicTickResult:
    """Result of one year of economic activity."""
    year: int
    trades_executed: int = 0
    gini: float = 0.0
    diversity: float = 0.0
    policy: str = "communal"
    efficiency_bonus: float = 0.0
    tax_collected: float = 0.0
    credits_awarded: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "trades": self.trades_executed,
            "gini": round(self.gini, 4),
            "diversity": round(self.diversity, 4),
            "policy": self.policy,
            "efficiency_bonus": round(self.efficiency_bonus, 4),
            "tax_collected": round(self.tax_collected, 4),
            "credits_awarded": round(self.credits_awarded, 4),
        }


# -- pure functions ----------------------------------------------------------

def compute_gini(values: list[float]) -> float:
    """Compute the Gini coefficient of a list of non-negative values.

    Returns 0.0 for empty, single-element, or all-zero inputs.
    """
    n = len(values)
    if n < 2:
        return 0.0
    total = sum(values)
    if total <= 0.0:
        return 0.0
    sorted_vals = sorted(values)
    cumulative = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        weighted_sum += (2 * (i + 1) - n - 1) * v
    return weighted_sum / (n * total)


def compute_diversity(actions: dict[str, str], action_pool: list[str]) -> float:
    """Compute labor diversity as normalized entropy.

    0.0 = everyone does the same action.
    1.0 = perfectly uniform distribution across all actions.
    """
    if not actions:
        return 0.0
    counts: dict[str, int] = {}
    for action in actions.values():
        counts[action] = counts.get(action, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    max_entropy = math.log(len(action_pool)) if len(action_pool) > 1 else 1.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0


def determine_policy(gov_type: str) -> str:
    """Map governance type to economic policy."""
    return GOV_TO_POLICY.get(gov_type, "mixed")


def compute_efficiency_bonus(gini: float, diversity: float) -> float:
    """Compute the resource-production efficiency modifier.

    Diversity is rewarded (up to +8 %).
    Extreme inequality is penalized (up to -6 %).
    Moderate inequality (0.2-0.35) gives a mild incentive bonus (+2 %).
    """
    diversity_bonus = diversity * 0.08
    if gini < 0.2:
        inequality_effect = 0.0
    elif gini < 0.35:
        inequality_effect = 0.02
    elif gini < 0.55:
        inequality_effect = -0.02 * (gini - 0.35) / 0.20
    else:
        inequality_effect = -0.02 - 0.04 * min(1.0, (gini - 0.55) / 0.45)
    return diversity_bonus + inequality_effect


# -- tick function -----------------------------------------------------------

def tick_economy(
    eco: EconomicState,
    year: int,
    actions: dict[str, str],
    active_ids: list[str],
    social_get: Any,
    gov_type: str,
    rng: random.Random,
) -> EconomicTickResult:
    """Advance the colony economy by one year.

    Parameters
    ----------
    eco : EconomicState
        Mutable economic state.
    year : int
        Current simulation year.
    actions : dict[str, str]
        Colonist ID → action chosen this year.
    active_ids : list[str]
        IDs of living, non-exiled colonists.
    social_get : callable(from_id, to_id) -> Relationship
        Accessor for social-graph trust values.
    gov_type : str
        Current governance type (e.g. "council", "anarchy").
    rng : random.Random
        Dedicated economics RNG.

    Returns
    -------
    EconomicTickResult
        Summary of this year's economic activity.
    """
    result = EconomicTickResult(year=year)

    # -- 0. Ensure all active colonists have accounts --
    for cid in active_ids:
        if cid not in eco.credits:
            eco.credits[cid] = STARTING_CREDITS

    # -- 1. Decay existing credits (wealth sink) --
    for cid in list(eco.credits):
        eco.credits[cid] *= (1.0 - CREDIT_DECAY_RATE)

    # -- 2. Award credits from productive actions --
    total_awarded = 0.0
    for cid in active_ids:
        action = actions.get(cid, "rest")
        earned = ACTION_CREDIT_VALUES.get(action, 0.5)
        earned *= (1.0 + rng.gauss(0, 0.05))
        earned = max(0.0, earned)
        eco.credits[cid] = eco.credits.get(cid, 0.0) + earned
        total_awarded += earned
    result.credits_awarded = total_awarded

    # -- 3. Taxation & redistribution --
    policy = determine_policy(gov_type)
    eco.policy = policy
    result.policy = policy
    tax_rate = POLICY_TAX_RATES.get(policy, 0.20)
    tax_pool = 0.0
    for cid in active_ids:
        tax = eco.credits[cid] * tax_rate
        eco.credits[cid] -= tax
        tax_pool += tax
    result.tax_collected = tax_pool
    if active_ids and tax_pool > 0:
        share = tax_pool / len(active_ids)
        for cid in active_ids:
            eco.credits[cid] += share

    # -- 4. Bilateral trades based on trust --
    trades = 0
    shuffled = list(active_ids)
    rng.shuffle(shuffled)
    paired: set[str] = set()
    for i, cid_a in enumerate(shuffled):
        if cid_a in paired or trades >= MAX_TRADES_PER_YEAR:
            break
        best_partner = None
        best_trust = TRADE_TRUST_THRESHOLD
        for cid_b in shuffled[i + 1:]:
            if cid_b in paired:
                continue
            rel = social_get(cid_a, cid_b)
            trust = rel.trust if hasattr(rel, "trust") else 0.0
            if trust > best_trust:
                best_trust = trust
                best_partner = cid_b
        if best_partner is None:
            continue
        # Wealthier partner shares surplus with poorer
        a_credits = eco.credits.get(cid_a, 0.0)
        b_credits = eco.credits.get(best_partner, 0.0)
        if a_credits > b_credits:
            donor, receiver = cid_a, best_partner
            donor_credits = a_credits
        else:
            donor, receiver = best_partner, cid_a
            donor_credits = b_credits
        transfer = donor_credits * TRADE_FRACTION * best_trust
        transfer = max(0.0, transfer)
        eco.credits[donor] -= transfer
        eco.credits[receiver] += transfer
        paired.add(cid_a)
        paired.add(best_partner)
        trades += 1
    result.trades_executed = trades
    eco.total_trades += trades

    # -- 5. Prune dead/exiled accounts (keep for history but zero out) --
    for cid in list(eco.credits):
        if cid not in active_ids:
            eco.credits[cid] = 0.0

    # -- 6. Compute metrics --
    active_credits = [eco.credits.get(cid, 0.0) for cid in active_ids]
    result.gini = compute_gini(active_credits)
    eco.gini_history.append(result.gini)

    from src.mars100.engine import ACTIONS as ACTION_POOL
    result.diversity = compute_diversity(actions, ACTION_POOL)

    result.efficiency_bonus = compute_efficiency_bonus(result.gini, result.diversity)

    return result
