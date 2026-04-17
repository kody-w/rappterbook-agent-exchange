"""
Economics organ for Mars-100.

Models a credit-based economy where colonists earn credits from labor,
trade small amounts based on social trust, and pay taxes. Colony policy
(free_market / mixed / collectivist) determines redistribution.

Credits are an abstraction over contribution — they do NOT represent
private ownership of physical resources (food/water/power/air/medicine
remain a shared pool). Instead credits affect:
  - Governance pressure: high inequality triggers proposals
  - Social dynamics: wealth disparity breeds resentment
  - Hoard action: now earns credits instead of only reducing shared food
  - Earth relations: GDP reported in diplomatic messages

Conservation law: total credits = sum(agent.wealth) + treasury.
Credits are created only through production and destroyed only through
explicit destruction events (none in v1). This means total_credits
is monotonically non-decreasing.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


# -- constants ---------------------------------------------------------------

DEFAULT_TAX_RATE = 0.15
MAX_TAX_RATE = 0.60
MIN_TAX_RATE = 0.0
MAX_TRADE_PARTNERS = 3       # cap pairwise trades per colonist per year
TRADE_TRUST_THRESHOLD = 0.55  # minimum trust to trade
TRADE_FRACTION = 0.10         # max fraction of wealth transferred per trade
UBI_FLOOR = 0.05              # minimum UBI payout under collectivist policy

# Production multipliers per action
PRODUCTION_TABLE: dict[str, tuple[str, float]] = {
    "terraform": ("terraforming", 0.12),
    "farm":      ("hydroponics", 0.14),
    "code":      ("coding", 0.10),
    "research":  ("coding", 0.08),
    "mediate":   ("mediation", 0.06),
    "pray":      ("prayer", 0.04),
    "cooperate": ("mediation", 0.07),
    "explore":   ("terraforming", 0.05),
    "hoard":     ("hoarding", 0.09),    # hoarding is a stat, not a skill
    "sabotage":  ("sabotage", 0.02),
    "rest":      ("", 0.01),
}


# -- data classes ------------------------------------------------------------

@dataclass
class EconomicAgent:
    """Per-colonist economic state."""
    colonist_id: str
    wealth: float = 0.0
    labor_output: float = 0.0
    tax_paid: float = 0.0
    trade_balance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "colonist_id": self.colonist_id,
            "wealth": round(self.wealth, 4),
            "labor_output": round(self.labor_output, 4),
            "tax_paid": round(self.tax_paid, 4),
            "trade_balance": round(self.trade_balance, 4),
        }


@dataclass
class EconomicState:
    """Colony-level economic state."""
    agents: dict[str, EconomicAgent] = field(default_factory=dict)
    treasury: float = 0.0
    tax_rate: float = DEFAULT_TAX_RATE
    policy: str = "mixed"     # "free_market" | "mixed" | "collectivist"
    gini: float = 0.0
    gdp: float = 0.0         # total production this year (credits created)
    year_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "treasury": round(self.treasury, 4),
            "tax_rate": round(self.tax_rate, 4),
            "policy": self.policy,
            "gini": round(self.gini, 4),
            "gdp": round(self.gdp, 4),
            "agent_count": len(self.agents),
            "total_wealth": round(sum(a.wealth for a in self.agents.values()), 4),
            "year_history": self.year_history[-10:],
        }

    def summary(self) -> dict:
        """Short summary for YearResult."""
        return {
            "gdp": round(self.gdp, 4),
            "gini": round(self.gini, 4),
            "treasury": round(self.treasury, 4),
            "tax_rate": round(self.tax_rate, 4),
            "policy": self.policy,
            "total_wealth": round(
                sum(a.wealth for a in self.agents.values()), 4),
        }


# -- pure functions ----------------------------------------------------------

def compute_gini(wealth_values: list[float]) -> float:
    """Compute Gini coefficient from a list of wealth values.

    Returns 0.0 for perfect equality, approaches 1.0 for extreme inequality.
    Handles edge cases: empty list → 0.0, all zeros → 0.0, single value → 0.0.
    """
    n = len(wealth_values)
    if n < 2:
        return 0.0
    total = sum(wealth_values)
    if total <= 0.0:
        return 0.0
    sorted_w = sorted(wealth_values)
    cumulative = 0.0
    weighted_sum = 0.0
    for i, w in enumerate(sorted_w):
        cumulative += w
        weighted_sum += (2 * (i + 1) - n - 1) * w
    return weighted_sum / (n * total)


def produce(action: str, skills: dict[str, float],
            stats: dict[str, float], resource_avg: float,
            rng: random.Random) -> float:
    """Compute labor output (credits) for one colonist-year.

    Production depends on the chosen action, the relevant skill/stat,
    and a small random noise factor. Higher scarcity (lower resource_avg)
    slightly depresses output — hard to work when starving.
    """
    entry = PRODUCTION_TABLE.get(action, ("", 0.01))
    skill_or_stat_name, base_rate = entry

    if skill_or_stat_name:
        # Check skills first, then stats
        ability = skills.get(skill_or_stat_name,
                             stats.get(skill_or_stat_name, 0.3))
    else:
        ability = 0.3  # rest or unknown action

    scarcity_factor = 0.5 + 0.5 * min(1.0, resource_avg / 0.5)
    noise = rng.gauss(1.0, 0.1)
    output = base_rate * (0.3 + 0.7 * ability) * scarcity_factor * max(0.5, noise)
    return max(0.0, output)


def trade(state: EconomicState, social_get: object,
          active_ids: list[str], rng: random.Random) -> list[dict]:
    """Simulate bilateral credit trades based on social trust.

    Each colonist attempts up to MAX_TRADE_PARTNERS trades per year.
    Only pairs with trust > TRADE_TRUST_THRESHOLD trade.
    The wealthier partner transfers a small fraction to the poorer one,
    weighted by trust level. Returns a log of trades.

    social_get must be a callable(id_a, id_b) -> Relationship-like object
    with a .trust attribute.
    """
    trades: list[dict] = []
    if len(active_ids) < 2:
        return trades

    shuffled = list(active_ids)
    rng.shuffle(shuffled)

    trade_counts: dict[str, int] = {cid: 0 for cid in active_ids}

    for i, cid_a in enumerate(shuffled):
        if trade_counts[cid_a] >= MAX_TRADE_PARTNERS:
            continue
        agent_a = state.agents.get(cid_a)
        if agent_a is None:
            continue

        # Find trade partners (next few in shuffled order)
        for j in range(i + 1, min(i + 6, len(shuffled))):
            cid_b = shuffled[j]
            if trade_counts[cid_a] >= MAX_TRADE_PARTNERS:
                break
            if trade_counts[cid_b] >= MAX_TRADE_PARTNERS:
                continue
            agent_b = state.agents.get(cid_b)
            if agent_b is None:
                continue

            rel = social_get(cid_a, cid_b)
            trust = rel.trust if hasattr(rel, "trust") else 0.0
            if trust < TRADE_TRUST_THRESHOLD:
                continue

            # Wealthier gives to poorer, amount proportional to trust
            if agent_a.wealth >= agent_b.wealth:
                giver, receiver = agent_a, agent_b
            else:
                giver, receiver = agent_b, agent_a

            if giver.wealth <= 0.0:
                continue

            amount = giver.wealth * TRADE_FRACTION * trust * rng.uniform(0.5, 1.0)
            amount = min(amount, giver.wealth)  # can't give more than you have

            giver.wealth -= amount
            receiver.wealth += amount
            giver.trade_balance -= amount
            receiver.trade_balance += amount
            trade_counts[giver.colonist_id] += 1
            trade_counts[receiver.colonist_id] += 1

            trades.append({
                "from": giver.colonist_id,
                "to": receiver.colonist_id,
                "amount": round(amount, 4),
                "trust": round(trust, 2),
            })

    return trades


def tax_and_redistribute(state: EconomicState) -> dict:
    """Apply tax and redistribute based on colony policy.

    Conservation: total wealth before == total wealth after.
    Returns summary of tax/redistribution.
    """
    total_tax = 0.0
    for agent in state.agents.values():
        if agent.wealth <= 0.0:
            agent.tax_paid = 0.0
            continue
        tax = agent.wealth * state.tax_rate
        agent.wealth -= tax
        agent.tax_paid = tax
        total_tax += tax

    state.treasury += total_tax

    # Redistribute
    n = len(state.agents)
    distributed = 0.0
    if n > 0 and state.treasury > 0:
        if state.policy == "collectivist":
            # Equal distribution of entire treasury
            per_agent = state.treasury / n
            for agent in state.agents.values():
                agent.wealth += per_agent
                distributed += per_agent
            state.treasury = 0.0

        elif state.policy == "mixed":
            # Distribute half the treasury as UBI
            ubi_pool = state.treasury * 0.5
            per_agent = ubi_pool / n
            for agent in state.agents.values():
                agent.wealth += per_agent
                distributed += per_agent
            state.treasury -= ubi_pool

        # "free_market": no redistribution, treasury accumulates

    return {
        "total_tax": round(total_tax, 4),
        "distributed": round(distributed, 4),
        "treasury_after": round(state.treasury, 4),
        "policy": state.policy,
    }


def sync_agents(state: EconomicState, active_ids: list[str]) -> None:
    """Ensure economic agents match the current active colonist roster.

    Adds missing agents, removes dead/exiled ones (preserving their
    final wealth record in history but removing from active tracking).
    """
    active_set = set(active_ids)

    # Add new colonists
    for cid in active_ids:
        if cid not in state.agents:
            state.agents[cid] = EconomicAgent(colonist_id=cid)

    # Remove inactive colonists — transfer their wealth to treasury
    to_remove = [cid for cid in state.agents if cid not in active_set]
    for cid in to_remove:
        agent = state.agents.pop(cid)
        state.treasury += max(0.0, agent.wealth)


def tick_economics(
    state: EconomicState,
    active_ids: list[str],
    actions: dict[str, str],
    skills_map: dict[str, dict[str, float]],
    stats_map: dict[str, dict[str, float]],
    resource_avg: float,
    social_get: object,
    rng: random.Random,
) -> dict:
    """Advance the economy by one year.

    Steps:
    1. Sync agent roster
    2. Produce (earn credits from labor)
    3. Trade (trust-based credit transfers)
    4. Tax and redistribute
    5. Compute Gini
    6. Record history

    Returns a summary dict.
    """
    # 1. Sync roster
    sync_agents(state, active_ids)

    # Reset per-year accumulators
    for agent in state.agents.values():
        agent.labor_output = 0.0
        agent.tax_paid = 0.0
        agent.trade_balance = 0.0

    # 2. Production
    total_production = 0.0
    for cid in active_ids:
        agent = state.agents.get(cid)
        if agent is None:
            continue
        action = actions.get(cid, "rest")
        skills = skills_map.get(cid, {})
        stats = stats_map.get(cid, {})
        output = produce(action, skills, stats, resource_avg, rng)
        agent.wealth += output
        agent.labor_output = output
        total_production += output

    state.gdp = total_production

    # 3. Trade
    trade_log = trade(state, social_get, active_ids, rng)

    # 4. Tax and redistribute
    tax_summary = tax_and_redistribute(state)

    # 5. Gini coefficient
    wealth_values = [a.wealth for a in state.agents.values()]
    state.gini = compute_gini(wealth_values)

    # 6. Record history
    year_record = {
        "gdp": round(total_production, 4),
        "gini": round(state.gini, 4),
        "treasury": round(state.treasury, 4),
        "policy": state.policy,
        "trades": len(trade_log),
        "total_wealth": round(sum(a.wealth for a in state.agents.values()), 4),
    }
    state.year_history.append(year_record)
    if len(state.year_history) > 20:
        state.year_history = state.year_history[-20:]

    return {
        "gdp": round(total_production, 4),
        "gini": round(state.gini, 4),
        "treasury": round(state.treasury, 4),
        "tax_summary": tax_summary,
        "trade_count": len(trade_log),
        "trade_log": trade_log[:5],  # keep log small
        "policy": state.policy,
        "top_earners": _top_earners(state, 3),
        "wealth_distribution": _wealth_distribution(state),
    }


def economic_governance_pressure(state: EconomicState) -> float:
    """Return extra governance proposal probability from inequality.

    High Gini (>0.35) increases the chance of economic reform proposals.
    Returns a value 0.0-0.3 to add to base proposal probability.
    """
    if state.gini < 0.2:
        return 0.0
    return min(0.3, (state.gini - 0.2) * 0.75)


def _top_earners(state: EconomicState, n: int) -> list[dict]:
    """Return the top N earners by labor output."""
    sorted_agents = sorted(state.agents.values(),
                           key=lambda a: a.labor_output, reverse=True)
    return [{"id": a.colonist_id, "output": round(a.labor_output, 4)}
            for a in sorted_agents[:n]]


def _wealth_distribution(state: EconomicState) -> dict:
    """Summarize wealth distribution into quartiles."""
    if not state.agents:
        return {"q1": 0.0, "median": 0.0, "q3": 0.0, "max": 0.0}
    values = sorted(a.wealth for a in state.agents.values())
    n = len(values)
    return {
        "q1": round(values[n // 4] if n >= 4 else values[0], 4),
        "median": round(values[n // 2], 4),
        "q3": round(values[3 * n // 4] if n >= 4 else values[-1], 4),
        "max": round(values[-1], 4),
    }
