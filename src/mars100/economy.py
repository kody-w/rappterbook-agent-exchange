"""
Economy engine for Mars-100.

Manages individual colonist wealth, income from productive actions, theft via
sabotage, tax collection under governance, inheritance on death, and colony-wide
inequality metrics (Gini coefficient).

Resources remain communal — this module tracks *credits*, an abstract unit of
personal economic standing that influences governance proposals, social
dynamics, and colonist decision-making.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# Income rates per action, scaled by relevant skill (0.0-1.0)
ACTION_INCOME: dict[str, tuple[str, float]] = {
    # (skill_name, base_income)
    "terraform": ("terraforming", 0.06),
    "farm":      ("hydroponics", 0.07),
    "code":      ("coding", 0.05),
    "mediate":   ("mediation", 0.04),
    "pray":      ("prayer", 0.02),
    "cooperate": ("mediation", 0.03),
    "explore":   ("terraforming", 0.03),
    "rest":      ("", 0.01),
    "hoard":     ("", 0.0),   # hoarding earns nothing — it keeps more
    "sabotage":  ("sabotage", 0.0),  # income comes from theft instead
}

THEFT_FRACTION = 0.15  # fraction of victim's wealth stolen per sabotage
TAX_RATES: dict[str, float] = {
    "anarchy": 0.0,
    "council": 0.10,
    "dictator": 0.20,
    "lottery": 0.05,
    "consensus": 0.08,
    "ai_governor": 0.12,
    "direct_democracy": 0.10,
}
TREASURY_RESOURCE_CONVERSION = 0.02  # treasury → resource bonus per tick
MAX_WEALTH = 5.0  # wealth is uncapped upward to this ceiling
DEFAULT_STARTING_WEALTH = 0.1


@dataclass
class Wallet:
    """Personal wealth of a colonist."""
    credits: float = DEFAULT_STARTING_WEALTH
    lifetime_income: float = 0.0
    lifetime_tax_paid: float = 0.0
    lifetime_stolen: float = 0.0
    trades_completed: int = 0

    def clamp(self) -> None:
        self.credits = max(0.0, min(MAX_WEALTH, self.credits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "credits": round(self.credits, 4),
            "lifetime_income": round(self.lifetime_income, 4),
            "lifetime_tax_paid": round(self.lifetime_tax_paid, 4),
            "lifetime_stolen": round(self.lifetime_stolen, 4),
            "trades_completed": self.trades_completed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Wallet:
        return cls(
            credits=d.get("credits", DEFAULT_STARTING_WEALTH),
            lifetime_income=d.get("lifetime_income", 0.0),
            lifetime_tax_paid=d.get("lifetime_tax_paid", 0.0),
            lifetime_stolen=d.get("lifetime_stolen", 0.0),
            trades_completed=d.get("trades_completed", 0),
        )


@dataclass
class EconomySnapshot:
    """One year of colony economic activity."""
    year: int
    gini: float
    total_income: float
    total_tax: float
    total_theft: float
    treasury: float
    wealth_distribution: dict[str, float]  # colonist_id → credits
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "gini": round(self.gini, 4),
            "total_income": round(self.total_income, 4),
            "total_tax": round(self.total_tax, 4),
            "total_theft": round(self.total_theft, 4),
            "treasury": round(self.treasury, 4),
            "wealth_distribution": {k: round(v, 4) for k, v in self.wealth_distribution.items()},
            "events": self.events,
        }


@dataclass
class ColonyEconomy:
    """Colony-wide economic state."""
    wallets: dict[str, Wallet] = field(default_factory=dict)
    treasury: float = 0.0
    custom_tax_rate: float | None = None  # governance override
    year_trade_volume: int = 0

    def ensure_wallet(self, colonist_id: str) -> Wallet:
        """Get or create a wallet for a colonist."""
        if colonist_id not in self.wallets:
            self.wallets[colonist_id] = Wallet()
        return self.wallets[colonist_id]

    def wealth_of(self, colonist_id: str) -> float:
        """Get a colonist's current wealth."""
        return self.wallets.get(colonist_id, Wallet()).credits

    def to_dict(self) -> dict:
        return {
            "wallets": {k: v.to_dict() for k, v in self.wallets.items()},
            "treasury": round(self.treasury, 4),
            "custom_tax_rate": self.custom_tax_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ColonyEconomy:
        wallets = {k: Wallet.from_dict(v) for k, v in d.get("wallets", {}).items()}
        return cls(
            wallets=wallets,
            treasury=d.get("treasury", 0.0),
            custom_tax_rate=d.get("custom_tax_rate"),
        )


def compute_income(action: str, colonist: Any) -> float:
    """Compute income for a colonist based on their action and skill level.

    The hoarding stat acts as an income retention multiplier: high hoarding
    means less is lost to communal sharing (but earns social distrust).
    """
    skill_name, base = ACTION_INCOME.get(action, ("", 0.01))
    if skill_name:
        skill_val = getattr(colonist.skills, skill_name, 0.0)
        income = base * (0.5 + 0.5 * skill_val)
    else:
        income = base

    # Hoarding acts as retention multiplier — high hoarders keep more
    retention = 0.6 + 0.4 * colonist.stats.hoarding
    return income * retention


def process_theft(saboteur_id: str, victim_id: str, economy: ColonyEconomy,
                  rng: random.Random) -> dict | None:
    """Process a sabotage-as-theft event. Returns event dict or None."""
    saboteur_wallet = economy.ensure_wallet(saboteur_id)
    victim_wallet = economy.ensure_wallet(victim_id)

    if victim_wallet.credits < 0.01:
        return None

    stolen = victim_wallet.credits * THEFT_FRACTION * rng.uniform(0.5, 1.0)
    victim_wallet.credits -= stolen
    saboteur_wallet.credits += stolen
    saboteur_wallet.lifetime_stolen += stolen
    saboteur_wallet.clamp()
    victim_wallet.clamp()

    return {
        "type": "theft",
        "saboteur": saboteur_id,
        "victim": victim_id,
        "amount": round(stolen, 4),
    }


def collect_taxes(economy: ColonyEconomy, gov_type: str,
                  active_ids: list[str]) -> float:
    """Collect taxes from all active colonists based on governance type.

    High-hoarding colonists evade a fraction of their taxes.
    Returns total tax collected.
    """
    rate = economy.custom_tax_rate if economy.custom_tax_rate is not None else TAX_RATES.get(gov_type, 0.0)
    if rate <= 0.0:
        return 0.0

    total_collected = 0.0
    for cid in active_ids:
        wallet = economy.ensure_wallet(cid)
        # Import hoarding stat — needed for tax avoidance
        base_tax = wallet.credits * rate

        total_collected += base_tax
        wallet.credits -= base_tax
        wallet.lifetime_tax_paid += base_tax
        wallet.clamp()

    economy.treasury += total_collected
    return total_collected


def apply_tax_evasion(economy: ColonyEconomy, colonist: Any) -> float:
    """Compute tax evasion based on hoarding stat. Returns evaded fraction."""
    evasion = colonist.stats.hoarding * 0.3  # max 30% evasion
    return evasion


def process_inheritance(dead_id: str, children_ids: list[str],
                        economy: ColonyEconomy) -> dict:
    """Distribute a dead colonist's wealth.

    If children exist, split among them. Otherwise, goes to treasury.
    Returns inheritance event dict.
    """
    wallet = economy.wallets.get(dead_id, Wallet())
    amount = wallet.credits
    wallet.credits = 0.0

    if children_ids:
        share = amount / len(children_ids) if children_ids else 0.0
        for child_id in children_ids:
            child_wallet = economy.ensure_wallet(child_id)
            child_wallet.credits += share
            child_wallet.clamp()
        return {
            "type": "inheritance",
            "deceased": dead_id,
            "amount": round(amount, 4),
            "heirs": children_ids,
            "share_each": round(share, 4),
        }
    else:
        economy.treasury += amount
        return {
            "type": "inheritance_to_treasury",
            "deceased": dead_id,
            "amount": round(amount, 4),
        }


def spend_treasury(economy: ColonyEconomy, resources: Any) -> float:
    """Convert treasury into colony resource maintenance.

    Returns amount spent from treasury.
    """
    if economy.treasury <= 0.0:
        return 0.0

    from src.mars100.colony import RESOURCE_NAMES
    spend = min(economy.treasury, TREASURY_RESOURCE_CONVERSION * 5)
    per_resource = spend / len(RESOURCE_NAMES)
    for name in RESOURCE_NAMES:
        current = getattr(resources, name)
        setattr(resources, name, min(1.0, current + per_resource))
    economy.treasury -= spend
    return spend


def compute_gini(economy: ColonyEconomy, active_ids: list[str]) -> float:
    """Compute the Gini coefficient for active colonists.

    Returns 0.0 (perfect equality) to 1.0 (maximum inequality).
    """
    if len(active_ids) < 2:
        return 0.0

    values = sorted(economy.wealth_of(cid) for cid in active_ids)
    n = len(values)
    total = sum(values)
    if total <= 0.0:
        return 0.0

    # Gini via mean absolute difference
    numerator = sum(abs(values[i] - values[j])
                    for i in range(n) for j in range(n))
    return numerator / (2 * n * total)


def tick_economy(
    economy: ColonyEconomy,
    actions: dict[str, str],
    colonists: list[Any],
    active_ids: list[str],
    gov_type: str,
    year: int,
    social: Any,
    rng: random.Random,
    deaths: list[dict],
    lineage: dict[str, list[str]],
    resources: Any,
) -> EconomySnapshot:
    """Run one year of economic activity.

    Call order in engine.tick():
      1. After actions chosen and skill bonuses computed
      2. Before death/exile processing (so we can handle inheritance)
    """
    econ_events: list[dict] = []
    total_income = 0.0
    total_theft = 0.0

    colonist_map = {c.id: c for c in colonists}

    # Phase 1: Income
    for cid in active_ids:
        action = actions.get(cid, "rest")
        colonist = colonist_map.get(cid)
        if colonist is None:
            continue
        income = compute_income(action, colonist)
        wallet = economy.ensure_wallet(cid)
        wallet.credits += income
        wallet.lifetime_income += income
        wallet.clamp()
        total_income += income

    # Phase 2: Theft (sabotage actions steal from random colonists)
    for cid, action in actions.items():
        if action != "sabotage":
            continue
        possible_victims = [v for v in active_ids if v != cid]
        if not possible_victims:
            continue
        # Prefer least-trusted target
        if social:
            victim = social.most_trusted_by(cid, possible_victims)
            if victim is None:
                victim = rng.choice(possible_victims)
            # Actually saboteurs target enemies, not friends
            # Find LEAST trusted
            edges = social.edges.get(cid, {})
            if edges:
                candidates = [(v, edges[v].trust) for v in possible_victims if v in edges]
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    victim = candidates[0][0]
        else:
            victim = rng.choice(possible_victims)
        event = process_theft(cid, victim, economy, rng)
        if event:
            econ_events.append(event)
            total_theft += event["amount"]

    # Phase 3: Tax collection
    total_tax = collect_taxes(economy, gov_type, active_ids)
    if total_tax > 0:
        econ_events.append({
            "type": "tax_collection",
            "rate": economy.custom_tax_rate or TAX_RATES.get(gov_type, 0.0),
            "total": round(total_tax, 4),
        })

    # Phase 4: Treasury spending on resources
    treasury_spend = spend_treasury(economy, resources)
    if treasury_spend > 0:
        econ_events.append({
            "type": "treasury_spend",
            "amount": round(treasury_spend, 4),
        })

    # Phase 5: Inheritance for deaths this year
    for death in deaths:
        dead_id = death.get("id", "")
        children = lineage.get(dead_id, [])
        active_children = [c for c in children if c in active_ids]
        event = process_inheritance(dead_id, active_children, economy)
        econ_events.append(event)

    # Phase 6: Inequality crisis detection
    gini = compute_gini(economy, active_ids)
    if gini > 0.6:
        econ_events.append({
            "type": "inequality_crisis",
            "gini": round(gini, 4),
            "message": f"Year {year}: Wealth inequality at critical level (Gini {gini:.2f})",
        })

    wealth_dist = {cid: economy.wealth_of(cid) for cid in active_ids}

    return EconomySnapshot(
        year=year,
        gini=gini,
        total_income=total_income,
        total_tax=total_tax,
        total_theft=total_theft,
        treasury=economy.treasury,
        wealth_distribution=wealth_dist,
        events=econ_events,
    )
