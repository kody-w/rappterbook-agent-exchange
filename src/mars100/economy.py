"""
Mars-100 economy organ.

Scarcity-driven wages, specialisation tracking, Gini coefficient.
Integrates into the engine via a single tick_economy() call per year.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def compute_gini(balances: list[float]) -> float:
    """Gini coefficient (0 = perfect equality, 1 = one colonist has everything)."""
    n = len(balances)
    if n == 0:
        return 0.0
    total = sum(balances)
    if total <= 0:
        return 0.0
    sorted_b = sorted(balances)
    weighted_sum = sum((2 * (i + 1) - n - 1) * b for i, b in enumerate(sorted_b))
    return weighted_sum / (n * total)


def scarcity_multiplier(resource_level: float, threshold: float = 0.3) -> float:
    """Pay premium when a resource is scarce. Returns >= 1.0."""
    if resource_level >= threshold:
        return 1.0
    if resource_level <= 0:
        return 3.0
    return 1.0 + 2.0 * (1.0 - resource_level / threshold)


RESOURCE_ACTIONS = {
    "terraform": "materials",
    "farm": "food",
    "cooperate": "air",
    "explore": "energy",
    "rest": "water",
    "research": "energy",
}

BASE_WAGE = 10.0
TAX_RATE = 0.10


@dataclass
class ColonistLedger:
    """Economic state for one colonist."""
    balance: float = 0.0
    lifetime_earnings: float = 0.0
    action_counts: dict[str, int] = field(default_factory=dict)

    def record_action(self, action: str, wage: float) -> None:
        """Record an action and pay the colonist (minus tax)."""
        tax = wage * TAX_RATE
        net = wage - tax
        self.balance += net
        self.lifetime_earnings += net
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

    def specialisation(self) -> dict[str, float]:
        """Fraction of lifetime actions per action type."""
        total = sum(self.action_counts.values())
        if total == 0:
            return {}
        return {a: c / total for a, c in self.action_counts.items()}

    def specialist_bonus(self) -> float:
        """Bonus multiplier if colonist has >40% of actions in one type."""
        spec = self.specialisation()
        if not spec:
            return 0.0
        top = max(spec.values())
        if top < 0.4:
            return 0.0
        return min(0.30, (top - 0.4) * 0.5)

    def to_dict(self) -> dict:
        return {
            "balance": round(self.balance, 2),
            "lifetime_earnings": round(self.lifetime_earnings, 2),
            "action_counts": dict(self.action_counts),
            "specialisation": {k: round(v, 3) for k, v in self.specialisation().items()},
            "specialist_bonus": round(self.specialist_bonus(), 3),
        }


@dataclass
class EconomyState:
    """Colony-wide economic state."""
    ledgers: dict[int, ColonistLedger] = field(default_factory=dict)
    treasury: float = 0.0
    total_minted: float = 0.0
    year_wages: float = 0.0

    def get_ledger(self, colonist_id: int) -> ColonistLedger:
        """Get or create a ledger for a colonist."""
        if colonist_id not in self.ledgers:
            self.ledgers[colonist_id] = ColonistLedger()
        return self.ledgers[colonist_id]

    def gini(self) -> float:
        """Colony-wide Gini coefficient."""
        return compute_gini([l.balance for l in self.ledgers.values()])

    def to_dict(self) -> dict:
        return {
            "treasury": round(self.treasury, 2),
            "total_minted": round(self.total_minted, 2),
            "gini": round(self.gini(), 4),
            "ledgers": {str(k): v.to_dict() for k, v in self.ledgers.items()},
        }


def tick_economy(
    economy: EconomyState,
    alive_colonists: list,
    resources,
    actions: dict,
    year: int,
) -> dict:
    """Process one year of economic activity. Returns summary dict.

    actions: dict mapping colonist_id -> action_name (engine format).
    """
    economy.year_wages = 0.0

    for cid, action in actions.items():

        ledger = economy.get_ledger(cid)
        wage = BASE_WAGE

        resource_key = RESOURCE_ACTIONS.get(action)
        if resource_key is not None:
            level = getattr(resources, resource_key, 1.0)
            wage *= scarcity_multiplier(level)

        wage *= (1.0 + ledger.specialist_bonus())

        tax = wage * TAX_RATE
        economy.treasury += tax
        economy.total_minted += wage
        economy.year_wages += wage

        ledger.record_action(action, wage)

    return {
        "year": year,
        "treasury": round(economy.treasury, 2),
        "total_minted": round(economy.total_minted, 2),
        "gini": round(economy.gini(), 4),
        "year_wages": round(economy.year_wages, 2),
        "alive_count": len(alive_colonists),
    }
