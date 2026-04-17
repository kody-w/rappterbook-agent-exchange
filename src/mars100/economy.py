"""
Economy organ for Mars-100.

Personal inventories, specialization streaks, Gini inequality tracking,
black market mechanics, and economic pressure on governance.

Conservation rule: colony_resources + sum(personal_stockpiles) = total.
Personal stockpiles are private ration caches drawn from the colony.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


TRADEABLE_RESOURCES = ("food", "water", "medicine")

HOARD_RATE = 0.03
HOARD_COLONY_COST = 0.03
BLACK_MARKET_RATIO = 2.0
BLACK_MARKET_TRUST_PENALTY = 0.08
SPECIALIZATION_THRESHOLD = 5
SPECIALIZATION_BONUS = 0.20
STOCKPILE_DECAY = 0.02
GINI_GOVERNANCE_THRESHOLD = 0.6
GINI_REDISTRIBUTION_THRESHOLD = 0.8
MAX_BLACK_MARKET_PER_YEAR = 2
TRADE_TRUST_THRESHOLD = 0.55
MAX_TRADES_PER_COLONIST = 1


@dataclass
class Stockpile:
    """Personal resource cache for a colonist."""
    food: float = 0.0
    water: float = 0.0
    medicine: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"food": self.food, "water": self.water, "medicine": self.medicine}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Stockpile:
        return cls(
            food=d.get("food", 0.0),
            water=d.get("water", 0.0),
            medicine=d.get("medicine", 0.0),
        )

    def total(self) -> float:
        return self.food + self.water + self.medicine

    def clamp(self) -> None:
        self.food = max(0.0, min(1.0, self.food))
        self.water = max(0.0, min(1.0, self.water))
        self.medicine = max(0.0, min(1.0, self.medicine))

    def decay(self, rate: float = STOCKPILE_DECAY) -> dict[str, float]:
        """Apply spoilage decay. Returns amounts lost."""
        lost: dict[str, float] = {}
        for name in TRADEABLE_RESOURCES:
            current = getattr(self, name)
            loss = current * rate
            setattr(self, name, current - loss)
            lost[name] = loss
        self.clamp()
        return lost


@dataclass
class Specialization:
    """Track a colonist's action specialization streak."""
    last_action: str = ""
    consecutive_years: int = 0

    def update(self, action: str) -> None:
        """Update streak based on this year's action."""
        if action == self.last_action:
            self.consecutive_years += 1
        else:
            self.last_action = action
            self.consecutive_years = 1

    @property
    def is_specialized(self) -> bool:
        return self.consecutive_years >= SPECIALIZATION_THRESHOLD

    @property
    def bonus(self) -> float:
        if not self.is_specialized:
            return 0.0
        return SPECIALIZATION_BONUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_action": self.last_action,
            "consecutive_years": self.consecutive_years,
            "is_specialized": self.is_specialized,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Specialization:
        return cls(
            last_action=d.get("last_action", ""),
            consecutive_years=d.get("consecutive_years", 0),
        )


@dataclass
class TradeRecord:
    """Record of a resource transfer between colonists."""
    year: int
    from_id: str
    to_id: str
    resource: str
    amount: float
    trust_before: float
    trust_after: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year, "from_id": self.from_id, "to_id": self.to_id,
            "resource": self.resource, "amount": self.amount,
            "trust_before": self.trust_before, "trust_after": self.trust_after,
        }


@dataclass
class BlackMarketRecord:
    """Record of a black market transaction."""
    year: int
    colonist_id: str
    resource: str
    colony_cost: float
    personal_gain: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year, "colonist_id": self.colonist_id,
            "resource": self.resource, "colony_cost": self.colony_cost,
            "personal_gain": self.personal_gain,
        }


@dataclass
class EconomyState:
    """Colony-wide economic state for one year."""
    gini: float = 0.0
    total_wealth: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    black_market: list[BlackMarketRecord] = field(default_factory=list)
    redistribution_triggered: bool = False
    specializations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "gini": round(self.gini, 4),
            "total_wealth": round(self.total_wealth, 4),
            "trade_count": len(self.trades),
            "trades": [t.to_dict() for t in self.trades],
            "black_market_count": len(self.black_market),
            "black_market": [b.to_dict() for b in self.black_market],
            "redistribution_triggered": self.redistribution_triggered,
            "specializations": self.specializations,
        }


def compute_gini(values: list[float]) -> float:
    """Compute the Gini coefficient for a list of non-negative values.

    Returns 0.0 for perfect equality, approaches 1.0 for maximum inequality.
    Returns 0.0 for empty or single-element lists.
    """
    if len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    weighted_sum = 0.0
    for i, val in enumerate(sorted_vals):
        cumulative += val
        weighted_sum += (2 * (i + 1) - n - 1) * val
    return weighted_sum / (n * total)


def process_hoarding(
    colonist_id: str,
    action: str,
    stockpile: Stockpile,
    colony_resources: Any,
    hoarding_stat: float,
    rng: random.Random,
) -> dict[str, float]:
    """Process hoarding action: transfer colony resources to personal stockpile.

    Returns dict of amounts transferred (positive = gained personally).
    Colony resources are reduced by the same amounts.
    """
    if action != "hoard":
        return {}

    transfers: dict[str, float] = {}
    for resource in TRADEABLE_RESOURCES:
        colony_val = getattr(colony_resources, resource, 0.0)
        amount = HOARD_RATE * hoarding_stat * rng.uniform(0.5, 1.0)
        actual = min(amount, colony_val * 0.5)
        if actual > 0.001:
            setattr(colony_resources, resource, colony_val - actual)
            current_personal = getattr(stockpile, resource)
            setattr(stockpile, resource, min(1.0, current_personal + actual))
            transfers[resource] = actual

    return transfers


def process_trade(
    from_id: str,
    to_id: str,
    from_stockpile: Stockpile,
    to_stockpile: Stockpile,
    trust: float,
    empathy_from: float,
    empathy_to: float,
    year: int,
    rng: random.Random,
) -> TradeRecord | None:
    """Attempt a resource transfer from one colonist to another.

    High-trust, high-empathy colonists are more generous.
    Returns None if trade doesn't happen.
    """
    if trust < TRADE_TRUST_THRESHOLD:
        return None

    generosity = empathy_from * trust * rng.uniform(0.3, 1.0)
    if generosity < 0.2:
        return None

    # Find resource where donor has surplus and recipient has need
    best_resource = None
    best_delta = -1.0
    for resource in TRADEABLE_RESOURCES:
        donor_val = getattr(from_stockpile, resource)
        recipient_val = getattr(to_stockpile, resource)
        delta = donor_val - recipient_val
        if delta > 0.02 and donor_val > 0.05 and delta > best_delta:
            best_resource = resource
            best_delta = delta

    if best_resource is None:
        return None

    amount = min(best_delta * 0.5 * generosity, getattr(from_stockpile, best_resource) * 0.3)
    if amount < 0.005:
        return None

    trust_before = trust

    from_val = getattr(from_stockpile, best_resource)
    to_val = getattr(to_stockpile, best_resource)
    setattr(from_stockpile, best_resource, from_val - amount)
    setattr(to_stockpile, best_resource, min(1.0, to_val + amount))
    from_stockpile.clamp()
    to_stockpile.clamp()

    trust_after = min(1.0, trust + 0.03)

    return TradeRecord(
        year=year, from_id=from_id, to_id=to_id,
        resource=best_resource, amount=round(amount, 4),
        trust_before=round(trust_before, 4),
        trust_after=round(trust_after, 4),
    )


def process_black_market(
    colonist_id: str,
    stockpile: Stockpile,
    colony_resources: Any,
    sabotage_skill: float,
    paranoia: float,
    rng: random.Random,
    year: int,
) -> BlackMarketRecord | None:
    """Attempt a black market transaction when resources are critical.

    Sabotage-skilled, paranoid colonists extract colony resources at
    a 2:1 ratio (wasteful). Returns None if conditions not met.
    """
    if sabotage_skill < 0.5:
        return None
    if paranoia < 0.4:
        return None

    critical = []
    for resource in TRADEABLE_RESOURCES:
        if getattr(colony_resources, resource, 1.0) < 0.3:
            critical.append(resource)

    if not critical:
        return None

    if rng.random() > sabotage_skill * 0.6:
        return None

    resource = rng.choice(critical)
    colony_val = getattr(colony_resources, resource)
    colony_cost = min(0.04 * sabotage_skill, colony_val * 0.3)
    personal_gain = colony_cost / BLACK_MARKET_RATIO

    if colony_cost < 0.005:
        return None

    setattr(colony_resources, resource, colony_val - colony_cost)
    current_personal = getattr(stockpile, resource)
    setattr(stockpile, resource, min(1.0, current_personal + personal_gain))

    return BlackMarketRecord(
        year=year, colonist_id=colonist_id, resource=resource,
        colony_cost=round(colony_cost, 4),
        personal_gain=round(personal_gain, 4),
    )


def liquidate_stockpile(
    stockpile: Stockpile,
    colony_resources: Any,
) -> dict[str, float]:
    """Return a dead/exiled colonist's stockpile to the colony.

    Returns the amounts returned.
    """
    returned: dict[str, float] = {}
    for resource in TRADEABLE_RESOURCES:
        amount = getattr(stockpile, resource)
        if amount > 0.001:
            colony_val = getattr(colony_resources, resource, 0.0)
            setattr(colony_resources, resource, min(1.0, colony_val + amount))
            setattr(stockpile, resource, 0.0)
            returned[resource] = amount
    return returned


def redistribute_wealth(
    stockpiles: dict[str, Stockpile],
    active_ids: list[str],
) -> dict[str, float]:
    """Redistribute personal stockpiles equally among active colonists.

    Triggered when Gini exceeds redistribution threshold.
    Returns per-colonist allocation amounts.
    """
    if not active_ids:
        return {}

    totals: dict[str, float] = {r: 0.0 for r in TRADEABLE_RESOURCES}
    for cid in active_ids:
        sp = stockpiles.get(cid)
        if sp:
            for resource in TRADEABLE_RESOURCES:
                totals[resource] += getattr(sp, resource)

    per_capita: dict[str, float] = {
        r: totals[r] / len(active_ids) for r in TRADEABLE_RESOURCES
    }

    for cid in active_ids:
        sp = stockpiles.get(cid)
        if sp:
            for resource in TRADEABLE_RESOURCES:
                setattr(sp, resource, per_capita[resource])
            sp.clamp()

    return per_capita


def tick_economy(
    colonists: list[Any],
    actions: dict[str, str],
    colony_resources: Any,
    social_graph: Any,
    stockpiles: dict[str, Stockpile],
    specializations: dict[str, Specialization],
    year: int,
    rng: random.Random,
) -> EconomyState:
    """Run one year of economic simulation.

    Phases:
    1. Decay personal stockpiles (spoilage)
    2. Process hoarding actions
    3. Update specialization streaks
    4. Process trades between high-trust pairs
    5. Process black market transactions
    6. Compute Gini coefficient
    7. Check for redistribution trigger
    """
    active = [c for c in colonists if c.is_active()]
    active_ids = [c.id for c in active]
    economy = EconomyState()

    # Ensure all active colonists have entries
    for c in active:
        if c.id not in stockpiles:
            stockpiles[c.id] = Stockpile()
        if c.id not in specializations:
            specializations[c.id] = Specialization()

    # Phase 1: Decay
    for cid in active_ids:
        stockpiles[cid].decay()

    # Phase 2: Hoarding
    for c in active:
        action = actions.get(c.id, "rest")
        process_hoarding(c.id, action, stockpiles[c.id], colony_resources,
                         c.stats.hoarding, rng)

    # Phase 3: Specialization
    spec_count = 0
    for c in active:
        action = actions.get(c.id, "rest")
        specializations[c.id].update(action)
        if specializations[c.id].is_specialized:
            spec_count += 1
    economy.specializations = spec_count

    # Phase 4: Trades (max 1 per colonist)
    traded_this_year: set[str] = set()
    shuffled_active = list(active)
    rng.shuffle(shuffled_active)
    for c in shuffled_active:
        if c.id in traded_this_year:
            continue
        if c.stats.empathy < 0.3:
            continue
        if stockpiles[c.id].total() < 0.03:
            continue

        # Find best trade partner
        best_partner = None
        best_trust = 0.0
        for other in active:
            if other.id == c.id or other.id in traded_this_year:
                continue
            rel = social_graph.get(c.id, other.id)
            if rel.trust > best_trust and rel.trust >= TRADE_TRUST_THRESHOLD:
                best_partner = other
                best_trust = rel.trust

        if best_partner is None:
            continue

        record = process_trade(
            from_id=c.id, to_id=best_partner.id,
            from_stockpile=stockpiles[c.id],
            to_stockpile=stockpiles[best_partner.id],
            trust=best_trust,
            empathy_from=c.stats.empathy,
            empathy_to=best_partner.stats.empathy,
            year=year, rng=rng,
        )
        if record:
            economy.trades.append(record)
            traded_this_year.add(c.id)
            traded_this_year.add(best_partner.id)
            # Update social trust
            social_graph.update_from_cooperation(c.id, best_partner.id, rng)

    # Phase 5: Black market
    bm_count = 0
    for c in active:
        if bm_count >= MAX_BLACK_MARKET_PER_YEAR:
            break
        record = process_black_market(
            colonist_id=c.id, stockpile=stockpiles[c.id],
            colony_resources=colony_resources,
            sabotage_skill=c.skills.sabotage, paranoia=c.stats.paranoia,
            rng=rng, year=year,
        )
        if record:
            economy.black_market.append(record)
            bm_count += 1
            # Trust penalty from all other colonists
            for other in active:
                if other.id != c.id:
                    social_graph.update_from_conflict(c.id, other.id, rng)

    # Phase 6: Gini
    wealth_values = [stockpiles[cid].total() for cid in active_ids]
    economy.gini = compute_gini(wealth_values)
    economy.total_wealth = sum(wealth_values)

    # Phase 7: Redistribution
    if economy.gini > GINI_REDISTRIBUTION_THRESHOLD and len(active_ids) >= 3:
        redistribute_wealth(stockpiles, active_ids)
        economy.redistribution_triggered = True
        # Recompute gini after redistribution
        wealth_values = [stockpiles[cid].total() for cid in active_ids]
        economy.gini = compute_gini(wealth_values)
        economy.total_wealth = sum(wealth_values)

    return economy


def get_specialization_bonus(
    colonist_id: str,
    action: str,
    specializations: dict[str, Specialization],
) -> float:
    """Return the specialization bonus multiplier for a colonist's action.

    Returns 0.0 if not specialized in this action, SPECIALIZATION_BONUS if they are.
    """
    spec = specializations.get(colonist_id)
    if spec is None:
        return 0.0
    if spec.last_action == action and spec.is_specialized:
        return spec.bonus
    return 0.0
