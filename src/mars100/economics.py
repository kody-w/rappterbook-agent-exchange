"""
Economics organ for Mars-100 colony simulation.

Models personal resource stockpiles, barter trade between colonists,
inequality metrics (GINI coefficient), and emergent currency detection.

Key design: economics operates AFTER the shared resource tick but BEFORE
death checks.  Colonists divert a fraction of each year's production
surplus into personal reserves.  When colony commons dip below survival
thresholds, private reserves can be burned to avoid starvation deaths.

The split between personal and commons depends on governance type and
individual hoarding stats.  Trade happens along the social graph: high-
trust pairs exchange surplus for deficit resources at favorable rates.
Inequality feeds back into governance proposals — high GINI triggers
redistribution votes.

Currency emerges organically when one resource dominates barter trades
over a sustained period (>40% of exchanges over 10+ years, with a
minimum absolute trade count).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


# -- constants ---------------------------------------------------------------

RESOURCE_NAMES = ("food", "water", "power", "air", "medicine")

# Fraction of surplus diverted to personal stockpile (base, before mods)
BASE_DIVERSION = 0.20

# Governance modifiers to diversion fraction
GOVERNANCE_DIVERSION: dict[str, float] = {
    "council": 0.20,
    "dictator": 0.10,
    "lottery": 0.30,
    "consensus": 0.25,
    "ai_governor": 0.15,
    "anarchy": 0.50,
}

# Don't divert from commons if resource is below this floor
COMMONS_FLOOR = 0.30

# Trust threshold for trade
TRADE_TRUST_THRESHOLD = 0.40

# Maximum diversion fraction (even with high hoarding + anarchy)
MAX_DIVERSION = 0.60

# Minimum trades needed before currency can emerge
MIN_TRADES_FOR_CURRENCY = 15

# Fraction of trades a single resource must dominate for currency
CURRENCY_DOMINANCE_THRESHOLD = 0.40

# Personal stockpile decay per year (spoilage)
STOCKPILE_DECAY = 0.05

# Max personal stockpile per resource
MAX_STOCKPILE = 1.0

# GINI threshold that triggers governance proposals
INEQUALITY_GOVERNANCE_THRESHOLD = 0.55

# Years of trade history to consider for currency detection
CURRENCY_LOOKBACK_YEARS = 10


# -- data classes ------------------------------------------------------------

@dataclass
class PersonalStockpile:
    """A colonist's personal resource reserves."""
    food: float = 0.0
    water: float = 0.0
    power: float = 0.0
    air: float = 0.0
    medicine: float = 0.0

    def total(self) -> float:
        return sum(getattr(self, n) for n in RESOURCE_NAMES)

    def to_dict(self) -> dict[str, float]:
        return {n: round(getattr(self, n), 4) for n in RESOURCE_NAMES}

    def clamp(self) -> None:
        for n in RESOURCE_NAMES:
            val = getattr(self, n)
            setattr(self, n, max(0.0, min(MAX_STOCKPILE, val)))


@dataclass
class TradeRecord:
    """Record of a single barter exchange."""
    year: int
    from_id: str
    to_id: str
    gave_resource: str
    gave_amount: float
    received_resource: str
    received_amount: float

    def to_dict(self) -> dict:
        return {
            "year": self.year, "from": self.from_id, "to": self.to_id,
            "gave": self.gave_resource, "gave_amt": round(self.gave_amount, 4),
            "received": self.received_resource,
            "received_amt": round(self.received_amount, 4),
        }


@dataclass
class EconomicsState:
    """Colony-wide economic state persisted across years."""
    stockpiles: dict[str, PersonalStockpile] = field(default_factory=dict)
    trade_history: list[TradeRecord] = field(default_factory=list)
    gini_history: list[float] = field(default_factory=list)
    currency_resource: str | None = None
    currency_emerged_year: int | None = None
    total_trades: int = 0

    def to_dict(self) -> dict:
        return {
            "stockpiles": {k: v.to_dict() for k, v in self.stockpiles.items()},
            "gini_history": [round(g, 4) for g in self.gini_history[-20:]],
            "currency_resource": self.currency_resource,
            "currency_emerged_year": self.currency_emerged_year,
            "total_trades": self.total_trades,
            "recent_trades": [t.to_dict() for t in self.trade_history[-10:]],
        }

    def summary(self) -> dict:
        """Compact summary for year results."""
        current_gini = self.gini_history[-1] if self.gini_history else 0.0
        return {
            "gini": round(current_gini, 4),
            "total_trades": self.total_trades,
            "currency": self.currency_resource,
            "currency_year": self.currency_emerged_year,
            "stockpile_count": len(self.stockpiles),
        }


@dataclass
class EconomicTickResult:
    """Result of one year's economic activity."""
    year: int
    diversions: dict[str, dict[str, float]] = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)
    gini: float = 0.0
    currency_emerged: bool = False
    redistribution_triggered: bool = False
    survival_burns: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "trade_count": len(self.trades),
            "gini": round(self.gini, 4),
            "currency_emerged": self.currency_emerged,
            "redistribution_triggered": self.redistribution_triggered,
            "survival_burn_count": len(self.survival_burns),
        }


# -- initialization ----------------------------------------------------------

def initialize_stockpiles(colonist_ids: list[str],
                          rng: random.Random) -> dict[str, PersonalStockpile]:
    """Give each colonist a small initial personal stockpile."""
    stockpiles: dict[str, PersonalStockpile] = {}
    for cid in colonist_ids:
        sp = PersonalStockpile()
        for n in RESOURCE_NAMES:
            setattr(sp, n, max(0.0, rng.gauss(0.05, 0.02)))
        sp.clamp()
        stockpiles[cid] = sp
    return stockpiles


# -- diversion ---------------------------------------------------------------

def _compute_diversion_fraction(governance_type: str,
                                hoarding_stat: float) -> float:
    """Compute the fraction of surplus a colonist diverts to personal use."""
    base = GOVERNANCE_DIVERSION.get(governance_type, BASE_DIVERSION)
    # Hoarding stat adds up to 0.15 extra diversion
    personal = base + hoarding_stat * 0.15
    return min(MAX_DIVERSION, max(0.0, personal))


def _divert_surplus(stockpiles: dict[str, PersonalStockpile],
                    colonists: list, resource_delta: dict[str, float],
                    commons: object, governance_type: str,
                    rng: random.Random) -> dict[str, dict[str, float]]:
    """Divert a fraction of positive resource deltas to personal stockpiles.

    Only diverts when commons is above the floor. Returns a dict of
    colonist_id → {resource: amount_diverted}.
    """
    diversions: dict[str, dict[str, float]] = {}
    active_count = len(colonists)
    if active_count == 0:
        return diversions

    for colonist in colonists:
        cid = colonist.id
        if cid not in stockpiles:
            stockpiles[cid] = PersonalStockpile()

        frac = _compute_diversion_fraction(
            governance_type, colonist.stats.hoarding)
        per_colonist: dict[str, float] = {}

        for res_name in RESOURCE_NAMES:
            delta = resource_delta.get(res_name, 0.0)
            commons_level = getattr(commons, res_name, 0.0)

            # Only divert from positive deltas when commons is healthy
            if delta <= 0 or commons_level < COMMONS_FLOOR:
                continue

            # Per-colonist share of the surplus
            share = delta / active_count
            divert_amount = share * frac * rng.uniform(0.8, 1.0)
            divert_amount = min(divert_amount, 0.05)  # cap per-year

            # Actually remove from commons and add to personal
            current_commons = getattr(commons, res_name)
            if current_commons - divert_amount >= COMMONS_FLOOR:
                setattr(commons, res_name, current_commons - divert_amount)
                current_personal = getattr(stockpiles[cid], res_name)
                setattr(stockpiles[cid], res_name,
                        min(MAX_STOCKPILE, current_personal + divert_amount))
                per_colonist[res_name] = divert_amount

        if per_colonist:
            diversions[cid] = per_colonist

    return diversions


# -- trade -------------------------------------------------------------------

def _find_trade_pairs(colonists: list, stockpiles: dict[str, PersonalStockpile],
                      social: object, rng: random.Random) -> list[tuple]:
    """Find potential trade pairs based on complementary needs and trust."""
    pairs: list[tuple] = []
    ids = [c.id for c in colonists]

    for i, c_a in enumerate(colonists):
        for c_b in colonists[i + 1:]:
            trust_ab = social.get(c_a.id, c_b.id).trust
            trust_ba = social.get(c_b.id, c_a.id).trust
            avg_trust = (trust_ab + trust_ba) / 2

            if avg_trust < TRADE_TRUST_THRESHOLD:
                continue

            sp_a = stockpiles.get(c_a.id, PersonalStockpile())
            sp_b = stockpiles.get(c_b.id, PersonalStockpile())

            # Find complementary surplus/deficit
            a_surplus = max(RESOURCE_NAMES,
                            key=lambda n: getattr(sp_a, n))
            b_surplus = max(RESOURCE_NAMES,
                            key=lambda n: getattr(sp_b, n))

            a_deficit = min(RESOURCE_NAMES,
                            key=lambda n: getattr(sp_a, n))
            b_deficit = min(RESOURCE_NAMES,
                            key=lambda n: getattr(sp_b, n))

            # Trade if A's surplus matches B's deficit or vice versa
            if (a_surplus == b_deficit and getattr(sp_a, a_surplus) > 0.02
                    and b_surplus == a_deficit
                    and getattr(sp_b, b_surplus) > 0.02):
                pairs.append((c_a, c_b, a_surplus, b_surplus, avg_trust))

    rng.shuffle(pairs)
    return pairs


def _execute_trades(pairs: list[tuple],
                    stockpiles: dict[str, PersonalStockpile],
                    year: int, rng: random.Random) -> list[TradeRecord]:
    """Execute barter trades between paired colonists."""
    records: list[TradeRecord] = []
    traded_this_year: set[str] = set()

    for c_a, c_b, res_a, res_b, trust in pairs:
        # Each colonist trades at most once per year
        if c_a.id in traded_this_year or c_b.id in traded_this_year:
            continue

        sp_a = stockpiles[c_a.id]
        sp_b = stockpiles[c_b.id]

        # Trade amount scales with trust
        amount_a = min(getattr(sp_a, res_a) * 0.3,
                       trust * 0.05 * rng.uniform(0.8, 1.2))
        amount_b = min(getattr(sp_b, res_b) * 0.3,
                       trust * 0.05 * rng.uniform(0.8, 1.2))

        if amount_a < 0.001 or amount_b < 0.001:
            continue

        # Execute the exchange
        setattr(sp_a, res_a, getattr(sp_a, res_a) - amount_a)
        setattr(sp_a, res_b, getattr(sp_a, res_b) + amount_b)
        setattr(sp_b, res_b, getattr(sp_b, res_b) - amount_b)
        setattr(sp_b, res_a, getattr(sp_b, res_a) + amount_a)

        sp_a.clamp()
        sp_b.clamp()

        records.append(TradeRecord(
            year=year, from_id=c_a.id, to_id=c_b.id,
            gave_resource=res_a, gave_amount=amount_a,
            received_resource=res_b, received_amount=amount_b,
        ))

        traded_this_year.add(c_a.id)
        traded_this_year.add(c_b.id)

    return records


# -- survival burns ----------------------------------------------------------

def burn_stockpile_for_survival(colonist_id: str,
                                stockpiles: dict[str, PersonalStockpile],
                                critical_resources: list[str],
                                commons: object) -> list[dict]:
    """Burn personal reserves to prevent starvation when commons are low.

    Returns a list of burn records.
    """
    burns: list[dict] = []
    sp = stockpiles.get(colonist_id)
    if sp is None:
        return burns

    for res_name in critical_resources:
        personal_val = getattr(sp, res_name, 0.0)
        if personal_val > 0.01:
            burn_amount = min(personal_val, 0.05)
            setattr(sp, res_name, personal_val - burn_amount)
            # Contribute back to commons
            commons_val = getattr(commons, res_name, 0.0)
            setattr(commons, res_name, min(1.0, commons_val + burn_amount))
            burns.append({
                "colonist_id": colonist_id,
                "resource": res_name,
                "amount": round(burn_amount, 4),
            })

    return burns


# -- inequality metrics ------------------------------------------------------

def compute_gini(stockpiles: dict[str, PersonalStockpile]) -> float:
    """Compute GINI coefficient across all colonist total holdings.

    Returns 0.0 (perfect equality) to approaching 1.0 (one has all).
    """
    if len(stockpiles) < 2:
        return 0.0

    totals = sorted(sp.total() for sp in stockpiles.values())
    n = len(totals)
    total_sum = sum(totals)

    if total_sum < 0.001:
        return 0.0  # everyone has nothing — that's equal

    # Standard GINI formula
    numerator = sum((2 * (i + 1) - n - 1) * totals[i] for i in range(n))
    return numerator / (n * total_sum)


# -- currency detection ------------------------------------------------------

def detect_currency(trade_history: list[TradeRecord],
                    current_year: int) -> str | None:
    """Detect if a resource has emerged as de facto currency.

    Requires minimum trade volume and dominance over a lookback window.
    """
    cutoff_year = current_year - CURRENCY_LOOKBACK_YEARS
    recent = [t for t in trade_history if t.year >= cutoff_year]

    if len(recent) < MIN_TRADES_FOR_CURRENCY:
        return None

    # Count how often each resource appears in trades (given or received)
    counts: dict[str, int] = {}
    total = 0
    for trade in recent:
        counts[trade.gave_resource] = counts.get(trade.gave_resource, 0) + 1
        counts[trade.received_resource] = counts.get(
            trade.received_resource, 0) + 1
        total += 2

    if total == 0:
        return None

    # Find the most-traded resource
    dominant = max(counts, key=counts.get)
    if counts[dominant] / total >= CURRENCY_DOMINANCE_THRESHOLD:
        return dominant

    return None


# -- economic pressure -------------------------------------------------------

def compute_economic_pressure(state: EconomicsState) -> dict[str, float]:
    """Compute action weight modifiers from economic conditions.

    High inequality encourages cooperation and mediation.
    Currency emergence encourages coding (market systems).
    """
    pressure: dict[str, float] = {}
    gini = state.gini_history[-1] if state.gini_history else 0.0

    if gini > 0.5:
        pressure["cooperate"] = 0.1 * gini
        pressure["mediate"] = 0.05 * gini
        pressure["hoard"] = -0.1 * gini
    if gini < 0.2:
        pressure["hoard"] = 0.05  # low inequality → less stigma

    if state.currency_resource is not None:
        pressure["code"] = 0.05  # incentivize market systems

    return pressure


# -- stockpile decay ---------------------------------------------------------

def _decay_stockpiles(stockpiles: dict[str, PersonalStockpile]) -> None:
    """Apply yearly spoilage to all personal stockpiles."""
    for sp in stockpiles.values():
        for n in RESOURCE_NAMES:
            current = getattr(sp, n)
            setattr(sp, n, max(0.0, current * (1.0 - STOCKPILE_DECAY)))


# -- add/remove colonists ---------------------------------------------------

def add_colonist_stockpile(state: EconomicsState, colonist_id: str,
                           rng: random.Random) -> None:
    """Initialize stockpile for a new colonist (birth or immigrant)."""
    sp = PersonalStockpile()
    for n in RESOURCE_NAMES:
        setattr(sp, n, max(0.0, rng.gauss(0.03, 0.01)))
    sp.clamp()
    state.stockpiles[colonist_id] = sp


def archive_colonist_stockpile(state: EconomicsState, colonist_id: str,
                                commons: object) -> dict[str, float]:
    """When a colonist dies/is exiled, return their stockpile to commons.

    Legacy, not delete: the stockpile record stays in history but
    resources flow back to the colony.
    """
    sp = state.stockpiles.get(colonist_id)
    returned: dict[str, float] = {}
    if sp is None:
        return returned

    for n in RESOURCE_NAMES:
        val = getattr(sp, n, 0.0)
        if val > 0:
            commons_val = getattr(commons, n, 0.0)
            setattr(commons, n, min(1.0, commons_val + val))
            returned[n] = round(val, 4)
        setattr(sp, n, 0.0)

    return returned


# -- main tick ---------------------------------------------------------------

def tick_economics(state: EconomicsState, colonists: list,
                   resources: object, resource_delta: dict[str, float],
                   social: object, governance_type: str,
                   year: int, rng: random.Random) -> EconomicTickResult:
    """Advance colony economics by one Martian year.

    Called after resource tick, before death checks.

    1. Spoilage on existing stockpiles
    2. Divert surplus from commons to personal stockpiles
    3. Trade between trusted colonist pairs
    4. Compute GINI and detect currency
    5. Return result for logging
    """
    result = EconomicTickResult(year=year)
    active = [c for c in colonists if c.is_active()]

    if not active:
        return result

    # 1. Decay existing stockpiles
    _decay_stockpiles(state.stockpiles)

    # 2. Divert surplus
    diversions = _divert_surplus(
        state.stockpiles, active, resource_delta,
        resources, governance_type, rng)
    result.diversions = {k: {r: round(v, 4) for r, v in d.items()}
                         for k, d in diversions.items()}

    # 3. Trade
    pairs = _find_trade_pairs(active, state.stockpiles, social, rng)
    trades = _execute_trades(pairs, state.stockpiles, year, rng)
    state.trade_history.extend(trades)
    state.total_trades += len(trades)
    result.trades = [t.to_dict() for t in trades]

    # Trim trade history to last 50 years
    if len(state.trade_history) > 200:
        cutoff = year - 50
        state.trade_history = [t for t in state.trade_history
                                if t.year >= cutoff]

    # 4. GINI
    gini = compute_gini(state.stockpiles)
    state.gini_history.append(gini)
    if len(state.gini_history) > 100:
        state.gini_history = state.gini_history[-100:]
    result.gini = gini

    # 5. Currency detection
    if state.currency_resource is None:
        currency = detect_currency(state.trade_history, year)
        if currency is not None:
            state.currency_resource = currency
            state.currency_emerged_year = year
            result.currency_emerged = True

    # 6. Check if inequality triggers redistribution pressure
    if gini > INEQUALITY_GOVERNANCE_THRESHOLD:
        result.redistribution_triggered = True

    return result
