"""
Earth relations organ for Mars-100.

Models the persistent relationship between Mars colony and Earth:
supply ships, funding cycles, political opinion, and the independence arc.

Earth provides maintenance supplies (spare parts, specialty equipment)
that reduce infrastructure operating costs. It does NOT inject raw
resources — the colony must feed and water itself. The dependency
creates genuine tension: Earth support keeps the machines running
cheaply, but political strings come attached.

Key dynamics:
  - Supply ships launch every 2 Mars-years (launch windows), transit 1 year
  - Earth opinion drifts based on colony performance and crises
  - Funding tracks opinion with lag — bureaucracy is slow
  - Policy ranges from supportive → neutral → restrictive → hostile
  - Independence becomes possible after year 40 with sufficient population
    and resources; requires a governance vote to declare
  - After independence, maintenance modifier worsens but self-determination
    is gained; ships already in transit still arrive
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# -- constants ---------------------------------------------------------------

LAUNCH_WINDOW_INTERVAL = 2   # Mars years between launch windows
TRANSIT_TIME = 1             # Mars years for ship transit
SHIP_LOSS_BASE_PROB = 0.05   # 5% base chance of losing a ship in transit

INDEPENDENCE_MIN_YEAR = 40
INDEPENDENCE_MIN_POP = 12
INDEPENDENCE_MIN_RESOURCES = 0.55

POLICY_THRESHOLDS = {
    "supportive": 0.65,
    "neutral": 0.40,
    "restrictive": 0.20,
    # below 0.20 → hostile
}

MAINTENANCE_MODIFIERS = {
    "supportive": 0.7,
    "neutral": 0.9,
    "restrictive": 1.1,
    "hostile": 1.3,
    "independent": 1.4,
}


# -- data classes ------------------------------------------------------------

@dataclass
class SupplyShip:
    """A supply ship in transit or arrived."""
    id: str
    launched_year: int
    arrival_year: int
    cargo: dict[str, float] = field(default_factory=dict)
    lost: bool = False
    arrived: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "launched_year": self.launched_year,
            "arrival_year": self.arrival_year, "cargo": dict(self.cargo),
            "lost": self.lost, "arrived": self.arrived,
        }


@dataclass
class EarthMessage:
    """A message from Earth (news, policy change, cultural exchange)."""
    year: int
    category: str   # "policy", "news", "cultural", "warning"
    content: str

    def to_dict(self) -> dict:
        return {"year": self.year, "category": self.category,
                "content": self.content}


@dataclass
class EarthState:
    """Persistent Earth relationship state across simulation years."""
    opinion: float = 0.6           # 0.0 (hostile) → 1.0 (enthusiastic)
    funding: float = 0.5           # 0.0 (cut off) → 1.0 (generous)
    policy: str = "neutral"
    independent: bool = False
    independence_year: int | None = None
    next_launch_window: int = 2
    ships_launched: int = 0
    ships_lost: int = 0
    ships_arrived: int = 0
    ships_in_transit: list[SupplyShip] = field(default_factory=list)
    messages: list[EarthMessage] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "opinion": round(self.opinion, 4),
            "funding": round(self.funding, 4),
            "policy": self.policy,
            "independent": self.independent,
            "independence_year": self.independence_year,
            "next_launch_window": self.next_launch_window,
            "ships_launched": self.ships_launched,
            "ships_lost": self.ships_lost,
            "ships_arrived": self.ships_arrived,
            "ships_in_transit": [s.to_dict() for s in self.ships_in_transit],
            "messages": [m.to_dict() for m in self.messages[-10:]],
        }


@dataclass
class EarthTickResult:
    """Result of one year's Earth interaction."""
    year: int
    arrivals: list[dict] = field(default_factory=list)
    losses: list[str] = field(default_factory=list)
    ship_launched: dict | None = None
    messages: list[dict] = field(default_factory=list)
    maintenance_modifier: float = 1.0
    opinion_delta: float = 0.0
    funding_delta: float = 0.0
    policy_changed: bool = False
    independence_declared: bool = False

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "arrivals": self.arrivals,
            "losses": self.losses,
            "ship_launched": self.ship_launched,
            "messages": self.messages,
            "maintenance_modifier": round(self.maintenance_modifier, 4),
            "opinion_delta": round(self.opinion_delta, 4),
            "funding_delta": round(self.funding_delta, 4),
            "policy_changed": self.policy_changed,
            "independence_declared": self.independence_declared,
        }


# -- cargo generation --------------------------------------------------------

def _generate_cargo(funding: float, rng: random.Random) -> dict[str, float]:
    """Generate supply ship cargo based on current funding level.

    Cargo is spare parts and specialty supplies that reduce maintenance
    costs — not raw food/water/power.
    """
    base = 0.3 + funding * 0.5   # 0.3–0.8 based on funding
    return {
        "spare_parts": round(base * rng.uniform(0.8, 1.2), 3),
        "medical_supplies": round(base * 0.4 * rng.uniform(0.7, 1.3), 3),
        "scientific_equipment": round(base * 0.3 * rng.uniform(0.6, 1.4), 3),
    }


# -- opinion / policy / funding updates -------------------------------------

def _update_opinion(state: EarthState, year: int, death_count: int,
                    resource_avg: float, rng: random.Random) -> float:
    """Update Earth's opinion of the colony. Returns delta."""
    delta = 0.0
    # Deaths erode support
    delta -= death_count * 0.04
    # Good resource management impresses
    if resource_avg > 0.6:
        delta += 0.02
    elif resource_avg < 0.3:
        delta -= 0.03
    # Slow drift toward apathy over time
    if year > 50:
        delta -= 0.005
    # Random noise (public opinion is fickle)
    delta += rng.gauss(0, 0.015)
    state.opinion = max(0.0, min(1.0, state.opinion + delta))
    return delta


def _update_policy(state: EarthState) -> bool:
    """Update Earth policy based on opinion. Returns True if changed."""
    old = state.policy
    if state.independent:
        state.policy = "independent"
    elif state.opinion >= POLICY_THRESHOLDS["supportive"]:
        state.policy = "supportive"
    elif state.opinion >= POLICY_THRESHOLDS["neutral"]:
        state.policy = "neutral"
    elif state.opinion >= POLICY_THRESHOLDS["restrictive"]:
        state.policy = "restrictive"
    else:
        state.policy = "hostile"
    return state.policy != old


def _update_funding(state: EarthState, rng: random.Random) -> float:
    """Update funding level — tracks opinion with bureaucratic lag."""
    if state.independent:
        target = 0.0
    elif state.policy == "hostile":
        target = 0.05
    else:
        target = state.opinion * 0.8
    # Funding moves slowly toward target (bureaucracy)
    delta = (target - state.funding) * 0.3 + rng.gauss(0, 0.01)
    state.funding = max(0.0, min(1.0, state.funding + delta))
    return delta


# -- ship management ---------------------------------------------------------

def _maybe_launch_ship(state: EarthState, year: int,
                       rng: random.Random) -> SupplyShip | None:
    """Launch a supply ship if it's a launch window and policy allows."""
    if year < state.next_launch_window:
        return None

    # Always advance the window, even if we don't launch
    state.next_launch_window = year + LAUNCH_WINDOW_INTERVAL

    if state.independent:
        return None
    if state.policy == "hostile" and rng.random() > 0.2:
        return None  # hostile Earth rarely sends ships
    if state.funding < 0.1:
        return None

    ship_id = f"ship-{state.ships_launched + 1}"
    cargo = _generate_cargo(state.funding, rng)
    ship = SupplyShip(
        id=ship_id, launched_year=year,
        arrival_year=year + TRANSIT_TIME, cargo=cargo,
    )
    state.ships_in_transit.append(ship)
    state.ships_launched += 1
    return ship


def _process_arrivals(state: EarthState, year: int,
                      rng: random.Random) -> tuple[list[SupplyShip], list[str]]:
    """Process ships that arrive or are lost this year."""
    arrivals: list[SupplyShip] = []
    losses: list[str] = []
    still_in_transit: list[SupplyShip] = []

    for ship in state.ships_in_transit:
        if ship.arrival_year > year:
            still_in_transit.append(ship)
            continue
        # Ship arrives — check for loss
        loss_prob = SHIP_LOSS_BASE_PROB
        if state.policy == "hostile":
            loss_prob += 0.03  # hostile Earth → less careful launches
        if rng.random() < loss_prob:
            ship.lost = True
            state.ships_lost += 1
            losses.append(ship.id)
        else:
            ship.arrived = True
            state.ships_arrived += 1
            arrivals.append(ship)

    state.ships_in_transit = still_in_transit
    return arrivals, losses


# -- maintenance modifier ----------------------------------------------------

def compute_maintenance_modifier(state: EarthState) -> float:
    """Compute the infrastructure operating cost multiplier.

    Lower is better: supportive Earth provides spare parts that reduce
    maintenance. Independent colony must source everything locally.
    """
    base = MAINTENANCE_MODIFIERS.get(state.policy, 1.0)
    # Recent arrivals provide a small bonus
    if state.ships_arrived > 0 and not state.independent:
        bonus = min(0.1, state.ships_arrived * 0.01)
        base = max(0.6, base - bonus)
    return round(base, 3)


# -- independence ------------------------------------------------------------

def check_independence_conditions(year: int, population: int,
                                  resource_avg: float,
                                  state: EarthState) -> bool:
    """Check if the colony meets the prerequisites for independence."""
    if state.independent:
        return False
    if year < INDEPENDENCE_MIN_YEAR:
        return False
    if population < INDEPENDENCE_MIN_POP:
        return False
    if resource_avg < INDEPENDENCE_MIN_RESOURCES:
        return False
    return True


def declare_independence(state: EarthState, year: int) -> None:
    """Declare independence from Earth."""
    state.independent = True
    state.independence_year = year
    state.policy = "independent"
    state.messages.append(EarthMessage(
        year=year, category="policy",
        content="The colony has declared independence from Earth.",
    ))


# -- messages ----------------------------------------------------------------

def _generate_messages(state: EarthState, year: int,
                       arrivals: list[SupplyShip], losses: list[str],
                       rng: random.Random) -> list[EarthMessage]:
    """Generate Earth messages for the year."""
    msgs: list[EarthMessage] = []

    for ship in arrivals:
        msgs.append(EarthMessage(
            year=year, category="news",
            content=f"Supply ship {ship.id} arrived with cargo: {ship.cargo}",
        ))

    for ship_id in losses:
        msgs.append(EarthMessage(
            year=year, category="warning",
            content=f"Supply ship {ship_id} lost in transit.",
        ))

    # Occasional policy commentary
    if year % 5 == 0 and not state.independent:
        sentiment = {
            "supportive": "Earth reaffirms commitment to Mars program.",
            "neutral": "Mars program funding under routine review.",
            "restrictive": "Budget committee proposes Mars funding cuts.",
            "hostile": "Public protests demand end to Mars funding.",
        }
        msg = sentiment.get(state.policy, "")
        if msg:
            msgs.append(EarthMessage(year=year, category="policy", content=msg))

    return msgs


# -- main tick ---------------------------------------------------------------

def tick_earth(state: EarthState, year: int, death_count: int,
               resource_avg: float, population: int,
               rng: random.Random) -> EarthTickResult:
    """Advance Earth relations by one Martian year.

    Called after resource tick and deaths. Returns a result with
    arrivals, messages, and the maintenance modifier for this year.
    """
    result = EarthTickResult(year=year)

    # 1. Process ship arrivals/losses first
    arrivals, losses = _process_arrivals(state, year, rng)
    result.arrivals = [s.to_dict() for s in arrivals]
    result.losses = losses

    # 2. Update opinion based on colony performance
    result.opinion_delta = _update_opinion(
        state, year, death_count, resource_avg, rng)

    # 3. Update policy
    result.policy_changed = _update_policy(state)

    # 4. Update funding (tracks opinion with lag)
    result.funding_delta = _update_funding(state, rng)

    # 5. Maybe launch a new ship
    ship = _maybe_launch_ship(state, year, rng)
    if ship:
        result.ship_launched = ship.to_dict()

    # 6. Generate messages
    msgs = _generate_messages(state, year, arrivals, losses, rng)
    state.messages.extend(msgs)
    result.messages = [m.to_dict() for m in msgs]

    # 7. Compute maintenance modifier for this year
    result.maintenance_modifier = compute_maintenance_modifier(state)

    return result
