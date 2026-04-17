"""
Earth parallel evolution for Mars-100.

Earth exists alongside the colony as a living entity that evolves over
100 years — technology advancing, climate deteriorating, political
interest in Mars waxing and waning.  Earth sends signals (supply drops,
immigration offers, demands, embargoes) that inject external pressure
into colony governance and resources.  Climate collapse permanently
severs contact.

Two-phase integration in the engine tick:
  1. ``apply_signal_effects`` — before resource tick, applies last year's
     queued signal effects to colony resources.
  2. ``tick_earth`` + ``generate_signal`` — after the year resolves,
     Earth observes colony outcomes and decides next year's signal.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

SIGNAL_TYPES = (
    "supply_drop", "demand_tribute", "immigration_offer",
    "tech_transfer", "embargo", "radio_silence",
)

# Tunables
TECH_GROWTH = 0.004
STABILITY_DRIFT = 0.025
CLIMATE_BASE_GROWTH = 0.006
CLIMATE_SPIKE_PROB = 0.08
INTEREST_DECAY = 0.003
SIGNAL_BASE_PROB = 0.35
COLLAPSE_THRESHOLD = 0.92


@dataclass
class EarthState:
    """Earth's parallel state across 100 Martian years."""
    tech_level: float = 0.50
    political_stability: float = 0.70
    mars_interest: float = 0.60
    climate_crisis: float = 0.15
    contact_active: bool = True
    signals: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tech_level": round(self.tech_level, 4),
            "political_stability": round(self.political_stability, 4),
            "mars_interest": round(self.mars_interest, 4),
            "climate_crisis": round(self.climate_crisis, 4),
            "contact_active": self.contact_active,
            "total_signals": len(self.signals),
        }

    def snapshot(self) -> dict:
        """Lightweight per-year snapshot (no signal history)."""
        return {
            "tech": round(self.tech_level, 4),
            "stability": round(self.political_stability, 4),
            "interest": round(self.mars_interest, 4),
            "climate": round(self.climate_crisis, 4),
            "contact": self.contact_active,
        }


@dataclass
class EarthSignal:
    """A signal sent from Earth to the colony."""
    year: int
    signal_type: str
    magnitude: float
    description: str

    def to_dict(self) -> dict:
        return {
            "year": self.year, "signal_type": self.signal_type,
            "magnitude": round(self.magnitude, 4),
            "description": self.description,
        }


def _clamp(v: float) -> float:
    """Clamp a value to [0.0, 1.0]."""
    return max(0.0, min(1.0, v))


def tick_earth(state: EarthState, year: int,
               colony_pop: int, colony_resource_avg: float,
               rng: random.Random) -> None:
    """Advance Earth's internal state by one Martian year.

    Called *after* the colony year resolves so Earth reacts to outcomes.
    """
    if not state.contact_active:
        return

    # Technology grows steadily
    state.tech_level = _clamp(state.tech_level + TECH_GROWTH
                              + rng.gauss(0, 0.002))

    # Political stability random-walks
    state.political_stability = _clamp(
        state.political_stability + rng.gauss(0, STABILITY_DRIFT))

    # Climate crisis accumulates, occasional spikes
    climate_delta = CLIMATE_BASE_GROWTH + rng.gauss(0, 0.003)
    if rng.random() < CLIMATE_SPIKE_PROB:
        climate_delta += rng.uniform(0.03, 0.08)
    state.climate_crisis = _clamp(state.climate_crisis + climate_delta)

    # Mars interest influenced by colony success
    interest_boost = 0.0
    if colony_pop > 15:
        interest_boost += 0.005
    if colony_resource_avg > 0.5:
        interest_boost += 0.004
    state.mars_interest = _clamp(
        state.mars_interest - INTEREST_DECAY + interest_boost
        + rng.gauss(0, 0.01))

    # Collapse check — permanent contact loss
    if state.climate_crisis >= COLLAPSE_THRESHOLD:
        state.contact_active = False


def generate_signal(state: EarthState, year: int,
                    colony_pop: int, colony_resource_avg: float,
                    rng: random.Random) -> EarthSignal | None:
    """Decide whether Earth sends a signal this year.

    Called after ``tick_earth``.  Returns None most years.
    """
    if not state.contact_active:
        return None

    prob = (SIGNAL_BASE_PROB
            * state.mars_interest
            * state.political_stability)
    if rng.random() > prob:
        return None

    return _choose_signal(state, year, colony_pop,
                          colony_resource_avg, rng)


# ------------------------------------------------------------------
# Signal generation helpers
# ------------------------------------------------------------------

_SIGNAL_DESCRIPTIONS: dict[str, list[str]] = {
    "supply_drop": [
        "Earth cargo pod enters Mars orbit — food and medicine inbound.",
        "Automated resupply vessel docks at the colony landing pad.",
        "Supply capsule from Earth survives aerobraking — rations secured.",
    ],
    "demand_tribute": [
        "Earth legislature demands mineral shipment as colony tax.",
        "Earth corporation asserts IP rights over colony research output.",
        "Political faction on Earth calls for Mars colony to 'pay its debt'.",
    ],
    "immigration_offer": [
        "Earth offers to send a trained specialist on next launch window.",
        "Immigration lottery winner requests transfer to Mars colony.",
        "Earth diplomat proposes cultural exchange — one colonist inbound.",
    ],
    "tech_transfer": [
        "Earth uplinks blueprints for improved power grid components.",
        "New medical techniques transmitted — colony medicine advances.",
        "Earth shares breakthrough water-recycling schematics.",
    ],
    "embargo": [
        "Earth alliance votes to suspend Mars supply missions.",
        "Political crisis on Earth freezes all interplanetary shipments.",
        "Trade embargo imposed — colony must be fully self-sufficient.",
    ],
    "radio_silence": [
        "No transmissions from Earth this year — static on all bands.",
        "Earth's deep-space antenna array offline for maintenance.",
        "Solar interference blocks Earth-Mars communication window.",
    ],
}


def _choose_signal(state: EarthState, year: int,
                   colony_pop: int, colony_resource_avg: float,
                   rng: random.Random) -> EarthSignal:
    """Weight signal types by Earth's current state and colony needs."""
    weights: dict[str, float] = {}

    # Supply drop — high interest + stability + colony struggling
    weights["supply_drop"] = (state.mars_interest * 0.4
                              + state.political_stability * 0.3
                              + max(0, 0.5 - colony_resource_avg) * 0.6)

    # Demand tribute — low stability, high interest
    weights["demand_tribute"] = (max(0, 0.7 - state.political_stability) * 0.5
                                 + state.mars_interest * 0.3)

    # Immigration — high tech + colony healthy
    weights["immigration_offer"] = (state.tech_level * 0.4
                                    + min(colony_resource_avg, 0.6) * 0.4)
    if colony_pop < 5:
        weights["immigration_offer"] *= 0.2  # too risky

    # Tech transfer — high tech + colony struggling
    weights["tech_transfer"] = (state.tech_level * 0.5
                                + max(0, 0.5 - colony_resource_avg) * 0.3)

    # Embargo — low stability
    weights["embargo"] = max(0, 0.6 - state.political_stability) * 0.6

    # Radio silence — low interest or high climate crisis
    weights["radio_silence"] = (max(0, 0.5 - state.mars_interest) * 0.4
                                + state.climate_crisis * 0.3)

    # Ensure all positive
    weights = {k: max(0.01, v) for k, v in weights.items()}

    total = sum(weights.values())
    r = rng.random() * total
    cumulative = 0.0
    chosen = "radio_silence"
    for sig_type, w in weights.items():
        cumulative += w
        if r <= cumulative:
            chosen = sig_type
            break

    magnitude = rng.uniform(0.3, 0.9)
    desc = rng.choice(_SIGNAL_DESCRIPTIONS[chosen])

    signal = EarthSignal(year=year, signal_type=chosen,
                         magnitude=magnitude, description=desc)
    state.signals.append(signal.to_dict())
    return signal


# ------------------------------------------------------------------
# Signal effect application
# ------------------------------------------------------------------

def apply_signal_effects(signal: EarthSignal | None) -> dict[str, float]:
    """Compute resource deltas from an Earth signal.

    Returns a dict of resource-name → delta.  The engine applies these
    during the resource tick phase.  Immigration and other non-resource
    effects are handled by the engine reading ``signal.signal_type``.
    """
    if signal is None:
        return {}

    m = signal.magnitude
    effects: dict[str, float] = {}

    if signal.signal_type == "supply_drop":
        effects = {"food": 0.10 * m, "water": 0.06 * m,
                   "medicine": 0.08 * m}
    elif signal.signal_type == "demand_tribute":
        effects = {"food": -0.06 * m, "power": -0.04 * m}
    elif signal.signal_type == "tech_transfer":
        effects = {"power": 0.06 * m, "medicine": 0.03 * m}
    elif signal.signal_type == "embargo":
        effects = {"food": -0.04 * m, "water": -0.03 * m,
                   "medicine": -0.05 * m}
    # immigration_offer and radio_silence have no direct resource effects

    return effects
