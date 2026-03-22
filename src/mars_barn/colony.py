"""Colony population dynamics — logistic growth with resource constraints.

Each colony tracks: population, resources, infrastructure, tech level,
morale, and history. The population model uses a modified logistic equation
with stochastic perturbations and resource-limited carrying capacity.

dN/dt = r * N * (1 - N/K_eff) + noise

Where K_eff = min(K_base, K_resources) and K_resources is derived from
the most limiting resource (Liebig's law of the minimum).
"""
from __future__ import annotations

import math
import random


# Per-capita resource consumption (per sol)
CONSUMPTION = {
    "o2": 0.84,      # kg O2 per person per sol
    "h2o": 2.5,      # liters water per person per sol
    "food": 1.8,     # kg food per person per sol
    "power": 3.0,    # kWh per person per sol
}

# Colony presets
COLONY_PRESETS: dict[str, dict] = {
    "ares_prime": {
        "name": "Ares Prime",
        "population": 50,
        "carrying_capacity": 200,
        "birth_rate": 0.020,
        "death_rate": 0.005,
        "infrastructure": 1.2,
        "tech_level": 1.0,
        "morale": 0.8,
        "latitude": -4.5,
        "description": "Balanced colony in Jezero Crater. Reliable but conservative.",
        "resources": {"o2": 5000, "h2o": 20000, "food": 8000, "power": 10000},
        "resource_capacity": {"o2": 20000, "h2o": 80000, "food": 40000, "power": 50000},
    },
    "hellas_basin": {
        "name": "Hellas Basin",
        "population": 30,
        "carrying_capacity": 300,
        "birth_rate": 0.028,
        "death_rate": 0.008,
        "infrastructure": 0.8,
        "tech_level": 0.9,
        "morale": 0.7,
        "latitude": -42.7,
        "description": "Aggressive growth colony in the deepest basin. High risk, high reward.",
        "resources": {"o2": 3000, "h2o": 15000, "food": 5000, "power": 6000},
        "resource_capacity": {"o2": 30000, "h2o": 100000, "food": 50000, "power": 60000},
    },
    "olympus_station": {
        "name": "Olympus Station",
        "population": 20,
        "carrying_capacity": 150,
        "birth_rate": 0.015,
        "death_rate": 0.003,
        "infrastructure": 1.5,
        "tech_level": 1.3,
        "morale": 0.9,
        "latitude": 18.65,
        "description": "High-tech station near Olympus Mons. Small but resilient.",
        "resources": {"o2": 4000, "h2o": 18000, "food": 7000, "power": 15000},
        "resource_capacity": {"o2": 15000, "h2o": 60000, "food": 30000, "power": 80000},
    },
}


def create_colony(preset_key: str, index: int) -> dict:
    """Create a colony from a preset configuration."""
    preset = COLONY_PRESETS[preset_key]
    return {
        "index": index,
        "name": preset["name"],
        "preset": preset_key,
        "population": preset["population"],
        "carrying_capacity": preset["carrying_capacity"],
        "birth_rate": preset["birth_rate"],
        "death_rate": preset["death_rate"],
        "infrastructure": preset["infrastructure"],
        "tech_level": preset["tech_level"],
        "morale": preset["morale"],
        "latitude": preset["latitude"],
        "description": preset["description"],
        "resources": dict(preset["resources"]),
        "resource_capacity": dict(preset["resource_capacity"]),
        "births_total": 0,
        "deaths_total": 0,
        "peak_population": preset["population"],
        "history": [],
        "events_log": [],
        "active_storms": [],
    }


def effective_carrying_capacity(colony: dict) -> float:
    """Compute effective K from base capacity and resource limits.

    Uses Liebig's law: the most limiting resource sets the cap.
    """
    k_base = colony["carrying_capacity"]
    resources = colony["resources"]

    # How many people can each resource support for ~30 sols?
    resource_caps = []
    for res, rate in CONSUMPTION.items():
        stock = resources.get(res, 0)
        if rate > 0:
            supportable = stock / (rate * 30)  # 30-sol buffer
            resource_caps.append(supportable)

    if not resource_caps:
        return k_base

    k_resource = min(resource_caps)
    return min(k_base, max(k_resource, 2))  # floor of 2 to prevent div-by-zero


def tick_colony(
    colony: dict,
    sol: int,
    production: dict[str, float],
    events: list[dict],
    rng: random.Random,
) -> dict:
    """Advance colony by one sol. Returns a snapshot dict for history.

    This is the core population dynamics engine:
    1. Apply resource production and consumption
    2. Compute effective carrying capacity
    3. Apply logistic growth with stochastic noise
    4. Apply events (storms, epidemics, supply drops, etc.)
    5. Update infrastructure and tech (slow growth)
    6. Record snapshot
    """
    pop = colony["population"]
    if pop <= 0:
        colony["population"] = 0
        return _snapshot(colony, sol)

    # --- 1. Resources ---
    for res in CONSUMPTION:
        # Production
        colony["resources"][res] = min(
            colony["resources"].get(res, 0) + production.get(res, 0),
            colony["resource_capacity"].get(res, 10000),
        )
        # Consumption
        consumed = CONSUMPTION[res] * pop
        colony["resources"][res] = max(0, colony["resources"][res] - consumed)

    # --- 2. Effective carrying capacity ---
    k_eff = effective_carrying_capacity(colony)

    # --- 3. Logistic growth ---
    r = colony["birth_rate"] - colony["death_rate"]
    # Morale modifier
    r *= (0.5 + colony["morale"] * 0.5)

    # Stochastic noise (demographic + environmental)
    demographic_noise = rng.gauss(0, 0.003) * math.sqrt(pop)
    environmental_noise = rng.gauss(0, 0.005)

    # Growth rate with logistic term
    if k_eff > 0:
        growth_rate = r * pop * (1 - pop / k_eff) + demographic_noise + environmental_noise
    else:
        growth_rate = -pop * 0.1  # catastrophic decline

    # Resource stress: if any resource is critically low, increase death rate
    resource_stress = 0.0
    for res in CONSUMPTION:
        stock = colony["resources"].get(res, 0)
        daily_need = CONSUMPTION[res] * pop
        if daily_need > 0:
            days_remaining = stock / daily_need
            if days_remaining < 10:
                resource_stress = max(resource_stress, 1.0 - days_remaining / 10)

    if resource_stress > 0:
        starvation_deaths = pop * resource_stress * 0.02
        growth_rate -= starvation_deaths
        colony["morale"] = max(0.1, colony["morale"] - resource_stress * 0.05)

    # --- 4. Apply events ---
    birth_multiplier = 1.0
    for event in events:
        if event["target"] != colony["index"] and event["target"] != -1:
            continue

        etype = event["type"]
        mag = event["magnitude"]

        if etype == "dust_storm" or etype == "global_dust_storm":
            colony["active_storms"].append({
                "type": etype,
                "target": event["target"],
                "magnitude": mag,
                "remaining": event.get("duration", 5),
            })
            colony["morale"] = max(0.1, colony["morale"] - 0.05 * mag)

        elif etype == "equipment_failure":
            # Reduce infrastructure temporarily
            colony["infrastructure"] = max(0.3, colony["infrastructure"] - mag * 0.2)
            colony["morale"] = max(0.1, colony["morale"] - 0.03)

        elif etype == "supply_drop":
            # Boost resources and carrying capacity
            for res in colony["resources"]:
                colony["resources"][res] = min(
                    colony["resources"][res] + mag * 10,
                    colony["resource_capacity"][res],
                )
            colony["carrying_capacity"] += int(mag)
            colony["morale"] = min(1.0, colony["morale"] + 0.05)

        elif etype == "epidemic":
            # Direct population loss
            deaths = max(1, int(pop * mag))
            growth_rate -= deaths
            colony["morale"] = max(0.1, colony["morale"] - 0.1)

        elif etype == "birth_boom":
            birth_multiplier *= mag

        colony["events_log"].append({
            "sol": sol,
            "type": etype,
            "description": event["description"],
        })
        # Keep events log trimmed
        if len(colony["events_log"]) > 200:
            colony["events_log"] = colony["events_log"][-150:]

    growth_rate *= birth_multiplier

    # --- 5. Apply population change ---
    delta = growth_rate
    # Discretize: at least track births and deaths separately
    births = max(0, int(round(max(delta, 0) + rng.random())))
    deaths_natural = max(0, int(round(abs(min(delta, 0)) + rng.random() * 0.5)))

    if pop + births - deaths_natural < 1 and pop > 0:
        # Minimum viable population protection (extinction is hard)
        deaths_natural = min(deaths_natural, pop - 1) if pop > 1 else 0

    colony["population"] = max(0, pop + births - deaths_natural)
    colony["births_total"] += births
    colony["deaths_total"] += deaths_natural
    colony["peak_population"] = max(colony["peak_population"], colony["population"])

    # --- 6. Slow infrastructure/tech growth ---
    if colony["population"] > 10:
        colony["infrastructure"] = min(
            5.0, colony["infrastructure"] + 0.001 * colony["tech_level"]
        )
        colony["tech_level"] = min(
            3.0, colony["tech_level"] + 0.0005 * colony["infrastructure"]
        )

    # Morale recovery (slow drift toward 0.7)
    colony["morale"] += (0.7 - colony["morale"]) * 0.02

    # Decay active storms
    remaining_storms = []
    for storm in colony["active_storms"]:
        storm["remaining"] -= 1
        if storm["remaining"] > 0:
            remaining_storms.append(storm)
    colony["active_storms"] = remaining_storms

    return _snapshot(colony, sol)


def _snapshot(colony: dict, sol: int) -> dict:
    """Create a history snapshot for this sol."""
    return {
        "sol": sol,
        "population": colony["population"],
        "births_total": colony["births_total"],
        "deaths_total": colony["deaths_total"],
        "morale": round(colony["morale"], 3),
        "infrastructure": round(colony["infrastructure"], 3),
        "tech_level": round(colony["tech_level"], 3),
        "k_eff": round(effective_carrying_capacity(colony), 1),
        "resources": {k: round(v, 1) for k, v in colony["resources"].items()},
    }
