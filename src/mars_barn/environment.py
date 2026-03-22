"""Mars environment model — seasons, dust storms, radiation, resource production.

Mars orbital mechanics drive everything: solar longitude (Ls) determines
season, dust storm probability, solar flux, and temperature. These feed
directly into colony resource production rates.

Reference: Allison & McEwen 2000 for Ls calculation.
"""
from __future__ import annotations

import math
import random


# Mars orbital constants
MARS_YEAR_SOLS = 668.6  # sols per Mars year
PERIHELION_LS = 251.0    # Ls at perihelion (closest to sun)

# Dust storm season: Ls 180-330 (southern spring/summer)
DUST_SEASON_START = 180.0
DUST_SEASON_END = 330.0

# Base probabilities per sol
BASE_DUST_STORM_PROB = 0.005       # outside dust season
DUST_SEASON_STORM_PROB = 0.04      # during dust season
GLOBAL_DUST_STORM_PROB = 0.002     # planet-encircling event
EQUIPMENT_FAILURE_PROB = 0.008
SUPPLY_DROP_PROB = 0.005
EPIDEMIC_PROB = 0.003
BIRTH_BOOM_PROB = 0.01


def sol_to_ls(sol: int) -> float:
    """Convert sol number to solar longitude (Ls) in degrees.

    Simplified model: linear mapping with eccentricity correction.
    Ls 0 = northern spring equinox.
    """
    mean_ls = (sol / MARS_YEAR_SOLS) * 360.0
    # Eccentricity correction (Mars e=0.0934)
    ecc_correction = 10.691 * math.sin(math.radians(mean_ls - PERIHELION_LS))
    return (mean_ls + ecc_correction) % 360.0


def solar_flux(ls: float) -> float:
    """Solar flux factor (0.0-1.0) based on season and distance from sun.

    Peaks near perihelion (Ls ~251), minimum near aphelion (Ls ~71).
    """
    angle = math.radians(ls - PERIHELION_LS)
    # Inverse square law with eccentricity
    ecc = 0.0934
    distance_factor = 1.0 - ecc * math.cos(angle)
    flux = 1.0 / (distance_factor ** 2)
    # Normalize to 0-1 range
    min_flux = 1.0 / ((1 + ecc) ** 2)
    max_flux = 1.0 / ((1 - ecc) ** 2)
    return (flux - min_flux) / (max_flux - min_flux)


def surface_temperature(ls: float, latitude: float = 0.0) -> float:
    """Surface temperature in Celsius at given Ls and latitude.

    Ranges roughly -80°C to +20°C depending on season and location.
    """
    base_temp = -40.0
    seasonal = 30.0 * math.sin(math.radians(ls - 90))
    lat_effect = -20.0 * abs(latitude) / 90.0
    return base_temp + seasonal + lat_effect


def is_dust_season(ls: float) -> bool:
    """Check if current Ls is in dust storm season."""
    if DUST_SEASON_START <= DUST_SEASON_END:
        return DUST_SEASON_START <= ls <= DUST_SEASON_END
    return ls >= DUST_SEASON_START or ls <= DUST_SEASON_END


def generate_events(sol: int, colonies: list[dict], rng: random.Random) -> list[dict]:
    """Generate random Mars events for this sol.

    Returns list of event dicts: {type, target, magnitude, description}.
    Target is colony index or -1 for global events.
    """
    ls = sol_to_ls(sol)
    events = []
    dust_season = is_dust_season(ls)

    # Local dust storms
    storm_prob = DUST_SEASON_STORM_PROB if dust_season else BASE_DUST_STORM_PROB
    for i in range(len(colonies)):
        if rng.random() < storm_prob:
            magnitude = rng.uniform(0.3, 0.8)
            duration = rng.randint(2, 15)
            events.append({
                "type": "dust_storm",
                "target": i,
                "magnitude": round(magnitude, 3),
                "duration": duration,
                "description": f"Dust storm hits colony {i} (intensity {magnitude:.1%})",
            })

    # Global dust storm (rare, devastating)
    if dust_season and rng.random() < GLOBAL_DUST_STORM_PROB:
        magnitude = rng.uniform(0.7, 1.0)
        duration = rng.randint(30, 90)
        events.append({
            "type": "global_dust_storm",
            "target": -1,
            "magnitude": round(magnitude, 3),
            "duration": duration,
            "description": f"GLOBAL dust storm! (intensity {magnitude:.1%}, {duration} sols)",
        })

    # Equipment failure
    for i in range(len(colonies)):
        if rng.random() < EQUIPMENT_FAILURE_PROB:
            system = rng.choice(["life_support", "power", "water_recycler", "greenhouse"])
            severity = rng.uniform(0.1, 0.5)
            events.append({
                "type": "equipment_failure",
                "target": i,
                "system": system,
                "magnitude": round(severity, 3),
                "description": f"Colony {i}: {system} failure (severity {severity:.0%})",
            })

    # Supply drop from Earth (every ~100 sols on average)
    if rng.random() < SUPPLY_DROP_PROB:
        target = rng.randint(0, len(colonies) - 1)
        supplies = rng.uniform(5, 20)
        events.append({
            "type": "supply_drop",
            "target": target,
            "magnitude": round(supplies, 1),
            "description": f"Supply drop at colony {target} (+{supplies:.0f} capacity)",
        })

    # Epidemic
    for i in range(len(colonies)):
        pop = colonies[i].get("population", 0)
        # Higher density = higher epidemic risk
        density_factor = min(pop / max(colonies[i].get("carrying_capacity", 100), 1), 1.5)
        if rng.random() < EPIDEMIC_PROB * density_factor:
            mortality = rng.uniform(0.02, 0.10)
            events.append({
                "type": "epidemic",
                "target": i,
                "magnitude": round(mortality, 3),
                "description": f"Epidemic at colony {i} ({mortality:.1%} mortality)",
            })

    # Birth boom (good conditions)
    flux = solar_flux(ls)
    for i in range(len(colonies)):
        boom_prob = BIRTH_BOOM_PROB * flux
        if rng.random() < boom_prob:
            boost = rng.uniform(1.2, 1.8)
            events.append({
                "type": "birth_boom",
                "target": i,
                "magnitude": round(boost, 3),
                "description": f"Birth boom at colony {i} ({boost:.0%} birth rate)",
            })

    return events


def compute_resource_production(
    colony: dict, ls: float, active_storms: list[dict]
) -> dict[str, float]:
    """Compute per-sol resource production rates for a colony.

    Resources: o2 (kg/sol), h2o (L/sol), food (kg/sol), power (kWh/sol).
    Nuclear reactors provide baseline power; solar adds on top.
    MOXIE units produce O2 from CO2. Water is 95% recycled + ice mining.
    Greenhouses use artificial light when solar is low.
    """
    tech = colony.get("tech_level", 1.0)
    pop = max(colony.get("population", 0), 1)
    infra = colony.get("infrastructure", 1.0)

    # Dust reduces solar but not nuclear
    flux = solar_flux(ls)
    dust_reduction = 1.0
    for storm in active_storms:
        # Active storms on the colony already filtered by target
        dust_reduction *= (1.0 - storm.get("magnitude", 0) * 0.8)

    # Power: nuclear baseline (70%) + solar (30%)
    nuclear_power = 150.0 * infra * tech
    solar_power = 80.0 * infra * tech * flux * max(dust_reduction, 0.05)
    power = nuclear_power + solar_power

    # O2 from MOXIE electrolysis (scales with power and infrastructure)
    o2 = 60.0 * infra * tech * min(power / (150.0 * infra * tech), 1.2)

    # Water: 95% recycling + ice mining
    h2o_recycling = pop * 2.5 * 0.95  # recycle 95% of consumption
    h2o_mining = 15.0 * infra * tech
    h2o = h2o_recycling + h2o_mining

    # Food from greenhouses (artificial light + solar supplement)
    greenhouse_base = 50.0 * infra * tech  # artificial lighting baseline
    greenhouse_solar = 40.0 * infra * tech * flux * max(dust_reduction, 0.1)
    food = greenhouse_base + greenhouse_solar

    return {
        "power": round(max(power, 0), 2),
        "o2": round(max(o2, 0), 2),
        "h2o": round(max(h2o, 0), 2),
        "food": round(max(food, 0), 2),
    }
