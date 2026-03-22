#!/usr/bin/env python3
"""Colony population model — demographics, resources, carrying capacity.

Each colony is a self-contained habitat with:
- Population with age structure
- Resources: O2, H2O, food, power (kWh)
- Infrastructure: habitable volume, shielding, greenhouses
- Morale: psychological wellbeing affecting birth/death rates
"""
from __future__ import annotations

import math
import math
import random
from dataclasses import dataclass, field


@dataclass
class Resources:
    """Colony resource stockpiles (per-capita days of supply)."""
    o2_days: float = 90.0
    h2o_days: float = 90.0
    food_days: float = 120.0
    power_kwh: float = 1000.0

    def min_supply_days(self) -> float:
        """Minimum supply across all life-critical resources."""
        return min(self.o2_days, self.h2o_days, self.food_days)

    def to_dict(self) -> dict:
        return {
            "o2_days": round(self.o2_days, 1),
            "h2o_days": round(self.h2o_days, 1),
            "food_days": round(self.food_days, 1),
            "power_kwh": round(self.power_kwh, 1),
        }


@dataclass
class ColonyConfig:
    """Static configuration for a colony type."""
    name: str
    strategy: str  # "greenhouse", "underground", "hybrid"
    initial_pop: int = 100
    habitat_volume_m3: float = 5000.0
    greenhouse_area_m2: float = 200.0
    solar_panel_m2: float = 500.0
    has_nuclear: bool = False
    has_geothermal: bool = False
    radiation_shielding: float = 0.5  # 0-1, fraction absorbed

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "initial_pop": self.initial_pop,
            "habitat_volume_m3": self.habitat_volume_m3,
            "greenhouse_area_m2": self.greenhouse_area_m2,
            "solar_panel_m2": self.solar_panel_m2,
            "has_nuclear": self.has_nuclear,
            "has_geothermal": self.has_geothermal,
            "radiation_shielding": self.radiation_shielding,
        }


# Colony configurations for the 3 colonies
OLYMPUS_CONFIG = ColonyConfig(
    name="Olympus Greenhouse",
    strategy="greenhouse",
    initial_pop=120,
    habitat_volume_m3=8000.0,
    greenhouse_area_m2=1600.0,  # large surface greenhouses
    solar_panel_m2=2000.0,
    has_nuclear=False,
    has_geothermal=False,
    radiation_shielding=0.3,
)

VALLES_CONFIG = ColonyConfig(
    name="Valles Caverns",
    strategy="underground",
    initial_pop=80,
    habitat_volume_m3=12000.0,  # vast cave network
    greenhouse_area_m2=600.0,   # grow-light racks in caverns
    solar_panel_m2=300.0,
    has_nuclear=False,
    has_geothermal=True,
    radiation_shielding=0.95,
)

HELLAS_CONFIG = ColonyConfig(
    name="Hellas Basin Hub",
    strategy="hybrid",
    initial_pop=100,
    habitat_volume_m3=10000.0,
    greenhouse_area_m2=1000.0,  # surface + underground growing
    solar_panel_m2=800.0,
    has_nuclear=True,
    has_geothermal=False,
    radiation_shielding=0.6,
)


def carrying_capacity(config: ColonyConfig, resources: Resources) -> int:
    """Compute carrying capacity from infrastructure and resources.

    K reflects the maximum sustainable population given infrastructure.
    Growth is possible when population < K.
    """
    # Space: ~40 m³ per person for long-term habitation
    space_k = int(config.habitat_volume_m3 / 40.0)

    # Food: estimate sustainable food production capacity
    # Greenhouse at ~0.25 kg/m²/sol, ~0.65 average solar efficiency
    food_per_sol = config.greenhouse_area_m2 * 0.25 * 0.65
    # Underground/nuclear colonies get grow-light bonus
    if config.strategy == "underground":
        food_per_sol += config.greenhouse_area_m2 * 0.15
    elif config.has_nuclear or config.has_geothermal:
        food_per_sol += config.greenhouse_area_m2 * 0.10
    food_k = int(food_per_sol / 1.8)  # 1.8 kg/person/sol

    # O2 production capacity
    o2_per_sol = config.greenhouse_area_m2 * 0.12 * 0.65
    if config.strategy == "underground":
        o2_per_sol += config.greenhouse_area_m2 * 0.08
    elif config.has_nuclear or config.has_geothermal:
        o2_per_sol += config.greenhouse_area_m2 * 0.05
    o2_k = int(o2_per_sol / 0.84)

    # K is the minimum bottleneck
    k = min(space_k, food_k, o2_k)

    # Resource pressure: if any supply < 30 days, effective K drops
    supply_min = resources.min_supply_days()
    if supply_min < 30:
        k = int(k * (supply_min / 30.0))

    return max(10, k)


def produce_resources(
    config: ColonyConfig,
    resources: Resources,
    population: int,
    solar_power_factor: float,
    temperature_c: float,
    rng: random.Random,
) -> Resources:
    """Produce resources for one sol. Returns updated resources.

    Resources are tracked as per-capita days of supply.
    Production rates are calibrated so colonies are marginally self-sustaining
    at initial population, with resource pressure emerging as population grows.

    Key rates (per sol, at full capacity):
      Greenhouse food:  ~0.25 kg/m²/sol (intensive hydroponics)
      O2 from plants:   ~0.05 kg/m²/sol (photosynthesis byproduct)
      O2 from MOXIE:    scales with power surplus
      H2O recycling:    95% + ice mining supplement
      Power (solar):    ~1 kWh/m²/sol peak, reduced by dust and season
    """
    pop = max(population, 1)

    # === Power (kWh) — computed as daily balance ===
    solar_power = config.solar_panel_m2 * 1.0 * solar_power_factor  # ~1 kWh/m²/sol
    nuclear_power = 300.0 if config.has_nuclear else 0.0
    geothermal_power = 200.0 if config.has_geothermal else 0.0
    power_produced = solar_power + nuclear_power + geothermal_power

    heating_factor = max(1.0, 1.0 + (-temperature_c - 30) / 200.0)
    power_consumed = pop * 5.0 * heating_factor  # 5 kWh base per person
    power_net = power_produced - power_consumed
    resources.power_kwh = max(0.0, min(10000.0, resources.power_kwh + power_net))

    # === Food (days of supply per capita) ===
    # Hydroponics: 0.25 kg/m²/sol, need ~1.8 kg/person/sol
    solar_eff = max(0.3, solar_power_factor)
    food_produced_kg = config.greenhouse_area_m2 * 0.25 * solar_eff
    # Grow-light supplement: any colony with surplus power can boost food
    if resources.power_kwh > 100:
        if config.strategy == "underground":
            food_produced_kg += config.greenhouse_area_m2 * 0.15
        elif config.has_nuclear or config.has_geothermal:
            food_produced_kg += config.greenhouse_area_m2 * 0.10
    food_consumed_kg = pop * 1.8
    food_net_days = (food_produced_kg - food_consumed_kg) / (pop * 1.8)
    resources.food_days = max(0.0, min(365.0, resources.food_days + food_net_days))

    # === O2 (days of supply per capita) ===
    # Greenhouse photosynthesis: 0.12 kg O2/m²/sol
    o2_from_plants = config.greenhouse_area_m2 * 0.12 * solar_eff
    if resources.power_kwh > 100:
        if config.strategy == "underground":
            o2_from_plants += config.greenhouse_area_m2 * 0.08
        elif config.has_nuclear or config.has_geothermal:
            o2_from_plants += config.greenhouse_area_m2 * 0.05
    # MOXIE electrolysis from CO2: scales with surplus power
    o2_from_moxie = max(0.0, power_net * 0.15) if power_net > 0 else 0.0
    o2_produced = o2_from_plants + o2_from_moxie
    o2_consumed = pop * 0.84  # 0.84 kg O2/person/sol
    o2_net_days = (o2_produced - o2_consumed) / (pop * 0.84)
    resources.o2_days = max(0.0, min(365.0, resources.o2_days + o2_net_days))

    # === H2O (days of supply per capita) ===
    # 95% recycling + ice mining: nearly self-sustaining
    h2o_recycled = pop * 3.0 * 0.95
    h2o_mined = 15.0 + (10.0 if config.strategy == "underground" else 0.0)
    h2o_consumed = pop * 3.0
    h2o_net_days = (h2o_recycled + h2o_mined - h2o_consumed) / (pop * 3.0)
    resources.h2o_days = max(0.0, min(365.0, resources.h2o_days + h2o_net_days))

    # === Random equipment malfunction ===
    if rng.random() < 0.003:
        resource_hit = rng.choice(["o2", "h2o", "food", "power"])
        loss = rng.uniform(0.05, 0.15)
        if resource_hit == "o2":
            resources.o2_days *= (1 - loss)
        elif resource_hit == "h2o":
            resources.h2o_days *= (1 - loss)
        elif resource_hit == "food":
            resources.food_days *= (1 - loss)
        else:
            resources.power_kwh *= (1 - loss)

    return resources


def _poisson(lam: float, rng: random.Random) -> int:
    """Sample from Poisson distribution using Knuth's algorithm.

    Efficient for small lambda values typical of per-sol birth/death rates.
    """
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p < L:
            return k - 1


def compute_morale(
    resources: Resources,
    population: int,
    config: ColonyConfig,
    radiation_msv: float,
    dust_storm: bool,
) -> float:
    """Compute colony morale [0-1].

    Affected by resource security, crowding, radiation, weather.
    """
    morale = 0.7  # baseline

    # Resource security bonus/penalty
    supply = resources.min_supply_days()
    if supply > 60:
        morale += 0.1
    elif supply < 14:
        morale -= 0.2
    elif supply < 30:
        morale -= 0.1

    # Crowding penalty
    density = population / max(config.habitat_volume_m3, 1) * 50  # people per 50m³
    if density > 1.0:
        morale -= 0.15 * (density - 1.0)

    # Radiation anxiety (surface colonies)
    if radiation_msv > 1.0:
        morale -= 0.1
    elif config.radiation_shielding > 0.8:
        morale += 0.05  # security from good shielding

    # Dust storm cabin fever
    if dust_storm:
        morale -= 0.15

    # Underground isolation penalty
    if config.strategy == "underground":
        morale -= 0.05

    return max(0.0, min(1.0, morale))


def population_delta(
    population: int,
    k: int,
    morale: float,
    resources: Resources,
    radiation_msv: float,
    rng: random.Random,
) -> tuple[int, int, list[str]]:
    """Compute births and deaths for one sol.

    Returns (births, deaths, events).
    Uses stochastic logistic growth with resource and morale modifiers.
    """
    events: list[str] = []

    if population <= 0:
        return 0, 0, ["Colony extinct"]

    # Birth rate: base ~0.003/person/sol
    # High rate reflects intentional colony growth program with fertility support.
    # Heavily damped by logistic factor near carrying capacity.
    base_birth_rate = 0.003
    food_factor = min(1.0, resources.food_days / 30.0)
    morale_factor = 0.5 + morale * 0.5
    radiation_fertility = max(0.5, 1.0 - radiation_msv * 0.05)
    birth_rate = base_birth_rate * food_factor * morale_factor * radiation_fertility

    # Logistic damping: soft curve (1 - (P/K)²) gives more growth room
    ratio = population / max(k, 1)
    logistic_factor = max(0.0, 1.0 - ratio * ratio)
    birth_rate *= logistic_factor

    # Stochastic births (Poisson-approximated via expected value + noise)
    expected_births = birth_rate * population
    births = 0
    # Use Poisson for small expected values
    if expected_births > 0:
        births = _poisson(expected_births, rng)

    # Death rate: base ~0.00008/person/sol ≈ 29/1000/year
    # Harsh Mars conditions. Modified by resource scarcity, radiation, crowding.
    base_death_rate = 0.00008
    supply_min = resources.min_supply_days()
    if supply_min < 7:
        base_death_rate *= 5.0  # critical shortage
        events.append("Critical resource shortage — elevated mortality")
    elif supply_min < 14:
        base_death_rate *= 2.0
        events.append("Resource shortage — increased mortality risk")

    # Radiation-induced mortality (chronic exposure)
    if radiation_msv > 2.0:
        base_death_rate += 0.00005 * radiation_msv

    # Overcrowding disease risk
    if population > k:
        overcrowd_factor = 1.0 + (population - k) / max(k, 1)
        base_death_rate *= overcrowd_factor

    deaths = 0
    expected_deaths = base_death_rate * population
    if expected_deaths > 0:
        deaths = _poisson(expected_deaths, rng)

    # Rare catastrophic events
    if rng.random() < 0.001:
        casualty_count = rng.randint(1, max(1, population // 20))
        deaths += casualty_count
        events.append(f"Habitat breach — {casualty_count} casualties")

    if births > 0:
        events.append(f"{births} born this sol")
    if deaths > 0 and not any("casualties" in e for e in events):
        events.append(f"{deaths} died this sol")

    return births, deaths, events


def compute_migration(
    colonies: list[dict],
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    """Compute migration between colonies based on resource pressure.

    Returns list of (from_idx, to_idx, count) tuples.
    Migration only triggers when resource differential > threshold.
    """
    moves: list[tuple[int, int, int]] = []

    for i, src in enumerate(colonies):
        src_supply = src["resources"].min_supply_days()
        src_pop = src["population"]

        if src_pop < 20:
            continue  # too small to lose people

        for j, dst in enumerate(colonies):
            if i == j:
                continue
            dst_supply = dst["resources"].min_supply_days()

            # Migrate if destination has >30 days more supply
            differential = dst_supply - src_supply
            if differential > 30 and src_supply < 30:
                # 1-3% of source population migrates
                rate = rng.uniform(0.01, 0.03)
                count = max(1, int(src_pop * rate))
                count = min(count, src_pop - 10)  # keep minimum 10
                if count > 0:
                    moves.append((i, j, count))

    return moves


@dataclass
class Colony:
    """A Mars colony with population and resources."""
    config: ColonyConfig
    population: int = 0
    resources: Resources = field(default_factory=Resources)
    morale: float = 0.7
    total_births: int = 0
    total_deaths: int = 0
    total_migrants_in: int = 0
    total_migrants_out: int = 0
    population_history: list[int] = field(default_factory=list)
    morale_history: list[float] = field(default_factory=list)
    resource_history: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.population == 0:
            self.population = self.config.initial_pop

    def to_dict(self) -> dict:
        """Serialize colony state."""
        return {
            "config": self.config.to_dict(),
            "population": self.population,
            "resources": self.resources.to_dict(),
            "morale": round(self.morale, 3),
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "total_migrants_in": self.total_migrants_in,
            "total_migrants_out": self.total_migrants_out,
            "population_history": self.population_history,
            "morale_history": [round(m, 3) for m in self.morale_history],
            "events": self.events[-100:],  # keep last 100 events
        }

    @classmethod
    def from_dict(cls, data: dict) -> Colony:
        """Deserialize colony state."""
        cfg_data = data["config"]
        config = ColonyConfig(**cfg_data)
        resources = Resources(**data["resources"])
        colony = cls(
            config=config,
            population=data["population"],
            resources=resources,
            morale=data.get("morale", 0.7),
            total_births=data.get("total_births", 0),
            total_deaths=data.get("total_deaths", 0),
            total_migrants_in=data.get("total_migrants_in", 0),
            total_migrants_out=data.get("total_migrants_out", 0),
            population_history=data.get("population_history", []),
            morale_history=data.get("morale_history", []),
            resource_history=data.get("resource_history", []),
            events=data.get("events", []),
        )
        return colony
