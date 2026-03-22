"""Mars colony population model.

Each colony has:
- location (lat, lon, altitude)
- infrastructure (habitat volume, greenhouse area, solar panels, ice miners)
- population (crew count, births, deaths)
- resources (water_kg, food_kg, power_kwh, regolith_kg)

Population follows modified logistic growth bounded by carrying capacity.
Carrying capacity = min(habitat_capacity, food_capacity, water_capacity, power_capacity).
"""
from __future__ import annotations

import math
import random

from .environment import MarsEnvironment


# Infrastructure growth rates (per sol, per person working)
HABITAT_BUILD_RATE_M3 = 0.5  # m³ of pressurized volume per worker per sol
GREENHOUSE_BUILD_RATE_M2 = 0.3  # m² per worker per sol
SOLAR_PANEL_BUILD_RATE_M2 = 0.2  # m² per worker per sol

# Resource constants
WATER_PER_PERSON_KG = 2.7  # kg/day minimum
FOOD_PER_PERSON_KG = 1.8  # kg/day dry mass
POWER_PER_PERSON_KWH = 10.0  # kWh/day for life support
HABITAT_VOLUME_PER_PERSON_M3 = 25.0  # minimum pressurized volume per person

# Agricultural model
FOOD_YIELD_KG_PER_M2_SOL = 0.04  # optimized hydroponic yield (NASA VEGGIE+)
WATER_FOR_CROPS_KG_PER_M2 = 2.0  # daily irrigation per m² greenhouse

# Ice mining
ICE_MINING_RATE_KG_PER_MINER = 50.0  # kg water ice per sol per mining unit

# Solar power
SOLAR_PANEL_EFFICIENCY = 0.25  # 25% efficient panels
SOLAR_HOURS_PER_SOL = 12.4  # average daylight hours

# Demographics
BASE_BIRTH_RATE = 0.0003  # per person per sol (~1 birth per 9 Earth years)
BASE_DEATH_RATE = 0.00005  # per person per sol (~1 death per 55 Earth years)
IMMIGRATION_WAVE_SIZE = 6  # crew per supply ship
IMMIGRATION_INTERVAL_SOLS = 260  # ~every Earth-Mars synodic half-period


class ColonyConfig:
    """Static configuration for a colony site."""

    __slots__ = ("name", "latitude", "longitude", "altitude_km",
                 "description", "initial_crew", "ice_accessibility",
                 "dust_exposure", "terrain_difficulty")

    def __init__(self, name: str, latitude: float, longitude: float,
                 altitude_km: float, description: str, initial_crew: int,
                 ice_accessibility: float, dust_exposure: float,
                 terrain_difficulty: float) -> None:
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude_km = altitude_km
        self.description = description
        self.initial_crew = initial_crew
        self.ice_accessibility = ice_accessibility  # 0–1, affects mining rate
        self.dust_exposure = dust_exposure  # 0–1, affects panel degradation
        self.terrain_difficulty = terrain_difficulty  # 0–1, affects build rates


# The three colony sites
COLONY_CONFIGS = [
    ColonyConfig(
        name="Ares Prime",
        latitude=-4.5,  # Near equator in Valles Marineris
        longitude=-62.0,
        altitude_km=-2.0,  # Canyon floor, below datum
        description="Equatorial canyon base. Moderate solar, sheltered from wind, "
                    "deep ice deposits in canyon walls.",
        initial_crew=8,
        ice_accessibility=0.7,
        dust_exposure=0.3,  # Canyon shelters from dust
        terrain_difficulty=0.5,
    ),
    ColonyConfig(
        name="Boreas Station",
        latitude=46.7,  # Arcadia Planitia
        longitude=-170.0,
        altitude_km=-1.5,
        description="Northern lowlands outpost. Abundant shallow subsurface ice, "
                    "flat terrain for expansion, colder temperatures.",
        initial_crew=6,
        ice_accessibility=0.9,  # Best ice access
        dust_exposure=0.5,
        terrain_difficulty=0.2,  # Flat, easy to build
    ),
    ColonyConfig(
        name="Hellas Deep",
        latitude=-42.4,  # Hellas Basin center
        longitude=70.5,
        altitude_km=-7.0,  # Deepest point on Mars
        description="Deep crater settlement. Highest atmospheric pressure on Mars "
                    "(~1100 Pa), warmest temperatures, but severe dust storms.",
        initial_crew=10,
        ice_accessibility=0.4,
        dust_exposure=0.9,  # Worst dust exposure
        terrain_difficulty=0.7,
    ),
]


class Colony:
    """A Mars colony with population, infrastructure, and resources."""

    def __init__(self, config: ColonyConfig) -> None:
        self.config = config
        self.population: int = config.initial_crew
        self.sol: int = 0

        # Infrastructure
        self.habitat_volume_m3: float = config.initial_crew * 30.0
        self.greenhouse_area_m2: float = config.initial_crew * 50.0
        self.solar_panel_area_m2: float = config.initial_crew * 30.0
        self.ice_miners: int = 2
        self.has_shielding: bool = True  # Start with basic shielding

        # Resources (start with supply ship reserves)
        self.water_kg: float = config.initial_crew * WATER_PER_PERSON_KG * 90
        self.food_kg: float = config.initial_crew * FOOD_PER_PERSON_KG * 90
        self.regolith_kg: float = 500.0
        self.stored_power_kwh: float = 200.0

        # Tracking
        self.total_births: int = 0
        self.total_deaths: int = 0
        self.total_immigrants: int = config.initial_crew
        self.morale: float = 0.8  # 0–1
        self.history: list[dict] = []
        self.events: list[dict] = []

    def carrying_capacity(self) -> int:
        """Current carrying capacity based on infrastructure.

        Bottleneck = min(habitat, food, water, power).
        """
        habitat_cap = self.habitat_volume_m3 / HABITAT_VOLUME_PER_PERSON_M3
        food_cap = (self.greenhouse_area_m2 * FOOD_YIELD_KG_PER_M2_SOL /
                    FOOD_PER_PERSON_KG) if FOOD_PER_PERSON_KG > 0 else 0
        water_production = (self.ice_miners * ICE_MINING_RATE_KG_PER_MINER *
                            self.config.ice_accessibility)
        water_cap = water_production / WATER_PER_PERSON_KG if WATER_PER_PERSON_KG > 0 else 0
        # Power capacity from solar generation (assume ~150 W/m² avg Mars irradiance)
        avg_irradiance = 150.0
        daily_gen_kwh = (self.solar_panel_area_m2 * SOLAR_PANEL_EFFICIENCY *
                         avg_irradiance / 1000.0 * SOLAR_HOURS_PER_SOL)
        power_cap = daily_gen_kwh / POWER_PER_PERSON_KWH if POWER_PER_PERSON_KWH > 0 else 0

        return max(1, int(min(habitat_cap, food_cap, water_cap, power_cap)))

    def tick(self, env: MarsEnvironment, rng: random.Random) -> dict:
        """Advance colony by one sol. Returns sol report dict."""
        self.sol = env.sol
        report: dict = {"sol": env.sol, "colony": self.config.name}

        # === POWER GENERATION ===
        raw_power = (self.solar_panel_area_m2 * SOLAR_PANEL_EFFICIENCY *
                     env.irradiance / 1000.0 * SOLAR_HOURS_PER_SOL)
        # Panel degradation from dust
        dust_degradation = 1.0 - (self.config.dust_exposure * env.tau * 0.05)
        dust_degradation = max(0.1, dust_degradation)
        power_generated = raw_power * dust_degradation
        power_demand = self.population * POWER_PER_PERSON_KWH
        self.stored_power_kwh = min(
            self.stored_power_kwh + power_generated - power_demand,
            self.population * 50.0 + 200  # battery cap
        )
        self.stored_power_kwh = max(0, self.stored_power_kwh)
        report["power_kwh"] = round(power_generated, 1)

        # === WATER MINING ===
        water_mined = (self.ice_miners * ICE_MINING_RATE_KG_PER_MINER *
                       self.config.ice_accessibility)
        if env.dust_storm:
            water_mined *= 0.3  # Storm disrupts outdoor ops
        water_demand = self.population * WATER_PER_PERSON_KG
        crop_water = self.greenhouse_area_m2 * WATER_FOR_CROPS_KG_PER_M2
        self.water_kg += water_mined - water_demand - crop_water
        self.water_kg = max(0, self.water_kg)
        report["water_kg"] = round(self.water_kg, 1)

        # === FOOD PRODUCTION ===
        light_factor = min(1.0, env.irradiance / 200.0)
        water_factor = 1.0 if self.water_kg > crop_water else (
            self.water_kg / crop_water if crop_water > 0 else 0
        )
        food_produced = (self.greenhouse_area_m2 * FOOD_YIELD_KG_PER_M2_SOL *
                         light_factor * water_factor)
        food_demand = self.population * FOOD_PER_PERSON_KG
        self.food_kg += food_produced - food_demand
        self.food_kg = max(0, self.food_kg)
        report["food_kg"] = round(self.food_kg, 1)

        # === INFRASTRUCTURE EXPANSION ===
        # Workers allocated: 20% of population (rest do maintenance/science)
        if self.population > 0:
            builders = max(1, int(self.population * 0.2))
            build_factor = 1.0 - self.config.terrain_difficulty * 0.5
            if env.dust_storm:
                build_factor *= 0.1  # Almost no outdoor work in storms

            self.habitat_volume_m3 += (builders * HABITAT_BUILD_RATE_M3 *
                                        build_factor * 0.3)
            self.greenhouse_area_m2 += (builders * GREENHOUSE_BUILD_RATE_M2 *
                                         build_factor * 0.2)
            self.solar_panel_area_m2 += (builders * SOLAR_PANEL_BUILD_RATE_M2 *
                                          build_factor * 0.2)

            # New ice miner every ~100 sols of accumulated work
            if self.sol > 0 and self.sol % 100 == 0:
                self.ice_miners += 1

        # === POPULATION DYNAMICS ===
        cap = self.carrying_capacity()

        # Birth rate modulated by surplus and morale
        surplus_factor = max(0, min(2.0, (cap - self.population) / max(cap, 1)))
        effective_birth_rate = BASE_BIRTH_RATE * surplus_factor * self.morale
        # Minimum population for natural births
        births = 0
        if self.population >= 4:
            for _ in range(self.population):
                if rng.random() < effective_birth_rate:
                    births += 1

        # Death rate modulated by resource stress
        stress = 0.0
        if self.food_kg < food_demand * 7:  # Less than 7 days food reserve
            stress += 0.3
        if self.water_kg < water_demand * 7:
            stress += 0.3
        if self.stored_power_kwh < power_demand * 2:
            stress += 0.2
        if env.temperature < -80:
            stress += 0.1

        # Radiation deaths (rare but real)
        rad_death_prob = max(0, (env.radiation - 2.0) * 0.01)

        effective_death_rate = BASE_DEATH_RATE * (1 + stress * 10) + rad_death_prob

        deaths = 0
        for _ in range(self.population):
            if rng.random() < effective_death_rate:
                deaths += 1

        # Event-driven casualties
        event_deaths = 0
        if env.event == "meteorite_impact":
            event_deaths = rng.randint(0, max(1, self.population // 10))
            self.events.append({
                "sol": env.sol, "type": "meteorite",
                "message": f"Meteorite impact! {event_deaths} casualties.",
            })
        elif env.event == "equipment_failure":
            if rng.random() < 0.3:
                event_deaths = rng.randint(0, 2)
                self.events.append({
                    "sol": env.sol, "type": "equipment_failure",
                    "message": f"Equipment failure. {event_deaths} killed.",
                })
        elif env.event == "scientific_discovery":
            self.morale = min(1.0, self.morale + 0.05)
            self.events.append({
                "sol": env.sol, "type": "discovery",
                "message": "Scientific breakthrough! Morale boosted.",
            })

        deaths += event_deaths

        # Immigration waves
        immigrants = 0
        if (self.sol > 0 and self.sol % IMMIGRATION_INTERVAL_SOLS == 0 and
                self.population + IMMIGRATION_WAVE_SIZE <= cap * 1.5):
            immigrants = IMMIGRATION_WAVE_SIZE
            # Supply ship brings 90 days of extra supplies
            self.water_kg += immigrants * WATER_PER_PERSON_KG * 90
            self.food_kg += immigrants * FOOD_PER_PERSON_KG * 90
            self.ice_miners += 1
            self.solar_panel_area_m2 += 40.0
            self.events.append({
                "sol": env.sol, "type": "immigration",
                "message": f"Supply ship arrived! {immigrants} new colonists.",
            })

        # Apply population changes
        deaths = min(deaths, self.population)  # Can't kill more than exist
        self.population = max(0, self.population + births - deaths + immigrants)
        self.total_births += births
        self.total_deaths += deaths
        self.total_immigrants += immigrants

        # Morale dynamics
        if self.population > 0:
            resource_satisfaction = min(1.0, (self.food_kg / (food_demand * 30 + 1)))
            density = self.population / max(cap, 1)
            crowding_penalty = max(0, density - 0.8) * 0.5
            self.morale = 0.7 * self.morale + 0.3 * (resource_satisfaction - crowding_penalty)
            self.morale = max(0.1, min(1.0, self.morale))

        # Dust storm morale hit
        if env.dust_storm:
            self.morale = max(0.1, self.morale - 0.02)

        report.update({
            "population": self.population,
            "births": births,
            "deaths": deaths,
            "immigrants": immigrants,
            "carrying_capacity": cap,
            "morale": round(self.morale, 3),
            "habitat_m3": round(self.habitat_volume_m3, 1),
            "greenhouse_m2": round(self.greenhouse_area_m2, 1),
            "solar_m2": round(self.solar_panel_area_m2, 1),
            "ice_miners": self.ice_miners,
        })

        # Record history snapshot
        self.history.append({
            "sol": env.sol,
            "population": self.population,
            "carrying_capacity": cap,
            "food_kg": round(self.food_kg, 1),
            "water_kg": round(self.water_kg, 1),
            "power_kwh": round(power_generated, 1),
            "morale": round(self.morale, 3),
            "temperature_c": round(env.temperature, 1),
            "dust_storm": env.dust_storm,
            "radiation_msv": round(env.radiation, 3),
        })

        return report

    def to_dict(self) -> dict:
        """Serialize colony state to JSON-safe dict."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "location": {
                "latitude": self.config.latitude,
                "longitude": self.config.longitude,
                "altitude_km": self.config.altitude_km,
            },
            "population": self.population,
            "sol": self.sol,
            "infrastructure": {
                "habitat_volume_m3": round(self.habitat_volume_m3, 1),
                "greenhouse_area_m2": round(self.greenhouse_area_m2, 1),
                "solar_panel_area_m2": round(self.solar_panel_area_m2, 1),
                "ice_miners": self.ice_miners,
                "has_shielding": self.has_shielding,
            },
            "resources": {
                "water_kg": round(self.water_kg, 1),
                "food_kg": round(self.food_kg, 1),
                "regolith_kg": round(self.regolith_kg, 1),
                "stored_power_kwh": round(self.stored_power_kwh, 1),
            },
            "demographics": {
                "total_births": self.total_births,
                "total_deaths": self.total_deaths,
                "total_immigrants": self.total_immigrants,
                "morale": round(self.morale, 3),
            },
            "history": self.history,
            "events": self.events[-50:],  # Keep last 50 events
        }
