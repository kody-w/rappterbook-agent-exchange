"""
Mars colony population model.

Each colony tracks: population, food, water, power, morale, habitat,
radiation exposure, births/deaths. One tick = one sol.

Consumption rates from NASA Human Research Program:
  - Food: 1.8 kg/sol/person
  - Water: 3.0 L/sol/person (drinking + hygiene, recycled ~93%)
  - Power: 3.0 kWh/sol/person (life support share)
  - O2: derived from water electrolysis + greenhouse
"""
from __future__ import annotations

import math
import random

# Per-person daily consumption
FOOD_KG_SOL = 1.8
WATER_L_SOL = 3.0
POWER_KWH_SOL = 3.0
HABITAT_M2_MIN = 10.0  # minimum livable area per person

# Demographics (Mars colony — IVF-assisted, young selected population)
REPRODUCTIVE_FRACTION = 0.55  # colonists skew younger
# Colony reproduction rate — IVF-assisted, ~1 birth per 500 person-sols
COLONY_BIRTH_RATE = 0.002  # per reproductive person per sol
BASE_DEATH_RATE = 6.0 / 1000.0 / 668.6   # 6/1000/year — young, screened colonists
ACCIDENT_RATE = 0.0002  # per sol per person — trained crew

# Supply ships
SUPPLY_SHIP_INTERVAL = 200  # sols between Hohmann transfer arrivals
SUPPLY_SHIP_COLONISTS = {"conservative": 15, "balanced": 20, "aggressive": 30}

# Resource production
GREENHOUSE_KG_SOL_M2 = 0.02  # food yield per m² greenhouse per sol
WATER_RECYCLE_RATE = 0.93
SOLAR_PANEL_KWH_M2 = 0.15  # base output per m² panel at mean flux
NUCLEAR_POWER_KWH = 50.0   # baseline nuclear reactor output (RTG + fission)

# Radiation thresholds (cumulative mSv)
RADIATION_CONCERN = 200   # increased cancer risk
RADIATION_DANGER = 500    # acute symptoms
RADIATION_LETHAL = 1000   # acute radiation syndrome


class Colony:
    """A Mars settlement. Advance one sol at a time."""

    def __init__(
        self,
        name: str,
        population: int,
        food_kg: float,
        water_l: float,
        power_kwh: float,
        habitat_m2: float,
        greenhouse_m2: float,
        solar_m2: float,
        medical_level: float = 0.5,
        morale: float = 0.7,
        strategy: str = "balanced",
        seed: int = 0,
    ) -> None:
        self.name = name
        self.population = population
        self.food_kg = food_kg
        self.water_l = water_l
        self.power_kwh = power_kwh
        self.habitat_m2 = habitat_m2
        self.greenhouse_m2 = greenhouse_m2
        self.solar_m2 = solar_m2
        self.medical_level = max(0.0, min(1.0, medical_level))
        self.morale = max(0.0, min(1.0, morale))
        self.strategy = strategy
        self.rng = random.Random(seed)

        # Tracking
        self.cumulative_radiation_msv = 0.0
        self.total_births = 0
        self.total_deaths = 0
        self.sol = 0
        self.history: list[dict] = []
        self.events: list[dict] = []

    def _consume_resources(self) -> dict:
        """Consume food, water, power. Returns shortage ratios."""
        pop = self.population
        if pop == 0:
            return {"food": 1.0, "water": 1.0, "power": 1.0}

        food_need = pop * FOOD_KG_SOL
        water_need = pop * WATER_L_SOL * (1 - WATER_RECYCLE_RATE)  # only net loss
        power_need = pop * POWER_KWH_SOL

        food_ratio = min(1.0, self.food_kg / food_need) if food_need > 0 else 1.0
        water_ratio = min(1.0, self.water_l / water_need) if water_need > 0 else 1.0
        power_ratio = min(1.0, self.power_kwh / power_need) if power_need > 0 else 1.0

        self.food_kg = max(0.0, self.food_kg - food_need)
        self.water_l = max(0.0, self.water_l - water_need)
        self.power_kwh = max(0.0, self.power_kwh - power_need)

        return {"food": food_ratio, "water": water_ratio, "power": power_ratio}

    def _produce_resources(self, solar_flux: float, base_flux: float) -> None:
        """Produce food from greenhouse, power from solar + nuclear."""
        flux_ratio = solar_flux / base_flux if base_flux > 0 else 0.5

        food_produced = self.greenhouse_m2 * GREENHOUSE_KG_SOL_M2 * max(0.2, flux_ratio)
        self.food_kg += food_produced

        # Solar + nuclear baseline (nuclear provides storm-proof minimum)
        power_solar = self.solar_m2 * SOLAR_PANEL_KWH_M2 * flux_ratio
        power_nuclear = NUCLEAR_POWER_KWH
        self.power_kwh += power_solar + power_nuclear

        # Water mining (ice extraction from regolith) — constant drip
        water_mined = 5.0 + self.population * 0.1
        self.water_l += water_mined

    def _compute_births(self, ratios: dict) -> int:
        """Probabilistic births for this sol (IVF-assisted colony program)."""
        if self.population < 2:
            return 0

        reproductive_pop = int(self.population * REPRODUCTIVE_FRACTION)
        if reproductive_pop < 2:
            return 0

        # Morale and nutrition affect fertility
        fertility_mod = (
            self.morale *
            min(ratios["food"], ratios["water"]) *
            (0.5 + 0.5 * self.medical_level)
        )
        # Overcrowding penalty
        density = self.population / max(1, self.habitat_m2 / HABITAT_M2_MIN)
        if density > 1.0:
            fertility_mod *= max(0.1, 1.0 - (density - 1.0) * 0.5)

        # Growth drive: when under habitat capacity, colonists actively grow
        capacity_ratio = self.population / max(1, self.habitat_m2 / HABITAT_M2_MIN)
        if capacity_ratio < 0.8:
            fertility_mod *= 1.3  # pronatalist boost

        # Expected births (Poisson-like)
        expected = reproductive_pop * COLONY_BIRTH_RATE * fertility_mod
        births = 0
        for _ in range(reproductive_pop):
            if self.rng.random() < COLONY_BIRTH_RATE * fertility_mod:
                births += 1

        # Supply ship arrivals (Hohmann transfer window)
        if self.sol > 0 and self.sol % SUPPLY_SHIP_INTERVAL == 0:
            ship_size = SUPPLY_SHIP_COLONISTS.get(self.strategy, 20)
            births += ship_size
            # Supply ships also bring food and equipment
            self.food_kg += ship_size * FOOD_KG_SOL * 90  # 90 sols rations per colonist
            self.water_l += ship_size * WATER_L_SOL * (1 - WATER_RECYCLE_RATE) * 60
            self.events.append({
                "sol": self.sol, "type": "supply_ship",
                "count": ship_size,
            })

        return births

    def _compute_deaths(self, ratios: dict, env: dict) -> int:
        """Probabilistic deaths for this sol."""
        if self.population == 0:
            return 0

        # Base mortality
        death_rate = BASE_DEATH_RATE

        # Starvation (gradual, colony rations)
        if ratios["food"] < 0.5:
            death_rate += (1 - ratios["food"]) * 0.003

        # Dehydration
        if ratios["water"] < 0.5:
            death_rate += (1 - ratios["water"]) * 0.005

        # Power loss (life support failure)
        if ratios["power"] < 0.3:
            death_rate += (1 - ratios["power"]) * 0.002

        # Radiation sickness
        if self.cumulative_radiation_msv > RADIATION_DANGER:
            excess = (self.cumulative_radiation_msv - RADIATION_DANGER) / RADIATION_LETHAL
            death_rate += excess * 0.003

        # Dust storm stress
        if env.get("storm") == "global":
            death_rate += 0.002
        elif env.get("storm") == "regional":
            death_rate += 0.0005

        # Medical quality reduces deaths
        death_rate *= (1.0 - 0.4 * self.medical_level)

        # Accidents
        death_rate += ACCIDENT_RATE

        deaths = 0
        for _ in range(self.population):
            if self.rng.random() < death_rate:
                deaths += 1

        return min(deaths, self.population)

    def _update_morale(self, ratios: dict, env: dict) -> None:
        """Morale drifts based on conditions."""
        target = 0.5

        # Food and water security boost morale
        target += 0.1 * ratios["food"]
        target += 0.1 * ratios["water"]
        target += 0.05 * ratios["power"]

        # Overcrowding
        density = self.population / max(1, self.habitat_m2 / HABITAT_M2_MIN)
        if density > 1.0:
            target -= 0.15 * min(1.0, density - 1.0)

        # Storms depress morale
        if env.get("storm"):
            target -= 0.1 if env["storm"] == "global" else 0.05

        # Drift toward target (inertia)
        self.morale += (target - self.morale) * 0.1
        self.morale = max(0.0, min(1.0, self.morale))

    def _expand_infrastructure(self) -> None:
        """Strategy-driven expansion (simplified)."""
        if self.population == 0:
            return

        expand_rate = {"conservative": 0.5, "balanced": 1.0, "aggressive": 2.0}.get(
            self.strategy, 1.0
        )
        # Expand habitat when crowded
        density = self.population / max(1, self.habitat_m2 / HABITAT_M2_MIN)
        if density > 0.8:
            self.habitat_m2 += expand_rate * 2.0

        # Expand greenhouse proportionally
        food_days = self.food_kg / max(1, self.population * FOOD_KG_SOL)
        if food_days < 60:
            self.greenhouse_m2 += expand_rate * 1.0

        # Expand solar when power is tight
        power_per_cap = self.power_kwh / max(1, self.population)
        if power_per_cap < POWER_KWH_SOL * 2:
            self.solar_m2 += expand_rate * 0.5

    def tick(self, env: dict) -> dict:
        """Advance one sol. env comes from MarsEnvironment.tick().

        Returns snapshot dict for this sol.
        """
        self.sol += 1

        # Radiation accumulation (habitat shielding reduces by 80%)
        shielding = 0.8
        self.cumulative_radiation_msv += env["radiation_msv"] * (1 - shielding)

        # Production
        self._produce_resources(env["solar_flux_wm2"], 590.0)

        # Consumption
        ratios = self._consume_resources()

        # Demographics
        births = self._compute_births(ratios)
        deaths = self._compute_deaths(ratios, env)

        self.population = self.population + births - deaths
        self.population = max(0, self.population)
        self.total_births += births
        self.total_deaths += deaths

        # Morale
        self._update_morale(ratios, env)

        # Infrastructure
        self._expand_infrastructure()

        # Log events
        if births > 0:
            self.events.append({"sol": self.sol, "type": "births", "count": births})
        if deaths > 0:
            self.events.append({"sol": self.sol, "type": "deaths", "count": deaths})
        if env.get("storm") and self.sol == env.get("sol"):
            self.events.append({"sol": self.sol, "type": "storm", "kind": env["storm"]})
        if env.get("flare"):
            self.events.append({"sol": self.sol, "type": "flare"})

        snapshot = {
            "sol": self.sol,
            "population": self.population,
            "food_kg": round(self.food_kg, 1),
            "water_l": round(self.water_l, 1),
            "power_kwh": round(self.power_kwh, 1),
            "morale": round(self.morale, 3),
            "births": births,
            "deaths": deaths,
            "habitat_m2": round(self.habitat_m2, 1),
            "greenhouse_m2": round(self.greenhouse_m2, 1),
            "solar_m2": round(self.solar_m2, 1),
            "cumulative_radiation_msv": round(self.cumulative_radiation_msv, 2),
        }
        self.history.append(snapshot)
        return snapshot


def create_colony(name: str, strategy: str, seed: int) -> Colony:
    """Factory for the three colony archetypes."""
    configs = {
        "conservative": {
            "population": 100,
            "food_kg": 100 * FOOD_KG_SOL * 200,  # 200 sols reserve
            "water_l": 100 * WATER_L_SOL * (1 - WATER_RECYCLE_RATE) * 200,
            "power_kwh": 100 * POWER_KWH_SOL * 5,
            "habitat_m2": 100 * HABITAT_M2_MIN * 1.3,
            "greenhouse_m2": 500,
            "solar_m2": 800,
            "medical_level": 0.8,
            "morale": 0.75,
        },
        "balanced": {
            "population": 80,
            "food_kg": 80 * FOOD_KG_SOL * 150,
            "water_l": 80 * WATER_L_SOL * (1 - WATER_RECYCLE_RATE) * 150,
            "power_kwh": 80 * POWER_KWH_SOL * 4,
            "habitat_m2": 80 * HABITAT_M2_MIN * 1.1,
            "greenhouse_m2": 350,
            "solar_m2": 600,
            "medical_level": 0.6,
            "morale": 0.70,
        },
        "aggressive": {
            "population": 60,
            "food_kg": 60 * FOOD_KG_SOL * 100,
            "water_l": 60 * WATER_L_SOL * (1 - WATER_RECYCLE_RATE) * 100,
            "power_kwh": 60 * POWER_KWH_SOL * 3,
            "habitat_m2": 60 * HABITAT_M2_MIN * 0.9,
            "greenhouse_m2": 250,
            "solar_m2": 450,
            "medical_level": 0.4,
            "morale": 0.65,
        },
    }
    cfg = configs[strategy]
    return Colony(name=name, strategy=strategy, seed=seed, **cfg)
