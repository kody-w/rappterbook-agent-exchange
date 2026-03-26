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

from src.tech_tree import ResearchEngine, TechUnlock

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
SUPPLY_SHIP_INTERVAL = 120  # sols between supply flights
SUPPLY_SHIP_COLONISTS = {"conservative": 20, "balanced": 25, "aggressive": 35}

# Terraforming output per colonist per sol (industrial greenhouse gas production)
# Rate tuned so 3 colonies (~400 total pop) reach ~10% in 365 sols (~0.1% progress/sol)
# Full terraforming takes ~5-10 Mars years of sustained effort.
TERRAFORM_BASE_RATE = 0.0000008
TERRAFORM_STRATEGY_MOD = {"conservative": 0.7, "balanced": 1.0, "aggressive": 1.5}

# Resource production
GREENHOUSE_KG_SOL_M2 = 0.08  # food yield per m² greenhouse per sol (vertical Mars farms)
WATER_RECYCLE_RATE = 0.93
SOLAR_PANEL_KWH_M2 = 0.22  # base output per m² panel (high-eff GaAs)
NUCLEAR_POWER_KWH = 100.0  # baseline fission reactor output (Kilopower)

# Radiation thresholds (cumulative mSv)
RADIATION_CONCERN = 200   # increased cancer risk
RADIATION_DANGER = 500    # acute symptoms
RADIATION_LETHAL = 1000   # acute radiation syndrome

# Epidemic parameters
EPIDEMIC_CHANCE_PER_SOL = 0.003  # ~1 epidemic per Mars year per colony
EPIDEMIC_MIN_POP = 20  # epidemics need enough hosts
EPIDEMIC_STRAINS = [
    {"name": "Mars Flu",       "severity": 0.3, "duration": (10, 25), "mortality": 0.002},
    {"name": "Regolith Lung",  "severity": 0.6, "duration": (15, 40), "mortality": 0.005},
    {"name": "Rad Fever",      "severity": 0.8, "duration": (20, 50), "mortality": 0.008},
]


class Epidemic:
    """Active disease outbreak in a colony."""
    __slots__ = ("strain", "severity", "remaining_sols", "total_duration",
                 "infected_count", "quarantined")

    def __init__(self, strain: dict, duration: int, population: int) -> None:
        self.strain = strain["name"]
        self.severity = strain["severity"]
        self.remaining_sols = duration
        self.total_duration = duration
        self.infected_count = max(1, int(population * 0.05))
        self.quarantined = False

    def infection_rate(self) -> float:
        """Current infection pressure (SIR-like curve)."""
        progress = 1.0 - (self.remaining_sols / self.total_duration)
        if progress < 0.3:
            return self.severity * progress / 0.3
        return self.severity * max(0.0, 1.0 - (progress - 0.3) / 0.7)

    def extra_mortality(self) -> float:
        """Additional death rate from the epidemic."""
        rate = self.severity * 0.005 * self.infection_rate()
        if self.quarantined:
            rate *= 0.4
        return rate

    def tick(self) -> bool:
        """Advance one sol. Returns True if epidemic still active."""
        self.remaining_sols -= 1
        return self.remaining_sols > 0


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
        self.water_mining_bonus = 0.0
        self.medical_breakthroughs = 0
        self.initial_population = population
        self.total_immigrants = 0
        self.total_emigrants = 0
        self.epidemic: Epidemic | None = None
        self.genetic_diversity = min(1.0, population / 200.0)
        self.cumulative_death_causes: dict[str, int] = {
            "baseline": 0, "starvation": 0, "dehydration": 0,
            "power_failure": 0, "radiation": 0, "storm": 0,
            "epidemic": 0, "accident": 0,
        }
        self.research_engine: ResearchEngine | None = None
        self.terraforming_output = 0.0  # cumulative contribution

    def habitability_index(self) -> float:
        """Composite 0-1 score of colony viability.

        Combines food security, morale, genetic diversity, medical capability,
        and radiation exposure into a single metric. Useful for dashboards,
        migration ranking, and narrative generation.

        The index is the geometric mean of five sub-scores so that a single
        catastrophic dimension drags the whole score toward zero.
        """
        if self.population == 0:
            return 0.0
        daily_food = max(1, self.population * FOOD_KG_SOL)
        food_sols = self.food_kg / daily_food
        food_score = min(1.0, food_sols / 120.0)
        rad_score = max(0.0, 1.0 - self.cumulative_radiation_msv / RADIATION_LETHAL)
        diversity_score = self.genetic_diversity
        morale_score = self.morale
        med_score = min(1.0, self.medical_level + self.medical_breakthroughs * 0.1)
        product = food_score * rad_score * diversity_score * morale_score * med_score
        return round(product ** 0.2, 4)

    def carrying_capacity(self) -> float:
        """Compute carrying capacity K from bottleneck resources.

        K = min(habitat_K, food_K, water_K, power_K).
        The binding constraint determines the population ceiling.
        Tech unlocks (aquaponics, construction bots) add bonus K.
        """
        habitat_k = self.habitat_m2 / HABITAT_M2_MIN
        food_daily = self.greenhouse_m2 * GREENHOUSE_KG_SOL_M2
        food_k = food_daily / FOOD_KG_SOL if FOOD_KG_SOL > 0 else 999
        water_daily = 5.0 + self.population * 0.1 + self.water_mining_bonus
        net_water_per_person = WATER_L_SOL * (1 - WATER_RECYCLE_RATE)
        water_k = water_daily / net_water_per_person if net_water_per_person > 0 else 999
        power_daily = self.solar_m2 * SOLAR_PANEL_KWH_M2 * 0.7 + NUCLEAR_POWER_KWH
        power_k = power_daily / POWER_KWH_SOL if POWER_KWH_SOL > 0 else 999
        base_k = max(2.0, min(habitat_k, food_k, water_k, power_k))
        tech_k_bonus = self._tech_k_bonus()
        return base_k + tech_k_bonus

    def _tech_k_bonus(self) -> float:
        """Bonus carrying capacity from unlocked techs."""
        if self.research_engine is None:
            return 0.0
        bonus = 0.0
        if self.research_engine.has_tech("Aquaponics Integration"):
            bonus += 10.0
        if self.research_engine.has_tech("Autonomous Construction Bots"):
            bonus += 20.0
        return bonus

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

    def _produce_resources(self, solar_flux: float, base_flux: float,
                          terraforming_progress: float = 0.0) -> None:
        """Produce food from greenhouse, power from solar + nuclear.

        Terraforming boosts greenhouse yield (warmer, thicker air).
        Tech modifiers: solar, greenhouse, nuclear, aquaponics, water.
        """
        flux_ratio = solar_flux / base_flux if base_flux > 0 else 0.5

        # Tech multipliers
        solar_mult = 1.0
        greenhouse_mult = 1.0
        nuclear_extra = 0.0
        food_extra_mult = 0.0
        water_mult = 1.0
        if self.research_engine is not None:
            solar_mult += self.research_engine.get_modifier("solar_boost")
            greenhouse_mult += self.research_engine.get_modifier("greenhouse_boost")
            nuclear_extra = self.research_engine.get_modifier("nuclear_boost")
            food_extra_mult = self.research_engine.get_modifier("aquaponics")
            water_mult += self.research_engine.get_modifier("water_efficiency")

        terraform_boost = 1.0 + terraforming_progress * 0.3
        food_produced = (self.greenhouse_m2 * GREENHOUSE_KG_SOL_M2
                         * max(0.2, flux_ratio) * terraform_boost
                         * greenhouse_mult * (1.0 + food_extra_mult))
        self.food_kg += food_produced

        # Solar + nuclear baseline (nuclear provides storm-proof minimum)
        power_solar = self.solar_m2 * SOLAR_PANEL_KWH_M2 * flux_ratio * solar_mult
        power_nuclear = NUCLEAR_POWER_KWH + nuclear_extra
        self.power_kwh += power_solar + power_nuclear

        # Water mining (ice extraction from regolith) — boosted by discoveries + tech
        water_mined = (5.0 + self.population * 0.1 + self.water_mining_bonus) * water_mult
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

        # Logistic damping — growth rate decreases as pop approaches K
        k = self.carrying_capacity()
        if k > 0:
            logistic_factor = max(0.0, 1.0 - self.population / k)
            fertility_mod *= logistic_factor

        # Genetic diversity penalty — inbreeding depression
        if self.genetic_diversity < 0.5:
            diversity_factor = 0.3 + 0.7 * (self.genetic_diversity / 0.5) ** 2
            fertility_mod *= diversity_factor

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

    def _compute_deaths(self, ratios: dict, env: dict) -> tuple[int, dict[str, int]]:
        """Probabilistic deaths with cause attribution.

        Each colonist is tested against causes in priority order.
        The first lethal cause wins. Returns (total_deaths, causes_dict).
        """
        causes: dict[str, int] = {
            "baseline": 0, "starvation": 0, "dehydration": 0,
            "power_failure": 0, "radiation": 0, "storm": 0,
            "epidemic": 0, "accident": 0,
        }
        if self.population == 0:
            return 0, causes

        # Build per-cause rates
        effective_medical = min(1.0, self.medical_level + self.medical_breakthroughs * 0.05)
        med_factor = 1.0 - 0.4 * effective_medical
        # AI Diagnostics tech reduces all death rates further
        if self.research_engine is not None:
            tech_mort_reduction = self.research_engine.get_modifier("mortality_reduction")
            med_factor *= (1.0 - tech_mort_reduction)

        rates: list[tuple[str, float]] = []
        rates.append(("baseline", BASE_DEATH_RATE * med_factor))
        if ratios["food"] < 0.5:
            rates.append(("starvation", (1 - ratios["food"]) * 0.003 * med_factor))
        if ratios["water"] < 0.5:
            rates.append(("dehydration", (1 - ratios["water"]) * 0.005 * med_factor))
        if ratios["power"] < 0.3:
            rates.append(("power_failure", (1 - ratios["power"]) * 0.002 * med_factor))
        if self.cumulative_radiation_msv > RADIATION_DANGER:
            excess = (self.cumulative_radiation_msv - RADIATION_DANGER) / RADIATION_LETHAL
            rates.append(("radiation", excess * 0.003 * med_factor))
        if env.get("storm") == "global":
            rates.append(("storm", 0.002 * med_factor))
        elif env.get("storm") == "regional":
            rates.append(("storm", 0.0005 * med_factor))
        if self.epidemic is not None:
            rates.append(("epidemic", self.epidemic.extra_mortality()))
        rates.append(("accident", ACCIDENT_RATE))

        deaths = 0
        for _ in range(self.population):
            for cause, rate in rates:
                if self.rng.random() < rate:
                    causes[cause] += 1
                    deaths += 1
                    break  # first lethal cause wins

        deaths = min(deaths, self.population)
        return deaths, causes

    def _storm_damage(self, env: dict) -> None:
        """Dust storms physically degrade solar panels and greenhouses.

        Global storms: 0.8%/sol solar, 0.5%/sol greenhouse.
        Regional storms: 0.2%/sol solar, 0.1%/sol greenhouse.
        Floor thresholds prevent total destruction.
        """
        storm = env.get("storm")
        if not storm:
            return
        if storm == "global":
            solar_loss = self.solar_m2 * 0.008
            greenhouse_loss = self.greenhouse_m2 * 0.005
        else:  # regional
            solar_loss = self.solar_m2 * 0.002
            greenhouse_loss = self.greenhouse_m2 * 0.001
        self.solar_m2 = max(50.0, self.solar_m2 - solar_loss)
        self.greenhouse_m2 = max(20.0, self.greenhouse_m2 - greenhouse_loss)
        if solar_loss > 0.5 or greenhouse_loss > 0.5:
            self.events.append({
                "sol": self.sol, "type": "storm_damage",
                "solar_loss_m2": round(solar_loss, 1),
                "greenhouse_loss_m2": round(greenhouse_loss, 1),
            })

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
        """Strategy-driven expansion. Construction bots tech doubles speed."""
        if self.population == 0:
            return

        expand_rate = {"conservative": 1.5, "balanced": 3.0, "aggressive": 5.0}.get(
            self.strategy, 1.0
        )
        # Construction bots multiply expansion speed
        if self.research_engine is not None:
            bot_mult = self.research_engine.get_modifier("construction_bots")
            if bot_mult > 0:
                expand_rate *= bot_mult
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

    def _roll_discoveries(self) -> None:
        """Rare permanent improvements — ice veins, medical breakthroughs.

        These are the surprises that change colony trajectories.
        ~1% per sol each, meaning ~3-4 discoveries per year per colony.
        """
        if self.population < 5:
            return

        # Ice vein discovery — permanent water mining boost
        if self.rng.random() < 0.008:
            bonus = self.rng.uniform(2.0, 8.0)
            self.water_mining_bonus += bonus
            self.morale = min(1.0, self.morale + 0.05)
            self.events.append({
                "sol": self.sol, "type": "discovery",
                "kind": "ice_vein", "bonus": round(bonus, 1),
            })

        # Medical breakthrough — permanent mortality reduction
        if self.rng.random() < 0.005 and self.medical_breakthroughs < 4:
            self.medical_breakthroughs += 1
            self.morale = min(1.0, self.morale + 0.08)
            self.events.append({
                "sol": self.sol, "type": "discovery",
                "kind": "medical",
            })

        # Crop strain adaptation — greenhouse efficiency boost
        if self.rng.random() < 0.006:
            boost = self.rng.uniform(5.0, 15.0)
            self.greenhouse_m2 += boost  # equivalent to adding greenhouse area
            self.events.append({
                "sol": self.sol, "type": "discovery",
                "kind": "crop_strain", "boost_m2": round(boost, 1),
            })

    def _tick_research(self) -> object:
        """Advance one sol of tech research. Returns unlock if one occurred."""
        if self.research_engine is None:
            return None
        unlock = self.research_engine.tick(self.population, self.morale, self.sol)
        if unlock is not None:
            # Aquaponics gives a morale boost on unlock
            if unlock.effect == "aquaponics":
                self.morale = min(1.0, self.morale + 0.05)
            self.events.append({
                "sol": self.sol, "type": "tech_unlock",
                "name": unlock.name, "branch": unlock.branch,
            })
        return unlock

    def _compute_terraforming(self) -> float:
        """Industrial greenhouse gas output for terraforming.

        Returns delta contribution for this sol (not cumulative).
        Scales with population, strategy, and industrial surplus.
        """
        if self.population == 0:
            return 0.0
        strategy_mod = TERRAFORM_STRATEGY_MOD.get(self.strategy, 1.0)
        base = self.population * TERRAFORM_BASE_RATE * strategy_mod
        # Power surplus bonus — spare energy goes to gas factories
        power_per_cap = self.power_kwh / max(1, self.population)
        if power_per_cap > POWER_KWH_SOL * 2:
            base *= 1.3
        # Greenhouse gas byproduct from farming
        base += self.greenhouse_m2 * 0.0000001
        self.terraforming_output += base
        return base

    def _tick_epidemic(self) -> None:
        """Advance epidemic state. May start, progress, or end outbreaks."""
        if self.epidemic is not None:
            alive = self.epidemic.tick()
            if not alive:
                self.events.append({
                    "sol": self.sol, "type": "epidemic_end",
                    "strain": self.epidemic.strain,
                })
                self.morale = min(1.0, self.morale + 0.05)
                self.epidemic = None
            elif not self.epidemic.quarantined and self.medical_level > 0.6:
                self.epidemic.quarantined = True
                self.events.append({
                    "sol": self.sol, "type": "quarantine",
                    "strain": self.epidemic.strain,
                })
        if (self.epidemic is None and
                self.population >= EPIDEMIC_MIN_POP and
                self.rng.random() < EPIDEMIC_CHANCE_PER_SOL):
            strain = self.rng.choice(EPIDEMIC_STRAINS)
            dur = self.rng.randint(strain["duration"][0], strain["duration"][1])
            self.epidemic = Epidemic(strain, dur, self.population)
            self.morale = max(0.0, self.morale - 0.1 * strain["severity"])
            self.events.append({
                "sol": self.sol, "type": "epidemic_start",
                "strain": strain["name"], "severity": strain["severity"],
            })

    def _drift_genetic_diversity(self) -> None:
        """Genetic drift — small populations lose diversity (Wright-Fisher)."""
        if self.population == 0:
            return
        ne = max(1, int(self.population * REPRODUCTIVE_FRACTION * 0.8))
        if self.sol % 30 == 0 and self.sol > 0:
            loss_rate = 1.0 / (2.0 * ne)
            self.genetic_diversity *= (1.0 - loss_rate)
            self.genetic_diversity = max(0.05, self.genetic_diversity)

    def receive_immigrants(self, count: int) -> None:
        """Boost genetic diversity when immigrants arrive."""
        if count <= 0 or self.population == 0:
            return
        boost = count / (self.population + count) * 0.3
        self.genetic_diversity = min(1.0, self.genetic_diversity + boost)

    def tick(self, env: dict) -> dict:
        """Advance one sol. env comes from MarsEnvironment.tick().

        Returns snapshot dict for this sol.
        """
        self.sol += 1

        # Radiation accumulation (habitat shielding reduces by 80%)
        shielding = 0.8
        if self.research_engine is not None:
            shielding += self.research_engine.get_modifier("rad_shielding")
            shielding = min(0.95, shielding)
        self.cumulative_radiation_msv += env["radiation_msv"] * (1 - shielding)

        # Production (terraforming boosts greenhouse yield)
        tf_progress = env.get("terraforming_progress", 0.0)
        self._produce_resources(env["solar_flux_wm2"], 590.0, tf_progress)

        # Consumption
        ratios = self._consume_resources()

        # Storm infrastructure damage
        self._storm_damage(env)

        # Demographics
        births = self._compute_births(ratios)
        deaths, death_causes = self._compute_deaths(ratios, env)

        self.population = self.population + births - deaths
        self.population = max(0, self.population)
        self.total_births += births
        self.total_deaths += deaths

        # Accumulate death cause tracking
        if not hasattr(self, "cumulative_death_causes"):
            self.cumulative_death_causes: dict[str, int] = {}
        for cause, count in death_causes.items():
            self.cumulative_death_causes[cause] = (
                self.cumulative_death_causes.get(cause, 0) + count
            )

        # Morale
        self._update_morale(ratios, env)

        # Infrastructure
        self._expand_infrastructure()

        # Rare discoveries
        self._roll_discoveries()

        # Technology research
        tech_unlock = self._tick_research()

        # Terraforming contribution
        terraform_delta = self._compute_terraforming()

        # Epidemics
        self._tick_epidemic()

        # Genetic diversity drift
        self._drift_genetic_diversity()

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
            "death_causes": death_causes,
            "habitat_m2": round(self.habitat_m2, 1),
            "greenhouse_m2": round(self.greenhouse_m2, 1),
            "solar_m2": round(self.solar_m2, 1),
            "cumulative_radiation_msv": round(self.cumulative_radiation_msv, 2),
            "carrying_capacity": round(self.carrying_capacity(), 1),
            "genetic_diversity": round(self.genetic_diversity, 4),
            "habitability_index": self.habitability_index(),
            "net_migration": 0,  # updated by Simulation after migration phase
            "terraforming_contribution": round(terraform_delta, 8),
            "tech": self.research_engine.snapshot() if self.research_engine else None,
        }
        self.history.append(snapshot)
        return snapshot


def create_colony(name: str, strategy: str, seed: int) -> Colony:
    """Factory for the three colony archetypes."""
    configs = {
        "conservative": {
            "population": 120,
            "food_kg": 120 * FOOD_KG_SOL * 200,  # 200 sols reserve
            "water_l": 120 * WATER_L_SOL * (1 - WATER_RECYCLE_RATE) * 200,
            "power_kwh": 120 * POWER_KWH_SOL * 5,
            "habitat_m2": 120 * HABITAT_M2_MIN * 1.5,
            "greenhouse_m2": 2500,
            "solar_m2": 2000,
            "medical_level": 0.8,
            "morale": 0.80,
        },
        "balanced": {
            "population": 80,
            "food_kg": 80 * FOOD_KG_SOL * 150,
            "water_l": 80 * WATER_L_SOL * (1 - WATER_RECYCLE_RATE) * 150,
            "power_kwh": 80 * POWER_KWH_SOL * 4,
            "habitat_m2": 80 * HABITAT_M2_MIN * 1.2,
            "greenhouse_m2": 1500,
            "solar_m2": 1500,
            "medical_level": 0.6,
            "morale": 0.70,
        },
        "aggressive": {
            "population": 60,
            "food_kg": 60 * FOOD_KG_SOL * 100,
            "water_l": 60 * WATER_L_SOL * (1 - WATER_RECYCLE_RATE) * 100,
            "power_kwh": 60 * POWER_KWH_SOL * 3,
            "habitat_m2": 60 * HABITAT_M2_MIN * 0.9,
            "greenhouse_m2": 800,
            "solar_m2": 1200,
            "medical_level": 0.4,
            "morale": 0.65,
        },
    }
    cfg = configs[strategy]
    return Colony(name=name, strategy=strategy, seed=seed, **cfg)
