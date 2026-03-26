"""food_storage.py — Mars Colony Food Preservation & Caloric Inventory.

The barn.  A colony that grows food but cannot store it is one failed
harvest from starvation.  Mars provides a gift: its thin atmosphere
(~636 Pa) and cold temperatures (-60°C average) make freeze-drying
trivially easy — expose wet food to the Martian environment and
physics does the rest.  But storage isn't free: containers must be
sealed against perchlorates in regolith dust, nutrient degradation
is relentless, and caloric accounting errors kill as surely as
vacuum exposure.

Physics
-------
* **Freeze-drying (lyophilisation)**: water sublimes directly from
  ice to vapor below the triple point (611 Pa).  Mars surface
  pressure (~636 Pa) is borderline — a slight vacuum pump brings
  it below threshold.  Energy: ~2.8 MJ/kg water removed (latent
  heat of sublimation: 2.83 MJ/kg at 0°C).
* **Shelf life**: freeze-dried food lasts 25-30 years at Earth
  ambient.  On Mars, lower temperatures extend this.  Modelled as
  exponential decay: Q(t) = Q₀ · exp(-λt), where λ depends on
  storage temperature and packaging integrity.
* **Nutrient degradation**: vitamins degrade faster than calories.
  Vitamin C half-life ~6 months in freeze-dried storage; B12 ~2 yr.
  Tracked as a separate quality metric (0-100%).
* **Caloric needs**: 2500 kcal/person/sol (moderate activity),
  scaling to 3200 kcal/sol during EVA-heavy periods and dropping
  to 1800 kcal/sol under emergency rationing.
* **Supply ships**: Earth deliveries every ~26 months (synodic
  period).  Each ship carries ~200 sols of food per crew member.
* **Perchlorate contamination**: Mars regolith contains 0.5-1%
  perchlorates (toxic).  Storage containers must maintain seal
  integrity; breach means contamination and food loss.

Reference: NASA Space Food Systems Lab; ISS food packaging (~1.8 kg/
  person/day); Mars DRA 5.0 logistics; Curiosity perchlorate measurements
  (0.5% ClO₄⁻ in Gale Crater soil).

One tick = one sol.  Mass in kg, energy in kWh, calories in kcal.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# -- Physical constants -------------------------------------------------------

WATER_SUBLIMATION_ENERGY_MJ_KG = 2.83   # latent heat of sublimation at 0°C
MJ_TO_KWH = 1.0 / 3.6                    # 1 kWh = 3.6 MJ
MARS_SURFACE_PRESSURE_PA = 636.0          # average Mars surface pressure
WATER_TRIPLE_POINT_PA = 611.0             # water triple point pressure
MARS_AVG_TEMP_C = -60.0                   # average Mars surface temperature

# Freeze-drying parameters
FREEZE_DRY_WATER_FRACTION = 0.80          # typical food water content (80%)
FREEZE_DRY_EFFICIENCY = 0.92              # practical efficiency of sublimation
VACUUM_PUMP_POWER_KW = 0.5               # small pump to drop below triple point
FREEZE_DRY_HOURS_PER_KG = 8.0            # processing time per kg wet food

# Caloric parameters
KCAL_PER_PERSON_SOL_NORMAL = 2500.0       # moderate activity
KCAL_PER_PERSON_SOL_EVA = 3200.0          # heavy EVA work
KCAL_PER_PERSON_SOL_RATION = 1800.0       # emergency rationing minimum
KCAL_PER_KG_FREEZE_DRIED = 3800.0         # average across food types
KCAL_PER_KG_FRESH = 800.0                 # average fresh produce

# Shelf life (exponential decay rate per sol at Mars ambient -60°C)
# At -60°C, freeze-dried food degrades ~10x slower than Earth ambient
CALORIC_DECAY_RATE_PER_SOL = 0.000005     # ~0.0005%/sol → ~17% loss/year
NUTRIENT_DECAY_RATE_PER_SOL = 0.0003      # ~0.03%/sol → ~10% loss/year

# Vitamin-specific half-lives (sols)
VITAMIN_C_HALF_LIFE_SOLS = 180.0          # ~6 months
VITAMIN_B12_HALF_LIFE_SOLS = 730.0        # ~2 years
VITAMIN_A_HALF_LIFE_SOLS = 365.0          # ~1 year

# Storage capacity
STORAGE_CAPACITY_KG = 5000.0              # maximum dry food storage (kg)
CONTAINER_SEAL_FAILURE_RATE = 0.001       # probability per container per sol
PERCHLORATE_CONTAMINATION_LOSS = 0.05     # 5% of container lost on seal breach

# Supply ships
SUPPLY_SHIP_INTERVAL_SOLS = 780.0         # ~26 months synodic period
SUPPLY_SHIP_FOOD_KG_PER_PERSON = 400.0    # ~200 sols of food per person
SUPPLY_SHIP_KCAL_PER_KG = 4200.0          # concentrated ship rations

# Ration thresholds (days of food remaining per person)
RATION_NORMAL_DAYS = 120                  # comfortable buffer
RATION_CAUTION_DAYS = 60                  # reduce waste, watch consumption
RATION_WARNING_DAYS = 30                  # begin rationing
RATION_CRITICAL_DAYS = 14                 # emergency rations
RATION_STARVATION_DAYS = 3                # colony in mortal danger

# Processing limits
MAX_FREEZE_DRY_KG_SOL = 50.0             # max fresh food processed per sol
COLD_STORAGE_POWER_KW = 2.0              # refrigeration for fresh produce
COLD_STORAGE_CAPACITY_KG = 500.0         # fresh produce cold storage


# -- Data structures ----------------------------------------------------------

@dataclass
class FoodBatch:
    """A batch of stored food with provenance and quality tracking."""
    batch_id: int
    sol_stored: int
    mass_kg: float
    kcal_total: float
    nutrient_quality: float    # 0.0-1.0, degrades over time
    is_freeze_dried: bool
    source: str                # "greenhouse", "supply_ship", "algae"
    contaminated: bool = False


@dataclass
class StorageState:
    """Mars colony food storage system state."""
    sol: int = 0
    crew_count: int = 20

    # Inventory
    freeze_dried_kg: float = 0.0
    fresh_kg: float = 0.0
    total_kcal: float = 0.0
    batches: List[FoodBatch] = field(default_factory=list)
    next_batch_id: int = 1

    # Quality tracking
    avg_nutrient_quality: float = 1.0
    vitamin_c_level: float = 1.0      # 0-1 relative to fresh
    vitamin_b12_level: float = 1.0
    vitamin_a_level: float = 1.0

    # Consumption
    kcal_consumed_today: float = 0.0
    cumulative_kcal_consumed: float = 0.0
    sols_rationed: int = 0
    sols_starving: int = 0

    # Processing
    freeze_dry_queue_kg: float = 0.0
    freeze_dry_processed_kg: float = 0.0
    processing_power_kwh: float = 0.0

    # Supply chain
    sols_since_supply: int = 0
    supply_ships_received: int = 0

    # Losses
    spoilage_kg: float = 0.0
    contamination_kg: float = 0.0
    total_loss_kg: float = 0.0

    # Alerts
    ration_level: str = "normal"
    power_consumed_kwh: float = 0.0


@dataclass
class SolReport:
    """Per-sol food storage results."""
    sol: int
    ration_level: str
    days_of_food: float
    total_kcal: float
    freeze_dried_kg: float
    fresh_kg: float
    kcal_consumed: float
    kcal_per_person: float
    nutrient_quality: float
    vitamin_c: float
    vitamin_b12: float
    fresh_processed_kg: float
    spoilage_kg: float
    contamination_kg: float
    power_kwh: float
    supply_ship: bool
    alerts: List[str]


# -- Pure functions -----------------------------------------------------------

def kcal_per_kg(is_freeze_dried: bool) -> float:
    """Caloric density by food type."""
    return KCAL_PER_KG_FREEZE_DRIED if is_freeze_dried else KCAL_PER_KG_FRESH


def freeze_dry_energy_kwh(wet_mass_kg: float) -> float:
    """Energy to freeze-dry a given mass of wet food.

    Removes water fraction, then accounts for sublimation energy
    plus vacuum pump operation.
    """
    if wet_mass_kg <= 0.0:
        return 0.0
    water_kg = wet_mass_kg * FREEZE_DRY_WATER_FRACTION
    sublimation_mj = water_kg * WATER_SUBLIMATION_ENERGY_MJ_KG / FREEZE_DRY_EFFICIENCY
    sublimation_kwh = sublimation_mj * MJ_TO_KWH
    pump_kwh = VACUUM_PUMP_POWER_KW * FREEZE_DRY_HOURS_PER_KG * wet_mass_kg
    return sublimation_kwh + pump_kwh


def freeze_dry_yield_kg(wet_mass_kg: float) -> float:
    """Dry mass remaining after freeze-drying."""
    if wet_mass_kg <= 0.0:
        return 0.0
    return wet_mass_kg * (1.0 - FREEZE_DRY_WATER_FRACTION)


def days_of_food(total_kcal: float, crew_count: int, kcal_per_sol: float) -> float:
    """Calculate how many sols of food remain."""
    if crew_count <= 0 or kcal_per_sol <= 0.0:
        return float("inf")
    daily_need = crew_count * kcal_per_sol
    if daily_need <= 0.0:
        return float("inf")
    return total_kcal / daily_need


def ration_level_from_days(food_days: float) -> str:
    """Determine rationing level from days of food remaining."""
    if food_days <= RATION_STARVATION_DAYS:
        return "starvation"
    if food_days <= RATION_CRITICAL_DAYS:
        return "critical"
    if food_days <= RATION_WARNING_DAYS:
        return "warning"
    if food_days <= RATION_CAUTION_DAYS:
        return "caution"
    return "normal"


def consumption_rate(ration_level: str) -> float:
    """Kcal per person per sol based on ration level."""
    rates = {
        "normal": KCAL_PER_PERSON_SOL_NORMAL,
        "caution": KCAL_PER_PERSON_SOL_NORMAL * 0.9,
        "warning": KCAL_PER_PERSON_SOL_RATION,
        "critical": KCAL_PER_PERSON_SOL_RATION * 0.85,
        "starvation": KCAL_PER_PERSON_SOL_RATION * 0.6,
    }
    return rates.get(ration_level, KCAL_PER_PERSON_SOL_NORMAL)


def nutrient_decay(quality: float, sols: int = 1) -> float:
    """Exponential nutrient quality degradation per sol."""
    return quality * math.exp(-NUTRIENT_DECAY_RATE_PER_SOL * sols)


def vitamin_decay(level: float, half_life_sols: float, sols: int = 1) -> float:
    """Exponential vitamin level decay based on half-life."""
    if half_life_sols <= 0.0:
        return 0.0
    decay_rate = math.log(2.0) / half_life_sols
    return level * math.exp(-decay_rate * sols)


def caloric_decay(kcal: float, sols: int = 1) -> float:
    """Exponential caloric degradation (very slow for freeze-dried)."""
    return kcal * math.exp(-CALORIC_DECAY_RATE_PER_SOL * sols)


def fresh_spoilage_rate(temp_c: float) -> float:
    """Fraction of fresh food spoiled per sol based on storage temp.

    Below -20°C: essentially zero (frozen solid).
    -20°C to 4°C: slow degradation.
    Above 4°C: rapid spoilage.
    """
    if temp_c < -20.0:
        return 0.001
    if temp_c < 4.0:
        return 0.005 + 0.002 * (temp_c + 20.0) / 24.0
    return 0.05 + 0.01 * min(temp_c - 4.0, 30.0)


def cold_storage_power_kwh(fresh_kg: float, sol_hours: float = 24.66) -> float:
    """Power for cold storage refrigeration."""
    if fresh_kg <= 0.0:
        return 0.0
    utilization = min(1.0, fresh_kg / COLD_STORAGE_CAPACITY_KG)
    return COLD_STORAGE_POWER_KW * utilization * sol_hours


def check_seal_breach(
    num_containers: int,
    rng: Optional[random.Random] = None,
) -> int:
    """Check how many storage containers suffer seal breaches."""
    r = rng or random.Random()
    breaches = 0
    for _ in range(num_containers):
        if r.random() < CONTAINER_SEAL_FAILURE_RATE:
            breaches += 1
    return breaches


def supply_ship_due(sol: int, sols_since_supply: int) -> bool:
    """Check if a supply ship arrives this sol."""
    return sols_since_supply >= SUPPLY_SHIP_INTERVAL_SOLS


# -- Tick function ------------------------------------------------------------

def tick_storage(
    state: StorageState,
    greenhouse_harvest_kg: float = 0.0,
    temp_c: float = MARS_AVG_TEMP_C,
    rng: Optional[random.Random] = None,
) -> SolReport:
    """Advance the food storage system by one sol.

    Args:
        state: Current storage state (mutated in place).
        greenhouse_harvest_kg: Fresh food from greenhouse this sol.
        temp_c: Storage ambient temperature.
        rng: Random number generator for stochastic events.

    Returns:
        SolReport with this sol's food storage metrics.
    """
    state.sol += 1
    r = rng or random.Random()
    alerts: List[str] = []
    sol_power = 0.0
    sol_spoilage = 0.0
    sol_contamination = 0.0

    # -- 1. Receive greenhouse harvest ------------------------------------
    if greenhouse_harvest_kg > 0.0:
        state.fresh_kg += greenhouse_harvest_kg
        fresh_kcal = greenhouse_harvest_kg * KCAL_PER_KG_FRESH
        state.total_kcal += fresh_kcal
        batch = FoodBatch(
            batch_id=state.next_batch_id,
            sol_stored=state.sol,
            mass_kg=greenhouse_harvest_kg,
            kcal_total=fresh_kcal,
            nutrient_quality=1.0,
            is_freeze_dried=False,
            source="greenhouse",
        )
        state.batches.append(batch)
        state.next_batch_id += 1

    # -- 2. Freeze-dry processing -----------------------------------------
    available_to_dry = min(state.fresh_kg, MAX_FREEZE_DRY_KG_SOL)
    if available_to_dry > 0.0:
        dry_mass = freeze_dry_yield_kg(available_to_dry)
        energy = freeze_dry_energy_kwh(available_to_dry)
        sol_power += energy

        # Fresh → freeze-dried conversion (calories preserved, mass reduced)
        fresh_kcal_removed = available_to_dry * KCAL_PER_KG_FRESH
        dry_kcal_added = dry_mass * KCAL_PER_KG_FREEZE_DRIED
        state.total_kcal += (dry_kcal_added - fresh_kcal_removed)

        state.fresh_kg -= available_to_dry
        state.freeze_dried_kg += dry_mass

        if state.freeze_dried_kg > STORAGE_CAPACITY_KG:
            overflow = state.freeze_dried_kg - STORAGE_CAPACITY_KG
            state.freeze_dried_kg = STORAGE_CAPACITY_KG
            overflow_kcal = overflow * KCAL_PER_KG_FREEZE_DRIED
            state.total_kcal -= overflow_kcal
            alerts.append(
                f"Storage overflow: {overflow:.1f} kg freeze-dried food lost"
            )

        state.freeze_dry_processed_kg += dry_mass

    # -- 3. Fresh food spoilage -------------------------------------------
    if state.fresh_kg > 0.0:
        spoil_rate = fresh_spoilage_rate(temp_c)
        spoiled = state.fresh_kg * spoil_rate
        spoiled_kcal = spoiled * KCAL_PER_KG_FRESH
        state.fresh_kg = max(0.0, state.fresh_kg - spoiled)
        state.total_kcal = max(0.0, state.total_kcal - spoiled_kcal)
        state.spoilage_kg += spoiled
        sol_spoilage = spoiled
        sol_power += cold_storage_power_kwh(state.fresh_kg)

    # -- 4. Caloric degradation of stored food ----------------------------
    decay_loss = state.total_kcal * (1.0 - math.exp(-CALORIC_DECAY_RATE_PER_SOL))
    state.total_kcal = max(0.0, state.total_kcal - decay_loss)

    # -- 5. Nutrient/vitamin degradation ----------------------------------
    state.avg_nutrient_quality = nutrient_decay(state.avg_nutrient_quality)
    state.vitamin_c_level = vitamin_decay(
        state.vitamin_c_level, VITAMIN_C_HALF_LIFE_SOLS
    )
    state.vitamin_b12_level = vitamin_decay(
        state.vitamin_b12_level, VITAMIN_B12_HALF_LIFE_SOLS
    )
    state.vitamin_a_level = vitamin_decay(
        state.vitamin_a_level, VITAMIN_A_HALF_LIFE_SOLS
    )

    # -- 6. Container seal breaches (perchlorate contamination) -----------
    num_containers = max(1, int(state.freeze_dried_kg / 25.0))  # ~25 kg/container
    breaches = check_seal_breach(num_containers, r)
    if breaches > 0:
        lost_per_breach = 25.0 * PERCHLORATE_CONTAMINATION_LOSS
        total_lost = min(breaches * lost_per_breach, state.freeze_dried_kg)
        lost_kcal = total_lost * KCAL_PER_KG_FREEZE_DRIED
        state.freeze_dried_kg = max(0.0, state.freeze_dried_kg - total_lost)
        state.total_kcal = max(0.0, state.total_kcal - lost_kcal)
        state.contamination_kg += total_lost
        sol_contamination = total_lost
        alerts.append(
            f"SEAL BREACH: {breaches} container(s), "
            f"{total_lost:.1f} kg contaminated by perchlorates"
        )

    # -- 7. Supply ship check ---------------------------------------------
    state.sols_since_supply += 1
    ship_arrived = False
    if supply_ship_due(state.sol, state.sols_since_supply):
        ship_kg = SUPPLY_SHIP_FOOD_KG_PER_PERSON * state.crew_count
        ship_kcal = ship_kg * SUPPLY_SHIP_KCAL_PER_KG
        state.freeze_dried_kg += ship_kg
        state.total_kcal += ship_kcal
        state.sols_since_supply = 0
        state.supply_ships_received += 1
        ship_arrived = True
        # Reset vitamin levels (fresh supply)
        state.vitamin_c_level = min(1.0, state.vitamin_c_level + 0.5)
        state.vitamin_b12_level = min(1.0, state.vitamin_b12_level + 0.5)
        state.vitamin_a_level = min(1.0, state.vitamin_a_level + 0.5)
        state.avg_nutrient_quality = min(
            1.0, state.avg_nutrient_quality + 0.3
        )
        alerts.append(
            f"SUPPLY SHIP: +{ship_kg:.0f} kg food ({ship_kcal:.0f} kcal)"
        )

    # -- 8. Consumption ---------------------------------------------------
    food_d = days_of_food(
        state.total_kcal, state.crew_count, KCAL_PER_PERSON_SOL_NORMAL
    )
    state.ration_level = ration_level_from_days(food_d)
    kcal_per_person = consumption_rate(state.ration_level)
    total_consumption = kcal_per_person * state.crew_count

    if total_consumption > state.total_kcal:
        total_consumption = state.total_kcal
        kcal_per_person = total_consumption / max(1, state.crew_count)
        if kcal_per_person < KCAL_PER_PERSON_SOL_RATION * 0.5:
            alerts.append("STARVATION: insufficient calories for crew")
            state.sols_starving += 1

    state.total_kcal = max(0.0, state.total_kcal - total_consumption)
    state.kcal_consumed_today = total_consumption
    state.cumulative_kcal_consumed += total_consumption

    # Deplete mass proportionally (freeze-dried first, then fresh)
    mass_consumed = total_consumption / KCAL_PER_KG_FREEZE_DRIED
    if mass_consumed <= state.freeze_dried_kg:
        state.freeze_dried_kg -= mass_consumed
    else:
        remainder_kcal = (
            (mass_consumed - state.freeze_dried_kg)
            * KCAL_PER_KG_FREEZE_DRIED
        )
        state.freeze_dried_kg = 0.0
        fresh_consumed = remainder_kcal / max(1.0, KCAL_PER_KG_FRESH)
        state.fresh_kg = max(0.0, state.fresh_kg - fresh_consumed)

    if state.ration_level in ("warning", "critical", "starvation"):
        state.sols_rationed += 1

    # -- 9. Vitamin deficiency alerts -------------------------------------
    if state.vitamin_c_level < 0.2:
        alerts.append("SCURVY RISK: Vitamin C critically low")
    if state.vitamin_b12_level < 0.2:
        alerts.append("B12 DEFICIENCY: neurological risk")
    if state.vitamin_a_level < 0.2:
        alerts.append("NIGHT BLINDNESS RISK: Vitamin A low")

    # -- 10. Power accounting ---------------------------------------------
    state.power_consumed_kwh += sol_power
    state.total_loss_kg = state.spoilage_kg + state.contamination_kg

    # Recalculate food days after consumption
    food_d_post = days_of_food(
        state.total_kcal, state.crew_count, KCAL_PER_PERSON_SOL_NORMAL
    )

    return SolReport(
        sol=state.sol,
        ration_level=state.ration_level,
        days_of_food=food_d_post,
        total_kcal=state.total_kcal,
        freeze_dried_kg=state.freeze_dried_kg,
        fresh_kg=state.fresh_kg,
        kcal_consumed=total_consumption,
        kcal_per_person=kcal_per_person,
        nutrient_quality=state.avg_nutrient_quality,
        vitamin_c=state.vitamin_c_level,
        vitamin_b12=state.vitamin_b12_level,
        fresh_processed_kg=available_to_dry if available_to_dry > 0.0 else 0.0,
        spoilage_kg=sol_spoilage,
        contamination_kg=sol_contamination,
        power_kwh=sol_power,
        supply_ship=ship_arrived,
        alerts=alerts,
    )


# -- Factory functions --------------------------------------------------------

def make_storage(
    crew_count: int = 20,
    initial_food_kg: float = 2000.0,
) -> StorageState:
    """Create a food storage system with initial provisions.

    Default: 2000 kg freeze-dried food = ~152 days for 20 crew.
    """
    initial_kcal = initial_food_kg * KCAL_PER_KG_FREEZE_DRIED
    state = StorageState(crew_count=crew_count)
    state.freeze_dried_kg = initial_food_kg
    state.total_kcal = initial_kcal
    state.batches.append(FoodBatch(
        batch_id=0,
        sol_stored=0,
        mass_kg=initial_food_kg,
        kcal_total=initial_kcal,
        nutrient_quality=1.0,
        is_freeze_dried=True,
        source="supply_ship",
    ))
    return state


def run_storage(
    sols: int = 365,
    crew_count: int = 20,
    initial_food_kg: float = 2000.0,
    greenhouse_kg_per_sol: float = 15.0,
    seed: Optional[int] = None,
) -> List[SolReport]:
    """Run the food storage simulation for N sols.

    Args:
        sols: Number of sols to simulate.
        crew_count: Number of crew members.
        initial_food_kg: Starting freeze-dried food (kg).
        greenhouse_kg_per_sol: Daily greenhouse harvest (kg fresh).
        seed: RNG seed for reproducibility.

    Returns:
        List of SolReport for each sol.
    """
    storage = make_storage(crew_count, initial_food_kg)
    rng = random.Random(seed)
    reports: List[SolReport] = []

    for _ in range(sols):
        # Greenhouse output varies ±20%
        harvest = greenhouse_kg_per_sol * rng.uniform(0.8, 1.2)
        report = tick_storage(storage, harvest, MARS_AVG_TEMP_C, rng)
        reports.append(report)

    return reports
