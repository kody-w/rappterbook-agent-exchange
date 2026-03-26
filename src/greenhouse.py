"""
greenhouse.py — Mars agricultural module.

Models food production via pressurised greenhouse hydroponics:
  - Crop growth cycles (planting → maturity → harvest)
  - Light requirements (PAR from solar + LED supplemental)
  - Water consumption (transpiration + nutrient solution)
  - CO2 uptake / O2 release (photosynthesis assists ECLSS)
  - Caloric yield per crop type
  - Crop failure from environmental stress (cold, low light, pests)

Physical references:
  - NASA VEGGIE (ISS): lettuce harvest in 28 days, ~0.4 m²/crew supplemental
  - Controlled Ecological Life Support System (CELSS): 50 m²/person full diet
  - Human caloric need: ~2500 kcal/person/sol
  - Lettuce yield: ~3.5 kg/m²/cycle (28 days), 150 kcal/kg
  - Potato yield: ~5.0 kg/m²/cycle (90 days), 770 kcal/kg
  - Wheat yield:  ~0.8 kg/m²/cycle (120 days), 3400 kcal/kg
  - Soybean yield: ~0.4 kg/m²/cycle (100 days), 4460 kcal/kg
  - Photosynthesis: 6CO2 + 6H2O → C6H12O6 + 6O2
  - CO2 uptake: ~0.02 kg CO2/m²/sol for active crops
  - O2 release: ~0.015 kg O2/m²/sol for active crops
  - Water use: ~5 L/m²/sol (recirculating hydro, 80% recycled)
  - PAR requirement: 200-600 µmol/m²/s, ~12 hrs/sol
  - LED grow light power: ~0.04 kWh/m²/sol supplemental

One tick = one sol. Mass in kg, energy in kWh, area in m².
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

CALORIES_PER_PERSON_SOL = 2500.0

# CO2/O2 exchange rates for active growing area
CO2_UPTAKE_KG_M2_SOL = 0.02     # kg CO2 absorbed per m² per sol
O2_RELEASE_KG_M2_SOL = 0.015    # kg O2 released per m² per sol

# Water usage (net after recirculation)
WATER_USE_L_M2_SOL = 5.0         # gross water flow
WATER_RECYCLE_FRACTION = 0.80    # 80% recaptured in hydroponic loop
WATER_NET_L_M2_SOL = WATER_USE_L_M2_SOL * (1.0 - WATER_RECYCLE_FRACTION)

# LED supplemental lighting
LED_KWH_M2_SOL = 0.04           # energy for supplemental grow lights

# Crop database: (cycle_days, yield_kg_m2, kcal_per_kg)
CROP_DATA: dict[str, tuple[int, float, float]] = {
    "lettuce":  (28,  3.5, 150.0),
    "potato":   (90,  5.0, 770.0),
    "wheat":    (120, 0.8, 3400.0),
    "soybean":  (100, 0.4, 4460.0),
}

# Growth efficiency modifiers
MIN_LIGHT_FRACTION = 0.15        # below this solar fraction, crops fail
COLD_STRESS_TEMP_C = -20.0       # below this, growth halts (greenhouse insulated)
OPTIMAL_TEMP_C = 22.0            # ideal greenhouse temp


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CropBed:
    """A single crop bed in the greenhouse.

    Attributes:
        crop: crop type key from CROP_DATA
        area_m2: planting area
        day_in_cycle: current growth day (0 = just planted)
        health: crop health 0.0-1.0 (stress reduces this)
        harvests: total number of completed harvest cycles
    """
    crop: str
    area_m2: float
    day_in_cycle: int = 0
    health: float = 1.0
    harvests: int = 0

    def __post_init__(self) -> None:
        if self.crop not in CROP_DATA:
            raise ValueError(f"Unknown crop: {self.crop}. "
                             f"Valid: {list(CROP_DATA.keys())}")
        self.area_m2 = max(0.0, self.area_m2)
        self.health = max(0.0, min(1.0, self.health))
        self.day_in_cycle = max(0, self.day_in_cycle)

    @property
    def cycle_days(self) -> int:
        """Total days in growth cycle for this crop."""
        return CROP_DATA[self.crop][0]

    @property
    def yield_kg_m2(self) -> float:
        """Yield in kg per m² at full health."""
        return CROP_DATA[self.crop][1]

    @property
    def kcal_per_kg(self) -> float:
        """Caloric density of harvested crop."""
        return CROP_DATA[self.crop][2]

    def growth_fraction(self) -> float:
        """How far through the growth cycle (0.0 to 1.0)."""
        if self.cycle_days <= 0:
            return 1.0
        return min(1.0, self.day_in_cycle / self.cycle_days)

    def is_mature(self) -> bool:
        """True if crop is ready to harvest."""
        return self.day_in_cycle >= self.cycle_days


@dataclass
class Greenhouse:
    """Complete greenhouse facility.

    Attributes:
        beds: list of active crop beds
        total_food_kg: cumulative food harvested (kg)
        total_calories: cumulative calories produced
        total_harvests: total harvest events
    """
    beds: list[CropBed] = field(default_factory=list)
    total_food_kg: float = 0.0
    total_calories: float = 0.0
    total_harvests: int = 0

    def active_area_m2(self) -> float:
        """Total planted area across all beds."""
        return sum(bed.area_m2 for bed in self.beds)

    def add_bed(self, crop: str, area_m2: float) -> CropBed:
        """Plant a new crop bed."""
        bed = CropBed(crop=crop, area_m2=area_m2)
        self.beds.append(bed)
        return bed


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def light_efficiency(solar_flux_fraction: float) -> float:
    """Growth efficiency from available light.

    solar_flux_fraction: fraction of nominal Mars solar flux reaching
    greenhouse (0.0 = total darkness, 1.0 = clear day).
    Below MIN_LIGHT_FRACTION, crops get no useful photons.

    Returns efficiency multiplier 0.0-1.0.
    """
    if solar_flux_fraction < MIN_LIGHT_FRACTION:
        return 0.0
    # Diminishing returns above 0.6 — plants saturate
    if solar_flux_fraction >= 0.6:
        return 1.0
    # Linear ramp from MIN_LIGHT_FRACTION to 0.6
    return (solar_flux_fraction - MIN_LIGHT_FRACTION) / (0.6 - MIN_LIGHT_FRACTION)


def temperature_efficiency(greenhouse_temp_c: float) -> float:
    """Growth efficiency from greenhouse temperature.

    Below COLD_STRESS_TEMP_C: zero growth.
    Optimal at ~22°C. Above 40°C: heat stress.

    Returns efficiency multiplier 0.0-1.0.
    """
    if greenhouse_temp_c < COLD_STRESS_TEMP_C:
        return 0.0
    if greenhouse_temp_c < 5.0:
        return (greenhouse_temp_c - COLD_STRESS_TEMP_C) / (5.0 - COLD_STRESS_TEMP_C) * 0.3
    if greenhouse_temp_c <= 30.0:
        # Bell-like curve peaking at OPTIMAL_TEMP_C
        delta = abs(greenhouse_temp_c - OPTIMAL_TEMP_C)
        return max(0.3, 1.0 - (delta / 20.0) ** 2)
    if greenhouse_temp_c <= 45.0:
        return max(0.0, 1.0 - (greenhouse_temp_c - 30.0) / 15.0)
    return 0.0


def grow_bed(
    bed: CropBed,
    light_eff: float,
    temp_eff: float,
) -> None:
    """Advance one crop bed by one sol.

    Growth efficiency affects health. If conditions are poor,
    health degrades. If conditions are good, health recovers.
    """
    combined_eff = min(1.0, light_eff * temp_eff)

    # Health dynamics: good conditions heal, bad conditions damage
    if combined_eff > 0.5:
        recovery = (combined_eff - 0.5) * 0.02  # slow recovery
        bed.health = min(1.0, bed.health + recovery)
    elif combined_eff < 0.3:
        damage = (0.3 - combined_eff) * 0.05    # faster damage
        bed.health = max(0.0, bed.health - damage)

    # Advance growth day (scaled by efficiency and health)
    if combined_eff > 0 and bed.health > 0.1:
        bed.day_in_cycle += 1


def harvest_bed(bed: CropBed) -> tuple[float, float]:
    """Harvest a mature crop bed. Returns (food_kg, calories).

    Yield is scaled by crop health at harvest time.
    Resets the bed for the next growth cycle.
    """
    if not bed.is_mature():
        return (0.0, 0.0)

    food_kg = bed.yield_kg_m2 * bed.area_m2 * bed.health
    calories = food_kg * bed.kcal_per_kg

    # Reset for next cycle
    bed.day_in_cycle = 0
    bed.harvests += 1

    return (round(food_kg, 4), round(calories, 2))


def greenhouse_gas_exchange(active_area_m2: float) -> dict[str, float]:
    """Calculate CO2/O2 exchange from photosynthesis for one sol.

    Returns dict with co2_absorbed_kg and o2_released_kg.
    """
    return {
        "co2_absorbed_kg": round(CO2_UPTAKE_KG_M2_SOL * active_area_m2, 6),
        "o2_released_kg": round(O2_RELEASE_KG_M2_SOL * active_area_m2, 6),
    }


def greenhouse_water_demand(active_area_m2: float) -> float:
    """Net water consumption for one sol (liters ≈ kg)."""
    return round(WATER_NET_L_M2_SOL * active_area_m2, 4)


def greenhouse_power_demand(active_area_m2: float) -> float:
    """LED supplemental lighting power for one sol (kWh)."""
    return round(LED_KWH_M2_SOL * active_area_m2, 4)


def food_coverage(daily_calories: float, crew_count: int) -> float:
    """Fraction of crew caloric needs met by greenhouse output.

    Returns 0.0-1.0+ (can exceed 1.0 if surplus).
    """
    if crew_count <= 0:
        return 1.0 if daily_calories > 0 else 0.0
    needed = CALORIES_PER_PERSON_SOL * crew_count
    if needed <= 0:
        return 1.0
    return daily_calories / needed


def tick_greenhouse(
    greenhouse: Greenhouse,
    solar_flux_fraction: float,
    greenhouse_temp_c: float,
) -> dict:
    """Advance greenhouse by one sol.

    Steps:
      1. Calculate light and temperature efficiency
      2. Grow all beds (advance day, adjust health)
      3. Harvest mature beds
      4. Calculate gas exchange, water, and power needs

    Args:
        greenhouse: the greenhouse state (mutated in place)
        solar_flux_fraction: fraction of nominal solar reaching greenhouse
        greenhouse_temp_c: internal greenhouse temperature (°C)

    Returns:
        Status dict with all greenhouse metrics.
    """
    light_eff = light_efficiency(solar_flux_fraction)
    temp_eff = temperature_efficiency(greenhouse_temp_c)

    # Grow all beds
    for bed in greenhouse.beds:
        grow_bed(bed, light_eff, temp_eff)

    # Harvest mature beds
    sol_food_kg = 0.0
    sol_calories = 0.0
    harvests_this_sol = 0
    for bed in greenhouse.beds:
        if bed.is_mature():
            food_kg, cal = harvest_bed(bed)
            sol_food_kg += food_kg
            sol_calories += cal
            harvests_this_sol += 1

    greenhouse.total_food_kg += sol_food_kg
    greenhouse.total_calories += sol_calories
    greenhouse.total_harvests += harvests_this_sol

    # Resource demands
    active_area = greenhouse.active_area_m2()
    gas = greenhouse_gas_exchange(active_area)
    water_demand = greenhouse_water_demand(active_area)
    power_demand = greenhouse_power_demand(active_area)

    # Bed summaries
    bed_status = []
    for bed in greenhouse.beds:
        bed_status.append({
            "crop": bed.crop,
            "area_m2": bed.area_m2,
            "day": bed.day_in_cycle,
            "cycle_days": bed.cycle_days,
            "growth": round(bed.growth_fraction(), 3),
            "health": round(bed.health, 4),
            "harvests": bed.harvests,
        })

    return {
        "light_efficiency": round(light_eff, 4),
        "temperature_efficiency": round(temp_eff, 4),
        "active_area_m2": round(active_area, 2),
        "food_harvested_kg": round(sol_food_kg, 4),
        "calories_harvested": round(sol_calories, 2),
        "harvests_this_sol": harvests_this_sol,
        "total_food_kg": round(greenhouse.total_food_kg, 4),
        "total_calories": round(greenhouse.total_calories, 2),
        "co2_absorbed_kg": gas["co2_absorbed_kg"],
        "o2_released_kg": gas["o2_released_kg"],
        "water_demand_kg": round(water_demand, 4),
        "power_demand_kwh": round(power_demand, 4),
        "beds": bed_status,
    }


def create_starter_greenhouse() -> Greenhouse:
    """Create a basic Mars greenhouse with mixed crop rotation.

    20 m² lettuce (fast harvest, morale food)
    30 m² potato (calorie staple)
    15 m² wheat (long-term grain)
    10 m² soybean (protein source)
    = 75 m² total
    """
    gh = Greenhouse()
    gh.add_bed("lettuce", 20.0)
    gh.add_bed("potato", 30.0)
    gh.add_bed("wheat", 15.0)
    gh.add_bed("soybean", 10.0)
    return gh
