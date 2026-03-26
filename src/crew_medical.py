"""
crew_medical.py — Mars colony crew health and medical system.

Models crew health as a function of environmental stressors:
  - Radiation sickness from accumulated dose (ties to rad_shield)
  - Nutritional deficiency from food shortages (ties to greenhouse)
  - Hypoxia risk from low O₂ partial pressure (ties to life_support)
  - CO₂ toxicity from high CO₂ levels (ties to life_support)
  - Psychological stress from isolation, low morale, and crowding
  - Injury from accidents (EVA, construction, dust storms)
  - Medical treatment: limited supplies, recovery rates, triage

Physical references:
  - NASA radiation limits: 600 mSv career (NSCR-2020 model)
  - Acute radiation syndrome: >700 mSv = nausea, >4000 mSv = lethal
  - Hypoxia onset: O₂ < 16 kPa, impairment < 14 kPa, lethal < 10 kPa
  - CO₂ toxicity: >2 kPa headache, >5 kPa dangerous, >8 kPa lethal
  - Caloric deficiency: <1500 kcal/day = health decline after 30 sols
  - Mars psych studies: isolation stress increases after ~200 sols
  - ISS medical: ~1 minor injury per crew-member per 6 months
  - Medical supply consumption: ~0.5 kg/person/sol (bandages, meds, etc.)

One tick = one sol. Dose in mSv, pressure in kPa, calories in kcal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants / medical thresholds
# ---------------------------------------------------------------------------

# Radiation thresholds (cumulative mSv)
RAD_CAREER_LIMIT_MSV = 600.0          # NASA career limit
RAD_NAUSEA_THRESHOLD_MSV = 700.0      # acute radiation syndrome onset
RAD_LETHAL_MSV = 4000.0               # lethal dose (unshielded SPE)
RAD_HEALTH_PENALTY_PER_MSV = 0.0001   # health loss per mSv above baseline

# Atmospheric thresholds (kPa)
O2_NORMAL_KPA = 21.3                  # Earth-normal O₂
O2_HYPOXIA_ONSET_KPA = 16.0           # mild impairment begins
O2_IMPAIRMENT_KPA = 14.0              # cognitive impairment
O2_LETHAL_KPA = 10.0                  # lethal without intervention

CO2_SAFE_KPA = 0.5                    # NASA long-duration limit
CO2_HEADACHE_KPA = 2.0                # headaches, drowsiness
CO2_DANGEROUS_KPA = 5.0               # dangerous — confusion, panic
CO2_LETHAL_KPA = 8.0                  # lethal

# Nutrition
CALORIES_DAILY_NEED = 2500.0          # kcal/person/sol
CALORIE_DEFICIT_GRACE_SOLS = 30       # sols before health impact
STARVATION_LETHAL_SOLS = 90           # death from total starvation

# Psychology
ISOLATION_STRESS_ONSET_SOLS = 200     # stress increases after this many sols
CROWDING_THRESHOLD = 25.0             # m² per person below which crowding stress
OPTIMAL_SPACE_M2 = 50.0              # ideal living space per person

# Injury
BASE_INJURY_RATE = 0.002              # probability of injury per person per sol
STORM_INJURY_MULTIPLIER = 3.0         # during dust storms
EVA_INJURY_MULTIPLIER = 2.0           # during EVA operations

# Medical capacity
MEDICAL_SUPPLIES_KG_PER_SOL = 0.5     # supply consumption per patient per sol
NATURAL_HEALING_RATE = 0.02           # health recovery per sol (untreated)
TREATED_HEALING_RATE = 0.05           # health recovery per sol (with treatment)
CRITICAL_THRESHOLD = 0.2              # below this, crew member is critical
DEATH_THRESHOLD = 0.0                 # at zero health, death


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CrewHealth:
    """Aggregate health state of the colony crew.

    Tracks average and worst-case health across all crew members.
    Individual tracking is abstracted to population-level statistics.

    Attributes:
        avg_health: mean crew health [0.0, 1.0] where 1.0 = perfect
        min_health: worst individual health score
        healthy_count: crew with health > 0.8
        injured_count: crew with health 0.2-0.8
        critical_count: crew with health < 0.2
        cumulative_rad_msv: average cumulative radiation dose
        calorie_deficit_sols: consecutive sols below calorie needs
        isolation_sols: total sols since colony founding
        medical_supplies_kg: remaining medical supplies
    """
    avg_health: float = 1.0
    min_health: float = 1.0
    healthy_count: int = 0
    injured_count: int = 0
    critical_count: int = 0
    cumulative_rad_msv: float = 0.0
    calorie_deficit_sols: int = 0
    isolation_sols: int = 0
    medical_supplies_kg: float = 500.0

    def __post_init__(self) -> None:
        self.avg_health = max(0.0, min(1.0, self.avg_health))
        self.min_health = max(0.0, min(1.0, self.min_health))
        self.healthy_count = max(0, self.healthy_count)
        self.injured_count = max(0, self.injured_count)
        self.critical_count = max(0, self.critical_count)
        self.cumulative_rad_msv = max(0.0, self.cumulative_rad_msv)
        self.calorie_deficit_sols = max(0, self.calorie_deficit_sols)
        self.isolation_sols = max(0, self.isolation_sols)
        self.medical_supplies_kg = max(0.0, self.medical_supplies_kg)


@dataclass
class MedicalReport:
    """Status report from one sol of medical operations.

    Attributes:
        health_delta: change in average health this sol
        rad_damage: health penalty from radiation
        nutrition_damage: health penalty from calorie deficit
        o2_damage: health penalty from low oxygen
        co2_damage: health penalty from high CO₂
        psych_damage: health penalty from psychological stress
        injury_damage: health penalty from accidents
        healing: health recovered from treatment + natural healing
        supplies_used_kg: medical supplies consumed
        deaths_medical: deaths caused by health reaching zero
    """
    health_delta: float = 0.0
    rad_damage: float = 0.0
    nutrition_damage: float = 0.0
    o2_damage: float = 0.0
    co2_damage: float = 0.0
    psych_damage: float = 0.0
    injury_damage: float = 0.0
    healing: float = 0.0
    supplies_used_kg: float = 0.0
    deaths_medical: int = 0


# ---------------------------------------------------------------------------
# Damage functions
# ---------------------------------------------------------------------------

def radiation_damage(dose_msv_sol: float, cumulative_msv: float) -> float:
    """Health damage from radiation exposure this sol.

    Chronic: slow damage from cumulative dose above career limit.
    Acute: rapid damage from high single-sol dose (SPE events).
    """
    dose_msv_sol = max(0.0, dose_msv_sol)
    cumulative_msv = max(0.0, cumulative_msv)

    chronic = 0.0
    if cumulative_msv > RAD_CAREER_LIMIT_MSV:
        excess = cumulative_msv - RAD_CAREER_LIMIT_MSV
        chronic = excess * RAD_HEALTH_PENALTY_PER_MSV * 0.01

    acute = 0.0
    if dose_msv_sol > 10.0:
        acute = (dose_msv_sol / RAD_NAUSEA_THRESHOLD_MSV) * 0.05
    if dose_msv_sol > RAD_NAUSEA_THRESHOLD_MSV:
        acute = 0.15
    if dose_msv_sol > RAD_LETHAL_MSV:
        acute = 0.8

    return min(1.0, chronic + acute)


def nutrition_damage(calories_available: float, population: int,
                     deficit_sols: int) -> float:
    """Health damage from calorie deficit.

    No damage if adequate food. Gradual onset after grace period.
    """
    if population <= 0:
        return 0.0
    calories_available = max(0.0, calories_available)
    per_capita = calories_available / population

    if per_capita >= CALORIES_DAILY_NEED:
        return 0.0

    shortfall_fraction = 1.0 - (per_capita / CALORIES_DAILY_NEED)

    if deficit_sols <= CALORIE_DEFICIT_GRACE_SOLS:
        return shortfall_fraction * 0.005
    else:
        severity = min(1.0, deficit_sols / STARVATION_LETHAL_SOLS)
        return shortfall_fraction * severity * 0.05


def oxygen_damage(o2_kpa: float) -> float:
    """Health damage from low oxygen partial pressure."""
    o2_kpa = max(0.0, o2_kpa)

    if o2_kpa >= O2_HYPOXIA_ONSET_KPA:
        return 0.0
    if o2_kpa >= O2_IMPAIRMENT_KPA:
        fraction = 1.0 - (o2_kpa - O2_IMPAIRMENT_KPA) / (O2_HYPOXIA_ONSET_KPA - O2_IMPAIRMENT_KPA)
        return fraction * 0.02
    if o2_kpa >= O2_LETHAL_KPA:
        fraction = 1.0 - (o2_kpa - O2_LETHAL_KPA) / (O2_IMPAIRMENT_KPA - O2_LETHAL_KPA)
        return 0.02 + fraction * 0.18
    return 0.5


def co2_damage(co2_kpa: float) -> float:
    """Health damage from high CO₂ partial pressure."""
    co2_kpa = max(0.0, co2_kpa)

    if co2_kpa <= CO2_SAFE_KPA:
        return 0.0
    if co2_kpa <= CO2_HEADACHE_KPA:
        fraction = (co2_kpa - CO2_SAFE_KPA) / (CO2_HEADACHE_KPA - CO2_SAFE_KPA)
        return fraction * 0.01
    if co2_kpa <= CO2_DANGEROUS_KPA:
        fraction = (co2_kpa - CO2_HEADACHE_KPA) / (CO2_DANGEROUS_KPA - CO2_HEADACHE_KPA)
        return 0.01 + fraction * 0.09
    if co2_kpa <= CO2_LETHAL_KPA:
        fraction = (co2_kpa - CO2_DANGEROUS_KPA) / (CO2_LETHAL_KPA - CO2_DANGEROUS_KPA)
        return 0.10 + fraction * 0.40
    return 0.8


def psychological_damage(morale: float, isolation_sols: int,
                         population: int, habitat_area_m2: float) -> float:
    """Health damage from psychological stressors.

    Factors: morale, isolation duration, crowding.
    """
    morale = max(0.0, min(1.0, morale))
    damage = 0.0

    # Low morale
    if morale < 0.5:
        damage += (0.5 - morale) * 0.02

    # Isolation stress (increases over time)
    if isolation_sols > ISOLATION_STRESS_ONSET_SOLS:
        excess_sols = isolation_sols - ISOLATION_STRESS_ONSET_SOLS
        iso_factor = min(1.0, excess_sols / 1000.0)
        damage += iso_factor * 0.005

    # Crowding
    if population > 0 and habitat_area_m2 > 0:
        space_per_person = habitat_area_m2 / population
        if space_per_person < CROWDING_THRESHOLD:
            crowd_factor = 1.0 - (space_per_person / CROWDING_THRESHOLD)
            damage += crowd_factor * 0.01

    return min(0.1, damage)


def injury_chance(base_rate: float, dust_storm_active: bool,
                  eva_active: bool) -> float:
    """Probability of injury per person per sol."""
    rate = max(0.0, base_rate)
    if dust_storm_active:
        rate *= STORM_INJURY_MULTIPLIER
    if eva_active:
        rate *= EVA_INJURY_MULTIPLIER
    return min(1.0, rate)


def healing_rate(has_supplies: bool) -> float:
    """Health recovery rate per sol based on medical supplies."""
    return TREATED_HEALING_RATE if has_supplies else NATURAL_HEALING_RATE


# ---------------------------------------------------------------------------
# Main tick function
# ---------------------------------------------------------------------------

def tick_medical(
    crew: CrewHealth,
    population: int,
    radiation_msv_sol: float,
    o2_kpa: float,
    co2_kpa: float,
    calories_available: float,
    morale: float,
    habitat_area_m2: float = 2000.0,
    dust_storm_active: bool = False,
    eva_active: bool = False,
) -> MedicalReport:
    """Advance crew medical status by one sol.

    Computes damage from all environmental stressors, applies healing,
    updates crew health categories, and determines medical deaths.

    Args:
        crew: crew health state (mutated in place)
        population: current colony population
        radiation_msv_sol: radiation dose received this sol
        o2_kpa: habitat oxygen partial pressure
        co2_kpa: habitat CO₂ partial pressure
        calories_available: total food calories available
        morale: colony morale [0.0, 1.0]
        habitat_area_m2: pressurised living area
        dust_storm_active: True during dust storm
        eva_active: True during EVA operations

    Returns:
        MedicalReport for this sol.
    """
    report = MedicalReport()
    population = max(0, population)

    if population == 0:
        crew.healthy_count = 0
        crew.injured_count = 0
        crew.critical_count = 0
        return report

    # Track time
    crew.isolation_sols += 1

    # Accumulate radiation
    radiation_msv_sol = max(0.0, radiation_msv_sol)
    crew.cumulative_rad_msv += radiation_msv_sol

    # --- Compute all damage sources ---
    rad_dmg = radiation_damage(radiation_msv_sol, crew.cumulative_rad_msv)
    report.rad_damage = rad_dmg

    # Nutrition tracking
    per_capita_cal = calories_available / population if population > 0 else 0
    if per_capita_cal < CALORIES_DAILY_NEED:
        crew.calorie_deficit_sols += 1
    else:
        crew.calorie_deficit_sols = 0

    nut_dmg = nutrition_damage(calories_available, population, crew.calorie_deficit_sols)
    report.nutrition_damage = nut_dmg

    o2_dmg = oxygen_damage(o2_kpa)
    report.o2_damage = o2_dmg

    co2_dmg = co2_damage(co2_kpa)
    report.co2_damage = co2_dmg

    psych_dmg = psychological_damage(morale, crew.isolation_sols,
                                     population, habitat_area_m2)
    report.psych_damage = psych_dmg

    # Injuries: probabilistic damage spread across population
    inj_rate = injury_chance(BASE_INJURY_RATE, dust_storm_active, eva_active)
    # Expected injury damage across population (averaged)
    inj_dmg = inj_rate * 0.1  # each injury removes ~10% health
    report.injury_damage = inj_dmg

    # --- Total damage ---
    total_damage = rad_dmg + nut_dmg + o2_dmg + co2_dmg + psych_dmg + inj_dmg

    # --- Healing ---
    patients = crew.injured_count + crew.critical_count
    has_supplies = crew.medical_supplies_kg > 0 and patients > 0
    heal = healing_rate(has_supplies)

    if has_supplies and patients > 0:
        supplies_needed = patients * MEDICAL_SUPPLIES_KG_PER_SOL
        supplies_used = min(supplies_needed, crew.medical_supplies_kg)
        crew.medical_supplies_kg -= supplies_used
        report.supplies_used_kg = supplies_used
        supply_fraction = supplies_used / supplies_needed if supplies_needed > 0 else 0
        heal = NATURAL_HEALING_RATE + (TREATED_HEALING_RATE - NATURAL_HEALING_RATE) * supply_fraction

    report.healing = heal

    # --- Update health ---
    net_delta = heal - total_damage
    report.health_delta = net_delta

    crew.avg_health += net_delta
    crew.avg_health = max(0.0, min(1.0, crew.avg_health))

    # Min health degrades faster (worst case individuals)
    crew.min_health += net_delta * 1.5  # worst-case degrades 50% faster
    crew.min_health = max(0.0, min(crew.avg_health, crew.min_health))

    # --- Categorize crew ---
    # Distribute population across health bands based on avg/min health
    if crew.avg_health > 0.8:
        crew.critical_count = 0
        crew.injured_count = max(0, int(population * (1.0 - crew.avg_health) * 2))
        crew.healthy_count = population - crew.injured_count
    elif crew.avg_health > CRITICAL_THRESHOLD:
        critical_frac = max(0.0, (CRITICAL_THRESHOLD + 0.3 - crew.avg_health) / 0.3)
        crew.critical_count = max(0, int(population * critical_frac * 0.3))
        crew.injured_count = max(0, int(population * 0.5))
        crew.healthy_count = max(0, population - crew.injured_count - crew.critical_count)
    else:
        crew.critical_count = max(1, int(population * 0.4))
        crew.injured_count = max(0, int(population * 0.4))
        crew.healthy_count = max(0, population - crew.injured_count - crew.critical_count)

    # --- Medical deaths ---
    if crew.min_health <= DEATH_THRESHOLD and crew.avg_health < 0.15:
        # Mortality rate scales with how far below threshold
        death_rate = (0.15 - crew.avg_health) / 0.15
        report.deaths_medical = max(0, int(population * death_rate * 0.1))

    return report
