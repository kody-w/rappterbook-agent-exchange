"""
waste_recycler.py — Mars closed-loop biological waste processing.

On Mars there is no "away" to throw things. Every gram of matter the
colony imports from Earth is precious. This module models the waste
processing pipeline that closes the loop: human waste and crop residue
go in; clean water, fertilizer, and biogas come out.

Subsystems modelled:
  - Urine processing (vapor compression distillation → clean water)
  - Solid waste composting (aerobic thermophilic → fertilizer)
  - Crop residue digestion (anaerobic → biogas CH₄ + CO₂)
  - Brine recovery (electrolysis of distillation brine → water + salts)
  - Biogas capture (CH₄ for fuel, CO₂ for greenhouse enrichment)

Physical references:
  - ISS Water Recovery System (WRS): 93.5% urine water recovery
  - ISS Urine Processor Assembly (UPA): vapor compression distillation
  - Human urine output: ~1.5 L/person/day (NASA ECLSS handbook)
  - Human solid waste: ~0.12 kg/person/day dry mass (NASA HRP)
  - Human wastewater (hygiene): ~12 L/person/day on ISS
  - Aerobic composting temperature: 55-65°C (thermophilic)
  - Compost cycle: 60-90 sols for complete stabilization
  - Anaerobic digestion biogas yield: ~0.4 m³ CH₄/kg volatile solids
  - Biogas composition: ~60% CH₄, ~40% CO₂
  - Brine residual: ~15-20% of urine volume after primary distillation
  - Brine electrolysis water recovery: ~85% of brine water
  - Fertilizer N-P-K from human waste: ~11-1-2.5 g/person/day

One tick = one sol.  Volume in litres, mass in kg, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Human waste generation rates (per person per sol)
# ---------------------------------------------------------------------------

URINE_L_PER_PERSON_SOL = 1.5
SOLID_WASTE_KG_PER_PERSON_SOL = 0.12   # dry mass
HYGIENE_WATER_L_PER_PERSON_SOL = 12.0  # greywater from washing/bathing
CROP_RESIDUE_RATIO = 0.40               # inedible fraction of greenhouse harvest

# ---------------------------------------------------------------------------
# Urine processing (vapor compression distillation)
# ---------------------------------------------------------------------------

UPA_WATER_RECOVERY = 0.935             # ISS UPA baseline
UPA_POWER_KWH_PER_L = 0.08            # energy per litre of urine processed
BRINE_FRACTION = 0.065                 # fraction of urine that becomes brine
BRINE_ELECTROLYSIS_RECOVERY = 0.85     # water fraction recoverable from brine
BRINE_ELECTROLYSIS_KWH_PER_L = 0.25   # energy per litre of brine processed

# ---------------------------------------------------------------------------
# Greywater processing
# ---------------------------------------------------------------------------

GREY_WATER_RECOVERY = 0.95            # filtration + UV sterilization
GREY_POWER_KWH_PER_L = 0.03          # energy per litre of greywater

# ---------------------------------------------------------------------------
# Solid waste composting
# ---------------------------------------------------------------------------

COMPOST_CYCLE_SOLS = 75               # mean time to stable compost
COMPOST_TEMP_C = 60.0                 # target thermophilic temperature
COMPOST_POWER_KWH_PER_KG = 0.15      # heating + aeration energy per kg input
COMPOST_MASS_REDUCTION = 0.60         # mass fraction lost as CO₂ + H₂O vapor
COMPOST_WATER_RELEASE = 0.35          # fraction of input mass released as water
COMPOST_CO2_RELEASE_KG_PER_KG = 0.20  # CO₂ released per kg input during composting

# Fertilizer output (N-P-K per kg of composted solid waste)
FERT_N_KG_PER_KG_WASTE = 0.090
FERT_P_KG_PER_KG_WASTE = 0.008
FERT_K_KG_PER_KG_WASTE = 0.021

# ---------------------------------------------------------------------------
# Anaerobic digestion (crop residue → biogas)
# ---------------------------------------------------------------------------

VOLATILE_SOLIDS_FRACTION = 0.80       # VS/TS ratio for crop residue
BIOGAS_M3_PER_KG_VS = 0.40           # biogas yield (m³ per kg volatile solids)
BIOGAS_CH4_FRACTION = 0.60            # methane content of biogas
BIOGAS_CO2_FRACTION = 0.40            # CO₂ content of biogas
CH4_DENSITY_KG_M3 = 0.657            # methane density at ~101 kPa, 25°C
CO2_DENSITY_KG_M3 = 1.842            # CO₂ density at ~101 kPa, 25°C
DIGESTER_POWER_KWH_PER_KG = 0.10     # heating + mixing per kg input
DIGESTATE_FRACTION = 0.30             # fraction remaining as digestate (fertilizer)

# ---------------------------------------------------------------------------
# System limits
# ---------------------------------------------------------------------------

MAX_DAILY_URINE_L = 500.0            # max processing capacity
MAX_DAILY_SOLIDS_KG = 50.0           # max composting intake per sol
MAX_DAILY_RESIDUE_KG = 200.0         # max digester intake per sol


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WasteInput:
    """Daily waste inputs from the colony."""

    urine_l: float = 0.0
    solid_waste_kg: float = 0.0
    greywater_l: float = 0.0
    crop_residue_kg: float = 0.0

    def __post_init__(self) -> None:
        self.urine_l = max(0.0, self.urine_l)
        self.solid_waste_kg = max(0.0, self.solid_waste_kg)
        self.greywater_l = max(0.0, self.greywater_l)
        self.crop_residue_kg = max(0.0, self.crop_residue_kg)

    @classmethod
    def from_population(cls, population: int, crop_harvest_kg: float = 0.0) -> "WasteInput":
        """Generate waste inputs from colony population and harvest."""
        return cls(
            urine_l=population * URINE_L_PER_PERSON_SOL,
            solid_waste_kg=population * SOLID_WASTE_KG_PER_PERSON_SOL,
            greywater_l=population * HYGIENE_WATER_L_PER_PERSON_SOL,
            crop_residue_kg=crop_harvest_kg * CROP_RESIDUE_RATIO,
        )


@dataclass
class RecyclerOutput:
    """Cumulative recovered resources."""

    water_recovered_l: float = 0.0
    fertilizer_kg: float = 0.0
    ch4_kg: float = 0.0
    co2_kg: float = 0.0
    brine_salts_kg: float = 0.0


@dataclass
class RecyclerState:
    """Equipment state for the waste processing plant."""

    upa_health: float = 1.0            # urine processor health
    composter_health: float = 1.0      # composting system health
    digester_health: float = 1.0       # anaerobic digester health

    compost_queue_kg: float = 0.0      # waste in active composting
    compost_age_sols: int = 0          # sols since last batch started

    UPA_DEGRADE_PER_SOL: float = field(default=0.0003, repr=False)
    COMPOSTER_DEGRADE_PER_SOL: float = field(default=0.0001, repr=False)
    DIGESTER_DEGRADE_PER_SOL: float = field(default=0.0002, repr=False)

    def __post_init__(self) -> None:
        self.upa_health = max(0.0, min(1.0, self.upa_health))
        self.composter_health = max(0.0, min(1.0, self.composter_health))
        self.digester_health = max(0.0, min(1.0, self.digester_health))
        self.compost_queue_kg = max(0.0, self.compost_queue_kg)
        self.compost_age_sols = max(0, self.compost_age_sols)


# ---------------------------------------------------------------------------
# Urine processing
# ---------------------------------------------------------------------------

def process_urine(
    urine_l: float,
    power_kwh: float,
    upa_health: float,
) -> dict[str, float]:
    """Process urine through vapor compression distillation.

    Returns dict: water_l, brine_l, power_consumed, processed_l.
    """
    zero = {"water_l": 0.0, "brine_l": 0.0, "power_consumed": 0.0, "processed_l": 0.0}
    if urine_l <= 0.0 or power_kwh <= 0.0 or upa_health <= 0.0:
        return zero

    capacity = min(urine_l, MAX_DAILY_URINE_L)
    energy_per_l = UPA_POWER_KWH_PER_L / max(0.01, upa_health)
    max_by_power = power_kwh / energy_per_l if energy_per_l > 0 else 0.0
    processed = min(capacity, max_by_power)

    water = processed * UPA_WATER_RECOVERY
    brine = processed * BRINE_FRACTION
    power_used = processed * energy_per_l

    return {"water_l": water, "brine_l": brine, "power_consumed": power_used, "processed_l": processed}


# ---------------------------------------------------------------------------
# Greywater processing
# ---------------------------------------------------------------------------

def process_greywater(
    greywater_l: float,
    power_kwh: float,
) -> dict[str, float]:
    """Filter and sterilize greywater.

    Returns dict: water_l, power_consumed.
    """
    if greywater_l <= 0.0 or power_kwh <= 0.0:
        return {"water_l": 0.0, "power_consumed": 0.0}

    max_by_power = power_kwh / GREY_POWER_KWH_PER_L if GREY_POWER_KWH_PER_L > 0 else 0.0
    processed = min(greywater_l, max_by_power)
    water = processed * GREY_WATER_RECOVERY

    return {"water_l": water, "power_consumed": processed * GREY_POWER_KWH_PER_L}


# ---------------------------------------------------------------------------
# Brine recovery
# ---------------------------------------------------------------------------

def process_brine(
    brine_l: float,
    power_kwh: float,
) -> dict[str, float]:
    """Electrolyse brine to recover additional water.

    Returns dict: water_l, salts_kg, power_consumed.
    """
    if brine_l <= 0.0 or power_kwh <= 0.0:
        return {"water_l": 0.0, "salts_kg": 0.0, "power_consumed": 0.0}

    max_by_power = power_kwh / BRINE_ELECTROLYSIS_KWH_PER_L
    processed = min(brine_l, max_by_power)
    water = processed * BRINE_ELECTROLYSIS_RECOVERY
    salts = processed * (1.0 - BRINE_ELECTROLYSIS_RECOVERY)

    return {
        "water_l": water,
        "salts_kg": salts,
        "power_consumed": processed * BRINE_ELECTROLYSIS_KWH_PER_L,
    }


# ---------------------------------------------------------------------------
# Composting
# ---------------------------------------------------------------------------

def compost_tick(
    new_waste_kg: float,
    state: RecyclerState,
    power_kwh: float,
) -> dict[str, float]:
    """Advance composting by one sol.

    New waste is added to the active queue. Once the queue has aged
    COMPOST_CYCLE_SOLS, the batch yields fertilizer and releases
    water + CO₂.

    Returns dict: fertilizer_kg, water_l, co2_kg, power_consumed.
    """
    result = {"fertilizer_kg": 0.0, "water_l": 0.0, "co2_kg": 0.0, "power_consumed": 0.0}

    if state.composter_health <= 0.0:
        return result

    # Add new waste (capacity-limited)
    intake = min(new_waste_kg, MAX_DAILY_SOLIDS_KG)
    energy_needed = intake * COMPOST_POWER_KWH_PER_KG / max(0.01, state.composter_health)
    if energy_needed > power_kwh and power_kwh > 0:
        intake = power_kwh / (COMPOST_POWER_KWH_PER_KG / max(0.01, state.composter_health))
        energy_needed = power_kwh

    if intake > 0 and power_kwh > 0:
        state.compost_queue_kg += intake
        result["power_consumed"] = energy_needed
        if state.compost_age_sols == 0:
            state.compost_age_sols = 1  # start the clock

    # Age the batch
    if state.compost_queue_kg > 0:
        state.compost_age_sols += 1

        # Continuous CO₂ and water release during composting
        daily_co2 = state.compost_queue_kg * COMPOST_CO2_RELEASE_KG_PER_KG / COMPOST_CYCLE_SOLS
        daily_water = state.compost_queue_kg * COMPOST_WATER_RELEASE / COMPOST_CYCLE_SOLS
        result["co2_kg"] = daily_co2
        result["water_l"] = daily_water

        # Batch complete?
        if state.compost_age_sols >= COMPOST_CYCLE_SOLS:
            fert = state.compost_queue_kg * (1.0 - COMPOST_MASS_REDUCTION)
            result["fertilizer_kg"] = fert
            state.compost_queue_kg = 0.0
            state.compost_age_sols = 0

    return result


# ---------------------------------------------------------------------------
# Anaerobic digestion
# ---------------------------------------------------------------------------

def digest_residue(
    residue_kg: float,
    power_kwh: float,
    digester_health: float,
) -> dict[str, float]:
    """Digest crop residue into biogas + digestate.

    Returns dict: ch4_kg, co2_kg, digestate_kg, power_consumed.
    """
    zero = {"ch4_kg": 0.0, "co2_kg": 0.0, "digestate_kg": 0.0, "power_consumed": 0.0}
    if residue_kg <= 0.0 or power_kwh <= 0.0 or digester_health <= 0.0:
        return zero

    capacity = min(residue_kg, MAX_DAILY_RESIDUE_KG)
    energy_per_kg = DIGESTER_POWER_KWH_PER_KG / max(0.01, digester_health)
    max_by_power = power_kwh / energy_per_kg
    processed = min(capacity, max_by_power)

    vs = processed * VOLATILE_SOLIDS_FRACTION
    biogas_m3 = vs * BIOGAS_M3_PER_KG_VS
    ch4_m3 = biogas_m3 * BIOGAS_CH4_FRACTION
    co2_m3 = biogas_m3 * BIOGAS_CO2_FRACTION
    ch4_kg = ch4_m3 * CH4_DENSITY_KG_M3
    co2_kg = co2_m3 * CO2_DENSITY_KG_M3
    digestate = processed * DIGESTATE_FRACTION

    return {
        "ch4_kg": ch4_kg,
        "co2_kg": co2_kg,
        "digestate_kg": digestate,
        "power_consumed": processed * energy_per_kg,
    }


# ---------------------------------------------------------------------------
# Daily tick — full waste processing pipeline
# ---------------------------------------------------------------------------

def tick_waste(
    waste: WasteInput,
    output: RecyclerOutput,
    state: RecyclerState,
    power_budget_kwh: float,
) -> dict[str, float | bool]:
    """Advance waste recycling by one sol.

    Power allocation: 40% urine/brine, 20% greywater, 20% composting, 20% digestion.

    Mutates output and state in place.

    Returns summary of this sol's recovery.
    """
    summary: dict[str, float | bool] = {
        "sol_water_recovered_l": 0.0,
        "sol_fertilizer_kg": 0.0,
        "sol_ch4_kg": 0.0,
        "sol_co2_kg": 0.0,
        "sol_power_consumed": 0.0,
    }

    if power_budget_kwh <= 0.0:
        return summary

    remaining = power_budget_kwh

    # Phase 1: Urine processing (40% of power)
    urine_power = remaining * 0.40
    urine_result = process_urine(waste.urine_l, urine_power, state.upa_health)
    water_total = urine_result["water_l"]
    remaining -= urine_result["power_consumed"]
    summary["sol_power_consumed"] = float(summary["sol_power_consumed"]) + urine_result["power_consumed"]

    # Phase 1b: Brine recovery from urine processing
    brine_power = min(remaining * 0.15, remaining)
    brine_result = process_brine(urine_result["brine_l"], brine_power)
    water_total += brine_result["water_l"]
    output.brine_salts_kg += brine_result["salts_kg"]
    remaining -= brine_result["power_consumed"]
    summary["sol_power_consumed"] = float(summary["sol_power_consumed"]) + brine_result["power_consumed"]

    # Phase 2: Greywater (20% of original budget)
    grey_power = min(power_budget_kwh * 0.20, remaining)
    grey_result = process_greywater(waste.greywater_l, grey_power)
    water_total += grey_result["water_l"]
    remaining -= grey_result["power_consumed"]
    summary["sol_power_consumed"] = float(summary["sol_power_consumed"]) + grey_result["power_consumed"]

    # Phase 3: Composting (20% of original budget)
    compost_power = min(power_budget_kwh * 0.20, remaining)
    compost_result = compost_tick(waste.solid_waste_kg, state, compost_power)
    water_total += compost_result["water_l"]
    output.fertilizer_kg += compost_result["fertilizer_kg"]
    output.co2_kg += compost_result["co2_kg"]
    remaining -= compost_result["power_consumed"]
    summary["sol_power_consumed"] = float(summary["sol_power_consumed"]) + compost_result["power_consumed"]
    summary["sol_fertilizer_kg"] = compost_result["fertilizer_kg"]
    summary["sol_co2_kg"] = float(summary.get("sol_co2_kg", 0.0)) + compost_result["co2_kg"]

    # Phase 4: Anaerobic digestion (remaining power)
    digest_power = max(0.0, remaining)
    digest_result = digest_residue(waste.crop_residue_kg, digest_power, state.digester_health)
    output.ch4_kg += digest_result["ch4_kg"]
    output.co2_kg += digest_result["co2_kg"]
    output.fertilizer_kg += digest_result["digestate_kg"]
    summary["sol_ch4_kg"] = digest_result["ch4_kg"]
    summary["sol_co2_kg"] = float(summary["sol_co2_kg"]) + digest_result["co2_kg"]
    summary["sol_power_consumed"] = float(summary["sol_power_consumed"]) + digest_result["power_consumed"]

    # Accumulate water
    output.water_recovered_l += water_total
    summary["sol_water_recovered_l"] = water_total

    # Equipment degradation
    state.upa_health = max(0.0, state.upa_health - state.UPA_DEGRADE_PER_SOL)
    state.composter_health = max(0.0, state.composter_health - state.COMPOSTER_DEGRADE_PER_SOL)
    state.digester_health = max(0.0, state.digester_health - state.DIGESTER_DEGRADE_PER_SOL)

    return summary


# ---------------------------------------------------------------------------
# Utility / planning helpers
# ---------------------------------------------------------------------------

def daily_waste_volume(population: int) -> dict[str, float]:
    """Estimate daily waste generation for a colony."""
    return {
        "urine_l": population * URINE_L_PER_PERSON_SOL,
        "solid_waste_kg": population * SOLID_WASTE_KG_PER_PERSON_SOL,
        "greywater_l": population * HYGIENE_WATER_L_PER_PERSON_SOL,
    }


def water_recovery_rate(population: int) -> float:
    """Theoretical maximum water recovery fraction for given population.

    Combines urine, brine, and greywater recovery.
    """
    urine = population * URINE_L_PER_PERSON_SOL
    grey = population * HYGIENE_WATER_L_PER_PERSON_SOL
    total_input = urine + grey

    if total_input == 0:
        return 0.0

    urine_water = urine * UPA_WATER_RECOVERY
    brine_water = urine * BRINE_FRACTION * BRINE_ELECTROLYSIS_RECOVERY
    grey_water = grey * GREY_WATER_RECOVERY

    return (urine_water + brine_water + grey_water) / total_input


def fertilizer_npk(waste_kg: float) -> dict[str, float]:
    """Estimate N-P-K output from composting a given mass of waste."""
    return {
        "nitrogen_kg": waste_kg * FERT_N_KG_PER_KG_WASTE,
        "phosphorus_kg": waste_kg * FERT_P_KG_PER_KG_WASTE,
        "potassium_kg": waste_kg * FERT_K_KG_PER_KG_WASTE,
    }
