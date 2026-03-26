"""soil_amendment.py - Mars Regolith-to-Arable-Soil Conversion.

The colony grows food in greenhouses (greenhouse.py) and gardens
(garden.py), but Martian regolith is toxic.  Perchlorates (ClO4-)
at 0.5-1% by mass poison humans at chronic exposure levels as low
as 0.01 mg/kg/day.  The soil has zero organic matter, negligible
bioavailable nitrogen, and poor water retention.

This module converts raw regolith into farmable substrate through
a 4-stage pipeline: perchlorate leaching, composting, nitrogen
enrichment, and pH buffering.  Each stage has real chemistry and
real kinetics.  The output is soil that greenhouses can use.

Physics modelled
----------------
* **Perchlorate leaching** - water wash dissolves ClO4- from
  regolith.  Solubility is high (~200 g/L at 25C), so a 3:1
  water-to-soil ratio removes >95% in one wash.  Rate limited
  by diffusion through particle pores.

* **Composting** - thermophilic decomposition of organic waste
  (crew waste, crop residue) at 50-70C.  First-order decay:
  dM/dt = -k * M, where k depends on temperature, moisture, and
  C:N ratio.  Optimal C:N = 25-30:1.

* **Nitrogen fixation** - biological: Azotobacter at ~8 mg N/kg/sol.
  Chemical backup: Haber-Bosch micro-reactor using N2 from
  nitrogen_generator.py and H2 from water_electrolysis.py.

* **pH buffering** - Mars regolith pH 7.7-8.4 (Phoenix data).
  Sulfur oxidation by Thiobacillus: S + 3/2 O2 + H2O -> H2SO4.

* **Water holding capacity** - pure regolith holds ~5% water by
  mass.  Adding 5% organic matter raises this to ~25%.

Conservation laws: mass balance (in >= out), perchlorate decreases
monotonically during leaching, nitrogen >= 0, pH bounded [4, 10].

One tick = one sol.  Mass in kg, volume in liters, temperature in K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# -- Physical constants -------------------------------------------------------

MARS_AMBIENT_TEMP_K = 210.0
SECONDS_PER_SOL = 88_775.0
HOURS_PER_SOL = 24.66

PERCHLORATE_INITIAL_PPM = 6000.0
PERCHLORATE_SAFE_PPM = 100.0
WASH_EFFICIENCY_PER_CYCLE = 0.92
WATER_PER_KG_SOIL_L = 3.0

COMPOST_DECAY_RATE_PER_DAY = 0.05
OPTIMAL_COMPOST_TEMP_K = 333.0
COMPOST_TEMP_SENSITIVITY = 0.08
COMPOST_HEAT_MJ_PER_KG = 17.0
OPTIMAL_CN_RATIO = 27.5

BIO_N_FIXATION_MG_PER_KG_PER_SOL = 8.0
CHEMICAL_N_FIXATION_KG_PER_KWH = 0.001
TARGET_N_MG_PER_KG = 200.0
MAX_N_MG_PER_KG = 500.0

MARS_REGOLITH_PH = 8.0
TARGET_PH_LOW = 6.0
TARGET_PH_HIGH = 7.0
SULFUR_PH_DROP_PER_KG_PER_SOL = 0.02
MIN_PH = 4.0
MAX_PH = 10.0

BASE_WATER_RETENTION_FRAC = 0.05
MAX_WATER_RETENTION_FRAC = 0.35
ORGANIC_MATTER_RETENTION_COEFF = 4.0

MIN_ORGANIC_MATTER_FRAC = 0.03
GOOD_ORGANIC_MATTER_FRAC = 0.08


# -- Pure physics functions ---------------------------------------------------

def perchlorate_after_wash(initial_ppm: float, num_cycles: int) -> float:
    """Perchlorate concentration (ppm) after N wash cycles."""
    if num_cycles <= 0 or initial_ppm <= 0.0:
        return max(0.0, initial_ppm)
    remaining = (1.0 - WASH_EFFICIENCY_PER_CYCLE) ** num_cycles
    return round(max(0.0, initial_ppm * remaining), 4)


def wash_water_needed_l(soil_mass_kg: float, num_cycles: int) -> float:
    """Total water (liters) needed for perchlorate washing."""
    if soil_mass_kg <= 0.0 or num_cycles <= 0:
        return 0.0
    return round(soil_mass_kg * WATER_PER_KG_SOIL_L * num_cycles, 4)


def washes_to_safe(initial_ppm: float = PERCHLORATE_INITIAL_PPM) -> int:
    """Minimum wash cycles to reach safe perchlorate level."""
    if initial_ppm <= PERCHLORATE_SAFE_PPM:
        return 0
    cycles = 0
    ppm = initial_ppm
    while ppm > PERCHLORATE_SAFE_PPM and cycles < 100:
        ppm *= (1.0 - WASH_EFFICIENCY_PER_CYCLE)
        cycles += 1
    return cycles


def compost_decay_fraction(temp_k: float, moisture_fraction: float,
                           cn_ratio: float) -> float:
    """Fraction of organic matter decomposed per sol."""
    if temp_k <= 273.0 or moisture_fraction <= 0.0:
        return 0.0
    temp_diff = abs(temp_k - OPTIMAL_COMPOST_TEMP_K)
    temp_factor = math.exp(-COMPOST_TEMP_SENSITIVITY * temp_diff)
    if moisture_fraction < 0.2:
        moisture_factor = moisture_fraction / 0.2
    elif moisture_fraction > 0.7:
        moisture_factor = max(0.0, 1.0 - (moisture_fraction - 0.7) / 0.3)
    else:
        moisture_factor = 1.0
    cn_diff = abs(cn_ratio - OPTIMAL_CN_RATIO)
    cn_factor = max(0.1, 1.0 - cn_diff / (OPTIMAL_CN_RATIO * 2))
    rate = COMPOST_DECAY_RATE_PER_DAY * temp_factor * moisture_factor * cn_factor
    return round(max(0.0, min(1.0, rate)), 6)


def compost_heat_kw(organic_mass_kg: float, decay_fraction: float) -> float:
    """Heat generated (kW) by composting in one sol."""
    if organic_mass_kg <= 0.0 or decay_fraction <= 0.0:
        return 0.0
    energy_kj = organic_mass_kg * decay_fraction * COMPOST_HEAT_MJ_PER_KG * 1000.0
    return round(max(0.0, energy_kj / SECONDS_PER_SOL), 6)


def nitrogen_per_sol(soil_mass_kg: float, bio_active: bool,
                     chemical_power_kw: float = 0.0) -> float:
    """Nitrogen added to soil (mg) per sol."""
    bio_n = BIO_N_FIXATION_MG_PER_KG_PER_SOL * soil_mass_kg if bio_active and soil_mass_kg > 0 else 0.0
    chem_n = chemical_power_kw * HOURS_PER_SOL * CHEMICAL_N_FIXATION_KG_PER_KWH * 1e6 if chemical_power_kw > 0 else 0.0
    return round(bio_n + chem_n, 4)


def ph_after_amendment(current_ph: float, sulfur_kg: float,
                       soil_mass_kg: float) -> float:
    """Soil pH after sulfur amendment for one sol."""
    if soil_mass_kg <= 0.0 or sulfur_kg <= 0.0:
        return current_ph
    ph_drop = SULFUR_PH_DROP_PER_KG_PER_SOL * (sulfur_kg / soil_mass_kg) * soil_mass_kg
    return round(max(MIN_PH, min(MAX_PH, current_ph - ph_drop)), 4)


def water_holding_capacity(organic_matter_fraction: float) -> float:
    """Water holding capacity as fraction of soil mass."""
    om = max(0.0, min(1.0, organic_matter_fraction))
    whc = BASE_WATER_RETENTION_FRAC + ORGANIC_MATTER_RETENTION_COEFF * om
    return round(min(MAX_WATER_RETENTION_FRAC, max(0.0, whc)), 6)


def soil_fertility_score(perchlorate_ppm: float, nitrogen_mg_kg: float,
                         organic_matter_frac: float, ph: float) -> float:
    """Composite fertility score 0.0 (dead) to 1.0 (excellent)."""
    if perchlorate_ppm <= PERCHLORATE_SAFE_PPM:
        perc_score = 1.0
    elif perchlorate_ppm >= PERCHLORATE_INITIAL_PPM:
        perc_score = 0.0
    else:
        perc_score = 1.0 - (perchlorate_ppm - PERCHLORATE_SAFE_PPM) / (
            PERCHLORATE_INITIAL_PPM - PERCHLORATE_SAFE_PPM)

    n_score = min(1.0, nitrogen_mg_kg / TARGET_N_MG_PER_KG)

    if organic_matter_frac >= GOOD_ORGANIC_MATTER_FRAC:
        om_score = 1.0
    elif organic_matter_frac >= MIN_ORGANIC_MATTER_FRAC:
        om_score = 0.5 + 0.5 * ((organic_matter_frac - MIN_ORGANIC_MATTER_FRAC) /
                                  (GOOD_ORGANIC_MATTER_FRAC - MIN_ORGANIC_MATTER_FRAC))
    else:
        om_score = 0.5 * organic_matter_frac / max(0.001, MIN_ORGANIC_MATTER_FRAC)

    if TARGET_PH_LOW <= ph <= TARGET_PH_HIGH:
        ph_score = 1.0
    elif ph < TARGET_PH_LOW:
        ph_score = max(0.0, 1.0 - (TARGET_PH_LOW - ph) / 2.0)
    else:
        ph_score = max(0.0, 1.0 - (ph - TARGET_PH_HIGH) / 2.0)

    score = 0.35 * perc_score + 0.25 * n_score + 0.25 * om_score + 0.15 * ph_score
    return round(max(0.0, min(1.0, score)), 4)


# -- State dataclass ----------------------------------------------------------

@dataclass
class SoilAmendmentBed:
    """State of one soil amendment processing bed."""
    soil_mass_kg: float = 0.0
    perchlorate_ppm: float = PERCHLORATE_INITIAL_PPM
    wash_cycles_completed: int = 0
    organic_matter_kg: float = 0.0
    organic_matter_fraction: float = 0.0
    compost_temp_k: float = MARS_AMBIENT_TEMP_K
    moisture_fraction: float = 0.0
    nitrogen_mg_per_kg: float = 0.0
    bio_fixation_active: bool = False
    ph: float = MARS_REGOLITH_PH
    water_holding_cap: float = BASE_WATER_RETENTION_FRAC
    sols_processed: int = 0
    total_water_used_l: float = 0.0
    total_compost_added_kg: float = 0.0
    fertility_score: float = 0.0
    arable_soil_produced_kg: float = 0.0


# -- Tick function (one sol) --------------------------------------------------

def tick(bed: SoilAmendmentBed,
         regolith_added_kg: float = 0.0,
         compost_added_kg: float = 0.0,
         water_available_l: float = 0.0,
         sulfur_added_kg: float = 0.0,
         chemical_n_power_kw: float = 0.0,
         wash_this_sol: bool = False,
         heated: bool = False) -> dict:
    """Advance the soil amendment bed by one sol."""
    bed.sols_processed += 1

    if regolith_added_kg > 0.0:
        if bed.soil_mass_kg > 0:
            total = bed.soil_mass_kg + regolith_added_kg
            bed.perchlorate_ppm = (bed.perchlorate_ppm * bed.soil_mass_kg +
                                   PERCHLORATE_INITIAL_PPM * regolith_added_kg) / total
            bed.nitrogen_mg_per_kg = bed.nitrogen_mg_per_kg * bed.soil_mass_kg / total
            bed.soil_mass_kg = total
        else:
            bed.soil_mass_kg = regolith_added_kg
            bed.perchlorate_ppm = PERCHLORATE_INITIAL_PPM
            bed.nitrogen_mg_per_kg = 0.0

    water_used = 0.0
    result = {"fertility_score": 0.0, "perchlorate_ppm": bed.perchlorate_ppm,
              "nitrogen_mg_kg": bed.nitrogen_mg_per_kg, "ph": bed.ph,
              "organic_matter_frac": bed.organic_matter_fraction,
              "water_holding_capacity": bed.water_holding_cap, "water_used_l": 0.0,
              "compost_heat_kw": 0.0, "soil_mass_kg": bed.soil_mass_kg,
              "arable_ready": False}

    if bed.soil_mass_kg <= 0.0:
        return result

    # Stage 1: Perchlorate washing
    if wash_this_sol and bed.perchlorate_ppm > PERCHLORATE_SAFE_PPM:
        water_needed = wash_water_needed_l(bed.soil_mass_kg, 1)
        if water_available_l >= water_needed:
            bed.perchlorate_ppm = perchlorate_after_wash(bed.perchlorate_ppm, 1)
            bed.wash_cycles_completed += 1
            water_used += water_needed
            water_available_l -= water_needed

    # Stage 2: Composting
    if compost_added_kg > 0.0:
        bed.organic_matter_kg += compost_added_kg
        bed.total_compost_added_kg += compost_added_kg

    if bed.organic_matter_kg > 0.0:
        if heated:
            bed.compost_temp_k = min(OPTIMAL_COMPOST_TEMP_K, bed.compost_temp_k + 20.0)
        else:
            if bed.organic_matter_kg > 5.0:
                bed.compost_temp_k = min(OPTIMAL_COMPOST_TEMP_K - 15.0, bed.compost_temp_k + 5.0)
            else:
                bed.compost_temp_k = max(MARS_AMBIENT_TEMP_K, bed.compost_temp_k - 2.0)

        moisture_water = max(0.0, min(water_available_l,
                                       bed.organic_matter_kg * 0.5 - bed.moisture_fraction * bed.organic_matter_kg))
        water_used += moisture_water
        water_available_l -= moisture_water
        if bed.organic_matter_kg > 0:
            bed.moisture_fraction = min(0.7,
                (bed.moisture_fraction * bed.organic_matter_kg + moisture_water) / max(0.01, bed.organic_matter_kg))

        decay_frac = compost_decay_fraction(bed.compost_temp_k, bed.moisture_fraction, 30.0)
        decayed_kg = bed.organic_matter_kg * decay_frac
        result["compost_heat_kw"] = compost_heat_kw(bed.organic_matter_kg, decay_frac)
        bed.organic_matter_kg -= decayed_kg
        if bed.soil_mass_kg > 0:
            bed.organic_matter_fraction = min(0.15, bed.organic_matter_fraction + decayed_kg / bed.soil_mass_kg)

    # Stage 3: Nitrogen enrichment
    bed.bio_fixation_active = bed.perchlorate_ppm <= PERCHLORATE_SAFE_PPM * 5
    n_added = nitrogen_per_sol(bed.soil_mass_kg, bed.bio_fixation_active, chemical_n_power_kw)
    if bed.soil_mass_kg > 0:
        bed.nitrogen_mg_per_kg = min(MAX_N_MG_PER_KG, bed.nitrogen_mg_per_kg + n_added / bed.soil_mass_kg)

    # Stage 4: pH adjustment
    if sulfur_added_kg > 0.0:
        bed.ph = ph_after_amendment(bed.ph, sulfur_added_kg, bed.soil_mass_kg)

    # Update derived properties
    bed.water_holding_cap = water_holding_capacity(bed.organic_matter_fraction)
    bed.total_water_used_l += water_used
    bed.fertility_score = soil_fertility_score(bed.perchlorate_ppm, bed.nitrogen_mg_per_kg,
                                               bed.organic_matter_fraction, bed.ph)

    arable = (bed.perchlorate_ppm <= PERCHLORATE_SAFE_PPM and
              bed.nitrogen_mg_per_kg >= TARGET_N_MG_PER_KG * 0.5 and
              bed.organic_matter_fraction >= MIN_ORGANIC_MATTER_FRAC and
              TARGET_PH_LOW <= bed.ph <= TARGET_PH_HIGH + 0.5)
    if arable:
        bed.arable_soil_produced_kg = bed.soil_mass_kg

    result.update({"fertility_score": bed.fertility_score,
                   "perchlorate_ppm": round(bed.perchlorate_ppm, 4),
                   "nitrogen_mg_kg": round(bed.nitrogen_mg_per_kg, 4),
                   "ph": round(bed.ph, 4),
                   "organic_matter_frac": round(bed.organic_matter_fraction, 6),
                   "water_holding_capacity": bed.water_holding_cap,
                   "water_used_l": round(water_used, 4),
                   "soil_mass_kg": round(bed.soil_mass_kg, 4),
                   "arable_ready": arable})
    return result
