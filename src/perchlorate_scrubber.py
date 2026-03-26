"""perchlorate_scrubber.py -- Mars Regolith Perchlorate Remediation System

Mars soil is toxic.  Phoenix lander (2008) found 0.5-1.0% perchlorate
(ClO4-) by mass -- a thyroid-disrupting oxidiser lethal to humans at
chronic exposure above 0.1%.  Before the colony grows a single potato,
every gram of greenhouse soil must be scrubbed clean.

Chemistry
---------
* Perchlorate reduction: ClO4- -> ClO3- -> ClO2- -> Cl- + 2O2
  Net: ClO4- + 8e- + 8H+ -> Cl- + 4H2O  (biological/electro)
  Or:  ClO4- -> Cl- + 2O2  (thermal, >400C)
* Three remediation pathways:
  1. **Thermal decomposition** (>400C): Fast, high energy, liberates O2.
     Works on dry regolith.  Energy: ~3.5 kWh/kg regolith processed.
  2. **Bioremediation**: Perchlorate-reducing bacteria (Dechloromonas)
     in anaerobic bioreactor.  Slow but low-energy (~0.3 kWh/kg).
     Requires electron donor (acetate/H2) and 20-37C.
  3. **Water wash + ion exchange**: Rinse regolith, run brine through
     IX resin.  Moderate energy (~1.0 kWh/kg).  Resin needs regeneration.

Physics
-------
* Perchlorate molar mass: 99.45 g/mol (ClO4-)
* Chloride molar mass: 35.45 g/mol (Cl-)
* O2 liberated per kg ClO4- (thermal): 0.644 kg  (2*32/99.45)
* Regolith perchlorate: 0.4-1.0 wt% (site-dependent)
* Safe threshold for agriculture: <0.01 wt% (100 ppm)
* Safe threshold for drinking water: <15 ug/L (EPA)

Reference: Hecht et al. 2009 (Phoenix WCL), Stern et al. 2017
(Curiosity SAM), Davila et al. 2013 (bioremediation feasibility).

One tick = one sol.  Mass in kg, energy in kWh, concentration in wt%.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# -- Physical constants -------------------------------------------------------

PERCHLORATE_MOLAR_MASS = 99.45      # g/mol ClO4-
CHLORIDE_MOLAR_MASS = 35.45         # g/mol Cl-
O2_MOLAR_MASS = 32.0                # g/mol O2

# Stoichiometry: ClO4- -> Cl- + 2O2
O2_PER_PERCHLORATE_KG = (2 * O2_MOLAR_MASS) / PERCHLORATE_MOLAR_MASS  # ~0.6435
CL_PER_PERCHLORATE_KG = CHLORIDE_MOLAR_MASS / PERCHLORATE_MOLAR_MASS   # ~0.3565

REGOLITH_DENSITY_KG_M3 = 1500.0    # bulk density

# Perchlorate levels on Mars (mass fraction)
PERCHLORATE_MARS_LOW = 0.004        # 0.4 wt% (benign sites)
PERCHLORATE_MARS_HIGH = 0.010       # 1.0 wt% (hot spots)
PERCHLORATE_MARS_MEAN = 0.007       # 0.7 wt% (Phoenix average)

# Safety thresholds
SAFE_AGRICULTURE_LIMIT = 0.0001     # 0.01 wt% (100 ppm)
SAFE_DRINKING_WATER_UG_L = 15.0     # EPA MCL

# -- Thermal decomposition parameters ----------------------------------------

THERMAL_MIN_TEMP_C = 400.0          # onset of decomposition
THERMAL_OPTIMAL_TEMP_C = 600.0      # rapid, complete decomposition
THERMAL_MAX_TEMP_C = 800.0          # equipment limit
THERMAL_KWH_PER_KG = 3.5            # energy per kg regolith
THERMAL_REMOVAL_EFFICIENCY = 0.995  # 99.5% perchlorate destroyed
THERMAL_THROUGHPUT_KG_SOL = 200.0   # max capacity per sol (single kiln)

# -- Bioremediation parameters ------------------------------------------------

BIO_MIN_TEMP_C = 10.0               # bacteria barely active below this
BIO_OPTIMAL_TEMP_C = 30.0           # optimal mesophilic growth
BIO_MAX_TEMP_C = 45.0               # lethal above this
BIO_KWH_PER_KG = 0.3                # mostly heating + mixing
BIO_BASE_RATE_KG_SOL = 50.0         # max bioreactor capacity per sol
BIO_CULTURE_GROWTH_PER_SOL = 0.02   # health recovery when conditions good
BIO_CULTURE_DECAY_PER_SOL = 0.05    # health decay when conditions bad
BIO_ELECTRON_DONOR_KG_PER_KG = 0.05 # acetate consumed per kg regolith

# -- Ion exchange parameters --------------------------------------------------

IX_KWH_PER_KG = 1.0                 # pumping + regeneration energy
IX_WATER_L_PER_KG = 3.0             # wash water per kg regolith
IX_THROUGHPUT_KG_SOL = 150.0        # max IX column capacity per sol
IX_REMOVAL_EFFICIENCY = 0.98        # 98% perchlorate captured
IX_RESIN_CYCLES = 500               # regeneration cycles before replacement
IX_RESIN_DEGRADE_PER_CYCLE = 0.001  # efficiency loss per cycle


# -- Data structures ----------------------------------------------------------

@dataclass
class ScrubberState:
    """State of the perchlorate remediation system."""
    sols_running: int = 0

    # Stockpiles (kg)
    contaminated_kg: float = 0.0        # regolith awaiting treatment
    clean_kg: float = 0.0               # scrubbed regolith (safe for agriculture)
    perchlorate_waste_kg: float = 0.0   # collected chloride salts
    o2_liberated_kg: float = 0.0        # oxygen recovered from perchlorate

    # Subsystem states
    kiln_temp_c: float = 20.0           # thermal kiln temperature
    bio_culture_health: float = 0.5     # bioreactor culture viability [0,1]
    bio_temp_c: float = 25.0            # bioreactor temperature
    ix_resin_cycles_used: int = 0       # ion exchange resin wear
    electron_donor_kg: float = 50.0     # acetate stock for bioreactor

    # Cumulative tracking
    total_processed_kg: float = 0.0
    total_energy_kwh: float = 0.0
    total_o2_recovered_kg: float = 0.0
    total_water_used_l: float = 0.0

    # Alert
    alert: str = "nominal"


@dataclass
class ScrubberResult:
    """Result of one sol of scrubbing."""
    thermal_processed_kg: float = 0.0
    bio_processed_kg: float = 0.0
    ix_processed_kg: float = 0.0
    total_processed_kg: float = 0.0

    perchlorate_removed_kg: float = 0.0
    o2_recovered_kg: float = 0.0
    chloride_produced_kg: float = 0.0

    energy_used_kwh: float = 0.0
    water_used_l: float = 0.0
    clean_kg_produced: float = 0.0

    bio_culture_health: float = 0.0
    ix_resin_remaining_cycles: int = 0
    alert: str = "nominal"


# -- Pure physics functions ---------------------------------------------------

def perchlorate_in_regolith_kg(regolith_kg: float,
                                concentration: float = PERCHLORATE_MARS_MEAN) -> float:
    """Mass of perchlorate in a batch of regolith (kg)."""
    return regolith_kg * max(0.0, min(1.0, concentration))


def o2_from_perchlorate_kg(perchlorate_kg: float) -> float:
    """O2 liberated from thermal decomposition of perchlorate (kg).

    ClO4- -> Cl- + 2O2
    """
    return max(0.0, perchlorate_kg) * O2_PER_PERCHLORATE_KG


def chloride_from_perchlorate_kg(perchlorate_kg: float) -> float:
    """Chloride salt produced from perchlorate reduction (kg)."""
    return max(0.0, perchlorate_kg) * CL_PER_PERCHLORATE_KG


def residual_perchlorate_wt(initial_concentration: float,
                             removal_efficiency: float) -> float:
    """Perchlorate concentration after treatment (wt fraction)."""
    return initial_concentration * (1.0 - max(0.0, min(1.0, removal_efficiency)))


def is_safe_for_agriculture(concentration: float) -> bool:
    """Check if perchlorate level is below agricultural safety limit."""
    return concentration <= SAFE_AGRICULTURE_LIMIT


def mass_balance_check(regolith_kg: float,
                        concentration: float,
                        clean_kg: float,
                        perchlorate_removed_kg: float) -> float:
    """Verify mass conservation.  Returns residual error (should be ~0)."""
    expected_clean = regolith_kg - perchlorate_removed_kg
    return abs(clean_kg - expected_clean)


# -- Thermal decomposition ---------------------------------------------------

def thermal_efficiency(kiln_temp_c: float) -> float:
    """Removal efficiency as function of kiln temperature.

    Below 400C: no decomposition.
    400-600C: linear ramp to full efficiency.
    Above 600C: capped at THERMAL_REMOVAL_EFFICIENCY.
    """
    if kiln_temp_c < THERMAL_MIN_TEMP_C:
        return 0.0
    if kiln_temp_c >= THERMAL_OPTIMAL_TEMP_C:
        return THERMAL_REMOVAL_EFFICIENCY
    frac = (kiln_temp_c - THERMAL_MIN_TEMP_C) / (THERMAL_OPTIMAL_TEMP_C - THERMAL_MIN_TEMP_C)
    return frac * THERMAL_REMOVAL_EFFICIENCY


def thermal_process_sol(contaminated_kg: float,
                         power_kwh: float,
                         kiln_temp_c: float,
                         concentration: float = PERCHLORATE_MARS_MEAN) -> dict:
    """Process regolith through thermal kiln for one sol.

    Returns dict with processed_kg, clean_kg, perchlorate_removed_kg,
    o2_recovered_kg, chloride_kg, energy_used_kwh.
    """
    if contaminated_kg <= 0 or power_kwh <= 0 or kiln_temp_c < THERMAL_MIN_TEMP_C:
        return {"processed_kg": 0.0, "clean_kg": 0.0,
                "perchlorate_removed_kg": 0.0, "o2_recovered_kg": 0.0,
                "chloride_kg": 0.0, "energy_used_kwh": 0.0}

    # Power-limited throughput
    max_by_power = power_kwh / THERMAL_KWH_PER_KG if THERMAL_KWH_PER_KG > 0 else 0.0
    processed = min(contaminated_kg, THERMAL_THROUGHPUT_KG_SOL, max_by_power)

    eff = thermal_efficiency(kiln_temp_c)
    perchlorate_in = perchlorate_in_regolith_kg(processed, concentration)
    perchlorate_removed = perchlorate_in * eff
    o2_out = o2_from_perchlorate_kg(perchlorate_removed)
    cl_out = chloride_from_perchlorate_kg(perchlorate_removed)
    clean = processed - perchlorate_removed
    energy = processed * THERMAL_KWH_PER_KG

    return {
        "processed_kg": processed,
        "clean_kg": max(0.0, clean),
        "perchlorate_removed_kg": perchlorate_removed,
        "o2_recovered_kg": o2_out,
        "chloride_kg": cl_out,
        "energy_used_kwh": energy,
    }


# -- Bioremediation ----------------------------------------------------------

def bio_culture_factor(health: float, temp_c: float) -> float:
    """Activity multiplier for bacterial culture [0, 1].

    Depends on culture health and temperature.  Peak at 30C.
    """
    health = max(0.0, min(1.0, health))
    if temp_c < BIO_MIN_TEMP_C or temp_c > BIO_MAX_TEMP_C:
        return 0.0
    # Bell curve around optimal temperature
    sigma = 10.0
    temp_factor = math.exp(-0.5 * ((temp_c - BIO_OPTIMAL_TEMP_C) / sigma) ** 2)
    return health * temp_factor


def bio_update_health(current_health: float, temp_c: float,
                       has_donor: bool) -> float:
    """Update bacterial culture health after one sol."""
    current_health = max(0.0, min(1.0, current_health))
    in_range = BIO_MIN_TEMP_C <= temp_c <= BIO_MAX_TEMP_C
    if in_range and has_donor:
        new = current_health + BIO_CULTURE_GROWTH_PER_SOL
    else:
        new = current_health - BIO_CULTURE_DECAY_PER_SOL
    return max(0.0, min(1.0, new))


def bio_process_sol(contaminated_kg: float,
                     power_kwh: float,
                     culture_health: float,
                     bio_temp_c: float,
                     electron_donor_kg: float,
                     concentration: float = PERCHLORATE_MARS_MEAN) -> dict:
    """Bioremediation for one sol.

    Returns dict with processed_kg, clean_kg, perchlorate_removed_kg,
    donor_consumed_kg, energy_used_kwh, culture_health.
    """
    if contaminated_kg <= 0 or power_kwh <= 0:
        return {"processed_kg": 0.0, "clean_kg": 0.0,
                "perchlorate_removed_kg": 0.0, "donor_consumed_kg": 0.0,
                "energy_used_kwh": 0.0, "culture_health": culture_health}

    activity = bio_culture_factor(culture_health, bio_temp_c)
    if activity <= 0:
        new_health = bio_update_health(culture_health, bio_temp_c,
                                        electron_donor_kg > 0)
        return {"processed_kg": 0.0, "clean_kg": 0.0,
                "perchlorate_removed_kg": 0.0, "donor_consumed_kg": 0.0,
                "energy_used_kwh": 0.0, "culture_health": new_health}

    # Throughput limited by: capacity * activity, power, donor stock
    max_by_power = power_kwh / BIO_KWH_PER_KG if BIO_KWH_PER_KG > 0 else 0.0
    max_by_donor = (electron_donor_kg / BIO_ELECTRON_DONOR_KG_PER_KG
                    if BIO_ELECTRON_DONOR_KG_PER_KG > 0 else 0.0)
    processed = min(contaminated_kg,
                    BIO_BASE_RATE_KG_SOL * activity,
                    max_by_power,
                    max_by_donor)
    processed = max(0.0, processed)

    perchlorate_in = perchlorate_in_regolith_kg(processed, concentration)
    # Bio removal ~90% effective (slower than thermal)
    bio_eff = 0.90 * activity
    perchlorate_removed = perchlorate_in * bio_eff
    clean = processed - perchlorate_removed
    donor_used = processed * BIO_ELECTRON_DONOR_KG_PER_KG
    energy = processed * BIO_KWH_PER_KG

    new_health = bio_update_health(culture_health, bio_temp_c,
                                    electron_donor_kg - donor_used > 0)

    return {
        "processed_kg": processed,
        "clean_kg": max(0.0, clean),
        "perchlorate_removed_kg": perchlorate_removed,
        "donor_consumed_kg": donor_used,
        "energy_used_kwh": energy,
        "culture_health": new_health,
    }


# -- Ion exchange wash --------------------------------------------------------

def ix_current_efficiency(cycles_used: int) -> float:
    """Ion exchange removal efficiency accounting for resin degradation."""
    degradation = cycles_used * IX_RESIN_DEGRADE_PER_CYCLE
    return max(0.0, IX_REMOVAL_EFFICIENCY - degradation)


def ix_process_sol(contaminated_kg: float,
                    power_kwh: float,
                    water_available_l: float,
                    cycles_used: int,
                    concentration: float = PERCHLORATE_MARS_MEAN) -> dict:
    """Ion-exchange wash for one sol.

    Returns dict with processed_kg, clean_kg, perchlorate_removed_kg,
    water_used_l, energy_used_kwh, cycles_used.
    """
    if contaminated_kg <= 0 or power_kwh <= 0 or water_available_l <= 0:
        return {"processed_kg": 0.0, "clean_kg": 0.0,
                "perchlorate_removed_kg": 0.0, "water_used_l": 0.0,
                "energy_used_kwh": 0.0, "cycles_used": cycles_used}

    eff = ix_current_efficiency(cycles_used)
    if eff <= 0:
        return {"processed_kg": 0.0, "clean_kg": 0.0,
                "perchlorate_removed_kg": 0.0, "water_used_l": 0.0,
                "energy_used_kwh": 0.0, "cycles_used": cycles_used}

    max_by_power = power_kwh / IX_KWH_PER_KG if IX_KWH_PER_KG > 0 else 0.0
    max_by_water = (water_available_l / IX_WATER_L_PER_KG
                    if IX_WATER_L_PER_KG > 0 else 0.0)
    processed = min(contaminated_kg, IX_THROUGHPUT_KG_SOL,
                    max_by_power, max_by_water)
    processed = max(0.0, processed)

    perchlorate_in = perchlorate_in_regolith_kg(processed, concentration)
    perchlorate_removed = perchlorate_in * eff
    clean = processed - perchlorate_removed
    water_used = processed * IX_WATER_L_PER_KG
    energy = processed * IX_KWH_PER_KG

    return {
        "processed_kg": processed,
        "clean_kg": max(0.0, clean),
        "perchlorate_removed_kg": perchlorate_removed,
        "water_used_l": water_used,
        "energy_used_kwh": energy,
        "cycles_used": cycles_used + (1 if processed > 0 else 0),
    }


# -- System tick --------------------------------------------------------------

def tick_scrubber(state: ScrubberState,
                   regolith_input_kg: float = 0.0,
                   concentration: float = PERCHLORATE_MARS_MEAN,
                   power_budget_kwh: float = 50.0,
                   water_available_l: float = 200.0,
                   allocation: dict | None = None) -> tuple[ScrubberState, ScrubberResult]:
    """Advance the perchlorate scrubber by one sol.

    Parameters
    ----------
    state : ScrubberState
        Current system state.
    regolith_input_kg : float
        Fresh contaminated regolith added this sol.
    concentration : float
        Perchlorate mass fraction in incoming regolith.
    power_budget_kwh : float
        Total energy budget for this sol.
    water_available_l : float
        Water available for IX washing.
    allocation : dict | None
        Power split: {"thermal": 0.5, "bio": 0.2, "ix": 0.3}.
        Defaults to equal thirds.

    Returns
    -------
    tuple[ScrubberState, ScrubberResult]
    """
    result = ScrubberResult()

    # Accept new contaminated regolith
    state.contaminated_kg += max(0.0, regolith_input_kg)

    if allocation is None:
        allocation = {"thermal": 0.40, "bio": 0.25, "ix": 0.35}

    total_alloc = sum(allocation.values())
    if total_alloc <= 0:
        total_alloc = 1.0

    pw_thermal = power_budget_kwh * allocation.get("thermal", 0.0) / total_alloc
    pw_bio = power_budget_kwh * allocation.get("bio", 0.0) / total_alloc
    pw_ix = power_budget_kwh * allocation.get("ix", 0.0) / total_alloc

    remaining = state.contaminated_kg

    # --- Thermal kiln ---
    t_res = thermal_process_sol(remaining, pw_thermal, state.kiln_temp_c,
                                 concentration)
    remaining -= t_res["processed_kg"]
    result.thermal_processed_kg = t_res["processed_kg"]

    # --- Bioreactor ---
    b_res = bio_process_sol(remaining, pw_bio, state.bio_culture_health,
                             state.bio_temp_c, state.electron_donor_kg,
                             concentration)
    remaining -= b_res["processed_kg"]
    state.bio_culture_health = b_res["culture_health"]
    state.electron_donor_kg = max(0.0,
                                   state.electron_donor_kg - b_res["donor_consumed_kg"])
    result.bio_processed_kg = b_res["processed_kg"]

    # --- Ion exchange ---
    i_res = ix_process_sol(remaining, pw_ix, water_available_l,
                            state.ix_resin_cycles_used, concentration)
    remaining -= i_res["processed_kg"]
    state.ix_resin_cycles_used = i_res["cycles_used"]
    result.ix_processed_kg = i_res["processed_kg"]

    # --- Aggregate ---
    total_proc = (t_res["processed_kg"] + b_res["processed_kg"]
                  + i_res["processed_kg"])
    total_perc = (t_res["perchlorate_removed_kg"]
                  + b_res["perchlorate_removed_kg"]
                  + i_res["perchlorate_removed_kg"])
    total_clean = (t_res["clean_kg"] + b_res["clean_kg"] + i_res["clean_kg"])
    total_o2 = t_res["o2_recovered_kg"]  # only thermal liberates O2
    total_cl = (t_res.get("chloride_kg", 0.0)
                + chloride_from_perchlorate_kg(b_res["perchlorate_removed_kg"])
                + chloride_from_perchlorate_kg(i_res["perchlorate_removed_kg"]))
    total_energy = (t_res["energy_used_kwh"] + b_res["energy_used_kwh"]
                    + i_res["energy_used_kwh"])
    total_water = i_res.get("water_used_l", 0.0)

    # Update state
    state.contaminated_kg = max(0.0, remaining)
    state.clean_kg += total_clean
    state.perchlorate_waste_kg += total_cl
    state.o2_liberated_kg += total_o2
    state.total_processed_kg += total_proc
    state.total_energy_kwh += total_energy
    state.total_o2_recovered_kg += total_o2
    state.total_water_used_l += total_water
    state.sols_running += 1

    # Populate result
    result.total_processed_kg = total_proc
    result.perchlorate_removed_kg = total_perc
    result.o2_recovered_kg = total_o2
    result.chloride_produced_kg = total_cl
    result.energy_used_kwh = total_energy
    result.water_used_l = total_water
    result.clean_kg_produced = total_clean
    result.bio_culture_health = state.bio_culture_health
    result.ix_resin_remaining_cycles = max(0, IX_RESIN_CYCLES - state.ix_resin_cycles_used)

    # Alert assessment
    if state.contaminated_kg > 5000:
        state.alert = "warning"
        result.alert = "warning"
    elif state.bio_culture_health < 0.2:
        state.alert = "warning"
        result.alert = "warning"
    else:
        state.alert = "nominal"
        result.alert = "nominal"

    return state, result


# -- Factory ------------------------------------------------------------------

def create_scrubber(kiln_temp_c: float = 600.0,
                     bio_temp_c: float = 30.0,
                     electron_donor_kg: float = 50.0) -> ScrubberState:
    """Create a perchlorate scrubber with sensible defaults."""
    return ScrubberState(
        kiln_temp_c=max(20.0, min(THERMAL_MAX_TEMP_C, kiln_temp_c)),
        bio_temp_c=max(0.0, min(50.0, bio_temp_c)),
        electron_donor_kg=max(0.0, electron_donor_kg),
        bio_culture_health=0.5,
    )


def scrubber_power_kwh(state: ScrubberState) -> float:
    """Estimate power draw for one sol at current throughput."""
    thermal_share = min(state.contaminated_kg * 0.4, THERMAL_THROUGHPUT_KG_SOL)
    bio_share = min(state.contaminated_kg * 0.25, BIO_BASE_RATE_KG_SOL)
    ix_share = min(state.contaminated_kg * 0.35, IX_THROUGHPUT_KG_SOL)
    return (thermal_share * THERMAL_KWH_PER_KG
            + bio_share * BIO_KWH_PER_KG
            + ix_share * IX_KWH_PER_KG)
