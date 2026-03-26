"""
atmo_recycler.py — Mars habitat atmosphere management.

Models O2 production, CO2 scrubbing, pressure regulation, and trace
contaminant removal for a sealed Mars habitat. Energy-limited:
every process consumes electrical power.

Physical references:
  - ISS ECLSS: 6 crew, 840 W for O2 generation (OGS), 484 W for CO2 removal (CDRA)
  - Human O2 consumption: 0.84 kg/person/day (NASA HIDH)
  - Human CO2 production: 1.04 kg/person/day (NASA HIDH)
  - Water electrolysis: 2 H2O → 2 H2 + O2; 1 kg O2 requires 1.125 kg H2O
  - Sabatier reaction: CO2 + 4 H2 → CH4 + 2 H2O (recovers water from CO2 + H2)
  - Target atmosphere: 21% O2, 0.04% CO2, ~79% N2 buffer at 101.3 kPa
  - Mars ambient: 95.3% CO2, 0.13% O2, 0.636 kPa — completely unbreathable

One tick = one sol. Mass in kg, pressure in kPa, power in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Human metabolic rates (NASA Human Integration Design Handbook)
O2_KG_PER_PERSON_SOL = 0.84      # O2 consumed per person per sol
CO2_KG_PER_PERSON_SOL = 1.04     # CO2 produced per person per sol
H2O_KG_PER_KG_O2 = 1.125         # water needed per kg O2 via electrolysis

# Electrolysis (OGS-equivalent)
ELECTROLYSIS_KWH_PER_KG_O2 = 6.0  # ~5-7 kWh/kg O2 for PEM electrolysis
ELECTROLYSIS_EFFICIENCY = 0.85     # real-world PEM cell efficiency

# CO2 scrubbing (CDRA-equivalent: zeolite adsorption bed)
SCRUBBER_KWH_PER_KG_CO2 = 2.0     # power to adsorb + desorb CO2
SCRUBBER_CAPACITY_KG_SOL = 3.0    # max CO2 one scrubber unit handles per sol

# Sabatier reactor (CO2 + 4H2 → CH4 + 2H2O)
SABATIER_H2O_RECOVERY = 0.45      # fraction of electrolysis water recovered
SABATIER_KWH_PER_KG_CO2 = 0.5     # exothermic, but needs heating to start

# Atmosphere targets
TARGET_O2_KPA = 21.3               # partial pressure O2 (sea-level equivalent)
TARGET_CO2_KPA = 0.04              # partial pressure CO2 (safe limit)
TOTAL_PRESSURE_KPA = 101.3         # Earth-standard cabin pressure
CO2_WARNING_KPA = 0.53             # OSHA 8-hour limit (5000 ppm)
CO2_DANGER_KPA = 4.0               # impaired judgment, headaches
CO2_LETHAL_KPA = 10.0              # loss of consciousness

# Leakage
LEAK_RATE_KPA_SOL = 0.01           # ambient pressure loss per sol (microleaks)
AIRLOCK_LOSS_KPA = 0.05            # pressure lost per airlock cycle
N2_BUFFER_RESERVE_KG = 500.0       # initial N2 buffer tank

# Trace contaminants
TRACE_CONTAMINANT_KG_PERSON_SOL = 0.001  # VOCs, ammonia, etc.
TRACE_FILTER_EFFICIENCY = 0.95     # activated carbon filter removal rate


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Atmosphere:
    """Habitat atmospheric state.

    All partial pressures in kPa. Mass values in kg.
    """
    o2_kpa: float = TARGET_O2_KPA
    co2_kpa: float = 0.04
    n2_kpa: float = TOTAL_PRESSURE_KPA - TARGET_O2_KPA - 0.04
    trace_contaminants_kg: float = 0.0
    volume_m3: float = 500.0       # habitat volume (pressurised)

    def total_pressure(self) -> float:
        """Total cabin pressure (kPa)."""
        return self.o2_kpa + self.co2_kpa + self.n2_kpa

    def o2_fraction(self) -> float:
        """O2 mole fraction."""
        total = self.total_pressure()
        if total <= 0:
            return 0.0
        return self.o2_kpa / total

    def co2_ppm(self) -> float:
        """CO2 concentration in ppm."""
        total = self.total_pressure()
        if total <= 0:
            return 0.0
        return (self.co2_kpa / total) * 1_000_000


@dataclass
class LifeSupport:
    """Life support hardware state.

    scrubber_units: number of CO2 scrubber modules
    electrolyzer_capacity_kg_sol: max O2 production per sol (kg)
    sabatier_active: whether Sabatier reactor is operational
    """
    scrubber_units: int = 2
    electrolyzer_capacity_kg_sol: float = 10.0
    sabatier_active: bool = True
    scrubber_health: float = 1.0   # 0-1, degrades over time
    electrolyzer_health: float = 1.0

    def __post_init__(self) -> None:
        self.scrubber_units = max(0, self.scrubber_units)
        self.electrolyzer_capacity_kg_sol = max(0.0, self.electrolyzer_capacity_kg_sol)
        self.scrubber_health = max(0.0, min(1.0, self.scrubber_health))
        self.electrolyzer_health = max(0.0, min(1.0, self.electrolyzer_health))


@dataclass
class AtmoTickResult:
    """Result of one sol of atmosphere management."""
    o2_produced_kg: float = 0.0
    o2_consumed_kg: float = 0.0
    co2_produced_kg: float = 0.0
    co2_scrubbed_kg: float = 0.0
    water_consumed_kg: float = 0.0
    water_recovered_kg: float = 0.0
    power_consumed_kwh: float = 0.0
    pressure_lost_kpa: float = 0.0
    trace_removed_kg: float = 0.0
    co2_alert: str = "nominal"       # nominal, warning, danger, lethal


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def o2_demand(population: int) -> float:
    """O2 needed by crew for one sol (kg)."""
    return population * O2_KG_PER_PERSON_SOL


def co2_production(population: int) -> float:
    """CO2 exhaled by crew for one sol (kg)."""
    return population * CO2_KG_PER_PERSON_SOL


def water_for_electrolysis(o2_kg: float) -> float:
    """Water required to produce given O2 mass via electrolysis (kg)."""
    return o2_kg * H2O_KG_PER_KG_O2


def electrolysis_power(o2_kg: float) -> float:
    """Power to electrolyze water for given O2 production (kWh)."""
    return o2_kg * ELECTROLYSIS_KWH_PER_KG_O2 / ELECTROLYSIS_EFFICIENCY


def scrubber_power(co2_kg: float) -> float:
    """Power for CO2 scrubbing (kWh)."""
    return co2_kg * SCRUBBER_KWH_PER_KG_CO2


def sabatier_water_recovery(co2_kg: float) -> float:
    """Water recovered from Sabatier reaction on scrubbed CO2 (kg).

    CO2 + 4H2 → CH4 + 2H2O. The H2 comes from electrolysis byproduct.
    Not all CO2 is processed — limited by H2 availability.
    """
    return co2_kg * SABATIER_H2O_RECOVERY * H2O_KG_PER_KG_O2


def co2_alert_level(co2_kpa: float) -> str:
    """Classify CO2 partial pressure into alert level."""
    if co2_kpa >= CO2_LETHAL_KPA:
        return "lethal"
    if co2_kpa >= CO2_DANGER_KPA:
        return "danger"
    if co2_kpa >= CO2_WARNING_KPA:
        return "warning"
    return "nominal"


def pressure_to_mass(delta_kpa: float, volume_m3: float, molar_mass: float) -> float:
    """Convert partial pressure change to mass change using ideal gas law.

    PV = nRT → n = PV/(RT) → mass = n * M
    T ≈ 293 K (habitat interior), R = 8.314 J/(mol·K)

    Args:
        delta_kpa: change in partial pressure (kPa)
        volume_m3: habitat volume (m³)
        molar_mass: g/mol of the gas (O2=32, CO2=44, N2=28)

    Returns:
        Mass in kg.
    """
    r = 8.314        # J/(mol·K)
    t = 293.0        # K (~20°C habitat)
    # PV = nRT → n = PV/(RT)
    # P in Pa = kPa * 1000, V in m³
    n_mol = (delta_kpa * 1000.0 * volume_m3) / (r * t)
    return n_mol * molar_mass / 1000.0  # g → kg


def apply_leakage(atmo: Atmosphere, airlock_cycles: int = 0) -> float:
    """Apply ambient pressure loss from microleaks and airlock use.

    Proportional reduction across all gas species.
    Returns total pressure lost (kPa).
    """
    leak = LEAK_RATE_KPA_SOL + airlock_cycles * AIRLOCK_LOSS_KPA
    total = atmo.total_pressure()
    if total <= 0:
        return 0.0
    # Can't lose more than we have
    leak = min(leak, total * 0.1)  # cap at 10% per sol (catastrophe)
    fraction = leak / total
    atmo.o2_kpa -= atmo.o2_kpa * fraction
    atmo.co2_kpa -= atmo.co2_kpa * fraction
    atmo.n2_kpa -= atmo.n2_kpa * fraction
    return leak


def tick_atmosphere(
    atmo: Atmosphere,
    life_support: LifeSupport,
    population: int,
    power_available_kwh: float,
    water_available_kg: float,
    airlock_cycles: int = 0,
) -> AtmoTickResult:
    """Advance atmosphere management by one sol.

    Priority order:
      1. Crew breathes (O2 consumed, CO2 produced — non-negotiable)
      2. CO2 scrubbing (prevent toxic buildup)
      3. O2 production via electrolysis (if water and power available)
      4. Sabatier water recovery (if active)
      5. Trace contaminant filtering
      6. Pressure leakage

    Conservation laws:
      - O2 consumed ≤ O2 available (partial pressure)
      - CO2 scrubbed ≤ CO2 present + CO2 produced
      - Water consumed ≤ water available
      - Power consumed ≤ power available
      - All values ≥ 0

    Args:
        atmo: habitat atmosphere (mutated in place)
        life_support: hardware state
        population: crew count
        power_available_kwh: power budget for life support this sol
        water_available_kg: water budget for electrolysis this sol
        airlock_cycles: number of airlock open/close cycles this sol

    Returns:
        AtmoTickResult with all mass/energy flows.
    """
    result = AtmoTickResult()
    power_remaining = max(0.0, power_available_kwh)
    water_remaining = max(0.0, water_available_kg)

    # --- Step 1: Crew metabolism (always happens) ---
    o2_needed = o2_demand(population)
    co2_made = co2_production(population)
    trace_made = population * TRACE_CONTAMINANT_KG_PERSON_SOL

    # Convert O2 consumption to pressure change
    o2_mass_available = pressure_to_mass(atmo.o2_kpa, atmo.volume_m3, 32.0)
    o2_actually_consumed = min(o2_needed, o2_mass_available)
    o2_kpa_drop = (o2_actually_consumed / max(o2_mass_available, 1e-9)) * atmo.o2_kpa
    atmo.o2_kpa -= o2_kpa_drop
    atmo.o2_kpa = max(0.0, atmo.o2_kpa)
    result.o2_consumed_kg = o2_actually_consumed

    # CO2 added to atmosphere
    co2_kpa_gain = (co2_made / max(pressure_to_mass(100.0, atmo.volume_m3, 44.0), 1e-9)) * 100.0
    atmo.co2_kpa += co2_kpa_gain
    result.co2_produced_kg = co2_made

    # Trace contaminants accumulate
    atmo.trace_contaminants_kg += trace_made

    # --- Step 2: CO2 scrubbing (highest priority after breathing) ---
    scrub_capacity = (
        life_support.scrubber_units
        * SCRUBBER_CAPACITY_KG_SOL
        * life_support.scrubber_health
    )
    co2_mass_present = pressure_to_mass(atmo.co2_kpa, atmo.volume_m3, 44.0)
    co2_to_scrub = min(co2_mass_present, scrub_capacity)

    scrub_power_needed = scrubber_power(co2_to_scrub)
    if scrub_power_needed > power_remaining:
        # Power-limited scrubbing
        co2_to_scrub = co2_to_scrub * (power_remaining / scrub_power_needed)
        scrub_power_needed = power_remaining

    power_remaining -= scrub_power_needed
    result.co2_scrubbed_kg = co2_to_scrub
    result.power_consumed_kwh += scrub_power_needed

    # Remove scrubbed CO2 from atmosphere
    if co2_mass_present > 0:
        co2_kpa_removed = (co2_to_scrub / co2_mass_present) * atmo.co2_kpa
        atmo.co2_kpa -= co2_kpa_removed
        atmo.co2_kpa = max(0.0, atmo.co2_kpa)

    # --- Step 3: O2 production via electrolysis ---
    o2_target = o2_needed  # replace what was consumed
    o2_producible = life_support.electrolyzer_capacity_kg_sol * life_support.electrolyzer_health
    o2_to_produce = min(o2_target, o2_producible)

    water_needed = water_for_electrolysis(o2_to_produce)
    if water_needed > water_remaining:
        # Water-limited
        o2_to_produce = o2_to_produce * (water_remaining / water_needed)
        water_needed = water_remaining

    elec_power_needed = electrolysis_power(o2_to_produce)
    if elec_power_needed > power_remaining:
        # Power-limited
        o2_to_produce = o2_to_produce * (power_remaining / elec_power_needed)
        water_needed = water_for_electrolysis(o2_to_produce)
        elec_power_needed = power_remaining

    power_remaining -= elec_power_needed
    water_remaining -= water_needed
    result.o2_produced_kg = o2_to_produce
    result.water_consumed_kg = water_needed
    result.power_consumed_kwh += elec_power_needed

    # Add produced O2 to atmosphere
    o2_mass_now = pressure_to_mass(atmo.o2_kpa, atmo.volume_m3, 32.0)
    if o2_mass_now > 0:
        o2_kpa_gain = (o2_to_produce / o2_mass_now) * atmo.o2_kpa
    else:
        # Atmosphere was depleted — recompute from mass directly
        o2_kpa_gain = (o2_to_produce * 1000.0 / 32.0) * 8.314 * 293.0 / (atmo.volume_m3 * 1000.0)
    atmo.o2_kpa += o2_kpa_gain

    # --- Step 4: Sabatier water recovery ---
    if life_support.sabatier_active and result.co2_scrubbed_kg > 0:
        sab_power = result.co2_scrubbed_kg * SABATIER_KWH_PER_KG_CO2
        if sab_power <= power_remaining:
            recovered = sabatier_water_recovery(result.co2_scrubbed_kg)
            result.water_recovered_kg = recovered
            result.power_consumed_kwh += sab_power
            power_remaining -= sab_power

    # --- Step 5: Trace contaminant filtering ---
    removed = atmo.trace_contaminants_kg * TRACE_FILTER_EFFICIENCY
    atmo.trace_contaminants_kg -= removed
    result.trace_removed_kg = removed

    # --- Step 6: Pressure leakage ---
    result.pressure_lost_kpa = apply_leakage(atmo, airlock_cycles)

    # --- CO2 alert classification ---
    result.co2_alert = co2_alert_level(atmo.co2_kpa)

    return result
