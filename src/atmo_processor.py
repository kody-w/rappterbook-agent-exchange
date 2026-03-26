"""
atmo_processor.py — Mars atmospheric CO2 → O2 conversion (MOXIE-class SOCE).

Models solid-oxide CO2 electrolysis (SOCE) for in-situ O2 production,
based on NASA's MOXIE experiment aboard Perseverance (2021-2023).

Physical references:
  - Mars atmosphere: 95.3% CO2, 0.636 kPa surface pressure
  - MOXIE: 6-10 g O2/hr, ~300W input, 800°C SOCE stack
  - Human O2 need: 0.84 kg/person/sol (NASA HRP)
  - Electrolysis: CO2 → CO + ½O2 (ΔH = 283 kJ/mol at 800°C)
  - Molar masses: CO2 = 44 g/mol, O2 = 32 g/mol
  - Theoretical yield: 32/44 = 0.727 kg O2 per kg CO2
  - MOXIE measured efficiency: ~50-60% of theoretical
  - Intake compression: 0.636 kPa → ~101 kPa (160x, scroll pump)

One tick = one sol. Mass in kg, power in kWh, temperature in °C.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

CO2_FRACTION_ATMOSPHERE = 0.953        # Mars atmo composition
MARS_SURFACE_PRESSURE_KPA = 0.636      # ambient surface pressure
SOCE_OPERATING_TEMP_C = 800.0          # solid oxide cell stack temperature
SOCE_MIN_TEMP_C = 600.0               # below this, cell output drops sharply
SOCE_MAX_TEMP_C = 1000.0              # above this, thermal stress degrades

# Thermodynamics
ENTHALPY_KJ_PER_MOL_CO2 = 283.0       # ΔH for CO2 → CO + ½O2 at 800°C
MOLAR_MASS_CO2_KG = 0.044             # kg/mol
MOLAR_MASS_O2_KG = 0.032              # kg/mol
THEORETICAL_YIELD = MOLAR_MASS_O2_KG / MOLAR_MASS_CO2_KG  # 0.7273 kg O2 / kg CO2

# MOXIE-scaled constants
SOCE_BASE_EFFICIENCY = 0.55           # 55% of theoretical (MOXIE measured)
COMPRESSION_POWER_KWH_PER_KG_CO2 = 0.15   # scroll pump work per kg CO2 intake
HEATING_POWER_KWH_PER_HOUR = 0.08    # maintaining 800°C stack temperature
KJ_PER_KWH = 3600.0
SOL_HOURS = 24.66                     # Mars sol in hours

# Human consumption
O2_KG_PER_PERSON_SOL = 0.84           # NASA HRP breathing requirement
CO2_EXHALED_KG_PER_PERSON_SOL = 1.04  # human CO2 output (can supplement intake)

# Equipment degradation
DEGRADATION_RATE_PER_SOL = 0.00005    # thermal cycling wears SOCE membranes
DUST_INTAKE_PENALTY = 0.15            # dust storms clog intake filters
MAINTENANCE_REPAIR_FRACTION = 0.50    # maintenance restores 50% of degradation

# Storage
O2_STORAGE_LEAK_RATE_PER_SOL = 0.001  # 0.1%/sol micro-leak from seals


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SOCEUnit:
    """A solid-oxide CO2 electrolysis unit (MOXIE-class).

    Attributes:
        capacity_kg_sol: max O2 production in kg/sol at full efficiency
        degradation: cumulative wear [0, 1] (1 = fully degraded)
        operating_hours: total hours the unit has operated
    """
    capacity_kg_sol: float
    degradation: float = 0.0
    operating_hours: float = 0.0

    def __post_init__(self) -> None:
        self.capacity_kg_sol = max(0.0, self.capacity_kg_sol)
        self.degradation = max(0.0, min(1.0, self.degradation))
        self.operating_hours = max(0.0, self.operating_hours)

    def effective_capacity(self) -> float:
        """Capacity after degradation losses."""
        return self.capacity_kg_sol * (1.0 - self.degradation)


@dataclass
class O2Tank:
    """Oxygen storage tank.

    Attributes:
        capacity_kg: maximum O2 storage
        level_kg: current O2 stored
    """
    capacity_kg: float
    level_kg: float = 0.0

    def __post_init__(self) -> None:
        self.capacity_kg = max(0.0, self.capacity_kg)
        self.level_kg = max(0.0, min(self.level_kg, self.capacity_kg))

    def headroom(self) -> float:
        """Available storage capacity (kg)."""
        return max(0.0, self.capacity_kg - self.level_kg)

    def store(self, kg: float) -> float:
        """Store O2. Returns actual kg stored."""
        if kg <= 0:
            return 0.0
        actual = min(kg, self.headroom())
        self.level_kg += actual
        return actual

    def draw(self, kg_needed: float) -> float:
        """Draw O2 from tank. Returns actual kg delivered."""
        if kg_needed <= 0:
            return 0.0
        delivered = min(kg_needed, self.level_kg)
        self.level_kg -= delivered
        return delivered

    def apply_leak(self) -> float:
        """Apply daily micro-leak. Returns kg lost."""
        lost = self.level_kg * O2_STORAGE_LEAK_RATE_PER_SOL
        self.level_kg -= lost
        return lost

    def days_of_reserve(self, population: int) -> float:
        """How many sols of O2 remain for a given population."""
        if population <= 0:
            return float('inf')
        daily_need = population * O2_KG_PER_PERSON_SOL
        if daily_need <= 0:
            return float('inf')
        return self.level_kg / daily_need


# ---------------------------------------------------------------------------
# Core physics functions
# ---------------------------------------------------------------------------

def soce_efficiency(
    dust_opacity: float,
    pressure_kpa: float,
    degradation: float,
) -> float:
    """SOCE conversion efficiency [0, 1].

    Factors:
      - Base efficiency (MOXIE-measured ~55%)
      - Dust storms clog intake filters, reducing throughput
      - Lower pressure means harder compression (marginal effect)
      - Equipment degradation reduces membrane performance

    Returns fraction of theoretical yield actually achieved.
    """
    dust_penalty = DUST_INTAKE_PENALTY * min(1.0, dust_opacity)
    pressure_factor = min(1.0, pressure_kpa / MARS_SURFACE_PRESSURE_KPA)
    equip_factor = 1.0 - degradation

    eff = SOCE_BASE_EFFICIENCY * (1.0 - dust_penalty) * pressure_factor * equip_factor
    return max(0.0, min(1.0, eff))


def power_required_kwh(co2_intake_kg: float) -> float:
    """Total power to process a given mass of CO2 in one sol.

    Components:
      1. Electrolysis energy (thermodynamic minimum × overhead)
      2. Compression energy (scroll pump from 0.636 to ~101 kPa)
      3. Heating energy (maintaining 800°C stack)
    """
    if co2_intake_kg <= 0:
        return 0.0

    # Electrolysis: moles × ΔH, with 2x overhead for real-world losses
    moles = co2_intake_kg / MOLAR_MASS_CO2_KG
    electrolysis_kj = moles * ENTHALPY_KJ_PER_MOL_CO2 * 2.0
    electrolysis_kwh = electrolysis_kj / KJ_PER_KWH

    # Compression
    compression_kwh = co2_intake_kg * COMPRESSION_POWER_KWH_PER_KG_CO2

    # Heating (constant while operating — one full sol)
    heating_kwh = HEATING_POWER_KWH_PER_HOUR * SOL_HOURS

    return electrolysis_kwh + compression_kwh + heating_kwh


def o2_from_co2(co2_kg: float, efficiency: float) -> float:
    """O2 produced from a given CO2 intake at a given efficiency.

    Theoretical: 0.727 kg O2 per kg CO2.
    Actual: theoretical × efficiency.
    """
    if co2_kg <= 0 or efficiency <= 0:
        return 0.0
    return co2_kg * THEORETICAL_YIELD * efficiency


def co2_required_for_o2(o2_target_kg: float, efficiency: float) -> float:
    """CO2 intake needed to produce a target amount of O2."""
    if o2_target_kg <= 0 or efficiency <= 0:
        return 0.0
    return o2_target_kg / (THEORETICAL_YIELD * efficiency)


def degrade_unit(unit: SOCEUnit, radiation_msv: float) -> float:
    """Apply one sol of thermal cycling and radiation degradation.

    Higher radiation accelerates SOCE membrane degradation.
    Returns degradation delta.
    """
    rad_factor = max(1.0, radiation_msv / 0.67)
    delta = DEGRADATION_RATE_PER_SOL * rad_factor
    unit.degradation = min(1.0, unit.degradation + delta)
    unit.operating_hours += SOL_HOURS
    return delta


def maintain_unit(unit: SOCEUnit) -> float:
    """Perform maintenance on SOCE unit. Returns degradation recovered."""
    recovered = unit.degradation * MAINTENANCE_REPAIR_FRACTION
    unit.degradation -= recovered
    unit.degradation = max(0.0, unit.degradation)
    return recovered


# ---------------------------------------------------------------------------
# Sol-level tick
# ---------------------------------------------------------------------------

def tick_atmo_processor(
    unit: SOCEUnit,
    tank: O2Tank,
    population: int,
    power_available_kwh: float,
    dust_opacity: float,
    pressure_kpa: float,
    radiation_msv: float,
) -> dict:
    """Advance atmospheric processor by one sol.

    Pipeline:
      1. Degrade SOCE unit (thermal cycling + radiation)
      2. Calculate efficiency
      3. Determine CO2 intake (power-limited)
      4. Produce O2
      5. Tank leak
      6. Meet crew demand from production + tank
      7. Surplus to tank

    Args:
        unit: SOCE processor (mutated in place)
        tank: O2 storage (mutated in place)
        population: number of crew requiring O2
        power_available_kwh: power budget for this sol
        dust_opacity: current dust opacity [0, 1]
        pressure_kpa: atmospheric pressure (kPa)
        radiation_msv: ambient radiation (mSv/sol)

    Returns:
        Snapshot dict with all O2 metrics for the sol.
    """
    # 1. Degrade
    deg_delta = degrade_unit(unit, radiation_msv)

    # 2. Efficiency
    eff = soce_efficiency(dust_opacity, pressure_kpa, unit.degradation)

    # 3. Power-limited CO2 intake
    # How much O2 can the unit produce at current efficiency?
    max_o2 = unit.effective_capacity()

    # How much CO2 is needed for that O2?
    co2_for_max = co2_required_for_o2(max_o2, eff) if eff > 0 else 0.0

    # How much power would that require?
    power_for_max = power_required_kwh(co2_for_max)

    # Scale down if power-limited
    if power_for_max > 0 and power_available_kwh < power_for_max:
        scale = power_available_kwh / power_for_max
    else:
        scale = 1.0

    co2_intake = co2_for_max * scale
    o2_produced = o2_from_co2(co2_intake, eff)
    power_consumed = power_required_kwh(co2_intake)

    # 4. Tank leak
    leak_kg = tank.apply_leak()

    # 5. Crew demand
    demand_kg = population * O2_KG_PER_PERSON_SOL

    # 6. Meet demand: production first, then tank
    from_production = min(o2_produced, demand_kg)
    remaining_demand = demand_kg - from_production
    surplus_production = o2_produced - from_production

    from_tank = 0.0
    if remaining_demand > 0:
        from_tank = tank.draw(remaining_demand)

    # 7. Surplus to tank
    stored = 0.0
    if surplus_production > 0:
        stored = tank.store(surplus_production)

    delivered = from_production + from_tank
    deficit = max(0.0, demand_kg - delivered)
    reserve_sols = tank.days_of_reserve(population)

    return {
        "o2_produced_kg": round(o2_produced, 4),
        "co2_intake_kg": round(co2_intake, 4),
        "power_consumed_kwh": round(power_consumed, 4),
        "demand_kg": round(demand_kg, 4),
        "delivered_kg": round(delivered, 4),
        "deficit_kg": round(deficit, 4),
        "from_production_kg": round(from_production, 4),
        "from_tank_kg": round(from_tank, 4),
        "tank_stored_kg": round(stored, 4),
        "tank_level_kg": round(tank.level_kg, 4),
        "tank_leak_kg": round(leak_kg, 4),
        "reserve_sols": round(reserve_sols, 2),
        "efficiency": round(eff, 4),
        "degradation": round(unit.degradation, 6),
        "degradation_delta": round(deg_delta, 6),
        "operating_hours": round(unit.operating_hours, 2),
        "power_scale": round(scale, 4),
    }
