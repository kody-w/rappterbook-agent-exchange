"""
thermal_control.py — Mars habitat thermal regulation model.

Models the thermal balance of a pressurized Mars habitat:
  - Heat sources: electrical heaters, RTG waste heat, crew metabolic heat,
    equipment waste heat, greenhouse thermal mass
  - Heat losses: conduction through walls, radiation to space/ground,
    airlock cycling, atmospheric leak
  - Active cooling: radiator panels for when internal heat exceeds target

Physical references:
  - Mars surface temp: -60°C mean, range -120°C to +20°C
  - Human metabolic heat: ~100 W per person (resting) to ~300 W (working)
  - RTG (MMRTG): ~2000 W thermal, ~110 W electrical (5.5% conversion)
  - ISS thermal control: ammonia loop radiators, ~70 kW rejection capacity
  - Habitat wall U-value: 0.1-0.5 W/(m²·K) depending on insulation
  - Airlock cycle heat loss: ~5 MJ per cycle (depressurize + repressurize)
  - Stefan-Boltzmann radiation: ~5.67e-8 W/(m²·K⁴)

One tick = one sol. Temperature in °C, power in kW, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

STEFAN_BOLTZMANN = 5.670374419e-8     # W/(m²·K⁴)
MARS_SOL_HOURS = 24.66               # hours per sol

# Crew metabolic heat
METABOLIC_HEAT_W = 120.0              # average per person (mix of rest/work)

# RTG (Multi-Mission Radioisotope Thermoelectric Generator)
RTG_THERMAL_W = 2000.0               # thermal output per RTG unit
RTG_ELECTRICAL_W = 110.0             # electrical output per RTG unit
RTG_DECAY_PER_SOL = 0.0000035        # Pu-238 half-life ~87.7 years

# Habitat insulation
DEFAULT_WALL_U = 0.25                 # W/(m²·K) — aerogel-insulated regolith
REGOLITH_BERM_U = 0.10               # W/(m²·K) — heavy regolith berm insulation
MINIMAL_INSULATION_U = 0.50          # W/(m²·K) — basic tent/inflatable

# Equipment heat
EQUIPMENT_HEAT_FRACTION = 0.15        # fraction of total electrical power → waste heat

# Airlock
AIRLOCK_HEAT_LOSS_KWH = 1.4          # kWh per cycle (depressurize + cold air influx)
DEFAULT_AIRLOCK_CYCLES_SOL = 4.0      # EVA crew, 2 exits + 2 entries

# Radiator panel
RADIATOR_EMISSIVITY = 0.90           # high-emissivity coating
RADIATOR_AREA_M2_PER_UNIT = 10.0     # m² per deployable radiator unit
RADIATOR_TEMP_K = 300.0              # typical radiator operating temperature

# Thermal mass
THERMAL_MASS_KWH_PER_C = 0.5         # habitat thermal mass per °C (kWh)
# This represents the energy stored in walls, air, water tanks, regolith.
# A well-designed habitat is a thermal flywheel.

# Comfort range
COMFORT_MIN_C = 18.0
COMFORT_MAX_C = 26.0
TARGET_TEMP_C = 22.0

# Emergency thresholds
HYPOTHERMIA_C = 10.0
HYPERTHERMIA_C = 35.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HabitatThermal:
    """Thermal state of a Mars habitat.

    Attributes:
        interior_temp_c: current interior temperature
        wall_area_m2: total habitat wall/ceiling/floor area
        wall_u_value: thermal transmittance W/(m²·K)
        rtg_count: number of RTG units providing waste heat
        rtg_efficiency: RTG thermal output fraction (decays over time)
        radiator_units: number of deployable radiator panels
        thermal_mass_factor: multiplier on base thermal mass
    """
    interior_temp_c: float = TARGET_TEMP_C
    wall_area_m2: float = 400.0
    wall_u_value: float = DEFAULT_WALL_U
    rtg_count: int = 2
    rtg_efficiency: float = 1.0
    radiator_units: int = 1
    thermal_mass_factor: float = 1.0

    def __post_init__(self) -> None:
        self.interior_temp_c = max(-50.0, min(60.0, self.interior_temp_c))
        self.wall_area_m2 = max(1.0, self.wall_area_m2)
        self.wall_u_value = max(0.01, min(2.0, self.wall_u_value))
        self.rtg_count = max(0, self.rtg_count)
        self.rtg_efficiency = max(0.0, min(1.0, self.rtg_efficiency))
        self.radiator_units = max(0, self.radiator_units)
        self.thermal_mass_factor = max(0.1, self.thermal_mass_factor)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def conduction_loss_kwh(
    wall_area_m2: float,
    wall_u: float,
    interior_c: float,
    exterior_c: float,
) -> float:
    """Heat lost through habitat walls per sol (kWh).

    Q = U × A × ΔT × time
    Positive = heat flowing out (interior warmer than exterior).
    """
    delta_t = interior_c - exterior_c
    watts = wall_u * wall_area_m2 * delta_t
    kwh = watts * MARS_SOL_HOURS / 1000.0
    return round(kwh, 4)


def radiation_loss_kwh(
    radiator_units: int,
    radiator_temp_k: float = RADIATOR_TEMP_K,
) -> float:
    """Heat rejected through radiator panels per sol (kWh).

    Stefan-Boltzmann: P = ε σ A T⁴
    Only active when habitat needs cooling (radiators deployed).
    """
    if radiator_units <= 0:
        return 0.0
    area = radiator_units * RADIATOR_AREA_M2_PER_UNIT
    watts = RADIATOR_EMISSIVITY * STEFAN_BOLTZMANN * area * (radiator_temp_k ** 4)
    kwh = watts * MARS_SOL_HOURS / 1000.0
    return round(kwh, 4)


def metabolic_heat_kwh(population: int) -> float:
    """Total crew metabolic heat per sol (kWh)."""
    watts = max(0, population) * METABOLIC_HEAT_W
    return round(watts * MARS_SOL_HOURS / 1000.0, 4)


def rtg_heat_kwh(count: int, efficiency: float) -> float:
    """RTG waste heat per sol (kWh).

    RTGs produce thermal energy continuously. The electrical portion
    is used for power; the rest is waste heat that heats the habitat.
    """
    thermal_watts = max(0, count) * RTG_THERMAL_W * max(0.0, min(1.0, efficiency))
    # Subtract electrical output — that's already counted as power
    electrical_watts = max(0, count) * RTG_ELECTRICAL_W * max(0.0, min(1.0, efficiency))
    waste_watts = thermal_watts - electrical_watts
    return round(max(0.0, waste_watts) * MARS_SOL_HOURS / 1000.0, 4)


def equipment_heat_kwh(total_power_kwh: float) -> float:
    """Waste heat from all electrical equipment per sol (kWh)."""
    return round(max(0.0, total_power_kwh) * EQUIPMENT_HEAT_FRACTION, 4)


def airlock_loss_kwh(cycles: float) -> float:
    """Heat lost from airlock cycling per sol (kWh)."""
    return round(max(0.0, cycles) * AIRLOCK_HEAT_LOSS_KWH, 4)


def heater_demand_kwh(
    heat_deficit_kwh: float,
    heater_efficiency: float = 0.95,
) -> float:
    """Electrical power needed for supplemental heating (kWh).

    Returns the power the heater must draw to make up a thermal deficit.
    Resistance heaters are nearly 100% efficient (all electricity → heat).
    """
    if heat_deficit_kwh <= 0:
        return 0.0
    return round(heat_deficit_kwh / max(0.01, heater_efficiency), 4)


def comfort_score(temp_c: float) -> float:
    """Crew comfort score based on interior temperature [0, 1].

    1.0 in comfort range, drops linearly toward 0 at extremes.
    """
    if COMFORT_MIN_C <= temp_c <= COMFORT_MAX_C:
        return 1.0
    if temp_c < COMFORT_MIN_C:
        if temp_c <= HYPOTHERMIA_C:
            return 0.0
        return (temp_c - HYPOTHERMIA_C) / (COMFORT_MIN_C - HYPOTHERMIA_C)
    # temp_c > COMFORT_MAX_C
    if temp_c >= HYPERTHERMIA_C:
        return 0.0
    return (HYPERTHERMIA_C - temp_c) / (HYPERTHERMIA_C - COMFORT_MAX_C)


def tick_thermal(
    habitat: HabitatThermal,
    exterior_temp_c: float,
    population: int,
    total_equipment_power_kwh: float,
    airlock_cycles: float = DEFAULT_AIRLOCK_CYCLES_SOL,
    heater_power_available_kwh: float = 100.0,
) -> dict:
    """Advance habitat thermal state by one sol.

    Thermal balance:
      heat_in  = RTG + metabolic + equipment + heater
      heat_out = conduction + airlock + radiator (if overheating)
      net = heat_in - heat_out
      ΔT = net / thermal_mass

    Args:
        habitat: thermal state (mutated in place)
        exterior_temp_c: Mars surface temperature this sol
        population: number of crew
        total_equipment_power_kwh: total colony power draw (for waste heat calc)
        airlock_cycles: EVA airlock cycles this sol
        heater_power_available_kwh: power budget for heating

    Returns:
        Snapshot dict with all thermal metrics.
    """
    pop = max(0, population)

    # --- Heat inputs (kWh) ---
    h_rtg = rtg_heat_kwh(habitat.rtg_count, habitat.rtg_efficiency)
    h_metabolic = metabolic_heat_kwh(pop)
    h_equipment = equipment_heat_kwh(total_equipment_power_kwh)

    # --- Heat losses (kWh) ---
    h_conduction = conduction_loss_kwh(
        habitat.wall_area_m2,
        habitat.wall_u_value,
        habitat.interior_temp_c,
        exterior_temp_c,
    )
    h_airlock = airlock_loss_kwh(airlock_cycles)

    # --- Preliminary balance (without heater or radiator) ---
    passive_in = h_rtg + h_metabolic + h_equipment
    passive_out = h_conduction + h_airlock
    balance = passive_in - passive_out

    # --- Active thermal control ---
    heater_kwh = 0.0
    radiator_kwh = 0.0

    thermal_mass = THERMAL_MASS_KWH_PER_C * habitat.thermal_mass_factor
    if thermal_mass <= 0:
        thermal_mass = 0.1

    # Predict temperature change without active control
    predicted_delta = balance / thermal_mass
    predicted_temp = habitat.interior_temp_c + predicted_delta

    if predicted_temp < TARGET_TEMP_C:
        # Need heating
        deficit = (TARGET_TEMP_C - predicted_temp) * thermal_mass
        available_heat = min(deficit, heater_power_available_kwh * 0.95)  # 95% eff
        heater_kwh = heater_demand_kwh(available_heat)
        balance += available_heat
    elif predicted_temp > COMFORT_MAX_C:
        # Need cooling — activate radiators
        max_rejection = radiation_loss_kwh(habitat.radiator_units)
        excess = (predicted_temp - TARGET_TEMP_C) * thermal_mass
        radiator_kwh = min(max_rejection, excess)
        balance -= radiator_kwh

    # --- Temperature update ---
    delta_t = balance / thermal_mass
    habitat.interior_temp_c += delta_t
    habitat.interior_temp_c = max(-50.0, min(60.0, habitat.interior_temp_c))

    # --- RTG decay ---
    habitat.rtg_efficiency = max(0.0, habitat.rtg_efficiency - RTG_DECAY_PER_SOL)

    # --- Comfort ---
    comfort = comfort_score(habitat.interior_temp_c)

    return {
        "interior_temp_c": round(habitat.interior_temp_c, 2),
        "exterior_temp_c": round(exterior_temp_c, 2),
        "heat_rtg_kwh": h_rtg,
        "heat_metabolic_kwh": h_metabolic,
        "heat_equipment_kwh": h_equipment,
        "heater_kwh": round(heater_kwh, 4),
        "loss_conduction_kwh": round(h_conduction, 4),
        "loss_airlock_kwh": round(h_airlock, 4),
        "radiator_rejection_kwh": round(radiator_kwh, 4),
        "net_balance_kwh": round(balance, 4),
        "comfort_score": round(comfort, 4),
        "rtg_efficiency": round(habitat.rtg_efficiency, 6),
    }
