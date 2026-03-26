"""
thermal_control.py -- Mars habitat thermal regulation system.

Models heat balance for pressurised habitat modules on Mars.
Without active thermal control, habitats freeze in hours.

Heat sources:
  - Electric heaters (primary, powered by solar/nuclear)
  - Crew metabolic heat (~100 W/person awake, ~70 W sleeping)
  - Equipment waste heat (computing, life support exhaust)
  - Solar gain through windows/greenhouse panels

Heat sinks:
  - Radiation to Mars sky (effective sky temp ~ -80C)
  - Conduction through habitat walls to regolith
  - Airlock cycling losses (depressurisation dumps warm air)
  - Atmospheric convection on Mars (very thin, but non-zero)

Physical references:
  - Mars surface temp: -120C to +20C (seasonal, diurnal)
  - Human metabolic heat: 80-120 W (NASA HIDH, activity-dependent)
  - ISS thermal control: ammonia loop, 14 kW rejection capacity
  - Habitat insulation: aerogel R-value ~ 10 m2K/W per cm
  - Stefan-Boltzmann: q = epsilon * sigma * A * (T_s^4 - T_sky^4)
  - Mars sky effective temp: ~ -80C (193 K) for radiative loss

One tick = one sol. Temperature in Celsius, power in kW, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

STEFAN_BOLTZMANN = 5.67e-8           # W/(m2 K4)
HABITAT_EMISSIVITY = 0.85           # painted aluminum shell
MARS_SKY_TEMP_K = 193.0             # effective radiative sky temperature

METABOLIC_HEAT_W = 100.0            # per person, average over sol
EQUIPMENT_HEAT_W_PER_KWH = 50.0     # waste heat per kWh consumed by equipment

# Insulation
AEROGEL_R_PER_CM = 10.0             # m2 K/W per cm of aerogel
DEFAULT_INSULATION_CM = 5.0         # 5 cm aerogel baseline

# Habitat geometry (cylinder approximation)
DEFAULT_RADIUS_M = 5.0              # habitat radius
DEFAULT_LENGTH_M = 20.0             # habitat length
DEFAULT_WALL_AREA_M2 = 628.0        # 2 * pi * 5 * 20 = ~628 m2

# Conduction through regolith floor
REGOLITH_CONDUCTIVITY_W_MK = 0.02   # Mars regolith, very low (vacuum pores)
FLOOR_AREA_M2 = 100.0               # floor contact area
REGOLITH_DEPTH_M = 2.0              # depth to stable temp layer

# Airlock losses
AIRLOCK_VOLUME_M3 = 5.0             # volume lost per cycle
AIRLOCK_CYCLES_PER_SOL = 4.0        # average EVA sorties per sol
AIR_HEAT_CAPACITY_J_M3K = 1200.0    # volumetric heat capacity of habitat air

# Thermal targets
TARGET_TEMP_C = 21.0                # comfortable habitat temperature
TEMP_TOLERANCE_C = 3.0              # +/- 3C acceptable
MIN_SAFE_TEMP_C = 10.0              # hypothermia risk below this
MAX_SAFE_TEMP_C = 35.0              # heat stress above this

# Heater
HEATER_COP = 1.0                    # resistive heater: 1 kW electric = 1 kW thermal
HEATER_MAX_KW = 50.0                # maximum heater output

SOL_SECONDS = 88775.0               # Mars sol in seconds (24h 39m 35s)
SOL_HOURS = SOL_SECONDS / 3600.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InsulationSpec:
    """Habitat insulation configuration.

    Attributes:
        thickness_cm: aerogel insulation thickness
        wall_area_m2: total external wall area
        floor_area_m2: regolith contact area
    """
    thickness_cm: float = DEFAULT_INSULATION_CM
    wall_area_m2: float = DEFAULT_WALL_AREA_M2
    floor_area_m2: float = FLOOR_AREA_M2

    def __post_init__(self) -> None:
        self.thickness_cm = max(0.1, self.thickness_cm)
        self.wall_area_m2 = max(1.0, self.wall_area_m2)
        self.floor_area_m2 = max(1.0, self.floor_area_m2)

    @property
    def r_value(self) -> float:
        """Thermal resistance (m2 K/W) of the insulation."""
        return self.thickness_cm * AEROGEL_R_PER_CM


@dataclass
class ThermalState:
    """Habitat thermal state.

    Attributes:
        interior_temp_c: current interior temperature
        heater_output_kw: current heater power
        total_heat_input_kwh: cumulative heat added
        total_heat_lost_kwh: cumulative heat lost
        sols_operational: number of sols tracked
    """
    interior_temp_c: float = TARGET_TEMP_C
    heater_output_kw: float = 0.0
    total_heat_input_kwh: float = 0.0
    total_heat_lost_kwh: float = 0.0
    sols_operational: int = 0


# ---------------------------------------------------------------------------
# Heat flow calculations
# ---------------------------------------------------------------------------

def radiative_loss_kw(
    wall_area_m2: float,
    surface_temp_c: float,
    sky_temp_k: float = MARS_SKY_TEMP_K,
) -> float:
    """Radiative heat loss from habitat shell to Mars sky (kW).

    Stefan-Boltzmann: q = epsilon * sigma * A * (Ts^4 - Tsky^4)
    """
    t_surface_k = surface_temp_c + 273.15
    if t_surface_k <= 0:
        return 0.0
    q_w = (HABITAT_EMISSIVITY * STEFAN_BOLTZMANN * wall_area_m2
           * (t_surface_k**4 - sky_temp_k**4))
    return max(0.0, q_w / 1000.0)


def conductive_loss_kw(
    interior_temp_c: float,
    exterior_temp_c: float,
    insulation: InsulationSpec,
) -> float:
    """Conductive heat loss through insulated walls (kW).

    Q = A * (Ti - Te) / R_total
    """
    r_total = insulation.r_value
    if r_total <= 0:
        return 0.0
    delta_t = interior_temp_c - exterior_temp_c
    if delta_t <= 0:
        return 0.0
    q_w = insulation.wall_area_m2 * delta_t / r_total
    return q_w / 1000.0


def floor_loss_kw(
    interior_temp_c: float,
    regolith_temp_c: float = -40.0,
) -> float:
    """Conductive loss through floor to Mars regolith (kW).

    Simple 1D conduction: Q = k * A * dT / depth
    """
    delta_t = interior_temp_c - regolith_temp_c
    if delta_t <= 0:
        return 0.0
    q_w = (REGOLITH_CONDUCTIVITY_W_MK * FLOOR_AREA_M2
           * delta_t / REGOLITH_DEPTH_M)
    return q_w / 1000.0


def airlock_loss_kwh(
    interior_temp_c: float,
    exterior_temp_c: float,
    cycles: float = AIRLOCK_CYCLES_PER_SOL,
) -> float:
    """Heat lost per sol from airlock cycling (kWh).

    Each cycle vents warm air and replaces with cold Mars air.
    """
    delta_t = interior_temp_c - exterior_temp_c
    if delta_t <= 0:
        return 0.0
    q_j = cycles * AIRLOCK_VOLUME_M3 * AIR_HEAT_CAPACITY_J_M3K * delta_t
    return q_j / 3.6e6  # J to kWh


def metabolic_heat_kw(population: int) -> float:
    """Total metabolic heat from crew (kW)."""
    return max(0, population) * METABOLIC_HEAT_W / 1000.0


def equipment_heat_kw(equipment_power_kwh: float) -> float:
    """Waste heat from equipment over one sol (average kW).

    Converts total sol energy consumption to average thermal output.
    """
    if equipment_power_kwh <= 0:
        return 0.0
    avg_power_kw = equipment_power_kwh / SOL_HOURS
    return avg_power_kw * EQUIPMENT_HEAT_W_PER_KWH / 1000.0


def required_heating_kw(
    heat_loss_kw: float,
    passive_heat_kw: float,
) -> float:
    """Heater power needed to maintain target temperature (kW).

    If passive heat (metabolic + equipment) exceeds losses,
    no heating needed (may need cooling, but we model heating only).
    """
    deficit = heat_loss_kw - passive_heat_kw
    return max(0.0, min(deficit, HEATER_MAX_KW))


def temperature_drift(
    net_heat_kw: float,
    habitat_mass_kg: float = 50000.0,
    specific_heat_j_kgk: float = 900.0,
) -> float:
    """Temperature change per sol from net heat imbalance (C/sol).

    Uses thermal mass of habitat structure + air.
    """
    if habitat_mass_kg <= 0:
        return 0.0
    q_j = net_heat_kw * 1000.0 * SOL_SECONDS
    return q_j / (habitat_mass_kg * specific_heat_j_kgk)


def comfort_status(temp_c: float) -> str:
    """Assess thermal comfort level."""
    if temp_c < MIN_SAFE_TEMP_C:
        return "critical_cold"
    if temp_c > MAX_SAFE_TEMP_C:
        return "critical_hot"
    if abs(temp_c - TARGET_TEMP_C) <= TEMP_TOLERANCE_C:
        return "comfortable"
    if temp_c < TARGET_TEMP_C:
        return "cool"
    return "warm"


def tick_thermal(
    state: ThermalState,
    insulation: InsulationSpec,
    exterior_temp_c: float,
    population: int,
    equipment_power_kwh: float,
    power_for_heating_kwh: float,
) -> dict:
    """Advance thermal control by one sol.

    Args:
        state: thermal state (mutated in place)
        insulation: habitat insulation spec
        exterior_temp_c: Mars surface temperature this sol
        population: crew count
        equipment_power_kwh: total equipment energy this sol
        power_for_heating_kwh: power budget for heating

    Returns:
        Snapshot dict with thermal metrics.
    """
    state.sols_operational += 1
    t_int = state.interior_temp_c

    # --- Heat losses ---
    # Radiative loss from outer shell is NOT interior heat loss —
    # the insulated shell is near exterior temp and in thermal
    # equilibrium with Mars atmosphere. Interior losses are conductive.
    rad_loss = 0.0
    # Conductive (walls)
    cond_loss = conductive_loss_kw(t_int, exterior_temp_c, insulation)
    # Floor
    fl_loss = floor_loss_kw(t_int)
    # Airlock
    al_loss_kwh = airlock_loss_kwh(t_int, exterior_temp_c)
    al_loss_avg_kw = al_loss_kwh / SOL_HOURS if SOL_HOURS > 0 else 0.0

    total_loss_kw = rad_loss + cond_loss + fl_loss + al_loss_avg_kw

    # --- Heat gains ---
    met_heat = metabolic_heat_kw(population)
    equip_heat = equipment_heat_kw(equipment_power_kwh)
    passive_heat = met_heat + equip_heat

    # Heater
    needed = required_heating_kw(total_loss_kw, passive_heat)
    available_kw = power_for_heating_kwh / SOL_HOURS if SOL_HOURS > 0 else 0.0
    heater_kw = min(needed, available_kw, HEATER_MAX_KW)

    total_gain_kw = passive_heat + heater_kw

    # --- Net heat and temperature change ---
    net_kw = total_gain_kw - total_loss_kw
    drift = temperature_drift(net_kw)
    state.interior_temp_c += drift

    # Clamp to physical bounds
    state.interior_temp_c = max(-50.0, min(50.0, state.interior_temp_c))
    state.heater_output_kw = heater_kw

    # Cumulative tracking
    state.total_heat_input_kwh += total_gain_kw * SOL_HOURS
    state.total_heat_lost_kwh += total_loss_kw * SOL_HOURS

    return {
        "sol": state.sols_operational,
        "interior_temp_c": round(state.interior_temp_c, 2),
        "exterior_temp_c": round(exterior_temp_c, 1),
        "comfort": comfort_status(state.interior_temp_c),
        "heat_loss_kw": round(total_loss_kw, 4),
        "heat_gain_kw": round(total_gain_kw, 4),
        "heater_kw": round(heater_kw, 4),
        "metabolic_heat_kw": round(met_heat, 4),
        "equipment_heat_kw": round(equip_heat, 4),
        "radiative_loss_kw": round(rad_loss, 4),
        "conductive_loss_kw": round(cond_loss, 4),
        "floor_loss_kw": round(fl_loss, 4),
        "airlock_loss_kwh": round(al_loss_kwh, 4),
        "temp_drift_c": round(drift, 4),
        "net_heat_kw": round(net_kw, 4),
    }
