"""
thermal.py — Mars colony thermal management system.

Models heat balance for a Mars habitat: waste heat sources, radiative
losses, active heating, and thermal storage. Mars surface averages -60°C;
habitats must maintain 18-24°C. The thermal gradient is both the colony's
greatest enemy (freezing) and its hidden resource (waste heat recycling).

Physical references:
  - Mars mean surface temp: -60°C (varies -120°C to +20°C)
  - Stefan-Boltzmann constant: 5.67e-8 W/m²/K⁴
  - Mars atmospheric convection: negligible (0.636 kPa)
  - ISS thermal control: ~70 kW rejection via ammonia loop radiators
  - Human metabolic heat: ~100 W/person average (2.4 kWh/sol)
  - Kilopower reactor waste heat: ~7 kW thermal per 1 kW electric
  - SOCE waste heat: ~60% of input power dissipated as heat
  - Habitat insulation: multi-layer aerogel R-value ~10 m²K/W

One tick = one sol. Temperature in °C, energy in kWh, power in kW.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

STEFAN_BOLTZMANN = 5.67e-8            # W/m²/K⁴
MARS_MEAN_TEMP_C = -60.0             # average surface temperature
KELVIN_OFFSET = 273.15               # °C to K
SOL_HOURS = 24.66                    # Mars sol in hours
SECONDS_PER_SOL = SOL_HOURS * 3600   # seconds per sol

# Habitat parameters
HABITAT_TARGET_TEMP_C = 21.0         # comfortable interior temperature
HABITAT_TEMP_TOLERANCE_C = 3.0       # ±3°C acceptable range
HABITAT_MIN_TEMP_C = 5.0             # pipes freeze below this
HABITAT_MAX_TEMP_C = 40.0            # heat stroke risk above this

# Insulation
AEROGEL_R_VALUE = 10.0               # m²·K/W for multi-layer aerogel
DEFAULT_WALL_AREA_M2 = 200.0         # typical habitat exterior surface area

# Human metabolic heat
METABOLIC_HEAT_KWH_SOL = 2.4         # ~100W average per person over a sol

# Waste heat fractions (of electrical power consumed)
SOCE_WASTE_HEAT_FRACTION = 0.60      # SOCE converts 40% to chemistry, 60% waste
NUCLEAR_WASTE_HEAT_RATIO = 7.0       # 7 kW thermal per 1 kW electric (Kilopower)
SOLAR_INVERTER_WASTE = 0.05          # 5% of solar power lost as heat in electronics

# Radiator parameters
RADIATOR_EMISSIVITY = 0.90           # high-emissivity coating
RADIATOR_ABSORPTIVITY = 0.15         # low solar absorptivity (selective surface)
RADIATOR_EFFICIENCY_FLOOR = 0.30     # minimum radiator efficiency (dust-covered)

# Thermal storage
THERMAL_MASS_KWH_PER_C = 0.5        # habitat thermal mass: 0.5 kWh to shift 1°C
STORAGE_LEAK_RATE = 0.02             # 2% of stored thermal energy lost per sol


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HabitatThermal:
    """Thermal state of a habitat module.

    Attributes:
        interior_temp_c: current interior temperature (°C)
        wall_area_m2: exterior surface area (m²)
        insulation_r: thermal resistance (m²·K/W)
        thermal_mass_kwh_c: energy to shift interior by 1°C
    """
    interior_temp_c: float = HABITAT_TARGET_TEMP_C
    wall_area_m2: float = DEFAULT_WALL_AREA_M2
    insulation_r: float = AEROGEL_R_VALUE
    thermal_mass_kwh_c: float = THERMAL_MASS_KWH_PER_C

    def __post_init__(self) -> None:
        self.wall_area_m2 = max(1.0, self.wall_area_m2)
        self.insulation_r = max(0.1, self.insulation_r)
        self.thermal_mass_kwh_c = max(0.01, self.thermal_mass_kwh_c)


@dataclass
class Radiator:
    """External radiator panel for heat rejection.

    Attributes:
        area_m2: radiator surface area
        emissivity: surface emissivity (0-1)
        dust_fraction: dust coverage reducing performance (0-1)
    """
    area_m2: float
    emissivity: float = RADIATOR_EMISSIVITY
    dust_fraction: float = 0.0

    def __post_init__(self) -> None:
        self.area_m2 = max(0.0, self.area_m2)
        self.emissivity = max(0.0, min(1.0, self.emissivity))
        self.dust_fraction = max(0.0, min(1.0, self.dust_fraction))

    def effective_emissivity(self) -> float:
        """Emissivity after dust coverage."""
        floor = RADIATOR_EFFICIENCY_FLOOR * self.emissivity
        return max(floor, self.emissivity * (1.0 - self.dust_fraction))


# ---------------------------------------------------------------------------
# Core physics functions
# ---------------------------------------------------------------------------

def conductive_loss_kwh(
    interior_c: float,
    exterior_c: float,
    wall_area_m2: float,
    insulation_r: float,
) -> float:
    """Heat loss through habitat walls per sol (kWh).

    Q = A × ΔT / R × time
    Positive = heat flowing OUT (interior warmer than exterior).
    """
    delta_t = interior_c - exterior_c
    watts = wall_area_m2 * delta_t / insulation_r
    kwh = watts * SOL_HOURS / 1000.0
    return kwh


def radiative_rejection_kwh(radiator: Radiator, radiator_temp_c: float) -> float:
    """Heat rejected by radiator to space per sol (kWh).

    Stefan-Boltzmann: P = ε × σ × A × T⁴
    Mars atmosphere is nearly vacuum — radiative cooling dominates.
    """
    if radiator.area_m2 <= 0:
        return 0.0
    temp_k = max(0.0, radiator_temp_c + KELVIN_OFFSET)
    eff_emissivity = radiator.effective_emissivity()
    watts = eff_emissivity * STEFAN_BOLTZMANN * radiator.area_m2 * (temp_k ** 4)
    kwh = watts * SOL_HOURS / 1000.0
    return kwh


def metabolic_heat_kwh(population: int) -> float:
    """Total metabolic heat from crew per sol (kWh)."""
    return max(0, population) * METABOLIC_HEAT_KWH_SOL


def waste_heat_kwh(
    solar_kwh: float,
    nuclear_kwh: float,
    soce_kwh: float,
) -> float:
    """Total waste heat from colony systems per sol (kWh).

    Sources:
      - Solar inverter losses
      - Nuclear reactor thermal waste (dominant)
      - SOCE processor waste heat
    """
    solar_waste = max(0.0, solar_kwh) * SOLAR_INVERTER_WASTE
    # Nuclear: electric output × waste ratio (thermal/electric)
    nuclear_waste = max(0.0, nuclear_kwh) * NUCLEAR_WASTE_HEAT_RATIO
    soce_waste = max(0.0, soce_kwh) * SOCE_WASTE_HEAT_FRACTION
    return solar_waste + nuclear_waste + soce_waste


def temperature_delta(
    net_heat_kwh: float,
    thermal_mass_kwh_c: float,
) -> float:
    """Temperature change from net heat input (°C).

    ΔT = Q / C where C = thermal mass (kWh/°C).
    """
    if thermal_mass_kwh_c <= 0:
        return 0.0
    return net_heat_kwh / thermal_mass_kwh_c


def heating_power_needed(
    habitat: HabitatThermal,
    exterior_temp_c: float,
    waste_heat_available_kwh: float,
    metabolic_kwh: float,
) -> float:
    """Active heating power needed to maintain target temperature (kWh).

    Calculates the gap between heat losses and free heat sources.
    Returns 0 if waste heat + metabolic covers the losses.
    """
    loss = conductive_loss_kwh(
        habitat.interior_temp_c, exterior_temp_c,
        habitat.wall_area_m2, habitat.insulation_r,
    )
    free_heat = waste_heat_available_kwh + metabolic_kwh
    deficit = max(0.0, loss - free_heat)
    return deficit


def comfort_score(interior_temp_c: float) -> float:
    """Thermal comfort score [0, 1].

    1.0 = within ±3°C of target (21°C)
    0.0 = at or beyond survival limits (5°C or 40°C)
    """
    target = HABITAT_TARGET_TEMP_C
    tolerance = HABITAT_TEMP_TOLERANCE_C
    diff = abs(interior_temp_c - target)

    if diff <= tolerance:
        return 1.0

    if interior_temp_c <= HABITAT_MIN_TEMP_C:
        return 0.0
    if interior_temp_c >= HABITAT_MAX_TEMP_C:
        return 0.0

    # Linear ramp between tolerance boundary and survival limit
    if interior_temp_c < target:
        range_c = (target - tolerance) - HABITAT_MIN_TEMP_C
        dist = (target - tolerance) - interior_temp_c
    else:
        range_c = HABITAT_MAX_TEMP_C - (target + tolerance)
        dist = interior_temp_c - (target + tolerance)

    if range_c <= 0:
        return 0.0
    return max(0.0, 1.0 - dist / range_c)


# ---------------------------------------------------------------------------
# Sol-level tick
# ---------------------------------------------------------------------------

def tick_thermal(
    habitat: HabitatThermal,
    radiator: Radiator,
    exterior_temp_c: float,
    population: int,
    solar_kwh: float,
    nuclear_kwh: float,
    soce_kwh: float,
    active_heating_kwh: float = 0.0,
) -> dict:
    """Advance thermal system by one sol.

    Pipeline:
      1. Calculate heat sources (waste + metabolic + active heating)
      2. Calculate heat losses (conductive through walls)
      3. Calculate radiator rejection (if overheating)
      4. Net heat balance → temperature change
      5. Clamp interior temperature to physical bounds
      6. Calculate comfort score

    Args:
        habitat: habitat thermal state (mutated in place)
        radiator: heat rejection radiator
        exterior_temp_c: outside temperature this sol
        population: crew count
        solar_kwh: solar electrical generation this sol
        nuclear_kwh: nuclear electrical generation this sol
        soce_kwh: SOCE power consumption this sol
        active_heating_kwh: additional electrical heating applied

    Returns:
        Snapshot dict with all thermal metrics.
    """
    # 1. Heat sources
    meta_heat = metabolic_heat_kwh(population)
    sys_waste = waste_heat_kwh(solar_kwh, nuclear_kwh, soce_kwh)
    total_heat_in = meta_heat + sys_waste + max(0.0, active_heating_kwh)

    # 2. Conductive losses
    cond_loss = conductive_loss_kwh(
        habitat.interior_temp_c, exterior_temp_c,
        habitat.wall_area_m2, habitat.insulation_r,
    )

    # 3. Radiator rejection (only active when interior > target + tolerance)
    rad_rejection = 0.0
    if habitat.interior_temp_c > HABITAT_TARGET_TEMP_C + HABITAT_TEMP_TOLERANCE_C:
        rad_rejection = radiative_rejection_kwh(radiator, habitat.interior_temp_c)

    # 4. Net heat balance
    total_loss = cond_loss + rad_rejection
    net_heat = total_heat_in - total_loss

    # 5. Temperature change
    delta_t = temperature_delta(net_heat, habitat.thermal_mass_kwh_c)
    old_temp = habitat.interior_temp_c
    habitat.interior_temp_c += delta_t

    # Physical bounds: interior cannot go below exterior (no free cooling
    # below ambient) and cannot exceed reasonable upper bound
    habitat.interior_temp_c = max(exterior_temp_c, habitat.interior_temp_c)
    habitat.interior_temp_c = min(80.0, habitat.interior_temp_c)

    # 6. Comfort
    comfort = comfort_score(habitat.interior_temp_c)

    # Heating needed to return to target
    heating_needed = heating_power_needed(
        habitat, exterior_temp_c, sys_waste, meta_heat,
    )

    return {
        "interior_temp_c": round(habitat.interior_temp_c, 2),
        "exterior_temp_c": round(exterior_temp_c, 2),
        "temp_delta_c": round(habitat.interior_temp_c - old_temp, 2),
        "metabolic_heat_kwh": round(meta_heat, 4),
        "waste_heat_kwh": round(sys_waste, 4),
        "active_heating_kwh": round(max(0.0, active_heating_kwh), 4),
        "total_heat_in_kwh": round(total_heat_in, 4),
        "conductive_loss_kwh": round(cond_loss, 4),
        "radiator_rejection_kwh": round(rad_rejection, 4),
        "net_heat_kwh": round(net_heat, 4),
        "comfort_score": round(comfort, 4),
        "heating_needed_kwh": round(heating_needed, 4),
        "pipe_freeze_risk": habitat.interior_temp_c <= HABITAT_MIN_TEMP_C,
        "heat_stroke_risk": habitat.interior_temp_c >= HABITAT_MAX_TEMP_C,
    }
