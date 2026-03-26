"""wind_turbine.py — Mars Wind Turbine Power Generation

Mars atmosphere is ~1/100th Earth's density, so wind turbines produce
far less power than on Earth.  But they fill a critical niche: during
dust storms, when solar panels are useless (τ > 3), wind speeds climb
to 60-100 km/h — exactly when the colony needs backup power most.

Physics
-------
* **Wind power**: P = 0.5 · ρ · A · v³ · Cp
  - ρ (Mars air density): ~0.020 kg/m³ (vs Earth 1.225 kg/m³)
  - A (swept area): π · r² for rotor radius r
  - v (wind speed): Mars surface 2-7 m/s calm, 20-30 m/s in storms
  - Cp (power coefficient): Betz limit 16/27 ≈ 0.593; practical ≈ 0.35
* **Betz limit**: No turbine can extract more than 59.3% of wind kinetic
  energy — this is a thermodynamic law, same on Mars as Earth.
* **Cut-in / cut-out speeds**: Turbine needs minimum wind to start (cut-in),
  and must feather/brake above structural limit (cut-out).
  - Cut-in: ~3 m/s (designed for thin Mars air)
  - Rated: ~25 m/s (peak efficiency)
  - Cut-out: ~45 m/s (structural protection; Mars gusts can exceed this)
* **Air density varies** with temperature and pressure:
  ρ = P / (R_specific · T)
  Mars: ~600 Pa, ~210 K → ρ ≈ 0.020 kg/m³ (±30% seasonal)
* **Dust erosion**: Mars dust at high speed erodes blade surfaces,
  reducing aerodynamic efficiency ~0.01%/sol during storms.
* **Capacity factor**: On Mars, expect 15-25% (vs Earth onshore ~30-35%).
  Higher during dust storm season (Ls 200-330).

References:
  James & Borer — "Wind Energy for Mars" (2018 AIAA)
  Holstein-Rathlou et al. — InSight meteorology (2018)
  Delgado-Bonal et al. — "Mars atmospheric density" (Icarus 2020)

One tick = one sol.  Power in kW, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MARS_AIR_DENSITY_KG_M3 = 0.020       # surface mean (CO₂ atmosphere, 600 Pa, 210 K)
MARS_GAS_CONSTANT_CO2 = 188.9        # J/(kg·K) specific gas constant for CO₂
MARS_SURFACE_PRESSURE_PA = 610.0     # mean surface pressure (Pa)
MARS_MEAN_TEMP_K = 210.0             # mean surface temperature (K)

BETZ_LIMIT = 16.0 / 27.0             # ≈ 0.593, max extractable fraction
PRACTICAL_CP = 0.35                   # realistic power coefficient for Mars turbine
GENERATOR_EFFICIENCY = 0.92           # electrical generator efficiency
GEARBOX_EFFICIENCY = 0.95             # mechanical gearbox efficiency

CUT_IN_SPEED_M_S = 3.0               # minimum wind to generate power
RATED_SPEED_M_S = 25.0               # wind speed at rated (peak) power
CUT_OUT_SPEED_M_S = 45.0             # structural limit — turbine feathers

BLADE_EROSION_RATE_PER_SOL = 0.0001  # efficiency loss per sol during dust storm
BLADE_EROSION_RATE_CALM = 0.00001    # efficiency loss per sol in calm weather
MAX_BLADE_EROSION = 0.30             # max cumulative erosion (30% loss, then replace)

SOL_HOURS = 24.66                     # Mars sol in hours

# Maintenance
MAINTENANCE_INTERVAL_SOLS = 90        # check every ~90 sols
MAINTENANCE_EFFICIENCY_RESTORE = 0.05 # restores 5% of lost blade efficiency


# ---------------------------------------------------------------------------
# Air density model
# ---------------------------------------------------------------------------

def air_density(pressure_pa: float, temperature_k: float) -> float:
    """Compute Mars air density from pressure and temperature.

    Uses ideal gas law:  ρ = P / (R_specific · T)
    where R_specific = 188.9 J/(kg·K) for CO₂-dominated Mars atmosphere.

    Returns density in kg/m³, clamped to physical bounds.
    """
    if pressure_pa <= 0.0 or temperature_k <= 0.0:
        return 0.0
    rho = pressure_pa / (MARS_GAS_CONSTANT_CO2 * temperature_k)
    return max(0.0, min(rho, 1.0))  # Mars air can't exceed 1 kg/m³


# ---------------------------------------------------------------------------
# Turbine geometry
# ---------------------------------------------------------------------------

def swept_area(rotor_radius_m: float) -> float:
    """Swept area of turbine rotor in m²."""
    if rotor_radius_m <= 0.0:
        return 0.0
    return math.pi * rotor_radius_m ** 2


# ---------------------------------------------------------------------------
# Power curve
# ---------------------------------------------------------------------------

def wind_power_available_w(density_kg_m3: float, area_m2: float,
                           wind_speed_m_s: float) -> float:
    """Total kinetic power in wind passing through swept area (W).

    P_wind = 0.5 · ρ · A · v³
    """
    if density_kg_m3 <= 0.0 or area_m2 <= 0.0 or wind_speed_m_s <= 0.0:
        return 0.0
    return 0.5 * density_kg_m3 * area_m2 * wind_speed_m_s ** 3


def turbine_power_output_w(wind_speed_m_s: float, rotor_radius_m: float,
                           density_kg_m3: float = MARS_AIR_DENSITY_KG_M3,
                           cp: float = PRACTICAL_CP,
                           blade_erosion: float = 0.0) -> float:
    """Electrical power output from a single turbine (W).

    Applies cut-in/cut-out limits, Betz-capped Cp, generator and
    gearbox losses, and blade erosion degradation.

    Between cut-in and rated speed: power follows cubic wind law.
    Between rated and cut-out: power is capped at rated output.
    Above cut-out: turbine feathers, output is zero.
    """
    if wind_speed_m_s < CUT_IN_SPEED_M_S or wind_speed_m_s > CUT_OUT_SPEED_M_S:
        return 0.0

    cp_effective = min(cp, BETZ_LIMIT) * (1.0 - min(blade_erosion, MAX_BLADE_EROSION))
    area = swept_area(rotor_radius_m)

    if wind_speed_m_s <= RATED_SPEED_M_S:
        p_aero = wind_power_available_w(density_kg_m3, area, wind_speed_m_s)
    else:
        # Capped at rated power (pitch-controlled)
        p_aero = wind_power_available_w(density_kg_m3, area, RATED_SPEED_M_S)

    p_shaft = p_aero * cp_effective * GEARBOX_EFFICIENCY
    p_elec = p_shaft * GENERATOR_EFFICIENCY
    return max(0.0, p_elec)


def rated_power_w(rotor_radius_m: float,
                  density_kg_m3: float = MARS_AIR_DENSITY_KG_M3,
                  cp: float = PRACTICAL_CP) -> float:
    """Nameplate rated power output at rated wind speed (W)."""
    return turbine_power_output_w(RATED_SPEED_M_S, rotor_radius_m,
                                  density_kg_m3, cp, blade_erosion=0.0)


# ---------------------------------------------------------------------------
# Capacity factor
# ---------------------------------------------------------------------------

def capacity_factor(power_output_w: float, rated_w: float) -> float:
    """Instantaneous capacity factor: actual / rated output."""
    if rated_w <= 0.0:
        return 0.0
    return max(0.0, min(1.0, power_output_w / rated_w))


# ---------------------------------------------------------------------------
# Turbine state
# ---------------------------------------------------------------------------

@dataclass
class WindTurbine:
    """State of a single Mars wind turbine.

    rotor_radius_m: rotor blade radius (metres)
    hub_height_m: height of hub above ground (metres)
    blade_erosion: cumulative blade erosion [0, MAX_BLADE_EROSION]
    sols_since_maintenance: sols since last maintenance check
    total_energy_kwh: cumulative energy produced (kWh)
    operational: whether turbine is running (not feathered/broken)
    sol: current sol counter
    """
    rotor_radius_m: float = 5.0
    hub_height_m: float = 15.0
    blade_erosion: float = 0.0
    sols_since_maintenance: int = 0
    total_energy_kwh: float = 0.0
    operational: bool = True
    sol: int = 0

    def __post_init__(self) -> None:
        """Clamp fields to valid physical ranges."""
        self.rotor_radius_m = max(0.1, self.rotor_radius_m)
        self.hub_height_m = max(1.0, self.hub_height_m)
        self.blade_erosion = max(0.0, min(MAX_BLADE_EROSION, self.blade_erosion))
        self.total_energy_kwh = max(0.0, self.total_energy_kwh)
        self.sol = max(0, self.sol)


@dataclass
class WindConditions:
    """Wind and atmospheric conditions for a single sol.

    wind_speed_m_s: average wind speed at hub height (m/s)
    pressure_pa: atmospheric pressure (Pa)
    temperature_k: air temperature (K)
    dust_storm_active: whether a dust storm is in progress
    """
    wind_speed_m_s: float = 5.0
    pressure_pa: float = MARS_SURFACE_PRESSURE_PA
    temperature_k: float = MARS_MEAN_TEMP_K
    dust_storm_active: bool = False

    def __post_init__(self) -> None:
        """Clamp to physical bounds."""
        self.wind_speed_m_s = max(0.0, self.wind_speed_m_s)
        self.pressure_pa = max(0.0, self.pressure_pa)
        self.temperature_k = max(0.0, self.temperature_k)


@dataclass
class TickResult:
    """Result of one sol tick for a wind turbine.

    power_w: average electrical power output (W)
    energy_kwh: energy produced this sol (kWh)
    capacity_factor: fraction of rated power achieved
    air_density_kg_m3: computed air density for this sol
    blade_erosion_delta: erosion accumulated this sol
    maintenance_performed: whether maintenance happened this sol
    feathered: whether turbine was feathered (wind > cut-out)
    """
    power_w: float = 0.0
    energy_kwh: float = 0.0
    capacity_factor: float = 0.0
    air_density_kg_m3: float = 0.0
    blade_erosion_delta: float = 0.0
    maintenance_performed: bool = False
    feathered: bool = False


# ---------------------------------------------------------------------------
# Wind shear (height correction)
# ---------------------------------------------------------------------------

def wind_at_height(reference_speed_m_s: float, reference_height_m: float,
                   target_height_m: float, alpha: float = 0.20) -> float:
    """Adjust wind speed for height using power law wind profile.

    v(z) = v_ref · (z / z_ref)^α

    Mars wind shear exponent α ≈ 0.20 (less boundary layer turbulence
    than Earth due to lower density and smoother terrain in many areas).
    """
    if reference_speed_m_s <= 0.0 or reference_height_m <= 0.0 or target_height_m <= 0.0:
        return 0.0
    return reference_speed_m_s * (target_height_m / reference_height_m) ** alpha


# ---------------------------------------------------------------------------
# Tick function
# ---------------------------------------------------------------------------

def tick(turbine: WindTurbine, conditions: WindConditions,
         reference_wind_height_m: float = 2.0) -> TickResult:
    """Advance turbine by one sol.

    1. Compute air density from pressure/temperature.
    2. Adjust wind speed from reference height to hub height.
    3. Compute power output (with erosion).
    4. Accumulate blade erosion.
    5. Check maintenance schedule.

    Returns TickResult with all outputs.
    """
    result = TickResult()

    # Air density for this sol
    rho = air_density(conditions.pressure_pa, conditions.temperature_k)
    result.air_density_kg_m3 = rho

    if not turbine.operational:
        turbine.sol += 1
        return result

    # Wind at hub height
    hub_wind = wind_at_height(
        conditions.wind_speed_m_s, reference_wind_height_m,
        turbine.hub_height_m
    )

    # Feather check
    if hub_wind > CUT_OUT_SPEED_M_S:
        result.feathered = True
        turbine.sol += 1
        turbine.sols_since_maintenance += 1
        return result

    # Power output
    power = turbine_power_output_w(
        hub_wind, turbine.rotor_radius_m, rho,
        PRACTICAL_CP, turbine.blade_erosion
    )
    result.power_w = power

    # Energy for the sol
    energy = power * SOL_HOURS / 1000.0  # W · h → kWh
    result.energy_kwh = energy
    turbine.total_energy_kwh += energy

    # Capacity factor
    rated = rated_power_w(turbine.rotor_radius_m, rho)
    result.capacity_factor = capacity_factor(power, rated)

    # Blade erosion
    if conditions.dust_storm_active:
        erosion_delta = BLADE_EROSION_RATE_PER_SOL
    else:
        erosion_delta = BLADE_EROSION_RATE_CALM
    result.blade_erosion_delta = erosion_delta
    turbine.blade_erosion = min(
        MAX_BLADE_EROSION, turbine.blade_erosion + erosion_delta
    )

    # Maintenance
    turbine.sols_since_maintenance += 1
    if turbine.sols_since_maintenance >= MAINTENANCE_INTERVAL_SOLS:
        restored = min(turbine.blade_erosion, MAINTENANCE_EFFICIENCY_RESTORE)
        turbine.blade_erosion = max(0.0, turbine.blade_erosion - restored)
        turbine.sols_since_maintenance = 0
        result.maintenance_performed = True

    turbine.sol += 1
    return result


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_turbine(rotor_radius_m: float = 5.0,
                   hub_height_m: float = 15.0) -> WindTurbine:
    """Create a fresh wind turbine with default Mars specs."""
    return WindTurbine(rotor_radius_m=rotor_radius_m,
                       hub_height_m=hub_height_m)


def create_storm_conditions(wind_speed_m_s: float = 25.0) -> WindConditions:
    """Create dust storm wind conditions (high wind, standard pressure)."""
    return WindConditions(
        wind_speed_m_s=wind_speed_m_s,
        dust_storm_active=True,
    )


def create_calm_conditions(wind_speed_m_s: float = 5.0) -> WindConditions:
    """Create calm weather conditions."""
    return WindConditions(wind_speed_m_s=wind_speed_m_s)


# ---------------------------------------------------------------------------
# Multi-turbine farm
# ---------------------------------------------------------------------------

@dataclass
class WindFarm:
    """A collection of wind turbines forming a power farm.

    turbines: list of individual WindTurbine instances
    wake_loss_fraction: power loss from turbine wake interference [0, 1]
    """
    turbines: list = field(default_factory=list)
    wake_loss_fraction: float = 0.10  # 10% wake losses typical

    def __post_init__(self) -> None:
        self.wake_loss_fraction = max(0.0, min(0.5, self.wake_loss_fraction))

    def total_rated_power_w(self, density: float = MARS_AIR_DENSITY_KG_M3) -> float:
        """Sum of rated power for all turbines (W)."""
        return sum(rated_power_w(t.rotor_radius_m, density)
                   for t in self.turbines)

    def tick_all(self, conditions: WindConditions,
                 reference_wind_height_m: float = 2.0) -> list[TickResult]:
        """Tick all turbines, apply wake losses to total output."""
        results = []
        for t in self.turbines:
            r = tick(t, conditions, reference_wind_height_m)
            # Apply wake loss
            r.power_w *= (1.0 - self.wake_loss_fraction)
            r.energy_kwh *= (1.0 - self.wake_loss_fraction)
            results.append(r)
        return results

    def total_power_w(self, results: list[TickResult]) -> float:
        """Total farm power from tick results."""
        return sum(r.power_w for r in results)

    def total_energy_kwh(self, results: list[TickResult]) -> float:
        """Total farm energy from tick results."""
        return sum(r.energy_kwh for r in results)


def create_wind_farm(num_turbines: int = 4,
                     rotor_radius_m: float = 5.0,
                     hub_height_m: float = 15.0) -> WindFarm:
    """Create a wind farm with identical turbines."""
    turbines = [create_turbine(rotor_radius_m, hub_height_m)
                for _ in range(max(1, num_turbines))]
    return WindFarm(turbines=turbines)
