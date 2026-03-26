"""rover.py -- Mars Surface Rover Simulation.

Models an autonomous Mars rover for colony support operations.
Each tick = 1 sol of rover activity.

Physics modelled
----------------
* **Solar power** -- panel area, dust accumulation, seasonal irradiance.
  Mars receives ~590 W/m^2 at perihelion, ~490 W/m^2 at aphelion.
  Dust reduces panel efficiency; cleaning events partially restore it.
* **Battery** -- Li-ion bank with round-trip efficiency, self-discharge,
  and temperature-dependent capacity.  Cold Mars nights reduce capacity.
* **Drivetrain** -- energy cost scales with terrain slope, regolith type,
  and cargo mass.  Speed is limited by power budget and terrain.
* **Navigation** -- great-circle distance on Mars (radius 3389.5 km).
  Bearing and distance between waypoints.  Odometry tracks total travel.
* **Thermal** -- RTG/heater keeps electronics warm overnight.  Cold soak
  degrades battery and can halt the rover if heater fails.
* **Wear** -- wheels, motors, and sensors degrade with distance travelled.
  Maintenance at the colony restores health.
* **Sample collection** -- the rover can collect regolith/rock samples
  up to its cargo capacity.  Samples have mass that affects drivetrain.

Reference vehicles:
  - Curiosity: 900 kg, RTG, ~100 m/sol (variable)
  - Perseverance: 1025 kg, RTG, ~200 m/sol
  - Spirit/Opportunity: 185 kg, solar, ~100 m/sol max
  - Colony rover (this model): ~400 kg, solar+battery, ~2 km/sol cruise
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# -- Physical constants -------------------------------------------------------
MARS_RADIUS_KM = 3389.5
MARS_GRAVITY_M_S2 = 3.72
MARS_SOL_HOURS = 24.66

# Solar
SOLAR_FLUX_PERIHELION_W_M2 = 590.0
SOLAR_FLUX_APHELION_W_M2 = 490.0
PANEL_AREA_M2 = 4.0
PANEL_EFFICIENCY = 0.22
DUST_ACCUMULATION_PER_SOL = 0.002
DUST_STORM_ACCUMULATION = 0.05
CLEANING_RESTORE_FRACTION = 0.8

# Battery
BATTERY_CAPACITY_WH = 3000.0
BATTERY_ROUND_TRIP_EFF = 0.92
BATTERY_SELF_DISCHARGE_PER_SOL = 0.005
COLD_CAPACITY_PENALTY = 0.15

# Drivetrain
ROVER_MASS_KG = 400.0
DRIVE_EFFICIENCY = 0.35
ROLLING_RESISTANCE = 0.08
MAX_SPEED_KM_SOL = 5.0
MIN_DRIVE_POWER_W = 20.0

# Thermal
HEATER_POWER_W = 40.0
HEATER_HOURS_PER_SOL = 12.0
ELECTRONICS_MIN_TEMP_C = -40.0
MARS_NIGHT_TEMP_C = -90.0

# Wear
WHEEL_LIFE_KM = 50.0
SENSOR_LIFE_SOLS = 2000
MAINTENANCE_RESTORE = 0.8

# Samples
MAX_SAMPLE_CAPACITY_KG = 100.0
SAMPLE_COLLECT_ENERGY_WH = 50.0


# -- Data structures ----------------------------------------------------------

@dataclass
class RoverState:
    """Mutable state of a Mars surface rover."""
    sol: int = 0
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    battery_wh: float = BATTERY_CAPACITY_WH * 0.8
    dust_factor: float = 0.0
    wheel_wear: float = 0.0
    sensor_wear: float = 0.0
    total_distance_km: float = 0.0
    total_samples_kg: float = 0.0
    cargo_kg: float = 0.0
    operational: bool = True
    total_energy_generated_wh: float = 0.0
    total_energy_consumed_wh: float = 0.0

    def __post_init__(self) -> None:
        self.battery_wh = max(0.0, min(self.battery_wh, BATTERY_CAPACITY_WH))
        self.dust_factor = max(0.0, min(self.dust_factor, 1.0))
        self.wheel_wear = max(0.0, min(self.wheel_wear, 1.0))
        self.sensor_wear = max(0.0, min(self.sensor_wear, 1.0))
        self.cargo_kg = max(0.0, min(self.cargo_kg, MAX_SAMPLE_CAPACITY_KG))
        self.total_distance_km = max(0.0, self.total_distance_km)
        self.total_samples_kg = max(0.0, self.total_samples_kg)
        self.total_energy_generated_wh = max(0.0, self.total_energy_generated_wh)
        self.total_energy_consumed_wh = max(0.0, self.total_energy_consumed_wh)


@dataclass
class RoverSol:
    """Output of one sol of rover simulation."""
    sol: int = 0
    energy_generated_wh: float = 0.0
    energy_consumed_wh: float = 0.0
    distance_km: float = 0.0
    samples_collected_kg: float = 0.0
    speed_km_sol: float = 0.0
    battery_level_wh: float = 0.0
    dust_factor: float = 0.0
    wheel_health: float = 1.0
    sensor_health: float = 1.0
    warnings: List[str] = field(default_factory=list)
    halted: bool = False


# -- Core simulation functions ------------------------------------------------

def mars_distance_km(
    lat1_deg: float, lon1_deg: float,
    lat2_deg: float, lon2_deg: float,
) -> float:
    """Great-circle distance between two points on Mars surface (km).

    Uses the Haversine formula with Mars radius.
    """
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    dlat = lat2 - lat1
    dlon = math.radians(lon2_deg - lon1_deg)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    a = min(1.0, max(0.0, a))
    c = 2 * math.asin(math.sqrt(a))
    return MARS_RADIUS_KM * c


def solar_power_wh(
    panel_area_m2: float,
    dust_factor: float,
    sol_of_year: int = 0,
) -> float:
    """Solar energy generated in one sol (watt-hours).

    Accounts for panel area, efficiency, dust, and seasonal variation.
    Mars orbital period ≈ 668.6 sols.
    """
    season = 0.5 + 0.5 * math.cos(2 * math.pi * sol_of_year / 668.6)
    flux = SOLAR_FLUX_APHELION_W_M2 + season * (SOLAR_FLUX_PERIHELION_W_M2 - SOLAR_FLUX_APHELION_W_M2)
    dust_mult = max(0.0, 1.0 - dust_factor)
    daylight_hours = MARS_SOL_HOURS * 0.5
    return panel_area_m2 * PANEL_EFFICIENCY * flux * dust_mult * daylight_hours


def drive_energy_wh(
    distance_km: float,
    slope_deg: float = 0.0,
    cargo_kg: float = 0.0,
) -> float:
    """Energy required to drive a given distance (watt-hours).

    Accounts for slope, cargo mass, rolling resistance, and Mars gravity.
    """
    distance_km = max(0.0, distance_km)
    total_mass = ROVER_MASS_KG + max(0.0, cargo_kg)
    distance_m = distance_km * 1000.0

    grade_force = total_mass * MARS_GRAVITY_M_S2 * math.sin(math.radians(abs(slope_deg)))
    rolling_force = total_mass * MARS_GRAVITY_M_S2 * ROLLING_RESISTANCE
    total_force = rolling_force + grade_force

    mechanical_energy_j = total_force * distance_m
    mechanical_energy_wh = mechanical_energy_j / 3600.0

    if DRIVE_EFFICIENCY > 0:
        return mechanical_energy_wh / DRIVE_EFFICIENCY
    return 0.0


def max_range_km(
    available_wh: float,
    slope_deg: float = 0.0,
    cargo_kg: float = 0.0,
) -> float:
    """Maximum distance the rover can drive with available energy.

    Inverse of drive_energy_wh.
    """
    if available_wh <= 0:
        return 0.0
    energy_per_km = drive_energy_wh(1.0, slope_deg, cargo_kg)
    if energy_per_km <= 0:
        return 0.0
    return min(MAX_SPEED_KM_SOL, available_wh / energy_per_km)


def effective_battery_capacity(dust_factor: float) -> float:
    """Effective battery capacity accounting for cold Mars nights.

    Cold temperatures reduce Li-ion capacity by COLD_CAPACITY_PENALTY.
    More dust = less daytime solar heating = colder battery = more penalty.
    """
    cold_penalty = COLD_CAPACITY_PENALTY * (0.5 + 0.5 * dust_factor)
    return BATTERY_CAPACITY_WH * (1.0 - cold_penalty)


def tick_rover(
    rover: RoverState,
    drive_km: float = 0.0,
    slope_deg: float = 0.0,
    collect_samples_kg: float = 0.0,
    dust_storm: bool = False,
    cleaning_event: bool = False,
    maintenance: bool = False,
    sol_of_year: int = 0,
) -> RoverSol:
    """Advance the rover simulation by one sol.

    Parameters
    ----------
    rover : RoverState
        Mutable rover state (modified in place).
    drive_km : float
        Requested driving distance this sol.
    slope_deg : float
        Average terrain slope (degrees).
    collect_samples_kg : float
        Mass of samples to attempt collecting.
    dust_storm : bool
        Whether a dust storm is occurring.
    cleaning_event : bool
        Whether panels are cleaned this sol.
    maintenance : bool
        Whether rover receives maintenance at colony.
    sol_of_year : int
        Day of Martian year for seasonal solar calculation.

    Returns
    -------
    RoverSol with this sol's metrics and warnings.
    """
    rover.sol += 1
    result = RoverSol(sol=rover.sol)
    warnings: List[str] = []

    if not rover.operational:
        result.halted = True
        warnings.append("ROVER_OFFLINE: Not operational")
        result.warnings = warnings
        return result

    # -- Maintenance --
    if maintenance:
        rover.wheel_wear = max(0.0, rover.wheel_wear - MAINTENANCE_RESTORE * rover.wheel_wear)
        rover.sensor_wear = max(0.0, rover.sensor_wear - MAINTENANCE_RESTORE * rover.sensor_wear)

    # -- Dust --
    if dust_storm:
        rover.dust_factor = min(1.0, rover.dust_factor + DUST_STORM_ACCUMULATION)
        warnings.append("DUST_STORM: Severe solar reduction")
    else:
        rover.dust_factor = min(1.0, rover.dust_factor + DUST_ACCUMULATION_PER_SOL)

    if cleaning_event:
        rover.dust_factor *= (1.0 - CLEANING_RESTORE_FRACTION)

    # -- Solar generation --
    generated = solar_power_wh(PANEL_AREA_M2, rover.dust_factor, sol_of_year)
    result.energy_generated_wh = generated
    rover.total_energy_generated_wh += generated

    # -- Heater (overnight survival) --
    heater_wh = HEATER_POWER_W * HEATER_HOURS_PER_SOL
    energy_budget = rover.battery_wh + generated - heater_wh
    rover.battery_wh = max(0.0, rover.battery_wh + generated * BATTERY_ROUND_TRIP_EFF - heater_wh)

    if energy_budget < 0:
        warnings.append("LOW_POWER: Heater consuming reserves")

    # -- Self-discharge --
    discharge_loss = rover.battery_wh * BATTERY_SELF_DISCHARGE_PER_SOL
    rover.battery_wh = max(0.0, rover.battery_wh - discharge_loss)

    # -- Cap battery at effective capacity --
    eff_cap = effective_battery_capacity(rover.dust_factor)
    rover.battery_wh = min(rover.battery_wh, eff_cap)

    # -- Driving --
    drive_km = max(0.0, drive_km)
    actual_drive = 0.0
    drive_consumed = 0.0

    if drive_km > 0 and rover.battery_wh > MIN_DRIVE_POWER_W:
        wheel_penalty = 1.0 + rover.wheel_wear * 2.0
        requested_energy = drive_energy_wh(drive_km, slope_deg, rover.cargo_kg) * wheel_penalty
        available_for_drive = max(0.0, rover.battery_wh - MIN_DRIVE_POWER_W)

        if requested_energy <= available_for_drive:
            actual_drive = drive_km
            drive_consumed = requested_energy
        else:
            fraction = available_for_drive / requested_energy if requested_energy > 0 else 0
            actual_drive = drive_km * fraction
            drive_consumed = available_for_drive

        actual_drive = min(actual_drive, MAX_SPEED_KM_SOL)
        rover.battery_wh = max(0.0, rover.battery_wh - drive_consumed)
        rover.total_distance_km += actual_drive

        # Wheel wear
        if WHEEL_LIFE_KM > 0:
            rover.wheel_wear = min(1.0, rover.wheel_wear + actual_drive / WHEEL_LIFE_KM)

    result.distance_km = actual_drive
    result.speed_km_sol = actual_drive

    # -- Sample collection --
    collected = 0.0
    if collect_samples_kg > 0 and rover.battery_wh > SAMPLE_COLLECT_ENERGY_WH:
        capacity_remaining = MAX_SAMPLE_CAPACITY_KG - rover.cargo_kg
        collected = min(collect_samples_kg, capacity_remaining)
        if collected > 0:
            rover.battery_wh = max(0.0, rover.battery_wh - SAMPLE_COLLECT_ENERGY_WH)
            rover.cargo_kg += collected
            rover.total_samples_kg += collected
            drive_consumed += SAMPLE_COLLECT_ENERGY_WH

    result.samples_collected_kg = collected

    # -- Sensor wear --
    if SENSOR_LIFE_SOLS > 0:
        rover.sensor_wear = min(1.0, rover.sensor_wear + 1.0 / SENSOR_LIFE_SOLS)

    # -- Total energy consumed --
    total_consumed = heater_wh + drive_consumed + discharge_loss
    result.energy_consumed_wh = total_consumed
    rover.total_energy_consumed_wh += total_consumed

    # -- Warnings --
    if rover.battery_wh < BATTERY_CAPACITY_WH * 0.2:
        warnings.append("LOW_BATTERY: %.0f Wh remaining" % rover.battery_wh)
    if rover.wheel_wear > 0.8:
        warnings.append("WHEEL_WEAR: %.0f%% degraded" % (rover.wheel_wear * 100))
    if rover.sensor_wear > 0.8:
        warnings.append("SENSOR_WEAR: %.0f%% degraded" % (rover.sensor_wear * 100))
    if rover.dust_factor > 0.5:
        warnings.append("HIGH_DUST: %.0f%% panel coverage" % (rover.dust_factor * 100))

    # -- Check operational status --
    if rover.wheel_wear >= 1.0:
        rover.operational = False
        warnings.append("WHEELS_FAILED: Rover immobilized")
    if rover.battery_wh <= 0 and generated < heater_wh:
        rover.operational = False
        warnings.append("FROZEN: Battery depleted, cannot heat electronics")

    result.battery_level_wh = rover.battery_wh
    result.dust_factor = rover.dust_factor
    result.wheel_health = 1.0 - rover.wheel_wear
    result.sensor_health = 1.0 - rover.sensor_wear
    result.warnings = warnings
    result.halted = not rover.operational
    return result


def create_rover(strategy: str = "explorer") -> RoverState:
    """Create a rover configured for a given mission profile."""
    configs = {
        "explorer": RoverState(battery_wh=BATTERY_CAPACITY_WH * 0.9),
        "hauler": RoverState(battery_wh=BATTERY_CAPACITY_WH, cargo_kg=0.0),
        "scout": RoverState(battery_wh=BATTERY_CAPACITY_WH * 0.7),
    }
    return configs.get(strategy, configs["explorer"])
