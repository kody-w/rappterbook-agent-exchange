"""airlock.py -- Mars Colony Airlock Simulation.

Models the pressure-cycling airlock connecting a pressurized habitat to
the Martian surface.  Each tick = 1 sol of airlock operations.

Physics modelled
----------------
* **Pressure cycling** -- pump-down from habitat pressure (~101.3 kPa) to
  Mars ambient (~0.636 kPa) and re-pressurization.  Real pumps need
  multiple stages for a ~160:1 pressure ratio.
* **Air loss** -- each depressurization cycle vents some gas.  The pump
  recovers a fraction (pump_efficiency); the rest is lost to Mars.
  At 101.3 kPa and 293 K in a 12 m³ chamber:
    PV = nRT  →  n ≈ 500 mol  →  ~14.5 kg of air per full chamber.
  A 95 % efficient pump loses ~0.73 kg per cycle.
* **Dust ingress** -- every return cycle drags Martian regolith particles
  into the habitat.  Dust mass scales with number of crew, EVA duration,
  and surface conditions.  Electrostatic perchlorate dust is toxic.
* **Seal wear** -- rubber/silicone seals degrade with thermal cycling
  (ΔT ≈ 180 K between habitat and Mars surface).  Leak rate increases
  with wear.
* **Power consumption** -- vacuum pumps, lighting, heating, sensors.
  Proportional to cycles per sol and chamber volume.
* **Thermal management** -- the chamber cools during depressurization
  (adiabatic expansion) and must be reheated before crew re-entry.
* **Cycle time** -- pump-down takes longer than repressurization
  (expanding into vacuum vs. filling from pressurized supply).
  Typical: 15-30 min depressurize, 5-10 min repressurize.

Reference:
  - ISS Quest airlock: 5.5 m³, ~30 min depress, ~5 min repress
  - Mars habitat: bigger chamber (12 m³), similar pump tech
  - Mars ambient: 0.636 kPa avg (CO₂), varies ±26 % with season
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

# -- Physical constants -------------------------------------------------------

# Pressures (kPa)
HABITAT_PRESSURE_KPA = 101.3          # Earth-normal (N₂/O₂ mix)
MARS_AMBIENT_KPA = 0.636              # Mars surface average (CO₂)
SAFE_OPEN_THRESHOLD_KPA = 1.0         # Max delta-P for outer door opening

# Chamber
CHAMBER_VOLUME_M3 = 12.0             # Airlock internal volume
AIR_DENSITY_KG_M3 = 1.225            # At habitat conditions (101.3 kPa, 20°C)
AIR_MOLAR_MASS_KG = 0.029            # Weighted avg N₂/O₂

# Pump
PUMP_EFFICIENCY = 0.95                # Fraction of air recovered during depress
PUMP_POWER_KW = 2.5                   # Vacuum pump power draw
DEPRESS_TIME_MIN = 25.0               # Full depressurization time
REPRESS_TIME_MIN = 8.0                # Full repressurization time

# Thermal
HABITAT_TEMP_C = 20.0
MARS_SURFACE_TEMP_C = -60.0           # Average daytime
ADIABATIC_COOLING_C_PER_CYCLE = 15.0  # Chamber cooling during depress
HEATER_POWER_KW = 1.5                 # Chamber heater
HEATER_TIME_MIN = 10.0                # Time to reheat chamber

# Dust
DUST_PER_PERSON_PER_EVA_G = 50.0      # Grams of regolith per returning crew
PERCHLORATE_FRACTION = 0.005          # ClO₄⁻ mass fraction in Mars dust
DUST_FILTER_EFFICIENCY = 0.85         # HEPA-equivalent filter capture rate
FILTER_LIFE_CYCLES = 500              # Cycles before filter replacement needed

# Seals
SEAL_LIFE_CYCLES = 2000               # Thermal cycles before seal failure
SEAL_LEAK_RATE_BASE_KPA_HR = 0.01     # Baseline leak rate (new seals)
SEAL_LEAK_RATE_MAX_KPA_HR = 1.0       # Leak rate at full wear
MAINTENANCE_SEAL_RESTORE = 0.7        # Fraction of wear removed by maintenance

# Operations
MAX_CYCLES_PER_SOL = 10               # Safety limit on daily cycles
MAX_CREW_PER_CYCLE = 4                # Chamber capacity
MARS_SOL_HOURS = 24.66                # Sol length


# -- Data structures ----------------------------------------------------------

@dataclass
class AirlockState:
    """Mutable state of a Mars colony airlock."""

    sol: int = 0
    pressure_kpa: float = HABITAT_PRESSURE_KPA
    chamber_temp_c: float = HABITAT_TEMP_C
    seal_wear: float = 0.0             # 0.0 = new, 1.0 = failed
    filter_wear: float = 0.0           # 0.0 = new, 1.0 = needs replacement
    total_cycles: int = 0
    total_air_lost_kg: float = 0.0
    total_dust_ingress_g: float = 0.0
    total_perchlorate_g: float = 0.0
    total_energy_kwh: float = 0.0
    operational: bool = True
    inner_door_open: bool = False      # Habitat side
    outer_door_open: bool = False      # Mars surface side

    def __post_init__(self) -> None:
        self.pressure_kpa = max(0.0, min(self.pressure_kpa, HABITAT_PRESSURE_KPA * 1.1))
        self.seal_wear = max(0.0, min(self.seal_wear, 1.0))
        self.filter_wear = max(0.0, min(self.filter_wear, 1.0))
        self.total_air_lost_kg = max(0.0, self.total_air_lost_kg)
        self.total_dust_ingress_g = max(0.0, self.total_dust_ingress_g)
        self.total_perchlorate_g = max(0.0, self.total_perchlorate_g)
        self.total_energy_kwh = max(0.0, self.total_energy_kwh)
        self.total_cycles = max(0, self.total_cycles)


@dataclass
class CycleResult:
    """Output of a single airlock depressurize→open→close→repressurize cycle."""

    cycle_number: int = 0
    air_lost_kg: float = 0.0
    dust_ingress_g: float = 0.0
    perchlorate_ingress_g: float = 0.0
    energy_kwh: float = 0.0
    cycle_time_min: float = 0.0
    leak_rate_kpa_hr: float = 0.0
    warnings: List[str] = field(default_factory=list)
    aborted: bool = False


@dataclass
class AirlockSol:
    """Aggregated output for one sol of airlock operations."""

    sol: int = 0
    cycles_completed: int = 0
    total_air_lost_kg: float = 0.0
    total_dust_ingress_g: float = 0.0
    total_perchlorate_g: float = 0.0
    total_energy_kwh: float = 0.0
    seal_health: float = 1.0
    filter_health: float = 1.0
    leak_rate_kpa_hr: float = 0.0
    warnings: List[str] = field(default_factory=list)
    halted: bool = False


# -- Core simulation functions ------------------------------------------------

def air_mass_in_chamber(pressure_kpa: float, volume_m3: float = CHAMBER_VOLUME_M3) -> float:
    """Mass of air in the chamber at given pressure (kg).

    Uses ideal gas law: PV = nRT → m = PV × M / (RT)
    At T = 293 K, R = 8.314 J/(mol·K), M = 0.029 kg/mol
    """
    if pressure_kpa <= 0 or volume_m3 <= 0:
        return 0.0
    pressure_pa = pressure_kpa * 1000.0
    temp_k = 293.0  # Standard habitat temperature
    n_mol = (pressure_pa * volume_m3) / (8.314 * temp_k)
    return n_mol * AIR_MOLAR_MASS_KG


def air_lost_per_cycle(pump_efficiency: float = PUMP_EFFICIENCY) -> float:
    """Air lost to Mars atmosphere per depressurization cycle (kg).

    The pump recovers pump_efficiency fraction of chamber air.
    The remainder escapes when the outer door opens.
    """
    total_air = air_mass_in_chamber(HABITAT_PRESSURE_KPA)
    return total_air * (1.0 - max(0.0, min(1.0, pump_efficiency)))


def leak_rate(seal_wear: float) -> float:
    """Current leak rate in kPa/hr based on seal condition.

    Linear interpolation from base to max leak rate.
    """
    wear = max(0.0, min(1.0, seal_wear))
    return SEAL_LEAK_RATE_BASE_KPA_HR + wear * (SEAL_LEAK_RATE_MAX_KPA_HR - SEAL_LEAK_RATE_BASE_KPA_HR)


def cycle_energy_kwh() -> float:
    """Total energy for one complete airlock cycle (kWh).

    Pump (depress) + pump (repress) + heater (reheat chamber).
    """
    pump_depress = PUMP_POWER_KW * (DEPRESS_TIME_MIN / 60.0)
    pump_repress = PUMP_POWER_KW * (REPRESS_TIME_MIN / 60.0)
    heater = HEATER_POWER_KW * (HEATER_TIME_MIN / 60.0)
    return pump_depress + pump_repress + heater


def cycle_time_min() -> float:
    """Total wall-clock time for one complete cycle (minutes).

    Depressurize + repressurize + reheat.
    """
    return DEPRESS_TIME_MIN + REPRESS_TIME_MIN + HEATER_TIME_MIN


def dust_ingress_g(crew_count: int, dust_storm: bool = False) -> float:
    """Dust entering habitat per cycle (grams).

    Scales with crew count.  Dust storm doubles ingress.
    Filter captures DUST_FILTER_EFFICIENCY fraction.
    """
    crew = max(0, min(crew_count, MAX_CREW_PER_CYCLE))
    raw_dust = crew * DUST_PER_PERSON_PER_EVA_G
    if dust_storm:
        raw_dust *= 2.0
    filtered = raw_dust * (1.0 - DUST_FILTER_EFFICIENCY)
    return filtered


def run_cycle(
    state: AirlockState,
    crew_count: int = 1,
    dust_storm: bool = False,
) -> CycleResult:
    """Execute one complete airlock cycle (egress or ingress).

    Steps:
    1. Close inner door (isolate from habitat)
    2. Depressurize chamber (pump recovers most air)
    3. Open outer door (crew exits/enters)
    4. Close outer door
    5. Repressurize chamber from habitat supply
    6. Open inner door (crew enters habitat)

    Modifies state in-place.  Returns CycleResult with metrics.
    """
    result = CycleResult()
    warnings: List[str] = []

    if not state.operational:
        result.aborted = True
        warnings.append("AIRLOCK_OFFLINE: Not operational")
        result.warnings = warnings
        return result

    if state.seal_wear >= 1.0:
        state.operational = False
        result.aborted = True
        warnings.append("SEAL_FAILURE: Seals completely degraded")
        result.warnings = warnings
        return result

    # -- Cycle counter --
    state.total_cycles += 1
    result.cycle_number = state.total_cycles

    # -- Close inner door, depressurize --
    state.inner_door_open = False
    air_lost = air_lost_per_cycle(PUMP_EFFICIENCY)
    state.total_air_lost_kg += air_lost
    result.air_lost_kg = air_lost
    state.pressure_kpa = MARS_AMBIENT_KPA

    # -- Chamber cools during depressurization --
    state.chamber_temp_c -= ADIABATIC_COOLING_C_PER_CYCLE

    # -- Open outer door (crew transit) --
    state.outer_door_open = True

    # -- Dust ingress (crew returning with dust) --
    dust = dust_ingress_g(crew_count, dust_storm)
    perchlorate = dust * PERCHLORATE_FRACTION
    state.total_dust_ingress_g += dust
    state.total_perchlorate_g += perchlorate
    result.dust_ingress_g = dust
    result.perchlorate_ingress_g = perchlorate

    # -- Close outer door, repressurize --
    state.outer_door_open = False
    state.pressure_kpa = HABITAT_PRESSURE_KPA

    # -- Reheat chamber --
    state.chamber_temp_c = HABITAT_TEMP_C

    # -- Open inner door --
    state.inner_door_open = True

    # -- Energy --
    energy = cycle_energy_kwh()
    state.total_energy_kwh += energy
    result.energy_kwh = energy

    # -- Cycle time --
    result.cycle_time_min = cycle_time_min()

    # -- Seal wear from thermal cycling --
    if SEAL_LIFE_CYCLES > 0:
        state.seal_wear = min(1.0, state.seal_wear + 1.0 / SEAL_LIFE_CYCLES)

    # -- Filter wear --
    if FILTER_LIFE_CYCLES > 0:
        state.filter_wear = min(1.0, state.filter_wear + 1.0 / FILTER_LIFE_CYCLES)

    # -- Leak rate --
    current_leak = leak_rate(state.seal_wear)
    result.leak_rate_kpa_hr = current_leak

    # -- Warnings --
    if state.seal_wear > 0.8:
        warnings.append("SEAL_DEGRADED: %.0f%% worn" % (state.seal_wear * 100))
    if state.filter_wear > 0.8:
        warnings.append("FILTER_DEGRADED: %.0f%% worn — replace soon" % (state.filter_wear * 100))
    if current_leak > 0.5:
        warnings.append("HIGH_LEAK_RATE: %.2f kPa/hr" % current_leak)
    if state.total_perchlorate_g > 100.0:
        warnings.append("PERCHLORATE_BUILDUP: %.1f g — toxic threshold" % state.total_perchlorate_g)

    # -- Check for seal failure --
    if state.seal_wear >= 1.0:
        state.operational = False
        warnings.append("SEAL_FAILURE: Airlock seals failed — not safe for use")

    result.warnings = warnings
    return result


def tick_airlock(
    state: AirlockState,
    egress_cycles: int = 0,
    ingress_cycles: int = 0,
    crew_per_cycle: int = 2,
    dust_storm: bool = False,
    maintenance: bool = False,
    filter_replacement: bool = False,
) -> AirlockSol:
    """Advance the airlock simulation by one sol.

    Parameters
    ----------
    state : AirlockState
        Mutable airlock state (modified in-place).
    egress_cycles : int
        Number of crew going OUT this sol.
    ingress_cycles : int
        Number of crew coming IN this sol.
    crew_per_cycle : int
        Crew members per cycle (max MAX_CREW_PER_CYCLE).
    dust_storm : bool
        Whether a dust storm is active (doubles dust ingress).
    maintenance : bool
        Whether seal maintenance is performed this sol.
    filter_replacement : bool
        Whether the dust filter is replaced this sol.

    Returns
    -------
    AirlockSol with aggregated metrics and warnings.
    """
    state.sol += 1
    result = AirlockSol(sol=state.sol)
    all_warnings: List[str] = []

    if not state.operational:
        result.halted = True
        all_warnings.append("AIRLOCK_OFFLINE: Not operational")
        result.warnings = all_warnings
        return result

    # -- Maintenance --
    if maintenance:
        restored = state.seal_wear * MAINTENANCE_SEAL_RESTORE
        state.seal_wear = max(0.0, state.seal_wear - restored)

    if filter_replacement:
        state.filter_wear = 0.0

    # -- Background leak (continuous, even without cycles) --
    current_leak = leak_rate(state.seal_wear)
    leak_loss_sol = current_leak * MARS_SOL_HOURS  # kPa lost over full sol
    leak_air_kg = air_mass_in_chamber(min(leak_loss_sol, HABITAT_PRESSURE_KPA))
    state.total_air_lost_kg += leak_air_kg

    # -- Execute cycles --
    total_requested = egress_cycles + ingress_cycles
    total_cycles = min(total_requested, MAX_CYCLES_PER_SOL)

    if total_requested > MAX_CYCLES_PER_SOL:
        all_warnings.append("CYCLE_LIMIT: Requested %d, max %d per sol" % (total_requested, MAX_CYCLES_PER_SOL))

    # Check time budget
    time_per_cycle = cycle_time_min()
    available_min = MARS_SOL_HOURS * 60.0
    max_by_time = int(available_min / time_per_cycle) if time_per_cycle > 0 else 0
    total_cycles = min(total_cycles, max_by_time)

    sol_air_lost = leak_air_kg
    sol_dust = 0.0
    sol_perchlorate = 0.0
    sol_energy = 0.0
    cycles_done = 0

    for _ in range(total_cycles):
        if not state.operational:
            break
        cr = run_cycle(state, crew_count=crew_per_cycle, dust_storm=dust_storm)
        if cr.aborted:
            all_warnings.extend(cr.warnings)
            break
        cycles_done += 1
        sol_air_lost += cr.air_lost_kg
        sol_dust += cr.dust_ingress_g
        sol_perchlorate += cr.perchlorate_ingress_g
        sol_energy += cr.energy_kwh
        all_warnings.extend(cr.warnings)

    result.cycles_completed = cycles_done
    result.total_air_lost_kg = sol_air_lost
    result.total_dust_ingress_g = sol_dust
    result.total_perchlorate_g = sol_perchlorate
    result.total_energy_kwh = sol_energy
    result.seal_health = 1.0 - state.seal_wear
    result.filter_health = 1.0 - state.filter_wear
    result.leak_rate_kpa_hr = leak_rate(state.seal_wear)
    result.warnings = all_warnings
    result.halted = not state.operational
    return result


def create_airlock(config: str = "standard") -> AirlockState:
    """Create an airlock configured for a given role.

    Configurations:
    - standard: default colony airlock
    - heavy: larger seal tolerance, pre-worn filter (high-traffic)
    - emergency: fresh seals, low cycle count (backup)
    """
    configs = {
        "standard": AirlockState(),
        "heavy": AirlockState(filter_wear=0.1),
        "emergency": AirlockState(seal_wear=0.0, filter_wear=0.0),
    }
    return configs.get(config, configs["standard"])
