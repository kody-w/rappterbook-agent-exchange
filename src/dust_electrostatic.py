"""dust_electrostatic.py — Mars Electrostatic Dust Mitigation System.

Active dust removal using the Electrodynamic Dust Shield (EDS), a real
NASA technology (Kennedy Space Center, 2010-present).  Multi-phase AC
electric fields on transparent ITO-coated surfaces repel charged dust
particles via Coulomb force.

Mars dust is the colony's #1 environmental hazard:
  - Coats solar panels -> 30-50% power loss over months
  - Infiltrates airlock seals -> pressure leaks
  - Contains perchlorates (0.5-1 wt%) -> toxic if inhaled
  - Abrades optical instruments and EVA suit visors
  - Blocks thermal radiators -> overheating risk

Physics modelled
----------------
* **Triboelectric charging**: Mars dust grains acquire charge through
  grain-grain and grain-surface collisions.  Typical charge: 1e-15 to
  1e-13 C per grain (10-100 fC).  Fine grains (<5 um) tend negative.
* **Coulomb force vs adhesion**: Electric field E exerts F = qE on a
  charged grain.  Must overcome van der Waals adhesion (proportional to d)
  and gravity (proportional to d cubed).  Threshold field ~1-5 kV/cm for
  typical Mars dust (1-10 um).
* **Multi-phase AC drive**: 4-phase standing wave pushes grains
  directionally (traveling wave dielectrophoresis).  Frequency 5-25 Hz
  optimal for Mars grain sizes.
* **Power consumption**: ~0.1 W/m2 continuous for panel shields.
  Negligible compared to panel output.
* **Cleaning efficiency**: 90-95% removal for grains >5 um; drops to
  60-70% for sub-micron particles (too light to overcome adhesion).
  Overall ~85% integrated efficiency (NASA EDS tests, 2019).
* **Electrode degradation**: ITO coating thins from UV + thermal cycling.
  ~0.02% per sol degradation rate.  At 50% health, efficiency halved.
* **Temperature effects**: Below -80C, ITO resistivity rises, reducing
  field strength.  Above +20C, thermal expansion stresses electrodes.
* **Dust storm surge**: During active storms, deposition rate exceeds
  cleaning rate.  Shield runs continuously but net dust increases.
  After storm clears, shield catches up over 2-10 sols.

Reference technology:
  - NASA Electrodynamic Dust Shield (Kennedy Space Center)
  - Mazumder et al., "Self-cleaning transparent dust shields" (2019)
  - Calle et al., "Particle removal by EDS from solar panels" (2011)
  - Typical Mars dust: basaltic, 1-3 um mode, rho ~ 3000 kg/m3

One tick = one sol.  Power in watts, area in m2, charge in coulombs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


# -- Physical constants ---------------------------------------------------

# Mars dust properties
DUST_DENSITY_KG_M3 = 3000.0          # basaltic dust grain density
DUST_MODE_DIAMETER_UM = 2.5           # most common grain size (um)
DUST_CHARGE_FC = 50.0                 # typical grain charge (femtocoulombs)
DUST_CHARGE_C = DUST_CHARGE_FC * 1e-15  # convert to coulombs

# Adhesion (van der Waals)
HAMAKER_CONSTANT_J = 1.0e-19         # grain-glass Hamaker constant
VDW_CONTACT_SEPARATION_M = 4.0e-10   # contact separation ~0.4 nm

# Mars gravity
MARS_GRAVITY_M_S2 = 3.72

# Electric field parameters
THRESHOLD_FIELD_V_M = 3.0e5          # ~3 kV/cm to lift typical grain
OPERATING_FIELD_V_M = 5.0e5          # ~5 kV/cm nominal operating field
MAX_FIELD_V_M = 1.0e6                # breakdown limit for Mars atmo
AC_FREQUENCY_HZ = 15.0               # optimal frequency for Mars dust
NUM_PHASES = 4                        # 4-phase traveling wave

# Cleaning efficiency by grain size regime
EFFICIENCY_LARGE = 0.93               # >10 um grains
EFFICIENCY_MEDIUM = 0.85              # 2-10 um grains
EFFICIENCY_FINE = 0.65                # <2 um grains
INTEGRATED_EFFICIENCY = 0.85          # overall weighted average

# Power
POWER_W_PER_M2 = 0.1                 # continuous power draw per m2 shield
STARTUP_ENERGY_WH_PER_M2 = 0.5       # energy to charge electrodes initially
BURST_POWER_MULTIPLIER = 3.0          # burst cleaning mode power multiplier

# Electrode degradation
ITO_DEGRADATION_PER_SOL = 0.0002     # fractional loss per sol (UV + thermal)
ITO_MIN_HEALTH = 0.10                # below this, shield non-functional
ITO_REPLACEMENT_THRESHOLD = 0.40     # recommended replacement health

# Temperature effects on ITO conductivity
ITO_OPTIMAL_TEMP_LOW_C = -60.0       # lower optimal bound
ITO_OPTIMAL_TEMP_HIGH_C = 20.0       # upper optimal bound
ITO_COLD_PENALTY_PER_C = 0.005       # efficiency loss per deg C below -60
ITO_HOT_PENALTY_PER_C = 0.003        # efficiency loss per deg C above +20
ITO_MIN_TEMP_C = -120.0              # absolute minimum operating temp
ITO_MAX_TEMP_C = 80.0                # absolute maximum operating temp

# Dust storm parameters (interaction with dust_storm.py)
NORMAL_DEPOSITION_RATE = 0.003        # fraction/sol in clear weather
STORM_DEPOSITION_RATE = 0.015         # fraction/sol during storm
RECOVERY_SOLS_AFTER_STORM = 5         # sols to clear accumulated dust

# Mars sol
SECONDS_PER_SOL = 88_775.0
HOURS_PER_SOL = SECONDS_PER_SOL / 3600.0  # ~24.66


# -- Physics functions ----------------------------------------------------

def van_der_waals_force(diameter_m: float) -> float:
    """Van der Waals adhesion force for a sphere on a flat surface (N).

    F_vdw = A * d / (24 * z**2)
    where A = Hamaker constant, d = grain diameter, z = contact separation.
    """
    if diameter_m <= 0:
        return 0.0
    return (HAMAKER_CONSTANT_J * diameter_m /
            (24.0 * VDW_CONTACT_SEPARATION_M ** 2))


def gravity_force(diameter_m: float) -> float:
    """Gravitational force on a spherical dust grain on Mars (N).

    F_g = (pi/6) * d**3 * rho * g
    """
    if diameter_m <= 0:
        return 0.0
    volume = (math.pi / 6.0) * diameter_m ** 3
    return volume * DUST_DENSITY_KG_M3 * MARS_GRAVITY_M_S2


def coulomb_force(charge_c: float, field_v_m: float) -> float:
    """Electrostatic force on a charged grain in an electric field (N).

    F_e = q * E
    """
    return abs(charge_c) * abs(field_v_m)


def removal_threshold_field(diameter_um: float) -> float:
    """Minimum electric field to remove a grain of given diameter (V/m).

    The field must produce a Coulomb force exceeding both van der Waals
    adhesion and gravity.  Adhesion dominates for small grains.
    """
    if diameter_um <= 0:
        return float('inf')
    d_m = diameter_um * 1e-6
    f_vdw = van_der_waals_force(d_m)
    f_grav = gravity_force(d_m)
    total_adhesion = f_vdw + f_grav

    # Charge scales roughly with surface area proportional to d**2
    charge = DUST_CHARGE_C * (diameter_um / DUST_MODE_DIAMETER_UM) ** 2
    if charge <= 0:
        return float('inf')

    return total_adhesion / charge


def cleaning_efficiency(field_v_m: float, electrode_health: float,
                        temperature_c: float) -> float:
    """Overall dust removal efficiency given operating conditions.

    Combines field strength, electrode health, and temperature effects.
    Returns fraction of dust removed per cleaning cycle (0..1).
    """
    if electrode_health <= ITO_MIN_HEALTH:
        return 0.0

    # Field strength factor (logistic curve centered on threshold)
    if field_v_m <= 0:
        return 0.0
    field_ratio = field_v_m / THRESHOLD_FIELD_V_M
    field_factor = 1.0 / (1.0 + math.exp(-3.0 * (field_ratio - 1.0)))

    # Temperature penalty
    temp_factor = 1.0
    if temperature_c < ITO_OPTIMAL_TEMP_LOW_C:
        temp_factor = max(0.2, 1.0 - ITO_COLD_PENALTY_PER_C *
                          (ITO_OPTIMAL_TEMP_LOW_C - temperature_c))
    elif temperature_c > ITO_OPTIMAL_TEMP_HIGH_C:
        temp_factor = max(0.2, 1.0 - ITO_HOT_PENALTY_PER_C *
                          (temperature_c - ITO_OPTIMAL_TEMP_HIGH_C))

    # Electrode health scales linearly
    health_factor = max(0.0, (electrode_health - ITO_MIN_HEALTH) /
                        (1.0 - ITO_MIN_HEALTH))

    raw_efficiency = INTEGRATED_EFFICIENCY * field_factor * temp_factor * health_factor
    return max(0.0, min(1.0, raw_efficiency))


def power_consumption_w(area_m2: float, mode: str = "normal") -> float:
    """Power draw for the dust shield (watts).

    Modes: 'standby' (monitoring only), 'normal' (continuous cleaning),
    'burst' (aggressive post-storm cleaning).
    """
    if area_m2 <= 0:
        return 0.0
    base = POWER_W_PER_M2 * area_m2
    multipliers = {"standby": 0.1, "normal": 1.0, "burst": BURST_POWER_MULTIPLIER}
    return base * multipliers.get(mode, 1.0)


def electrode_degradation(current_health: float, sol_count: int = 1,
                          temperature_c: float = -40.0) -> float:
    """Calculate new electrode health after sol_count sols.

    UV radiation and thermal cycling degrade the ITO coating.
    Extreme temperatures accelerate degradation.
    """
    if current_health <= 0:
        return 0.0

    rate = ITO_DEGRADATION_PER_SOL
    # Temperature stress accelerates degradation
    if temperature_c < ITO_OPTIMAL_TEMP_LOW_C:
        rate *= 1.0 + 0.01 * (ITO_OPTIMAL_TEMP_LOW_C - temperature_c)
    elif temperature_c > ITO_OPTIMAL_TEMP_HIGH_C:
        rate *= 1.0 + 0.02 * (temperature_c - ITO_OPTIMAL_TEMP_HIGH_C)

    new_health = current_health - rate * sol_count
    return max(0.0, min(1.0, new_health))


def net_dust_change(current_dust: float, deposition_rate: float,
                    cleaning_eff: float) -> float:
    """Net change in dust coverage fraction per sol.

    Positive = dust accumulating, negative = shield winning.
    """
    deposited = deposition_rate * (1.0 - current_dust)
    removed = cleaning_eff * current_dust
    return deposited - removed


# -- State ----------------------------------------------------------------

@dataclass
class DustShieldState:
    """Complete state of one electrostatic dust shield installation."""

    # Geometry
    area_m2: float = 100.0              # total shielded area
    coverage_zones: int = 1             # number of independent zones

    # Electrode health
    electrode_health: float = 1.0       # ITO coating condition (0..1)

    # Operating state
    active: bool = True                 # shield powered on
    mode: str = "normal"                # standby / normal / burst
    field_v_m: float = OPERATING_FIELD_V_M  # current electric field strength

    # Dust state
    dust_fraction: float = 0.0          # fraction of surface covered by dust

    # Environment
    temperature_c: float = -40.0        # surface temperature
    storm_active: bool = False          # is a dust storm currently active

    # Counters
    sol: int = 0
    total_energy_wh: float = 0.0        # cumulative energy consumed
    cleaning_cycles: int = 0            # total cleaning cycles performed
    dust_removed_kg_m2: float = 0.0     # cumulative dust removed

    def __post_init__(self) -> None:
        """Clamp all fields to physical bounds."""
        self.area_m2 = max(0.0, self.area_m2)
        self.coverage_zones = max(1, self.coverage_zones)
        self.electrode_health = max(0.0, min(1.0, self.electrode_health))
        self.dust_fraction = max(0.0, min(1.0, self.dust_fraction))
        self.temperature_c = max(ITO_MIN_TEMP_C, min(ITO_MAX_TEMP_C,
                                                      self.temperature_c))
        self.field_v_m = max(0.0, min(MAX_FIELD_V_M, self.field_v_m))
        if self.mode not in ("standby", "normal", "burst"):
            self.mode = "normal"


@dataclass
class DustShieldResult:
    """Result of one tick of the dust shield."""
    dust_before: float = 0.0
    dust_after: float = 0.0
    dust_deposited: float = 0.0
    dust_removed: float = 0.0
    cleaning_efficiency: float = 0.0
    power_used_wh: float = 0.0
    electrode_health: float = 1.0
    mode: str = "normal"
    warning: str = ""


# -- Tick function --------------------------------------------------------

def tick_dust_shield(state: DustShieldState,
                     storm_active: bool = False,
                     temperature_c: float | None = None) -> DustShieldResult:
    """Advance the dust shield by one sol.

    Parameters
    ----------
    state : DustShieldState
        Mutable shield state (modified in place).
    storm_active : bool
        Whether a dust storm is currently active.
    temperature_c : float or None
        Override surface temperature (None = keep current).

    Returns
    -------
    DustShieldResult with metrics from this sol.
    """
    result = DustShieldResult()
    result.dust_before = state.dust_fraction

    # Update environment
    state.storm_active = storm_active
    if temperature_c is not None:
        state.temperature_c = max(ITO_MIN_TEMP_C,
                                   min(ITO_MAX_TEMP_C, temperature_c))

    # Auto-mode selection
    if storm_active and state.mode != "burst":
        state.mode = "burst"
    elif not storm_active and state.dust_fraction > 0.3:
        state.mode = "burst"
    elif not storm_active and state.dust_fraction < 0.05:
        state.mode = "standby"
    elif not storm_active:
        state.mode = "normal"

    result.mode = state.mode

    # Deposition this sol
    dep_rate = STORM_DEPOSITION_RATE if storm_active else NORMAL_DEPOSITION_RATE
    deposited = dep_rate * (1.0 - state.dust_fraction)
    result.dust_deposited = deposited

    # Cleaning
    eff = 0.0
    if state.active and state.electrode_health > ITO_MIN_HEALTH:
        eff = cleaning_efficiency(state.field_v_m, state.electrode_health,
                                   state.temperature_c)
        # Burst mode gets extra cleaning passes
        if state.mode == "burst":
            eff = min(1.0, eff * 1.3)

    result.cleaning_efficiency = eff
    removed = eff * state.dust_fraction
    result.dust_removed = removed

    # Net dust change
    new_dust = state.dust_fraction + deposited - removed
    state.dust_fraction = max(0.0, min(1.0, new_dust))
    result.dust_after = state.dust_fraction

    # Dust mass removed (approximate: 1 mg/m2 per 1% coverage)
    state.dust_removed_kg_m2 += removed * 0.001 * state.area_m2

    # Power consumption
    power_w = power_consumption_w(state.area_m2, state.mode) if state.active else 0.0
    energy_wh = power_w * HOURS_PER_SOL
    state.total_energy_wh += energy_wh
    result.power_used_wh = energy_wh

    # Electrode degradation
    state.electrode_health = electrode_degradation(
        state.electrode_health, sol_count=1,
        temperature_c=state.temperature_c
    )
    result.electrode_health = state.electrode_health

    # Warnings
    if state.electrode_health < ITO_REPLACEMENT_THRESHOLD:
        result.warning = "electrode_degraded"
    if state.electrode_health <= ITO_MIN_HEALTH:
        result.warning = "electrode_failed"
        state.active = False

    state.sol += 1
    state.cleaning_cycles += 1

    return result


# -- Factory --------------------------------------------------------------

def create_dust_shield(scenario: str = "solar_panels") -> DustShieldState:
    """Create a DustShieldState for a named scenario.

    Scenarios
    ---------
    solar_panels : Shield covering colony solar array (~500 m2)
    airlock : Small shield for airlock decontamination (~10 m2)
    greenhouse : Medium shield for greenhouse windows (~50 m2)
    visor : EVA suit visor shield (~0.05 m2)
    """
    configs = {
        "solar_panels": dict(
            area_m2=500.0, coverage_zones=10, temperature_c=-40.0,
        ),
        "airlock": dict(
            area_m2=10.0, coverage_zones=2, temperature_c=-20.0,
        ),
        "greenhouse": dict(
            area_m2=50.0, coverage_zones=4, temperature_c=5.0,
        ),
        "visor": dict(
            area_m2=0.05, coverage_zones=1, temperature_c=-30.0,
        ),
    }
    cfg = configs.get(scenario, configs["solar_panels"])
    return DustShieldState(**cfg)
