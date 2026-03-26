"""nuclear_reactor.py — Kilopower-class Mars Fission Reactor.

Models a small fission reactor providing baseload electrical power
to a Mars colony.  Unlike solar arrays, a fission reactor operates
through dust storms, at night, and at any latitude — the backbone
that keeps the colony alive when the sun disappears.

Each tick = 1 sol of reactor operations.

Physics modelled
----------------
* **Fission heat generation** — U-235 core with negative temperature
  coefficient of reactivity (self-regulating).  Thermal power is set
  by control drum position.  Reference: NASA KRUSTY (Kilopower Reactor
  Using Stirling Technology), 2018 test at NNSS.
* **Stirling conversion** — free-piston Stirling engines convert
  thermal power to electricity.  Carnot-limited efficiency depends on
  hot-side (core) and cold-side (radiator) temperatures.
* **Radiator cooling** — NaK (sodium-potassium) coolant loop rejects
  waste heat through deployable radiator fins.  Radiator performance
  depends on Mars ambient temperature and fin area.  Dust accumulation
  on radiator fins reduces effectiveness.
* **Fuel burnup** — U-235 depletes over reactor lifetime.  KRUSTY
  designed for 10+ year operation at 10 kWe.  We model linear burnup
  with reactivity margin.
* **Control drums** — BeO reflector drums with B4C poison arcs.
  Rotation angle controls neutron reflection → reactivity → power.
  Range: 0° (shutdown) to 180° (full power).
* **Radiation shielding** — Shadow shield (LiH + depleted uranium)
  protects crew.  Dose rate at habitat distance modelled as inverse
  square law with shielding attenuation.
* **Startup / shutdown** — reactor must be brought to criticality
  from cold state (takes ~2 sols).  SCRAM (emergency shutdown) is
  instantaneous but requires 1 sol cooldown before restart.

Reference hardware:
  - KRUSTY/Kilopower: 1-10 kWe, ~400 kg total, 10+ yr life
  - SNAP-10A (1965): 0.5 kWe, first reactor in space
  - Megapower concept: 2 MWe for Mars base (future)
  - This model: 10 kWe nominal, scalable to 40 kWe with 4 units
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


# -- Physical constants -------------------------------------------------------

BOLTZMANN_W_M2_K4 = 5.67e-8       # Stefan-Boltzmann constant
MARS_AMBIENT_TEMP_K = 210.0       # average Mars surface temperature
MARS_SOL_SECONDS = 88775.0        # one sol in seconds

# Core
U235_ENERGY_DENSITY_MJ_KG = 82_000_000.0   # theoretical max, 82 TJ/kg
CORE_MASS_KG = 28.0                # HEU core mass (KRUSTY reference)
CORE_ENRICHMENT = 0.932            # 93.2% HEU (KRUSTY spec)
FUEL_MASS_KG = CORE_MASS_KG * CORE_ENRICHMENT  # fissile U-235 mass
THERMAL_POWER_MAX_KW = 43.0       # max thermal output
CORE_TEMP_NOMINAL_K = 1073.0      # 800°C hot side (KRUSTY)
CORE_TEMP_MAX_K = 1173.0          # emergency limit
TEMP_REACTIVITY_COEFF = -0.002    # dk/k per K (negative = self-regulating)

# Stirling engines
NUM_STIRLING_ENGINES = 8           # KRUSTY: 8 free-piston Stirling
STIRLING_FRACTION_OF_CARNOT = 0.50 # realistic fraction of Carnot efficiency
ENGINE_MASS_KG_EACH = 12.0        # per engine

# Radiator
RADIATOR_AREA_M2 = 6.0            # total fin area
RADIATOR_EMISSIVITY = 0.85        # high-emissivity coating
DUST_ACCUMULATION_PER_SOL = 0.0005 # emissivity loss per sol from dust
DUST_STORM_ACCUMULATION = 0.01    # dust during storms
CLEANING_RESTORE_FRACTION = 0.9   # cleaning restores this much emissivity
RADIATOR_MIN_EMISSIVITY = 0.30    # floor even with heavy dust

# Fuel burnup
DESIGN_LIFE_SOLS = 6680           # ~10 Earth years
BURNUP_FRACTION_AT_EOL = 0.05     # 5% U-235 consumed over design life

# Control drums
DRUM_ANGLE_SHUTDOWN = 0.0         # degrees — fully poisoned
DRUM_ANGLE_FULL_POWER = 180.0     # degrees — max reflection
DRUM_RATE_DEG_PER_SOL = 90.0      # max rotation rate

# Shielding
SHIELD_MASS_KG = 150.0            # LiH + depleted uranium
SHIELD_ATTENUATION_FACTOR = 1e-6  # dose transmission through shield
UNSHIELDED_DOSE_RATE_MSV_HR = 500.0  # at 1 m from bare core (lethal)
HAB_DISTANCE_M = 100.0            # reactor-to-habitat distance

# Startup
STARTUP_SOLS = 2                  # sols to reach criticality from cold
SCRAM_COOLDOWN_SOLS = 1           # sols before restart after SCRAM


# -- Enums / State ------------------------------------------------------------

class ReactorState(Enum):
    """Operational state of the reactor."""
    COLD = "cold"                  # never started or fully cooled
    STARTING = "starting"          # coming up to criticality
    RUNNING = "running"            # normal power generation
    SCRAMMED = "scrammed"          # emergency shutdown, cooling


@dataclass
class FissionReactor:
    """Complete state of a Kilopower-class fission reactor."""
    state: ReactorState = ReactorState.COLD
    drum_angle_deg: float = DRUM_ANGLE_SHUTDOWN
    core_temp_k: float = MARS_AMBIENT_TEMP_K
    radiator_emissivity: float = RADIATOR_EMISSIVITY
    fuel_remaining_fraction: float = 1.0
    sols_operated: int = 0
    startup_sols_remaining: int = 0
    cooldown_sols_remaining: int = 0
    total_energy_kwh: float = 0.0
    scram_count: int = 0
    electrical_output_kw: float = 0.0


# -- Physics functions --------------------------------------------------------

def thermal_power_from_drums(drum_angle_deg: float,
                             fuel_fraction: float) -> float:
    """Thermal power (kW) as function of control drum angle and fuel state.

    Power is approximately sinusoidal with drum angle (0°=shutdown,
    180°=full power).  Depleted fuel reduces max reactivity.
    """
    if drum_angle_deg <= 0.0 or fuel_fraction <= 0.0:
        return 0.0
    angle = max(0.0, min(DRUM_ANGLE_FULL_POWER, drum_angle_deg))
    # Sinusoidal response: P = Pmax * sin²(angle/2 * π/180)
    # At 180°: sin²(90°) = 1.0, at 0°: sin²(0) = 0.0
    angle_rad = math.radians(angle / 2.0)
    power_fraction = math.sin(angle_rad) ** 2
    # Fuel depletion reduces available reactivity
    reactivity_margin = min(1.0, fuel_fraction / (1.0 - BURNUP_FRACTION_AT_EOL))
    reactivity_margin = max(0.0, reactivity_margin)
    thermal_kw = THERMAL_POWER_MAX_KW * power_fraction * reactivity_margin
    return round(thermal_kw, 4)


def radiator_rejection_kw(radiator_temp_k: float,
                          emissivity: float) -> float:
    """Heat rejected by radiator (kW) via Stefan-Boltzmann radiation.

    Q = ε σ A (T_rad⁴ - T_ambient⁴)
    Both surfaces radiate; effective area is 2× for a flat fin.
    """
    if radiator_temp_k <= MARS_AMBIENT_TEMP_K:
        return 0.0
    eff_emissivity = max(RADIATOR_MIN_EMISSIVITY, min(1.0, emissivity))
    # Two-sided flat plate radiator
    area_effective = RADIATOR_AREA_M2 * 2.0
    q_watts = (eff_emissivity * BOLTZMANN_W_M2_K4 * area_effective *
               (radiator_temp_k ** 4 - MARS_AMBIENT_TEMP_K ** 4))
    return round(q_watts / 1000.0, 4)  # convert to kW


def stirling_efficiency(hot_temp_k: float, cold_temp_k: float) -> float:
    """Electrical conversion efficiency of Stirling engines.

    Fraction of Carnot efficiency: η = f × (1 - T_cold/T_hot)
    """
    if hot_temp_k <= cold_temp_k or hot_temp_k <= 0:
        return 0.0
    carnot = 1.0 - (cold_temp_k / hot_temp_k)
    efficiency = STIRLING_FRACTION_OF_CARNOT * carnot
    return round(max(0.0, min(0.45, efficiency)), 6)


def core_temperature_step(current_temp_k: float, thermal_power_kw: float,
                          radiator_emissivity: float,
                          sol_fraction: float) -> float:
    """Update core temperature based on heat balance.

    Lumped capacitance model with sub-stepping.  At each sub-step,
    radiator rejection is recomputed from current temperature so the
    core converges toward thermal equilibrium naturally.

    Core + coolant + structure thermal mass ~500 kJ/K.
    """
    core_thermal_mass_j_per_k = 500_000.0
    total_seconds = sol_fraction * MARS_SOL_SECONDS
    if total_seconds <= 0:
        return current_temp_k
    n_steps = 100
    dt = total_seconds / n_steps
    temp = current_temp_k
    for _ in range(n_steps):
        # Stirling cold side approximation
        cold_side_k = min(temp, 550.0)
        # Dynamic radiator rejection at current temperature
        rejection_kw = radiator_rejection_kw(cold_side_k, radiator_emissivity)
        net_watts = (thermal_power_kw - rejection_kw) * 1000.0
        delta_t = (net_watts * dt) / core_thermal_mass_j_per_k
        temp += delta_t
        temp = max(MARS_AMBIENT_TEMP_K, min(2000.0, temp))
    return round(temp, 2)


def fuel_burnup_per_sol(thermal_power_kw: float) -> float:
    """Fraction of fuel consumed per sol at given thermal power.

    Linear model: at max power, fuel lasts DESIGN_LIFE_SOLS.
    """
    if thermal_power_kw <= 0:
        return 0.0
    power_fraction = thermal_power_kw / THERMAL_POWER_MAX_KW
    burnup = (BURNUP_FRACTION_AT_EOL / DESIGN_LIFE_SOLS) * power_fraction
    return round(burnup, 12)


def radiation_dose_at_hab(shielded: bool) -> float:
    """Radiation dose rate (mSv/hr) at habitat distance.

    Inverse square law from core, attenuated by shadow shield.
    """
    # Dose at 1 m, unshielded
    dose_1m = UNSHIELDED_DOSE_RATE_MSV_HR
    # Inverse square to habitat distance
    dose_at_hab = dose_1m / (HAB_DISTANCE_M ** 2)
    if shielded:
        dose_at_hab *= SHIELD_ATTENUATION_FACTOR
    return round(dose_at_hab, 10)


def move_drum_toward(current_deg: float, target_deg: float,
                     sol_fraction: float) -> float:
    """Move control drum toward target angle at limited rate.

    Returns new drum angle (degrees).
    """
    target = max(DRUM_ANGLE_SHUTDOWN, min(DRUM_ANGLE_FULL_POWER, target_deg))
    max_move = DRUM_RATE_DEG_PER_SOL * sol_fraction
    delta = target - current_deg
    if abs(delta) <= max_move:
        return round(target, 4)
    direction = 1.0 if delta > 0 else -1.0
    new_angle = current_deg + direction * max_move
    return round(max(DRUM_ANGLE_SHUTDOWN, min(DRUM_ANGLE_FULL_POWER, new_angle)), 4)


# -- Tick function (one sol) --------------------------------------------------

def tick(reactor: FissionReactor,
         demand_kw: float = 10.0,
         dust_storm: bool = False,
         scram: bool = False) -> dict:
    """Advance the reactor by one sol.

    Parameters
    ----------
    reactor : FissionReactor
        Mutable reactor state (modified in place).
    demand_kw : float
        Requested electrical output (kW).  Reactor adjusts drums to match.
    dust_storm : bool
        Whether a dust storm is active (affects radiator dust).
    scram : bool
        Emergency shutdown trigger.

    Returns
    -------
    dict with keys:
        electrical_kw, thermal_kw, efficiency, core_temp_k,
        fuel_remaining, state, dose_msv_hr, radiator_emissivity
    """
    result = {
        "electrical_kw": 0.0,
        "thermal_kw": 0.0,
        "efficiency": 0.0,
        "core_temp_k": reactor.core_temp_k,
        "fuel_remaining": reactor.fuel_remaining_fraction,
        "state": reactor.state.value,
        "dose_msv_hr": 0.0,
        "radiator_emissivity": reactor.radiator_emissivity,
    }

    # -- Handle SCRAM ---
    if scram and reactor.state == ReactorState.RUNNING:
        reactor.state = ReactorState.SCRAMMED
        reactor.drum_angle_deg = DRUM_ANGLE_SHUTDOWN
        reactor.cooldown_sols_remaining = SCRAM_COOLDOWN_SOLS
        reactor.scram_count += 1
        reactor.electrical_output_kw = 0.0
        result["state"] = reactor.state.value
        return result  # SCRAM consumes the rest of this sol

    # -- State machine ---
    if reactor.state == ReactorState.COLD:
        # Can start if fuel remains
        if demand_kw > 0 and reactor.fuel_remaining_fraction > 0:
            reactor.state = ReactorState.STARTING
            reactor.startup_sols_remaining = STARTUP_SOLS
        result["state"] = reactor.state.value
        return result

    if reactor.state == ReactorState.STARTING:
        reactor.startup_sols_remaining -= 1
        if reactor.startup_sols_remaining <= 0:
            reactor.state = ReactorState.RUNNING
            reactor.drum_angle_deg = 10.0  # initial low-power criticality
        result["state"] = reactor.state.value
        result["core_temp_k"] = reactor.core_temp_k
        return result

    if reactor.state == ReactorState.SCRAMMED:
        reactor.cooldown_sols_remaining -= 1
        # Core cools toward ambient
        reactor.core_temp_k = core_temperature_step(
            reactor.core_temp_k, 0.0, reactor.radiator_emissivity, 1.0
        )
        if reactor.cooldown_sols_remaining <= 0:
            reactor.state = ReactorState.COLD
        result["state"] = reactor.state.value
        result["core_temp_k"] = reactor.core_temp_k
        return result

    # -- RUNNING state ---
    reactor.sols_operated += 1

    # -- Radiator dust ---
    dust_rate = DUST_STORM_ACCUMULATION if dust_storm else DUST_ACCUMULATION_PER_SOL
    reactor.radiator_emissivity = max(
        RADIATOR_MIN_EMISSIVITY,
        reactor.radiator_emissivity - dust_rate
    )
    result["radiator_emissivity"] = round(reactor.radiator_emissivity, 6)

    # -- Determine target drum angle from demand ---
    # Estimate needed thermal power: P_th = P_el / η (guess η ~0.25)
    eta_guess = 0.25
    needed_thermal = demand_kw / eta_guess if eta_guess > 0 else 0
    target_power_fraction = min(1.0, needed_thermal / THERMAL_POWER_MAX_KW)
    # Invert the sinusoidal power curve to find target angle
    # power_frac = sin²(angle/2 in rad) → angle = 2 × arcsin(√frac) in deg
    safe_frac = max(0.0, min(1.0, target_power_fraction))
    target_angle = 2.0 * math.degrees(math.asin(math.sqrt(safe_frac)))
    reactor.drum_angle_deg = move_drum_toward(
        reactor.drum_angle_deg, target_angle, 1.0
    )

    # -- Thermal power ---
    thermal_kw = thermal_power_from_drums(
        reactor.drum_angle_deg, reactor.fuel_remaining_fraction
    )
    result["thermal_kw"] = thermal_kw

    # -- Radiator heat rejection (for reporting) ---
    cold_side_k = min(reactor.core_temp_k, 550.0)
    heat_rejected = radiator_rejection_kw(cold_side_k, reactor.radiator_emissivity)

    # -- Core temperature (dynamic sub-stepping) ---
    reactor.core_temp_k = core_temperature_step(
        reactor.core_temp_k, thermal_kw, reactor.radiator_emissivity, 1.0
    )
    result["core_temp_k"] = reactor.core_temp_k

    # -- Safety: auto-SCRAM on over-temperature ---
    if reactor.core_temp_k > CORE_TEMP_MAX_K:
        reactor.state = ReactorState.SCRAMMED
        reactor.drum_angle_deg = DRUM_ANGLE_SHUTDOWN
        reactor.cooldown_sols_remaining = SCRAM_COOLDOWN_SOLS
        reactor.scram_count += 1
        result["state"] = reactor.state.value
        result["electrical_kw"] = 0.0
        reactor.electrical_output_kw = 0.0
        return result

    # -- Stirling electrical conversion ---
    cold_temp_k = max(MARS_AMBIENT_TEMP_K, cold_side_k)
    efficiency = stirling_efficiency(reactor.core_temp_k, cold_temp_k)
    electrical_kw = thermal_kw * efficiency
    result["efficiency"] = efficiency
    result["electrical_kw"] = round(electrical_kw, 4)
    reactor.electrical_output_kw = round(electrical_kw, 4)

    # -- Fuel burnup ---
    burnup = fuel_burnup_per_sol(thermal_kw)
    reactor.fuel_remaining_fraction = round(
        max(0.0, reactor.fuel_remaining_fraction - burnup), 10
    )
    result["fuel_remaining"] = reactor.fuel_remaining_fraction

    # -- Energy accumulation ---
    energy_kwh = electrical_kw * (MARS_SOL_SECONDS / 3600.0)
    reactor.total_energy_kwh = round(reactor.total_energy_kwh + energy_kwh, 4)

    # -- Radiation dose ---
    result["dose_msv_hr"] = radiation_dose_at_hab(shielded=True)

    # -- Fuel exhaustion check ---
    if reactor.fuel_remaining_fraction <= 0.0:
        reactor.state = ReactorState.COLD
        reactor.drum_angle_deg = DRUM_ANGLE_SHUTDOWN
        reactor.electrical_output_kw = 0.0

    result["state"] = reactor.state.value
    return result
