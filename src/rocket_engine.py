"""rocket_engine.py -- Mars Ascent Vehicle Methalox Engine.

The colony makes fuel (fuel_production.py), stores it (propellant_depot.py),
and knows when to launch (launch_window.py).  But it has no engine.
Without this module, the colony is a one-way trip.

This is the return ticket: a pressure-fed methalox (CH4/LOX) rocket
engine sized for a Mars Ascent Vehicle (MAV).  It models the complete
engine cycle from ignition through shutdown -- thrust, propellant flow,
chamber thermodynamics, nozzle expansion, and thermal limits.

Physics modelled
----------------
* Thrust: F = mdot*Ve + (Pe - P_amb)*Ae
* Specific impulse: Isp = F / (mdot*g0).  Methalox vacuum ~363 s.
* Tsiolkovsky equation: dv = Isp*g0*ln(m0/mf)
* Combustion: CH4 + 2O2 -> CO2 + 2H2O.  O/F ~ 3.5 by mass.
* Nozzle: isentropic expansion, bisection solver for exit Mach.
* Chamber thermal: Newton's law of cooling with regen CH4 coolant.
* Gravity & drag losses during vertical ascent.

Conservation laws
-----------------
- Mass: propellant_consumed = LOX_consumed + CH4_consumed (exact)
- Mixture ratio: LOX/CH4 = O/F (exact per tick)
- Thrust >= 0, Isp in [0, theoretical_max], propellant >= 0
- Chamber wall temp <= max operating temp (or engine fails)
- Delta-v <= Tsiolkovsky limit for mass ratio

One tick = one second of burn.  Mass in kg, force in N, velocity in m/s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# -- Physical constants -------------------------------------------------------

G0_M_S2 = 9.80665
MARS_GRAVITY_M_S2 = 3.72076
MARS_RADIUS_M = 3_389_500.0
MARS_SURFACE_PRESSURE_PA = 610.0
MARS_SCALE_HEIGHT_M = 11_100.0
UNIV_GAS_J_MOLKT = 8.314

# -- Methalox combustion ------------------------------------------------------

MIXTURE_RATIO_OF = 3.5
EXHAUST_MW_G_MOL = 22.0
EXHAUST_GAMMA = 1.25
CHAMBER_TEMP_K = 3500.0
CHAMBER_PRESSURE_PA = 2.0e6

# -- Thermal limits -----------------------------------------------------------

MAX_WALL_TEMP_K = 1200.0
WALL_HEAT_COEFF = 0.005
WALL_COOL_COEFF = 0.020
COOLANT_TEMP_K = 200.0

# -- Default engine sizing (MAV-class) ----------------------------------------

DEFAULT_THRUST_N = 44_000.0
DEFAULT_NOZZLE_AREA_RATIO = 60.0
DEFAULT_THROAT_AREA_M2 = 0.005
DEFAULT_LOX_KG = 3150.0
DEFAULT_CH4_KG = 900.0
DEFAULT_DRY_MASS_KG = 500.0

MARS_ORBIT_DV_M_S = 4100.0
GRAVITY_LOSS_M_S_PER_S = 3.0
DRAG_LOSS_TOTAL_M_S = 50.0


# -- Nozzle physics -----------------------------------------------------------

def exhaust_velocity_m_s(
    chamber_temp_k: float,
    chamber_pressure_pa: float,
    exit_pressure_pa_val: float,
    gamma: float = EXHAUST_GAMMA,
    mw_g_mol: float = EXHAUST_MW_G_MOL,
) -> float:
    """Ideal exhaust velocity from isentropic nozzle expansion."""
    if chamber_pressure_pa <= 0.0 or exit_pressure_pa_val < 0.0:
        return 0.0
    if exit_pressure_pa_val >= chamber_pressure_pa:
        return 0.0
    mw_kg_mol = mw_g_mol / 1000.0
    r_specific = UNIV_GAS_J_MOLKT / mw_kg_mol
    pressure_ratio = exit_pressure_pa_val / chamber_pressure_pa
    gm1 = gamma - 1.0
    exponent = gm1 / gamma
    term = 1.0 - pressure_ratio ** exponent
    ve_sq = (2.0 * gamma / gm1) * r_specific * chamber_temp_k * term
    if ve_sq <= 0.0:
        return 0.0
    return math.sqrt(ve_sq)


def _area_mach_ratio(mach: float, gamma: float) -> float:
    """Area ratio A/A* for given supersonic Mach number."""
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    m2 = mach * mach
    inner = (2.0 / gp1) * (1.0 + 0.5 * gm1 * m2)
    exp = 0.5 * gp1 / gm1
    return (1.0 / mach) * (inner ** exp)


def exit_pressure_pa(
    chamber_pressure_pa: float,
    area_ratio: float,
    gamma: float = EXHAUST_GAMMA,
) -> float:
    """Nozzle exit pressure via bisection on supersonic area-Mach relation."""
    if area_ratio <= 1.0 or chamber_pressure_pa <= 0.0:
        return chamber_pressure_pa
    gm1 = gamma - 1.0
    lo, hi = 1.001, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _area_mach_ratio(mid, gamma) < area_ratio:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    mach = 0.5 * (lo + hi)
    p_exit = chamber_pressure_pa * (1.0 + 0.5 * gm1 * mach * mach) ** (-gamma / gm1)
    return max(p_exit, 0.0)


def specific_impulse_s(
    exhaust_vel_m_s: float,
    exit_press_pa: float,
    ambient_pressure_pa: float,
    exit_area_m2: float,
    mass_flow_kg_s: float,
) -> float:
    """Effective Isp including pressure thrust."""
    if mass_flow_kg_s <= 0.0:
        return 0.0
    f = (mass_flow_kg_s * exhaust_vel_m_s
         + (exit_press_pa - ambient_pressure_pa) * exit_area_m2)
    return max(f / (mass_flow_kg_s * G0_M_S2), 0.0)


def thrust_n(
    exhaust_vel_m_s: float,
    exit_press_pa: float,
    ambient_pressure_pa: float,
    exit_area_m2: float,
    mass_flow_kg_s: float,
) -> float:
    """Total thrust: momentum + pressure contribution."""
    if mass_flow_kg_s <= 0.0:
        return 0.0
    f = (mass_flow_kg_s * exhaust_vel_m_s
         + (exit_press_pa - ambient_pressure_pa) * exit_area_m2)
    return max(f, 0.0)


def mass_flow_rate_kg_s(thrust_target_n: float, isp_s: float) -> float:
    """Required mass flow rate for target thrust at given Isp."""
    if isp_s <= 0.0 or thrust_target_n <= 0.0:
        return 0.0
    return thrust_target_n / (isp_s * G0_M_S2)


def tsiolkovsky_delta_v(isp_s: float, m_initial_kg: float,
                        m_final_kg: float) -> float:
    """Ideal delta-v from the Tsiolkovsky rocket equation."""
    if isp_s <= 0.0 or m_initial_kg <= 0.0 or m_final_kg <= 0.0:
        return 0.0
    if m_final_kg >= m_initial_kg:
        return 0.0
    return isp_s * G0_M_S2 * math.log(m_initial_kg / m_final_kg)


def gravity_at_altitude_m_s2(altitude_m: float) -> float:
    """Mars gravity at altitude (inverse square law)."""
    r = MARS_RADIUS_M + max(altitude_m, 0.0)
    return MARS_GRAVITY_M_S2 * (MARS_RADIUS_M / r) ** 2


def atmospheric_pressure_pa(altitude_m: float) -> float:
    """Mars atmospheric pressure at altitude (exponential model)."""
    if altitude_m < 0.0:
        return MARS_SURFACE_PRESSURE_PA
    return MARS_SURFACE_PRESSURE_PA * math.exp(-altitude_m / MARS_SCALE_HEIGHT_M)


def atmospheric_density_kg_m3(altitude_m: float) -> float:
    """Mars atmospheric density at altitude."""
    p = atmospheric_pressure_pa(altitude_m)
    return 0.020 * (p / MARS_SURFACE_PRESSURE_PA)


def drag_force_n(velocity_m_s: float, altitude_m: float,
                 drag_area_m2: float, cd: float = 0.3) -> float:
    """Aerodynamic drag on the ascending vehicle."""
    if velocity_m_s <= 0.0:
        return 0.0
    rho = atmospheric_density_kg_m3(altitude_m)
    return 0.5 * rho * velocity_m_s ** 2 * cd * drag_area_m2


# -- Engine state -------------------------------------------------------------

@dataclass
class EngineConfig:
    """Immutable engine configuration."""
    thrust_target_n: float = DEFAULT_THRUST_N
    chamber_pressure_pa: float = CHAMBER_PRESSURE_PA
    chamber_temp_k: float = CHAMBER_TEMP_K
    nozzle_area_ratio: float = DEFAULT_NOZZLE_AREA_RATIO
    throat_area_m2: float = DEFAULT_THROAT_AREA_M2
    mixture_ratio: float = MIXTURE_RATIO_OF
    max_wall_temp_k: float = MAX_WALL_TEMP_K


@dataclass
class VehicleState:
    """Mutable state of the MAV during ascent."""
    lox_kg: float = DEFAULT_LOX_KG
    ch4_kg: float = DEFAULT_CH4_KG
    dry_mass_kg: float = DEFAULT_DRY_MASS_KG
    altitude_m: float = 0.0
    velocity_m_s: float = 0.0
    delta_v_m_s: float = 0.0
    burn_time_s: float = 0.0
    wall_temp_k: float = 300.0
    engine_running: bool = False
    engine_failed: bool = False
    total_lox_consumed_kg: float = 0.0
    total_ch4_consumed_kg: float = 0.0
    total_impulse_ns: float = 0.0
    peak_acceleration_g: float = 0.0
    peak_altitude_m: float = 0.0
    gravity_loss_m_s: float = 0.0
    drag_loss_m_s: float = 0.0

    @property
    def total_mass_kg(self) -> float:
        return self.dry_mass_kg + self.lox_kg + self.ch4_kg

    @property
    def propellant_kg(self) -> float:
        return self.lox_kg + self.ch4_kg

    @property
    def mass_ratio(self) -> float:
        if self.dry_mass_kg <= 0.0:
            return 1.0
        return self.total_mass_kg / self.dry_mass_kg


@dataclass
class TickResult:
    """Output of a single engine tick (1 second)."""
    tick: int = 0
    thrust_n: float = 0.0
    isp_s: float = 0.0
    mass_flow_kg_s: float = 0.0
    lox_flow_kg_s: float = 0.0
    ch4_flow_kg_s: float = 0.0
    exhaust_velocity_m_s: float = 0.0
    acceleration_m_s2: float = 0.0
    acceleration_g: float = 0.0
    altitude_m: float = 0.0
    velocity_m_s: float = 0.0
    delta_v_m_s: float = 0.0
    ambient_pressure_pa: float = 0.0
    wall_temp_k: float = 0.0
    propellant_remaining_kg: float = 0.0
    gravity_loss_m_s: float = 0.0
    drag_loss_m_s: float = 0.0
    engine_running: bool = False
    engine_failed: bool = False


# -- Engine factory -----------------------------------------------------------

def create_engine(
    thrust_n: float = DEFAULT_THRUST_N,
    lox_kg: float = DEFAULT_LOX_KG,
    ch4_kg: float = DEFAULT_CH4_KG,
    dry_mass_kg: float = DEFAULT_DRY_MASS_KG,
) -> "tuple[EngineConfig, VehicleState]":
    """Create a default MAV engine + vehicle."""
    config = EngineConfig(thrust_target_n=thrust_n)
    state = VehicleState(lox_kg=lox_kg, ch4_kg=ch4_kg, dry_mass_kg=dry_mass_kg)
    return config, state


# -- Tick engine --------------------------------------------------------------

def tick(config: EngineConfig, state: VehicleState,
         dt_s: float = 1.0) -> TickResult:
    """Advance the engine by one tick (default 1 second)."""
    result = TickResult(tick=int(state.burn_time_s))

    if state.engine_failed:
        result.engine_failed = True
        result.altitude_m = state.altitude_m
        result.velocity_m_s = state.velocity_m_s
        result.propellant_remaining_kg = state.propellant_kg
        return result

    if state.lox_kg <= 0.0 or state.ch4_kg <= 0.0:
        state.engine_running = False
        result.altitude_m = state.altitude_m
        result.velocity_m_s = state.velocity_m_s
        result.propellant_remaining_kg = state.propellant_kg
        return result

    if not state.engine_running:
        state.engine_running = True
        state.wall_temp_k = 300.0

    p_amb = atmospheric_pressure_pa(state.altitude_m)
    result.ambient_pressure_pa = p_amb

    p_exit = exit_pressure_pa(config.chamber_pressure_pa,
                              config.nozzle_area_ratio, EXHAUST_GAMMA)
    exit_area = config.throat_area_m2 * config.nozzle_area_ratio

    v_e = exhaust_velocity_m_s(config.chamber_temp_k,
                               config.chamber_pressure_pa, p_exit)
    result.exhaust_velocity_m_s = v_e

    # Compute mass flow directly: F = mdot*Ve + (Pe-Pamb)*Ae
    pressure_thrust = (p_exit - p_amb) * exit_area
    if v_e > 0.0:
        mdot = (config.thrust_target_n - pressure_thrust) / v_e
        mdot = max(mdot, 0.0)
    else:
        mdot = 0.0

    f = thrust_n(v_e, p_exit, p_amb, exit_area, mdot)
    isp = specific_impulse_s(v_e, p_exit, p_amb, exit_area, mdot)

    result.thrust_n = f
    result.isp_s = isp
    result.mass_flow_kg_s = mdot

    of_ratio = config.mixture_ratio
    lox_flow = mdot * of_ratio / (1.0 + of_ratio)
    ch4_flow = mdot / (1.0 + of_ratio)
    result.lox_flow_kg_s = lox_flow
    result.ch4_flow_kg_s = ch4_flow

    lox_consumed = min(lox_flow * dt_s, state.lox_kg)
    ch4_consumed = min(ch4_flow * dt_s, state.ch4_kg)
    state.lox_kg -= lox_consumed
    state.ch4_kg -= ch4_consumed
    state.total_lox_consumed_kg += lox_consumed
    state.total_ch4_consumed_kg += ch4_consumed

    # Chamber wall thermal management (Newton's law of cooling)
    heat_in = WALL_HEAT_COEFF * (config.chamber_temp_k - state.wall_temp_k) * dt_s
    cool_out = WALL_COOL_COEFF * (state.wall_temp_k - COOLANT_TEMP_K) * dt_s
    state.wall_temp_k += heat_in - cool_out
    state.wall_temp_k = max(state.wall_temp_k, COOLANT_TEMP_K)
    result.wall_temp_k = state.wall_temp_k

    if state.wall_temp_k > config.max_wall_temp_k:
        state.engine_failed = True
        state.engine_running = False
        result.engine_failed = True
        result.engine_running = False
        result.altitude_m = state.altitude_m
        result.velocity_m_s = state.velocity_m_s
        result.propellant_remaining_kg = state.propellant_kg
        return result

    total_mass = state.total_mass_kg
    if total_mass <= 0.0:
        total_mass = 1.0

    g_local = gravity_at_altitude_m_s2(state.altitude_m)
    f_drag = drag_force_n(state.velocity_m_s, state.altitude_m, 3.0)

    accel = (f - total_mass * g_local - f_drag) / total_mass
    result.acceleration_m_s2 = accel
    result.acceleration_g = accel / G0_M_S2

    state.gravity_loss_m_s += g_local * dt_s
    state.drag_loss_m_s += (f_drag / total_mass * dt_s if total_mass > 0 else 0.0)
    result.gravity_loss_m_s = state.gravity_loss_m_s
    result.drag_loss_m_s = state.drag_loss_m_s

    state.velocity_m_s += accel * dt_s
    state.velocity_m_s = max(state.velocity_m_s, 0.0)
    state.altitude_m += state.velocity_m_s * dt_s
    state.altitude_m = max(state.altitude_m, 0.0)

    if total_mass > 0 and mdot > 0:
        state.delta_v_m_s += (f / total_mass) * dt_s
    result.delta_v_m_s = state.delta_v_m_s

    state.burn_time_s += dt_s
    state.total_impulse_ns += f * dt_s
    state.peak_acceleration_g = max(state.peak_acceleration_g, abs(accel) / G0_M_S2)
    state.peak_altitude_m = max(state.peak_altitude_m, state.altitude_m)

    result.altitude_m = state.altitude_m
    result.velocity_m_s = state.velocity_m_s
    result.propellant_remaining_kg = state.propellant_kg
    result.engine_running = state.engine_running
    return result


# -- Simulation runner --------------------------------------------------------

def run_burn(
    config: "EngineConfig | None" = None,
    state: "VehicleState | None" = None,
    max_seconds: int = 600,
    dt_s: float = 1.0,
) -> "list[TickResult]":
    """Run a complete engine burn until propellant exhaustion or timeout."""
    if config is None or state is None:
        config, state = create_engine()
    results = []
    for _ in range(max_seconds):
        result = tick(config, state, dt_s)
        results.append(result)
        if not state.engine_running or state.engine_failed:
            break
        if state.propellant_kg <= 0.0:
            break
    return results


def can_reach_orbit(
    config: "EngineConfig | None" = None,
    state: "VehicleState | None" = None,
    target_dv_m_s: float = MARS_ORBIT_DV_M_S,
) -> "dict[str, Any]":
    """Check if the MAV can reach Mars orbit with current propellant."""
    if config is None or state is None:
        config, state = create_engine()

    p_exit = exit_pressure_pa(config.chamber_pressure_pa,
                              config.nozzle_area_ratio)
    v_e = exhaust_velocity_m_s(config.chamber_temp_k,
                               config.chamber_pressure_pa, p_exit)
    exit_area = config.throat_area_m2 * config.nozzle_area_ratio
    mdot_est = (config.thrust_target_n - (p_exit * exit_area)) / v_e if v_e > 0 else 1.0
    isp_vac = specific_impulse_s(v_e, p_exit, 0.0, exit_area, max(mdot_est, 0.01))

    ideal_dv = tsiolkovsky_delta_v(isp_vac, state.total_mass_kg, state.dry_mass_kg)
    est_burn_time = state.propellant_kg / max(mdot_est, 0.01)
    est_gravity_loss = GRAVITY_LOSS_M_S_PER_S * min(est_burn_time, 300.0)
    effective_dv = ideal_dv - est_gravity_loss - DRAG_LOSS_TOTAL_M_S

    return {
        "ideal_delta_v_m_s": round(ideal_dv, 1),
        "isp_vacuum_s": round(isp_vac, 1),
        "estimated_gravity_loss_m_s": round(est_gravity_loss, 1),
        "estimated_drag_loss_m_s": round(DRAG_LOSS_TOTAL_M_S, 1),
        "effective_delta_v_m_s": round(effective_dv, 1),
        "target_delta_v_m_s": target_dv_m_s,
        "margin_m_s": round(effective_dv - target_dv_m_s, 1),
        "go": effective_dv >= target_dv_m_s,
        "mass_ratio": round(state.mass_ratio, 2),
        "propellant_kg": round(state.propellant_kg, 1),
        "burn_time_estimate_s": round(est_burn_time, 1),
    }


def run_simulation(
    sols: int = 1,
    config: "EngineConfig | None" = None,
    state: "VehicleState | None" = None,
) -> "dict[str, Any]":
    """Run a complete MAV launch simulation."""
    if config is None or state is None:
        config, state = create_engine()

    preflight = can_reach_orbit(config, state)
    trajectory = run_burn(config, state, max_seconds=600)

    return {
        "preflight": preflight,
        "burn_ticks": len(trajectory),
        "final_altitude_m": round(state.altitude_m, 1),
        "final_velocity_m_s": round(state.velocity_m_s, 1),
        "total_delta_v_m_s": round(state.delta_v_m_s, 1),
        "peak_altitude_m": round(state.peak_altitude_m, 1),
        "peak_acceleration_g": round(state.peak_acceleration_g, 2),
        "total_impulse_ns": round(state.total_impulse_ns, 1),
        "lox_consumed_kg": round(state.total_lox_consumed_kg, 1),
        "ch4_consumed_kg": round(state.total_ch4_consumed_kg, 1),
        "gravity_loss_m_s": round(state.gravity_loss_m_s, 1),
        "drag_loss_m_s": round(state.drag_loss_m_s, 1),
        "engine_failed": state.engine_failed,
        "propellant_remaining_kg": round(state.propellant_kg, 1),
    }
