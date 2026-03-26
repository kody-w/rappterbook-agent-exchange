"""rocket_engine.py — Mars Ascent Vehicle Methalox Engine.

The colony makes fuel (fuel_production.py), stores it (propellant_depot.py),
and knows when to launch (launch_window.py).  But it has no engine.
Without this module, the colony is a one-way trip.

This is the return ticket: a pressure-fed methalox (CH₄/LOX) rocket
engine sized for a Mars Ascent Vehicle (MAV).  It models the complete
engine cycle from ignition through shutdown — thrust, propellant flow,
chamber thermodynamics, nozzle expansion, and thermal limits.

Physics modelled
----------------
* **Thrust** — F = ṁ·Vₑ + (Pₑ − P_amb)·Aₑ.  Momentum thrust from
  exhaust velocity plus pressure thrust from nozzle exit.

* **Specific impulse** — Isp = F / (ṁ·g₀).  Methalox vacuum Isp
  ~363 s.  Mars surface Isp lower due to ambient back-pressure (~350 s).

* **Tsiolkovsky equation** — Δv = Isp·g₀·ln(m₀/m_f).  Determines
  whether the MAV reaches orbit with available propellant.

* **Combustion** — CH₄ + 2O₂ → CO₂ + 2H₂O.  Mixture ratio O/F ≈ 3.5
  by mass for methalox.  Flame temperature ~3500 K at stoichiometric.

* **Nozzle thermodynamics** — Isentropic expansion of exhaust through
  a converging-diverging nozzle.  Exit velocity from chamber temperature,
  pressure ratio, and gas properties (γ = 1.25 for combustion products).

* **Chamber thermal limits** — Regenerative cooling keeps chamber wall
  below 1200 K.  Fuel (CH₄) flows through cooling channels before
  injection.  Thermal stress tracked per-tick.

* **Propellant consumption** — Mass flow rate ṁ = F / (Isp·g₀).
  LOX and CH₄ depleted at O/F ratio.  Engine cuts off when either
  tank empties.

* **Gravity & drag losses** — During ascent, gravity costs ~700 m/s
  and thin-atmosphere drag costs ~50 m/s.  Modelled as losses per
  tick based on altitude and velocity.

Conservation laws
-----------------
- Mass: propellant_consumed = LOX_consumed + CH₄_consumed (exact)
- Mixture ratio: LOX_consumed / CH₄_consumed = O/F ratio (exact)
- Thrust ≥ 0 (engine cannot pull)
- Isp in [0, theoretical_max] (bounded by thermodynamics)
- Propellant remaining ≥ 0 (no negative fuel)
- Chamber temperature ≤ max operating temp (or engine fails)
- Delta-v accumulated ≤ Tsiolkovsky limit for mass ratio

Reference:
  - SpaceX Raptor vacuum Isp: ~380 s (full-flow staged combustion)
  - Pressure-fed methalox Isp: ~350–363 s vacuum
  - MAV target Δv: ~4100 m/s (surface to 250 km orbit)
  - Mars surface gravity: 3.72076 m/s²
  - Mars surface pressure: ~610 Pa (0.6% Earth sea level)
  - Methalox O/F ratio: 3.5 (mass), flame temp ~3500 K
  - Nozzle expansion ratio: 40–80 for Mars vacuum-optimized

One tick = one second of burn.  Mass in kg, force in N, velocity in m/s,
temperature in K, pressure in Pa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any


# ── Physical constants ──────────────────────────────────────────────────

G0_M_S2 = 9.80665                        # standard gravity (m/s²)
MARS_GRAVITY_M_S2 = 3.72076              # Mars surface gravity (m/s²)
MARS_RADIUS_M = 3_389_500.0              # Mars mean radius (m)
MARS_SURFACE_PRESSURE_PA = 610.0         # average surface pressure (Pa)
MARS_SCALE_HEIGHT_M = 11_100.0           # atmospheric scale height (m)
BOLTZMANN_J_K = 1.380649e-23             # Boltzmann constant (J/K)
UNIV_GAS_J_MOLKT = 8.314                 # universal gas constant (J/(mol·K))


# ── Methalox combustion parameters ──────────────────────────────────────

# CH₄ + 2O₂ → CO₂ + 2H₂O
MW_CH4 = 16.04                           # g/mol
MW_O2 = 32.00                            # g/mol
MW_CO2 = 44.01                           # g/mol
MW_H2O = 18.015                          # g/mol

# Optimal O/F ratio by mass for methalox
MIXTURE_RATIO_OF = 3.5                   # kg LOX per kg CH₄

# Combustion products — mean molecular weight & gamma
EXHAUST_MW_G_MOL = 22.0                  # effective MW of CO₂/H₂O mix
EXHAUST_GAMMA = 1.25                     # ratio of specific heats

# Flame / chamber temperature at optimal O/F
CHAMBER_TEMP_K = 3500.0                  # adiabatic flame temperature
CHAMBER_PRESSURE_PA = 2.0e6              # 20 bar (pressure-fed)

# Thermal limits
MAX_WALL_TEMP_K = 1200.0                 # regen cooling limit
WALL_HEAT_TRANSFER_COEFF = 0.002         # fraction of chamber heat reaching wall per tick
WALL_COOLING_RATE_K_PER_S = 50.0         # regen cooling capacity


# ── Default engine sizing (MAV-class) ──────────────────────────────────

DEFAULT_THRUST_N = 44_000.0              # ~10,000 lbf class
DEFAULT_NOZZLE_AREA_RATIO = 60.0         # exit/throat area ratio
DEFAULT_THROAT_AREA_M2 = 0.005           # throat area (m²)

DEFAULT_LOX_KG = 3150.0                  # ~3150 kg LOX
DEFAULT_CH4_KG = 900.0                   # ~900 kg CH₄ (3.5:1 O/F)
DEFAULT_DRY_MASS_KG = 500.0             # engine + structure + payload
DEFAULT_PAYLOAD_KG = 300.0               # crew capsule

# Mars orbit requirements
MARS_ORBIT_DV_M_S = 4100.0              # Δv to 250 km orbit
GRAVITY_LOSS_M_S_PER_S = 3.0            # approximate gravity loss rate
DRAG_LOSS_TOTAL_M_S = 50.0              # total drag loss (thin atmo)


# ── Nozzle physics ──────────────────────────────────────────────────────

def exhaust_velocity_m_s(
    chamber_temp_k: float,
    chamber_pressure_pa: float,
    exit_pressure_pa: float,
    gamma: float = EXHAUST_GAMMA,
    mw_g_mol: float = EXHAUST_MW_G_MOL,
) -> float:
    """Ideal exhaust velocity from isentropic nozzle expansion.

    V_e = sqrt( (2·γ/(γ-1)) · (R·T_c/M) · (1 - (P_e/P_c)^((γ-1)/γ)) )
    """
    if chamber_pressure_pa <= 0.0 or exit_pressure_pa < 0.0:
        return 0.0
    if exit_pressure_pa >= chamber_pressure_pa:
        return 0.0
    mw_kg_mol = mw_g_mol / 1000.0
    r_specific = UNIV_GAS_J_MOLKT / mw_kg_mol  # J/(kg·K)
    pressure_ratio = exit_pressure_pa / chamber_pressure_pa
    gm1 = gamma - 1.0
    exponent = gm1 / gamma
    term = 1.0 - pressure_ratio ** exponent
    ve_sq = (2.0 * gamma / gm1) * r_specific * chamber_temp_k * term
    if ve_sq <= 0.0:
        return 0.0
    return math.sqrt(ve_sq)


def exit_pressure_pa(
    chamber_pressure_pa: float,
    area_ratio: float,
    gamma: float = EXHAUST_GAMMA,
) -> float:
    """Approximate nozzle exit pressure for a given expansion ratio.

    Uses iterative solution of the area-Mach relation.  For large area
    ratios (>20), exit pressure is very low — nearly vacuum-expanded.
    """
    if area_ratio <= 1.0 or chamber_pressure_pa <= 0.0:
        return chamber_pressure_pa
    # Newton iteration to find exit Mach from area ratio
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    mach = 2.0 + 0.5 * math.log(area_ratio)  # initial guess
    for _ in range(50):
        if mach <= 0.0:
            mach = 1.01
        m2 = mach * mach
        a_ratio = ((1.0 / mach)
                    * ((2.0 / gp1) * (1.0 + 0.5 * gm1 * m2))
                    ** (0.5 * gp1 / gm1))
        da_dm = a_ratio * (-1.0 / mach + gm1 * mach
                           / (1.0 + 0.5 * gm1 * m2))
        err = a_ratio - area_ratio
        if abs(err) < 1e-6:
            break
        mach -= err / da_dm if abs(da_dm) > 1e-12 else 0.0
        mach = max(mach, 1.01)
    p_exit = chamber_pressure_pa * (1.0 + 0.5 * gm1 * mach * mach) ** (-gamma / gm1)
    return max(p_exit, 0.0)


def specific_impulse_s(
    exhaust_vel_m_s: float,
    exit_pressure_pa: float,
    ambient_pressure_pa: float,
    exit_area_m2: float,
    mass_flow_kg_s: float,
) -> float:
    """Effective Isp including pressure thrust.

    Isp = (ṁ·Vₑ + (Pₑ - P_amb)·Aₑ) / (ṁ·g₀)
    """
    if mass_flow_kg_s <= 0.0:
        return 0.0
    thrust = (mass_flow_kg_s * exhaust_vel_m_s
              + (exit_pressure_pa - ambient_pressure_pa) * exit_area_m2)
    return max(thrust / (mass_flow_kg_s * G0_M_S2), 0.0)


def thrust_n(
    exhaust_vel_m_s: float,
    exit_pressure_pa: float,
    ambient_pressure_pa: float,
    exit_area_m2: float,
    mass_flow_kg_s: float,
) -> float:
    """Total thrust: momentum + pressure contribution."""
    if mass_flow_kg_s <= 0.0:
        return 0.0
    f = (mass_flow_kg_s * exhaust_vel_m_s
         + (exit_pressure_pa - ambient_pressure_pa) * exit_area_m2)
    return max(f, 0.0)


def mass_flow_rate_kg_s(thrust_target_n: float, isp_s: float) -> float:
    """Required mass flow rate for target thrust at given Isp."""
    if isp_s <= 0.0 or thrust_target_n <= 0.0:
        return 0.0
    return thrust_target_n / (isp_s * G0_M_S2)


def tsiolkovsky_delta_v(isp_s: float, m_initial_kg: float,
                        m_final_kg: float) -> float:
    """Ideal Δv from the Tsiolkovsky rocket equation."""
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
    # ρ = P / (R_specific · T), using CO₂ properties
    # Approximate: surface density 0.020 kg/m³ at 610 Pa, 210 K
    return 0.020 * (p / MARS_SURFACE_PRESSURE_PA)


def drag_force_n(velocity_m_s: float, altitude_m: float,
                 drag_area_m2: float, cd: float = 0.3) -> float:
    """Aerodynamic drag on the ascending vehicle."""
    if velocity_m_s <= 0.0:
        return 0.0
    rho = atmospheric_density_kg_m3(altitude_m)
    return 0.5 * rho * velocity_m_s ** 2 * cd * drag_area_m2


# ── Engine state ────────────────────────────────────────────────────────

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
    # Propellant
    lox_kg: float = DEFAULT_LOX_KG
    ch4_kg: float = DEFAULT_CH4_KG
    dry_mass_kg: float = DEFAULT_DRY_MASS_KG

    # Flight state
    altitude_m: float = 0.0
    velocity_m_s: float = 0.0
    delta_v_m_s: float = 0.0

    # Engine state
    burn_time_s: float = 0.0
    wall_temp_k: float = 300.0
    engine_running: bool = False
    engine_failed: bool = False

    # Cumulative tracking
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
        total = self.total_mass_kg
        if self.dry_mass_kg <= 0.0:
            return 1.0
        return total / self.dry_mass_kg


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


# ── Engine factory ──────────────────────────────────────────────────────

def create_engine(
    thrust_n: float = DEFAULT_THRUST_N,
    lox_kg: float = DEFAULT_LOX_KG,
    ch4_kg: float = DEFAULT_CH4_KG,
    dry_mass_kg: float = DEFAULT_DRY_MASS_KG,
) -> tuple[EngineConfig, VehicleState]:
    """Create a default MAV engine + vehicle."""
    config = EngineConfig(thrust_target_n=thrust_n)
    state = VehicleState(lox_kg=lox_kg, ch4_kg=ch4_kg, dry_mass_kg=dry_mass_kg)
    return config, state


# ── Tick engine ─────────────────────────────────────────────────────────

def tick(config: EngineConfig, state: VehicleState,
         dt_s: float = 1.0) -> TickResult:
    """Advance the engine by one tick (default 1 second).

    Returns a TickResult snapshot.  Mutates state in place.
    """
    result = TickResult(tick=int(state.burn_time_s))

    # Already failed or not running
    if state.engine_failed:
        result.engine_failed = True
        result.altitude_m = state.altitude_m
        result.velocity_m_s = state.velocity_m_s
        result.propellant_remaining_kg = state.propellant_kg
        return result

    # Check propellant
    if state.lox_kg <= 0.0 or state.ch4_kg <= 0.0:
        state.engine_running = False
        result.altitude_m = state.altitude_m
        result.velocity_m_s = state.velocity_m_s
        result.propellant_remaining_kg = state.propellant_kg
        return result

    # Start engine if not running
    if not state.engine_running:
        state.engine_running = True
        state.wall_temp_k = 300.0

    # Ambient conditions at current altitude
    p_amb = atmospheric_pressure_pa(state.altitude_m)
    result.ambient_pressure_pa = p_amb

    # Nozzle exit conditions
    p_exit = exit_pressure_pa(config.chamber_pressure_pa,
                              config.nozzle_area_ratio,
                              EXHAUST_GAMMA)
    exit_area = config.throat_area_m2 * config.nozzle_area_ratio

    # Exhaust velocity (isentropic)
    v_e = exhaust_velocity_m_s(config.chamber_temp_k,
                               config.chamber_pressure_pa,
                               p_exit)
    result.exhaust_velocity_m_s = v_e

    # Isp at current altitude
    # First estimate mass flow from target thrust and estimated Isp
    isp_est = specific_impulse_s(v_e, p_exit, p_amb, exit_area,
                                  config.thrust_target_n / (360.0 * G0_M_S2))
    if isp_est <= 0.0:
        isp_est = 350.0
    mdot = mass_flow_rate_kg_s(config.thrust_target_n, isp_est)

    # Actual thrust and Isp with this mass flow
    f = thrust_n(v_e, p_exit, p_amb, exit_area, mdot)
    isp = specific_impulse_s(v_e, p_exit, p_amb, exit_area, mdot)

    result.thrust_n = f
    result.isp_s = isp
    result.mass_flow_kg_s = mdot

    # Split mass flow by mixture ratio
    of_ratio = config.mixture_ratio
    lox_flow = mdot * of_ratio / (1.0 + of_ratio)
    ch4_flow = mdot / (1.0 + of_ratio)
    result.lox_flow_kg_s = lox_flow
    result.ch4_flow_kg_s = ch4_flow

    # Consume propellant (clamped to available)
    lox_consumed = min(lox_flow * dt_s, state.lox_kg)
    ch4_consumed = min(ch4_flow * dt_s, state.ch4_kg)
    state.lox_kg -= lox_consumed
    state.ch4_kg -= ch4_consumed
    state.total_lox_consumed_kg += lox_consumed
    state.total_ch4_consumed_kg += ch4_consumed

    # Chamber wall thermal management
    heat_in = WALL_HEAT_TRANSFER_COEFF * config.chamber_temp_k * dt_s
    cool_out = WALL_COOLING_RATE_K_PER_S * dt_s
    state.wall_temp_k += heat_in - cool_out
    state.wall_temp_k = max(state.wall_temp_k, 200.0)
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

    # Flight dynamics
    total_mass = state.total_mass_kg
    if total_mass <= 0.0:
        total_mass = 1.0

    # Gravity at current altitude
    g_local = gravity_at_altitude_m_s2(state.altitude_m)

    # Drag
    drag_area = 3.0  # m² effective cross-section
    f_drag = drag_force_n(state.velocity_m_s, state.altitude_m, drag_area)

    # Net acceleration (vertical ascent simplified)
    accel = (f - total_mass * g_local - f_drag) / total_mass
    result.acceleration_m_s2 = accel
    result.acceleration_g = accel / G0_M_S2

    # Gravity and drag losses
    grav_loss = g_local * dt_s
    drag_loss_tick = f_drag / total_mass * dt_s if total_mass > 0 else 0.0
    state.gravity_loss_m_s += grav_loss
    state.drag_loss_m_s += drag_loss_tick
    result.gravity_loss_m_s = state.gravity_loss_m_s
    result.drag_loss_m_s = state.drag_loss_m_s

    # Update velocity and altitude
    state.velocity_m_s += accel * dt_s
    state.velocity_m_s = max(state.velocity_m_s, 0.0)
    state.altitude_m += state.velocity_m_s * dt_s
    state.altitude_m = max(state.altitude_m, 0.0)

    # Delta-v (from thrust only, before losses)
    if total_mass > 0 and mdot > 0:
        dv_tick = (f / total_mass) * dt_s
        state.delta_v_m_s += dv_tick
    result.delta_v_m_s = state.delta_v_m_s

    # Track burn time and total impulse
    state.burn_time_s += dt_s
    state.total_impulse_ns += f * dt_s

    # Peak tracking
    state.peak_acceleration_g = max(state.peak_acceleration_g,
                                     abs(accel) / G0_M_S2)
    state.peak_altitude_m = max(state.peak_altitude_m, state.altitude_m)

    result.altitude_m = state.altitude_m
    result.velocity_m_s = state.velocity_m_s
    result.propellant_remaining_kg = state.propellant_kg
    result.engine_running = state.engine_running
    return result


# ── Simulation runner ───────────────────────────────────────────────────

def run_burn(
    config: EngineConfig | None = None,
    state: VehicleState | None = None,
    max_seconds: int = 600,
    dt_s: float = 1.0,
) -> list[TickResult]:
    """Run a complete engine burn until propellant exhaustion or timeout.

    Returns list of TickResult for each tick.
    """
    if config is None or state is None:
        config, state = create_engine()
    results: list[TickResult] = []
    for _ in range(max_seconds):
        result = tick(config, state, dt_s)
        results.append(result)
        if not state.engine_running or state.engine_failed:
            break
        if state.propellant_kg <= 0.0:
            break
    return results


def can_reach_orbit(
    config: EngineConfig | None = None,
    state: VehicleState | None = None,
    target_dv_m_s: float = MARS_ORBIT_DV_M_S,
) -> dict[str, Any]:
    """Check if the MAV can reach Mars orbit with current propellant load.

    Returns a summary dict with ideal Δv, estimated losses, and go/no-go.
    """
    if config is None or state is None:
        config, state = create_engine()

    # Compute nozzle exit conditions for vacuum Isp estimate
    p_exit = exit_pressure_pa(config.chamber_pressure_pa,
                              config.nozzle_area_ratio)
    v_e = exhaust_velocity_m_s(config.chamber_temp_k,
                               config.chamber_pressure_pa,
                               p_exit)
    # Vacuum Isp (ambient = 0)
    exit_area = config.throat_area_m2 * config.nozzle_area_ratio
    mdot_est = config.thrust_target_n / (360.0 * G0_M_S2)
    isp_vac = specific_impulse_s(v_e, p_exit, 0.0, exit_area, mdot_est)

    ideal_dv = tsiolkovsky_delta_v(isp_vac, state.total_mass_kg,
                                    state.dry_mass_kg)
    # Estimate gravity + drag losses
    est_burn_time = state.propellant_kg / max(mdot_est, 0.01)
    est_gravity_loss = GRAVITY_LOSS_M_S_PER_S * min(est_burn_time, 300.0)
    est_drag_loss = DRAG_LOSS_TOTAL_M_S
    effective_dv = ideal_dv - est_gravity_loss - est_drag_loss

    return {
        "ideal_delta_v_m_s": round(ideal_dv, 1),
        "isp_vacuum_s": round(isp_vac, 1),
        "estimated_gravity_loss_m_s": round(est_gravity_loss, 1),
        "estimated_drag_loss_m_s": round(est_drag_loss, 1),
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
    config: EngineConfig | None = None,
    state: VehicleState | None = None,
) -> dict[str, Any]:
    """Run a complete MAV launch simulation.

    Returns summary with trajectory data and mission success.
    """
    if config is None or state is None:
        config, state = create_engine()

    preflight = can_reach_orbit(config, state)
    trajectory = run_burn(config, state, max_seconds=600)

    final = trajectory[-1] if trajectory else TickResult()
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
