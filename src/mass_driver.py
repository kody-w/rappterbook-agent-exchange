"""mass_driver.py -- Mars Electromagnetic Launch Rail (Mass Driver).

The colony has a rocket engine for crewed ascent (rocket_engine.py) and
a propellant depot (propellant_depot.py) for fuel storage.  But burning
methalox to loft every kilogram of cargo is ruinously expensive.

This module is the freight elevator: an electromagnetic mass driver that
accelerates cargo sleds along a track and flings them to Mars orbit,
Phobos, or Earth-transfer trajectories -- zero propellant consumed.
Power comes from the colony's nuclear reactor and solar arrays.

Physics modelled
----------------
* Lorentz force: F = n * I * L_coil * B, where n = active coils.
  The sled rides a sequence of superconducting coil stages that fire
  in rapid succession (like a coilgun / linear synchronous motor).

* Kinetic energy: E_k = 0.5 * m * v^2 (joules delivered to payload).

* Electrical-to-kinetic efficiency: eta = E_k / E_in.  Real linear
  accelerators achieve 50-80 pct; we model stage-by-stage losses.

* Track dynamics: constant-force approximation per stage.
  a = F/m, v^2 = v0^2 + 2*a*d per stage.  Cumulative velocity along
  the full track.

* Atmospheric drag: F_drag = 0.5 * rho * v^2 * C_d * A.  Mars rho is
  about 0.020 kg/m^3 at surface; exponential decay with altitude.

* G-load: payload peak acceleration must stay below structural
  limits (typically 20-50 g for hardened cargo, 3 g for fragile).

* Track thermal load: resistive heating per stage, Q = I^2 * R * t.
  Track must cool between launches.

* Magnetic field energy: E_B = B^2 / (2 * mu_0) per unit volume.

* Launch geometry: track is built on a slope (Olympus Mons flank
  or a constructed ramp), elevation angle theta determines how much
  velocity goes vertical vs horizontal.

* Orbital mechanics: required velocity at rail exit depends on
  target orbit.  Low Mars orbit ~ 3,550 m/s.  Mars escape ~ 5,030 m/s.
  Earth transfer ~ 5,700 m/s.

Conservation laws
-----------------
- Energy: E_electrical_in >= E_kinetic + E_drag_loss + E_thermal_loss
- Momentum: F * dt = m * dv per stage (impulse-momentum theorem)
- Mass: payload mass is constant during acceleration
- Temperature: track segments heat during launch, cool between launches
- Velocity: monotonically non-decreasing along track (no braking zones)

One tick = one coil stage firing (milliseconds).  A full launch is
hundreds of ticks.  Mass in kg, force in N, velocity in m/s, energy
in joules, distance in meters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# -- Physical constants -------------------------------------------------------

G0_M_S2 = 9.80665
MARS_GRAVITY_M_S2 = 3.72076
MARS_SURFACE_PRESSURE_PA = 610.0
MARS_SURFACE_DENSITY_KG_M3 = 0.020
MARS_SCALE_HEIGHT_M = 11_100.0
MARS_SURFACE_TEMP_K = 210.0
VACUUM_PERMEABILITY_H_M = 4.0e-7 * math.pi  # mu_0

# -- Orbital targets (required velocity at rail exit) -------------------------

LMO_VELOCITY_M_S = 3550.0        # Low Mars Orbit
ESCAPE_VELOCITY_M_S = 5030.0     # Mars escape
EARTH_TRANSFER_M_S = 5700.0      # Trans-Earth injection

# -- Default track geometry ---------------------------------------------------

DEFAULT_TRACK_LENGTH_M = 2000.0
DEFAULT_ELEVATION_DEG = 30.0
DEFAULT_NUM_STAGES = 200
DEFAULT_COIL_LENGTH_M = 0.5      # each coil stage length
DEFAULT_COILS_PER_STAGE = 4

# -- Electromagnetic parameters -----------------------------------------------

DEFAULT_CURRENT_A = 50_000.0     # per coil
DEFAULT_MAGNETIC_FIELD_T = 2.0   # superconducting coils
DEFAULT_COIL_RESISTANCE_OHM = 0.001  # superconducting, near-zero

# -- Efficiency and thermal ---------------------------------------------------

ELECTRICAL_EFFICIENCY = 0.65     # E_kinetic / E_electrical
DRAG_COEFF = 0.15               # streamlined sled
SLED_CROSS_SECTION_M2 = 0.5     # frontal area of cargo sled
MAX_TRACK_TEMP_K = 400.0        # track shutdown threshold
AMBIENT_TRACK_TEMP_K = MARS_SURFACE_TEMP_K
TRACK_THERMAL_MASS_J_PER_K = 500_000.0  # per stage segment
TRACK_COOLING_RATE_W = 2000.0   # radiative + conductive cooling per segment

# -- G-load limits ------------------------------------------------------------

HARDENED_CARGO_G_LIMIT = 50.0
FRAGILE_CARGO_G_LIMIT = 3.0
DEFAULT_G_LIMIT = 30.0

# -- Default payload ----------------------------------------------------------

DEFAULT_PAYLOAD_KG = 100.0
DEFAULT_SLED_KG = 50.0          # reusable sled (stays on track or is caught)


# -- Pure physics functions ---------------------------------------------------

def lorentz_force_n(
    current_a: float,
    coil_length_m: float,
    magnetic_field_t: float,
    num_coils: int = 1,
) -> float:
    """Lorentz force from active coils: F = n * I * L * B."""
    if current_a <= 0.0 or coil_length_m <= 0.0 or magnetic_field_t <= 0.0:
        return 0.0
    return float(num_coils) * current_a * coil_length_m * magnetic_field_t


def kinetic_energy_j(mass_kg: float, velocity_m_s: float) -> float:
    """Kinetic energy: E = 0.5 * m * v^2."""
    if mass_kg <= 0.0:
        return 0.0
    return 0.5 * mass_kg * velocity_m_s ** 2


def velocity_from_energy_m_s(energy_j: float, mass_kg: float) -> float:
    """Velocity from kinetic energy: v = sqrt(2E/m)."""
    if energy_j <= 0.0 or mass_kg <= 0.0:
        return 0.0
    return math.sqrt(2.0 * energy_j / mass_kg)


def mars_air_density_kg_m3(altitude_m: float) -> float:
    """Mars atmospheric density at altitude (exponential model)."""
    if altitude_m < 0.0:
        altitude_m = 0.0
    return MARS_SURFACE_DENSITY_KG_M3 * math.exp(-altitude_m / MARS_SCALE_HEIGHT_M)


def drag_force_n(
    velocity_m_s: float,
    altitude_m: float = 0.0,
    drag_coeff: float = DRAG_COEFF,
    cross_section_m2: float = SLED_CROSS_SECTION_M2,
) -> float:
    """Aerodynamic drag: F = 0.5 * rho * v^2 * C_d * A."""
    if velocity_m_s <= 0.0:
        return 0.0
    rho = mars_air_density_kg_m3(altitude_m)
    return 0.5 * rho * velocity_m_s ** 2 * drag_coeff * cross_section_m2


def gravity_component_n(mass_kg: float, elevation_deg: float) -> float:
    """Gravity force along track (opposing motion on upward slope)."""
    if mass_kg <= 0.0:
        return 0.0
    theta = math.radians(max(0.0, min(90.0, elevation_deg)))
    return mass_kg * MARS_GRAVITY_M_S2 * math.sin(theta)


def stage_acceleration_m_s2(
    force_n: float,
    drag_n: float,
    gravity_n: float,
    mass_kg: float,
) -> float:
    """Net acceleration along track for a single stage."""
    if mass_kg <= 0.0:
        return 0.0
    net = force_n - drag_n - gravity_n
    return max(0.0, net / mass_kg)


def exit_velocity_m_s(
    entry_velocity_m_s: float,
    acceleration_m_s2: float,
    stage_length_m: float,
) -> float:
    """Velocity after traversing one stage: v^2 = v0^2 + 2*a*d."""
    if acceleration_m_s2 <= 0.0 and entry_velocity_m_s <= 0.0:
        return 0.0
    v_sq = entry_velocity_m_s ** 2 + 2.0 * acceleration_m_s2 * stage_length_m
    return math.sqrt(max(0.0, v_sq))


def stage_transit_time_s(
    entry_velocity_m_s: float,
    exit_velocity_m_s: float,
    stage_length_m: float,
) -> float:
    """Time to traverse one stage (constant acceleration)."""
    avg_v = 0.5 * (entry_velocity_m_s + exit_velocity_m_s)
    if avg_v <= 0.0:
        return float("inf")
    return stage_length_m / avg_v


def resistive_heat_j(
    current_a: float,
    resistance_ohm: float,
    duration_s: float,
) -> float:
    """Ohmic heating in coils: Q = I^2 * R * t."""
    if current_a <= 0.0 or resistance_ohm <= 0.0 or duration_s <= 0.0:
        return 0.0
    return current_a ** 2 * resistance_ohm * duration_s


def magnetic_field_energy_j(
    field_t: float,
    volume_m3: float,
) -> float:
    """Energy stored in a magnetic field: E = B^2 / (2*mu_0) * V."""
    if field_t <= 0.0 or volume_m3 <= 0.0:
        return 0.0
    return (field_t ** 2 / (2.0 * VACUUM_PERMEABILITY_H_M)) * volume_m3


def altitude_gain_m(track_distance_m: float, elevation_deg: float) -> float:
    """Altitude gained along track."""
    theta = math.radians(max(0.0, min(90.0, elevation_deg)))
    return track_distance_m * math.sin(theta)


def required_exit_velocity_m_s(
    target_velocity_m_s: float,
    elevation_deg: float,
    payload_mass_kg: float,
    track_length_m: float,
) -> float:
    """Exit velocity needed at rail end to reach target after drag/gravity
    losses during atmospheric transit above the rail."""
    alt = altitude_gain_m(track_length_m, elevation_deg)
    drag_loss_estimate = drag_force_n(target_velocity_m_s, alt) * 50.0 / max(payload_mass_kg, 1.0)
    grav_loss = MARS_GRAVITY_M_S2 * math.sin(math.radians(elevation_deg)) * 10.0
    return target_velocity_m_s + drag_loss_estimate + grav_loss


def power_required_w(energy_j: float, duration_s: float) -> float:
    """Average power: P = E / t."""
    if duration_s <= 0.0:
        return float("inf")
    return energy_j / duration_s


def g_load(acceleration_m_s2: float) -> float:
    """Convert acceleration to g-load."""
    return acceleration_m_s2 / G0_M_S2


# -- Track configuration dataclass -------------------------------------------

@dataclass
class TrackConfig:
    """Immutable track parameters."""
    track_length_m: float = DEFAULT_TRACK_LENGTH_M
    elevation_deg: float = DEFAULT_ELEVATION_DEG
    num_stages: int = DEFAULT_NUM_STAGES
    coil_length_m: float = DEFAULT_COIL_LENGTH_M
    coils_per_stage: int = DEFAULT_COILS_PER_STAGE
    current_a: float = DEFAULT_CURRENT_A
    magnetic_field_t: float = DEFAULT_MAGNETIC_FIELD_T
    coil_resistance_ohm: float = DEFAULT_COIL_RESISTANCE_OHM
    efficiency: float = ELECTRICAL_EFFICIENCY
    drag_coeff: float = DRAG_COEFF
    sled_cross_section_m2: float = SLED_CROSS_SECTION_M2
    g_limit: float = DEFAULT_G_LIMIT

    @property
    def stage_length_m(self) -> float:
        if self.num_stages <= 0:
            return 0.0
        return self.track_length_m / self.num_stages

    @property
    def coil_volume_m3(self) -> float:
        r_outer = 0.3
        shell_thickness = 0.05
        return (math.pi * ((r_outer + shell_thickness) ** 2 - r_outer ** 2)
                * self.coil_length_m * self.coils_per_stage)

    @property
    def max_force_n(self) -> float:
        return lorentz_force_n(
            self.current_a, self.coil_length_m,
            self.magnetic_field_t, self.coils_per_stage,
        )


# -- Launch state dataclass --------------------------------------------------

@dataclass
class LaunchState:
    """Mutable state during a launch sequence."""
    payload_mass_kg: float = DEFAULT_PAYLOAD_KG
    sled_mass_kg: float = DEFAULT_SLED_KG
    velocity_m_s: float = 0.0
    position_m: float = 0.0
    altitude_m: float = 0.0
    current_stage: int = 0
    total_energy_in_j: float = 0.0
    total_drag_loss_j: float = 0.0
    total_thermal_loss_j: float = 0.0
    peak_g_load: float = 0.0
    peak_velocity_m_s: float = 0.0
    launch_complete: bool = False
    launch_failed: bool = False
    failure_reason: str = ""
    elapsed_time_s: float = 0.0
    stage_temps_k: list[float] = field(default_factory=list)

    @property
    def total_mass_kg(self) -> float:
        return self.payload_mass_kg + self.sled_mass_kg

    def init_stage_temps(self, num_stages: int) -> None:
        if not self.stage_temps_k:
            self.stage_temps_k = [AMBIENT_TRACK_TEMP_K] * num_stages


# -- Stage tick result -------------------------------------------------------

@dataclass
class StageResult:
    """Result of a single stage tick."""
    stage_index: int = 0
    entry_velocity_m_s: float = 0.0
    exit_velocity_m_s: float = 0.0
    acceleration_m_s2: float = 0.0
    g_load: float = 0.0
    force_n: float = 0.0
    drag_n: float = 0.0
    gravity_n: float = 0.0
    energy_in_j: float = 0.0
    thermal_loss_j: float = 0.0
    drag_loss_j: float = 0.0
    transit_time_s: float = 0.0
    stage_temp_k: float = AMBIENT_TRACK_TEMP_K
    altitude_m: float = 0.0


# -- Factory ------------------------------------------------------------------

def create_track(
    track_length_m: float = DEFAULT_TRACK_LENGTH_M,
    elevation_deg: float = DEFAULT_ELEVATION_DEG,
    num_stages: int = DEFAULT_NUM_STAGES,
    current_a: float = DEFAULT_CURRENT_A,
    field_t: float = DEFAULT_MAGNETIC_FIELD_T,
    g_limit: float = DEFAULT_G_LIMIT,
) -> TrackConfig:
    """Create a track configuration."""
    return TrackConfig(
        track_length_m=track_length_m,
        elevation_deg=elevation_deg,
        num_stages=num_stages,
        current_a=current_a,
        magnetic_field_t=field_t,
        g_limit=g_limit,
    )


def create_launch(
    payload_kg: float = DEFAULT_PAYLOAD_KG,
    sled_kg: float = DEFAULT_SLED_KG,
) -> LaunchState:
    """Create a fresh launch state."""
    return LaunchState(payload_mass_kg=payload_kg, sled_mass_kg=sled_kg)


# -- Core tick: one coil stage fires -----------------------------------------

def tick(config: TrackConfig, state: LaunchState) -> StageResult:
    """Advance the launch by one coil stage."""
    result = StageResult(stage_index=state.current_stage)
    state.init_stage_temps(config.num_stages)

    if state.launch_complete or state.launch_failed:
        result.exit_velocity_m_s = state.velocity_m_s
        result.altitude_m = state.altitude_m
        return result

    if state.current_stage >= config.num_stages:
        state.launch_complete = True
        result.exit_velocity_m_s = state.velocity_m_s
        result.altitude_m = state.altitude_m
        return result

    stage_len = config.stage_length_m
    entry_v = state.velocity_m_s
    total_mass = state.total_mass_kg
    result.entry_velocity_m_s = entry_v

    em_force = config.max_force_n
    drag = drag_force_n(entry_v, state.altitude_m,
                        config.drag_coeff, config.sled_cross_section_m2)
    grav = gravity_component_n(total_mass, config.elevation_deg)
    result.force_n = em_force
    result.drag_n = drag
    result.gravity_n = grav

    accel = stage_acceleration_m_s2(em_force, drag, grav, total_mass)
    max_accel = config.g_limit * G0_M_S2
    if accel > max_accel:
        accel = max_accel
    result.acceleration_m_s2 = accel
    result.g_load = g_load(accel)

    if result.g_load > config.g_limit * 1.01:
        state.launch_failed = True
        state.failure_reason = f"G-load {result.g_load:.1f} exceeds limit {config.g_limit}"
        result.exit_velocity_m_s = entry_v
        return result

    v_exit = exit_velocity_m_s(entry_v, accel, stage_len)
    result.exit_velocity_m_s = v_exit

    dt = stage_transit_time_s(entry_v, v_exit, stage_len)
    result.transit_time_s = dt

    ke_gain = kinetic_energy_j(total_mass, v_exit) - kinetic_energy_j(total_mass, entry_v)
    drag_loss = drag * stage_len
    grav_work = grav * stage_len
    total_useful = ke_gain + drag_loss + grav_work
    energy_in = total_useful / max(config.efficiency, 0.01)
    result.energy_in_j = energy_in
    result.drag_loss_j = drag_loss

    thermal = resistive_heat_j(config.current_a, config.coil_resistance_ohm, dt)
    result.thermal_loss_j = thermal

    stage_idx = state.current_stage
    if stage_idx < len(state.stage_temps_k):
        temp = state.stage_temps_k[stage_idx]
        temp += thermal / TRACK_THERMAL_MASS_J_PER_K
        state.stage_temps_k[stage_idx] = temp
        result.stage_temp_k = temp

        if temp > MAX_TRACK_TEMP_K:
            state.launch_failed = True
            state.failure_reason = f"Track stage {stage_idx} overheated: {temp:.1f} K"
            result.exit_velocity_m_s = entry_v
            return result

    state.velocity_m_s = v_exit
    state.position_m += stage_len
    state.altitude_m = altitude_gain_m(state.position_m, config.elevation_deg)
    state.current_stage += 1
    state.total_energy_in_j += energy_in
    state.total_drag_loss_j += drag_loss
    state.total_thermal_loss_j += thermal
    state.elapsed_time_s += dt
    state.peak_g_load = max(state.peak_g_load, result.g_load)
    state.peak_velocity_m_s = max(state.peak_velocity_m_s, v_exit)
    result.altitude_m = state.altitude_m

    if state.current_stage >= config.num_stages:
        state.launch_complete = True

    return result


# -- Cooling between launches ------------------------------------------------

def cool_track(state: LaunchState, wait_seconds: float) -> None:
    """Cool all track stages toward ambient over a wait period."""
    if not state.stage_temps_k:
        return
    for i in range(len(state.stage_temps_k)):
        temp = state.stage_temps_k[i]
        if temp > AMBIENT_TRACK_TEMP_K:
            delta = temp - AMBIENT_TRACK_TEMP_K
            cooling = TRACK_COOLING_RATE_W * wait_seconds / TRACK_THERMAL_MASS_J_PER_K
            new_delta = delta * math.exp(-cooling)
            state.stage_temps_k[i] = AMBIENT_TRACK_TEMP_K + new_delta


# -- Full launch simulation --------------------------------------------------

def run_launch(
    config: TrackConfig | None = None,
    state: LaunchState | None = None,
) -> list[StageResult]:
    """Run a complete launch sequence through all stages."""
    if config is None:
        config = create_track()
    if state is None:
        state = create_launch()
    results: list[StageResult] = []
    for _ in range(config.num_stages + 1):
        result = tick(config, state)
        results.append(result)
        if state.launch_complete or state.launch_failed:
            break
    return results


def can_reach_target(
    target_velocity_m_s: float,
    config: TrackConfig | None = None,
    payload_kg: float = DEFAULT_PAYLOAD_KG,
) -> dict[str, Any]:
    """Estimate whether the track can launch a payload to target velocity."""
    if config is None:
        config = create_track()
    state = create_launch(payload_kg=payload_kg)
    run_launch(config, state)

    ke_payload = kinetic_energy_j(payload_kg, state.velocity_m_s)

    return {
        "exit_velocity_m_s": round(state.velocity_m_s, 1),
        "target_velocity_m_s": target_velocity_m_s,
        "velocity_margin_m_s": round(state.velocity_m_s - target_velocity_m_s, 1),
        "go": state.velocity_m_s >= target_velocity_m_s and not state.launch_failed,
        "peak_g_load": round(state.peak_g_load, 1),
        "total_energy_mj": round(state.total_energy_in_j / 1e6, 2),
        "kinetic_energy_payload_mj": round(ke_payload / 1e6, 2),
        "kinetic_energy_total_mj": round(
            kinetic_energy_j(state.total_mass_kg, state.velocity_m_s) / 1e6, 2),
        "drag_loss_mj": round(state.total_drag_loss_j / 1e6, 2),
        "thermal_loss_mj": round(state.total_thermal_loss_j / 1e6, 2),
        "launch_time_s": round(state.elapsed_time_s, 4),
        "stages_fired": state.current_stage,
        "failed": state.launch_failed,
        "failure_reason": state.failure_reason,
        "altitude_at_exit_m": round(state.altitude_m, 1),
        "payload_kg": payload_kg,
    }


def optimal_track_for_target(
    target_velocity_m_s: float,
    payload_kg: float = DEFAULT_PAYLOAD_KG,
    max_g: float = DEFAULT_G_LIMIT,
    min_track_m: float = 500.0,
    max_track_m: float = 20_000.0,
) -> dict[str, Any]:
    """Find the shortest track that can reach target velocity within g-limit."""
    best: dict[str, Any] = {}
    lo, hi = min_track_m, max_track_m

    for _ in range(30):
        mid = (lo + hi) / 2.0
        cfg = create_track(track_length_m=mid, g_limit=max_g)
        result = can_reach_target(target_velocity_m_s, cfg, payload_kg)

        if result["go"]:
            best = result
            best["track_length_m"] = round(mid, 1)
            hi = mid
        else:
            lo = mid

        if hi - lo < 1.0:
            break

    if not best:
        cfg = create_track(track_length_m=max_track_m, g_limit=max_g)
        result = can_reach_target(target_velocity_m_s, cfg, payload_kg)
        result["track_length_m"] = max_track_m
        return result

    return best


# -- High-level simulation entry point ----------------------------------------

def run_simulation(
    payload_kg: float = DEFAULT_PAYLOAD_KG,
    target: str = "lmo",
    track_length_m: float = DEFAULT_TRACK_LENGTH_M,
    elevation_deg: float = DEFAULT_ELEVATION_DEG,
    g_limit: float = DEFAULT_G_LIMIT,
) -> dict[str, Any]:
    """Run a complete mass driver simulation."""
    targets = {
        "lmo": LMO_VELOCITY_M_S,
        "escape": ESCAPE_VELOCITY_M_S,
        "earth": EARTH_TRANSFER_M_S,
    }
    target_v = targets.get(target, LMO_VELOCITY_M_S)

    config = create_track(
        track_length_m=track_length_m,
        elevation_deg=elevation_deg,
        g_limit=g_limit,
    )
    state = create_launch(payload_kg=payload_kg)
    run_launch(config, state)

    ke_payload = kinetic_energy_j(payload_kg, state.velocity_m_s)
    efficiency = ke_payload / max(state.total_energy_in_j, 1.0)

    power_avg = power_required_w(state.total_energy_in_j,
                                 max(state.elapsed_time_s, 1e-9))

    return {
        "target": target,
        "target_velocity_m_s": target_v,
        "exit_velocity_m_s": round(state.velocity_m_s, 1),
        "go": state.velocity_m_s >= target_v and not state.launch_failed,
        "payload_kg": payload_kg,
        "total_mass_kg": state.total_mass_kg,
        "track_length_m": track_length_m,
        "elevation_deg": elevation_deg,
        "stages_fired": state.current_stage,
        "launch_time_s": round(state.elapsed_time_s, 4),
        "peak_g_load": round(state.peak_g_load, 1),
        "exit_altitude_m": round(state.altitude_m, 1),
        "total_energy_mj": round(state.total_energy_in_j / 1e6, 2),
        "kinetic_energy_mj": round(ke_payload / 1e6, 2),
        "drag_loss_mj": round(state.total_drag_loss_j / 1e6, 2),
        "thermal_loss_mj": round(state.total_thermal_loss_j / 1e6, 2),
        "payload_efficiency": round(efficiency, 4),
        "average_power_mw": round(power_avg / 1e6, 2),
        "failed": state.launch_failed,
        "failure_reason": state.failure_reason,
    }
