"""mass_driver.py -- Mars Electromagnetic Launch Rail (Mass Driver).

The colony has a rocket engine for crewed ascent (rocket_engine.py) and
a propellant depot (propellant_depot.py) for fuel storage.  But burning
methalox to loft every kilogram of cargo is ruinously expensive.

This module is the freight elevator: an electromagnetic mass driver that
accelerates cargo sleds along a track and flings them to Mars orbit,
Phobos, or Earth-transfer trajectories -- zero propellant consumed.
Power comes from the colony nuclear reactor and solar arrays.

Physics: Lorentz force, kinetic energy, atmospheric drag, g-load limits,
track thermal load, magnetic field energy, launch geometry, orbital mechanics.

Conservation laws: energy, momentum, mass, temperature, velocity monotonic.
One tick = one coil stage firing.  Mass in kg, force in N, velocity in m/s.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any

G0_M_S2 = 9.80665
MARS_GRAVITY_M_S2 = 3.72076
MARS_SURFACE_PRESSURE_PA = 610.0
MARS_SURFACE_DENSITY_KG_M3 = 0.020
MARS_SCALE_HEIGHT_M = 11_100.0
MARS_SURFACE_TEMP_K = 210.0
VACUUM_PERMEABILITY_H_M = 4.0e-7 * math.pi
LMO_VELOCITY_M_S = 3550.0
ESCAPE_VELOCITY_M_S = 5030.0
EARTH_TRANSFER_M_S = 5700.0
DEFAULT_TRACK_LENGTH_M = 2000.0
DEFAULT_ELEVATION_DEG = 30.0
DEFAULT_NUM_STAGES = 200
DEFAULT_COIL_LENGTH_M = 0.5
DEFAULT_COILS_PER_STAGE = 4
DEFAULT_CURRENT_A = 50_000.0
DEFAULT_MAGNETIC_FIELD_T = 2.0
DEFAULT_COIL_RESISTANCE_OHM = 0.001
ELECTRICAL_EFFICIENCY = 0.65
DRAG_COEFF = 0.15
SLED_CROSS_SECTION_M2 = 0.5
MAX_TRACK_TEMP_K = 400.0
AMBIENT_TRACK_TEMP_K = MARS_SURFACE_TEMP_K
TRACK_THERMAL_MASS_J_PER_K = 500_000.0
TRACK_COOLING_RATE_W = 2000.0
HARDENED_CARGO_G_LIMIT = 50.0
FRAGILE_CARGO_G_LIMIT = 3.0
DEFAULT_G_LIMIT = 30.0
DEFAULT_PAYLOAD_KG = 100.0
DEFAULT_SLED_KG = 50.0

def lorentz_force_n(current_a, coil_length_m, magnetic_field_t, num_coils=1):
    """Lorentz force from active coils: F = n * I * L * B."""
    if current_a <= 0.0 or coil_length_m <= 0.0 or magnetic_field_t <= 0.0:
        return 0.0
    return float(num_coils) * current_a * coil_length_m * magnetic_field_t

def kinetic_energy_j(mass_kg, velocity_m_s):
    """Kinetic energy: E = 0.5 * m * v^2."""
    if mass_kg <= 0.0:
        return 0.0
    return 0.5 * mass_kg * velocity_m_s ** 2

def velocity_from_energy_m_s(energy_j, mass_kg):
    """Velocity from kinetic energy."""
    if energy_j <= 0.0 or mass_kg <= 0.0:
        return 0.0
    return math.sqrt(2.0 * energy_j / mass_kg)

def mars_air_density_kg_m3(altitude_m):
    """Mars atmospheric density at altitude (exponential model)."""
    if altitude_m < 0.0:
        altitude_m = 0.0
    return MARS_SURFACE_DENSITY_KG_M3 * math.exp(-altitude_m / MARS_SCALE_HEIGHT_M)

def drag_force_n(velocity_m_s, altitude_m=0.0, drag_coeff=DRAG_COEFF, cross_section_m2=SLED_CROSS_SECTION_M2):
    """Aerodynamic drag: F = 0.5 * rho * v^2 * Cd * A."""
    if velocity_m_s <= 0.0:
        return 0.0
    rho = mars_air_density_kg_m3(altitude_m)
    return 0.5 * rho * velocity_m_s ** 2 * drag_coeff * cross_section_m2

def gravity_component_n(mass_kg, elevation_deg):
    """Gravity force along track."""
    if mass_kg <= 0.0:
        return 0.0
    theta = math.radians(max(0.0, min(90.0, elevation_deg)))
    return mass_kg * MARS_GRAVITY_M_S2 * math.sin(theta)

def stage_acceleration_m_s2(force_n, drag_n, gravity_n, mass_kg):
    """Net acceleration along track for a single stage."""
    if mass_kg <= 0.0:
        return 0.0
    return max(0.0, (force_n - drag_n - gravity_n) / mass_kg)

def exit_velocity_m_s(entry_velocity_m_s, acceleration_m_s2, stage_length_m):
    """Velocity after traversing one stage."""
    if acceleration_m_s2 <= 0.0 and entry_velocity_m_s <= 0.0:
        return 0.0
    v_sq = entry_velocity_m_s ** 2 + 2.0 * acceleration_m_s2 * stage_length_m
    return math.sqrt(max(0.0, v_sq))

def stage_transit_time_s(entry_velocity_m_s, exit_velocity_m_s, stage_length_m):
    """Time to traverse one stage."""
    avg_v = 0.5 * (entry_velocity_m_s + exit_velocity_m_s)
    if avg_v <= 0.0:
        return float('inf')
    return stage_length_m / avg_v

def resistive_heat_j(current_a, resistance_ohm, duration_s):
    """Ohmic heating in coils: Q = I^2 * R * t."""
    if current_a <= 0.0 or resistance_ohm <= 0.0 or duration_s <= 0.0:
        return 0.0
    return current_a ** 2 * resistance_ohm * duration_s

def magnetic_field_energy_j(field_t, volume_m3):
    """Energy stored in magnetic field."""
    if field_t <= 0.0 or volume_m3 <= 0.0:
        return 0.0
    return (field_t ** 2 / (2.0 * VACUUM_PERMEABILITY_H_M)) * volume_m3

def altitude_gain_m(track_distance_m, elevation_deg):
    """Altitude gained along track."""
    theta = math.radians(max(0.0, min(90.0, elevation_deg)))
    return track_distance_m * math.sin(theta)

def power_required_w(energy_j, duration_s):
    """Average power: P = E / t."""
    if duration_s <= 0.0:
        return float('inf')
    return energy_j / duration_s

def g_load(acceleration_m_s2):
    """Convert acceleration to g-load."""
    return acceleration_m_s2 / G0_M_S2

@dataclass
class TrackConfig:
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
    def stage_length_m(self):
        if self.num_stages <= 0:
            return 0.0
        return self.track_length_m / self.num_stages

    @property
    def coil_volume_m3(self):
        r_outer = 0.3
        shell_thickness = 0.05
        return (math.pi * ((r_outer + shell_thickness) ** 2 - r_outer ** 2)
                * self.coil_length_m * self.coils_per_stage)

    @property
    def max_force_n(self):
        return lorentz_force_n(self.current_a, self.coil_length_m,
                               self.magnetic_field_t, self.coils_per_stage)

@dataclass
class LaunchState:
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
    failure_reason: str = ''
    elapsed_time_s: float = 0.0
    stage_temps_k: list[float] = field(default_factory=list)

    @property
    def total_mass_kg(self):
        return self.payload_mass_kg + self.sled_mass_kg

    def init_stage_temps(self, num_stages):
        if not self.stage_temps_k:
            self.stage_temps_k = [AMBIENT_TRACK_TEMP_K] * num_stages

@dataclass
class StageResult:
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

def create_track(track_length_m=DEFAULT_TRACK_LENGTH_M, elevation_deg=DEFAULT_ELEVATION_DEG,
                 num_stages=DEFAULT_NUM_STAGES, current_a=DEFAULT_CURRENT_A,
                 field_t=DEFAULT_MAGNETIC_FIELD_T, g_limit=DEFAULT_G_LIMIT):
    return TrackConfig(track_length_m=track_length_m, elevation_deg=elevation_deg,
                       num_stages=num_stages, current_a=current_a,
                       magnetic_field_t=field_t, g_limit=g_limit)

def create_launch(payload_kg=DEFAULT_PAYLOAD_KG, sled_kg=DEFAULT_SLED_KG):
    return LaunchState(payload_mass_kg=payload_kg, sled_mass_kg=sled_kg)

def tick(config, state):
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
    drag = drag_force_n(entry_v, state.altitude_m, config.drag_coeff, config.sled_cross_section_m2)
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
        state.failure_reason = 'G-load exceeded'
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
            state.failure_reason = f'Track stage {stage_idx} overheated: {temp:.1f} K'
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

def cool_track(state, wait_seconds):
    if not state.stage_temps_k:
        return
    for i in range(len(state.stage_temps_k)):
        temp = state.stage_temps_k[i]
        if temp > AMBIENT_TRACK_TEMP_K:
            delta = temp - AMBIENT_TRACK_TEMP_K
            cooling = TRACK_COOLING_RATE_W * wait_seconds / TRACK_THERMAL_MASS_J_PER_K
            state.stage_temps_k[i] = AMBIENT_TRACK_TEMP_K + delta * math.exp(-cooling)

def run_launch(config=None, state=None):
    if config is None:
        config = create_track()
    if state is None:
        state = create_launch()
    results = []
    for _ in range(config.num_stages + 1):
        result = tick(config, state)
        results.append(result)
        if state.launch_complete or state.launch_failed:
            break
    return results

def can_reach_target(target_velocity_m_s, config=None, payload_kg=DEFAULT_PAYLOAD_KG):
    if config is None:
        config = create_track()
    state = create_launch(payload_kg=payload_kg)
    run_launch(config, state)
    ke_payload = kinetic_energy_j(payload_kg, state.velocity_m_s)
    return {
        'exit_velocity_m_s': round(state.velocity_m_s, 1),
        'target_velocity_m_s': target_velocity_m_s,
        'velocity_margin_m_s': round(state.velocity_m_s - target_velocity_m_s, 1),
        'go': state.velocity_m_s >= target_velocity_m_s and not state.launch_failed,
        'peak_g_load': round(state.peak_g_load, 1),
        'total_energy_mj': round(state.total_energy_in_j / 1e6, 2),
        'kinetic_energy_payload_mj': round(ke_payload / 1e6, 2),
        'kinetic_energy_total_mj': round(kinetic_energy_j(state.total_mass_kg, state.velocity_m_s) / 1e6, 2),
        'drag_loss_mj': round(state.total_drag_loss_j / 1e6, 2),
        'thermal_loss_mj': round(state.total_thermal_loss_j / 1e6, 2),
        'launch_time_s': round(state.elapsed_time_s, 4),
        'stages_fired': state.current_stage,
        'failed': state.launch_failed,
        'failure_reason': state.failure_reason,
        'altitude_at_exit_m': round(state.altitude_m, 1),
        'payload_kg': payload_kg,
    }

def optimal_track_for_target(target_velocity_m_s, payload_kg=DEFAULT_PAYLOAD_KG,
                             max_g=DEFAULT_G_LIMIT, min_track_m=500.0, max_track_m=20_000.0):
    best = {}
    lo, hi = min_track_m, max_track_m
    for _ in range(30):
        mid = (lo + hi) / 2.0
        cfg = create_track(track_length_m=mid, g_limit=max_g)
        result = can_reach_target(target_velocity_m_s, cfg, payload_kg)
        if result['go']:
            best = result
            best['track_length_m'] = round(mid, 1)
            hi = mid
        else:
            lo = mid
        if hi - lo < 1.0:
            break
    if not best:
        cfg = create_track(track_length_m=max_track_m, g_limit=max_g)
        result = can_reach_target(target_velocity_m_s, cfg, payload_kg)
        result['track_length_m'] = max_track_m
        return result
    return best

def run_simulation(payload_kg=DEFAULT_PAYLOAD_KG, target='lmo',
                   track_length_m=DEFAULT_TRACK_LENGTH_M, elevation_deg=DEFAULT_ELEVATION_DEG,
                   g_limit=DEFAULT_G_LIMIT):
    targets = {'lmo': LMO_VELOCITY_M_S, 'escape': ESCAPE_VELOCITY_M_S, 'earth': EARTH_TRANSFER_M_S}
    target_v = targets.get(target, LMO_VELOCITY_M_S)
    config = create_track(track_length_m=track_length_m, elevation_deg=elevation_deg, g_limit=g_limit)
    state = create_launch(payload_kg=payload_kg)
    run_launch(config, state)
    ke_payload = kinetic_energy_j(payload_kg, state.velocity_m_s)
    efficiency = ke_payload / max(state.total_energy_in_j, 1.0)
    power_avg = power_required_w(state.total_energy_in_j, max(state.elapsed_time_s, 1e-9))
    return {
        'target': target, 'target_velocity_m_s': target_v,
        'exit_velocity_m_s': round(state.velocity_m_s, 1),
        'go': state.velocity_m_s >= target_v and not state.launch_failed,
        'payload_kg': payload_kg, 'total_mass_kg': state.total_mass_kg,
        'track_length_m': track_length_m, 'elevation_deg': elevation_deg,
        'stages_fired': state.current_stage,
        'launch_time_s': round(state.elapsed_time_s, 4),
        'peak_g_load': round(state.peak_g_load, 1),
        'exit_altitude_m': round(state.altitude_m, 1),
        'total_energy_mj': round(state.total_energy_in_j / 1e6, 2),
        'kinetic_energy_mj': round(ke_payload / 1e6, 2),
        'drag_loss_mj': round(state.total_drag_loss_j / 1e6, 2),
        'thermal_loss_mj': round(state.total_thermal_loss_j / 1e6, 2),
        'payload_efficiency': round(efficiency, 4),
        'average_power_mw': round(power_avg / 1e6, 2),
        'failed': state.launch_failed, 'failure_reason': state.failure_reason,
    }
