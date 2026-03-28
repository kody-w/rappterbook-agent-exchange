"""centrifuge_habitat.py -- Mars Artificial Gravity Centrifuge.

Mars gravity is 0.38g.  We don't know the long-term health floor
for humans -- maybe 0.38g is fine, maybe it causes irreversible
bone loss, cardiac remodeling, and immune suppression over years.
The colony cannot gamble.  A rotating habitat section provides
supplemental gravity so crew can sleep at 0.5-1.0g nightly.

This is a tethered centrifuge: two habitat pods connected by a
tensile cable, spinning around their common centre of mass.  Unlike
a rigid ring, a tethered system can be built from existing materials
(pressurized_tunnel.py, regolith_sintering.py) and scaled up
incrementally.

Physics modelled
----------------
* **Centripetal acceleration:**  a = ω²r.  At radius r with angular
  velocity ω, the apparent gravity felt is a + g_mars (along the
  radial-vertical composite direction).

* **Coriolis effect:**  F_cor = −2m(ω × v).  Walking along the spin
  direction feels heavier; against feels lighter.  At 2 RPM and
  50m radius, Coriolis is ~4% of centripetal for a 1 m/s walk.

* **Tether tension:**  T = m·ω²·r for each pod.  The tether must
  hold the combined centripetal load of pod + crew + equipment.
  Safety factor ≥ 3 required.  Kevlar tether: tensile strength
  3,620 MPa, density 1,440 kg/m³.

* **Spin-up energy:**  E = ½Iω².  Moment of inertia I = Σ mᵢrᵢ².
  An electric motor at the hub inputs torque; spin-up time depends
  on motor power and total I.

* **Bearing friction:**  The hub bearing dissipates energy.
  P_friction = μ·F_axial·ω·r_bearing.  Must be continuously
  compensated or the centrifuge spins down.

* **Precession:**  On Mars, the rotating plane precesses due to
  the Coriolis force from Mars's rotation (very small: Mars rotates
  once per 24.6h).  Modelled as a perturbation.

* **Comfort criteria (NASA STD):**
  - Max rotation rate: 4 RPM (above → motion sickness)
  - Min radius for 1g at 4 RPM: ~56 m
  - Gravity gradient head-to-foot < 10% (requires r > 14× person height)

Conservation laws
-----------------
- Angular momentum: L = Iω (conserved when no external torque)
- Energy: KE_rotational = ½Iω² (conserved minus friction)
- Tension: T > 0 whenever ω > 0 (structural integrity)
- Acceleration: a ≥ 0, bounded by max_rpm constraint
- Mass balance: total mass = 2×pod + tether + hub (constant)

One tick = one sol.  Angular velocity in rad/s, force in N, energy in J.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# -- Physical constants -------------------------------------------------------

MARS_GRAVITY_M_S2 = 3.72076
MARS_ROTATION_RAD_S = 7.088e-5     # Mars sidereal rotation rate
G0_M_S2 = 9.80665
TWO_PI = 2.0 * math.pi

# -- Material properties ------------------------------------------------------

KEVLAR_TENSILE_MPA = 3_620.0
KEVLAR_DENSITY_KG_M3 = 1_440.0
STEEL_TENSILE_MPA = 500.0
STEEL_DENSITY_KG_M3 = 7_800.0

# -- Default design parameters ------------------------------------------------

DEFAULT_RADIUS_M = 56.0             # pod distance from hub
DEFAULT_POD_MASS_KG = 8_000.0       # each pod (structure + life support)
DEFAULT_CREW_PER_POD = 3
DEFAULT_CREW_MASS_KG = 80.0
DEFAULT_TETHER_DIAMETER_M = 0.03    # Kevlar cable
DEFAULT_HUB_MASS_KG = 2_000.0
DEFAULT_TARGET_G = 0.7              # target apparent gravity (Mars + centripetal)
DEFAULT_MAX_RPM = 4.0
DEFAULT_MOTOR_POWER_W = 5_000.0     # spin-up/maintenance motor
DEFAULT_BEARING_FRICTION_COEFF = 0.005
DEFAULT_BEARING_RADIUS_M = 0.3
DEFAULT_SAFETY_FACTOR = 3.0

SECONDS_PER_SOL = 88_775.0
HOURS_PER_SOL = 24.66

# comfort thresholds
MAX_RPM = 6.0                       # absolute structural limit
COMFORT_MAX_RPM = 4.0               # crew comfort limit
MIN_RADIUS_FOR_COMFORT_M = 14.0     # ~8× average height
MAX_GRAVITY_GRADIENT_FRAC = 0.10    # head-to-foot < 10%
PERSON_HEIGHT_M = 1.75


# =============================================================================
# Pure physics functions
# =============================================================================

def rpm_to_rad_s(rpm: float) -> float:
    """Convert rotations per minute to radians per second."""
    return rpm * TWO_PI / 60.0


def rad_s_to_rpm(omega: float) -> float:
    """Convert radians per second to rotations per minute."""
    if omega <= 0.0:
        return 0.0
    return omega * 60.0 / TWO_PI


def centripetal_acceleration_m_s2(omega_rad_s: float, radius_m: float) -> float:
    """Centripetal acceleration: a = ω²r."""
    if omega_rad_s <= 0.0 or radius_m <= 0.0:
        return 0.0
    return omega_rad_s * omega_rad_s * radius_m


def apparent_gravity_g(omega_rad_s: float, radius_m: float) -> float:
    """Apparent gravity in g-units (centripetal + Mars gravity vector sum).

    For a floor-mounted centrifuge on Mars, the apparent gravity
    is the vector sum of centripetal (radial) and Mars gravity
    (vertical).  For a horizontally-spinning centrifuge:
    g_apparent = sqrt(a_cent² + g_mars²).
    """
    a_cent = centripetal_acceleration_m_s2(omega_rad_s, radius_m)
    g_total = math.sqrt(a_cent**2 + MARS_GRAVITY_M_S2**2)
    return g_total / G0_M_S2


def omega_for_target_g(target_g: float, radius_m: float) -> float:
    """Angular velocity (rad/s) needed for target apparent gravity.

    Solves: sqrt((ω²r)² + g_mars²) = target_g * g0
    → ω²r = sqrt((target_g*g0)² - g_mars²)
    → ω = sqrt(above / r)
    """
    if target_g <= 0.0 or radius_m <= 0.0:
        return 0.0
    target_accel = target_g * G0_M_S2
    if target_accel <= MARS_GRAVITY_M_S2:
        return 0.0  # Mars gravity alone exceeds or meets target
    centripetal_needed = math.sqrt(target_accel**2 - MARS_GRAVITY_M_S2**2)
    return math.sqrt(centripetal_needed / radius_m)


def coriolis_acceleration_m_s2(omega_rad_s: float,
                                velocity_m_s: float) -> float:
    """Coriolis acceleration magnitude: a_cor = 2·ω·v.

    This is the sideways force felt when walking in the rotating frame.
    """
    if omega_rad_s <= 0.0 or velocity_m_s <= 0.0:
        return 0.0
    return 2.0 * omega_rad_s * velocity_m_s


def coriolis_fraction(omega_rad_s: float, radius_m: float,
                      walk_speed_m_s: float = 1.0) -> float:
    """Coriolis acceleration as fraction of centripetal (comfort metric)."""
    a_cent = centripetal_acceleration_m_s2(omega_rad_s, radius_m)
    if a_cent <= 0.0:
        return 0.0
    a_cor = coriolis_acceleration_m_s2(omega_rad_s, walk_speed_m_s)
    return a_cor / a_cent


def gravity_gradient_fraction(radius_m: float,
                               person_height_m: float = PERSON_HEIGHT_M) -> float:
    """Head-to-foot gravity gradient as a fraction.

    At the feet (radius r), a = ω²r.
    At the head (radius r - h), a = ω²(r-h).
    Gradient = h/r.
    """
    if radius_m <= 0.0:
        return 1.0  # infinite gradient
    return person_height_m / radius_m


def tether_tension_n(pod_mass_kg: float, omega_rad_s: float,
                     radius_m: float) -> float:
    """Centripetal tension in tether: T = m·ω²·r."""
    if pod_mass_kg <= 0.0 or omega_rad_s <= 0.0 or radius_m <= 0.0:
        return 0.0
    return pod_mass_kg * omega_rad_s**2 * radius_m


def tether_cross_section_m2(tension_n: float, safety_factor: float,
                             tensile_strength_mpa: float = KEVLAR_TENSILE_MPA) -> float:
    """Required tether cross-section area for given tension."""
    if tension_n <= 0.0 or safety_factor <= 0.0 or tensile_strength_mpa <= 0.0:
        return 0.0
    required_strength_pa = tensile_strength_mpa * 1e6
    return (tension_n * safety_factor) / required_strength_pa


def tether_mass_kg(length_m: float, cross_section_m2: float,
                   density_kg_m3: float = KEVLAR_DENSITY_KG_M3) -> float:
    """Mass of one tether cable."""
    if length_m <= 0.0 or cross_section_m2 <= 0.0:
        return 0.0
    return length_m * cross_section_m2 * density_kg_m3


def moment_of_inertia_kg_m2(masses_and_radii: list[tuple[float, float]]) -> float:
    """Total moment of inertia: I = Σ mᵢ·rᵢ²."""
    total = 0.0
    for mass, radius in masses_and_radii:
        if mass > 0.0 and radius >= 0.0:
            total += mass * radius * radius
    return total


def rotational_energy_j(moment_of_inertia: float,
                        omega_rad_s: float) -> float:
    """Rotational kinetic energy: E = ½Iω²."""
    if moment_of_inertia <= 0.0 or omega_rad_s <= 0.0:
        return 0.0
    return 0.5 * moment_of_inertia * omega_rad_s**2


def angular_momentum_kg_m2_s(moment_of_inertia: float,
                              omega_rad_s: float) -> float:
    """Angular momentum: L = Iω."""
    if moment_of_inertia <= 0.0 or omega_rad_s <= 0.0:
        return 0.0
    return moment_of_inertia * omega_rad_s


def spin_up_time_s(moment_of_inertia: float, target_omega: float,
                   motor_power_w: float) -> float:
    """Time to spin up from rest: t = E / P = ½Iω² / P."""
    energy = rotational_energy_j(moment_of_inertia, target_omega)
    if motor_power_w <= 0.0:
        return float("inf") if energy > 0.0 else 0.0
    return energy / motor_power_w


def bearing_friction_power_w(friction_coeff: float, axial_load_n: float,
                              omega_rad_s: float,
                              bearing_radius_m: float) -> float:
    """Power lost to bearing friction: P = μ·F·ω·r."""
    if (friction_coeff <= 0.0 or axial_load_n <= 0.0
            or omega_rad_s <= 0.0 or bearing_radius_m <= 0.0):
        return 0.0
    return friction_coeff * axial_load_n * omega_rad_s * bearing_radius_m


def spindown_rate_rad_s2(friction_power_w: float,
                          moment_of_inertia: float,
                          omega_rad_s: float) -> float:
    """Angular deceleration from friction: α = P / (I·ω)."""
    if (friction_power_w <= 0.0 or moment_of_inertia <= 0.0
            or omega_rad_s <= 0.0):
        return 0.0
    return friction_power_w / (moment_of_inertia * omega_rad_s)


def spindown_time_s(omega_rad_s: float, decel_rad_s2: float) -> float:
    """Time for centrifuge to stop from friction alone."""
    if omega_rad_s <= 0.0 or decel_rad_s2 <= 0.0:
        return 0.0
    return omega_rad_s / decel_rad_s2


def comfort_check(rpm: float, radius_m: float) -> dict[str, Any]:
    """Evaluate comfort criteria for a centrifuge configuration."""
    omega = rpm_to_rad_s(rpm)
    g_grad = gravity_gradient_fraction(radius_m)
    cor_frac = coriolis_fraction(omega, radius_m, 1.0)
    app_g = apparent_gravity_g(omega, radius_m)

    issues: list[str] = []
    if rpm > COMFORT_MAX_RPM:
        issues.append(f"RPM {rpm:.1f} exceeds comfort limit {COMFORT_MAX_RPM}")
    if g_grad > MAX_GRAVITY_GRADIENT_FRAC:
        issues.append(f"Gravity gradient {g_grad:.1%} exceeds {MAX_GRAVITY_GRADIENT_FRAC:.0%}")
    if radius_m < MIN_RADIUS_FOR_COMFORT_M:
        issues.append(f"Radius {radius_m:.1f}m below minimum {MIN_RADIUS_FOR_COMFORT_M}m")
    if cor_frac > 0.25:
        issues.append(f"Coriolis fraction {cor_frac:.1%} exceeds 25%")

    return {
        "rpm": round(rpm, 2),
        "radius_m": radius_m,
        "apparent_gravity_g": round(app_g, 3),
        "gravity_gradient_frac": round(g_grad, 4),
        "coriolis_fraction": round(cor_frac, 4),
        "comfortable": len(issues) == 0,
        "issues": issues,
    }


def design_centrifuge(target_g: float = DEFAULT_TARGET_G,
                      max_rpm: float = COMFORT_MAX_RPM,
                      pod_mass_kg: float = DEFAULT_POD_MASS_KG,
                      crew_per_pod: int = DEFAULT_CREW_PER_POD) -> dict[str, Any]:
    """Design a centrifuge to meet target gravity within comfort limits."""
    max_omega = rpm_to_rad_s(max_rpm)

    # Minimum radius from max RPM: a_cent = ω²r
    # target_accel = sqrt((target_g*g0)² - g_mars²)
    target_accel = target_g * G0_M_S2
    if target_accel <= MARS_GRAVITY_M_S2:
        return {"error": "Target g is less than Mars gravity alone",
                "target_g": target_g}

    cent_needed = math.sqrt(target_accel**2 - MARS_GRAVITY_M_S2**2)
    min_radius = cent_needed / (max_omega**2) if max_omega > 0 else float("inf")
    min_radius = max(min_radius, MIN_RADIUS_FOR_COMFORT_M)

    # Check gravity gradient
    while gravity_gradient_fraction(min_radius) > MAX_GRAVITY_GRADIENT_FRAC:
        min_radius += 1.0
        if min_radius > 500.0:
            break

    omega = omega_for_target_g(target_g, min_radius)
    rpm = rad_s_to_rpm(omega)

    pod_total = pod_mass_kg + crew_per_pod * DEFAULT_CREW_MASS_KG
    tension = tether_tension_n(pod_total, omega, min_radius)
    tether_cs = tether_cross_section_m2(tension, DEFAULT_SAFETY_FACTOR)
    tether_diam = 2.0 * math.sqrt(tether_cs / math.pi) if tether_cs > 0 else 0.0
    t_mass = tether_mass_kg(min_radius * 2.0, tether_cs)

    masses_radii = [
        (pod_total, min_radius),
        (pod_total, min_radius),
        (t_mass, min_radius / 2.0),  # tether CoM at half radius
        (DEFAULT_HUB_MASS_KG, 0.0),
    ]
    moi = moment_of_inertia_kg_m2(masses_radii)
    energy = rotational_energy_j(moi, omega)
    spinup = spin_up_time_s(moi, omega, DEFAULT_MOTOR_POWER_W)

    return {
        "radius_m": round(min_radius, 1),
        "rpm": round(rpm, 2),
        "omega_rad_s": round(omega, 4),
        "apparent_gravity_g": round(apparent_gravity_g(omega, min_radius), 3),
        "centripetal_g": round(centripetal_acceleration_m_s2(omega, min_radius) / G0_M_S2, 3),
        "tether_tension_kn": round(tension / 1000.0, 1),
        "tether_diameter_mm": round(tether_diam * 1000.0, 1),
        "tether_mass_kg": round(t_mass, 1),
        "total_mass_kg": round(2 * pod_total + t_mass + DEFAULT_HUB_MASS_KG, 1),
        "moment_of_inertia_kg_m2": round(moi, 0),
        "rotational_energy_mj": round(energy / 1e6, 2),
        "spin_up_time_hours": round(spinup / 3600.0, 1),
        "gravity_gradient_frac": round(gravity_gradient_fraction(min_radius), 4),
        "coriolis_fraction": round(coriolis_fraction(omega, min_radius), 4),
        "comfort": comfort_check(rpm, min_radius),
    }


# =============================================================================
# State dataclass
# =============================================================================

@dataclass
class CentrifugeState:
    """Mutable state for the centrifuge habitat, advanced each sol."""
    radius_m: float = DEFAULT_RADIUS_M
    pod_mass_kg: float = DEFAULT_POD_MASS_KG
    crew_per_pod: int = DEFAULT_CREW_PER_POD
    hub_mass_kg: float = DEFAULT_HUB_MASS_KG

    omega_rad_s: float = 0.0
    target_omega_rad_s: float = 0.0
    phase: str = "stopped"  # stopped | spinning_up | nominal | spinning_down | emergency_stop

    motor_power_w: float = DEFAULT_MOTOR_POWER_W
    bearing_friction_coeff: float = DEFAULT_BEARING_FRICTION_COEFF
    bearing_radius_m: float = DEFAULT_BEARING_RADIUS_M

    # tether state
    tether_diameter_m: float = DEFAULT_TETHER_DIAMETER_M
    tether_safety_factor: float = DEFAULT_SAFETY_FACTOR

    # cumulative tracking
    sol: int = 0
    total_energy_input_j: float = 0.0
    total_friction_loss_j: float = 0.0
    crew_hours_at_target_g: float = 0.0
    crew_hours_total: float = 0.0
    peak_rpm: float = 0.0
    peak_tension_kn: float = 0.0
    emergency_stops: int = 0
    structural_warnings: int = 0

    @property
    def rpm(self) -> float:
        return rad_s_to_rpm(self.omega_rad_s)

    @property
    def pod_total_mass_kg(self) -> float:
        return self.pod_mass_kg + self.crew_per_pod * DEFAULT_CREW_MASS_KG

    @property
    def tether_cross_section_m2(self) -> float:
        if self.tether_diameter_m <= 0.0:
            return 0.0
        r = self.tether_diameter_m / 2.0
        return math.pi * r * r

    @property
    def tether_max_tension_n(self) -> float:
        return self.tether_cross_section_m2 * KEVLAR_TENSILE_MPA * 1e6

    @property
    def total_moment_of_inertia(self) -> float:
        t_mass = tether_mass_kg(self.radius_m * 2.0,
                                self.tether_cross_section_m2)
        return moment_of_inertia_kg_m2([
            (self.pod_total_mass_kg, self.radius_m),
            (self.pod_total_mass_kg, self.radius_m),
            (t_mass, self.radius_m / 2.0),
            (self.hub_mass_kg, 0.0),
        ])

    @property
    def current_tension_n(self) -> float:
        return tether_tension_n(self.pod_total_mass_kg,
                                self.omega_rad_s, self.radius_m)

    @property
    def apparent_gravity_g(self) -> float:
        return apparent_gravity_g(self.omega_rad_s, self.radius_m)

    @property
    def angular_momentum(self) -> float:
        return angular_momentum_kg_m2_s(self.total_moment_of_inertia,
                                        self.omega_rad_s)


@dataclass
class SolRecord:
    """Record of one sol's centrifuge operation."""
    sol: int = 0
    phase: str = "stopped"
    omega_rad_s: float = 0.0
    rpm: float = 0.0
    apparent_gravity_g: float = 0.0
    tension_kn: float = 0.0
    energy_input_kwh: float = 0.0
    friction_loss_kwh: float = 0.0
    crew_hours_at_target: float = 0.0
    structural_ok: bool = True
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Tick engine (one sol)
# =============================================================================

def start_spin(state: CentrifugeState, target_g: float = DEFAULT_TARGET_G) -> str | None:
    """Command the centrifuge to begin spinning up to target gravity."""
    if state.phase not in ("stopped", "nominal"):
        return f"Cannot start spin: centrifuge is {state.phase}"
    omega = omega_for_target_g(target_g, state.radius_m)
    max_omega = rpm_to_rad_s(MAX_RPM)
    if omega > max_omega:
        return f"Target {target_g}g requires {rad_s_to_rpm(omega):.1f} RPM, exceeds max {MAX_RPM}"
    state.target_omega_rad_s = omega
    state.phase = "spinning_up"
    return None


def stop_spin(state: CentrifugeState, emergency: bool = False) -> str | None:
    """Command the centrifuge to spin down."""
    if state.omega_rad_s <= 0.0 and state.phase == "stopped":
        return "Already stopped"
    if emergency:
        state.phase = "emergency_stop"
        state.emergency_stops += 1
    else:
        state.phase = "spinning_down"
        state.target_omega_rad_s = 0.0
    return None


def tick(state: CentrifugeState, available_power_w: float | None = None,
         dt_hours: float = HOURS_PER_SOL) -> SolRecord:
    """Advance the centrifuge by one sol."""
    state.sol += 1
    dt_s = dt_hours * 3600.0
    motor_power = min(available_power_w, state.motor_power_w) if available_power_w is not None else state.motor_power_w

    record = SolRecord(sol=state.sol, phase=state.phase)
    moi = state.total_moment_of_inertia

    if state.phase == "stopped":
        record.omega_rad_s = 0.0
        record.rpm = 0.0
        record.apparent_gravity_g = MARS_GRAVITY_M_S2 / G0_M_S2
        return record

    if state.phase == "emergency_stop":
        # Immediate stop (braking — ignoring energy dissipation for safety)
        state.omega_rad_s = 0.0
        state.phase = "stopped"
        record.phase = "stopped"
        record.omega_rad_s = 0.0
        record.warnings.append("Emergency stop executed")
        return record

    # Bearing friction (continuous)
    axial_load = 2.0 * state.pod_total_mass_kg * MARS_GRAVITY_M_S2
    friction_w = bearing_friction_power_w(
        state.bearing_friction_coeff, axial_load,
        state.omega_rad_s, state.bearing_radius_m)
    friction_energy_j = friction_w * dt_s
    state.total_friction_loss_j += friction_energy_j

    if state.phase == "spinning_up":
        _tick_spin_up(state, moi, motor_power, friction_w, dt_s)

    elif state.phase == "nominal":
        _tick_nominal(state, moi, motor_power, friction_w, dt_s, record)

    elif state.phase == "spinning_down":
        _tick_spin_down(state, moi, friction_w, dt_s)

    # Structural check
    tension = state.current_tension_n
    max_tension = state.tether_max_tension_n
    safety = max_tension / tension if tension > 0 else float("inf")

    if safety < 1.0:
        record.warnings.append("CRITICAL: Tether tension exceeds rated capacity!")
        record.structural_ok = False
        state.structural_warnings += 1
        # Auto emergency stop
        stop_spin(state, emergency=True)
    elif safety < state.tether_safety_factor:
        record.warnings.append(f"Tether safety factor {safety:.1f} below design {state.tether_safety_factor:.1f}")
        state.structural_warnings += 1

    # Crew tracking
    if state.phase == "nominal":
        hours_this_sol = dt_hours
        state.crew_hours_total += hours_this_sol * state.crew_per_pod * 2
        # Check if within 5% of target
        target_g_val = apparent_gravity_g(state.target_omega_rad_s, state.radius_m)
        current_g_val = state.apparent_gravity_g
        if target_g_val > 0 and abs(current_g_val - target_g_val) / target_g_val < 0.05:
            state.crew_hours_at_target_g += hours_this_sol * state.crew_per_pod * 2
            record.crew_hours_at_target = hours_this_sol * state.crew_per_pod * 2

    # Update peaks
    state.peak_rpm = max(state.peak_rpm, state.rpm)
    state.peak_tension_kn = max(state.peak_tension_kn, tension / 1000.0)

    # Fill record
    record.phase = state.phase
    record.omega_rad_s = round(state.omega_rad_s, 6)
    record.rpm = round(state.rpm, 2)
    record.apparent_gravity_g = round(state.apparent_gravity_g, 3)
    record.tension_kn = round(tension / 1000.0, 2)
    record.friction_loss_kwh = round(friction_energy_j / 3_600_000.0, 4)
    record.structural_ok = record.structural_ok if record.warnings else True
    return record


def _tick_spin_up(state: CentrifugeState, moi: float,
                  motor_power: float, friction_w: float,
                  dt_s: float) -> None:
    """Spin up: motor torque minus friction."""
    if moi <= 0.0 or motor_power <= 0.0:
        return

    net_power = motor_power - friction_w
    if net_power <= 0.0:
        # Motor can't overcome friction
        return

    # Energy method: ΔE = P_net · Δt = ½I(ω₂² - ω₁²)
    energy_in = net_power * dt_s
    state.total_energy_input_j += motor_power * dt_s

    omega_sq = state.omega_rad_s**2 + 2.0 * energy_in / moi
    if omega_sq < 0.0:
        omega_sq = 0.0
    new_omega = math.sqrt(omega_sq)

    if new_omega >= state.target_omega_rad_s:
        state.omega_rad_s = state.target_omega_rad_s
        state.phase = "nominal"
    else:
        state.omega_rad_s = new_omega


def _tick_nominal(state: CentrifugeState, moi: float,
                  motor_power: float, friction_w: float,
                  dt_s: float, record: SolRecord) -> None:
    """Maintain speed: compensate friction."""
    # Motor compensates friction to hold omega constant
    compensation_power = min(friction_w, motor_power)
    state.total_energy_input_j += compensation_power * dt_s
    record.energy_input_kwh = round(compensation_power * dt_s / 3_600_000.0, 4)

    if compensation_power < friction_w:
        # Not enough power to maintain — slowly spinning down
        deficit_power = friction_w - compensation_power
        energy_lost = deficit_power * dt_s
        omega_sq = state.omega_rad_s**2 - 2.0 * energy_lost / moi
        if omega_sq <= 0.0:
            state.omega_rad_s = 0.0
            state.phase = "stopped"
        else:
            state.omega_rad_s = math.sqrt(omega_sq)
        record.warnings.append("Insufficient power to maintain spin")


def _tick_spin_down(state: CentrifugeState, moi: float,
                    friction_w: float, dt_s: float) -> None:
    """Spin down using friction (no motor, passive deceleration)."""
    if moi <= 0.0 or state.omega_rad_s <= 0.0:
        state.omega_rad_s = 0.0
        state.phase = "stopped"
        return

    # Use friction + controlled braking
    # For controlled spindown, we add motor-as-brake at same power
    brake_power = friction_w + state.motor_power_w
    energy_removed = brake_power * dt_s

    omega_sq = state.omega_rad_s**2 - 2.0 * energy_removed / moi
    if omega_sq <= 0.0:
        state.omega_rad_s = 0.0
        state.phase = "stopped"
    else:
        state.omega_rad_s = math.sqrt(omega_sq)


# =============================================================================
# Simulation runner
# =============================================================================

def run_simulation(
    target_g: float = DEFAULT_TARGET_G,
    radius_m: float = DEFAULT_RADIUS_M,
    max_sols: int = 30,
    pod_mass_kg: float = DEFAULT_POD_MASS_KG,
    crew_per_pod: int = DEFAULT_CREW_PER_POD,
) -> dict[str, Any]:
    """Run a complete centrifuge lifecycle: spin up → hold → spin down."""
    state = CentrifugeState(
        radius_m=radius_m,
        pod_mass_kg=pod_mass_kg,
        crew_per_pod=crew_per_pod,
    )

    err = start_spin(state, target_g)
    if err:
        return {"error": err}

    records: list[SolRecord] = []
    hold_sols = 0
    spin_down_started = False

    for sol in range(max_sols):
        rec = tick(state)
        records.append(rec)

        if state.phase == "nominal":
            hold_sols += 1
            # Hold for 10 sols then spin down
            if hold_sols >= 10 and not spin_down_started:
                stop_spin(state)
                spin_down_started = True

        if state.phase == "stopped" and sol > 0:
            break

    design = design_centrifuge(target_g, pod_mass_kg=pod_mass_kg,
                               crew_per_pod=crew_per_pod)

    return {
        "design": design,
        "sols_simulated": len(records),
        "final_phase": state.phase,
        "peak_rpm": round(state.peak_rpm, 2),
        "peak_tension_kn": round(state.peak_tension_kn, 2),
        "total_energy_kwh": round(state.total_energy_input_j / 3_600_000.0, 2),
        "total_friction_loss_kwh": round(state.total_friction_loss_j / 3_600_000.0, 2),
        "crew_hours_at_target_g": round(state.crew_hours_at_target_g, 1),
        "crew_hours_total": round(state.crew_hours_total, 1),
        "emergency_stops": state.emergency_stops,
        "structural_warnings": state.structural_warnings,
    }
