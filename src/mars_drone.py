"""mars_drone.py -- Mars Autonomous Aerial Scout (Rotorcraft).

Ingenuity proved powered flight on Mars in 2021.  But the colony needs
more than a tech demo — it needs persistent aerial capability for ice
deposit surveys, landing-zone clearance, hab exterior inspection, and
emergency search-and-rescue across terrain no rover can cross.

This module models a colony-scale Mars rotorcraft: coaxial contra-
rotating blades, solar-charged batteries, autonomous navigation via
visual odometry, and a modular payload bay (camera, LIDAR, relay radio,
sample grabber).

Physics modelled
----------------
* **Rotor aerodynamics** -- Thrust from momentum theory:
  T = C_T * rho * A * (Omega*R)^2, where rho is Mars atmospheric
  density (~0.02 kg/m^3), A is disk area, Omega is angular velocity,
  R is rotor radius.  Mars air is ~1/60th Earth sea level density,
  so tip speeds must be near-sonic (~Mach 0.7 at Mars speed of sound
  ~240 m/s) and blade area must be large relative to vehicle mass.

* **Power required** -- hover power from ideal momentum theory:
  P_hover = T^(3/2) / sqrt(2 * rho * A).  Forward flight:
  P_fwd = P_hover * (1 + mu^2) where mu = V_forward / V_tip.
  Parasite drag adds P_drag = 0.5 * rho * Cd * A_body * V^3.

* **Battery energy** -- lithium-ion at Mars temperatures.  Capacity
  degrades below -20 C (Arrhenius model).  Charge from body-mounted
  solar cells at Mars insolation (~590 W/m^2 peak, reduced by dust
  and season).

* **Thermal model** -- electronics and battery must stay above -40 C.
  Heater power competes with flight power.  Mars ambient: 180-290 K
  depending on season/time.  Insulation + heater keep battery in
  operating range.

* **Navigation** -- visual odometry with downward camera + IMU.
  Position uncertainty grows with distance: sigma ~ k * sqrt(d).
  No GPS on Mars.  Range limited by nav uncertainty budget.

* **Communication link** -- UHF relay to base.  Free-space path loss
  increases with distance.  Beyond line of sight, terrain blocks signal.
  Max effective range ~5 km with 1 W transmitter.

* **Blade erosion** -- Mars dust (basaltic, ~100 um particles) erodes
  blade leading edges.  Erosion rate proportional to tip speed and
  dust optical depth.  Blade efficiency degrades over flight hours.

* **Payload** -- modular bay carries one payload at a time.  Payload
  mass reduces flight time.  Types: survey camera, LIDAR mapper,
  comms relay, sample grabber, emergency beacon.

Conservation laws
-----------------
- Energy: battery_used = P_flight * dt + P_heater * dt (<= battery_capacity)
- Thrust >= weight for flight (T >= m*g_mars)
- Range <= V * t_flight (<= battery / power * velocity)
- Blade health in [0, 1], monotonically decreasing
- Temperature: ambient <= T_battery <= T_max
- Nav uncertainty >= 0, monotonically increasing during flight
- Altitude >= 0

One tick = one sol of operations (charge + flight + maintenance).
Mass in kg, distance in metres, velocity in m/s, power in watts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MARS_GRAVITY_M_S2 = 3.72076
MARS_SURFACE_DENSITY_KG_M3 = 0.020
MARS_SCALE_HEIGHT_M = 11_100.0
MARS_SPEED_OF_SOUND_M_S = 240.0
MARS_AMBIENT_TEMP_K = 210.0
MARS_SOLAR_FLUX_W_M2 = 590.0

SECONDS_PER_SOL = 88_775.0
HOURS_PER_SOL = 24.66
DAYLIGHT_FRACTION = 0.5

# ---------------------------------------------------------------------------
# Rotor aerodynamics
# ---------------------------------------------------------------------------

DEFAULT_ROTOR_RADIUS_M = 0.65
DEFAULT_NUM_BLADES = 4
DEFAULT_BLADE_CHORD_M = 0.06
DEFAULT_ROTOR_RPM = 2_700.0
DEFAULT_CT = 0.015                        # thrust coefficient (4-blade optimised)
FIGURE_OF_MERIT = 0.65                    # rotor efficiency
MAX_TIP_MACH = 0.75

# ---------------------------------------------------------------------------
# Vehicle parameters
# ---------------------------------------------------------------------------

DEFAULT_EMPTY_MASS_KG = 4.5
DEFAULT_BATTERY_CAPACITY_WH = 180.0
DEFAULT_SOLAR_PANEL_AREA_M2 = 0.12
DEFAULT_SOLAR_EFFICIENCY = 0.28
DEFAULT_BODY_DRAG_COEFF = 0.5
DEFAULT_BODY_AREA_M2 = 0.04

# ---------------------------------------------------------------------------
# Battery thermal model
# ---------------------------------------------------------------------------

BATTERY_MIN_TEMP_K = 233.0               # -40 C: below this, capacity -> 0
BATTERY_NOMINAL_TEMP_K = 293.0           # +20 C: nominal
BATTERY_MAX_TEMP_K = 333.0               # +60 C: thermal runaway
HEATER_POWER_W = 3.5
BATTERY_THERMAL_MASS_J_K = 120.0
INSULATION_CONDUCTANCE_W_K = 0.25

BATTERY_TEMP_CAPACITY_FACTOR_SLOPE = 0.012

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

NAV_UNCERTAINTY_COEFF = 0.02
MAX_NAV_UNCERTAINTY_M = 50.0
COMM_MAX_RANGE_M = 5_000.0
COMM_POWER_W = 1.0

# ---------------------------------------------------------------------------
# Blade erosion
# ---------------------------------------------------------------------------

BLADE_EROSION_RATE_PER_HOUR = 0.002
DUST_EROSION_MULTIPLIER = 0.5
BLADE_EFFICIENCY_AT_ZERO_HEALTH = 0.60
MIN_BLADE_HEALTH_FOR_FLIGHT = 0.10

# ---------------------------------------------------------------------------
# Payload types
# ---------------------------------------------------------------------------

PAYLOAD_TYPES = {
    "camera":    {"mass_kg": 0.3, "power_w": 2.0, "description": "survey camera"},
    "lidar":     {"mass_kg": 0.8, "power_w": 5.0, "description": "terrain mapper"},
    "relay":     {"mass_kg": 0.5, "power_w": 3.0, "description": "comms relay"},
    "grabber":   {"mass_kg": 1.2, "power_w": 1.0, "description": "sample grabber"},
    "beacon":    {"mass_kg": 0.4, "power_w": 4.0, "description": "emergency beacon"},
}


# ---------------------------------------------------------------------------
# Helper physics functions
# ---------------------------------------------------------------------------

def air_density_at_altitude(altitude_m: float) -> float:
    """Atmospheric density at altitude using exponential model."""
    if altitude_m < 0.0:
        altitude_m = 0.0
    return MARS_SURFACE_DENSITY_KG_M3 * math.exp(-altitude_m / MARS_SCALE_HEIGHT_M)


def rotor_thrust_n(rotor_radius_m: float, rpm: float,
                   ct: float, rho: float, blade_health: float) -> float:
    """Thrust from coaxial rotor system using blade-element momentum theory.

    T = C_T * rho * A * (Omega*R)^2, adjusted for blade health.
    Coaxial factor: ~1.8x single rotor (interference loss).
    """
    if rpm <= 0.0 or rotor_radius_m <= 0.0 or rho <= 0.0:
        return 0.0
    omega = rpm * 2.0 * math.pi / 60.0
    tip_speed = omega * rotor_radius_m
    tip_mach = tip_speed / MARS_SPEED_OF_SOUND_M_S
    if tip_mach > 1.0:
        return 0.0
    mach_penalty = 1.0
    if tip_mach > MAX_TIP_MACH:
        mach_penalty = max(0.0, 1.0 - 2.0 * (tip_mach - MAX_TIP_MACH))
    disk_area = math.pi * rotor_radius_m ** 2
    blade_eff = BLADE_EFFICIENCY_AT_ZERO_HEALTH + (
        1.0 - BLADE_EFFICIENCY_AT_ZERO_HEALTH) * max(0.0, min(1.0, blade_health))
    coaxial_factor = 1.8
    thrust = ct * rho * disk_area * tip_speed ** 2 * blade_eff * mach_penalty * coaxial_factor
    return max(0.0, thrust)


def hover_power_w(thrust_n: float, rho: float, disk_area_m2: float) -> float:
    """Ideal hover power: P = T^(3/2) / sqrt(2*rho*A), divided by FOM."""
    if thrust_n <= 0.0 or rho <= 0.0 or disk_area_m2 <= 0.0:
        return 0.0
    denom = math.sqrt(2.0 * rho * disk_area_m2)
    if denom <= 0.0:
        return 0.0
    ideal = thrust_n ** 1.5 / denom
    return ideal / FIGURE_OF_MERIT


def forward_flight_power_w(hover_power: float, v_forward: float,
                           v_tip: float, rho: float,
                           body_drag_coeff: float,
                           body_area_m2: float) -> float:
    """Power in forward flight: rotor + parasite drag."""
    if hover_power <= 0.0:
        return 0.0
    if v_tip <= 0.0:
        return hover_power
    mu = v_forward / v_tip if v_tip > 0.0 else 0.0
    rotor_power = hover_power * (1.0 + mu ** 2)
    drag_power = 0.5 * rho * body_drag_coeff * body_area_m2 * max(0.0, v_forward) ** 3
    return rotor_power + drag_power


def battery_capacity_factor(temp_k: float) -> float:
    """Battery capacity as fraction of nominal, based on temperature."""
    if temp_k >= BATTERY_NOMINAL_TEMP_K:
        return 1.0
    if temp_k <= BATTERY_MIN_TEMP_K:
        return 0.0
    delta = BATTERY_NOMINAL_TEMP_K - temp_k
    factor = 1.0 - BATTERY_TEMP_CAPACITY_FACTOR_SLOPE * delta
    return max(0.0, min(1.0, factor))


def solar_charge_wh(panel_area_m2: float, efficiency: float,
                    hours: float, dust_factor: float) -> float:
    """Energy harvested from solar panels during daylight hours."""
    if panel_area_m2 <= 0.0 or hours <= 0.0:
        return 0.0
    effective_flux = MARS_SOLAR_FLUX_W_M2 * max(0.0, min(1.0, dust_factor))
    avg_cosine = 0.637
    power_w = panel_area_m2 * efficiency * effective_flux * avg_cosine
    return power_w * hours


def nav_uncertainty_m(distance_m: float) -> float:
    """Position uncertainty after flying distance_m (visual odometry)."""
    if distance_m <= 0.0:
        return 0.0
    return NAV_UNCERTAINTY_COEFF * math.sqrt(distance_m)


def comm_link_margin_db(distance_m: float, freq_hz: float = 400e6) -> float:
    """Free-space path loss margin for UHF link to base."""
    if distance_m <= 0.0:
        return 100.0
    wavelength = 3e8 / freq_hz
    fspl_db = 20.0 * math.log10(4.0 * math.pi * distance_m / wavelength)
    tx_power_dbm = 10.0 * math.log10(COMM_POWER_W * 1000.0)
    rx_sensitivity_dbm = -110.0
    margin = tx_power_dbm - fspl_db - rx_sensitivity_dbm
    return margin


def blade_erosion(flight_hours: float, dust_tau: float) -> float:
    """Blade health loss for given flight hours and dust conditions."""
    if flight_hours <= 0.0:
        return 0.0
    base_loss = BLADE_EROSION_RATE_PER_HOUR * flight_hours
    dust_loss = DUST_EROSION_MULTIPLIER * max(0.0, dust_tau) * flight_hours
    return base_loss + dust_loss


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DroneState:
    """Full state of the Mars aerial scout."""

    rotor_radius_m: float = DEFAULT_ROTOR_RADIUS_M
    num_blades: int = DEFAULT_NUM_BLADES
    rotor_rpm: float = DEFAULT_ROTOR_RPM
    ct: float = DEFAULT_CT
    empty_mass_kg: float = DEFAULT_EMPTY_MASS_KG
    body_drag_coeff: float = DEFAULT_BODY_DRAG_COEFF
    body_area_m2: float = DEFAULT_BODY_AREA_M2

    battery_capacity_wh: float = DEFAULT_BATTERY_CAPACITY_WH
    battery_charge_wh: float = DEFAULT_BATTERY_CAPACITY_WH
    battery_temp_k: float = BATTERY_NOMINAL_TEMP_K
    solar_panel_area_m2: float = DEFAULT_SOLAR_PANEL_AREA_M2
    solar_efficiency: float = DEFAULT_SOLAR_EFFICIENCY

    blade_health: float = 1.0

    payload_type: str = ""
    payload_mass_kg: float = 0.0
    payload_power_w: float = 0.0

    total_flights: int = 0
    total_flight_hours: float = 0.0
    total_distance_m: float = 0.0
    total_energy_used_wh: float = 0.0
    failed_flights: int = 0
    sol: int = 0

    last_range_m: float = 0.0
    last_altitude_m: float = 0.0
    last_flight_time_s: float = 0.0
    last_energy_wh: float = 0.0


@dataclass
class FlightPlan:
    """Parameters for a single drone flight."""
    target_altitude_m: float = 10.0
    target_range_m: float = 500.0
    cruise_speed_m_s: float = 10.0
    payload_type: str = "camera"
    dust_tau: float = 0.5


@dataclass
class FlightResult:
    """Telemetry from a single flight."""
    success: bool = False
    range_m: float = 0.0
    altitude_m: float = 0.0
    flight_time_s: float = 0.0
    energy_used_wh: float = 0.0
    hover_power_w: float = 0.0
    cruise_power_w: float = 0.0
    nav_uncertainty_m: float = 0.0
    comm_margin_db: float = 0.0
    max_thrust_n: float = 0.0
    weight_n: float = 0.0
    blade_health_after: float = 0.0
    battery_after_wh: float = 0.0
    failure_reason: str = ""


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def create_drone(**overrides) -> DroneState:
    """Factory: create a drone with optional parameter overrides."""
    return DroneState(**overrides)


def attach_payload(state: DroneState, payload_type: str) -> bool:
    """Attach a payload to the drone.  Returns True on success."""
    if payload_type not in PAYLOAD_TYPES:
        return False
    info = PAYLOAD_TYPES[payload_type]
    state.payload_type = payload_type
    state.payload_mass_kg = info["mass_kg"]
    state.payload_power_w = info["power_w"]
    return True


def detach_payload(state: DroneState) -> str:
    """Remove current payload.  Returns the removed payload type."""
    removed = state.payload_type
    state.payload_type = ""
    state.payload_mass_kg = 0.0
    state.payload_power_w = 0.0
    return removed


def total_mass_kg(state: DroneState) -> float:
    """Total vehicle mass including payload."""
    return state.empty_mass_kg + state.payload_mass_kg


def can_fly(state: DroneState) -> tuple:
    """Pre-flight check.  Returns (ok, reason)."""
    if state.blade_health < MIN_BLADE_HEALTH_FOR_FLIGHT:
        return False, "blades_worn"
    if state.battery_charge_wh <= 0.0:
        return False, "battery_empty"
    if state.battery_temp_k < BATTERY_MIN_TEMP_K:
        return False, "battery_too_cold"
    if state.battery_temp_k > BATTERY_MAX_TEMP_K:
        return False, "battery_too_hot"
    mass = total_mass_kg(state)
    rho = MARS_SURFACE_DENSITY_KG_M3
    thrust = rotor_thrust_n(state.rotor_radius_m, state.rotor_rpm,
                            state.ct, rho, state.blade_health)
    weight = mass * MARS_GRAVITY_M_S2
    if thrust < weight:
        return False, "insufficient_thrust"
    return True, "ok"


def simulate_flight(state: DroneState, plan: FlightPlan) -> FlightResult:
    """Simulate a single drone flight and update state.

    Flight profile: vertical climb -> cruise to target -> loiter -> return.
    Total range = 2 * target_range (out and back).
    """
    result = FlightResult()

    flyable, reason = can_fly(state)
    if not flyable:
        result.failure_reason = reason
        state.failed_flights += 1
        return result

    if plan.payload_type and state.payload_type != plan.payload_type:
        if not attach_payload(state, plan.payload_type):
            result.failure_reason = "invalid_payload"
            state.failed_flights += 1
            return result

    mass = total_mass_kg(state)
    weight = mass * MARS_GRAVITY_M_S2

    rho = air_density_at_altitude(plan.target_altitude_m)
    disk_area = math.pi * state.rotor_radius_m ** 2

    thrust = rotor_thrust_n(state.rotor_radius_m, state.rotor_rpm,
                            state.ct, rho, state.blade_health)
    if thrust < weight:
        result.failure_reason = "insufficient_thrust_at_altitude"
        result.max_thrust_n = thrust
        result.weight_n = weight
        state.failed_flights += 1
        return result

    result.max_thrust_n = thrust
    result.weight_n = weight

    p_hover = hover_power_w(weight, rho, disk_area)
    result.hover_power_w = p_hover

    climb_speed = 2.0
    climb_time_s = plan.target_altitude_m / climb_speed if plan.target_altitude_m > 0 else 0.0
    climb_energy_wh = p_hover * 1.2 * climb_time_s / 3600.0

    tip_speed = state.rotor_rpm * 2.0 * math.pi / 60.0 * state.rotor_radius_m
    cruise_speed = max(0.1, min(plan.cruise_speed_m_s, tip_speed * 0.3))
    total_cruise_distance = 2.0 * max(0.0, plan.target_range_m)
    cruise_time_s = total_cruise_distance / cruise_speed if cruise_speed > 0 else 0.0

    p_cruise = forward_flight_power_w(p_hover, cruise_speed, tip_speed,
                                      rho, state.body_drag_coeff,
                                      state.body_area_m2)
    result.cruise_power_w = p_cruise
    cruise_energy_wh = p_cruise * cruise_time_s / 3600.0

    loiter_time_s = 60.0
    loiter_energy_wh = p_hover * loiter_time_s / 3600.0

    descent_time_s = climb_time_s * 1.5
    descent_energy_wh = p_hover * 0.5 * descent_time_s / 3600.0

    total_flight_time_s = climb_time_s + cruise_time_s + loiter_time_s + descent_time_s
    payload_energy_wh = state.payload_power_w * total_flight_time_s / 3600.0
    comm_energy_wh = COMM_POWER_W * total_flight_time_s / 3600.0

    heater_energy_wh = 0.0
    if state.battery_temp_k < BATTERY_NOMINAL_TEMP_K:
        heater_energy_wh = HEATER_POWER_W * total_flight_time_s / 3600.0

    total_energy_wh = (climb_energy_wh + cruise_energy_wh + loiter_energy_wh +
                       descent_energy_wh + payload_energy_wh + comm_energy_wh +
                       heater_energy_wh)

    cap_factor = battery_capacity_factor(state.battery_temp_k)
    available_wh = state.battery_charge_wh * cap_factor

    if total_energy_wh > available_wh:
        usable_for_cruise = available_wh - (climb_energy_wh + loiter_energy_wh +
                                            descent_energy_wh + payload_energy_wh +
                                            comm_energy_wh + heater_energy_wh)
        if usable_for_cruise <= 0.0:
            result.failure_reason = "insufficient_battery"
            result.energy_used_wh = total_energy_wh
            state.failed_flights += 1
            return result
        fraction = usable_for_cruise / cruise_energy_wh if cruise_energy_wh > 0 else 0.0
        total_cruise_distance *= fraction
        cruise_time_s *= fraction
        cruise_energy_wh = usable_for_cruise
        total_flight_time_s = climb_time_s + cruise_time_s + loiter_time_s + descent_time_s
        total_energy_wh = available_wh

    max_distance = total_cruise_distance / 2.0
    nav_unc = nav_uncertainty_m(max_distance)
    if nav_unc > MAX_NAV_UNCERTAINTY_M:
        result.failure_reason = "nav_uncertainty_exceeded"
        result.nav_uncertainty_m = nav_unc
        state.failed_flights += 1
        return result

    comm_margin = comm_link_margin_db(max_distance)
    if comm_margin < 0.0 and max_distance > COMM_MAX_RANGE_M:
        result.failure_reason = "comm_link_lost"
        result.comm_margin_db = comm_margin
        state.failed_flights += 1
        return result

    # Flight succeeds
    state.battery_charge_wh -= total_energy_wh / cap_factor if cap_factor > 0 else total_energy_wh
    state.battery_charge_wh = max(0.0, state.battery_charge_wh)

    flight_hours = total_flight_time_s / 3600.0
    erosion = blade_erosion(flight_hours, plan.dust_tau)
    state.blade_health = max(0.0, state.blade_health - erosion)

    state.total_flights += 1
    state.total_flight_hours += flight_hours
    state.total_distance_m += total_cruise_distance
    state.total_energy_used_wh += total_energy_wh

    state.last_range_m = max_distance
    state.last_altitude_m = plan.target_altitude_m
    state.last_flight_time_s = total_flight_time_s
    state.last_energy_wh = total_energy_wh

    result.success = True
    result.range_m = max_distance
    result.altitude_m = plan.target_altitude_m
    result.flight_time_s = total_flight_time_s
    result.energy_used_wh = total_energy_wh
    result.hover_power_w = p_hover
    result.cruise_power_w = p_cruise
    result.nav_uncertainty_m = nav_unc
    result.comm_margin_db = comm_margin
    result.blade_health_after = state.blade_health
    result.battery_after_wh = state.battery_charge_wh
    return result


def charge_battery(state: DroneState, hours: float,
                   dust_factor: float = 0.8) -> float:
    """Charge battery from solar panels.  Returns Wh added."""
    if hours <= 0.0:
        return 0.0
    daylight_hours = hours * DAYLIGHT_FRACTION
    energy = solar_charge_wh(state.solar_panel_area_m2,
                             state.solar_efficiency,
                             daylight_hours, dust_factor)
    old_charge = state.battery_charge_wh
    state.battery_charge_wh = min(state.battery_capacity_wh,
                                  state.battery_charge_wh + energy)
    return state.battery_charge_wh - old_charge


def thermal_update(state: DroneState, ambient_temp_k: float,
                   seconds: float) -> float:
    """Update battery temperature toward ambient.  Returns new temp."""
    if seconds <= 0.0:
        return state.battery_temp_k
    delta_t = state.battery_temp_k - ambient_temp_k
    heat_leak = INSULATION_CONDUCTANCE_W_K * delta_t
    heater = 0.0
    if state.battery_temp_k < BATTERY_NOMINAL_TEMP_K and state.battery_charge_wh > 0.0:
        heater = HEATER_POWER_W
        heater_energy = heater * seconds / 3600.0
        state.battery_charge_wh = max(0.0, state.battery_charge_wh - heater_energy)
    net_heat = heater - heat_leak
    temp_change = net_heat * seconds / BATTERY_THERMAL_MASS_J_K
    state.battery_temp_k += temp_change
    state.battery_temp_k = max(ambient_temp_k, state.battery_temp_k)
    state.battery_temp_k = min(BATTERY_MAX_TEMP_K, state.battery_temp_k)
    return state.battery_temp_k


def replace_blades(state: DroneState) -> None:
    """Replace rotor blades, restoring health to 1.0."""
    state.blade_health = 1.0


def tick(state: DroneState,
         flight_plans: list = None,
         ambient_temp_k: float = MARS_AMBIENT_TEMP_K,
         dust_tau: float = 0.5) -> dict:
    """Advance the drone by one sol.

    Sol schedule:
    1. Thermal equilibration overnight (12 hours at ambient)
    2. Solar charging during daylight (~12 hours)
    3. Execute flight plans in sequence
    4. Post-flight charging with remaining daylight
    """
    state.sol += 1
    plans = flight_plans or []

    night_seconds = SECONDS_PER_SOL / 2.0
    thermal_update(state, ambient_temp_k, night_seconds)

    dust_factor = max(0.0, min(1.0, math.exp(-dust_tau)))
    pre_charge = charge_battery(state, 6.0, dust_factor)

    thermal_update(state, ambient_temp_k, 6.0 * 3600.0)

    flight_results = []
    flights_ok = 0
    flights_fail = 0
    total_distance = 0.0

    for plan in plans:
        plan.dust_tau = dust_tau
        res = simulate_flight(state, plan)
        flight_results.append({
            "success": res.success,
            "range_m": round(res.range_m, 1),
            "altitude_m": round(res.altitude_m, 1),
            "flight_time_s": round(res.flight_time_s, 1),
            "energy_wh": round(res.energy_used_wh, 2),
            "nav_uncertainty_m": round(res.nav_uncertainty_m, 2),
            "comm_margin_db": round(res.comm_margin_db, 1),
            "failure_reason": res.failure_reason,
        })
        if res.success:
            flights_ok += 1
            total_distance += res.range_m
        else:
            flights_fail += 1

    post_charge = charge_battery(state, 6.0, dust_factor)

    return {
        "sol": state.sol,
        "flights_attempted": len(plans),
        "flights_succeeded": flights_ok,
        "flights_failed": flights_fail,
        "total_distance_m": round(total_distance, 1),
        "battery_wh": round(state.battery_charge_wh, 2),
        "battery_temp_k": round(state.battery_temp_k, 2),
        "blade_health": round(state.blade_health, 4),
        "pre_charge_wh": round(pre_charge, 2),
        "post_charge_wh": round(post_charge, 2),
        "dust_tau": dust_tau,
        "cumulative_flights": state.total_flights,
        "cumulative_distance_m": round(state.total_distance_m, 1),
        "cumulative_flight_hours": round(state.total_flight_hours, 4),
        "results": flight_results,
    }


def status(state: DroneState) -> dict:
    """Non-mutating snapshot for dashboards."""
    flyable, reason = can_fly(state)
    return {
        "sol": state.sol,
        "operational": flyable,
        "status_reason": reason,
        "battery_pct": round(100.0 * state.battery_charge_wh / max(1.0, state.battery_capacity_wh), 1),
        "battery_temp_k": round(state.battery_temp_k, 1),
        "blade_health_pct": round(100.0 * state.blade_health, 1),
        "payload": state.payload_type or "none",
        "total_flights": state.total_flights,
        "total_distance_m": round(state.total_distance_m, 1),
        "total_flight_hours": round(state.total_flight_hours, 2),
    }
