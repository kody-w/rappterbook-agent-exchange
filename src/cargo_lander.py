"""cargo_lander.py — Mars Cargo Entry, Descent & Landing (EDL) Physics.

"Seven minutes of terror."  A cargo capsule arrives from Mars orbit at
~3.5 km/s and must reach the surface intact.  No runway, no ocean —
just a thin CO₂ atmosphere (0.6% of Earth's surface pressure) and a
rocky floor.  The three-phase descent — atmospheric entry, parachute
deceleration, powered terminal descent — must shed >99.9% of the
capsule's kinetic energy without crushing the payload or running out
of propellant.

Physics modelled
----------------
* **Atmospheric entry** — Hypersonic drag decelerates the capsule from
  orbital velocity.  F_drag = 0.5 × ρ × v² × Cd × A.  Mars atmosphere
  density follows an exponential profile: ρ(h) = ρ₀ × exp(−h / H)
  where scale height H ≈ 11.1 km and ρ₀ ≈ 0.020 kg/m³ at the surface.
  Peak deceleration typically 10–15 g for unmanned cargo.

* **Heat shield ablation** — Kinetic energy converts to heat during
  entry.  Peak heating ~1600°C (~1873 K).  Ablative thermal protection
  mass fraction 15–20% of entry mass.  Heat shield temperature modelled
  via convective heating: q̇ ∝ √(ρ/r_nose) × v³.  Ablation rate
  proportional to heat flux above threshold temperature.

* **Supersonic parachute** — Deployed at ~Mach 2 (~450 m/s) at ~10 km
  altitude.  Disk-gap-band chute with Cd ≈ 0.4–0.6.  Even large chutes
  only slow to ~60–100 m/s on Mars — not enough for safe landing.

* **Powered descent** — Throttleable retrorockets shed remaining
  velocity.  Thrust-to-weight > 1.0 (Mars g = 3.72 m/s²).  Propellant
  consumption follows the Tsiolkovsky rocket equation:
  Δv = Isp × g₀ × ln(m_initial / m_final).

* **Ballistic coefficient** — β = m / (Cd × A).  Determines the
  deceleration profile through the atmosphere.  Higher β means deeper
  penetration and later deceleration.

* **Abort conditions** — Dust storm (optical depth τ > 2.0),
  surface wind > 25 m/s, fuel margin < 5%.

Conservation laws
-----------------
- Total energy: KE + PE + heat_dissipated + fuel_energy_used = initial_energy
  (within numerical tolerance of the integrator)
- Mass: payload + heatshield + parachute + propellant + structure = entry_mass
  (exact, enforced every tick)
- Propellant consumed ≤ propellant loaded (no negative fuel)
- Velocity ≥ 0 during descent (capsule doesn't fly upward)
- Altitude monotonically decreasing during active descent
- Deceleration bounded by structural limit (capsule integrity)
- Heat shield temperature bounded by physics (ablation rate)

Reference:
  - Mars atmospheric density: ρ₀ ≈ 0.020 kg/m³ (Seiff et al. 1997)
  - Scale height: H ≈ 11.1 km (varies 8–12 km with season)
  - Mars surface gravity: 3.72076 m/s²
  - Speed of sound on Mars: ~240 m/s at surface (CO₂ atmosphere)
  - Peak heating rate for MSL-class entry: ~200 W/cm²
  - PICA heat shield ablation threshold: ~2500 K
  - Supersonic parachute Cd (DGB): 0.4–0.6 (Cruz et al. 2014)
  - Mars 2020 sky-crane: Isp ~226 s (hydrazine monopropellant)
  - Mars sol: 24 h 39 m 35 s ≈ 88,775 seconds

One tick = one second.  Velocity in m/s, altitude in m, mass in kg,
force in N, temperature in K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any


# -- Mars environment constants -----------------------------------------------

MARS_SURFACE_GRAVITY_M_S2 = 3.72076          # Mars surface gravity (m/s²)
MARS_RADIUS_M = 3_389_500.0                  # Mars mean radius (m)
MARS_SURFACE_PRESSURE_PA = 610.0             # average surface pressure (Pa)
MARS_SURFACE_TEMP_K = 210.0                  # average surface temperature (K)
MARS_ATM_DENSITY_SURFACE = 0.020             # surface atmospheric density (kg/m³)
MARS_SCALE_HEIGHT_M = 11_100.0               # atmospheric scale height (m)
MARS_SPEED_OF_SOUND_M_S = 240.0              # approx speed of sound on Mars (m/s)
MARS_CO2_GAMMA = 1.3                         # heat capacity ratio for CO₂

# -- Entry phase constants ----------------------------------------------------

DEFAULT_ENTRY_VELOCITY_M_S = 3_500.0         # orbital insertion entry velocity (m/s)
DEFAULT_ENTRY_ALTITUDE_M = 125_000.0         # atmospheric interface (m)
DEFAULT_ENTRY_ANGLE_DEG = -15.0              # flight path angle (negative = descending)

HEATSHIELD_MASS_FRACTION = 0.17              # 17% of entry mass for heat shield
HEATSHIELD_ABLATION_THRESHOLD_K = 2_500.0    # PICA ablation onset temperature (K)
HEATSHIELD_MAX_TEMP_K = 3_500.0              # structural failure temperature (K)
HEATSHIELD_SPECIFIC_HEAT_J_KGK = 1_200.0     # specific heat of PICA ablator (J/kg·K)
HEATSHIELD_ABLATION_ENERGY_J_KG = 8.0e6      # energy to ablate 1 kg of heat shield (J)

# Convective heating coefficient: q_dot = k_h * sqrt(rho / r_nose) * v^3
# Units: W/m² when rho in kg/m³, r_nose in m, v in m/s
CONVECTIVE_HEATING_COEFF = 1.9e-8            # Sutton-Graves constant for CO₂ (simplified)
DEFAULT_NOSE_RADIUS_M = 2.25                 # nose radius of capsule (m)

# Drag
DEFAULT_ENTRY_CD = 1.05                      # hypersonic drag coefficient (blunt body)
DEFAULT_CAPSULE_AREA_M2 = 15.9               # cross-sectional area ~4.5m diameter (m²)

# Structural limits
MAX_DECELERATION_G = 20.0                    # structural g-limit for cargo
MAX_DECELERATION_M_S2 = MAX_DECELERATION_G * 9.80665  # in m/s²

# -- Parachute phase constants ------------------------------------------------

PARACHUTE_DEPLOY_MACH = 2.0                  # deploy at Mach 2
PARACHUTE_DEPLOY_VELOCITY_M_S = PARACHUTE_DEPLOY_MACH * MARS_SPEED_OF_SOUND_M_S
PARACHUTE_DEPLOY_ALTITUDE_MIN_M = 5_000.0    # minimum altitude for deployment (m)
PARACHUTE_CD = 0.50                          # drag coefficient (disk-gap-band)
PARACHUTE_MASS_FRACTION = 0.02               # 2% of entry mass
DEFAULT_PARACHUTE_AREA_M2 = 200.0            # parachute reference area (m²)

# -- Powered descent constants ------------------------------------------------

POWERED_DESCENT_TRIGGER_M_S = 90.0           # switch to rockets below this velocity (m/s)
POWERED_DESCENT_ALTITUDE_MIN_M = 1_500.0     # minimum altitude for powered descent start
DEFAULT_ENGINE_ISP_S = 226.0                 # specific impulse (hydrazine mono, s)
DEFAULT_MAX_THRUST_N = 31_000.0              # max thrust (N), ~8 MLE engines
DEFAULT_MIN_THROTTLE = 0.20                  # minimum throttle fraction
G0_M_S2 = 9.80665                            # standard gravity for Isp (m/s²)
PROPELLANT_MASS_FRACTION = 0.15              # 15% of entry mass for fuel

# -- Landing constants --------------------------------------------------------

LANDING_VELOCITY_MAX_M_S = 2.5               # maximum safe touchdown velocity (m/s)
LANDING_ALTITUDE_M = 0.0                     # surface altitude (m)

# -- Abort conditions ---------------------------------------------------------

DUST_TAU_ABORT_LIMIT = 2.0                   # optical depth abort threshold
WIND_ABORT_LIMIT_M_S = 25.0                  # surface wind abort limit (m/s)
FUEL_MARGIN_ABORT_FRACTION = 0.05            # abort if fuel below 5% remaining

# -- Time constants -----------------------------------------------------------

SECONDS_PER_SOL = 88_775.0                   # Mars sol in seconds
DEFAULT_DT_S = 1.0                           # default simulation timestep (s)


# -- Phase enum ---------------------------------------------------------------

PHASE_PREENTRY = "pre_entry"
PHASE_ENTRY = "entry"
PHASE_PARACHUTE = "parachute"
PHASE_POWERED = "powered_descent"
PHASE_LANDED = "landed"
PHASE_ABORTED = "aborted"
PHASE_CRASHED = "crashed"


# -- Pure physics helper functions --------------------------------------------

def atmosphere_density(altitude_m: float) -> float:
    """Mars atmospheric density at a given altitude.

    Uses exponential atmosphere model:
        ρ(h) = ρ₀ × exp(−h / H)
    where ρ₀ = 0.020 kg/m³ at surface, H = 11,100 m scale height.

    Returns 0 for altitudes above 200 km (effectively vacuum).
    """
    if altitude_m < 0.0:
        return MARS_ATM_DENSITY_SURFACE
    if altitude_m > 200_000.0:
        return 0.0
    return MARS_ATM_DENSITY_SURFACE * math.exp(-altitude_m / MARS_SCALE_HEIGHT_M)


def drag_force(density: float, velocity: float, cd: float, area: float) -> float:
    """Aerodynamic drag force.

    F_drag = 0.5 × ρ × v² × Cd × A
    Always non-negative (opposes motion).
    """
    return 0.5 * density * velocity * velocity * cd * area


def gravity_at_altitude(altitude_m: float) -> float:
    """Mars gravitational acceleration at altitude.

    g(h) = g₀ × (R / (R + h))²
    Accounts for decreasing gravity with altitude.
    """
    r = MARS_RADIUS_M + max(0.0, altitude_m)
    return MARS_SURFACE_GRAVITY_M_S2 * (MARS_RADIUS_M / r) ** 2


def mach_number(velocity_m_s: float) -> float:
    """Mach number on Mars (speed of sound ~240 m/s in CO₂)."""
    if MARS_SPEED_OF_SOUND_M_S <= 0.0:
        return 0.0
    return abs(velocity_m_s) / MARS_SPEED_OF_SOUND_M_S


def convective_heat_flux(density: float, velocity: float, nose_radius: float) -> float:
    """Convective heating rate on heat shield.

    q̇ = k_h × √(ρ / r_nose) × v³  (Sutton-Graves approximation)
    Returns heat flux in W/m².
    """
    if density <= 0.0 or nose_radius <= 0.0 or velocity <= 0.0:
        return 0.0
    # Sutton-Graves: q_dot = k * sqrt(rho / r_n) * v^3
    return CONVECTIVE_HEATING_COEFF * math.sqrt(density / nose_radius) * velocity ** 3


def heat_shield_temp(heat_flux_w_m2: float, shield_mass_kg: float,
                     current_temp_k: float, area_m2: float, dt_s: float) -> float:
    """Update heat shield temperature from absorbed heat flux.

    ΔT = (q̇ × A × dt) / (m × Cp)
    Temperature rises from absorbed flux, with Stefan-Boltzmann
    radiative cooling.
    """
    if shield_mass_kg <= 0.0 or area_m2 <= 0.0:
        return current_temp_k
    # Energy absorbed this step
    energy_j = heat_flux_w_m2 * area_m2 * dt_s
    # Temperature rise
    delta_t = energy_j / (shield_mass_kg * HEATSHIELD_SPECIFIC_HEAT_J_KGK)
    # Radiative cooling (Stefan-Boltzmann, simplified)
    # P_rad = ε × σ × A × T⁴, with ε ≈ 0.9
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    emissivity = 0.9
    radiated_power = emissivity * sigma * area_m2 * current_temp_k ** 4
    cooling_delta = (radiated_power * dt_s) / (shield_mass_kg * HEATSHIELD_SPECIFIC_HEAT_J_KGK)
    new_temp = current_temp_k + delta_t - cooling_delta
    return max(MARS_SURFACE_TEMP_K, min(new_temp, HEATSHIELD_MAX_TEMP_K))


def ablation_mass_loss(heat_flux_w_m2: float, area_m2: float, dt_s: float,
                       shield_temp_k: float) -> float:
    """Mass of heat shield ablated this timestep.

    Ablation occurs when temperature exceeds threshold.
    dm = (q̇ × A × dt × f_temp) / Q_ablation
    where f_temp is the fraction above threshold.
    """
    if shield_temp_k < HEATSHIELD_ABLATION_THRESHOLD_K:
        return 0.0
    if heat_flux_w_m2 <= 0.0 or area_m2 <= 0.0:
        return 0.0
    # Fraction above threshold drives ablation rate
    temp_ratio = (shield_temp_k - HEATSHIELD_ABLATION_THRESHOLD_K) / \
                 (HEATSHIELD_MAX_TEMP_K - HEATSHIELD_ABLATION_THRESHOLD_K)
    temp_ratio = max(0.0, min(1.0, temp_ratio))
    energy_absorbed = heat_flux_w_m2 * area_m2 * dt_s * temp_ratio
    return max(0.0, energy_absorbed / HEATSHIELD_ABLATION_ENERGY_J_KG)


def parachute_drag_force(density: float, velocity: float,
                         cd: float, area: float) -> float:
    """Drag force from deployed parachute.

    Same physics as aerodynamic drag: F = 0.5 × ρ × v² × Cd × A
    """
    return drag_force(density, velocity, cd, area)


def ballistic_coefficient(mass_kg: float, cd: float, area_m2: float) -> float:
    """Ballistic coefficient β = m / (Cd × A).

    Higher β means less deceleration per unit of atmospheric drag.
    Units: kg/m².
    """
    denom = cd * area_m2
    if denom <= 0.0:
        return float("inf")
    return mass_kg / denom


def tsiolkovsky_delta_v(isp_s: float, mass_initial: float,
                        mass_final: float) -> float:
    """Tsiolkovsky rocket equation: Δv = Isp × g₀ × ln(m_i / m_f).

    Returns achievable Δv in m/s for given propellant fraction.
    """
    if mass_final <= 0.0 or mass_initial <= mass_final:
        return 0.0
    return isp_s * G0_M_S2 * math.log(mass_initial / mass_final)


def required_propellant(delta_v: float, isp_s: float,
                        dry_mass: float) -> float:
    """Propellant mass needed for a given Δv.

    From Tsiolkovsky: m_prop = m_dry × (exp(Δv / (Isp × g₀)) − 1)
    """
    if isp_s <= 0.0 or delta_v <= 0.0:
        return 0.0
    mass_ratio = math.exp(delta_v / (isp_s * G0_M_S2))
    return dry_mass * (mass_ratio - 1.0)


def thrust_acceleration(thrust_n: float, mass_kg: float) -> float:
    """Acceleration from rocket thrust: a = F / m."""
    if mass_kg <= 0.0:
        return 0.0
    return thrust_n / mass_kg


def fuel_mass_flow_rate(thrust_n: float, isp_s: float) -> float:
    """Propellant mass flow rate: ṁ = F / (Isp × g₀).

    Returns kg/s consumed at given thrust level.
    """
    if isp_s <= 0.0:
        return 0.0
    return thrust_n / (isp_s * G0_M_S2)


def kinetic_energy(mass_kg: float, velocity_m_s: float) -> float:
    """Kinetic energy: KE = 0.5 × m × v²."""
    return 0.5 * mass_kg * velocity_m_s * velocity_m_s


def potential_energy(mass_kg: float, altitude_m: float) -> float:
    """Gravitational potential energy: PE = m × g × h.

    Uses surface gravity for simplicity (valid for h << R_mars).
    """
    return mass_kg * MARS_SURFACE_GRAVITY_M_S2 * max(0.0, altitude_m)


def dynamic_pressure(density: float, velocity: float) -> float:
    """Dynamic pressure: q = 0.5 × ρ × v² (Pa)."""
    return 0.5 * density * velocity * velocity


# -- State dataclass ----------------------------------------------------------

@dataclass
class LanderState:
    """State of a Mars cargo EDL capsule.

    Tracks position, velocity, mass breakdown, thermal state, and
    phase through the entry-descent-landing sequence.
    """

    # -- Configuration (set at creation, constant during flight) ---------------
    payload_mass_kg: float = 5_000.0          # cargo payload mass
    entry_mass_kg: float = 0.0                # total mass at atmospheric entry (computed)
    capsule_cd: float = DEFAULT_ENTRY_CD      # hypersonic drag coefficient
    capsule_area_m2: float = DEFAULT_CAPSULE_AREA_M2  # cross-section area
    nose_radius_m: float = DEFAULT_NOSE_RADIUS_M      # nose radius for heating calc
    engine_isp_s: float = DEFAULT_ENGINE_ISP_S         # engine specific impulse
    max_thrust_n: float = DEFAULT_MAX_THRUST_N         # maximum engine thrust
    min_throttle: float = DEFAULT_MIN_THROTTLE         # minimum throttle fraction
    parachute_area_m2: float = DEFAULT_PARACHUTE_AREA_M2  # chute area
    parachute_cd: float = PARACHUTE_CD                 # chute drag coefficient
    structural_g_limit: float = MAX_DECELERATION_G     # structural deceleration limit

    # -- Mass breakdown (kg) --------------------------------------------------
    heatshield_mass_kg: float = 0.0           # ablative heat shield (computed)
    parachute_mass_kg: float = 0.0            # parachute system mass (computed)
    propellant_mass_kg: float = 0.0           # loaded propellant (computed)
    propellant_remaining_kg: float = 0.0      # current propellant level
    structure_mass_kg: float = 0.0            # structural mass (computed)
    heatshield_remaining_kg: float = 0.0      # remaining heat shield mass

    # -- Kinematic state ------------------------------------------------------
    altitude_m: float = DEFAULT_ENTRY_ALTITUDE_M   # current altitude above surface
    velocity_m_s: float = DEFAULT_ENTRY_VELOCITY_M_S  # current speed (scalar, positive = moving)
    flight_path_angle_deg: float = DEFAULT_ENTRY_ANGLE_DEG  # angle below horizontal (negative = descending)

    # -- Thermal state --------------------------------------------------------
    heatshield_temp_k: float = MARS_SURFACE_TEMP_K  # current heat shield temperature
    peak_heating_w_m2: float = 0.0            # peak heat flux experienced
    peak_deceleration_g: float = 0.0          # peak deceleration experienced
    total_heat_dissipated_j: float = 0.0      # cumulative heat energy dissipated

    # -- Phase tracking -------------------------------------------------------
    phase: str = PHASE_PREENTRY               # current EDL phase
    parachute_deployed: bool = False           # whether chute has been deployed
    engines_active: bool = False               # whether retrorockets are firing
    throttle: float = 0.0                     # current throttle setting [0, 1]

    # -- Simulation state -----------------------------------------------------
    time_s: float = 0.0                       # elapsed time since entry (s)
    events: List[str] = field(default_factory=list)

    # -- Environment (updated per tick) ---------------------------------------
    dust_tau: float = 0.3                     # atmospheric optical depth
    surface_wind_m_s: float = 5.0             # surface wind speed (m/s)

    def current_mass_kg(self) -> float:
        """Total current mass of the capsule."""
        return (self.payload_mass_kg + self.heatshield_remaining_kg +
                self.parachute_mass_kg + self.propellant_remaining_kg +
                self.structure_mass_kg)

    def mass_check(self) -> float:
        """Difference between current mass sum and entry mass minus ablated/burned."""
        ablated = self.heatshield_mass_kg - self.heatshield_remaining_kg
        burned = self.propellant_mass_kg - self.propellant_remaining_kg
        expected = self.entry_mass_kg - ablated - burned
        return abs(self.current_mass_kg() - expected)

    def fuel_fraction(self) -> float:
        """Fraction of propellant remaining."""
        if self.propellant_mass_kg <= 0.0:
            return 0.0
        return self.propellant_remaining_kg / self.propellant_mass_kg

    def is_terminal(self) -> bool:
        """Whether the lander has reached a terminal state."""
        return self.phase in (PHASE_LANDED, PHASE_ABORTED, PHASE_CRASHED)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to a JSON-safe dict."""
        return {
            "payload_mass_kg": round(self.payload_mass_kg, 2),
            "entry_mass_kg": round(self.entry_mass_kg, 2),
            "altitude_m": round(self.altitude_m, 2),
            "velocity_m_s": round(self.velocity_m_s, 4),
            "phase": self.phase,
            "time_s": round(self.time_s, 2),
            "heatshield_temp_k": round(self.heatshield_temp_k, 1),
            "heatshield_remaining_kg": round(self.heatshield_remaining_kg, 3),
            "propellant_remaining_kg": round(self.propellant_remaining_kg, 3),
            "peak_heating_w_m2": round(self.peak_heating_w_m2, 1),
            "peak_deceleration_g": round(self.peak_deceleration_g, 3),
            "total_heat_dissipated_j": round(self.total_heat_dissipated_j, 1),
            "current_mass_kg": round(self.current_mass_kg(), 2),
            "mach": round(mach_number(self.velocity_m_s), 2),
            "fuel_fraction": round(self.fuel_fraction(), 4),
            "parachute_deployed": self.parachute_deployed,
            "engines_active": self.engines_active,
            "throttle": round(self.throttle, 3),
            "events": list(self.events),
        }


# -- Factory function ---------------------------------------------------------

def create_lander(
    payload_mass_kg: float = 5_000.0,
    entry_velocity_m_s: float = DEFAULT_ENTRY_VELOCITY_M_S,
    entry_altitude_m: float = DEFAULT_ENTRY_ALTITUDE_M,
    entry_angle_deg: float = DEFAULT_ENTRY_ANGLE_DEG,
    capsule_cd: float = DEFAULT_ENTRY_CD,
    capsule_area_m2: float = DEFAULT_CAPSULE_AREA_M2,
    nose_radius_m: float = DEFAULT_NOSE_RADIUS_M,
    engine_isp_s: float = DEFAULT_ENGINE_ISP_S,
    max_thrust_n: float = DEFAULT_MAX_THRUST_N,
    parachute_area_m2: float = DEFAULT_PARACHUTE_AREA_M2,
    parachute_cd: float = PARACHUTE_CD,
    heatshield_fraction: float = HEATSHIELD_MASS_FRACTION,
    parachute_fraction: float = PARACHUTE_MASS_FRACTION,
    propellant_fraction: float = PROPELLANT_MASS_FRACTION,
) -> LanderState:
    """Create a cargo lander ready for atmospheric entry.

    Computes mass breakdown from payload and mass fractions:
        entry_mass = payload / (1 − shield_frac − chute_frac − prop_frac − struct)
    Then allocates shield, chute, propellant, and structure budgets.
    """
    # Mass fractions must leave room for payload
    structure_fraction = 0.10  # 10% for structure, mechanisms, avionics
    total_fraction = heatshield_fraction + parachute_fraction + propellant_fraction + structure_fraction
    if total_fraction >= 1.0:
        raise ValueError(f"Mass fractions sum to {total_fraction:.2f} >= 1.0, no room for payload")

    payload_fraction = 1.0 - total_fraction
    if payload_fraction <= 0.0:
        entry_mass = 0.0
    else:
        entry_mass = payload_mass_kg / payload_fraction

    heatshield = entry_mass * heatshield_fraction
    parachute_mass = entry_mass * parachute_fraction
    propellant = entry_mass * propellant_fraction
    structure = entry_mass * structure_fraction

    state = LanderState(
        payload_mass_kg=payload_mass_kg,
        entry_mass_kg=entry_mass,
        capsule_cd=capsule_cd,
        capsule_area_m2=capsule_area_m2,
        nose_radius_m=nose_radius_m,
        engine_isp_s=engine_isp_s,
        max_thrust_n=max_thrust_n,
        parachute_area_m2=parachute_area_m2,
        parachute_cd=parachute_cd,
        heatshield_mass_kg=heatshield,
        parachute_mass_kg=parachute_mass,
        propellant_mass_kg=propellant,
        propellant_remaining_kg=propellant,
        structure_mass_kg=structure,
        heatshield_remaining_kg=heatshield,
        altitude_m=entry_altitude_m,
        velocity_m_s=entry_velocity_m_s,
        flight_path_angle_deg=entry_angle_deg,
        heatshield_temp_k=MARS_SURFACE_TEMP_K,
        phase=PHASE_PREENTRY,
    )
    return state


# -- Tick engine --------------------------------------------------------------

def tick(
    state: LanderState,
    dt_s: float = DEFAULT_DT_S,
    dust_tau: float | None = None,
    surface_wind_m_s: float | None = None,
) -> Dict[str, Any]:
    """Advance the lander state by one timestep.

    Parameters
    ----------
    state : LanderState
        Mutable lander state, modified in place.
    dt_s : float
        Timestep duration in seconds.
    dust_tau : float or None
        Atmospheric optical depth (None = use current state value).
    surface_wind_m_s : float or None
        Surface wind speed (None = use current state value).

    Returns
    -------
    dict with tick results including phase, altitude, velocity,
    forces, thermal state, and events.
    """
    state.events = []
    state.time_s += dt_s

    # Update environment if provided
    if dust_tau is not None:
        state.dust_tau = dust_tau
    if surface_wind_m_s is not None:
        state.surface_wind_m_s = surface_wind_m_s

    # Terminal states: no further physics
    if state.is_terminal():
        return _make_result(state, 0.0, 0.0, 0.0, 0.0)

    # -- Phase transitions ----------------------------------------------------

    # Pre-entry → Entry: begin atmospheric entry
    if state.phase == PHASE_PREENTRY:
        state.phase = PHASE_ENTRY
        state.events.append("ENTRY_INTERFACE")

    # -- Compute atmospheric properties ---------------------------------------
    rho = atmosphere_density(state.altitude_m)
    g_local = gravity_at_altitude(state.altitude_m)
    current_mass = state.current_mass_kg()

    # -- Force accumulation ---------------------------------------------------
    drag_n = 0.0
    thrust_n = 0.0
    chute_drag_n = 0.0
    heat_flux = 0.0

    # Flight path angle for vertical/horizontal velocity decomposition
    fpa_rad = math.radians(state.flight_path_angle_deg)
    sin_fpa = math.sin(abs(fpa_rad))
    cos_fpa = math.cos(fpa_rad)
    v_horizontal = state.velocity_m_s * cos_fpa

    if state.phase == PHASE_ENTRY:
        # Aerodynamic drag on capsule body
        drag_n = drag_force(rho, state.velocity_m_s, state.capsule_cd, state.capsule_area_m2)

        # Heat flux on heat shield
        heat_flux = convective_heat_flux(rho, state.velocity_m_s, state.nose_radius_m)
        state.peak_heating_w_m2 = max(state.peak_heating_w_m2, heat_flux)

        # Heat shield thermal response
        state.heatshield_temp_k = heat_shield_temp(
            heat_flux, state.heatshield_remaining_kg,
            state.heatshield_temp_k, state.capsule_area_m2, dt_s,
        )

        # Heat shield ablation
        ablated = ablation_mass_loss(
            heat_flux, state.capsule_area_m2, dt_s, state.heatshield_temp_k,
        )
        ablated = min(ablated, state.heatshield_remaining_kg)
        state.heatshield_remaining_kg -= ablated

        # Track total heat dissipated (drag work = F × v × dt)
        heat_dissipated = drag_n * state.velocity_m_s * dt_s
        state.total_heat_dissipated_j += heat_dissipated

        # Check for heat shield depletion
        if state.heatshield_remaining_kg <= 0.0 and heat_flux > 0.0:
            state.events.append("HEATSHIELD_DEPLETED")

        # Transition to parachute phase?
        mach = mach_number(state.velocity_m_s)
        if (mach <= PARACHUTE_DEPLOY_MACH and
                state.altitude_m > PARACHUTE_DEPLOY_ALTITUDE_MIN_M and
                not state.parachute_deployed):
            state.parachute_deployed = True
            state.phase = PHASE_PARACHUTE
            state.events.append("PARACHUTE_DEPLOYED")

    elif state.phase == PHASE_PARACHUTE:
        # Capsule body drag
        drag_n = drag_force(rho, state.velocity_m_s, state.capsule_cd, state.capsule_area_m2)

        # Parachute drag
        chute_drag_n = parachute_drag_force(
            rho, state.velocity_m_s, state.parachute_cd, state.parachute_area_m2,
        )

        # Track heat dissipated from drag
        total_drag_n = drag_n + chute_drag_n
        state.total_heat_dissipated_j += total_drag_n * state.velocity_m_s * dt_s

        # Heat shield cools toward ambient during parachute phase
        if state.heatshield_temp_k > MARS_SURFACE_TEMP_K:
            cooling_rate = 50.0  # K/s approx cooling in Mars atmosphere
            state.heatshield_temp_k = max(
                MARS_SURFACE_TEMP_K,
                state.heatshield_temp_k - cooling_rate * dt_s,
            )

        # Transition to powered descent?
        if (state.velocity_m_s <= POWERED_DESCENT_TRIGGER_M_S or
                state.altitude_m <= POWERED_DESCENT_ALTITUDE_MIN_M):
            state.phase = PHASE_POWERED
            state.engines_active = True
            state.events.append("POWERED_DESCENT_START")

    elif state.phase == PHASE_POWERED:
        # Capsule drag (still in atmosphere)
        drag_n = drag_force(rho, state.velocity_m_s, state.capsule_cd, state.capsule_area_m2)

        # Compute required thrust for gravity-turn descent
        # a_required = v² / (2h) to stop in remaining distance h
        if state.altitude_m > 0.0 and state.velocity_m_s > 0.0:
            a_needed = (state.velocity_m_s ** 2) / (2.0 * max(1.0, state.altitude_m))
            # Add gravity compensation to required deceleration
            total_a_needed = a_needed + g_local
            thrust_needed = total_a_needed * current_mass - drag_n
            thrust_needed = max(0.0, thrust_needed)
        else:
            # At surface or zero velocity — fight gravity
            thrust_needed = g_local * current_mass

        # Throttle to needed thrust within engine limits
        max_available = state.max_thrust_n
        min_available = state.max_thrust_n * state.min_throttle
        thrust_n = max(min_available, min(max_available, thrust_needed))
        state.throttle = thrust_n / state.max_thrust_n if state.max_thrust_n > 0 else 0.0

        # Propellant consumption: ṁ = F / (Isp × g₀)
        mdot = fuel_mass_flow_rate(thrust_n, state.engine_isp_s)
        fuel_used = mdot * dt_s
        fuel_used = min(fuel_used, state.propellant_remaining_kg)
        state.propellant_remaining_kg -= fuel_used

        # Track heat dissipated from aero drag
        state.total_heat_dissipated_j += drag_n * state.velocity_m_s * dt_s

        # Fuel exhaustion check
        if state.propellant_remaining_kg <= 0.0:
            state.propellant_remaining_kg = 0.0
            thrust_n = 0.0
            state.engines_active = False
            state.throttle = 0.0
            state.events.append("FUEL_EXHAUSTED")

        # Low fuel warning
        if (state.fuel_fraction() < FUEL_MARGIN_ABORT_FRACTION and
                state.propellant_remaining_kg > 0.0):
            state.events.append("LOW_FUEL_WARNING")

    # -- Equations of motion --------------------------------------------------
    total_drag = drag_n + chute_drag_n
    if current_mass > 0.0:
        # Deceleration from drag (opposes velocity)
        drag_decel = total_drag / current_mass
        # Deceleration from thrust (opposes velocity during descent)
        thrust_decel = thrust_n / current_mass
        # Net deceleration = aero + thrust − gravity component along flight path
        net_decel = drag_decel + thrust_decel - g_local * sin_fpa

        # Track peak structural loads (aero + thrust, excluding gravity)
        structural_decel_g = (drag_decel + thrust_decel) / G0_M_S2
        state.peak_deceleration_g = max(state.peak_deceleration_g, structural_decel_g)

        # Structural limit check
        if structural_decel_g > state.structural_g_limit:
            state.events.append("STRUCTURAL_LIMIT_EXCEEDED")

        # Update velocity (decrease toward zero)
        new_velocity = state.velocity_m_s - net_decel * dt_s
        state.velocity_m_s = max(0.0, new_velocity)
    else:
        drag_decel = 0.0
        thrust_decel = 0.0

    # Update altitude (descending)
    descent_rate = state.velocity_m_s * sin_fpa
    state.altitude_m -= descent_rate * dt_s
    state.altitude_m = max(0.0, state.altitude_m)

    # Flight path angle steepens as horizontal velocity is shed
    if state.velocity_m_s > 0.0 and v_horizontal > 1.0:
        h_drag_decel = total_drag / current_mass * cos_fpa if current_mass > 0 else 0.0
        new_v_h = max(0.0, v_horizontal - h_drag_decel * dt_s)
        new_v_v = state.velocity_m_s * sin_fpa
        if new_v_h > 0.01:
            state.flight_path_angle_deg = -math.degrees(math.atan2(new_v_v, new_v_h))
        else:
            state.flight_path_angle_deg = -90.0  # vertical descent

    # -- Abort conditions -----------------------------------------------------
    if state.dust_tau > DUST_TAU_ABORT_LIMIT and state.phase in (PHASE_ENTRY, PHASE_PARACHUTE):
        state.events.append("ABORT_DUST_STORM")

    if (state.surface_wind_m_s > WIND_ABORT_LIMIT_M_S and
            state.altitude_m < 2_000.0):
        state.events.append("ABORT_HIGH_WIND")

    # -- Landing detection ----------------------------------------------------
    if state.altitude_m <= 0.0:
        state.altitude_m = 0.0
        if state.velocity_m_s <= LANDING_VELOCITY_MAX_M_S:
            state.phase = PHASE_LANDED
            state.velocity_m_s = 0.0
            state.engines_active = False
            state.throttle = 0.0
            state.events.append("TOUCHDOWN")
        else:
            state.phase = PHASE_CRASHED
            state.events.append(f"CRASH_AT_{state.velocity_m_s:.1f}_M_S")

    return _make_result(state, drag_n, chute_drag_n, thrust_n, heat_flux)


def _make_result(
    state: LanderState,
    drag_n: float,
    chute_drag_n: float,
    thrust_n: float,
    heat_flux_w_m2: float,
) -> Dict[str, Any]:
    """Build standardized result dict for one tick."""
    return {
        "time_s": round(state.time_s, 3),
        "phase": state.phase,
        "altitude_m": round(state.altitude_m, 2),
        "velocity_m_s": round(state.velocity_m_s, 4),
        "mach": round(mach_number(state.velocity_m_s), 3),
        "current_mass_kg": round(state.current_mass_kg(), 3),
        "drag_n": round(drag_n, 2),
        "chute_drag_n": round(chute_drag_n, 2),
        "thrust_n": round(thrust_n, 2),
        "heat_flux_w_m2": round(heat_flux_w_m2, 2),
        "heatshield_temp_k": round(state.heatshield_temp_k, 1),
        "heatshield_remaining_kg": round(state.heatshield_remaining_kg, 3),
        "propellant_remaining_kg": round(state.propellant_remaining_kg, 3),
        "fuel_fraction": round(state.fuel_fraction(), 4),
        "peak_deceleration_g": round(state.peak_deceleration_g, 3),
        "total_heat_dissipated_j": round(state.total_heat_dissipated_j, 1),
        "dynamic_pressure_pa": round(
            dynamic_pressure(atmosphere_density(state.altitude_m), state.velocity_m_s), 2,
        ),
        "throttle": round(state.throttle, 3),
        "parachute_deployed": state.parachute_deployed,
        "engines_active": state.engines_active,
        "events": list(state.events),
    }


# -- Simulation runner --------------------------------------------------------

def run_edl(
    state: LanderState,
    max_time_s: float = 600.0,
    dt_s: float = DEFAULT_DT_S,
) -> List[Dict[str, Any]]:
    """Run a full EDL simulation until landing, crash, or timeout.

    Returns list of tick results for each timestep.
    """
    results: List[Dict[str, Any]] = []
    while state.time_s < max_time_s and not state.is_terminal():
        result = tick(state, dt_s=dt_s)
        results.append(result)
    return results
