"""magnetic_shield.py -- Mars Habitat Active Magnetic Shield.

The colony has passive shielding (rad_shield.py) and detectors
(radiation_monitor.py), but Mars has no magnetosphere.  Solar wind
strips atmosphere at ~100 g/s.  GCR delivers 0.67 mSv/sol to the
surface.  During solar particle events (SPE), doses spike 100-1000x.

Passive shielding (regolith, water, polyethylene) works for GCR but
adds massive structural weight.  This module is the colony's ACTIVE
magnetic defense: a superconducting solenoid that wraps the habitat
in a magnetic dipole field, deflecting charged particles by the
Lorentz force before they reach the hull.

Physics modelled
----------------
* Solenoid field: B = mu_0 * N * I / L (interior axial field).
* Magnetic dipole moment: m = N * I * A, where A = pi * R^2.
* Larmor radius: r_L = (m_p * v_perp) / (|q| * B).
  Particle deflected if r_L < R_shield (shield radius).
* Energy cutoff: E_cut = sqrt((q*B*R*c)^2 + (mc^2)^2) - mc^2.
  Particles with kinetic energy below E_cut are deflected.
* GCR spectrum: power-law, most particles <1 GeV/nucleon.
  A 0.5 T field with 5 m radius deflects protons up to ~260 MeV.
* SPE spectrum: softer than GCR.  0.5 T already deflects >90%.
* Stored energy: U = B^2 * V_coil / (2 * mu_0).
* Superconductor: YBCO (YBa2Cu3O7).  T_c = 93 K.  Cooled by
  LN2 at 77 K.  Mars ambient is 210 K -- cryocooler needed.
* Cryocooler power: Q_cryo = heat_leak / COP.
  Heat leak: conduction + radiation from 210 K environment.
* Quench: if coil temp exceeds T_c, resistance spikes,
  stored energy dumps into coil as heat.
* Ramp up/down: field changes at dB/dt limited by induced voltage.

Conservation laws
-----------------
- Energy: stored magnetic energy + thermal energy = constant (no quench)
- Coolant mass: boiloff = heat_leak / latent_heat_LN2
- Deflection fraction in [0, 1], monotonically increases with B
- Coil temperature >= coolant_temp (never below)
- Power >= 0, coolant >= 0

One tick = one sol.  Field in T, temperature in K, energy in MJ,
power in kW, coolant in kg, dose in mSv.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# -- Physical constants -------------------------------------------------------

MU_0 = 4.0e-7 * math.pi
PROTON_MASS_KG = 1.6726e-27
PROTON_CHARGE_C = 1.6022e-19
SPEED_OF_LIGHT = 2.998e8
PROTON_REST_ENERGY_GEV = 0.938272

MARS_AMBIENT_TEMP_K = 210.0
SECONDS_PER_SOL = 88_775.0
LN2_BOILING_POINT_K = 77.36
LN2_LATENT_HEAT_KJ_KG = 199.0

# -- YBCO superconductor properties -------------------------------------------

YBCO_TC_K = 93.0
YBCO_JC_A_M2 = 3.0e9
YBCO_NORMAL_RESISTIVITY = 1.0e-3
YBCO_SPECIFIC_HEAT_J_KG_K = 200.0

# -- Default coil geometry (habitat-class) ------------------------------------

DEFAULT_COIL_RADIUS_M = 5.0
DEFAULT_COIL_LENGTH_M = 2.5
DEFAULT_NUM_TURNS = 2000
DEFAULT_OPERATING_CURRENT_A = 500.0
DEFAULT_WIRE_CROSS_SECTION_M2 = 1.0e-4
DEFAULT_WIRE_DENSITY_KG_M3 = 6_300.0

DEFAULT_COOLANT_KG = 2000.0
DEFAULT_CRYOCOOLER_COP = 0.20

# -- Thermal model ------------------------------------------------------------

CRYO_VESSEL_AREA_M2 = 80.0
MLI_CONDUCTANCE_W_M2_K = 0.005
RADIATION_EMISSIVITY = 0.03
STEFAN_BOLTZMANN = 5.67e-8
JOINT_RESISTANCE_OHM = 1.0e-9
NUM_JOINTS = 20

# -- Radiation environment ----------------------------------------------------

GCR_SURFACE_MSV_SOL = 0.67
SPE_MILD_MSV = 50.0
SPE_SEVERE_MSV = 500.0
SOLAR_WIND_SPEED_M_S = 400_000.0

# -- Field ramp limits --------------------------------------------------------

MAX_DB_DT_T_PER_SOL = 0.05
MAX_OPERATING_FIELD_T = 2.0


# -- Pure physics functions ---------------------------------------------------

def solenoid_field_t(num_turns: int, current_a: float,
                     length_m: float) -> float:
    if length_m <= 0.0 or num_turns <= 0:
        return 0.0
    return MU_0 * num_turns * current_a / length_m


def magnetic_dipole_moment(num_turns: int, current_a: float,
                           radius_m: float) -> float:
    return num_turns * current_a * math.pi * radius_m ** 2


def stored_energy_mj(field_t: float, coil_volume_m3: float) -> float:
    if field_t <= 0.0 or coil_volume_m3 <= 0.0:
        return 0.0
    energy_j = (field_t ** 2) * coil_volume_m3 / (2.0 * MU_0)
    return energy_j / 1.0e6


def coil_volume_m3(radius_m: float, length_m: float) -> float:
    if radius_m <= 0.0 or length_m <= 0.0:
        return 0.0
    return math.pi * radius_m ** 2 * length_m


def larmor_radius_m(particle_mass_kg: float, velocity_m_s: float,
                    charge_c: float, field_t: float) -> float:
    if field_t <= 0.0 or charge_c == 0.0:
        return float("inf")
    return abs(particle_mass_kg * velocity_m_s / (charge_c * field_t))


def proton_larmor_radius_m(energy_gev: float, field_t: float) -> float:
    if field_t <= 0.0 or energy_gev <= 0.0:
        return float("inf")
    total_e_gev = energy_gev + PROTON_REST_ENERGY_GEV
    p_gev_c = math.sqrt(total_e_gev ** 2 - PROTON_REST_ENERGY_GEV ** 2)
    p_kg_m_s = p_gev_c * 1.0e9 * PROTON_CHARGE_C / SPEED_OF_LIGHT
    return p_kg_m_s / (PROTON_CHARGE_C * field_t)


def energy_cutoff_gev(field_t: float, shield_radius_m: float) -> float:
    if field_t <= 0.0 or shield_radius_m <= 0.0:
        return 0.0
    p_c_gev = (PROTON_CHARGE_C * field_t * shield_radius_m *
               SPEED_OF_LIGHT) / (1.0e9 * PROTON_CHARGE_C)
    total_e = math.sqrt(p_c_gev ** 2 + PROTON_REST_ENERGY_GEV ** 2)
    e_kinetic = total_e - PROTON_REST_ENERGY_GEV
    return max(0.0, e_kinetic)


def gcr_deflection_fraction(energy_cutoff_gev_val: float) -> float:
    if energy_cutoff_gev_val <= 0.0:
        return 0.0
    if energy_cutoff_gev_val >= 10.0:
        return 0.99
    if energy_cutoff_gev_val <= 0.3:
        return 0.35 * (energy_cutoff_gev_val / 0.3)
    elif energy_cutoff_gev_val <= 1.0:
        return 0.35 + 0.35 * ((energy_cutoff_gev_val - 0.3) / 0.7)
    elif energy_cutoff_gev_val <= 3.0:
        return 0.70 + 0.20 * ((energy_cutoff_gev_val - 1.0) / 2.0)
    else:
        return min(0.99, 0.90 + 0.09 * ((energy_cutoff_gev_val - 3.0) / 7.0))


def spe_deflection_fraction(field_t: float, shield_radius_m: float) -> float:
    if field_t <= 0.0 or shield_radius_m <= 0.0:
        return 0.0
    e_cut = energy_cutoff_gev(field_t, shield_radius_m)
    if e_cut >= 0.5:
        return 0.99
    if e_cut >= 0.1:
        return 0.85 + 0.14 * ((e_cut - 0.1) / 0.4)
    return 0.85 * (e_cut / 0.1)


def heat_leak_kw(vessel_area_m2: float, ambient_temp_k: float,
                 coil_temp_k: float) -> float:
    if ambient_temp_k <= coil_temp_k:
        return 0.0
    conduction = MLI_CONDUCTANCE_W_M2_K * vessel_area_m2 * (ambient_temp_k - coil_temp_k)
    radiation = (RADIATION_EMISSIVITY * STEFAN_BOLTZMANN * vessel_area_m2 *
                 (ambient_temp_k ** 4 - coil_temp_k ** 4))
    return max(0.0, (conduction + radiation) / 1000.0)


def joint_heating_kw(current_a: float, num_joints: int = NUM_JOINTS,
                     resistance_per_joint: float = JOINT_RESISTANCE_OHM) -> float:
    return num_joints * resistance_per_joint * current_a ** 2 / 1000.0


def cryocooler_power_kw(heat_load_kw: float,
                        cop: float = DEFAULT_CRYOCOOLER_COP) -> float:
    if heat_load_kw <= 0.0 or cop <= 0.0:
        return 0.0
    return heat_load_kw / cop


def coolant_boiloff_kg_per_sol(heat_leak_kw_val: float) -> float:
    if heat_leak_kw_val <= 0.0:
        return 0.0
    heat_j_per_sol = heat_leak_kw_val * 1000.0 * SECONDS_PER_SOL
    return heat_j_per_sol / (LN2_LATENT_HEAT_KJ_KG * 1000.0)


def wire_mass_kg(radius_m: float, num_turns: int,
                 cross_section_m2: float = DEFAULT_WIRE_CROSS_SECTION_M2,
                 density_kg_m3: float = DEFAULT_WIRE_DENSITY_KG_M3) -> float:
    wire_length = num_turns * 2.0 * math.pi * radius_m
    return wire_length * cross_section_m2 * density_kg_m3


def quench_temperature_rise_k(stored_energy_mj_val: float,
                              wire_mass_kg_val: float) -> float:
    if wire_mass_kg_val <= 0.0:
        return 0.0
    energy_j = stored_energy_mj_val * 1.0e6
    return energy_j / (wire_mass_kg_val * YBCO_SPECIFIC_HEAT_J_KG_K)


# -- Simulation state ---------------------------------------------------------

@dataclass
class MagneticShieldConfig:
    coil_radius_m: float = DEFAULT_COIL_RADIUS_M
    coil_length_m: float = DEFAULT_COIL_LENGTH_M
    num_turns: int = DEFAULT_NUM_TURNS
    operating_current_a: float = DEFAULT_OPERATING_CURRENT_A
    cryocooler_cop: float = DEFAULT_CRYOCOOLER_COP
    vessel_area_m2: float = CRYO_VESSEL_AREA_M2
    max_field_t: float = MAX_OPERATING_FIELD_T


@dataclass
class ShieldState:
    field_t: float = 0.0
    coil_temp_k: float = LN2_BOILING_POINT_K
    coolant_kg: float = DEFAULT_COOLANT_KG
    current_a: float = 0.0
    stored_energy_mj: float = 0.0
    cryocooler_on: bool = True
    shield_active: bool = False
    quenched: bool = False
    sol: int = 0
    total_gcr_deflected_msv: float = 0.0
    total_spe_deflected_msv: float = 0.0
    total_power_consumed_kwh: float = 0.0
    total_coolant_consumed_kg: float = 0.0
    quench_count: int = 0
    peak_field_t: float = 0.0

    def __post_init__(self) -> None:
        self.field_t = max(0.0, self.field_t)
        self.coil_temp_k = max(LN2_BOILING_POINT_K, self.coil_temp_k)
        self.coolant_kg = max(0.0, self.coolant_kg)
        self.current_a = max(0.0, self.current_a)
        self.stored_energy_mj = max(0.0, self.stored_energy_mj)


@dataclass
class TickResult:
    sol: int = 0
    field_t: float = 0.0
    coil_temp_k: float = 0.0
    coolant_kg: float = 0.0
    heat_leak_kw: float = 0.0
    joint_heat_kw: float = 0.0
    cryocooler_power_kw: float = 0.0
    gcr_deflection: float = 0.0
    spe_deflection: float = 0.0
    energy_cutoff_gev: float = 0.0
    gcr_dose_reduced_msv: float = 0.0
    spe_dose_reduced_msv: float = 0.0
    stored_energy_mj: float = 0.0
    quenched: bool = False
    shield_active: bool = False


# -- Simulation functions -----------------------------------------------------

def create_shield(
    coil_radius_m: float = DEFAULT_COIL_RADIUS_M,
    coil_length_m: float = DEFAULT_COIL_LENGTH_M,
    num_turns: int = DEFAULT_NUM_TURNS,
    operating_current_a: float = DEFAULT_OPERATING_CURRENT_A,
    coolant_kg: float = DEFAULT_COOLANT_KG,
) -> "tuple[MagneticShieldConfig, ShieldState]":
    config = MagneticShieldConfig(
        coil_radius_m=max(0.1, coil_radius_m),
        coil_length_m=max(0.1, coil_length_m),
        num_turns=max(0, num_turns),
        operating_current_a=max(0.0, operating_current_a),
    )
    state = ShieldState(coolant_kg=max(0.0, coolant_kg))
    return config, state


def tick(config: MagneticShieldConfig, state: ShieldState,
         spe_active: bool = False, spe_msv: float = 0.0,
         ambient_temp_k: float = MARS_AMBIENT_TEMP_K) -> TickResult:
    state.sol += 1
    result = TickResult(sol=state.sol)

    # -- Quench recovery
    if state.quenched:
        if state.coolant_kg > 0.0 and state.coil_temp_k > YBCO_TC_K:
            state.coil_temp_k = max(LN2_BOILING_POINT_K, state.coil_temp_k - 5.0)
        if state.coil_temp_k < YBCO_TC_K - 5.0:
            state.quenched = False

    # -- Pre-ramp quench detection
    if (not state.quenched and state.coil_temp_k >= YBCO_TC_K
            and state.field_t > 0.001):
        w_mass = wire_mass_kg(config.coil_radius_m, config.num_turns)
        temp_spike = quench_temperature_rise_k(state.stored_energy_mj, w_mass)
        state.coil_temp_k += temp_spike
        state.field_t = 0.0
        state.current_a = 0.0
        state.stored_energy_mj = 0.0
        state.quenched = True
        state.shield_active = False
        state.quench_count += 1

    # -- Field ramp
    target_field = solenoid_field_t(config.num_turns,
                                   config.operating_current_a,
                                   config.coil_length_m)
    target_field = min(target_field, config.max_field_t)

    if (not state.quenched and state.coolant_kg > 0.0
            and state.cryocooler_on and state.coil_temp_k < YBCO_TC_K):
        delta = target_field - state.field_t
        max_step = MAX_DB_DT_T_PER_SOL
        if abs(delta) > max_step:
            delta = max_step if delta > 0 else -max_step
        state.field_t = max(0.0, state.field_t + delta)
        if target_field > 0.0:
            state.current_a = config.operating_current_a * (state.field_t / target_field)
        else:
            state.current_a = 0.0
    elif state.quenched or state.coil_temp_k >= YBCO_TC_K:
        state.field_t = 0.0
        state.current_a = 0.0
    else:
        state.field_t = max(0.0, state.field_t - MAX_DB_DT_T_PER_SOL * 0.5)
        if state.field_t == 0.0:
            state.current_a = 0.0

    state.shield_active = state.field_t > 0.001
    state.peak_field_t = max(state.peak_field_t, state.field_t)

    # -- Stored energy
    vol = coil_volume_m3(config.coil_radius_m, config.coil_length_m)
    state.stored_energy_mj = stored_energy_mj(state.field_t, vol)

    # -- Thermal management
    h_leak = heat_leak_kw(config.vessel_area_m2, ambient_temp_k, state.coil_temp_k)
    j_heat = joint_heating_kw(state.current_a)
    total_heat_kw = h_leak + j_heat

    if state.cryocooler_on and state.coolant_kg > 0.0 and not state.quenched:
        cryo_power = cryocooler_power_kw(total_heat_kw, config.cryocooler_cop)
        boiloff = coolant_boiloff_kg_per_sol(total_heat_kw * 0.01)
    else:
        cryo_power = 0.0
        boiloff = coolant_boiloff_kg_per_sol(total_heat_kw)
        if total_heat_kw > 0.0:
            w_mass = wire_mass_kg(config.coil_radius_m, config.num_turns)
            if w_mass > 0.0:
                energy_j = total_heat_kw * 1000.0 * SECONDS_PER_SOL
                state.coil_temp_k += energy_j / (w_mass * YBCO_SPECIFIC_HEAT_J_KG_K)

    state.coolant_kg = max(0.0, state.coolant_kg - boiloff)
    state.total_coolant_consumed_kg += boiloff
    state.total_power_consumed_kwh += cryo_power * SECONDS_PER_SOL / 3600.0

    # -- Post-thermal quench detection (backup)
    if (not state.quenched and state.coil_temp_k >= YBCO_TC_K
            and state.current_a > 0.0):
        w_mass = wire_mass_kg(config.coil_radius_m, config.num_turns)
        temp_spike = quench_temperature_rise_k(state.stored_energy_mj, w_mass)
        state.coil_temp_k += temp_spike
        state.field_t = 0.0
        state.current_a = 0.0
        state.stored_energy_mj = 0.0
        state.quenched = True
        state.shield_active = False
        state.quench_count += 1

    # -- Radiation deflection
    e_cut = energy_cutoff_gev(state.field_t, config.coil_radius_m)
    gcr_frac = gcr_deflection_fraction(e_cut)
    spe_frac = spe_deflection_fraction(state.field_t, config.coil_radius_m)

    gcr_reduced = GCR_SURFACE_MSV_SOL * gcr_frac
    spe_reduced = 0.0
    if spe_active:
        spe_reduced = max(0.0, spe_msv) * spe_frac

    state.total_gcr_deflected_msv += gcr_reduced
    state.total_spe_deflected_msv += spe_reduced

    result.field_t = round(state.field_t, 6)
    result.coil_temp_k = round(state.coil_temp_k, 2)
    result.coolant_kg = round(state.coolant_kg, 4)
    result.heat_leak_kw = round(h_leak, 4)
    result.joint_heat_kw = round(j_heat, 6)
    result.cryocooler_power_kw = round(cryo_power, 4)
    result.gcr_deflection = round(gcr_frac, 4)
    result.spe_deflection = round(spe_frac, 4)
    result.energy_cutoff_gev = round(e_cut, 4)
    result.gcr_dose_reduced_msv = round(gcr_reduced, 4)
    result.spe_dose_reduced_msv = round(spe_reduced, 4)
    result.stored_energy_mj = round(state.stored_energy_mj, 4)
    result.quenched = state.quenched
    result.shield_active = state.shield_active
    return result


def run_simulation(
    sols: int = 100,
    config: "MagneticShieldConfig | None" = None,
    state: "ShieldState | None" = None,
    spe_events: "dict[int, float] | None" = None,
) -> "dict[str, object]":
    if config is None or state is None:
        config, state = create_shield()
    if spe_events is None:
        spe_events = {}

    results: list[TickResult] = []
    for sol in range(1, sols + 1):
        spe_active = sol in spe_events
        spe_msv = spe_events.get(sol, 0.0)
        result = tick(config, state, spe_active=spe_active, spe_msv=spe_msv)
        results.append(result)

    final_field = results[-1].field_t if results else 0.0
    ramp_up_sols = 0
    if final_field > 0.001:
        for r in results:
            if r.field_t < final_field * 0.95:
                ramp_up_sols += 1
            else:
                break

    return {
        "sols_simulated": sols,
        "final_field_t": round(state.field_t, 6),
        "peak_field_t": round(state.peak_field_t, 6),
        "final_coil_temp_k": round(state.coil_temp_k, 2),
        "coolant_remaining_kg": round(state.coolant_kg, 2),
        "coolant_consumed_kg": round(state.total_coolant_consumed_kg, 2),
        "total_power_consumed_kwh": round(state.total_power_consumed_kwh, 2),
        "total_gcr_deflected_msv": round(state.total_gcr_deflected_msv, 4),
        "total_spe_deflected_msv": round(state.total_spe_deflected_msv, 4),
        "energy_cutoff_gev": round(energy_cutoff_gev(state.field_t,
                                                     config.coil_radius_m), 4),
        "stored_energy_mj": round(state.stored_energy_mj, 4),
        "ramp_up_sols": ramp_up_sols,
        "quench_count": state.quench_count,
        "shield_active": state.shield_active,
    }
