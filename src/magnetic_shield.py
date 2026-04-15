"""magnetic_shield.py -- Mars Habitat Active Magnetic Shield.

The colony's ACTIVE magnetic defense: a superconducting solenoid that
wraps the habitat in a dipole field, deflecting charged particles via
the Lorentz force.  Complements passive shielding (rad_shield.py).

Physics: solenoid B = mu0*N*I/L, relativistic Larmor radius, energy
cutoff E_cut = sqrt((qBRc)^2 + (mc^2)^2) - mc^2, GCR/SPE spectral
deflection, YBCO superconductor thermals, quench dynamics, cryocooler.
One tick = one sol.  Field in T, temp in K, energy in MJ, dose in mSv.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

MU_0 = 4.0e-7 * math.pi
PROTON_MASS_KG = 1.6726e-27
PROTON_CHARGE_C = 1.6022e-19
SPEED_OF_LIGHT = 2.998e8
PROTON_REST_ENERGY_GEV = 0.938272
MARS_AMBIENT_TEMP_K = 210.0
SECONDS_PER_SOL = 88_775.0
LN2_BOILING_POINT_K = 77.36
LN2_LATENT_HEAT_KJ_KG = 199.0
YBCO_TC_K = 93.0
YBCO_SPECIFIC_HEAT_J_KG_K = 200.0
DEFAULT_COIL_RADIUS_M = 5.0
DEFAULT_COIL_LENGTH_M = 2.5
DEFAULT_NUM_TURNS = 2000
DEFAULT_OPERATING_CURRENT_A = 500.0
DEFAULT_WIRE_CROSS_SECTION_M2 = 1.0e-4
DEFAULT_WIRE_DENSITY_KG_M3 = 6_300.0
DEFAULT_COOLANT_KG = 2000.0
DEFAULT_CRYOCOOLER_COP = 0.20
CRYO_VESSEL_AREA_M2 = 80.0
MLI_CONDUCTANCE_W_M2_K = 0.005
RADIATION_EMISSIVITY = 0.03
STEFAN_BOLTZMANN = 5.67e-8
JOINT_RESISTANCE_OHM = 1.0e-9
NUM_JOINTS = 20
GCR_SURFACE_MSV_SOL = 0.67
MAX_DB_DT_T_PER_SOL = 0.05
MAX_OPERATING_FIELD_T = 2.0


def solenoid_field_t(num_turns: int, current_a: float, length_m: float) -> float:
    """Axial magnetic field inside a solenoid (Tesla)."""
    if length_m <= 0.0 or num_turns <= 0:
        return 0.0
    return MU_0 * num_turns * current_a / length_m


def magnetic_dipole_moment(num_turns: int, current_a: float, radius_m: float) -> float:
    """Magnetic dipole moment (A*m^2)."""
    return num_turns * current_a * math.pi * radius_m ** 2


def stored_energy_mj(field_t: float, coil_volume_m3: float) -> float:
    """Magnetic energy stored in solenoid (MJ)."""
    if field_t <= 0.0 or coil_volume_m3 <= 0.0:
        return 0.0
    return (field_t ** 2) * coil_volume_m3 / (2.0 * MU_0) / 1.0e6


def coil_volume_m3(radius_m: float, length_m: float) -> float:
    """Volume enclosed by solenoid (m^3)."""
    if radius_m <= 0.0 or length_m <= 0.0:
        return 0.0
    return math.pi * radius_m ** 2 * length_m


def larmor_radius_m(particle_mass_kg: float, velocity_m_s: float,
                    charge_c: float, field_t: float) -> float:
    """Gyroradius of a charged particle (m)."""
    if field_t <= 0.0 or charge_c == 0.0:
        return float("inf")
    return abs(particle_mass_kg * velocity_m_s / (charge_c * field_t))


def proton_larmor_radius_m(energy_gev: float, field_t: float) -> float:
    """Relativistic Larmor radius of a proton (m)."""
    if field_t <= 0.0 or energy_gev <= 0.0:
        return float("inf")
    total_e = energy_gev + PROTON_REST_ENERGY_GEV
    p_gev_c = math.sqrt(total_e ** 2 - PROTON_REST_ENERGY_GEV ** 2)
    p_kg = p_gev_c * 1.0e9 * PROTON_CHARGE_C / SPEED_OF_LIGHT
    return p_kg / (PROTON_CHARGE_C * field_t)


def energy_cutoff_gev(field_t: float, shield_radius_m: float) -> float:
    """Max proton kinetic energy (GeV) deflected by the shield."""
    if field_t <= 0.0 or shield_radius_m <= 0.0:
        return 0.0
    p_c = field_t * shield_radius_m * SPEED_OF_LIGHT / 1.0e9
    total_e = math.sqrt(p_c ** 2 + PROTON_REST_ENERGY_GEV ** 2)
    return max(0.0, total_e - PROTON_REST_ENERGY_GEV)


def gcr_deflection_fraction(energy_cutoff_gev_val: float) -> float:
    """Fraction of GCR flux deflected given energy cutoff."""
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
    """Fraction of solar particle event flux deflected."""
    if field_t <= 0.0 or shield_radius_m <= 0.0:
        return 0.0
    e_cut = energy_cutoff_gev(field_t, shield_radius_m)
    if e_cut >= 0.5:
        return 0.99
    if e_cut >= 0.1:
        return 0.85 + 0.14 * ((e_cut - 0.1) / 0.4)
    return 0.85 * (e_cut / 0.1)


def heat_leak_kw(vessel_area_m2: float, ambient_temp_k: float, coil_temp_k: float) -> float:
    """Total heat leak into cryostat (kW): conduction + radiation."""
    if ambient_temp_k <= coil_temp_k:
        return 0.0
    cond = MLI_CONDUCTANCE_W_M2_K * vessel_area_m2 * (ambient_temp_k - coil_temp_k)
    rad = RADIATION_EMISSIVITY * STEFAN_BOLTZMANN * vessel_area_m2 * (ambient_temp_k**4 - coil_temp_k**4)
    return max(0.0, (cond + rad) / 1000.0)


def joint_heating_kw(current_a: float, num_joints: int = NUM_JOINTS,
                     resistance_per_joint: float = JOINT_RESISTANCE_OHM) -> float:
    """Resistive heating from non-ideal splice joints (kW)."""
    return num_joints * resistance_per_joint * current_a ** 2 / 1000.0


def cryocooler_power_kw(heat_load_kw: float, cop: float = DEFAULT_CRYOCOOLER_COP) -> float:
    """Electrical power for the cryocooler (kW)."""
    if heat_load_kw <= 0.0 or cop <= 0.0:
        return 0.0
    return heat_load_kw / cop


def coolant_boiloff_kg_per_sol(heat_leak_kw_val: float) -> float:
    """LN2 lost to boiloff per sol (kg)."""
    if heat_leak_kw_val <= 0.0:
        return 0.0
    return heat_leak_kw_val * 1000.0 * SECONDS_PER_SOL / (LN2_LATENT_HEAT_KJ_KG * 1000.0)


def wire_mass_kg(radius_m: float, num_turns: int,
                 cross_section_m2: float = DEFAULT_WIRE_CROSS_SECTION_M2,
                 density_kg_m3: float = DEFAULT_WIRE_DENSITY_KG_M3) -> float:
    """Total mass of superconducting wire (kg)."""
    return num_turns * 2.0 * math.pi * radius_m * cross_section_m2 * density_kg_m3


def quench_temperature_rise_k(stored_mj: float, wire_mass: float) -> float:
    """Temperature rise (K) if all stored energy dumps into wire."""
    if wire_mass <= 0.0:
        return 0.0
    return stored_mj * 1.0e6 / (wire_mass * YBCO_SPECIFIC_HEAT_J_KG_K)


@dataclass
class MagneticShieldConfig:
    """Immutable configuration for the magnetic shield."""
    coil_radius_m: float = DEFAULT_COIL_RADIUS_M
    coil_length_m: float = DEFAULT_COIL_LENGTH_M
    num_turns: int = DEFAULT_NUM_TURNS
    operating_current_a: float = DEFAULT_OPERATING_CURRENT_A
    cryocooler_cop: float = DEFAULT_CRYOCOOLER_COP
    vessel_area_m2: float = CRYO_VESSEL_AREA_M2
    max_field_t: float = MAX_OPERATING_FIELD_T


@dataclass
class ShieldState:
    """Mutable state -- updated each tick."""
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
    """Result of one sol tick."""
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


def create_shield(coil_radius_m: float = DEFAULT_COIL_RADIUS_M,
                  coil_length_m: float = DEFAULT_COIL_LENGTH_M,
                  num_turns: int = DEFAULT_NUM_TURNS,
                  operating_current_a: float = DEFAULT_OPERATING_CURRENT_A,
                  coolant_kg: float = DEFAULT_COOLANT_KG,
                  ) -> "tuple[MagneticShieldConfig, ShieldState]":
    """Create a new magnetic shield."""
    config = MagneticShieldConfig(
        coil_radius_m=max(0.1, coil_radius_m),
        coil_length_m=max(0.1, coil_length_m),
        num_turns=max(0, num_turns),
        operating_current_a=max(0.0, operating_current_a),
    )
    return config, ShieldState(coolant_kg=max(0.0, coolant_kg))


def tick(config: MagneticShieldConfig, state: ShieldState,
         spe_active: bool = False, spe_msv: float = 0.0,
         ambient_temp_k: float = MARS_AMBIENT_TEMP_K) -> TickResult:
    """Advance the magnetic shield by one sol."""
    state.sol += 1
    result = TickResult(sol=state.sol)

    # Quench recovery
    if state.quenched:
        if state.coolant_kg > 0.0 and state.coil_temp_k > YBCO_TC_K:
            state.coil_temp_k = max(LN2_BOILING_POINT_K, state.coil_temp_k - 5.0)
        if state.coil_temp_k < YBCO_TC_K - 5.0:
            state.quenched = False

    # Pre-ramp quench detection
    if not state.quenched and state.coil_temp_k >= YBCO_TC_K and state.field_t > 0.001:
        w_mass = wire_mass_kg(config.coil_radius_m, config.num_turns)
        state.coil_temp_k += quench_temperature_rise_k(state.stored_energy_mj, w_mass)
        state.field_t = 0.0
        state.current_a = 0.0
        state.stored_energy_mj = 0.0
        state.quenched = True
        state.shield_active = False
        state.quench_count += 1

    # Field ramp
    target = min(solenoid_field_t(config.num_turns, config.operating_current_a,
                                  config.coil_length_m), config.max_field_t)

    if (not state.quenched and state.coolant_kg > 0.0
            and state.cryocooler_on and state.coil_temp_k < YBCO_TC_K):
        delta = target - state.field_t
        if abs(delta) > MAX_DB_DT_T_PER_SOL:
            delta = MAX_DB_DT_T_PER_SOL if delta > 0 else -MAX_DB_DT_T_PER_SOL
        state.field_t = max(0.0, state.field_t + delta)
        state.current_a = config.operating_current_a * (state.field_t / target) if target > 0 else 0.0
    elif state.quenched or state.coil_temp_k >= YBCO_TC_K:
        state.field_t = 0.0
        state.current_a = 0.0
    else:
        state.field_t = max(0.0, state.field_t - MAX_DB_DT_T_PER_SOL * 0.5)
        if state.field_t == 0.0:
            state.current_a = 0.0

    state.shield_active = state.field_t > 0.001
    state.peak_field_t = max(state.peak_field_t, state.field_t)

    # Stored energy
    vol = coil_volume_m3(config.coil_radius_m, config.coil_length_m)
    state.stored_energy_mj = stored_energy_mj(state.field_t, vol)

    # Thermal management
    h_leak = heat_leak_kw(config.vessel_area_m2, ambient_temp_k, state.coil_temp_k)
    j_heat = joint_heating_kw(state.current_a)
    total_heat = h_leak + j_heat

    if state.cryocooler_on and state.coolant_kg > 0.0 and not state.quenched:
        cryo_power = cryocooler_power_kw(total_heat, config.cryocooler_cop)
        boiloff = coolant_boiloff_kg_per_sol(total_heat * 0.01)
    else:
        cryo_power = 0.0
        boiloff = coolant_boiloff_kg_per_sol(total_heat)
        if total_heat > 0.0:
            w_mass = wire_mass_kg(config.coil_radius_m, config.num_turns)
            if w_mass > 0.0:
                state.coil_temp_k += total_heat * 1000.0 * SECONDS_PER_SOL / (w_mass * YBCO_SPECIFIC_HEAT_J_KG_K)

    state.coolant_kg = max(0.0, state.coolant_kg - boiloff)
    state.total_coolant_consumed_kg += boiloff
    state.total_power_consumed_kwh += cryo_power * SECONDS_PER_SOL / 3600.0

    # Post-thermal quench backup
    if not state.quenched and state.coil_temp_k >= YBCO_TC_K and state.current_a > 0.0:
        w_mass = wire_mass_kg(config.coil_radius_m, config.num_turns)
        state.coil_temp_k += quench_temperature_rise_k(state.stored_energy_mj, w_mass)
        state.field_t = 0.0
        state.current_a = 0.0
        state.stored_energy_mj = 0.0
        state.quenched = True
        state.shield_active = False
        state.quench_count += 1

    # Radiation deflection
    e_cut = energy_cutoff_gev(state.field_t, config.coil_radius_m)
    gcr_frac = gcr_deflection_fraction(e_cut)
    spe_frac = spe_deflection_fraction(state.field_t, config.coil_radius_m)
    gcr_reduced = GCR_SURFACE_MSV_SOL * gcr_frac
    spe_reduced = max(0.0, spe_msv) * spe_frac if spe_active else 0.0
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


def run_simulation(sols: int = 100,
                   config: "MagneticShieldConfig | None" = None,
                   state: "ShieldState | None" = None,
                   spe_events: "dict[int, float] | None" = None,
                   ) -> "dict[str, object]":
    """Run the magnetic shield for N sols."""
    if config is None or state is None:
        config, state = create_shield()
    if spe_events is None:
        spe_events = {}
    results: list[TickResult] = []
    for sol in range(1, sols + 1):
        r = tick(config, state, spe_active=sol in spe_events, spe_msv=spe_events.get(sol, 0.0))
        results.append(r)
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
        "energy_cutoff_gev": round(energy_cutoff_gev(state.field_t, config.coil_radius_m), 4),
        "stored_energy_mj": round(state.stored_energy_mj, 4),
        "ramp_up_sols": ramp_up_sols,
        "quench_count": state.quench_count,
        "shield_active": state.shield_active,
    }
