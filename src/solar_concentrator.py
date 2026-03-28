"""Mars Parabolic Solar Thermal Concentrator simulation module.

Models solar flux on Mars, parabolic mirror optics, thermal receiver
performance, dust degradation, and molten salt thermal storage.
One simulation tick equals one Martian sol.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
STEFAN_BOLTZMANN: float = 5.670_374_419e-8  # W m^-2 K^-4

# Solar
SOLAR_CONSTANT_1AU: float = 1361.0  # W/m^2

# Mars orbital parameters
MARS_SEMI_MAJOR_AU: float = 1.524
MARS_ECCENTRICITY: float = 0.0934
MARS_YEAR_SOLS: float = 668.6
MARS_SOL_HOURS: float = 24.66
MARS_AMBIENT_TEMP_K: float = 210.0

# Default concentrator geometry
DEFAULT_MIRROR_AREA_M2: float = 100.0
DEFAULT_MIRROR_REFLECTIVITY: float = 0.92
DEFAULT_RECEIVER_AREA_M2: float = 0.05
DEFAULT_RECEIVER_ABSORPTIVITY: float = 0.95
DEFAULT_RECEIVER_EMISSIVITY: float = 0.90
MAX_PRACTICAL_CONCENTRATION: float = 3000.0
TRACKING_EFFICIENCY: float = 0.97

# Dust and degradation
DUST_REFLECTIVITY_LOSS_PER_SOL: float = 0.003
PERMANENT_WEATHERING_PER_CLEAN: float = 0.0005
MIN_REFLECTIVITY: float = 0.10

# Molten salt storage (NaCl-KCl eutectic)
SALT_SPECIFIC_HEAT_J_KGK: float = 1050.0
SALT_MIN_TEMP_K: float = 700.0
SALT_MAX_TEMP_K: float = 1100.0
SALT_TANK_LOSS_FRAC_PER_SOL: float = 0.02
DEFAULT_SALT_MASS_KG: float = 5000.0

# Operational
SOLAR_HOURS_CLEAR: float = 11.0
SOLAR_HOURS_STORM: float = 3.0
DEFAULT_RECEIVER_INSULATION_KW_K: float = 0.001

# ---------------------------------------------------------------------------
# Orbital / flux helpers
# ---------------------------------------------------------------------------

def mars_distance_au(sol_of_year: float) -> float:
    """Heliocentric distance of Mars in AU using simplified Kepler approx.

    r ~ a * (1 - e * cos(2*pi*sol / year))
    """
    mean_anomaly = 2.0 * math.pi * sol_of_year / MARS_YEAR_SOLS
    return MARS_SEMI_MAJOR_AU * (1.0 - MARS_ECCENTRICITY * math.cos(mean_anomaly))


def solar_flux_w_m2(
    sol_of_year: float,
    dust_optical_depth: float = 0.5,
    zenith_angle_deg: float = 0.0,
) -> float:
    """Instantaneous solar flux on the Martian surface (W/m^2).

    Applies inverse-square law and Beer-Lambert dust attenuation.
    Returns 0 when the sun is at or below the horizon.
    """
    if zenith_angle_deg >= 90.0:
        return 0.0
    cos_z = math.cos(math.radians(zenith_angle_deg))
    if cos_z <= 0.0:
        return 0.0
    r = mars_distance_au(sol_of_year)
    top_of_atmosphere = SOLAR_CONSTANT_1AU / (r * r)
    return top_of_atmosphere * math.exp(-dust_optical_depth / cos_z)


def daily_average_flux_w_m2(
    sol_of_year: float,
    dust_optical_depth: float = 0.5,
) -> float:
    """Average solar flux during daytime hours on Mars.

    Uses avg_cos(zenith) = 2/pi as the representative daytime value.
    """
    r = mars_distance_au(sol_of_year)
    top_of_atmosphere = SOLAR_CONSTANT_1AU / (r * r)
    avg_cos_zenith = 2.0 / math.pi
    return top_of_atmosphere * math.exp(-dust_optical_depth / avg_cos_zenith)


# ---------------------------------------------------------------------------
# Optics helpers
# ---------------------------------------------------------------------------

def concentration_ratio(mirror_area_m2: float, receiver_area_m2: float) -> float:
    """Geometric concentration ratio, capped at MAX_PRACTICAL_CONCENTRATION."""
    if receiver_area_m2 <= 0.0:
        return MAX_PRACTICAL_CONCENTRATION
    ratio = mirror_area_m2 / receiver_area_m2
    return min(ratio, MAX_PRACTICAL_CONCENTRATION)


def equilibrium_temperature_k(
    flux_w_m2: float,
    concentration: float,
    absorptivity: float,
    emissivity: float,
) -> float:
    """Stagnation temperature of the receiver (K).

    T = (C * S * alpha / (epsilon * sigma)) ^ 0.25
    Returns ambient temperature when inputs are non-positive.
    """
    if flux_w_m2 <= 0.0 or concentration <= 0.0 or emissivity <= 0.0:
        return MARS_AMBIENT_TEMP_K
    numerator = concentration * flux_w_m2 * absorptivity
    denominator = emissivity * STEFAN_BOLTZMANN
    return (numerator / denominator) ** 0.25


# ---------------------------------------------------------------------------
# Thermal helpers
# ---------------------------------------------------------------------------

def absorbed_power_kw(
    flux_w_m2: float,
    mirror_area_m2: float,
    reflectivity: float,
    absorptivity: float,
    tracking_eff: float = TRACKING_EFFICIENCY,
) -> float:
    """Thermal power absorbed by the receiver (kW)."""
    return (
        flux_w_m2 * mirror_area_m2 * reflectivity * absorptivity * tracking_eff
        / 1000.0
    )


def radiation_loss_kw(
    receiver_temp_k: float,
    receiver_area_m2: float,
    emissivity: float,
) -> float:
    """Radiation loss from the receiver surface (kW)."""
    dt4 = receiver_temp_k ** 4 - MARS_AMBIENT_TEMP_K ** 4
    return emissivity * STEFAN_BOLTZMANN * receiver_area_m2 * dt4 / 1000.0


def conduction_loss_kw(
    receiver_temp_k: float,
    insulation_kw_k: float = DEFAULT_RECEIVER_INSULATION_KW_K,
) -> float:
    """Conduction loss through receiver insulation (kW)."""
    return insulation_kw_k * (receiver_temp_k - MARS_AMBIENT_TEMP_K)


def net_thermal_power_kw(
    absorbed_kw: float,
    receiver_temp_k: float,
    receiver_area_m2: float,
    emissivity: float,
    insulation_kw_k: float = DEFAULT_RECEIVER_INSULATION_KW_K,
) -> float:
    """Net useful thermal power after radiation and conduction losses (kW)."""
    rad = radiation_loss_kw(receiver_temp_k, receiver_area_m2, emissivity)
    cond = conduction_loss_kw(receiver_temp_k, insulation_kw_k)
    return max(0.0, absorbed_kw - rad - cond)


# ---------------------------------------------------------------------------
# Dust degradation
# ---------------------------------------------------------------------------

def reflectivity_after_dust(
    current_reflectivity: float,
    sols_since_clean: int,
) -> float:
    """Mirror reflectivity after *sols_since_clean* additional sols of dust."""
    degraded = current_reflectivity * (
        (1.0 - DUST_REFLECTIVITY_LOSS_PER_SOL) ** sols_since_clean
    )
    return max(degraded, MIN_REFLECTIVITY)


def clean_mirror(
    current_reflectivity: float,
    baseline_reflectivity: float,
    cleanings_total: int,
) -> float:
    """Reflectivity after cleaning, accounting for permanent weathering.

    *cleanings_total* is the count **before** this cleaning event.
    """
    new_count = cleanings_total + 1
    weathered_baseline = baseline_reflectivity - (
        PERMANENT_WEATHERING_PER_CLEAN * new_count
    )
    return max(weathered_baseline, MIN_REFLECTIVITY)


# ---------------------------------------------------------------------------
# Molten-salt thermal storage
# ---------------------------------------------------------------------------

def salt_energy_capacity_kwh(
    mass_kg: float,
    temp_range_k: float = None,
) -> float:
    """Energy capacity of the molten salt store (kWh).

    E = m * cp * dT / 3.6e6
    """
    if temp_range_k is None:
        temp_range_k = SALT_MAX_TEMP_K - SALT_MIN_TEMP_K
    return mass_kg * SALT_SPECIFIC_HEAT_J_KGK * temp_range_k / 3.6e6


def salt_charge_kwh(
    available_kw: float,
    hours: float,
    current_kwh: float,
    capacity_kwh: float,
) -> float:
    """Energy actually charged into the salt store (kWh)."""
    energy = available_kw * hours
    room = capacity_kwh - current_kwh
    return max(0.0, min(energy, room))


def salt_discharge_kwh(
    demand_kw: float,
    hours: float,
    current_kwh: float,
) -> float:
    """Energy actually discharged from the salt store (kWh)."""
    needed = demand_kw * hours
    return max(0.0, min(needed, current_kwh))


def salt_thermal_loss_kwh(stored_kwh: float) -> float:
    """Thermal loss from the salt tank per sol (kWh)."""
    return stored_kwh * SALT_TANK_LOSS_FRAC_PER_SOL


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConcentratorConfig:
    """Immutable design parameters for a parabolic solar concentrator."""

    mirror_area_m2: float = DEFAULT_MIRROR_AREA_M2
    receiver_area_m2: float = DEFAULT_RECEIVER_AREA_M2
    receiver_absorptivity: float = DEFAULT_RECEIVER_ABSORPTIVITY
    receiver_emissivity: float = DEFAULT_RECEIVER_EMISSIVITY
    insulation_kw_k: float = DEFAULT_RECEIVER_INSULATION_KW_K
    baseline_reflectivity: float = DEFAULT_MIRROR_REFLECTIVITY
    salt_mass_kg: float = DEFAULT_SALT_MASS_KG

    @property
    def concentration(self) -> float:
        """Geometric concentration ratio."""
        return concentration_ratio(self.mirror_area_m2, self.receiver_area_m2)

    @property
    def salt_capacity_kwh(self) -> float:
        """Total thermal storage capacity (kWh)."""
        return salt_energy_capacity_kwh(self.salt_mass_kg)

    @property
    def mirror_diameter_m(self) -> float:
        """Equivalent circular mirror diameter (m)."""
        return 2.0 * math.sqrt(self.mirror_area_m2 / math.pi)


@dataclass
class ConcentratorState:
    """Mutable runtime state for the concentrator simulation."""

    reflectivity: float = DEFAULT_MIRROR_REFLECTIVITY
    sols_since_clean: int = 0
    cleanings_total: int = 0
    receiver_temp_k: float = MARS_AMBIENT_TEMP_K
    salt_stored_kwh: float = 0.0
    absorbed_kwh_today: float = 0.0
    delivered_kwh_today: float = 0.0
    lost_kwh_today: float = 0.0
    stored_delta_kwh_today: float = 0.0
    total_absorbed_kwh: float = 0.0
    total_delivered_kwh: float = 0.0
    total_lost_kwh: float = 0.0
    sols_operated: int = 0
    sol_of_year: float = 0.0
    dust_optical_depth: float = 0.5
    is_dust_storm: bool = False


# ---------------------------------------------------------------------------
# High-level operations
# ---------------------------------------------------------------------------

def create_concentrator(
    mirror_area_m2: float = DEFAULT_MIRROR_AREA_M2,
    receiver_area_m2: float = DEFAULT_RECEIVER_AREA_M2,
    receiver_absorptivity: float = DEFAULT_RECEIVER_ABSORPTIVITY,
    receiver_emissivity: float = DEFAULT_RECEIVER_EMISSIVITY,
    insulation_kw_k: float = DEFAULT_RECEIVER_INSULATION_KW_K,
    baseline_reflectivity: float = DEFAULT_MIRROR_REFLECTIVITY,
    salt_mass_kg: float = DEFAULT_SALT_MASS_KG,
    initial_sol: float = 0.0,
    initial_dust: float = 0.5,
) -> tuple:
    """Create a concentrator (config, state) pair."""
    config = ConcentratorConfig(
        mirror_area_m2=max(mirror_area_m2, 0.0),
        receiver_area_m2=max(receiver_area_m2, 1e-6),
        receiver_absorptivity=min(max(receiver_absorptivity, 0.0), 1.0),
        receiver_emissivity=min(max(receiver_emissivity, 0.01), 1.0),
        insulation_kw_k=max(insulation_kw_k, 0.0),
        baseline_reflectivity=min(max(baseline_reflectivity, MIN_REFLECTIVITY), 1.0),
        salt_mass_kg=max(salt_mass_kg, 0.0),
    )
    state = ConcentratorState(
        reflectivity=config.baseline_reflectivity,
        sol_of_year=initial_sol % MARS_YEAR_SOLS,
        dust_optical_depth=max(initial_dust, 0.0),
    )
    return config, state


def tick(
    config: ConcentratorConfig,
    state: ConcentratorState,
    thermal_demand_kw: float = 0.0,
    dust_optical_depth: float = None,
    is_dust_storm: bool = False,
    sol_of_year: float = None,
) -> dict:
    """Advance the simulation by one sol and return a results dict."""
    # 1. Increment sols
    state.sols_operated += 1

    # 2. Update sol_of_year (wrap at MARS_YEAR_SOLS) and dust params
    if sol_of_year is not None:
        state.sol_of_year = sol_of_year % MARS_YEAR_SOLS
    else:
        state.sol_of_year = (state.sol_of_year + 1) % MARS_YEAR_SOLS

    state.is_dust_storm = is_dust_storm
    if dust_optical_depth is not None:
        state.dust_optical_depth = dust_optical_depth

    # 3. During dust storms force tau >= 3.0
    tau = state.dust_optical_depth
    if is_dust_storm:
        tau = max(tau, 3.0)

    # 4. Daily average flux and solar hours
    flux = daily_average_flux_w_m2(state.sol_of_year, tau)
    solar_hours = SOLAR_HOURS_STORM if is_dust_storm else SOLAR_HOURS_CLEAR

    # 5. Absorbed power and receiver temperature
    p_absorbed = absorbed_power_kw(
        flux, config.mirror_area_m2, state.reflectivity,
        config.receiver_absorptivity, TRACKING_EFFICIENCY,
    )

    conc = config.concentration

    # Operating temperature capped at SALT_MAX_TEMP_K so losses stay below
    # absorbed power (a real receiver extracts heat, preventing stagnation).
    t_eq = equilibrium_temperature_k(
        flux, conc, config.receiver_absorptivity, config.receiver_emissivity,
    )
    state.receiver_temp_k = min(t_eq, SALT_MAX_TEMP_K)

    # 6. Absorbed energy over solar hours
    absorbed_kwh = p_absorbed * solar_hours

    # 7. Radiation + conduction losses over solar hours
    rad_kw = radiation_loss_kw(
        state.receiver_temp_k, config.receiver_area_m2, config.receiver_emissivity,
    )
    cond_kw = conduction_loss_kw(state.receiver_temp_k, config.insulation_kw_k)
    loss_kwh = (rad_kw + cond_kw) * solar_hours
    net_kwh = max(0.0, absorbed_kwh - loss_kwh)

    # 8. Deliver directly from solar
    demand_kwh = thermal_demand_kw * MARS_SOL_HOURS
    delivered_solar_kwh = min(net_kwh, demand_kwh)

    # 9. Surplus charges salt storage
    surplus_kwh = net_kwh - delivered_solar_kwh
    if surplus_kwh > 0.0 and solar_hours > 0.0:
        charged_kwh = salt_charge_kwh(
            surplus_kwh / solar_hours, solar_hours,
            state.salt_stored_kwh, config.salt_capacity_kwh,
        )
    else:
        charged_kwh = 0.0
    state.salt_stored_kwh += charged_kwh

    # 10. Unmet demand discharged from salt storage
    unmet_kwh = max(0.0, demand_kwh - delivered_solar_kwh)
    if unmet_kwh > 0.0 and state.salt_stored_kwh > 0.0:
        discharged_kwh = salt_discharge_kwh(
            unmet_kwh / MARS_SOL_HOURS, MARS_SOL_HOURS,
            state.salt_stored_kwh,
        )
    else:
        discharged_kwh = 0.0
    state.salt_stored_kwh -= discharged_kwh

    delivered_total_kwh = delivered_solar_kwh + discharged_kwh

    # 11. Storage thermal losses
    storage_loss_kwh = salt_thermal_loss_kwh(state.salt_stored_kwh)
    state.salt_stored_kwh = max(0.0, state.salt_stored_kwh - storage_loss_kwh)

    # 12. Dust degradation
    state.sols_since_clean += 1
    state.reflectivity = reflectivity_after_dust(state.reflectivity, 1)

    # Update daily / lifetime accumulators
    total_lost_kwh = loss_kwh + storage_loss_kwh
    state.absorbed_kwh_today = absorbed_kwh
    state.delivered_kwh_today = delivered_total_kwh
    state.lost_kwh_today = total_lost_kwh
    state.stored_delta_kwh_today = charged_kwh - discharged_kwh - storage_loss_kwh
    state.total_absorbed_kwh += absorbed_kwh
    state.total_delivered_kwh += delivered_total_kwh
    state.total_lost_kwh += total_lost_kwh

    return {
        "sol": state.sols_operated,
        "sol_of_year": state.sol_of_year,
        "flux_w_m2": flux,
        "solar_hours": solar_hours,
        "absorbed_kwh": absorbed_kwh,
        "delivered_kwh": delivered_total_kwh,
        "lost_kwh": total_lost_kwh,
        "net_kwh": net_kwh,
        "charged_kwh": charged_kwh,
        "discharged_kwh": discharged_kwh,
        "storage_loss_kwh": storage_loss_kwh,
        "salt_stored_kwh": state.salt_stored_kwh,
        "reflectivity": state.reflectivity,
        "receiver_temp_k": state.receiver_temp_k,
        "is_dust_storm": is_dust_storm,
        "dust_optical_depth": tau,
    }


def clean(config: ConcentratorConfig, state: ConcentratorState) -> dict:
    """Clean the mirror.  Returns old/new reflectivity."""
    old_refl = state.reflectivity
    state.reflectivity = clean_mirror(
        old_refl, config.baseline_reflectivity, state.cleanings_total,
    )
    state.cleanings_total += 1
    state.sols_since_clean = 0
    return {
        "old_reflectivity": old_refl,
        "new_reflectivity": state.reflectivity,
        "cleanings_total": state.cleanings_total,
    }


def status(config: ConcentratorConfig, state: ConcentratorState) -> dict:
    """Snapshot of concentrator status."""
    cap = config.salt_capacity_kwh
    storage_pct = (state.salt_stored_kwh / cap * 100.0) if cap > 0.0 else 0.0
    eff = (
        (state.total_delivered_kwh / state.total_absorbed_kwh * 100.0)
        if state.total_absorbed_kwh > 0.0
        else 0.0
    )
    return {
        "sols_operated": state.sols_operated,
        "sol_of_year": state.sol_of_year,
        "reflectivity": state.reflectivity,
        "reflectivity_pct": state.reflectivity * 100.0,
        "receiver_temp_k": state.receiver_temp_k,
        "salt_stored_kwh": state.salt_stored_kwh,
        "salt_capacity_kwh": cap,
        "salt_storage_pct": storage_pct,
        "total_absorbed_kwh": state.total_absorbed_kwh,
        "total_delivered_kwh": state.total_delivered_kwh,
        "total_lost_kwh": state.total_lost_kwh,
        "cleanings_total": state.cleanings_total,
        "efficiency_pct": eff,
        "is_dust_storm": state.is_dust_storm,
        "mirror_area_m2": config.mirror_area_m2,
        "concentration_ratio": config.concentration,
    }


def run_simulation(
    sols: int = 365,
    config: ConcentratorConfig = None,
    state: ConcentratorState = None,
    thermal_demand_kw: float = 50.0,
    clean_interval_sols: int = 30,
    dust_storm_sols: set = None,
) -> dict:
    """Run a multi-sol simulation and return history + summary."""
    if config is None or state is None:
        config, state = create_concentrator()
    if dust_storm_sols is None:
        dust_storm_sols = set()

    history = []  # type: list[dict]
    for sol_num in range(1, sols + 1):
        is_storm = sol_num in dust_storm_sols
        result = tick(
            config, state,
            thermal_demand_kw=thermal_demand_kw,
            is_dust_storm=is_storm,
        )
        history.append(result)

        if clean_interval_sols > 0 and sol_num % clean_interval_sols == 0:
            clean(config, state)

    return {
        "sols": sols,
        "history": history,
        "final_status": status(config, state),
        "config": {
            "mirror_area_m2": config.mirror_area_m2,
            "concentration_ratio": config.concentration,
            "salt_capacity_kwh": config.salt_capacity_kwh,
        },
    }
