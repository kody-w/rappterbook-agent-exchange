"""solar_concentrator.py -- Mars Parabolic Solar Thermal Concentrator.

The colony runs every industrial furnace on electricity: plasma arcs for
smelting (ore_smelter.py), resistive heaters for sintering (regolith_
sintering.py), electric kilns for concrete (martian_concrete.py).  But
Mars gets 590 W/m2 of free sunlight.  A parabolic mirror can concentrate
that to >2000 K at the focal point -- hot enough to melt regolith, fire
bricks, and calcine concrete -- without drawing a single watt from the
power grid.  This is the colony's free kiln.

Physics modelled
----------------
* Solar flux on Mars -- Inverse-square from 1 AU.  Seasonal variation
  from Mars eccentricity (e=0.0934).  Dust storm attenuation via optical
  depth tau: S = S0 * exp(-tau/cos(theta_z)).

* Parabolic optics -- Concentration ratio C = A_mirror / A_receiver.
  Practical limit ~3000 for dish concentrators.  Receiver temperature
  from radiative equilibrium: T = (C * S * alpha / (epsilon * sigma))^(1/4).

* Thermal receiver -- Cavity receiver with absorptivity ~0.95.  Heat loss
  by radiation (Stefan-Boltzmann) + conduction through insulation.

* Molten salt storage -- NaCl-KCl eutectic (Mars-producible from regolith
  chlorides).  Stores thermal energy at 800-1100 K for nighttime operation.

* Dust degradation -- Mirror reflectivity drops ~0.3%/sol from dust
  deposition.  Cleaning restores to baseline minus permanent weathering.

* Sun tracking -- Two-axis alt-azimuth.  Cosine loss from tracking error.

Conservation laws
-----------------
- Energy: P_absorbed = P_delivered + P_radiation_loss + P_conduction_loss
- Power: P_thermal <= C * S_mars * A_mirror * eta_optical * eta_receiver
- Temperature: T_receiver <= T_equilibrium(C, S)
- Reflectivity: 0 <= R <= R_baseline (degrades monotonically between cleans)
- Storage: 0 <= E_stored <= E_capacity

One tick = one sol.  Power in kW, energy in kWh, temperature in K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

STEFAN_BOLTZMANN = 5.670_374_419e-8
SOLAR_CONSTANT_1AU = 1361.0
MARS_SEMI_MAJOR_AU = 1.524
MARS_ECCENTRICITY = 0.0934
MARS_YEAR_SOLS = 668.6
MARS_SOL_HOURS = 24.66
MARS_SOL_SECONDS = 88_775.0
MARS_AMBIENT_TEMP_K = 210.0

SOLAR_HOURS_CLEAR = 11.0
SOLAR_HOURS_STORM = 3.0

# ---------------------------------------------------------------------------
# Default concentrator parameters
# ---------------------------------------------------------------------------

DEFAULT_MIRROR_AREA_M2 = 100.0
DEFAULT_MIRROR_REFLECTIVITY = 0.92
DEFAULT_RECEIVER_AREA_M2 = 0.05
DEFAULT_RECEIVER_ABSORPTIVITY = 0.95
DEFAULT_RECEIVER_EMISSIVITY = 0.90
DEFAULT_RECEIVER_INSULATION_KW_K = 0.001
MAX_PRACTICAL_CONCENTRATION = 3000.0
TRACKING_EFFICIENCY = 0.97

DUST_REFLECTIVITY_LOSS_PER_SOL = 0.003
PERMANENT_WEATHERING_PER_CLEAN = 0.0005
MIN_REFLECTIVITY = 0.10

# ---------------------------------------------------------------------------
# Molten salt thermal storage (NaCl-KCl eutectic)
# ---------------------------------------------------------------------------

SALT_SPECIFIC_HEAT_J_KGK = 1050.0
SALT_DENSITY_KG_M3 = 1600.0
SALT_MIN_TEMP_K = 700.0
SALT_MAX_TEMP_K = 1100.0
SALT_TANK_LOSS_FRAC_PER_SOL = 0.02
DEFAULT_SALT_MASS_KG = 5000.0


# ---------------------------------------------------------------------------
# Solar flux computation
# ---------------------------------------------------------------------------

def mars_distance_au(sol_of_year):
    """Mars-Sun distance in AU for a given sol of the Mars year."""
    mean_anomaly = 2.0 * math.pi * sol_of_year / MARS_YEAR_SOLS
    return MARS_SEMI_MAJOR_AU * (1.0 - MARS_ECCENTRICITY * math.cos(mean_anomaly))


def solar_flux_w_m2(sol_of_year=0.0, dust_optical_depth=0.3,
                     zenith_angle_deg=30.0):
    """Solar irradiance on Mars surface after atmospheric attenuation."""
    if zenith_angle_deg >= 90.0:
        return 0.0
    r_au = mars_distance_au(sol_of_year)
    if r_au <= 0.0:
        return 0.0
    s_top = SOLAR_CONSTANT_1AU / (r_au ** 2)
    cos_z = math.cos(math.radians(zenith_angle_deg))
    if cos_z <= 0.0:
        return 0.0
    attenuation = math.exp(-dust_optical_depth / cos_z)
    return max(0.0, s_top * attenuation)


def daily_average_flux_w_m2(sol_of_year=0.0, dust_optical_depth=0.3):
    """Average flux over solar hours (cosine-weighted average)."""
    r_au = mars_distance_au(sol_of_year)
    s_top = SOLAR_CONSTANT_1AU / (r_au ** 2)
    avg_cos = 2.0 / math.pi
    attenuation = math.exp(-dust_optical_depth / avg_cos)
    return max(0.0, s_top * attenuation * avg_cos)


# ---------------------------------------------------------------------------
# Concentrator optics
# ---------------------------------------------------------------------------

def concentration_ratio(mirror_area_m2, receiver_area_m2):
    """Geometric concentration ratio C = A_mirror / A_receiver."""
    if receiver_area_m2 <= 0.0:
        return 0.0
    return min(mirror_area_m2 / receiver_area_m2, MAX_PRACTICAL_CONCENTRATION)


def equilibrium_temperature_k(flux_w_m2, concentration,
                               absorptivity=DEFAULT_RECEIVER_ABSORPTIVITY,
                               emissivity=DEFAULT_RECEIVER_EMISSIVITY):
    """Radiative equilibrium temperature of the receiver."""
    if flux_w_m2 <= 0.0 or concentration <= 0.0:
        return MARS_AMBIENT_TEMP_K
    if emissivity <= 0.0:
        return MARS_AMBIENT_TEMP_K
    numerator = concentration * flux_w_m2 * absorptivity
    denominator = emissivity * STEFAN_BOLTZMANN
    return (numerator / denominator) ** 0.25


def absorbed_power_kw(flux_w_m2, mirror_area_m2, reflectivity,
                       absorptivity=DEFAULT_RECEIVER_ABSORPTIVITY,
                       tracking_eff=TRACKING_EFFICIENCY):
    """Thermal power absorbed by receiver."""
    if flux_w_m2 <= 0.0 or mirror_area_m2 <= 0.0:
        return 0.0
    p_w = flux_w_m2 * mirror_area_m2 * reflectivity * absorptivity * tracking_eff
    return max(0.0, p_w / 1000.0)


# ---------------------------------------------------------------------------
# Thermal losses
# ---------------------------------------------------------------------------

def radiation_loss_kw(receiver_temp_k,
                       receiver_area_m2=DEFAULT_RECEIVER_AREA_M2,
                       emissivity=DEFAULT_RECEIVER_EMISSIVITY):
    """Radiation loss from receiver."""
    if receiver_temp_k <= MARS_AMBIENT_TEMP_K:
        return 0.0
    q_w = (emissivity * STEFAN_BOLTZMANN * receiver_area_m2 *
           (receiver_temp_k ** 4 - MARS_AMBIENT_TEMP_K ** 4))
    return max(0.0, q_w / 1000.0)


def conduction_loss_kw(receiver_temp_k,
                        insulation_kw_k=DEFAULT_RECEIVER_INSULATION_KW_K):
    """Conductive loss through receiver insulation."""
    if receiver_temp_k <= MARS_AMBIENT_TEMP_K:
        return 0.0
    return max(0.0, insulation_kw_k * (receiver_temp_k - MARS_AMBIENT_TEMP_K))


def net_thermal_power_kw(absorbed_kw, receiver_temp_k,
                          receiver_area_m2=DEFAULT_RECEIVER_AREA_M2,
                          emissivity=DEFAULT_RECEIVER_EMISSIVITY,
                          insulation_kw_k=DEFAULT_RECEIVER_INSULATION_KW_K):
    """Net useful thermal power after losses."""
    rad = radiation_loss_kw(receiver_temp_k, receiver_area_m2, emissivity)
    cond = conduction_loss_kw(receiver_temp_k, insulation_kw_k)
    return max(0.0, absorbed_kw - rad - cond)


# ---------------------------------------------------------------------------
# Dust degradation model
# ---------------------------------------------------------------------------

def reflectivity_after_dust(current_reflectivity, sols_since_clean=1):
    """Mirror reflectivity after dust accumulation."""
    degraded = current_reflectivity - DUST_REFLECTIVITY_LOSS_PER_SOL * sols_since_clean
    return max(MIN_REFLECTIVITY, degraded)


def clean_mirror(current_reflectivity, baseline_reflectivity,
                  cleanings_total):
    """Restore reflectivity after cleaning, with permanent weathering."""
    weathering = PERMANENT_WEATHERING_PER_CLEAN * cleanings_total
    restored = baseline_reflectivity - weathering
    return max(MIN_REFLECTIVITY, min(restored, baseline_reflectivity))


# ---------------------------------------------------------------------------
# Thermal storage (molten salt)
# ---------------------------------------------------------------------------

def salt_energy_capacity_kwh(mass_kg, temp_range_k=None):
    """Maximum thermal energy storable in the salt tank."""
    if mass_kg <= 0.0:
        return 0.0
    dt = temp_range_k if temp_range_k is not None else (SALT_MAX_TEMP_K - SALT_MIN_TEMP_K)
    dt = max(0.0, dt)
    energy_j = mass_kg * SALT_SPECIFIC_HEAT_J_KGK * dt
    return energy_j / 3.6e6


def salt_charge_kwh(available_kw, hours, current_kwh, capacity_kwh):
    """Energy added to salt storage.  Capped at capacity."""
    if available_kw <= 0.0 or hours <= 0.0:
        return 0.0
    added = available_kw * hours
    headroom = max(0.0, capacity_kwh - current_kwh)
    return min(added, headroom)


def salt_discharge_kwh(demand_kw, hours, current_kwh):
    """Energy drawn from salt storage.  Capped at available."""
    if demand_kw <= 0.0 or hours <= 0.0:
        return 0.0
    needed = demand_kw * hours
    return min(needed, max(0.0, current_kwh))


def salt_thermal_loss_kwh(stored_kwh):
    """Heat lost from storage per sol (fractional loss)."""
    return max(0.0, stored_kwh * SALT_TANK_LOSS_FRAC_PER_SOL)


# ---------------------------------------------------------------------------
# Concentrator state
# ---------------------------------------------------------------------------

@dataclass
class ConcentratorConfig:
    """Immutable configuration for a parabolic concentrator."""

    mirror_area_m2: float = DEFAULT_MIRROR_AREA_M2
    receiver_area_m2: float = DEFAULT_RECEIVER_AREA_M2
    receiver_absorptivity: float = DEFAULT_RECEIVER_ABSORPTIVITY
    receiver_emissivity: float = DEFAULT_RECEIVER_EMISSIVITY
    insulation_kw_k: float = DEFAULT_RECEIVER_INSULATION_KW_K
    baseline_reflectivity: float = DEFAULT_MIRROR_REFLECTIVITY
    salt_mass_kg: float = DEFAULT_SALT_MASS_KG

    def __post_init__(self):
        self.mirror_area_m2 = max(0.1, self.mirror_area_m2)
        self.receiver_area_m2 = max(0.001, self.receiver_area_m2)
        self.baseline_reflectivity = max(0.1, min(1.0, self.baseline_reflectivity))
        self.salt_mass_kg = max(0.0, self.salt_mass_kg)

    @property
    def concentration(self):
        return concentration_ratio(self.mirror_area_m2, self.receiver_area_m2)

    @property
    def salt_capacity_kwh(self):
        return salt_energy_capacity_kwh(self.salt_mass_kg)

    @property
    def mirror_diameter_m(self):
        return 2.0 * math.sqrt(self.mirror_area_m2 / math.pi)


@dataclass
class ConcentratorState:
    """Mutable state of the concentrator, advanced each sol."""

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
    dust_optical_depth: float = 0.3
    is_dust_storm: bool = False


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def create_concentrator(mirror_area_m2=DEFAULT_MIRROR_AREA_M2,
                         receiver_area_m2=DEFAULT_RECEIVER_AREA_M2,
                         salt_mass_kg=DEFAULT_SALT_MASS_KG,
                         reflectivity=DEFAULT_MIRROR_REFLECTIVITY):
    """Create a new solar concentrator system."""
    config = ConcentratorConfig(
        mirror_area_m2=mirror_area_m2,
        receiver_area_m2=receiver_area_m2,
        salt_mass_kg=salt_mass_kg,
        baseline_reflectivity=reflectivity,
    )
    state = ConcentratorState(reflectivity=reflectivity)
    return config, state


def tick(config, state, thermal_demand_kw=0.0, dust_optical_depth=None,
         is_dust_storm=False, sol_of_year=None):
    """Advance the concentrator by one sol."""
    state.sols_operated += 1

    if sol_of_year is not None:
        state.sol_of_year = sol_of_year
    else:
        state.sol_of_year = (state.sol_of_year + 1.0) % MARS_YEAR_SOLS
    if dust_optical_depth is not None:
        state.dust_optical_depth = dust_optical_depth
    state.is_dust_storm = is_dust_storm

    tau = state.dust_optical_depth
    if is_dust_storm:
        tau = max(tau, 3.0)
    flux = daily_average_flux_w_m2(state.sol_of_year, tau)
    solar_hours = SOLAR_HOURS_STORM if is_dust_storm else SOLAR_HOURS_CLEAR

    p_absorbed = absorbed_power_kw(
        flux, config.mirror_area_m2, state.reflectivity,
        config.receiver_absorptivity,
    )

    C = config.concentration
    t_eq = equilibrium_temperature_k(flux, C, config.receiver_absorptivity,
                                     config.receiver_emissivity)
    state.receiver_temp_k = t_eq

    absorbed_kwh = p_absorbed * solar_hours
    lost_rad = radiation_loss_kw(state.receiver_temp_k, config.receiver_area_m2,
                                 config.receiver_emissivity) * solar_hours
    lost_cond = conduction_loss_kw(state.receiver_temp_k,
                                   config.insulation_kw_k) * solar_hours
    total_lost = lost_rad + lost_cond
    net_kwh = max(0.0, absorbed_kwh - total_lost)

    demand_kwh = max(0.0, thermal_demand_kw * MARS_SOL_HOURS)
    delivered_from_solar = min(net_kwh, demand_kwh)
    surplus = net_kwh - delivered_from_solar

    capacity = config.salt_capacity_kwh
    charged = salt_charge_kwh(surplus / max(solar_hours, 0.01), solar_hours,
                               state.salt_stored_kwh, capacity)
    state.salt_stored_kwh += charged

    unmet = demand_kwh - delivered_from_solar
    discharged = 0.0
    if unmet > 0.0:
        discharged = salt_discharge_kwh(unmet / MARS_SOL_HOURS, MARS_SOL_HOURS,
                                        state.salt_stored_kwh)
        state.salt_stored_kwh -= discharged

    total_delivered = delivered_from_solar + discharged

    storage_loss = salt_thermal_loss_kwh(state.salt_stored_kwh)
    state.salt_stored_kwh = max(0.0, state.salt_stored_kwh - storage_loss)

    state.reflectivity = reflectivity_after_dust(state.reflectivity, 1)
    state.sols_since_clean += 1

    state.absorbed_kwh_today = round(absorbed_kwh, 4)
    state.delivered_kwh_today = round(total_delivered, 4)
    state.lost_kwh_today = round(total_lost + storage_loss, 4)
    state.stored_delta_kwh_today = round(charged - discharged - storage_loss, 4)

    state.total_absorbed_kwh += absorbed_kwh
    state.total_delivered_kwh += total_delivered
    state.total_lost_kwh += total_lost + storage_loss

    return {
        "sol": state.sols_operated,
        "flux_w_m2": round(flux, 2),
        "absorbed_kwh": round(absorbed_kwh, 2),
        "delivered_kwh": round(total_delivered, 2),
        "lost_kwh": round(total_lost + storage_loss, 2),
        "net_kwh": round(net_kwh, 2),
        "receiver_temp_k": round(state.receiver_temp_k, 1),
        "reflectivity": round(state.reflectivity, 4),
        "salt_stored_kwh": round(state.salt_stored_kwh, 2),
        "salt_capacity_kwh": round(capacity, 2),
        "solar_hours": solar_hours,
        "dust_storm": is_dust_storm,
        "demand_met_pct": round(total_delivered / demand_kwh * 100, 1) if demand_kwh > 0 else 100.0,
    }


def clean(config, state):
    """Clean the mirror, restoring reflectivity."""
    state.cleanings_total += 1
    state.reflectivity = clean_mirror(
        state.reflectivity,
        config.baseline_reflectivity,
        state.cleanings_total,
    )
    state.sols_since_clean = 0
    return {
        "reflectivity": round(state.reflectivity, 4),
        "cleanings_total": state.cleanings_total,
        "permanent_weathering": round(
            PERMANENT_WEATHERING_PER_CLEAN * state.cleanings_total, 4),
    }


def status(config, state):
    """Full diagnostic snapshot."""
    return {
        "mirror_area_m2": config.mirror_area_m2,
        "mirror_diameter_m": round(config.mirror_diameter_m, 2),
        "concentration_ratio": round(config.concentration, 1),
        "reflectivity": round(state.reflectivity, 4),
        "baseline_reflectivity": config.baseline_reflectivity,
        "reflectivity_loss_pct": round(
            (1.0 - state.reflectivity / config.baseline_reflectivity) * 100, 2),
        "receiver_temp_k": round(state.receiver_temp_k, 1),
        "salt_stored_kwh": round(state.salt_stored_kwh, 2),
        "salt_capacity_kwh": round(config.salt_capacity_kwh, 2),
        "salt_fill_pct": round(
            state.salt_stored_kwh / config.salt_capacity_kwh * 100, 1
        ) if config.salt_capacity_kwh > 0 else 0.0,
        "sols_since_clean": state.sols_since_clean,
        "cleanings_total": state.cleanings_total,
        "total_absorbed_kwh": round(state.total_absorbed_kwh, 1),
        "total_delivered_kwh": round(state.total_delivered_kwh, 1),
        "total_lost_kwh": round(state.total_lost_kwh, 1),
        "lifetime_efficiency": round(
            state.total_delivered_kwh / state.total_absorbed_kwh, 4
        ) if state.total_absorbed_kwh > 0 else 0.0,
        "sols_operated": state.sols_operated,
    }


def run_simulation(sols=365, config=None, state=None, thermal_demand_kw=50.0,
                    clean_interval_sols=30, dust_storm_sols=None):
    """Run a multi-sol concentrator simulation."""
    if config is None or state is None:
        config, state = create_concentrator()

    storm_set = set(dust_storm_sols) if dust_storm_sols else set()
    timeline = []

    for sol in range(sols):
        is_storm = sol in storm_set
        result = tick(config, state, thermal_demand_kw, is_dust_storm=is_storm)
        timeline.append(result)
        if clean_interval_sols > 0 and (sol + 1) % clean_interval_sols == 0:
            clean(config, state)

    return {
        "sols_simulated": sols,
        "final_status": status(config, state),
        "total_delivered_kwh": round(state.total_delivered_kwh, 1),
        "total_absorbed_kwh": round(state.total_absorbed_kwh, 1),
        "total_lost_kwh": round(state.total_lost_kwh, 1),
        "average_delivered_kwh_sol": round(state.total_delivered_kwh / max(sols, 1), 2),
        "cleanings_performed": state.cleanings_total,
        "dust_storm_sols": len(storm_set),
        "peak_receiver_temp_k": round(
            max((t["receiver_temp_k"] for t in timeline), default=0), 1),
        "min_reflectivity": round(
            min((t["reflectivity"] for t in timeline), default=0), 4),
    }
