"""plasma_forge.py -- Mars Plasma Arc Metal Refinery.

The colony processes regolith into bricks (regolith_sintering.py) and
concrete (martian_concrete.py), but it cannot extract pure metals.
Every bolt, wire, beam, and solar-cell wafer is shipped from Earth at
$2M/kg.  This module closes the loop: feed raw regolith in, get
structural iron, aluminum, titanium, and electronics-grade silicon out.

The forge is a transferred-arc plasma furnace running on CO2 plasma gas
(abundant in the Martian atmosphere).  Electric arcs between graphite
electrodes ionise the gas to 15000-25000 K, melting the regolith
charge and reducing metal oxides via carbothermic reduction.

Physics modelled
----------------
* **Plasma arc** -- Joule heating P = V*I, arc voltage from Ayrton
  equation V = a + b*L + (c + d*L)/I.  Constants for CO2 transferred-
  arc plasma (higher voltage than atmospheric arcs).

* **Carbothermic reduction** -- dominant reactions:
    Fe2O3 + 3C  -> 2Fe + 3CO  (dH ~ +490 kJ/mol Fe2O3)
    Al2O3 + 3C  -> 2Al + 3CO  (dH ~ +1340 kJ/mol)
    TiO2  + 2C  -> Ti  + 2CO  (dH ~ +720 kJ/mol)
    SiO2  + 2C  -> Si  + 2CO  (dH ~ +690 kJ/mol)
  Rates follow Arrhenius kinetics limited by temperature.

* **Energy balance** -- electrical input = sensible heat + latent heat
  + reaction enthalpy + radiation losses + conduction losses.

* **Electrode consumption** -- graphite wear proportional to amp-hours.

Conservation laws
-----------------
- Mass: ore_in = metal_out + slag_out + gas_out (CO produced)
- Energy: P_elec = P_heat + P_react + P_rad + P_cond
- Metal fractions fixed by Mars regolith composition
- Electrode mass decreases monotonically
- Temperature bounded [ambient, equilibrium] with thermal inertia

One tick = one hour.  Mass in kg, power in kW, temperature in K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# -- Mars regolith composition (Viking/Pathfinder/MER) -----------------------

REGOLITH_FE2O3_FRAC = 0.180
REGOLITH_AL2O3_FRAC = 0.071
REGOLITH_TIO2_FRAC = 0.010
REGOLITH_SIO2_FRAC = 0.435
REGOLITH_OTHER_FRAC = 1.0 - (REGOLITH_FE2O3_FRAC + REGOLITH_AL2O3_FRAC +
                               REGOLITH_TIO2_FRAC + REGOLITH_SIO2_FRAC)

FE_YIELD_FROM_OXIDE = 2 * 55.845 / (2 * 55.845 + 3 * 15.999)
AL_YIELD_FROM_OXIDE = 2 * 26.982 / (2 * 26.982 + 3 * 15.999)
TI_YIELD_FROM_OXIDE = 47.867 / (47.867 + 2 * 15.999)
SI_YIELD_FROM_OXIDE = 28.086 / (28.086 + 2 * 15.999)

MAX_FE_PER_KG = REGOLITH_FE2O3_FRAC * FE_YIELD_FROM_OXIDE
MAX_AL_PER_KG = REGOLITH_AL2O3_FRAC * AL_YIELD_FROM_OXIDE
MAX_TI_PER_KG = REGOLITH_TIO2_FRAC * TI_YIELD_FROM_OXIDE
MAX_SI_PER_KG = REGOLITH_SIO2_FRAC * SI_YIELD_FROM_OXIDE

# -- Reaction enthalpies (kJ/mol oxide) --------------------------------------

ENTHALPY_FE2O3_KJ_MOL = 490.0
ENTHALPY_AL2O3_KJ_MOL = 1340.0
ENTHALPY_TIO2_KJ_MOL = 720.0
ENTHALPY_SIO2_KJ_MOL = 690.0

MW_FE2O3 = 2 * 55.845 + 3 * 15.999
MW_AL2O3 = 2 * 26.982 + 3 * 15.999
MW_TIO2 = 47.867 + 2 * 15.999
MW_SIO2 = 28.086 + 2 * 15.999

ENTHALPY_FE2O3_KJ_KG = ENTHALPY_FE2O3_KJ_MOL / MW_FE2O3 * 1000.0
ENTHALPY_AL2O3_KJ_KG = ENTHALPY_AL2O3_KJ_MOL / MW_AL2O3 * 1000.0
ENTHALPY_TIO2_KJ_KG = ENTHALPY_TIO2_KJ_MOL / MW_TIO2 * 1000.0
ENTHALPY_SIO2_KJ_KG = ENTHALPY_SIO2_KJ_MOL / MW_SIO2 * 1000.0

# -- Carbon demand (kg C per kg oxide, stoichiometric) -----------------------

CARBON_PER_KG_FE2O3 = 3 * 12.011 / MW_FE2O3
CARBON_PER_KG_AL2O3 = 3 * 12.011 / MW_AL2O3
CARBON_PER_KG_TIO2 = 2 * 12.011 / MW_TIO2
CARBON_PER_KG_SIO2 = 2 * 12.011 / MW_SIO2

# -- Thermal -----------------------------------------------------------------

REGOLITH_SPECIFIC_HEAT_J_KGK = 630.0
REGOLITH_MELTING_TEMP_K = 1400.0
LATENT_HEAT_MELTING_KJ_KG = 400.0
MARS_AMBIENT_TEMP_K = 210.0
STEFAN_BOLTZMANN = 5.670374419e-8
FURNACE_EMISSIVITY = 0.20
FURNACE_SURFACE_AREA_M2 = 0.8
COOLING_COEFF_KW_PER_K = 0.003

# -- Plasma arc (CO2 transferred arc, industrial scale) ----------------------

ARC_CONST_A = 200.0
ARC_CONST_B = 10.0
ARC_CONST_C = 20.0
ARC_CONST_D = 10.0
DEFAULT_ARC_LENGTH_CM = 5.0
DEFAULT_ARC_CURRENT_A = 800.0
MAX_ARC_CURRENT_A = 2000.0

# -- Electrode ---------------------------------------------------------------

ELECTRODE_DENSITY_KG_M3 = 1750.0
DEFAULT_ELECTRODE_DIAMETER_M = 0.10
DEFAULT_ELECTRODE_LENGTH_M = 1.00
ELECTRODE_WEAR_RATE_KG_PER_AH = 2.0e-5

# -- Operating defaults ------------------------------------------------------

DEFAULT_CHARGE_KG = 200.0
DEFAULT_POWER_KW = 250.0
TICK_DURATION_HOURS = 1.0
MIN_MELT_FRACTION = 0.95
REDUCTION_ACTIVATION_TEMP_K = 1200.0
ARRHENIUS_PREFACTOR = 0.10
ARRHENIUS_SCALE_K = 300.0


# -- Physics functions -------------------------------------------------------

def arc_voltage(current_a: float, arc_length_cm: float) -> float:
    """Ayrton equation: V = a + b*L + (c + d*L)/I."""
    if current_a <= 0.0 or arc_length_cm <= 0.0:
        return 0.0
    return (ARC_CONST_A + ARC_CONST_B * arc_length_cm +
            (ARC_CONST_C + ARC_CONST_D * arc_length_cm) / current_a)


def arc_power_kw(current_a: float, arc_length_cm: float) -> float:
    """Electrical power dissipated in the arc: P = V * I."""
    v = arc_voltage(current_a, arc_length_cm)
    return v * current_a / 1000.0


def radiation_loss_kw(temperature_k: float) -> float:
    """Stefan-Boltzmann radiation from furnace surface."""
    if temperature_k <= MARS_AMBIENT_TEMP_K:
        return 0.0
    q_w = (FURNACE_EMISSIVITY * STEFAN_BOLTZMANN * FURNACE_SURFACE_AREA_M2 *
           (temperature_k ** 4 - MARS_AMBIENT_TEMP_K ** 4))
    return q_w / 1000.0


def conduction_loss_kw(temperature_k: float) -> float:
    """Conductive heat loss through furnace walls."""
    if temperature_k <= MARS_AMBIENT_TEMP_K:
        return 0.0
    return COOLING_COEFF_KW_PER_K * (temperature_k - MARS_AMBIENT_TEMP_K)


def total_loss_kw(temperature_k: float) -> float:
    """Total thermal loss (radiation + conduction)."""
    return radiation_loss_kw(temperature_k) + conduction_loss_kw(temperature_k)


def heating_energy_kj(mass_kg: float, t_from: float, t_to: float) -> float:
    """Sensible heat + latent heat if crossing melting point."""
    if mass_kg <= 0.0 or t_to <= t_from:
        return 0.0
    sensible = mass_kg * REGOLITH_SPECIFIC_HEAT_J_KGK * (t_to - t_from) / 1000.0
    latent = 0.0
    if t_from < REGOLITH_MELTING_TEMP_K <= t_to:
        latent = mass_kg * LATENT_HEAT_MELTING_KJ_KG
    return sensible + latent


def reduction_rate(temperature_k: float) -> float:
    """Fraction of remaining oxide reduced per tick (Arrhenius)."""
    if temperature_k < REDUCTION_ACTIVATION_TEMP_K:
        return 0.0
    delta = temperature_k - REDUCTION_ACTIVATION_TEMP_K
    return ARRHENIUS_PREFACTOR * (1.0 - math.exp(-delta / ARRHENIUS_SCALE_K))


def reduction_energy_kj_per_kg(ore_kg: float) -> float:
    """Total endothermic energy to fully reduce all oxides in ore."""
    if ore_kg <= 0.0:
        return 0.0
    e_fe = ore_kg * REGOLITH_FE2O3_FRAC * ENTHALPY_FE2O3_KJ_KG
    e_al = ore_kg * REGOLITH_AL2O3_FRAC * ENTHALPY_AL2O3_KJ_KG
    e_ti = ore_kg * REGOLITH_TIO2_FRAC * ENTHALPY_TIO2_KJ_KG
    e_si = ore_kg * REGOLITH_SIO2_FRAC * ENTHALPY_SIO2_KJ_KG
    return e_fe + e_al + e_ti + e_si


def carbon_demand_kg(ore_kg: float) -> float:
    """Carbon needed to reduce all oxides stoichiometrically."""
    if ore_kg <= 0.0:
        return 0.0
    c_fe = ore_kg * REGOLITH_FE2O3_FRAC * CARBON_PER_KG_FE2O3
    c_al = ore_kg * REGOLITH_AL2O3_FRAC * CARBON_PER_KG_AL2O3
    c_ti = ore_kg * REGOLITH_TIO2_FRAC * CARBON_PER_KG_TIO2
    c_si = ore_kg * REGOLITH_SIO2_FRAC * CARBON_PER_KG_SIO2
    return c_fe + c_al + c_ti + c_si


def electrode_wear_kg(current_a: float, hours: float) -> float:
    """Graphite electrode consumption: proportional to amp-hours."""
    if current_a <= 0.0 or hours <= 0.0:
        return 0.0
    return ELECTRODE_WEAR_RATE_KG_PER_AH * current_a * hours


def electrode_volume_m3(diameter_m: float, length_m: float) -> float:
    """Cylindrical electrode volume."""
    if diameter_m <= 0.0 or length_m <= 0.0:
        return 0.0
    return math.pi * (diameter_m / 2.0) ** 2 * length_m


def electrode_mass_kg(diameter_m: float, length_m: float) -> float:
    """Electrode mass from geometry and density."""
    return electrode_volume_m3(diameter_m, length_m) * ELECTRODE_DENSITY_KG_M3


def max_metal_yield_kg(ore_kg: float) -> dict[str, float]:
    """Theoretical maximum metal extraction from a batch."""
    return {
        "iron_kg": ore_kg * MAX_FE_PER_KG,
        "aluminum_kg": ore_kg * MAX_AL_PER_KG,
        "titanium_kg": ore_kg * MAX_TI_PER_KG,
        "silicon_kg": ore_kg * MAX_SI_PER_KG,
    }


def co_produced_kg(ore_kg: float, reduction_frac: float) -> float:
    """CO gas produced by carbothermic reduction."""
    if ore_kg <= 0.0 or reduction_frac <= 0.0:
        return 0.0
    frac = min(reduction_frac, 1.0)
    c_used = carbon_demand_kg(ore_kg) * frac
    return c_used * 28.01 / 12.011


# -- Forge state -------------------------------------------------------------

@dataclass
class ForgeState:
    """Mutable state of the plasma arc forge."""
    ore_charge_kg: float = DEFAULT_CHARGE_KG
    ore_remaining_kg: float = DEFAULT_CHARGE_KG
    melt_fraction: float = 0.0
    reduction_fraction: float = 0.0
    temperature_k: float = MARS_AMBIENT_TEMP_K
    electrode_mass_kg: float = 0.0
    electrode_diameter_m: float = DEFAULT_ELECTRODE_DIAMETER_M
    electrode_length_m: float = DEFAULT_ELECTRODE_LENGTH_M
    arc_current_a: float = DEFAULT_ARC_CURRENT_A
    arc_length_cm: float = DEFAULT_ARC_LENGTH_CM
    iron_produced_kg: float = 0.0
    aluminum_produced_kg: float = 0.0
    titanium_produced_kg: float = 0.0
    silicon_produced_kg: float = 0.0
    slag_produced_kg: float = 0.0
    co_produced_kg: float = 0.0
    carbon_consumed_kg: float = 0.0
    total_energy_kwh: float = 0.0
    total_ticks: int = 0
    is_running: bool = False
    is_tapped: bool = False
    fault: str = ""

    def __post_init__(self) -> None:
        if self.electrode_mass_kg == 0.0:
            self.electrode_mass_kg = electrode_mass_kg(
                self.electrode_diameter_m, self.electrode_length_m)


def create_forge(
    charge_kg: float = DEFAULT_CHARGE_KG,
    arc_current_a: float = DEFAULT_ARC_CURRENT_A,
    arc_length_cm: float = DEFAULT_ARC_LENGTH_CM,
    electrode_diameter_m: float = DEFAULT_ELECTRODE_DIAMETER_M,
    electrode_length_m: float = DEFAULT_ELECTRODE_LENGTH_M,
    power_limit_kw: float = DEFAULT_POWER_KW,
) -> ForgeState:
    """Create a new forge with a fresh charge of regolith."""
    e_mass = electrode_mass_kg(electrode_diameter_m, electrode_length_m)
    actual_current = min(arc_current_a, MAX_ARC_CURRENT_A)
    return ForgeState(
        ore_charge_kg=max(charge_kg, 0.0),
        ore_remaining_kg=max(charge_kg, 0.0),
        electrode_mass_kg=e_mass,
        electrode_diameter_m=electrode_diameter_m,
        electrode_length_m=electrode_length_m,
        arc_current_a=actual_current,
        arc_length_cm=arc_length_cm,
    )


def ignite(state: ForgeState) -> ForgeState:
    """Start the plasma arc."""
    if state.is_tapped:
        state.fault = "already_tapped"
        return state
    if state.electrode_mass_kg <= 0.0:
        state.fault = "no_electrode"
        return state
    if state.ore_remaining_kg <= 0.0:
        state.fault = "no_charge"
        return state
    state.is_running = True
    state.fault = ""
    return state


def shutdown(state: ForgeState) -> ForgeState:
    """Shut down the arc."""
    state.is_running = False
    return state


def tick(state: ForgeState, power_limit_kw: float = DEFAULT_POWER_KW) -> ForgeState:
    """Advance the forge by one tick (1 hour)."""
    if not state.is_running:
        if state.temperature_k > MARS_AMBIENT_TEMP_K:
            loss = total_loss_kw(state.temperature_k) * TICK_DURATION_HOURS * 3600.0
            cap = max(state.ore_remaining_kg * REGOLITH_SPECIFIC_HEAT_J_KGK / 1000.0, 0.001)
            state.temperature_k = max(MARS_AMBIENT_TEMP_K, state.temperature_k - loss / cap)
        return state

    if state.is_tapped or state.fault:
        state.is_running = False
        return state

    p_arc = min(arc_power_kw(state.arc_current_a, state.arc_length_cm),
                max(power_limit_kw, 0.0))
    p_loss = total_loss_kw(state.temperature_k)
    p_net = max(p_arc - p_loss, 0.0)

    charge_mass = max(state.ore_remaining_kg, 0.001)
    energy_kj = p_net * TICK_DURATION_HOURS * 3600.0
    heat_cap = charge_mass * REGOLITH_SPECIFIC_HEAT_J_KGK / 1000.0

    if state.temperature_k < REGOLITH_MELTING_TEMP_K:
        e_to_melt = heat_cap * (REGOLITH_MELTING_TEMP_K - state.temperature_k)
        if energy_kj < e_to_melt:
            state.temperature_k += energy_kj / heat_cap
        else:
            rem = energy_kj - e_to_melt
            state.temperature_k = REGOLITH_MELTING_TEMP_K
            lat_total = charge_mass * LATENT_HEAT_MELTING_KJ_KG
            lat_needed = lat_total * (1.0 - state.melt_fraction)
            if rem >= lat_needed:
                state.melt_fraction = 1.0
                state.temperature_k += (rem - lat_needed) / heat_cap
            else:
                state.melt_fraction += rem / lat_total
                state.melt_fraction = min(state.melt_fraction, 1.0)
    else:
        if state.melt_fraction < 1.0:
            lat_total = charge_mass * LATENT_HEAT_MELTING_KJ_KG
            lat_needed = lat_total * (1.0 - state.melt_fraction)
            if energy_kj >= lat_needed:
                state.melt_fraction = 1.0
                state.temperature_k += (energy_kj - lat_needed) / heat_cap
            else:
                state.melt_fraction += energy_kj / lat_total
        else:
            state.temperature_k += energy_kj / heat_cap

    if state.melt_fraction >= MIN_MELT_FRACTION:
        rate = reduction_rate(state.temperature_k)
        remaining = 1.0 - state.reduction_fraction
        dr = rate * remaining
        state.reduction_fraction = min(state.reduction_fraction + dr, 1.0)
        state.carbon_consumed_kg += carbon_demand_kg(state.ore_charge_kg) * dr
        state.co_produced_kg += co_produced_kg(state.ore_charge_kg, dr)
        red_e = reduction_energy_kj_per_kg(state.ore_charge_kg) * dr
        state.temperature_k = max(REGOLITH_MELTING_TEMP_K,
                                  state.temperature_k - red_e / heat_cap)

    wear = electrode_wear_kg(state.arc_current_a, TICK_DURATION_HOURS)
    state.electrode_mass_kg = max(0.0, state.electrode_mass_kg - wear)
    if state.electrode_mass_kg <= 0.0:
        state.fault = "electrode_exhausted"
        state.is_running = False

    state.total_energy_kwh += p_arc * TICK_DURATION_HOURS
    state.total_ticks += 1
    return state


def tap(state: ForgeState) -> ForgeState:
    """Tap the furnace: extract metals and slag."""
    if state.is_tapped:
        return state
    if state.reduction_fraction <= 0.0:
        state.fault = "nothing_to_tap"
        return state
    state.is_running = False
    state.is_tapped = True
    y = max_metal_yield_kg(state.ore_charge_kg)
    rf = state.reduction_fraction
    state.iron_produced_kg = y["iron_kg"] * rf
    state.aluminum_produced_kg = y["aluminum_kg"] * rf
    state.titanium_produced_kg = y["titanium_kg"] * rf
    state.silicon_produced_kg = y["silicon_kg"] * rf
    total_metal = (state.iron_produced_kg + state.aluminum_produced_kg +
                   state.titanium_produced_kg + state.silicon_produced_kg)
    state.slag_produced_kg = max(0.0, state.ore_charge_kg - total_metal - state.co_produced_kg)
    state.ore_remaining_kg = 0.0
    return state


def efficiency(state: ForgeState) -> float:
    """Useful reduction energy / total electrical energy."""
    if state.total_energy_kwh <= 0.0:
        return 0.0
    total_metal = (state.iron_produced_kg + state.aluminum_produced_kg +
                   state.titanium_produced_kg + state.silicon_produced_kg)
    if total_metal <= 0.0:
        return 0.0
    useful_kj = reduction_energy_kj_per_kg(state.ore_charge_kg) * state.reduction_fraction
    total_kj = state.total_energy_kwh * 3600.0
    return min(useful_kj / total_kj, 1.0)


def energy_per_kg_metal(state: ForgeState) -> float:
    """kWh consumed per kg of metal produced."""
    total_metal = (state.iron_produced_kg + state.aluminum_produced_kg +
                   state.titanium_produced_kg + state.silicon_produced_kg)
    if total_metal <= 0.0:
        return float("inf")
    return state.total_energy_kwh / total_metal


def summary(state: ForgeState) -> dict:
    """Human-readable summary of forge state."""
    total_metal = (state.iron_produced_kg + state.aluminum_produced_kg +
                   state.titanium_produced_kg + state.silicon_produced_kg)
    return {
        "temperature_k": round(state.temperature_k, 1),
        "melt_fraction": round(state.melt_fraction, 4),
        "reduction_fraction": round(state.reduction_fraction, 4),
        "iron_kg": round(state.iron_produced_kg, 3),
        "aluminum_kg": round(state.aluminum_produced_kg, 3),
        "titanium_kg": round(state.titanium_produced_kg, 3),
        "silicon_kg": round(state.silicon_produced_kg, 3),
        "total_metal_kg": round(total_metal, 3),
        "slag_kg": round(state.slag_produced_kg, 3),
        "co_produced_kg": round(state.co_produced_kg, 3),
        "electrode_remaining_kg": round(state.electrode_mass_kg, 3),
        "carbon_consumed_kg": round(state.carbon_consumed_kg, 3),
        "energy_kwh": round(state.total_energy_kwh, 2),
        "ticks": state.total_ticks,
        "efficiency": round(efficiency(state), 4),
        "is_running": state.is_running,
        "is_tapped": state.is_tapped,
        "fault": state.fault,
    }


def run_batch(charge_kg: float = DEFAULT_CHARGE_KG,
              power_kw: float = DEFAULT_POWER_KW,
              max_ticks: int = 200) -> ForgeState:
    """Run a complete batch from ignition to tap."""
    forge = create_forge(charge_kg=charge_kg, power_limit_kw=power_kw)
    forge = ignite(forge)
    for _ in range(max_ticks):
        forge = tick(forge, power_limit_kw=power_kw)
        if forge.fault:
            break
        if forge.reduction_fraction >= 0.90:
            break
    forge = tap(forge)
    return forge
