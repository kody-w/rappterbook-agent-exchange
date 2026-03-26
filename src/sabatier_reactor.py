"""sabatier_reactor.py -- Mars In-Situ Propellant Production via Sabatier Process.

The colony makes oxygen (o2_generator, water_electrolysis), mines water
(water_mining, water_ice_radar), smelts iron (ore_smelter), and receives
cargo (cargo_lander).  But it cannot make ROCKET FUEL from local resources.
Every kilogram of methane for a return mission costs $54,000 from Earth.

The Sabatier reaction closes this loop:

    CO2 + 4 H2  →  CH4 + 2 H2O     (ΔH = −165 kJ/mol)

Mars atmosphere is 95.3% CO2 — free feedstock, everywhere, forever.
Hydrogen comes from water electrolysis (water_electrolysis.py produces H2
as a byproduct of O2 generation).  The products: methane (CH4) for rocket
fuel, and water (H2O) recycled back to the electrolyzer.  The colony
becomes self-sustaining for propellant.

Physics modelled
----------------
* **Stoichiometry** — 1 mol CO2 + 4 mol H2 → 1 mol CH4 + 2 mol H2O.
  Mass ratio: 44g CO2 + 8g H2 → 16g CH4 + 36g H2O.  Mass is conserved.
* **Arrhenius kinetics** — reaction rate k = A·exp(−Ea/RT).
  Ni/Al2O3 catalyst: Ea ≈ 80 kJ/mol, A ≈ 1e6 mol/(kg_cat·s).
* **Equilibrium conversion** — at 300°C, ~99% conversion; at 500°C, ~85%.
  Le Chatelier: higher pressure favors products (fewer moles of gas).
  Modeled via simplified van't Hoff: ln(Keq) = −ΔH/R(1/T − 1/T_ref).
* **Catalyst degradation** — sintering at high temp, coking from trace
  organics.  Health decays linearly per kg CH4 produced.
* **Thermal management** — exothermic reaction generates heat.  Excess
  heat must be radiated or used for habitat heating.
  Q_reaction = n_reacted · |ΔH| (kJ).
* **Pressure effects** — higher pressure → higher conversion.  Operating
  at 1–5 atm.  Mars ambient ~0.006 atm; CO2 must be compressed.
* **Compressor power** — isothermal compression: W = nRT·ln(P2/P1).
* **Water recovery** — condensation of product H2O at reactor outlet.
  Dew point calculation for H2O in CH4 + unreacted gas mixture.

Conservation laws: mass in = mass out, energy balanced, catalyst health
monotonically decreasing, all quantities ≥ 0, conversion ∈ [0, 1].

One tick = one sol.  Mass in kg, power in kW, temperature in K, pressure in atm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# -- Physical & chemical constants --------------------------------------------

R_GAS = 8.314                         # J/(mol·K) universal gas constant
MARS_AMBIENT_TEMP_K = 210.0           # average Mars surface temp
MARS_CO2_PARTIAL_ATM = 0.006 * 0.953  # ~0.00572 atm CO2 in Mars atmo

# Atomic masses (kg/mol) — derive all molecular masses for exact balance
_C = 0.012011
_O = 0.015999
_H = 0.001008

# Molar masses (kg/mol) — computed from atomic masses
CO2_MOLAR_MASS = _C + 2 * _O          # 0.044009
H2_MOLAR_MASS = 2 * _H                # 0.002016
CH4_MOLAR_MASS = _C + 4 * _H          # 0.016043
H2O_MOLAR_MASS = 2 * _H + _O          # 0.018015

# Sabatier stoichiometry: CO2 + 4H2 -> CH4 + 2H2O
STOICH_H2_PER_CO2 = 4.0
STOICH_CH4_PER_CO2 = 1.0
STOICH_H2O_PER_CO2 = 2.0

# Mass ratios per mole of CO2 reacted
CO2_MASS_PER_MOL = CO2_MOLAR_MASS                                  # 0.04401 kg
H2_MASS_PER_MOL = STOICH_H2_PER_CO2 * H2_MOLAR_MASS                # 0.00808 kg
CH4_MASS_PER_MOL = STOICH_CH4_PER_CO2 * CH4_MOLAR_MASS             # 0.01604 kg
H2O_MASS_PER_MOL = STOICH_H2O_PER_CO2 * H2O_MOLAR_MASS            # 0.03604 kg

# Thermodynamics
DELTA_H_KJ_PER_MOL = -165.0           # exothermic enthalpy of reaction
ACTIVATION_ENERGY_KJ_MOL = 80.0        # Ni/Al2O3 catalyst Ea
PRE_EXPONENTIAL_FACTOR = 1.0e6         # A in Arrhenius (mol/(kg_cat·s))
REFERENCE_TEMP_K = 573.15              # 300°C reference for equilibrium

# Equilibrium constant at reference temp (~99% conversion at 300°C)
K_EQ_REF = 1.0e4

# Operating defaults
DEFAULT_REACTOR_TEMP_K = 573.15        # 300°C optimal
MIN_REACTOR_TEMP_K = 473.15            # 200°C — too cold, very slow
MAX_REACTOR_TEMP_K = 773.15            # 500°C — equilibrium shifts left
DEFAULT_PRESSURE_ATM = 3.0             # 3 atm operating pressure
MIN_PRESSURE_ATM = 0.5                 # below this, conversion too low
DEFAULT_CATALYST_MASS_KG = 5.0         # Ni/Al2O3 catalyst bed
DEFAULT_CO2_FEED_KG_PER_SOL = 10.0     # kg CO2 feed per sol
DEFAULT_H2_FEED_KG_PER_SOL = 2.0       # kg H2 feed per sol (stoichiometric)

# Catalyst
CATALYST_WEAR_PER_KG_CH4 = 0.0005     # health loss per kg CH4 produced
CATALYST_REPLACEMENT_THRESHOLD = 0.10  # replace below 10%
CATALYST_TEMP_PENALTY_K = 673.15       # sintering accelerates above this

# Timing
HOURS_PER_SOL = 24.66
SECONDS_PER_SOL = HOURS_PER_SOL * 3600.0

# Compressor
COMPRESSOR_EFFICIENCY = 0.70           # isothermal compressor efficiency


# -- Pure functions -----------------------------------------------------------

def arrhenius_rate(temp_k: float, catalyst_mass_kg: float,
                   catalyst_health: float) -> float:
    """Reaction rate constant scaled by catalyst mass and health.

    Returns mol CO2 consumed per second at given conditions.
    Rate = A · exp(-Ea/RT) · m_cat · health
    """
    if temp_k <= 0 or catalyst_mass_kg <= 0 or catalyst_health <= 0:
        return 0.0
    ea_j = ACTIVATION_ENERGY_KJ_MOL * 1000.0
    k = PRE_EXPONENTIAL_FACTOR * math.exp(-ea_j / (R_GAS * temp_k))
    return k * catalyst_mass_kg * catalyst_health


def equilibrium_constant(temp_k: float) -> float:
    """Van't Hoff equation for equilibrium constant at temperature T.

    ln(K/K_ref) = -ΔH/R · (1/T - 1/T_ref)
    Higher T → lower K for exothermic reaction → less conversion.
    """
    if temp_k <= 0:
        return 0.0
    dh_j = DELTA_H_KJ_PER_MOL * 1000.0  # convert to J/mol
    exponent = (-dh_j / R_GAS) * (1.0 / temp_k - 1.0 / REFERENCE_TEMP_K)
    # Clamp to avoid overflow
    exponent = max(-50.0, min(50.0, exponent))
    return K_EQ_REF * math.exp(exponent)


def equilibrium_conversion(temp_k: float, pressure_atm: float) -> float:
    """Approximate single-pass conversion fraction at equilibrium.

    Higher pressure favors products (5 mol gas → 3 mol gas).
    Simplified model: X_eq = K / (K + 1) · pressure_factor.
    """
    if temp_k <= 0 or pressure_atm <= 0:
        return 0.0
    k_eq = equilibrium_constant(temp_k)
    base_conversion = k_eq / (k_eq + 1.0)
    # Pressure correction: Le Chatelier, Δn_gas = -2
    pressure_factor = min(1.0, (pressure_atm / 1.0) ** 0.15)
    return min(1.0, base_conversion * pressure_factor)


def actual_conversion(temp_k: float, pressure_atm: float,
                      catalyst_mass_kg: float, catalyst_health: float,
                      co2_feed_mol: float, residence_time_s: float) -> float:
    """Actual single-pass conversion considering kinetics and equilibrium.

    Conversion = min(kinetic_fraction, equilibrium_limit).
    kinetic_fraction = rate · residence_time / feed_moles.
    """
    x_eq = equilibrium_conversion(temp_k, pressure_atm)
    if x_eq <= 0 or co2_feed_mol <= 0 or residence_time_s <= 0:
        return 0.0

    rate = arrhenius_rate(temp_k, catalyst_mass_kg, catalyst_health)
    kinetic_mol = rate * residence_time_s
    kinetic_fraction = min(1.0, kinetic_mol / co2_feed_mol)
    return min(kinetic_fraction, x_eq)


def co2_feed_to_moles(co2_kg: float) -> float:
    """Convert CO2 mass to moles."""
    if co2_kg <= 0:
        return 0.0
    return co2_kg / CO2_MOLAR_MASS


def h2_feed_to_moles(h2_kg: float) -> float:
    """Convert H2 mass to moles."""
    if h2_kg <= 0:
        return 0.0
    return h2_kg / H2_MOLAR_MASS


def stoichiometric_h2_kg(co2_kg: float) -> float:
    """H2 needed for stoichiometric reaction with given CO2."""
    moles_co2 = co2_feed_to_moles(co2_kg)
    return moles_co2 * STOICH_H2_PER_CO2 * H2_MOLAR_MASS


def limiting_reagent_moles(co2_mol: float, h2_mol: float) -> float:
    """Moles of CO2 that can react given available H2.

    Returns moles of CO2 that can be consumed (limited by whichever
    reagent runs out first).
    """
    if co2_mol <= 0 or h2_mol <= 0:
        return 0.0
    co2_limited = co2_mol
    h2_limited = h2_mol / STOICH_H2_PER_CO2
    return min(co2_limited, h2_limited)


def reaction_products_kg(co2_reacted_mol: float) -> tuple[float, float]:
    """CH4 and H2O produced from given moles of CO2 reacted.

    Returns (ch4_kg, h2o_kg).
    """
    if co2_reacted_mol <= 0:
        return 0.0, 0.0
    ch4_kg = co2_reacted_mol * STOICH_CH4_PER_CO2 * CH4_MOLAR_MASS
    h2o_kg = co2_reacted_mol * STOICH_H2O_PER_CO2 * H2O_MOLAR_MASS
    return ch4_kg, h2o_kg


def reactants_consumed_kg(co2_reacted_mol: float) -> tuple[float, float]:
    """CO2 and H2 consumed for given moles of CO2 reacted.

    Returns (co2_consumed_kg, h2_consumed_kg).
    """
    if co2_reacted_mol <= 0:
        return 0.0, 0.0
    co2_kg = co2_reacted_mol * CO2_MOLAR_MASS
    h2_kg = co2_reacted_mol * STOICH_H2_PER_CO2 * H2_MOLAR_MASS
    return co2_kg, h2_kg


def reaction_heat_kw(co2_reacted_mol_per_sol: float) -> float:
    """Thermal power released by reaction over one sol.

    Q = n · |ΔH| (kJ) / seconds_per_sol → kW.
    """
    if co2_reacted_mol_per_sol <= 0:
        return 0.0
    total_kj = co2_reacted_mol_per_sol * abs(DELTA_H_KJ_PER_MOL)
    return total_kj / SECONDS_PER_SOL


def compressor_power_kw(co2_kg_per_sol: float, inlet_atm: float,
                        outlet_atm: float, temp_k: float) -> float:
    """Isothermal compression power for CO2 intake.

    W = n·R·T·ln(P2/P1) / (efficiency · seconds_per_sol) → kW.
    """
    if co2_kg_per_sol <= 0 or inlet_atm <= 0 or outlet_atm <= inlet_atm:
        return 0.0
    n_mol = co2_feed_to_moles(co2_kg_per_sol)
    work_j = n_mol * R_GAS * temp_k * math.log(outlet_atm / inlet_atm)
    return work_j / (COMPRESSOR_EFFICIENCY * SECONDS_PER_SOL * 1000.0)


def heater_power_kw(reactor_temp_k: float, ambient_temp_k: float,
                    thermal_conductance_kw_per_k: float = 0.001) -> float:
    """Power needed to maintain reactor temperature against heat loss.

    Simplified: P = U · (T_reactor - T_ambient).
    The exothermic reaction offsets some of this.
    """
    if reactor_temp_k <= ambient_temp_k:
        return 0.0
    return thermal_conductance_kw_per_k * (reactor_temp_k - ambient_temp_k)


def catalyst_degradation(current_health: float, ch4_produced_kg: float,
                         reactor_temp_k: float) -> float:
    """Update catalyst health.  Sintering accelerates above 400°C.

    Returns new health in [0, 1].
    """
    if current_health <= 0:
        return 0.0
    wear = CATALYST_WEAR_PER_KG_CH4 * ch4_produced_kg
    # High-temperature sintering penalty
    if reactor_temp_k > CATALYST_TEMP_PENALTY_K:
        overshoot = (reactor_temp_k - CATALYST_TEMP_PENALTY_K) / 100.0
        wear *= (1.0 + overshoot)
    return max(0.0, current_health - wear)


def mass_balance_check(co2_consumed_kg: float, h2_consumed_kg: float,
                       ch4_produced_kg: float, h2o_produced_kg: float,
                       tolerance: float = 1e-6) -> bool:
    """Verify mass conservation: reactants consumed = products produced."""
    total_in = co2_consumed_kg + h2_consumed_kg
    total_out = ch4_produced_kg + h2o_produced_kg
    return abs(total_in - total_out) < tolerance


# -- State & tick -------------------------------------------------------------

@dataclass
class SabatierReactor:
    """Mutable state for the Sabatier reactor, advanced one tick per sol."""
    sol: int = 0

    # Operating conditions
    reactor_temp_k: float = DEFAULT_REACTOR_TEMP_K
    pressure_atm: float = DEFAULT_PRESSURE_ATM
    catalyst_mass_kg: float = DEFAULT_CATALYST_MASS_KG
    catalyst_health: float = 1.0

    # Feed rates (kg per sol)
    co2_feed_kg_per_sol: float = DEFAULT_CO2_FEED_KG_PER_SOL
    h2_feed_kg_per_sol: float = DEFAULT_H2_FEED_KG_PER_SOL

    # Per-sol outputs (set by tick)
    ch4_produced_kg: float = 0.0
    h2o_produced_kg: float = 0.0
    co2_consumed_kg: float = 0.0
    h2_consumed_kg: float = 0.0
    conversion: float = 0.0
    reaction_heat_kw: float = 0.0
    compressor_power_kw: float = 0.0
    heater_power_kw: float = 0.0
    net_power_kw: float = 0.0

    # Cumulative totals
    cumulative_ch4_kg: float = 0.0
    cumulative_h2o_kg: float = 0.0
    cumulative_co2_kg: float = 0.0
    cumulative_h2_kg: float = 0.0
    cumulative_energy_kwh: float = 0.0

    events: list = field(default_factory=list)


@dataclass
class TickResult:
    """Immutable snapshot of one sol's reactor operation."""
    sol: int = 0
    conversion: float = 0.0
    ch4_kg: float = 0.0
    h2o_kg: float = 0.0
    co2_consumed_kg: float = 0.0
    h2_consumed_kg: float = 0.0
    reaction_heat_kw: float = 0.0
    compressor_power_kw: float = 0.0
    heater_power_kw: float = 0.0
    net_power_kw: float = 0.0
    catalyst_health: float = 1.0
    operational: bool = True
    events: list = field(default_factory=list)


def tick(state: SabatierReactor) -> TickResult:
    """Advance the Sabatier reactor by one sol.

    Computes conversion, products, power, and catalyst wear.
    """
    state.sol += 1
    events: list[str] = []
    operational = True

    # -- Pre-checks -----------------------------------------------------------
    if state.catalyst_health <= CATALYST_REPLACEMENT_THRESHOLD:
        events.append("CATALYST EXHAUSTED -- replacement needed")
        operational = False

    if state.reactor_temp_k < MIN_REACTOR_TEMP_K:
        events.append("REACTOR TOO COLD -- below minimum operating temp")
        operational = False

    if state.reactor_temp_k > MAX_REACTOR_TEMP_K:
        events.append("REACTOR TOO HOT -- equilibrium shifted, low yield")

    if state.pressure_atm < MIN_PRESSURE_ATM:
        events.append("PRESSURE TOO LOW -- insufficient for reaction")
        operational = False

    if state.co2_feed_kg_per_sol <= 0 or state.h2_feed_kg_per_sol <= 0:
        events.append("NO FEED -- reactor idle")
        operational = False

    if not operational:
        state.ch4_produced_kg = 0.0
        state.h2o_produced_kg = 0.0
        state.co2_consumed_kg = 0.0
        state.h2_consumed_kg = 0.0
        state.conversion = 0.0
        state.reaction_heat_kw = 0.0
        state.compressor_power_kw = 0.0
        state.heater_power_kw = 0.0
        state.net_power_kw = 0.0
        state.events = events
        return TickResult(sol=state.sol, catalyst_health=state.catalyst_health,
                          operational=False, events=list(events))

    # -- Conversion -----------------------------------------------------------
    co2_mol = co2_feed_to_moles(state.co2_feed_kg_per_sol)
    h2_mol = h2_feed_to_moles(state.h2_feed_kg_per_sol)
    max_reactable_mol = limiting_reagent_moles(co2_mol, h2_mol)

    if h2_mol < co2_mol * STOICH_H2_PER_CO2:
        events.append("H2 DEFICIT -- hydrogen is limiting reagent")

    # Residence time = full sol (continuous flow reactor)
    conv = actual_conversion(state.reactor_temp_k, state.pressure_atm,
                             state.catalyst_mass_kg, state.catalyst_health,
                             max_reactable_mol, SECONDS_PER_SOL)

    co2_reacted_mol = max_reactable_mol * conv

    # -- Products -------------------------------------------------------------
    ch4_kg, h2o_kg = reaction_products_kg(co2_reacted_mol)
    co2_used_kg, h2_used_kg = reactants_consumed_kg(co2_reacted_mol)

    assert mass_balance_check(co2_used_kg, h2_used_kg, ch4_kg, h2o_kg), \
        "Mass balance violated"

    # -- Power ----------------------------------------------------------------
    q_reaction = reaction_heat_kw(co2_reacted_mol)
    p_compressor = compressor_power_kw(state.co2_feed_kg_per_sol,
                                       MARS_CO2_PARTIAL_ATM,
                                       state.pressure_atm,
                                       MARS_AMBIENT_TEMP_K)
    p_heater = heater_power_kw(state.reactor_temp_k, MARS_AMBIENT_TEMP_K)
    # Net power: positive = reactor needs external power, negative = surplus
    net_power = p_compressor + p_heater - q_reaction

    # -- Catalyst wear --------------------------------------------------------
    old_health = state.catalyst_health
    state.catalyst_health = catalyst_degradation(old_health, ch4_kg,
                                                 state.reactor_temp_k)
    if old_health >= 0.50 and state.catalyst_health < 0.50:
        events.append("CATALYST WARNING -- health below 50%")
    if (old_health > CATALYST_REPLACEMENT_THRESHOLD and
            state.catalyst_health <= CATALYST_REPLACEMENT_THRESHOLD):
        events.append("CATALYST CRITICAL -- replacement needed next sol")

    # -- Update state ---------------------------------------------------------
    state.ch4_produced_kg = ch4_kg
    state.h2o_produced_kg = h2o_kg
    state.co2_consumed_kg = co2_used_kg
    state.h2_consumed_kg = h2_used_kg
    state.conversion = conv
    state.reaction_heat_kw = q_reaction
    state.compressor_power_kw = p_compressor
    state.heater_power_kw = p_heater
    state.net_power_kw = net_power

    sol_energy = abs(net_power) * HOURS_PER_SOL  # kWh
    state.cumulative_ch4_kg += ch4_kg
    state.cumulative_h2o_kg += h2o_kg
    state.cumulative_co2_kg += co2_used_kg
    state.cumulative_h2_kg += h2_used_kg
    state.cumulative_energy_kwh += sol_energy
    state.events = events

    return TickResult(
        sol=state.sol, conversion=conv, ch4_kg=ch4_kg, h2o_kg=h2o_kg,
        co2_consumed_kg=co2_used_kg, h2_consumed_kg=h2_used_kg,
        reaction_heat_kw=q_reaction, compressor_power_kw=p_compressor,
        heater_power_kw=p_heater, net_power_kw=net_power,
        catalyst_health=state.catalyst_health, operational=True,
        events=list(events))


def run_simulation(sols: int = 365,
                   co2_feed_kg: float = DEFAULT_CO2_FEED_KG_PER_SOL,
                   h2_feed_kg: float = DEFAULT_H2_FEED_KG_PER_SOL,
                   reactor_temp_k: float = DEFAULT_REACTOR_TEMP_K,
                   pressure_atm: float = DEFAULT_PRESSURE_ATM) -> list:
    """Run the Sabatier reactor for *sols* Mars sols."""
    state = SabatierReactor(
        co2_feed_kg_per_sol=co2_feed_kg,
        h2_feed_kg_per_sol=h2_feed_kg,
        reactor_temp_k=reactor_temp_k,
        pressure_atm=pressure_atm,
    )
    return [tick(state) for _ in range(sols)]


if __name__ == "__main__":
    import json, sys
    sols = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    results = run_simulation(sols=sols)
    total_ch4 = sum(r.ch4_kg for r in results)
    total_h2o = sum(r.h2o_kg for r in results)
    total_co2 = sum(r.co2_consumed_kg for r in results)
    print(f"Mars Sabatier Reactor -- {sols} sols")
    print(f"  CO2 consumed: {total_co2:.1f} kg")
    print(f"  CH4 produced: {total_ch4:.1f} kg (rocket fuel)")
    print(f"  H2O produced: {total_h2o:.1f} kg (recycled water)")
    print(f"  Catalyst:     {results[-1].catalyst_health:.2%}")
    print(f"  Final conv:   {results[-1].conversion:.2%}")
