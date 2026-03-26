"""
atmo_processor.py — Mars In-Situ Resource Utilization (ISRU) atmospheric processor.

Models three coupled chemical processes that keep a Mars colony alive:

1. MOXIE (Solid Oxide CO₂ Electrolysis):  CO₂ → CO + ½O₂
   Primary oxygen source.  Perseverance demonstrated this at 6-10 g/hr.
   Industrial scale: 2-5 kg O₂/sol per unit.

2. Sabatier Reactor:  CO₂ + 4H₂ → CH₄ + 2H₂O
   Produces methane propellant + bonus water from CO₂ and hydrogen.
   Exothermic — produces usable waste heat.

3. Water Electrolysis:  2H₂O → 2H₂ + O₂
   Splits water into hydrogen (Sabatier feedstock) and oxygen.
   Closes the loop: Sabatier water → electrolysis → H₂ back to Sabatier.

Conservation of mass is enforced at every step.  Atom counts (C, O, H) balance.

Physical references:
  - Human O₂ consumption: 0.84 kg/person/sol (NASA HRP)
  - Mars atmosphere: 95.3% CO₂ at 0.636 kPa (unlimited feedstock)
  - MOXIE (Perseverance): 6-10 g O₂/hr, ~300 W
  - PEM electrolysis: ~5.3 kWh per kg H₂O
  - Sabatier reaction: -165 kJ/mol (exothermic)
  - NASA DRA 5.0 MAV propellant target: ~33 tonnes LOX+CH₄

One tick = one sol.  Mass in kg.  Energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Human oxygen needs (NASA Human Research Program)
O2_KG_PER_PERSON_SOL = 0.84

# --- MOXIE: CO₂ → CO + ½O₂ ---
# Stoichiometry (molar masses): CO₂=44, CO=28, O₂=32
# Per mol CO₂: 1 mol CO + 0.5 mol O₂
# Mass ratio per kg CO₂ input:
MOXIE_O2_PER_KG_CO2 = 16.0 / 44.0    # 0.3636 kg O₂ per kg CO₂
MOXIE_CO_PER_KG_CO2 = 28.0 / 44.0    # 0.6364 kg CO per kg CO₂
MOXIE_KWH_PER_KG_O2 = 25.0           # electrical energy cost
MOXIE_RATED_KG_O2_SOL = 5.0          # industrial MOXIE: 5 kg O₂/sol per unit

# --- Sabatier: CO₂ + 4H₂ → CH₄ + 2H₂O ---
# Molar: 44 + 8 → 16 + 36
# Per kg CO₂ input:
SABATIER_H2_PER_KG_CO2 = 8.0 / 44.0    # 0.1818 kg H₂ consumed
SABATIER_CH4_PER_KG_CO2 = 16.0 / 44.0  # 0.3636 kg CH₄ produced
SABATIER_H2O_PER_KG_CO2 = 36.0 / 44.0  # 0.8182 kg H₂O produced
SABATIER_HEAT_KJ_PER_KG_CO2 = 165.0 * 1000.0 / 44.0  # exothermic heat per kg CO₂

# --- Water Electrolysis: 2H₂O → 2H₂ + O₂ ---
# Molar: 36 → 4 + 32
# Per kg H₂O input:
ELECTROLYSIS_H2_PER_KG_H2O = 4.0 / 36.0    # 0.1111 kg H₂
ELECTROLYSIS_O2_PER_KG_H2O = 32.0 / 36.0   # 0.8889 kg O₂
ELECTROLYSIS_KWH_PER_KG_H2O = 5.3           # PEM electrolysis energy cost

# Dust filter degradation
DUST_FILTER_CLOG_RATE = 0.003    # per sol at dust_opacity=1.0
FILTER_MAINTENANCE_CLEAR = 0.5   # maintenance removes this fraction of clogging

# Temperature efficiency (SOEC operates at ~800°C; cold ambient = more startup energy)
COLD_PENALTY_FLOOR = 0.70        # minimum efficiency at -120°C
WARM_EFFICIENCY_CEIL = 1.0       # full efficiency at 0°C and above

# Propellant target (NASA Design Reference Architecture 5.0)
MAV_PROPELLANT_TARGET_KG = 33000.0  # LOX + CH₄ for Mars Ascent Vehicle


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MoxieBank:
    """Bank of industrial MOXIE units (Solid Oxide CO₂ Electrolysis).

    Attributes:
        units: number of active MOXIE units
        filter_clog: dust clogging fraction [0, 1] — reduces throughput
        age_sols: total operational sols (for tracking)
    """
    units: int = 1
    filter_clog: float = 0.0
    age_sols: int = 0

    def __post_init__(self) -> None:
        self.units = max(0, self.units)
        self.filter_clog = max(0.0, min(1.0, self.filter_clog))
        self.age_sols = max(0, self.age_sols)


@dataclass
class SabatierBank:
    """Bank of Sabatier methanation reactors.

    Attributes:
        units: number of active reactors
        rated_kg_co2_sol: CO₂ throughput per reactor per sol
    """
    units: int = 0
    rated_kg_co2_sol: float = 10.0

    def __post_init__(self) -> None:
        self.units = max(0, self.units)
        self.rated_kg_co2_sol = max(0.0, self.rated_kg_co2_sol)


@dataclass
class AtmoState:
    """Atmospheric processing state for one colony.

    Tracks O₂ reserves, propellant stockpile, and cumulative production.
    """
    o2_buffer_kg: float = 0.0
    ch4_stockpile_kg: float = 0.0
    h2_stockpile_kg: float = 0.0
    total_o2_produced_kg: float = 0.0
    total_ch4_produced_kg: float = 0.0
    total_h2o_produced_kg: float = 0.0
    total_power_consumed_kwh: float = 0.0
    deficit_sols: int = 0

    def __post_init__(self) -> None:
        self.o2_buffer_kg = max(0.0, self.o2_buffer_kg)
        self.ch4_stockpile_kg = max(0.0, self.ch4_stockpile_kg)
        self.h2_stockpile_kg = max(0.0, self.h2_stockpile_kg)


@dataclass
class TickResult:
    """Result of one sol of atmospheric processing.

    All fields in kg or kWh.  Mass conservation is guaranteed:
    total input mass == total output mass for each reaction.
    """
    o2_produced_kg: float = 0.0
    co_produced_kg: float = 0.0
    ch4_produced_kg: float = 0.0
    h2o_produced_kg: float = 0.0
    h2_produced_kg: float = 0.0
    o2_consumed_kg: float = 0.0
    h2_consumed_kg: float = 0.0
    h2o_consumed_kg: float = 0.0
    co2_consumed_kg: float = 0.0
    power_consumed_kwh: float = 0.0
    sabatier_heat_kj: float = 0.0
    o2_deficit_kg: float = 0.0
    mav_progress: float = 0.0


# ---------------------------------------------------------------------------
# Pure physics functions
# ---------------------------------------------------------------------------

def temperature_efficiency(temp_c: float) -> float:
    """ISRU efficiency factor from ambient temperature.

    SOEC cells operate at ~800°C internally; colder ambient means more
    energy wasted on thermal management.  Linear interpolation.

    Returns a factor in [COLD_PENALTY_FLOOR, WARM_EFFICIENCY_CEIL].
    """
    t = max(-120.0, min(0.0, temp_c))
    frac = (t + 120.0) / 120.0
    return COLD_PENALTY_FLOOR + frac * (WARM_EFFICIENCY_CEIL - COLD_PENALTY_FLOOR)


def dust_throughput_factor(filter_clog: float) -> float:
    """Throughput reduction from dust clogging intake filters.

    Quadratic: light dust has little effect, heavy clogging is catastrophic.
    """
    c = max(0.0, min(1.0, filter_clog))
    return max(0.0, 1.0 - c * c)


def moxie_output(
    bank: MoxieBank,
    power_available_kwh: float,
    temp_c: float,
) -> tuple[float, float, float, float]:
    """Compute one sol of MOXIE production.

    Args:
        bank: MOXIE unit bank state
        power_available_kwh: power budget allocated to MOXIE
        temp_c: ambient surface temperature

    Returns:
        (o2_kg, co_kg, co2_consumed_kg, power_consumed_kwh)

    Conservation: co2_consumed = o2 / MOXIE_O2_PER_KG_CO2
                  co = co2_consumed * MOXIE_CO_PER_KG_CO2
                  co2_consumed = o2 + co  (mass balance)
    """
    if bank.units <= 0 or power_available_kwh <= 0:
        return (0.0, 0.0, 0.0, 0.0)

    eff_temp = temperature_efficiency(temp_c)
    eff_dust = dust_throughput_factor(bank.filter_clog)
    effective_rate = MOXIE_RATED_KG_O2_SOL * bank.units * eff_temp * eff_dust

    # Power-limited: can't produce more O₂ than power allows
    power_limited_o2 = power_available_kwh / MOXIE_KWH_PER_KG_O2
    o2_kg = min(effective_rate, power_limited_o2)
    o2_kg = max(0.0, o2_kg)

    # Mass balance
    co2_consumed = o2_kg / MOXIE_O2_PER_KG_CO2
    co_kg = co2_consumed * MOXIE_CO_PER_KG_CO2
    power_consumed = o2_kg * MOXIE_KWH_PER_KG_O2

    return (o2_kg, co_kg, co2_consumed, power_consumed)


def sabatier_output(
    bank: SabatierBank,
    h2_available_kg: float,
    temp_c: float,
) -> tuple[float, float, float, float, float]:
    """Compute one sol of Sabatier reactor production.

    Args:
        bank: Sabatier reactor bank state
        h2_available_kg: hydrogen feedstock available
        temp_c: ambient temperature (affects catalyst efficiency)

    Returns:
        (ch4_kg, h2o_kg, co2_consumed_kg, h2_consumed_kg, heat_kj)

    Conservation: co2_consumed + h2_consumed == ch4 + h2o
    """
    if bank.units <= 0 or h2_available_kg <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    eff_temp = temperature_efficiency(temp_c)
    max_co2 = bank.rated_kg_co2_sol * bank.units * eff_temp

    # H₂-limited: each kg CO₂ needs SABATIER_H2_PER_KG_CO2 kg H₂
    h2_limited_co2 = h2_available_kg / SABATIER_H2_PER_KG_CO2
    co2_consumed = min(max_co2, h2_limited_co2)
    co2_consumed = max(0.0, co2_consumed)

    h2_consumed = co2_consumed * SABATIER_H2_PER_KG_CO2
    ch4_kg = co2_consumed * SABATIER_CH4_PER_KG_CO2
    h2o_kg = co2_consumed * SABATIER_H2O_PER_KG_CO2
    heat_kj = co2_consumed * SABATIER_HEAT_KJ_PER_KG_CO2

    return (ch4_kg, h2o_kg, co2_consumed, h2_consumed, heat_kj)


def electrolyze_water(
    h2o_available_kg: float,
    power_available_kwh: float,
    target_h2_kg: float | None = None,
) -> tuple[float, float, float, float]:
    """Electrolyze water into hydrogen and oxygen.

    Args:
        h2o_available_kg: water available for electrolysis
        power_available_kwh: power budget for electrolysis
        target_h2_kg: optional cap on H₂ production (for Sabatier feedstock)

    Returns:
        (h2_kg, o2_kg, h2o_consumed_kg, power_consumed_kwh)

    Conservation: h2o_consumed == h2 + o2
    """
    if h2o_available_kg <= 0 or power_available_kwh <= 0:
        return (0.0, 0.0, 0.0, 0.0)

    # Power-limited water throughput
    power_limited_h2o = power_available_kwh / ELECTROLYSIS_KWH_PER_KG_H2O
    h2o_consumed = min(h2o_available_kg, power_limited_h2o)

    h2_kg = h2o_consumed * ELECTROLYSIS_H2_PER_KG_H2O
    o2_kg = h2o_consumed * ELECTROLYSIS_O2_PER_KG_H2O

    # Cap to target H₂ if specified
    if target_h2_kg is not None and target_h2_kg >= 0 and h2_kg > target_h2_kg:
        ratio = target_h2_kg / h2_kg if h2_kg > 0 else 0.0
        h2_kg *= ratio
        o2_kg *= ratio
        h2o_consumed *= ratio

    power_consumed = h2o_consumed * ELECTROLYSIS_KWH_PER_KG_H2O

    return (h2_kg, o2_kg, h2o_consumed, power_consumed)


def colony_o2_demand(population: int) -> float:
    """Total O₂ demand for the colony in kg/sol."""
    return max(0, population) * O2_KG_PER_PERSON_SOL


def mav_progress_fraction(ch4_kg: float) -> float:
    """Fraction of Mars Ascent Vehicle propellant target achieved.

    LOX is produced alongside CH₄ via MOXIE, so CH₄ is the bottleneck.
    MAV needs ~7.5 tonnes CH₄ + ~25.5 tonnes LOX = 33 tonnes total.
    CH₄ fraction of target: 7500 kg.
    """
    ch4_target = MAV_PROPELLANT_TARGET_KG * (16.0 / (16.0 + 32.0 * 2))
    # Simplified: ~33% CH₄ by mass (stoichiometric O₂:CH₄ ratio ~3.4:1)
    ch4_target = MAV_PROPELLANT_TARGET_KG / 4.4  # ~7500 kg CH₄
    return min(1.0, max(0.0, ch4_kg / ch4_target)) if ch4_target > 0 else 0.0


# ---------------------------------------------------------------------------
# Tick function — one sol of atmospheric processing
# ---------------------------------------------------------------------------

def tick_atmo(
    moxie: MoxieBank,
    sabatier: SabatierBank,
    state: AtmoState,
    population: int,
    power_budget_kwh: float,
    water_budget_kg: float,
    temp_c: float,
    dust_opacity: float,
) -> TickResult:
    """Advance atmospheric processing by one sol.

    Orchestrates MOXIE, Sabatier, and electrolysis to produce O₂ for
    breathing and CH₄ for propellant.  Power and water are consumed.
    State is mutated in-place.

    Priority order:
      1. MOXIE runs first (primary O₂ source, highest priority)
      2. Electrolysis runs second (produces H₂ for Sabatier + bonus O₂)
      3. Sabatier runs third (uses H₂ to make CH₄ + H₂O)

    Args:
        moxie: MOXIE unit bank
        sabatier: Sabatier reactor bank
        state: atmospheric processing state (mutated)
        population: colony population (drives O₂ demand)
        power_budget_kwh: total power allocated to ISRU this sol
        water_budget_kg: water available for electrolysis
        temp_c: surface temperature
        dust_opacity: dust storm opacity [0, 1]

    Returns:
        TickResult with all production/consumption totals
    """
    result = TickResult()

    # --- Update dust filter clogging ---
    moxie.filter_clog += dust_opacity * DUST_FILTER_CLOG_RATE
    moxie.filter_clog = min(1.0, moxie.filter_clog)
    moxie.age_sols += 1

    # --- 1. MOXIE: CO₂ → O₂ + CO ---
    remaining_power = max(0.0, power_budget_kwh)
    o2_moxie, co_kg, co2_moxie, pwr_moxie = moxie_output(
        moxie, remaining_power, temp_c,
    )
    remaining_power -= pwr_moxie
    result.o2_produced_kg += o2_moxie
    result.co_produced_kg = co_kg
    result.co2_consumed_kg += co2_moxie
    result.power_consumed_kwh += pwr_moxie

    # --- 2. Water Electrolysis: H₂O → H₂ + O₂ ---
    h2_electro, o2_electro, h2o_consumed, pwr_electro = electrolyze_water(
        water_budget_kg, remaining_power,
    )
    remaining_power -= pwr_electro
    result.o2_produced_kg += o2_electro
    result.h2_produced_kg = h2_electro
    result.h2o_consumed_kg = h2o_consumed
    result.power_consumed_kwh += pwr_electro

    # --- 3. Sabatier: CO₂ + H₂ → CH₄ + H₂O ---
    total_h2 = state.h2_stockpile_kg + h2_electro
    ch4_kg, h2o_sab, co2_sab, h2_consumed, heat_kj = sabatier_output(
        sabatier, total_h2, temp_c,
    )
    result.ch4_produced_kg = ch4_kg
    result.h2o_produced_kg = h2o_sab
    result.co2_consumed_kg += co2_sab
    result.h2_consumed_kg = h2_consumed
    result.sabatier_heat_kj = heat_kj

    # --- O₂ budget ---
    o2_demand = colony_o2_demand(population)
    total_o2_available = state.o2_buffer_kg + result.o2_produced_kg
    o2_after_consumption = total_o2_available - o2_demand
    result.o2_consumed_kg = o2_demand

    if o2_after_consumption < 0:
        result.o2_deficit_kg = abs(o2_after_consumption)
        state.deficit_sols += 1
        state.o2_buffer_kg = 0.0
    else:
        result.o2_deficit_kg = 0.0
        state.o2_buffer_kg = o2_after_consumption

    # --- Update stockpiles ---
    state.h2_stockpile_kg = max(0.0, total_h2 - h2_consumed)
    state.ch4_stockpile_kg += ch4_kg
    state.total_o2_produced_kg += result.o2_produced_kg
    state.total_ch4_produced_kg += ch4_kg
    state.total_h2o_produced_kg += h2o_sab
    state.total_power_consumed_kwh += result.power_consumed_kwh

    # --- MAV progress ---
    result.mav_progress = mav_progress_fraction(state.ch4_stockpile_kg)

    return result


# ---------------------------------------------------------------------------
# Factory: default ISRU config for colony archetypes
# ---------------------------------------------------------------------------

def create_isru(strategy: str) -> tuple[MoxieBank, SabatierBank, AtmoState]:
    """Create default ISRU equipment for a colony strategy.

    Conservative: more units, larger O₂ buffer (safety margin).
    Balanced: moderate setup.
    Aggressive: fewer units, smaller buffer (prioritize expansion).
    """
    # Each MOXIE unit produces ~5 kg O₂/sol at peak.  Average efficiency
    # with temperature/dust losses is ~80-85%.  Provision enough units to
    # cover population demand (0.84 kg/person/sol) with safety margin.
    configs = {
        "conservative": {
            "moxie_units": 30,   # 120 people × 0.84 / (5 × 0.8) = 25.2 → 30 w/ margin
            "sabatier_units": 3,
            "o2_buffer_days": 30,
            "population": 120,
        },
        "balanced": {
            "moxie_units": 20,   # 80 people × 0.84 / (5 × 0.8) = 16.8 → 20 w/ margin
            "sabatier_units": 2,
            "o2_buffer_days": 20,
            "population": 80,
        },
        "aggressive": {
            "moxie_units": 14,   # 60 people × 0.84 / (5 × 0.8) = 12.6 → 14 tight
            "sabatier_units": 1,
            "o2_buffer_days": 10,
            "population": 60,
        },
    }
    cfg = configs.get(strategy, configs["balanced"])

    moxie = MoxieBank(units=cfg["moxie_units"])
    sabatier = SabatierBank(units=cfg["sabatier_units"])
    o2_buffer = cfg["population"] * O2_KG_PER_PERSON_SOL * cfg["o2_buffer_days"]
    state = AtmoState(o2_buffer_kg=o2_buffer)

    return moxie, sabatier, state


def perform_maintenance(moxie: MoxieBank) -> float:
    """Perform filter maintenance on MOXIE bank.

    Returns the amount of clogging cleared.
    """
    cleared = moxie.filter_clog * FILTER_MAINTENANCE_CLEAR
    moxie.filter_clog -= cleared
    moxie.filter_clog = max(0.0, moxie.filter_clog)
    return cleared
