"""battery_bank.py -- Mars Colony Energy Storage (Lithium-Sulfur Battery Array).

Models a colony-scale battery bank for buffering solar/nuclear power.
Each tick = 1 sol of charge/discharge cycling.

Physics modelled
----------------
* **Lithium-sulfur chemistry** -- high specific energy (500 Wh/kg theoretical,
  ~350 Wh/kg practical) chosen for Mars mass constraints.  Li-S avoids
  cobalt supply-chain issues.  Reference: Manthiram, 2020.
* **Charge/discharge** -- round-trip coulombic efficiency ~92%.  Discharge
  depth limited to protect cell life (default 80% DoD max).
* **Capacity fade** -- cycle-dependent degradation.  Li-S cells lose ~0.05%
  capacity per full-equivalent cycle due to polysulfide shuttle.
  End-of-life at 60% of original capacity.
* **Thermal management** -- Mars ambient ~210 K (-63C).  Battery operates
  optimally at 293 K.  Heater power required to maintain temperature.
  Cold batteries suffer reversible capacity loss (~0.3% per K below 273 K).
* **Self-discharge** -- Li-S has higher self-discharge than Li-ion:
  ~8% per month at 293 K, temperature-dependent (Arrhenius).
* **Module architecture** -- bank is N parallel strings of M series cells.
  Individual module failures reduce total capacity proportionally.
* **Charge controller** -- constant-current / constant-voltage (CC/CV).
  Prevents overcharge above 2.4 V/cell, cuts off below 1.7 V/cell.

Reference systems:
  - ISS batteries: 48 Li-ion, 3.6 kWh each = 173 kWh total
  - Mars colony (this model): 500 kWh nameplate, ~400 kWh usable (80% DoD)
  - Tesla Megapack: 3.9 MWh per unit (Earth reference)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

# -- Physical constants ---------------------------------------------------

MARS_AMBIENT_K = 210.0          # Average Mars surface temperature (K)
OPTIMAL_TEMP_K = 293.0          # Optimal battery operating temp (K)
BOLTZMANN_EV = 8.617e-5         # Boltzmann constant (eV/K)

# -- Battery chemistry (Li-S) --------------------------------------------

CELL_VOLTAGE_NOM = 2.1          # Nominal cell voltage (V)
CELL_VOLTAGE_MAX = 2.4          # Max charge voltage (V)
CELL_VOLTAGE_MIN = 1.7          # Cutoff discharge voltage (V)
SPECIFIC_ENERGY_WH_KG = 350.0   # Practical specific energy (Wh/kg)
ROUND_TRIP_EFFICIENCY = 0.92    # Coulombic round-trip efficiency
MAX_DOD = 0.80                  # Maximum depth of discharge (fraction)
CYCLE_DEGRADATION = 0.0005      # Capacity loss per full-equivalent cycle
EOL_CAPACITY_FRAC = 0.60        # End-of-life capacity fraction

# -- Thermal --------------------------------------------------------------

COLD_PENALTY_PER_K = 0.003      # Reversible capacity loss per K below 273 K
HEATER_WATTS = 200.0            # Heater power to maintain optimal temp (W)
HEATER_EFFICIENCY = 0.95        # Resistive heater efficiency
SELF_DISCHARGE_MONTHLY = 0.08   # Self-discharge fraction per month at 293 K
SELF_DISCHARGE_EA_EV = 0.30     # Activation energy for self-discharge (eV)
SECONDS_PER_SOL = 88775.0       # Mars sol duration (seconds)
HOURS_PER_SOL = SECONDS_PER_SOL / 3600.0  # ~24.66 hours

# -- Default bank sizing --------------------------------------------------

DEFAULT_CAPACITY_KWH = 500.0    # Nameplate capacity (kWh)
DEFAULT_MODULES = 20            # Number of parallel modules
DEFAULT_MAX_CHARGE_KW = 100.0   # Max charge rate (kW)
DEFAULT_MAX_DISCHARGE_KW = 120.0  # Max discharge rate (kW)


# -- Dataclasses -----------------------------------------------------------

@dataclass
class BatteryState:
    """Mutable state of the colony battery bank."""
    sol: int = 0
    stored_kwh: float = 0.0             # Current energy stored (kWh)
    nameplate_kwh: float = DEFAULT_CAPACITY_KWH
    health: float = 1.0                 # Capacity health fraction (1.0 = new)
    temperature_k: float = OPTIMAL_TEMP_K
    cumulative_cycles: float = 0.0      # Full-equivalent cycles
    modules_online: int = DEFAULT_MODULES
    modules_total: int = DEFAULT_MODULES
    heater_on: bool = True
    charged_kwh_today: float = 0.0      # Energy charged this sol
    discharged_kwh_today: float = 0.0   # Energy discharged this sol
    self_discharge_kwh_today: float = 0.0


@dataclass
class SolRecord:
    """Immutable record of one sol's battery activity."""
    sol: int
    stored_kwh: float
    health: float
    temperature_k: float
    charged_kwh: float
    discharged_kwh: float
    self_discharge_kwh: float
    usable_kwh: float
    modules_online: int
    heater_energy_kwh: float


# -- Pure functions --------------------------------------------------------

def usable_capacity(state: BatteryState) -> float:
    """Calculate current usable capacity (kWh) accounting for health,
    modules, depth-of-discharge, and temperature."""
    module_frac = state.modules_online / max(state.modules_total, 1)
    base = state.nameplate_kwh * state.health * module_frac * MAX_DOD

    # Cold penalty (reversible)
    if state.temperature_k < 273.0:
        cold_loss = COLD_PENALTY_PER_K * (273.0 - state.temperature_k)
        cold_loss = min(cold_loss, 0.5)  # Cap at 50% loss
        base *= (1.0 - cold_loss)

    return max(base, 0.0)


def self_discharge_rate(temperature_k: float) -> float:
    """Self-discharge fraction per sol, temperature-adjusted via Arrhenius.

    Returns fraction of stored energy lost per sol.
    """
    # Monthly rate at reference temp
    daily_rate_ref = 1.0 - (1.0 - SELF_DISCHARGE_MONTHLY) ** (1.0 / 30.0)

    # Arrhenius scaling
    if temperature_k > 0 and OPTIMAL_TEMP_K > 0:
        exponent = (SELF_DISCHARGE_EA_EV / BOLTZMANN_EV) * (
            (1.0 / OPTIMAL_TEMP_K) - (1.0 / temperature_k)
        )
        exponent = max(min(exponent, 20.0), -20.0)  # Clamp for stability
        arrhenius = math.exp(exponent)
    else:
        arrhenius = 1.0

    sol_rate = daily_rate_ref * arrhenius * (HOURS_PER_SOL / 24.0)
    return max(min(sol_rate, 0.5), 0.0)  # Clamp to physical bounds


def charge_energy(
    state: BatteryState,
    available_kwh: float,
    max_rate_kw: float = DEFAULT_MAX_CHARGE_KW,
) -> float:
    """Calculate how much energy can actually be stored this sol.

    Returns kWh actually stored (after efficiency losses).
    """
    cap = usable_capacity(state)
    headroom = cap - state.stored_kwh
    if headroom <= 0:
        return 0.0

    # Rate limit: max_rate_kw * hours per sol
    rate_limit = max_rate_kw * HOURS_PER_SOL
    input_kwh = min(available_kwh, rate_limit, headroom / ROUND_TRIP_EFFICIENCY)
    stored = input_kwh * ROUND_TRIP_EFFICIENCY
    stored = min(stored, headroom)
    return max(stored, 0.0)


def discharge_energy(
    state: BatteryState,
    demand_kwh: float,
    max_rate_kw: float = DEFAULT_MAX_DISCHARGE_KW,
) -> float:
    """Calculate how much energy can be delivered this sol.

    Returns kWh actually delivered to the load.
    """
    available = state.stored_kwh
    if available <= 0:
        return 0.0

    rate_limit = max_rate_kw * HOURS_PER_SOL
    delivered = min(demand_kwh, rate_limit, available)
    return max(delivered, 0.0)


def cycle_degradation(charged_kwh: float, discharged_kwh: float,
                       nameplate_kwh: float) -> float:
    """Calculate health loss from this sol's cycling.

    Full-equivalent cycles = average of charge+discharge / nameplate.
    """
    if nameplate_kwh <= 0:
        return 0.0
    throughput = (charged_kwh + discharged_kwh) / 2.0
    equivalent_cycles = throughput / nameplate_kwh
    return equivalent_cycles * CYCLE_DEGRADATION


def heater_energy_per_sol(ambient_k: float = MARS_AMBIENT_K,
                           heater_on: bool = True) -> float:
    """Calculate heater energy needed per sol (kWh).

    Scales linearly with temperature delta from optimal.
    """
    if not heater_on:
        return 0.0
    delta = max(OPTIMAL_TEMP_K - ambient_k, 0.0)
    # Proportional heating: full power at full delta
    fraction = delta / max(OPTIMAL_TEMP_K - 150.0, 1.0)  # 150K = deep cold
    fraction = min(fraction, 1.0)
    watts = HEATER_WATTS * fraction / HEATER_EFFICIENCY
    return watts * HOURS_PER_SOL / 1000.0  # Convert Wh to kWh


def battery_temperature(ambient_k: float, heater_on: bool) -> float:
    """Determine battery temperature based on heating status."""
    if heater_on:
        return OPTIMAL_TEMP_K
    # Without heater, battery drifts toward ambient (simplified)
    return ambient_k + 10.0  # Slight insulation benefit


def should_shed_module(state: BatteryState, rng_val: float) -> bool:
    """Determine if a module fails this sol.

    Failure probability increases with age (cycles) and cold.
    """
    if state.modules_online <= 0:
        return False
    base_rate = 0.0001  # 0.01% per sol base failure rate
    age_factor = 1.0 + state.cumulative_cycles / 1000.0
    cold_factor = 1.0 if state.temperature_k >= 273.0 else 1.5
    probability = base_rate * age_factor * cold_factor
    return rng_val < probability


# -- Tick function ---------------------------------------------------------

def tick_battery(
    state: BatteryState,
    charge_available_kwh: float = 0.0,
    discharge_demand_kwh: float = 0.0,
    ambient_k: float = MARS_AMBIENT_K,
    rng_val: float = 0.5,
    max_charge_kw: float = DEFAULT_MAX_CHARGE_KW,
    max_discharge_kw: float = DEFAULT_MAX_DISCHARGE_KW,
) -> SolRecord:
    """Advance battery state by one sol. Returns a SolRecord."""
    state.sol += 1

    # Thermal management
    state.temperature_k = battery_temperature(ambient_k, state.heater_on)
    heater_kwh = heater_energy_per_sol(ambient_k, state.heater_on)

    # Self-discharge
    sd_rate = self_discharge_rate(state.temperature_k)
    sd_kwh = state.stored_kwh * sd_rate
    state.stored_kwh = max(state.stored_kwh - sd_kwh, 0.0)
    state.self_discharge_kwh_today = sd_kwh

    # Charge
    charged = charge_energy(state, charge_available_kwh, max_charge_kw)
    state.stored_kwh += charged
    state.charged_kwh_today = charged

    # Discharge (heater draws from battery if no external source)
    total_demand = discharge_demand_kwh + heater_kwh
    discharged = discharge_energy(state, total_demand, max_discharge_kw)
    state.stored_kwh = max(state.stored_kwh - discharged, 0.0)
    state.discharged_kwh_today = discharged

    # Degradation
    health_loss = cycle_degradation(charged, discharged, state.nameplate_kwh)
    state.health = max(state.health - health_loss, EOL_CAPACITY_FRAC)
    state.cumulative_cycles += (charged + discharged) / (
        2.0 * max(state.nameplate_kwh, 1.0)
    )

    # Module failure check
    if should_shed_module(state, rng_val):
        state.modules_online = max(state.modules_online - 1, 0)

    # Clamp stored energy to usable capacity
    cap = usable_capacity(state)
    state.stored_kwh = min(state.stored_kwh, cap)

    return SolRecord(
        sol=state.sol,
        stored_kwh=state.stored_kwh,
        health=state.health,
        temperature_k=state.temperature_k,
        charged_kwh=charged,
        discharged_kwh=discharged,
        self_discharge_kwh=sd_kwh,
        usable_kwh=cap,
        modules_online=state.modules_online,
        heater_energy_kwh=heater_kwh,
    )


# -- Factory and runner ----------------------------------------------------

def make_battery(
    capacity_kwh: float = DEFAULT_CAPACITY_KWH,
    modules: int = DEFAULT_MODULES,
    initial_charge_frac: float = 0.5,
) -> BatteryState:
    """Create a new battery bank with given specs."""
    state = BatteryState(
        nameplate_kwh=capacity_kwh,
        modules_online=modules,
        modules_total=modules,
    )
    cap = usable_capacity(state)
    state.stored_kwh = cap * initial_charge_frac
    return state


def run_battery(
    state: BatteryState,
    sols: int = 100,
    charge_per_sol: float = 50.0,
    demand_per_sol: float = 40.0,
    ambient_k: float = MARS_AMBIENT_K,
) -> List[SolRecord]:
    """Run the battery for multiple sols. Returns list of SolRecords."""
    records = []
    for i in range(sols):
        rng_val = ((i * 7 + 13) % 1000) / 1000.0  # Deterministic pseudo-random
        record = tick_battery(
            state,
            charge_available_kwh=charge_per_sol,
            discharge_demand_kwh=demand_per_sol,
            ambient_k=ambient_k,
            rng_val=rng_val,
        )
        records.append(record)
    return records
