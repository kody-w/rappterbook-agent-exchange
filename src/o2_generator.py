"""
o2_generator.py — Mars ISRU oxygen production model.

Models atmospheric CO2 electrolysis using solid oxide electrolysis
cells (SOEC), identical in principle to NASA's MOXIE experiment
on the Perseverance rover.

Process: CO2 → CO + ½O2 (at ~800°C in a ceramic SOEC stack)

Physical references:
  - MOXIE: 300 W input → 10 g O2/hr at peak (JPL, 2023)
  - Mars atmosphere: 95.3% CO2, 0.636 kPa mean surface pressure
  - SOEC operating temperature: 800°C (electrical heating required)
  - Human O2 consumption: 0.84 kg/person/sol (NASA HRP)
  - CO2 molecular mass: 44.01 g/mol
  - O2 molecular mass: 32.00 g/mol
  - CO molecular mass: 28.01 g/mol
  - Stoichiometry: 2 CO2 → 2 CO + O2

One tick = one sol. Mass in kg, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MARS_CO2_FRACTION = 0.953          # atmospheric CO2 mole fraction
MARS_SURFACE_PRESSURE_KPA = 0.636  # mean surface pressure

# Molecular masses (g/mol)
MM_CO2 = 44.01
MM_O2 = 32.00
MM_CO = 28.01

# Stoichiometry: 2 CO2 → 2 CO + O2
# Mass ratio: 1 kg CO2 → (32/88) kg O2 = 0.3636 kg O2
O2_PER_CO2_MASS = (MM_O2) / (2 * MM_CO2)       # 0.36363...
CO_PER_CO2_MASS = (2 * MM_CO) / (2 * MM_CO2)    # 0.63636...

# MOXIE-derived performance
MOXIE_POWER_W = 300.0              # electrical input (W)
MOXIE_O2_RATE_G_HR = 10.0          # peak O2 output (g/hr)
# Energy cost: 300 W / 10 g/hr = 30 Wh/g = 30 kWh/kg
ENERGY_KWH_PER_KG_O2 = MOXIE_POWER_W / MOXIE_O2_RATE_G_HR  # 30.0 kWh/kg

# SOEC stack parameters
SOEC_OPERATING_TEMP_C = 800.0      # nominal operating temperature
SOEC_MIN_TEMP_C = 600.0            # below this, electrolysis stops
SOEC_EFFICIENCY_AT_NOMINAL = 0.85  # thermal-to-useful conversion

# Dust filter
DUST_CLOG_RATE_PER_SOL = 0.003     # fraction of filter lost per sol (clear sky)
DUST_STORM_CLOG_RATE = 0.020       # during active dust storm
FILTER_CLEANING_RESTORES = 0.80    # manual cleaning restores 80% of capacity

# Human O2 consumption
O2_KG_PER_PERSON_SOL = 0.84        # NASA Human Research Program

# Sol duration for unit conversion
SOL_HOURS = 24.66                  # Mars sol in hours

# O2 storage
STORAGE_LEAK_RATE_PER_SOL = 0.001  # 0.1% leak per sol (micrometeorite wear)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SOECStack:
    """A solid oxide electrolysis cell stack for O2 production.

    Attributes:
        n_cells: number of MOXIE-equivalent cells in the stack
        filter_health: dust filter condition [0, 1]
        cumulative_runtime_hrs: total operating hours (affects degradation)
    """
    n_cells: int
    filter_health: float = 1.0
    cumulative_runtime_hrs: float = 0.0

    def __post_init__(self) -> None:
        if self.n_cells < 0:
            raise ValueError("n_cells must be non-negative")
        self.filter_health = max(0.0, min(1.0, self.filter_health))
        self.cumulative_runtime_hrs = max(0.0, self.cumulative_runtime_hrs)

    def max_o2_rate_kg_hr(self) -> float:
        """Maximum O2 production rate (kg/hr) at full filter and power."""
        base_g_hr = self.n_cells * MOXIE_O2_RATE_G_HR
        return base_g_hr * self.filter_health / 1000.0

    def power_demand_kw(self) -> float:
        """Power draw when running at full capacity (kW)."""
        return self.n_cells * MOXIE_POWER_W / 1000.0


@dataclass
class O2Storage:
    """Pressurised oxygen storage tank.

    Attributes:
        capacity_kg: maximum O2 the tank can hold
        stored_kg: current O2 in the tank
    """
    capacity_kg: float
    stored_kg: float = 0.0

    def __post_init__(self) -> None:
        self.capacity_kg = max(0.0, self.capacity_kg)
        self.stored_kg = max(0.0, min(self.stored_kg, self.capacity_kg))

    def headroom_kg(self) -> float:
        """Remaining capacity (kg)."""
        return max(0.0, self.capacity_kg - self.stored_kg)

    def store(self, kg: float) -> float:
        """Add O2. Returns amount actually stored."""
        if kg <= 0:
            return 0.0
        actual = min(kg, self.headroom_kg())
        self.stored_kg += actual
        return actual

    def draw(self, kg: float) -> float:
        """Remove O2. Returns amount actually delivered."""
        if kg <= 0:
            return 0.0
        actual = min(kg, self.stored_kg)
        self.stored_kg -= actual
        return actual

    def apply_leak(self) -> float:
        """Daily micro-leak. Returns kg lost."""
        lost = self.stored_kg * STORAGE_LEAK_RATE_PER_SOL
        self.stored_kg -= lost
        return lost


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------

def co2_intake_kg_hr(pressure_kpa: float) -> float:
    """CO2 mass available for intake per hour (kg/hr) per MOXIE-cell.

    Intake scales linearly with atmospheric pressure (pump-limited).
    At nominal Mars pressure (0.636 kPa), one cell can ingest enough
    CO2 to produce 10 g O2/hr.
    """
    if pressure_kpa <= 0:
        return 0.0
    pressure_ratio = pressure_kpa / MARS_SURFACE_PRESSURE_KPA
    # At nominal pressure, one cell needs ~27.5 g CO2/hr to make 10 g O2/hr
    nominal_co2_g_hr = MOXIE_O2_RATE_G_HR / O2_PER_CO2_MASS
    return nominal_co2_g_hr * pressure_ratio / 1000.0


def soec_efficiency(operating_temp_c: float) -> float:
    """Electrolysis efficiency as a function of stack temperature.

    Below 600°C: electrolysis cannot proceed (insufficient ion mobility).
    600–800°C: efficiency ramps linearly from 0.2 to 0.85.
    Above 800°C: slight gains up to 0.90 at 1000°C, then degradation.
    """
    if operating_temp_c < SOEC_MIN_TEMP_C:
        return 0.0
    if operating_temp_c <= SOEC_OPERATING_TEMP_C:
        frac = (operating_temp_c - SOEC_MIN_TEMP_C) / (
            SOEC_OPERATING_TEMP_C - SOEC_MIN_TEMP_C
        )
        return 0.20 + frac * (SOEC_EFFICIENCY_AT_NOMINAL - 0.20)
    # Above nominal: diminishing returns, then degradation above 1000°C
    if operating_temp_c <= 1000.0:
        bonus = (operating_temp_c - SOEC_OPERATING_TEMP_C) / 200.0 * 0.05
        return SOEC_EFFICIENCY_AT_NOMINAL + bonus
    # Above 1000°C: thermal degradation
    penalty = (operating_temp_c - 1000.0) / 500.0 * 0.4
    return max(0.0, 0.90 - penalty)


def produce_o2_sol(
    stack: SOECStack,
    power_kwh: float,
    pressure_kpa: float,
    dust_opacity: float = 0.0,
) -> dict:
    """Run the SOEC stack for one sol, return production metrics.

    Args:
        stack: the electrolysis stack (mutated: filter_health, runtime)
        power_kwh: energy budget allocated for O2 production this sol
        pressure_kpa: current atmospheric pressure (affects CO2 intake)
        dust_opacity: current dust level (affects filter clogging)

    Returns:
        dict with o2_produced_kg, co2_consumed_kg, co_produced_kg,
        power_consumed_kwh, filter_health, efficiency.

    Conservation laws enforced:
        - co2_consumed = o2_produced / O2_PER_CO2_MASS
        - co_produced = co2_consumed * CO_PER_CO2_MASS
        - mass in (CO2) = mass out (CO + O2)
        - power_consumed <= power_kwh
    """
    if stack.n_cells == 0 or power_kwh <= 0 or pressure_kpa <= 0:
        return _zero_result(stack)

    # Filter degradation from dust
    clog_rate = DUST_STORM_CLOG_RATE if dust_opacity > 0.3 else DUST_CLOG_RATE_PER_SOL
    clog_rate *= (1.0 + dust_opacity)  # higher dust = faster clogging
    stack.filter_health = max(0.0, stack.filter_health - clog_rate)

    if stack.filter_health <= 0:
        return _zero_result(stack)

    # SOEC operates at fixed internal temp (electrically heated)
    efficiency = soec_efficiency(SOEC_OPERATING_TEMP_C)

    # Maximum O2 from available power
    effective_energy = power_kwh * efficiency * stack.filter_health
    max_o2_from_power = effective_energy / ENERGY_KWH_PER_KG_O2

    # Maximum O2 from CO2 intake (pressure-limited)
    co2_rate_per_cell = co2_intake_kg_hr(pressure_kpa)
    max_co2_sol = co2_rate_per_cell * stack.n_cells * SOL_HOURS * stack.filter_health
    max_o2_from_co2 = max_co2_sol * O2_PER_CO2_MASS

    # Binding constraint
    o2_produced = min(max_o2_from_power, max_o2_from_co2)
    o2_produced = max(0.0, o2_produced)

    # Mass conservation
    co2_consumed = o2_produced / O2_PER_CO2_MASS
    co_produced = co2_consumed * CO_PER_CO2_MASS

    # Power actually used
    if max_o2_from_power > 0 and o2_produced < max_o2_from_power:
        power_used = power_kwh * (o2_produced / max_o2_from_power)
    else:
        power_used = power_kwh
    power_used = min(power_used, power_kwh)

    # Runtime tracking
    stack.cumulative_runtime_hrs += SOL_HOURS

    return {
        "o2_produced_kg": round(o2_produced, 6),
        "co2_consumed_kg": round(co2_consumed, 6),
        "co_produced_kg": round(co_produced, 6),
        "power_consumed_kwh": round(power_used, 4),
        "filter_health": round(stack.filter_health, 6),
        "efficiency": round(efficiency, 4),
        "runtime_hrs": round(stack.cumulative_runtime_hrs, 2),
    }


def _zero_result(stack: SOECStack) -> dict:
    """Return a zero-production result."""
    return {
        "o2_produced_kg": 0.0,
        "co2_consumed_kg": 0.0,
        "co_produced_kg": 0.0,
        "power_consumed_kwh": 0.0,
        "filter_health": round(stack.filter_health, 6),
        "efficiency": 0.0,
        "runtime_hrs": round(stack.cumulative_runtime_hrs, 2),
    }


def clean_filter(stack: SOECStack) -> float:
    """Manual filter cleaning. Returns health restored."""
    before = stack.filter_health
    restored = (1.0 - stack.filter_health) * FILTER_CLEANING_RESTORES
    stack.filter_health = min(1.0, stack.filter_health + restored)
    return round(stack.filter_health - before, 6)


def crew_o2_demand(population: int) -> float:
    """Daily O2 demand for crew breathing (kg/sol)."""
    return population * O2_KG_PER_PERSON_SOL


def o2_sufficiency(stored_kg: float, population: int, days: int = 1) -> float:
    """Ratio of stored O2 to demand over given days. ≥1.0 means sufficient."""
    demand = crew_o2_demand(population) * days
    if demand <= 0:
        return float("inf") if stored_kg > 0 else 1.0
    return stored_kg / demand


def tick_o2_system(
    stack: SOECStack,
    storage: O2Storage,
    power_kwh: float,
    pressure_kpa: float,
    dust_opacity: float,
    population: int,
) -> dict:
    """Advance the O2 life support system by one sol.

    1. Produce O2 from atmospheric CO2
    2. Store surplus
    3. Apply storage leak
    4. Crew consumption draw
    5. Report sufficiency

    Returns snapshot dict with all O2 metrics.
    """
    # 1. Produce
    production = produce_o2_sol(stack, power_kwh, pressure_kpa, dust_opacity)
    o2_made = production["o2_produced_kg"]

    # 2. Crew demand
    demand = crew_o2_demand(population)

    # Direct supply (production feeds crew first)
    direct_supply = min(o2_made, demand)
    surplus = o2_made - direct_supply
    remaining_demand = demand - direct_supply

    # 3. Store surplus
    stored = storage.store(surplus)

    # 4. Leak
    leaked = storage.apply_leak()

    # 5. Draw from storage if production insufficient
    from_storage = storage.draw(remaining_demand)
    total_delivered = direct_supply + from_storage
    shortfall = max(0.0, demand - total_delivered)

    # Sufficiency ratio
    sufficiency = o2_sufficiency(storage.stored_kg, population)

    return {
        "o2_produced_kg": production["o2_produced_kg"],
        "co2_consumed_kg": production["co2_consumed_kg"],
        "co_produced_kg": production["co_produced_kg"],
        "power_consumed_kwh": production["power_consumed_kwh"],
        "o2_demand_kg": round(demand, 4),
        "o2_delivered_kg": round(total_delivered, 4),
        "o2_shortfall_kg": round(shortfall, 4),
        "o2_stored_kg": round(storage.stored_kg, 4),
        "o2_leaked_kg": round(leaked, 6),
        "o2_surplus_stored_kg": round(stored, 4),
        "filter_health": production["filter_health"],
        "efficiency": production["efficiency"],
        "sufficiency_ratio": round(sufficiency, 4),
    }


def create_colony_o2_system(
    strategy: str,
    population: int,
) -> tuple[SOECStack, O2Storage]:
    """Create an O2 system sized for a colony strategy.

    Conservative: oversized, large reserves.
    Balanced: matched to population with moderate buffer.
    Aggressive: lean, minimal reserves.
    """
    # Size stack so daily production ≥ daily demand at full efficiency
    daily_demand = crew_o2_demand(population)
    # Each cell produces ~10 g/hr * 24.66 hr * 0.85 eff ≈ 0.21 kg/sol
    o2_per_cell_sol = (
        MOXIE_O2_RATE_G_HR * SOL_HOURS * SOEC_EFFICIENCY_AT_NOMINAL / 1000.0
    )

    configs = {
        "conservative": {"margin": 2.0, "reserve_sols": 30},
        "balanced":     {"margin": 1.5, "reserve_sols": 15},
        "aggressive":   {"margin": 1.2, "reserve_sols": 7},
    }
    cfg = configs.get(strategy, configs["balanced"])

    cells_needed = max(1, math.ceil(daily_demand * cfg["margin"] / o2_per_cell_sol))
    storage_capacity = daily_demand * cfg["reserve_sols"]

    stack = SOECStack(n_cells=cells_needed)
    storage = O2Storage(capacity_kg=storage_capacity, stored_kg=storage_capacity * 0.5)
    return stack, storage
