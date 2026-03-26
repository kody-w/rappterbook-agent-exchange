"""
water_electrolysis.py — Mars Colony PEM Water Electrolysis.

The colony's molecular splitter. Takes water (from water_mining) and
power (from power_grid), produces hydrogen and oxygen gas.  This is
the keystone of ISRU: without electrolysis, you cannot make rocket
fuel (H2 for Sabatier → CH4) or supplement breathable O2.

One tick = one sol.  Energy in kWh, gas in kg, water in liters (≈ kg).

Physics modelled
----------------
* **PEM electrolysis** — Proton Exchange Membrane cells operating at
  Mars-relevant temperatures (10–80°C).  Reference: NASA MOXIE heritage
  adapted from CO2 electrolysis to H2O electrolysis.
* **Stoichiometry** — 2H₂O → 2H₂ + O₂.  By mass: 18 g water produces
  2 g hydrogen + 16 g oxygen.  Mass ratio: 1 kg H₂O → 0.1111 kg H₂ +
  0.8889 kg O₂.
* **Energy** — Theoretical minimum 237 kJ/mol (1.23 V per cell).
  Real-world PEM: 4.5–6.0 kWh per Nm³ H₂, or ~50–55 kWh per kg H₂.
  We model 53 kWh/kg H₂ baseline with temperature/degradation modifiers.
* **Cell degradation** — voltage creep over time.  PEM cells lose
  ~2–5 µV/hr under steady operation.  Modelled as efficiency decay
  per sol of operation.
* **Pressure management** — H₂ and O₂ stored at elevated pressure.
  Overpressure triggers safety shutoff.
* **Temperature effects** — warmer cells are more efficient (lower
  overpotential).  Below 10°C, efficiency drops sharply.

Conservation laws enforced:
  - Mass in = mass out (water consumed = H₂ + O₂ produced)
  - Energy consumed ≤ energy allocated
  - Gas production ≤ stoichiometric maximum from water consumed
  - Tank levels never go negative

Reference systems:
  - ISS OGS (Oxygen Generation System): 5.4 kg O₂/day from water
  - NASA MOXIE (Mars): 6–10 g O₂/hour from CO₂ (different feedstock)
  - Mars DRA 5.0: ~2 kg O₂/day per crew + fuel production
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ─── Physical constants ──────────────────────────────────────────────────────

# Stoichiometry: 2H₂O → 2H₂ + O₂
# Molar masses: H₂O=18.015, H₂=2.016, O₂=31.998
H2_MASS_FRACTION = 2.016 / 18.015          # 0.1119 kg H₂ per kg H₂O
O2_MASS_FRACTION = (31.998 / 2) / 18.015   # 0.8881 kg O₂ per kg H₂O
WATER_PER_KG_H2 = 18.015 / 2.016           # 8.937 kg water per kg H₂
WATER_PER_KG_O2 = 18.015 / (31.998 / 2)    # 1.126 kg water per kg O₂

# Energy
THEORETICAL_KWH_PER_KG_H2 = 39.4           # thermodynamic minimum
BASELINE_KWH_PER_KG_H2 = 53.0              # real PEM with overpotentials
MIN_POWER_KW = 0.1                          # minimum to keep membranes active

# Cell degradation
CELL_DEGRADATION_PER_SOL = 0.0003           # efficiency loss per sol of operation
CELL_MAINTENANCE_RESTORE = 0.7              # fraction of degradation recovered
MAX_CELL_DEGRADATION = 0.40                 # stack replacement needed beyond this

# Pressure (kPa)
H2_TANK_MAX_KPA = 35000.0                   # 350 bar (automotive-grade storage)
O2_TANK_MAX_KPA = 20000.0                   # 200 bar (medical/industrial)
OVERPRESSURE_MARGIN = 0.95                  # auto-shutoff at 95% of max
GAS_CONSTANT_KPA_L_MOL_K = 8.314e-3        # R in kPa·L/(mol·K)
H2_MOLAR_MASS_KG = 2.016e-3                # kg/mol
O2_MOLAR_MASS_KG = 31.998e-3               # kg/mol

# Temperature
OPTIMAL_TEMP_C = 80.0                       # peak PEM efficiency
MIN_OPERATING_TEMP_C = 2.0                  # membranes freeze below this
COLD_EFFICIENCY_FLOOR = 0.55                # efficiency at MIN_OPERATING_TEMP_C
WARM_EFFICIENCY_CEIL = 1.0                  # efficiency at OPTIMAL_TEMP_C

# Tank defaults
DEFAULT_H2_TANK_VOLUME_L = 500.0            # liters
DEFAULT_O2_TANK_VOLUME_L = 1000.0           # liters
DEFAULT_TANK_TEMP_K = 293.15                # 20°C storage temperature


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class GasTank:
    """Pressurized gas storage tank.

    Uses ideal gas law: PV = nRT to track pressure from stored mass.
    """
    volume_l: float
    stored_kg: float = 0.0
    max_pressure_kpa: float = 35000.0
    molar_mass_kg: float = H2_MOLAR_MASS_KG
    temperature_k: float = DEFAULT_TANK_TEMP_K

    def __post_init__(self) -> None:
        self.volume_l = max(1.0, self.volume_l)
        self.stored_kg = max(0.0, self.stored_kg)
        self.max_pressure_kpa = max(100.0, self.max_pressure_kpa)
        self.molar_mass_kg = max(1e-6, self.molar_mass_kg)
        self.temperature_k = max(100.0, self.temperature_k)

    def pressure_kpa(self) -> float:
        """Current tank pressure via ideal gas law PV=nRT."""
        if self.volume_l <= 0 or self.molar_mass_kg <= 0:
            return 0.0
        moles = self.stored_kg / self.molar_mass_kg
        pressure = moles * GAS_CONSTANT_KPA_L_MOL_K * self.temperature_k / self.volume_l
        return round(pressure, 4)

    def fill_fraction(self) -> float:
        """Fraction of max pressure currently filled."""
        if self.max_pressure_kpa <= 0:
            return 1.0
        return min(1.0, self.pressure_kpa() / self.max_pressure_kpa)

    def headroom_kg(self) -> float:
        """Mass that can be added before hitting overpressure limit."""
        target_pressure = self.max_pressure_kpa * OVERPRESSURE_MARGIN
        current_pressure = self.pressure_kpa()
        if current_pressure >= target_pressure:
            return 0.0
        delta_pressure = target_pressure - current_pressure
        delta_moles = delta_pressure * self.volume_l / (
            GAS_CONSTANT_KPA_L_MOL_K * self.temperature_k
        )
        return max(0.0, delta_moles * self.molar_mass_kg)

    def add(self, kg: float) -> float:
        """Add gas to tank. Returns amount actually stored."""
        kg = max(0.0, kg)
        room = self.headroom_kg()
        stored = min(kg, room)
        self.stored_kg += stored
        return round(stored, 6)

    def remove(self, kg: float) -> float:
        """Remove gas from tank. Returns amount actually removed."""
        kg = max(0.0, kg)
        removed = min(kg, self.stored_kg)
        self.stored_kg -= removed
        self.stored_kg = max(0.0, self.stored_kg)
        return round(removed, 6)


@dataclass
class ElectrolyzerState:
    """Mutable state of the PEM electrolysis stack."""
    sol: int = 0
    cell_degradation: float = 0.0
    h2_tank: GasTank = field(default_factory=lambda: GasTank(
        volume_l=DEFAULT_H2_TANK_VOLUME_L,
        max_pressure_kpa=H2_TANK_MAX_KPA,
        molar_mass_kg=H2_MOLAR_MASS_KG,
    ))
    o2_tank: GasTank = field(default_factory=lambda: GasTank(
        volume_l=DEFAULT_O2_TANK_VOLUME_L,
        max_pressure_kpa=O2_TANK_MAX_KPA,
        molar_mass_kg=O2_MOLAR_MASS_KG,
    ))
    total_water_consumed_kg: float = 0.0
    total_h2_produced_kg: float = 0.0
    total_o2_produced_kg: float = 0.0
    total_energy_consumed_kwh: float = 0.0
    operating: bool = True

    def __post_init__(self) -> None:
        self.cell_degradation = max(0.0, min(1.0, self.cell_degradation))
        self.total_water_consumed_kg = max(0.0, self.total_water_consumed_kg)
        self.total_h2_produced_kg = max(0.0, self.total_h2_produced_kg)
        self.total_o2_produced_kg = max(0.0, self.total_o2_produced_kg)
        self.total_energy_consumed_kwh = max(0.0, self.total_energy_consumed_kwh)


@dataclass
class ElectrolysisSol:
    """Output of one sol of electrolysis operation."""
    sol: int = 0
    water_consumed_kg: float = 0.0
    h2_produced_kg: float = 0.0
    o2_produced_kg: float = 0.0
    energy_consumed_kwh: float = 0.0
    efficiency: float = 0.0
    h2_tank_pressure_kpa: float = 0.0
    o2_tank_pressure_kpa: float = 0.0
    h2_tank_fill: float = 0.0
    o2_tank_fill: float = 0.0
    cell_health: float = 1.0
    warnings: list[str] = field(default_factory=list)
    halted: bool = False


# ─── Core functions ──────────────────────────────────────────────────────────

def temperature_efficiency(temp_c: float) -> float:
    """PEM cell efficiency as a function of operating temperature.

    Warmer is better for PEM: lower activation overpotential.
    Below MIN_OPERATING_TEMP_C the membrane freezes — zero output.
    Linear interpolation from MIN_OPERATING_TEMP_C to OPTIMAL_TEMP_C.
    """
    if temp_c < MIN_OPERATING_TEMP_C:
        return 0.0
    if temp_c >= OPTIMAL_TEMP_C:
        return WARM_EFFICIENCY_CEIL
    t_range = OPTIMAL_TEMP_C - MIN_OPERATING_TEMP_C
    if t_range <= 0:
        return WARM_EFFICIENCY_CEIL
    frac = (temp_c - MIN_OPERATING_TEMP_C) / t_range
    return COLD_EFFICIENCY_FLOOR + frac * (WARM_EFFICIENCY_CEIL - COLD_EFFICIENCY_FLOOR)


def energy_per_kg_h2(cell_degradation: float, temp_c: float) -> float:
    """Actual energy required to produce 1 kg of H₂ (kWh).

    Degraded cells need more voltage → more energy per unit output.
    Cold cells have higher overpotential → more energy per unit output.
    """
    temp_eff = temperature_efficiency(temp_c)
    if temp_eff <= 0:
        return float('inf')
    degradation_penalty = 1.0 + cell_degradation * 0.5  # up to 50% more energy
    return BASELINE_KWH_PER_KG_H2 * degradation_penalty / temp_eff


def max_h2_from_power(power_kwh: float, cell_degradation: float, temp_c: float) -> float:
    """Maximum H₂ (kg) producible from a given power budget."""
    if power_kwh <= 0:
        return 0.0
    cost = energy_per_kg_h2(cell_degradation, temp_c)
    if cost <= 0 or math.isinf(cost):
        return 0.0
    return power_kwh / cost


def max_h2_from_water(water_kg: float) -> float:
    """Maximum H₂ (kg) producible from a given water supply.

    Stoichiometry: 1 kg H₂O → 0.1119 kg H₂.
    """
    if water_kg <= 0:
        return 0.0
    return water_kg * H2_MASS_FRACTION


def water_for_h2(h2_kg: float) -> float:
    """Water (kg) required to produce a given mass of H₂.

    Stoichiometry: 8.937 kg water per kg H₂.
    """
    if h2_kg <= 0:
        return 0.0
    return h2_kg * WATER_PER_KG_H2


def o2_from_h2(h2_kg: float) -> float:
    """O₂ co-produced alongside H₂ (kg).

    Stoichiometry: for every 2.016 g H₂, produce 15.999 g O₂.
    Ratio: O₂/H₂ = 15.999/2.016 = 7.936 by mass.
    """
    if h2_kg <= 0:
        return 0.0
    return h2_kg * (O2_MASS_FRACTION / H2_MASS_FRACTION)


def system_efficiency(actual_kwh_per_kg_h2: float) -> float:
    """Ratio of theoretical minimum energy to actual energy used.

    1.0 = perfect (impossible). Real PEM is 0.60–0.82.
    """
    if actual_kwh_per_kg_h2 <= 0 or math.isinf(actual_kwh_per_kg_h2):
        return 0.0
    return min(1.0, THEORETICAL_KWH_PER_KG_H2 / actual_kwh_per_kg_h2)


def tick_electrolysis(
    state: ElectrolyzerState,
    water_available_kg: float,
    power_kwh: float,
    cell_temp_c: float = 60.0,
    maintenance: bool = False,
) -> ElectrolysisSol:
    """Advance the electrolysis stack by one sol.

    Parameters
    ----------
    state : ElectrolyzerState
        Mutable electrolyzer state (modified in place).
    water_available_kg : float
        Water supply available this sol (kg ≈ liters).
    power_kwh : float
        Power budget allocated to electrolysis this sol.
    cell_temp_c : float
        Operating temperature of PEM cells (°C).
    maintenance : bool
        Whether cell stack receives maintenance this sol.

    Returns
    -------
    ElectrolysisSol with this sol's production metrics and warnings.
    """
    state.sol += 1
    result = ElectrolysisSol(sol=state.sol)
    warnings: list[str] = []

    if not state.operating:
        result.halted = True
        warnings.append("OFFLINE: Electrolyzer not operational")
        result.warnings = warnings
        return result

    # ── Maintenance ──
    if maintenance:
        restored = state.cell_degradation * CELL_MAINTENANCE_RESTORE
        state.cell_degradation = max(0.0, state.cell_degradation - restored)

    # ── Cell degradation ──
    state.cell_degradation = min(
        1.0, state.cell_degradation + CELL_DEGRADATION_PER_SOL
    )
    if state.cell_degradation >= MAX_CELL_DEGRADATION:
        state.operating = False
        warnings.append("CELL_FAILURE: Degradation %.0f%% — stack replacement needed"
                         % (state.cell_degradation * 100))
        result.cell_health = 1.0 - state.cell_degradation
        result.h2_tank_pressure_kpa = state.h2_tank.pressure_kpa()
        result.o2_tank_pressure_kpa = state.o2_tank.pressure_kpa()
        result.h2_tank_fill = state.h2_tank.fill_fraction()
        result.o2_tank_fill = state.o2_tank.fill_fraction()
        result.halted = True
        result.warnings = warnings
        return result

    # ── Temperature check ──
    temp_eff = temperature_efficiency(cell_temp_c)
    if temp_eff <= 0:
        warnings.append("FROZEN: Cell temp %.1f°C below minimum %.1f°C"
                         % (cell_temp_c, MIN_OPERATING_TEMP_C))
        result.halted = True
        result.cell_health = 1.0 - state.cell_degradation
        result.h2_tank_pressure_kpa = state.h2_tank.pressure_kpa()
        result.o2_tank_pressure_kpa = state.o2_tank.pressure_kpa()
        result.h2_tank_fill = state.h2_tank.fill_fraction()
        result.o2_tank_fill = state.o2_tank.fill_fraction()
        result.warnings = warnings
        return result

    # ── Determine production limits ──
    water_available_kg = max(0.0, water_available_kg)
    power_kwh = max(0.0, power_kwh)

    # Limit 1: power budget
    h2_from_power = max_h2_from_power(power_kwh, state.cell_degradation, cell_temp_c)

    # Limit 2: water supply
    h2_from_water = max_h2_from_water(water_available_kg)

    # Limit 3: H₂ tank headroom
    h2_tank_room = state.h2_tank.headroom_kg()

    # Limit 4: O₂ tank headroom (co-product must fit)
    o2_per_h2 = O2_MASS_FRACTION / H2_MASS_FRACTION  # ~7.936
    o2_tank_room = state.o2_tank.headroom_kg()
    h2_limited_by_o2_tank = o2_tank_room / o2_per_h2 if o2_per_h2 > 0 else 0.0

    # Take the minimum of all limits
    h2_produced = min(h2_from_power, h2_from_water, h2_tank_room, h2_limited_by_o2_tank)
    h2_produced = max(0.0, h2_produced)

    # ── Calculate actual consumption and production ──
    o2_produced = o2_from_h2(h2_produced)
    water_consumed = water_for_h2(h2_produced)
    actual_cost = energy_per_kg_h2(state.cell_degradation, cell_temp_c)
    energy_consumed = h2_produced * actual_cost if h2_produced > 0 else 0.0

    # ── Store gases ──
    h2_stored = state.h2_tank.add(h2_produced)
    o2_stored = state.o2_tank.add(o2_produced)

    # ── Update totals ──
    state.total_water_consumed_kg += water_consumed
    state.total_h2_produced_kg += h2_stored
    state.total_o2_produced_kg += o2_stored
    state.total_energy_consumed_kwh += energy_consumed

    # ── Populate result ──
    result.water_consumed_kg = round(water_consumed, 6)
    result.h2_produced_kg = round(h2_stored, 6)
    result.o2_produced_kg = round(o2_stored, 6)
    result.energy_consumed_kwh = round(energy_consumed, 6)
    result.efficiency = system_efficiency(actual_cost) if h2_produced > 0 else 0.0
    result.h2_tank_pressure_kpa = state.h2_tank.pressure_kpa()
    result.o2_tank_pressure_kpa = state.o2_tank.pressure_kpa()
    result.h2_tank_fill = state.h2_tank.fill_fraction()
    result.o2_tank_fill = state.o2_tank.fill_fraction()
    result.cell_health = 1.0 - state.cell_degradation

    # ── Warnings ──
    if state.h2_tank.fill_fraction() > 0.85:
        warnings.append("H2_TANK_HIGH: %.0f%% full" % (state.h2_tank.fill_fraction() * 100))
    if state.o2_tank.fill_fraction() > 0.85:
        warnings.append("O2_TANK_HIGH: %.0f%% full" % (state.o2_tank.fill_fraction() * 100))
    if water_available_kg < water_consumed * 2:
        warnings.append("LOW_WATER: Supply running thin")
    if state.cell_degradation > 0.25:
        warnings.append("CELL_WEAR: %.0f%% degraded — schedule maintenance"
                         % (state.cell_degradation * 100))
    if power_kwh < MIN_POWER_KW * 24.66:
        warnings.append("LOW_POWER: Below minimum membrane sustain threshold")
    if h2_produced == 0 and water_available_kg > 0 and power_kwh > 0:
        warnings.append("NO_OUTPUT: Tanks full or system constrained")

    result.warnings = warnings
    return result


def create_electrolyzer(strategy: str = "balanced") -> ElectrolyzerState:
    """Create an electrolyzer configured for a colony strategy.

    conservative: large tanks, fresh cells — maximum reserves
    balanced: moderate tanks, standard config
    aggressive: small tanks, frequent turnover — prioritizes throughput
    """
    configs = {
        "conservative": ElectrolyzerState(
            h2_tank=GasTank(volume_l=1000.0, max_pressure_kpa=H2_TANK_MAX_KPA,
                            molar_mass_kg=H2_MOLAR_MASS_KG),
            o2_tank=GasTank(volume_l=2000.0, max_pressure_kpa=O2_TANK_MAX_KPA,
                            molar_mass_kg=O2_MOLAR_MASS_KG),
        ),
        "balanced": ElectrolyzerState(),
        "aggressive": ElectrolyzerState(
            h2_tank=GasTank(volume_l=250.0, max_pressure_kpa=H2_TANK_MAX_KPA,
                            molar_mass_kg=H2_MOLAR_MASS_KG),
            o2_tank=GasTank(volume_l=500.0, max_pressure_kpa=O2_TANK_MAX_KPA,
                            molar_mass_kg=O2_MOLAR_MASS_KG),
        ),
    }
    return configs.get(strategy, configs["balanced"])
