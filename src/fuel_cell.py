"""fuel_cell.py -- Mars Colony PEM Fuel Cell Backup Power

The inverse of water_electrolysis.py.  Where the electrolyzer splits
water into H₂ + O₂ using electricity, the fuel cell recombines them
to PRODUCE electricity + water + heat.  It is the colony's last-resort
power source: when solar panels are buried in dust and the reactor
SCRAMs, the fuel cell keeps life support alive on stored hydrogen.

Physics
-------
* **PEM fuel cell** — Proton Exchange Membrane.  Same membrane chemistry
  as the electrolyzer, run in reverse.  Anode: H₂ → 2H⁺ + 2e⁻.
  Cathode: ½O₂ + 2H⁺ + 2e⁻ → H₂O.  Net: H₂ + ½O₂ → H₂O + energy.
* **Thermodynamics** — Gibbs free energy ΔG° = −237 kJ/mol at 25 °C.
  Reversible voltage E° = ΔG / (n·F) = 1.229 V per cell.  In practice,
  activation, ohmic, and mass-transport losses reduce operating voltage
  to 0.55–0.80 V per cell depending on current draw.
* **Efficiency** — Electrical: η_e = V_cell / E_thermo ≈ 45–65%.
  Combined heat+electric (CHP): η_chp ≈ 80–90%.  We model both.
* **Stoichiometry** — 2H₂ + O₂ → 2H₂O.  By mass: 1 kg H₂ + 7.937 kg
  O₂ → 8.937 kg H₂O.  Energy per kg H₂ at η_e = 55%: ~18.3 kWh.
* **Degradation** — PEM cells lose 2–10 µV/hr under steady load
  (~0.05–0.25 mV/sol).  Modelled as voltage decay per operating sol.
  At 80% of original voltage → end of life.
* **Thermal output** — Waste heat = chemical energy − electrical output.
  At η_e = 55%, ~45% of HHV emerges as heat (≈15 kWh/kg H₂).
  Habitat can capture this for heating (Mars ambient −63 °C average).
* **Cold start** — PEM membranes need >0 °C.  On Mars, startup heater
  draws from battery_bank.  Below −20 °C, membrane damage risk.
* **Water recovery** — Product water is ultrapure distilled.  Feeds
  directly back to water_reclamation, closing the loop.

Conservation laws enforced every tick:
  mass_in(H₂ + O₂)  = mass_out(H₂O)   (to 6 decimal places)
  energy_chemical     = energy_electric + energy_heat  (1st law)

Reference systems:
  - NASA Gemini: first PEM fuel cell in space (1 kW)
  - Apollo: 3× Bacon alkaline fuel cells, 1.5 kW each
  - Space Shuttle: 3× 12 kW Orbiter PEM stacks
  - ISS: no fuel cells (solar+battery), but studied for lunar gateway
  - Mars DRA 5.0: 10 kW PEM backup for ISRU plant

One tick = one sol.  Energy in kWh, mass in kg, voltage in V,
temperature in °C, power in kW.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Electrochemistry
FARADAY_C = 96_485.3329          # Coulombs per mole of electrons
R_GAS = 8.31446                  # J/(mol·K)
N_ELECTRONS = 2                  # electrons transferred per H₂ molecule
GIBBS_FREE_ENERGY_J = 237_200.0  # ΔG° at 25 °C, J/mol H₂O formed
H2_MOLAR_MASS_KG = 0.002016     # kg/mol
O2_MOLAR_MASS_KG = 0.032        # kg/mol
H2O_MOLAR_MASS_KG = 0.018015    # kg/mol
HHV_H2_KWH_KG = 39.41           # higher heating value of H₂, kWh/kg
LHV_H2_KWH_KG = 33.33           # lower heating value of H₂, kWh/kg

# Reversible (thermodynamic) cell voltage at STP
E_REVERSIBLE_V = GIBBS_FREE_ENERGY_J / (N_ELECTRONS * FARADAY_C)  # ~1.229 V

# Stoichiometric mass ratios (from 2H₂ + O₂ → 2H₂O)
O2_PER_H2_MASS = O2_MOLAR_MASS_KG / (2 * H2_MOLAR_MASS_KG)      # 7.937
# Derived from conservation: H₂O = H₂ + O₂ by mass (guarantees balance)
H2O_PER_H2_MASS = 1.0 + O2_PER_H2_MASS                           # 8.937

# Operating parameters
NOMINAL_CELL_VOLTAGE_V = 0.70    # typical PEM operating point
MIN_CELL_VOLTAGE_V = 0.40        # below this → cell reversal damage
MAX_CELL_VOLTAGE_V = E_REVERSIBLE_V  # open-circuit, no load
CELLS_PER_STACK = 120            # series cells in one stack
STACK_VOLTAGE_V = CELLS_PER_STACK * NOMINAL_CELL_VOLTAGE_V  # 84 V

# Degradation
VOLTAGE_DECAY_MV_PER_SOL = 0.15  # ~6 µV/hr × 24.66 hr
EOL_VOLTAGE_FRACTION = 0.80      # end-of-life at 80% of nominal voltage

# Thermal
MARS_AMBIENT_C = -63.0
MIN_MEMBRANE_TEMP_C = 0.0        # below this → ice damage
COLD_START_HEATER_KWH = 2.0      # energy to warm stack from Mars ambient
OPTIMAL_TEMP_C = 70.0            # PEM sweet spot
TEMP_EFFICIENCY_COEFF = 0.002    # efficiency gain per °C above 25°C

# System
SOL_HOURS = 24.66
DEFAULT_H2_CAPACITY_KG = 50.0    # onboard H₂ supply
DEFAULT_O2_CAPACITY_KG = 400.0   # onboard O₂ supply


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FuelCellStack:
    """State of a single PEM fuel cell stack."""

    num_cells: int = CELLS_PER_STACK
    nominal_cell_v: float = NOMINAL_CELL_VOLTAGE_V
    voltage_decay_mv_sol: float = VOLTAGE_DECAY_MV_PER_SOL
    operating_sols: float = 0.0
    total_h2_consumed_kg: float = 0.0
    total_energy_kwh: float = 0.0
    total_water_produced_kg: float = 0.0
    total_heat_kwh: float = 0.0
    membrane_temp_c: float = 25.0
    is_running: bool = False

    def cell_voltage(self) -> float:
        """Current per-cell voltage accounting for degradation."""
        decay = self.operating_sols * self.voltage_decay_mv_sol / 1000.0
        v = self.nominal_cell_v - decay
        return max(v, MIN_CELL_VOLTAGE_V)

    def stack_voltage(self) -> float:
        """Total stack voltage (cells in series)."""
        return self.num_cells * self.cell_voltage()

    def health(self) -> float:
        """Stack health as fraction of nominal voltage (1.0 = new)."""
        return self.cell_voltage() / self.nominal_cell_v

    def is_eol(self) -> bool:
        """True if stack has degraded past end-of-life threshold."""
        return self.health() < EOL_VOLTAGE_FRACTION

    def electrical_efficiency(self) -> float:
        """Electrical efficiency: V_cell / E_reversible."""
        return self.cell_voltage() / E_REVERSIBLE_V

    def chp_efficiency(self) -> float:
        """Combined heat + power efficiency (theoretical max ~95%)."""
        eta_e = self.electrical_efficiency()
        # CHP captures waste heat; realistic capture ~85% of waste
        eta_heat_capture = 0.85
        waste_fraction = 1.0 - (eta_e * E_REVERSIBLE_V / LHV_H2_KWH_KG
                                * FARADAY_C * N_ELECTRONS / GIBBS_FREE_ENERGY_J)
        # Simplified: η_chp ≈ η_e + 0.85 × (1 − η_e) but capped at 0.95
        eta_chp = eta_e + eta_heat_capture * (1.0 - eta_e)
        return min(eta_chp, 0.95)


@dataclass
class FuelCellState:
    """Complete fuel cell system state for the colony."""

    stack: FuelCellStack = field(default_factory=FuelCellStack)
    h2_supply_kg: float = DEFAULT_H2_CAPACITY_KG
    h2_capacity_kg: float = DEFAULT_H2_CAPACITY_KG
    o2_supply_kg: float = DEFAULT_O2_CAPACITY_KG
    o2_capacity_kg: float = DEFAULT_O2_CAPACITY_KG
    water_produced_kg: float = 0.0
    cold_start_energy_kwh: float = 0.0
    emergency_mode: bool = False

    def h2_fill(self) -> float:
        """Fraction of H₂ supply remaining."""
        if self.h2_capacity_kg <= 0:
            return 0.0
        return self.h2_supply_kg / self.h2_capacity_kg

    def o2_fill(self) -> float:
        """Fraction of O₂ supply remaining."""
        if self.o2_capacity_kg <= 0:
            return 0.0
        return self.o2_supply_kg / self.o2_capacity_kg

    def runtime_hours(self, power_kw: float) -> float:
        """Estimate hours of runtime at given power draw.

        Returns hours until H₂ or O₂ runs out, whichever is first.
        """
        if power_kw <= 0:
            return float("inf")
        h2_rate = h2_consumption_rate(power_kw, self.stack.cell_voltage())
        if h2_rate <= 0:
            return float("inf")
        o2_rate = h2_rate * O2_PER_H2_MASS
        h2_hours = self.h2_supply_kg / h2_rate if h2_rate > 0 else float("inf")
        o2_hours = self.o2_supply_kg / o2_rate if o2_rate > 0 else float("inf")
        return min(h2_hours, o2_hours)


@dataclass
class TickResult:
    """Output of one sol of fuel cell operation."""

    energy_produced_kwh: float = 0.0
    heat_produced_kwh: float = 0.0
    h2_consumed_kg: float = 0.0
    o2_consumed_kg: float = 0.0
    water_produced_kg: float = 0.0
    cell_voltage_v: float = 0.0
    stack_voltage_v: float = 0.0
    electrical_efficiency: float = 0.0
    stack_health: float = 1.0
    membrane_temp_c: float = 25.0
    runtime_remaining_h: float = 0.0
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure physics functions
# ---------------------------------------------------------------------------

def nernst_voltage(temp_c: float, p_h2_atm: float = 1.0,
                   p_o2_atm: float = 0.21) -> float:
    """Nernst equation: reversible cell voltage at given T and pressures.

    E = E° + (RT/nF) · ln(pH2 · √pO2)
    Higher temperature and pressure → higher voltage.
    """
    temp_k = temp_c + 273.15
    if temp_k <= 0:
        return 0.0
    if p_h2_atm <= 0 or p_o2_atm <= 0:
        return 0.0
    # Temperature correction on E° (empirical: −0.85 mV/K from 298 K)
    e0_corrected = E_REVERSIBLE_V - 0.00085 * (temp_k - 298.15)
    log_term = math.log(p_h2_atm * math.sqrt(p_o2_atm))
    e_nernst = e0_corrected + (R_GAS * temp_k / (N_ELECTRONS * FARADAY_C)) * log_term
    return max(e_nernst, 0.0)


def h2_consumption_rate(power_kw: float, cell_voltage_v: float) -> float:
    """H₂ consumption rate in kg/hour for a given electrical power output.

    From Faraday's law: I = P / V_stack.  Each amp-second consumes
    H₂_molar_mass / (n·F) kg of hydrogen.

    Simplified: kg_H₂/hr = P_kW / (η_e × LHV_kWh/kg)
    where η_e = V_cell / E_reversible.
    """
    if cell_voltage_v <= 0 or power_kw <= 0:
        return 0.0
    eta_e = cell_voltage_v / E_REVERSIBLE_V
    eta_e = min(eta_e, 1.0)
    if eta_e <= 0:
        return 0.0
    return power_kw / (eta_e * LHV_H2_KWH_KG)


def energy_from_h2(h2_kg: float, cell_voltage_v: float) -> float:
    """Electrical energy (kWh) produced from a given mass of H₂.

    energy = h2_kg × LHV × η_e
    """
    if h2_kg <= 0 or cell_voltage_v <= 0:
        return 0.0
    eta_e = min(cell_voltage_v / E_REVERSIBLE_V, 1.0)
    return h2_kg * LHV_H2_KWH_KG * eta_e


def heat_from_h2(h2_kg: float, cell_voltage_v: float) -> float:
    """Waste heat (kWh) produced from a given mass of H₂.

    heat = h2_kg × LHV × (1 − η_e)
    First law: chemical energy = electrical + heat.
    """
    if h2_kg <= 0 or cell_voltage_v <= 0:
        return 0.0
    eta_e = min(cell_voltage_v / E_REVERSIBLE_V, 1.0)
    return h2_kg * LHV_H2_KWH_KG * (1.0 - eta_e)


def water_from_h2(h2_kg: float) -> float:
    """Water produced (kg) from consuming h2_kg of hydrogen.

    Stoichiometry: 2H₂ + O₂ → 2H₂O.
    Mass ratio: 1 kg H₂ → 8.937 kg H₂O.
    """
    return h2_kg * H2O_PER_H2_MASS


def o2_for_h2(h2_kg: float) -> float:
    """O₂ required (kg) to react with h2_kg of hydrogen.

    Mass ratio: 1 kg H₂ requires 7.937 kg O₂.
    """
    return h2_kg * O2_PER_H2_MASS


def membrane_warmup(current_temp_c: float, heat_kwh: float,
                    thermal_mass_kwh_per_c: float = 0.5) -> float:
    """New membrane temperature after absorbing waste heat.

    Simple lumped thermal model.  Heat raises temperature;
    radiation to Mars ambient cools it.  Steady state at
    operating temperature.
    """
    if thermal_mass_kwh_per_c <= 0:
        return current_temp_c
    delta_from_heat = heat_kwh / thermal_mass_kwh_per_c
    # Radiative cooling toward Mars ambient (simplified Newton cooling)
    cooling_rate = 0.02  # fraction of delta-T lost per sol
    delta_cooling = (current_temp_c - MARS_AMBIENT_C) * cooling_rate
    new_temp = current_temp_c + delta_from_heat - delta_cooling
    return new_temp


# ---------------------------------------------------------------------------
# Tick function — advances the fuel cell by one sol
# ---------------------------------------------------------------------------

def tick(state: FuelCellState, power_demand_kw: float = 0.0,
         sol_fraction: float = 1.0) -> TickResult:
    """Advance the fuel cell system by one sol (or fraction thereof).

    Parameters
    ----------
    state : FuelCellState
        Mutable system state.  Modified in place.
    power_demand_kw : float
        Requested average power output in kW.  The cell delivers up to
        this amount, limited by fuel, oxidizer, and stack health.
        If 0, the cell idles (no fuel consumed, stack not degraded).
    sol_fraction : float
        Fraction of a sol to simulate (default 1.0 = full sol).

    Returns
    -------
    TickResult with energy/heat/mass accounting and diagnostics.
    """
    result = TickResult()
    warnings: List[str] = []
    hours = SOL_HOURS * sol_fraction

    # ── Cold start check ──
    if power_demand_kw > 0 and not state.stack.is_running:
        if state.stack.membrane_temp_c < MIN_MEMBRANE_TEMP_C:
            state.cold_start_energy_kwh += COLD_START_HEATER_KWH
            state.stack.membrane_temp_c = MIN_MEMBRANE_TEMP_C + 5.0
            warnings.append("COLD_START: Heater activated (%.1f kWh)"
                            % COLD_START_HEATER_KWH)
        state.stack.is_running = True

    # ── Idle mode ──
    if power_demand_kw <= 0:
        state.stack.is_running = False
        # Membrane cools toward ambient when idle
        state.stack.membrane_temp_c = membrane_warmup(
            state.stack.membrane_temp_c, 0.0)
        result.membrane_temp_c = state.stack.membrane_temp_c
        result.stack_health = state.stack.health()
        result.cell_voltage_v = state.stack.cell_voltage()
        result.stack_voltage_v = state.stack.stack_voltage()
        result.runtime_remaining_h = state.runtime_hours(0.0)
        result.warnings = warnings
        return result

    # ── End-of-life check ──
    if state.stack.is_eol():
        warnings.append("END_OF_LIFE: Stack at %.0f%% health — replace"
                         % (state.stack.health() * 100))
        state.stack.is_running = False
        result.stack_health = state.stack.health()
        result.cell_voltage_v = state.stack.cell_voltage()
        result.warnings = warnings
        return result

    # ── Calculate fuel demand ──
    cell_v = state.stack.cell_voltage()
    h2_rate_kg_hr = h2_consumption_rate(power_demand_kw, cell_v)
    h2_demand_kg = h2_rate_kg_hr * hours
    o2_demand_kg = o2_for_h2(h2_demand_kg)

    # ── Limit by available reactants ──
    h2_available = state.h2_supply_kg
    o2_available = state.o2_supply_kg

    # H₂ limited
    if h2_demand_kg > h2_available:
        h2_demand_kg = h2_available
        warnings.append("H2_LIMITED: Only %.2f kg available" % h2_available)
    # O₂ limited
    o2_needed = o2_for_h2(h2_demand_kg)
    if o2_needed > o2_available:
        h2_demand_kg = o2_available / O2_PER_H2_MASS if O2_PER_H2_MASS > 0 else 0.0
        o2_needed = o2_available
        warnings.append("O2_LIMITED: Only %.2f kg available" % o2_available)

    # ── Reaction ──
    h2_consumed = max(0.0, h2_demand_kg)
    o2_consumed = o2_for_h2(h2_consumed)
    water_produced = water_from_h2(h2_consumed)
    energy_kwh = energy_from_h2(h2_consumed, cell_v)
    heat_kwh = heat_from_h2(h2_consumed, cell_v)

    # ── Conservation law check (mass) ──
    mass_in = h2_consumed + o2_consumed
    mass_out = water_produced
    assert abs(mass_in - mass_out) < 1e-6, (
        f"Mass conservation violated: in={mass_in:.6f} out={mass_out:.6f}")

    # ── Conservation law check (energy, 1st law) ──
    chemical_energy = h2_consumed * LHV_H2_KWH_KG
    assert abs(chemical_energy - energy_kwh - heat_kwh) < 1e-6, (
        f"Energy conservation violated: chem={chemical_energy:.6f} "
        f"elec={energy_kwh:.6f} heat={heat_kwh:.6f}")

    # ── Update state ──
    state.h2_supply_kg -= h2_consumed
    state.o2_supply_kg -= o2_consumed
    state.water_produced_kg += water_produced
    state.stack.total_h2_consumed_kg += h2_consumed
    state.stack.total_energy_kwh += energy_kwh
    state.stack.total_water_produced_kg += water_produced
    state.stack.total_heat_kwh += heat_kwh

    # Degradation: only when running under load
    state.stack.operating_sols += sol_fraction

    # Thermal: waste heat warms the membrane
    state.stack.membrane_temp_c = membrane_warmup(
        state.stack.membrane_temp_c, heat_kwh * 0.1)  # 10% stays in stack

    # ── Warnings ──
    if state.h2_fill() < 0.15:
        warnings.append("H2_LOW: %.0f%% remaining" % (state.h2_fill() * 100))
        if state.h2_fill() < 0.05:
            state.emergency_mode = True
            warnings.append("EMERGENCY: H2 critically low")
    if state.o2_fill() < 0.15:
        warnings.append("O2_LOW: %.0f%% remaining" % (state.o2_fill() * 100))
    if state.stack.health() < 0.85:
        warnings.append("DEGRADED: Stack at %.0f%% health"
                         % (state.stack.health() * 100))
    if state.stack.membrane_temp_c > 90.0:
        warnings.append("OVERTEMP: Membrane at %.1f°C — reduce load"
                         % state.stack.membrane_temp_c)

    # ── Populate result ──
    result.energy_produced_kwh = round(energy_kwh, 6)
    result.heat_produced_kwh = round(heat_kwh, 6)
    result.h2_consumed_kg = round(h2_consumed, 6)
    result.o2_consumed_kg = round(o2_consumed, 6)
    result.water_produced_kg = round(water_produced, 6)
    result.cell_voltage_v = cell_v
    result.stack_voltage_v = state.stack.stack_voltage()
    result.electrical_efficiency = state.stack.electrical_efficiency()
    result.stack_health = state.stack.health()
    result.membrane_temp_c = state.stack.membrane_temp_c
    result.runtime_remaining_h = state.runtime_hours(power_demand_kw)
    result.warnings = warnings

    return result


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_fuel_cell(profile: str = "colony") -> FuelCellState:
    """Create a fuel cell system for the given deployment profile.

    colony   : 50 kg H₂, 400 kg O₂ — full habitat backup
    rover    : 5 kg H₂, 40 kg O₂ — mobile power for EVA rover
    emergency: 10 kg H₂, 80 kg O₂ — lifeboat minimum
    """
    profiles = {
        "colony": FuelCellState(
            h2_supply_kg=50.0, h2_capacity_kg=50.0,
            o2_supply_kg=400.0, o2_capacity_kg=400.0,
        ),
        "rover": FuelCellState(
            h2_supply_kg=5.0, h2_capacity_kg=5.0,
            o2_supply_kg=40.0, o2_capacity_kg=40.0,
            stack=FuelCellStack(num_cells=60),
        ),
        "emergency": FuelCellState(
            h2_supply_kg=10.0, h2_capacity_kg=10.0,
            o2_supply_kg=80.0, o2_capacity_kg=80.0,
            stack=FuelCellStack(num_cells=30),
        ),
    }
    return profiles.get(profile, profiles["colony"])
