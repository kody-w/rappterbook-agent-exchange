"""nitrogen_generator.py — Mars Habitat Nitrogen Extraction System.

The colony's invisible lifeline.  Habitat air is ~78% nitrogen by
volume — not for breathing, but as an inert buffer gas that prevents
oxygen toxicity and suppresses fire.  Mars atmosphere is only 2.7% N₂
(vs 78% on Earth).  Every airlock cycle, every micro-leak, every EVA
bleeds nitrogen into the Martian void.

Without active nitrogen extraction and replenishment, the habitat
atmosphere slowly enriches in O₂ until either:
  (a) Crew develops oxygen toxicity (O₂ > 50 kPa → seizures, death)
  (b) A spark ignites the O₂-rich air (flammability limit ~25% O₂)

This module models cryogenic separation of Mars atmosphere to extract
N₂, store it, and replenish habitat losses.

Physics modelled
----------------
* **Intake compression** — Mars air (~636 Pa) compressed to working
  pressure (~200 kPa) for separation.  Compressor work via polytropic
  process: W = (n/(n-1)) · P₁V₁ · ((P₂/P₁)^((n-1)/n) - 1).
* **CO₂ freeze-out** — at 200 kPa, CO₂ freezes at -78.5°C (dry ice).
  Removing 95.3% of intake mass as solid CO₂ concentrates N₂ from
  2.7% to ~57% in the remaining gas stream.
* **Cryogenic distillation** — remaining N₂/Ar/CO mixture separated
  by fractional distillation.  N₂ boils at -195.8°C (77 K), Ar at
  -185.8°C (87 K).  Energy for cryocooler: ~1.2 kWh per kg N₂.
* **Storage** — high-pressure N₂ tanks (20 MPa).  Tank capacity
  finite; overpressure vents to waste.
* **Habitat injection** — controlled release to maintain target N₂
  partial pressure.  Proportional control: inject enough to close gap.
* **Leak losses** — habitat N₂ loss from airlock cycling, seal
  degradation, and micro-meteorite damage.  Modelled as fraction
  of total N₂ per sol.

Conservation laws
-----------------
- Mass of N₂ extracted ≤ mass of Mars air processed × N₂ mass fraction
- Energy consumed ≥ thermodynamic minimum for compression + cooling
- Tank level: never negative, never exceeds capacity
- Habitat N₂: injection ≤ tank contents

Mars atmosphere composition (by volume):
  CO₂: 95.32%    N₂: 2.70%    Ar: 1.60%    O₂: 0.13%
  CO: 0.08%       H₂O: 0.03%  (trace NO, Ne, Kr, Xe)

Reference:
  - Mars atmospheric pressure: 636 Pa average (varies 400–870 Pa)
  - ISS N₂ resupply: ~0.5 kg/day (Progress tankers)
  - ISS leak rate: ~0.005% volume/day
  - Hab airlock N₂ loss: ~0.7 kg per full cycle at 70 kPa
  - O₂ toxicity: >50 kPa causes pulmonary edema, seizures
  - Fire risk threshold: O₂ > 25% volume at 70 kPa

One tick = one sol.  Mass in kg, pressure in kPa, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ── Physical constants ──────────────────────────────────────────────

# Mars atmosphere
MARS_SURFACE_PRESSURE_PA = 636.0     # average surface pressure
MARS_N2_FRACTION = 0.027             # 2.7% N₂ by volume
MARS_CO2_FRACTION = 0.9532           # 95.32% CO₂
MARS_AR_FRACTION = 0.016             # 1.6% Ar

# Molar masses (g/mol)
N2_MOLAR_MASS = 28.014
CO2_MOLAR_MASS = 44.01
AR_MOLAR_MASS = 39.948

# Compression
COMPRESSOR_OUTLET_PA = 200_000.0     # 200 kPa working pressure
POLYTROPIC_INDEX = 1.3               # between isothermal (1.0) and adiabatic (1.4)
COMPRESSOR_EFFICIENCY = 0.70         # mechanical efficiency

# CO₂ freeze-out
CO2_FREEZE_TEMP_C = -78.5            # sublimation point at ~200 kPa
FREEZE_OUT_EFFICIENCY = 0.98         # fraction of CO₂ removed as dry ice

# Cryogenic distillation
CRYO_KWH_PER_KG_N2 = 1.2            # energy to separate and liquefy N₂
DISTILLATION_N2_RECOVERY = 0.92      # fraction of N₂ recovered from gas stream

# Storage
TANK_CAPACITY_KG = 500.0             # max N₂ storage
TANK_PRESSURE_MPA = 20.0             # storage pressure (200 atm)

# Habitat parameters
HABITAT_VOLUME_M3 = 500.0           # pressurized volume
HABITAT_PRESSURE_KPA = 70.0         # reduced-pressure Mars habitat
TARGET_N2_FRACTION = 0.78           # 78% N₂ by volume
TARGET_N2_KPA = HABITAT_PRESSURE_KPA * TARGET_N2_FRACTION  # 54.6 kPa

# N₂ density at habitat conditions (~70 kPa, 20°C)
# PV = nRT → ρ = PM/(RT) = (54600 × 0.028) / (8.314 × 293) ≈ 0.628 kg/m³
N2_DENSITY_KG_M3 = 0.628

# Total habitat N₂ mass: ρ × V = 0.628 × 500 = 314 kg
HABITAT_N2_TOTAL_KG = N2_DENSITY_KG_M3 * HABITAT_VOLUME_M3

# Loss rates
AIRLOCK_N2_LOSS_KG = 0.7            # N₂ lost per airlock cycle
SEAL_LEAK_FRACTION = 0.0005         # 0.05% of habitat N₂ per sol (micro-leaks)
EMERGENCY_LEAK_KG = 5.0             # N₂ lost in a micro-meteorite event

# Safety thresholds
N2_LOW_WARNING_KPA = 45.0           # crew experiences O₂ enrichment effects
N2_CRITICAL_KPA = 35.0              # fire risk, O₂ toxicity imminent
O2_FIRE_THRESHOLD_FRACTION = 0.30   # flammability danger above 30% O₂

# Intake rate limits
MAX_INTAKE_M3_SOL = 5000.0          # max Mars air intake per sol (m³)


# ── State ───────────────────────────────────────────────────────────

@dataclass
class NitrogenState:
    """State of the nitrogen extraction and storage system."""

    # Storage
    tank_kg: float = 200.0              # N₂ in high-pressure tanks
    tank_capacity_kg: float = TANK_CAPACITY_KG

    # Habitat atmosphere
    hab_n2_kpa: float = TARGET_N2_KPA   # current N₂ partial pressure
    hab_n2_mass_kg: float = HABITAT_N2_TOTAL_KG

    # Cumulative
    total_extracted_kg: float = 0.0
    total_injected_kg: float = 0.0
    total_lost_kg: float = 0.0
    total_energy_kwh: float = 0.0

    # Operational
    sols_running: int = 0
    compressor_hours: float = 0.0
    intake_rate_m3_sol: float = 2000.0  # current intake setting

    # Alerts
    alert: str = "nominal"

    def __post_init__(self) -> None:
        self.tank_kg = max(0.0, min(self.tank_capacity_kg, self.tank_kg))
        self.hab_n2_kpa = max(0.0, self.hab_n2_kpa)
        self.hab_n2_mass_kg = max(0.0, self.hab_n2_mass_kg)
        self.total_extracted_kg = max(0.0, self.total_extracted_kg)
        self.total_injected_kg = max(0.0, self.total_injected_kg)
        self.total_lost_kg = max(0.0, self.total_lost_kg)
        self.total_energy_kwh = max(0.0, self.total_energy_kwh)
        self.intake_rate_m3_sol = max(0.0, min(MAX_INTAKE_M3_SOL,
                                                self.intake_rate_m3_sol))


@dataclass
class NitrogenTickResult:
    """Result of one sol of nitrogen system operation."""
    n2_extracted_kg: float = 0.0
    n2_injected_kg: float = 0.0
    n2_lost_kg: float = 0.0
    co2_byproduct_kg: float = 0.0
    compression_kwh: float = 0.0
    cryo_kwh: float = 0.0
    total_energy_kwh: float = 0.0
    hab_n2_kpa: float = 0.0
    tank_level_fraction: float = 0.0
    alert: str = "nominal"


# ── Pure physics functions ──────────────────────────────────────────

def mars_air_density_kg_m3(pressure_pa: float = MARS_SURFACE_PRESSURE_PA,
                           temperature_k: float = 210.0) -> float:
    """Density of Mars atmosphere at given conditions.

    Uses ideal gas law: ρ = P·M / (R·T)
    Mars air ~95% CO₂ → effective molar mass ≈ 43.3 g/mol.
    """
    pressure_pa = max(0.0, pressure_pa)
    temperature_k = max(1.0, temperature_k)
    effective_molar_mass = (MARS_CO2_FRACTION * CO2_MOLAR_MASS
                            + MARS_N2_FRACTION * N2_MOLAR_MASS
                            + MARS_AR_FRACTION * AR_MOLAR_MASS) / 1000.0
    return pressure_pa * effective_molar_mass / (8.314 * temperature_k)


def n2_in_mars_air_kg(intake_m3: float,
                      air_density: float = 0.0) -> float:
    """Mass of N₂ in a given volume of Mars air (kg).

    Parameters
    ----------
    intake_m3 : float
        Volume of Mars air processed (m³ at surface conditions).
    air_density : float
        Density of Mars air (kg/m³).  If 0, uses standard conditions.
    """
    intake_m3 = max(0.0, intake_m3)
    if air_density <= 0.0:
        air_density = mars_air_density_kg_m3()
    air_density = max(0.0, air_density)
    total_air_mass = intake_m3 * air_density
    n2_mass_frac = (MARS_N2_FRACTION * N2_MOLAR_MASS
                    / (MARS_CO2_FRACTION * CO2_MOLAR_MASS
                       + MARS_N2_FRACTION * N2_MOLAR_MASS
                       + MARS_AR_FRACTION * AR_MOLAR_MASS))
    return total_air_mass * n2_mass_frac


def compression_energy_kwh(intake_m3: float,
                           p_in: float = MARS_SURFACE_PRESSURE_PA,
                           p_out: float = COMPRESSOR_OUTLET_PA) -> float:
    """Energy to compress Mars air from surface pressure to working pressure.

    Polytropic compression: W = (n/(n-1)) · P₁V · ((P₂/P₁)^((n-1)/n) - 1)
    Divide by compressor efficiency.  Convert J to kWh.
    """
    intake_m3 = max(0.0, intake_m3)
    p_in = max(1.0, p_in)
    p_out = max(p_in, p_out)
    n = POLYTROPIC_INDEX
    ratio = p_out / p_in
    exponent = (n - 1.0) / n

    work_j_per_m3 = (n / (n - 1.0)) * p_in * (ratio ** exponent - 1.0)
    total_work_j = work_j_per_m3 * intake_m3 / max(0.01, COMPRESSOR_EFFICIENCY)
    return total_work_j / 3_600_000.0


def distillation_energy_kwh(n2_mass_kg: float) -> float:
    """Energy for cryogenic distillation to separate N₂ (kWh)."""
    return max(0.0, n2_mass_kg) * CRYO_KWH_PER_KG_N2


def n2_after_freeze_out(intake_m3: float,
                        air_density: float = 0.0) -> float:
    """N₂ mass recovered after CO₂ freeze-out and distillation (kg).

    Pipeline: intake → compress → freeze CO₂ → distill N₂.
    """
    raw_n2 = n2_in_mars_air_kg(intake_m3, air_density)
    return raw_n2 * DISTILLATION_N2_RECOVERY


def co2_byproduct_kg(intake_m3: float,
                     air_density: float = 0.0) -> float:
    """CO₂ captured as dry ice byproduct (kg).  Useful for Sabatier."""
    intake_m3 = max(0.0, intake_m3)
    if air_density <= 0.0:
        air_density = mars_air_density_kg_m3()
    air_density = max(0.0, air_density)
    total_mass = intake_m3 * air_density
    co2_mass_frac = (MARS_CO2_FRACTION * CO2_MOLAR_MASS
                     / (MARS_CO2_FRACTION * CO2_MOLAR_MASS
                        + MARS_N2_FRACTION * N2_MOLAR_MASS
                        + MARS_AR_FRACTION * AR_MOLAR_MASS))
    return total_mass * co2_mass_frac * FREEZE_OUT_EFFICIENCY


def n2_loss_kg(airlock_cycles: int, seal_leak_frac: float,
               hab_n2_mass_kg: float,
               emergency: bool = False) -> float:
    """Total N₂ lost from habitat this sol (kg).

    Parameters
    ----------
    airlock_cycles : int
        Number of airlock depressurization cycles.
    seal_leak_frac : float
        Fraction of hab N₂ lost through seal micro-leaks per sol.
    hab_n2_mass_kg : float
        Current habitat N₂ mass.
    emergency : bool
        If True, add emergency leak (micro-meteorite, etc).
    """
    airlock_cycles = max(0, airlock_cycles)
    seal_leak_frac = max(0.0, min(0.1, seal_leak_frac))
    hab_n2_mass_kg = max(0.0, hab_n2_mass_kg)

    airlock_loss = airlock_cycles * AIRLOCK_N2_LOSS_KG
    seal_loss = hab_n2_mass_kg * seal_leak_frac
    emergency_loss = EMERGENCY_LEAK_KG if emergency else 0.0

    total = airlock_loss + seal_loss + emergency_loss
    return min(total, hab_n2_mass_kg)


def injection_needed_kg(hab_n2_kpa: float,
                        target_kpa: float = TARGET_N2_KPA) -> float:
    """Mass of N₂ to inject to restore target partial pressure (kg).

    Uses ideal gas: mass = ΔP × V × M / (R × T) at 293 K.
    """
    deficit_kpa = max(0.0, target_kpa - hab_n2_kpa)
    mass_kg = (deficit_kpa * 1000.0 * HABITAT_VOLUME_M3 * N2_MOLAR_MASS / 1000.0
               / (8.314 * 293.0))
    return max(0.0, mass_kg)


def kpa_from_mass(n2_mass_kg: float,
                  volume_m3: float = HABITAT_VOLUME_M3) -> float:
    """Convert N₂ mass (kg) to partial pressure (kPa) in habitat.

    P = m·R·T / (M·V)  where M is molar mass of N₂.
    """
    n2_mass_kg = max(0.0, n2_mass_kg)
    volume_m3 = max(0.01, volume_m3)
    pressure_pa = (n2_mass_kg * 8.314 * 293.0) / (N2_MOLAR_MASS / 1000.0 * volume_m3)
    return pressure_pa / 1000.0


def assess_alert(hab_n2_kpa: float) -> str:
    """Determine alert level from habitat N₂ partial pressure."""
    if hab_n2_kpa < N2_CRITICAL_KPA:
        return "critical"
    elif hab_n2_kpa < N2_LOW_WARNING_KPA:
        return "warning"
    return "nominal"


# ── Tick function ───────────────────────────────────────────────────

def tick_nitrogen(state: NitrogenState,
                  airlock_cycles: int = 4,
                  emergency: bool = False,
                  power_available_kwh: float = 100.0) -> tuple:
    """Advance nitrogen system by one sol.

    Parameters
    ----------
    state : NitrogenState
        Current system state.
    airlock_cycles : int
        Number of airlock cycles this sol.
    emergency : bool
        If True, a micro-meteorite event causes extra N₂ loss.
    power_available_kwh : float
        Energy budget for this sol.

    Returns
    -------
    tuple[NitrogenState, NitrogenTickResult]
        Updated state and tick result.
    """
    result = NitrogenTickResult()

    # ── N₂ losses ───────────────────────────────────────────────────
    lost = n2_loss_kg(airlock_cycles, SEAL_LEAK_FRACTION,
                      state.hab_n2_mass_kg, emergency)
    state.hab_n2_mass_kg = max(0.0, state.hab_n2_mass_kg - lost)
    state.hab_n2_kpa = kpa_from_mass(state.hab_n2_mass_kg)
    state.total_lost_kg += lost
    result.n2_lost_kg = lost

    # ── Extraction (power-limited) ──────────────────────────────────
    intake = state.intake_rate_m3_sol
    comp_kwh = compression_energy_kwh(intake)
    raw_n2 = n2_after_freeze_out(intake)
    cryo_kwh = distillation_energy_kwh(raw_n2)
    total_energy = comp_kwh + cryo_kwh

    # Scale down if not enough power
    if total_energy > power_available_kwh and total_energy > 0:
        scale = power_available_kwh / total_energy
        intake *= scale
        comp_kwh *= scale
        raw_n2 *= scale
        cryo_kwh *= scale
        total_energy = power_available_kwh

    # Store in tanks (cap at capacity)
    space_in_tank = max(0.0, state.tank_capacity_kg - state.tank_kg)
    stored = min(raw_n2, space_in_tank)
    state.tank_kg += stored
    state.total_extracted_kg += stored
    state.total_energy_kwh += total_energy

    result.n2_extracted_kg = stored
    result.compression_kwh = comp_kwh
    result.cryo_kwh = cryo_kwh
    result.total_energy_kwh = total_energy
    result.co2_byproduct_kg = co2_byproduct_kg(intake)

    # ── Injection (replenish habitat) ───────────────────────────────
    needed = injection_needed_kg(state.hab_n2_kpa)
    injected = min(needed, state.tank_kg)
    state.tank_kg -= injected
    state.hab_n2_mass_kg += injected
    state.hab_n2_kpa = kpa_from_mass(state.hab_n2_mass_kg)
    state.total_injected_kg += injected
    result.n2_injected_kg = injected

    # ── Status ──────────────────────────────────────────────────────
    result.hab_n2_kpa = state.hab_n2_kpa
    result.tank_level_fraction = (state.tank_kg / state.tank_capacity_kg
                                  if state.tank_capacity_kg > 0 else 0.0)
    alert = assess_alert(state.hab_n2_kpa)
    result.alert = alert
    state.alert = alert

    state.sols_running += 1
    state.compressor_hours += 24.65  # one sol of operation

    return state, result


# ── Factory ─────────────────────────────────────────────────────────

def create_nitrogen_system(tank_kg: float = 200.0) -> NitrogenState:
    """Create a nitrogen system with given initial tank reserves."""
    return NitrogenState(tank_kg=max(0.0, min(TANK_CAPACITY_KG, tank_kg)))
