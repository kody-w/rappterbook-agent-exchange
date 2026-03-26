"""solar_panel_cleaner.py -- Automated Solar Panel Dust Removal for Mars.

Dust killed Opportunity.  After 5,111 sols, a planet-encircling dust
storm dropped optical depth to τ > 10 and the rover's solar panels
couldn't keep the heaters alive.  InSight watched its power fall from
4.6 kWh/sol to 0.5 kWh/sol over 1,400 sols of slow dust burial.

A permanent colony cannot rely on lucky dust devils.  This module
builds the cleaning infrastructure that keeps the lights on:

Physics modelled
----------------
* **Electrostatic Dust Removal (EDR)** — Transparent ITO electrodes
  embedded in the panel coverglass generate a traveling-wave electric
  field (1–10 kV, 1–10 Hz).  Mars dust is triboelectrically charged
  to ~10⁻¹⁵ C per particle from aeolian transport.  The wave sweeps
  charged particles off the surface.  Efficiency ~90 % on fresh panels,
  degrades with coverglass wear.  Power: ~0.5 W/m² during activation.
  Reference: Mazumder et al., "Self-Cleaning Transparent Dust Shields"
  (NASA/CR-2013-217961).

* **Mechanical wiper** — Motor-driven blade sweeps dust in one pass.
  Simple and reliable.  Micro-scratches accumulate, reducing panel
  transmittance ~0.01 % per cleaning cycle.  Power: 5 W per cycle
  (small DC motor).  30-second cycle time.

* **Compressed CO₂ blast** — Mars atmosphere is 95.3 % CO₂ at
  ~600 Pa.  A compressor stores gas at 200 kPa.  Nozzle blasts
  panels clean.  Good for heavy dust loads.  Power: 20 W for 60 s
  compressor run per panel area.  Reference: atmospheric processing
  studies for ISRU propellant production.

* **Dust adhesion** — van der Waals + electrostatic + capillary
  forces bind dust to glass.  Adhesion force per particle:
  F_adh ≈ A·d / (12·z²)  (Hamaker model, A ≈ 6.5e-20 J for
  silicate-on-glass, d = particle diameter, z = separation ~0.4 nm).
  Mars low humidity means negligible capillary forces.

* **Cleaning efficiency** — Each method removes a fraction of dust.
  Efficiency depends on dust load, particle size distribution,
  method-specific physics, and panel wear state.

* **Panel wear** — Mechanical cleaning scratches coverglass.
  Electrostatic cleaning is gentler.  CO₂ blast is intermediate.
  Scratches reduce optical transmittance and thus peak panel output.

Conservation laws
-----------------
* Energy balance: cleaning energy ≥ 0; net energy = power gained − power spent.
* Dust mass balance: dust removed from panels = dust collected + dust re-suspended.
* Wear monotonically increases (scratches don't heal).
* Dust fraction always in [0, 1].  Wear always in [0, 1].

One tick = one sol.  Energy in watt-hours, area in m², dust as fraction [0,1].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Physical constants ──────────────────────────────────────────────────

MARS_GRAVITY_M_S2 = 3.72
MARS_SURFACE_PRESSURE_PA = 636.0
MARS_CO2_FRACTION = 0.953

# Dust properties
DUST_PARTICLE_DIAMETER_UM = 1.5
DUST_PARTICLE_CHARGE_C = 1.0e-15
DUST_HAMAKER_CONSTANT_J = 6.5e-20
DUST_SEPARATION_M = 4.0e-10
DUST_DENSITY_KG_M3 = 3933.0

# Solar panel properties
PANEL_TRANSMITTANCE_CLEAN = 0.95
PANEL_EFFICIENCY_GAAS = 0.30
MARS_SOLAR_IRRADIANCE_W_M2 = 590.0
SOL_HOURS = 24.66
SOLAR_HOURS_PER_SOL = 6.0  # effective full-sun-equivalent hours

# Cleaning method parameters
EDR_POWER_W_PER_M2 = 0.5
EDR_ACTIVATION_SECONDS = 300.0
EDR_BASE_EFFICIENCY = 0.90
EDR_WEAR_PER_CYCLE = 0.00002

WIPER_POWER_W = 5.0
WIPER_CYCLE_SECONDS = 30.0
WIPER_BASE_EFFICIENCY = 0.85
WIPER_WEAR_PER_CYCLE = 0.0001

CO2_BLAST_POWER_W = 20.0
CO2_BLAST_SECONDS = 60.0
CO2_BLAST_BASE_EFFICIENCY = 0.80
CO2_BLAST_WEAR_PER_CYCLE = 0.00005

# Dust accumulation from solar_array.py for consistency
DUST_ACCUMULATION_PER_SOL = 0.0028
DUST_STORM_ACCUMULATION_PER_SOL = 0.015

# Cleaning thresholds
DUST_TRIGGER_THRESHOLD = 0.05
MAX_WEAR = 1.0


class CleaningMethod(Enum):
    """Available cleaning mechanisms."""
    ELECTROSTATIC = "electrostatic"
    WIPER = "mechanical_wiper"
    CO2_BLAST = "co2_blast"


def dust_adhesion_force_n(particle_diameter_m: float) -> float:
    """Van der Waals adhesion force for a single dust particle on glass.

    Uses the Hamaker model: F = A * d / (12 * z^2)
    where A = Hamaker constant, d = particle diameter, z = separation.
    Returns force in Newtons.  Always non-negative.
    """
    if particle_diameter_m <= 0.0:
        return 0.0
    return (DUST_HAMAKER_CONSTANT_J * particle_diameter_m
            / (12.0 * DUST_SEPARATION_M ** 2))


def electrostatic_removal_force_n(
    electric_field_v_m: float,
    particle_charge_c: float,
) -> float:
    """Force on a charged dust particle in an EDR traveling wave.

    F = q * E.  Always non-negative (magnitude).
    """
    return abs(electric_field_v_m * particle_charge_c)


def cleaning_efficiency(
    method: CleaningMethod,
    dust_fraction: float,
    panel_wear: float,
) -> float:
    """Fraction of dust removed by one cleaning cycle.

    Higher dust loads are slightly easier to clean (less adhesion per
    unit area when piled up).  More worn panels are harder to clean
    (scratches trap dust).

    Returns value in [0, 1].
    """
    if dust_fraction <= 0.0:
        return 0.0

    base = {
        CleaningMethod.ELECTROSTATIC: EDR_BASE_EFFICIENCY,
        CleaningMethod.WIPER: WIPER_BASE_EFFICIENCY,
        CleaningMethod.CO2_BLAST: CO2_BLAST_BASE_EFFICIENCY,
    }[method]

    # Wear degrades cleaning efficiency linearly
    wear_factor = max(0.0, 1.0 - panel_wear)

    # Heavy dust is slightly easier to dislodge (logarithmic bonus)
    dust_bonus = 1.0 + 0.05 * math.log1p(dust_fraction * 20.0)

    eff = base * wear_factor * dust_bonus
    return max(0.0, min(1.0, eff))


def cleaning_energy_wh(method: CleaningMethod, panel_area_m2: float) -> float:
    """Energy consumed by one cleaning cycle in watt-hours.

    Always non-negative.  Scales with panel area for EDR and CO2,
    fixed for wiper (one motor regardless of area, up to practical limits).
    """
    panel_area_m2 = max(0.0, panel_area_m2)

    if method == CleaningMethod.ELECTROSTATIC:
        watts = EDR_POWER_W_PER_M2 * panel_area_m2
        hours = EDR_ACTIVATION_SECONDS / 3600.0
        return watts * hours

    if method == CleaningMethod.WIPER:
        hours = WIPER_CYCLE_SECONDS / 3600.0
        return WIPER_POWER_W * hours

    if method == CleaningMethod.CO2_BLAST:
        hours = CO2_BLAST_SECONDS / 3600.0
        return CO2_BLAST_POWER_W * hours

    return 0.0


def wear_per_cycle(method: CleaningMethod) -> float:
    """Panel coverglass wear increment per cleaning cycle.

    Mechanical wipers cause the most wear.  Electrostatic is gentlest.
    Always non-negative.
    """
    return {
        CleaningMethod.ELECTROSTATIC: EDR_WEAR_PER_CYCLE,
        CleaningMethod.WIPER: WIPER_WEAR_PER_CYCLE,
        CleaningMethod.CO2_BLAST: CO2_BLAST_WEAR_PER_CYCLE,
    }[method]


def power_output_wh(
    panel_area_m2: float,
    dust_fraction: float,
    panel_wear: float,
    dust_storm_active: bool = False,
) -> float:
    """Solar power generated per sol in watt-hours.

    Accounts for dust coverage, panel wear (reduced transmittance),
    and dust storm dimming via Beer-Lambert law.

    Always non-negative.
    """
    panel_area_m2 = max(0.0, panel_area_m2)
    dust_fraction = max(0.0, min(1.0, dust_fraction))
    panel_wear = max(0.0, min(1.0, panel_wear))

    transmittance = PANEL_TRANSMITTANCE_CLEAN * (1.0 - panel_wear)
    dust_factor = 1.0 - dust_fraction

    # Dust storm dims sunlight via Beer-Lambert: I = I₀ · exp(-τ)
    storm_factor = math.exp(-2.0) if dust_storm_active else 1.0

    irradiance = MARS_SOLAR_IRRADIANCE_W_M2 * storm_factor
    watts = (irradiance * panel_area_m2 * transmittance
             * dust_factor * PANEL_EFFICIENCY_GAAS)

    return max(0.0, watts * SOLAR_HOURS_PER_SOL)


def net_energy_gain_wh(
    method: CleaningMethod,
    panel_area_m2: float,
    dust_before: float,
    dust_after: float,
    panel_wear: float,
) -> float:
    """Net energy gain from one cleaning cycle over one sol.

    net = (power_after - power_before) - cleaning_energy

    Can be negative if cleaning costs more than it gains.
    """
    power_before = power_output_wh(panel_area_m2, dust_before, panel_wear)
    power_after = power_output_wh(panel_area_m2, dust_after, panel_wear)
    cost = cleaning_energy_wh(method, panel_area_m2)
    return (power_after - power_before) - cost


def should_clean(
    dust_fraction: float,
    panel_wear: float,
    panel_area_m2: float,
    method: CleaningMethod = CleaningMethod.ELECTROSTATIC,
) -> bool:
    """Decide whether cleaning is worthwhile right now.

    True if: dust above threshold AND net energy gain is positive.
    """
    if dust_fraction < DUST_TRIGGER_THRESHOLD:
        return False

    eff = cleaning_efficiency(method, dust_fraction, panel_wear)
    dust_after = dust_fraction * (1.0 - eff)
    gain = net_energy_gain_wh(method, panel_area_m2, dust_fraction,
                              dust_after, panel_wear)
    return gain > 0.0


# ── Stateful panel cleaner ──────────────────────────────────────────────

@dataclass
class PanelCleanerState:
    """Mutable state for one solar panel cleaning system."""
    panel_area_m2: float = 100.0
    dust_fraction: float = 0.0
    panel_wear: float = 0.0
    total_cleaning_cycles: int = 0
    total_energy_spent_wh: float = 0.0
    total_energy_gained_wh: float = 0.0
    total_dust_removed: float = 0.0
    sol: int = 0

    def __post_init__(self) -> None:
        """Clamp fields to valid physical ranges."""
        self.panel_area_m2 = max(0.0, self.panel_area_m2)
        self.dust_fraction = max(0.0, min(1.0, self.dust_fraction))
        self.panel_wear = max(0.0, min(MAX_WEAR, self.panel_wear))
        self.total_cleaning_cycles = max(0, self.total_cleaning_cycles)
        self.total_energy_spent_wh = max(0.0, self.total_energy_spent_wh)
        self.total_dust_removed = max(0.0, self.total_dust_removed)


def create_cleaner(
    panel_area_m2: float = 100.0,
    initial_dust: float = 0.0,
    initial_wear: float = 0.0,
) -> PanelCleanerState:
    """Factory: create a new panel cleaning system."""
    return PanelCleanerState(
        panel_area_m2=panel_area_m2,
        dust_fraction=initial_dust,
        panel_wear=initial_wear,
    )


def accumulate_dust(
    state: PanelCleanerState,
    dust_storm_active: bool = False,
) -> PanelCleanerState:
    """Add one sol of dust accumulation."""
    rate = (DUST_STORM_ACCUMULATION_PER_SOL if dust_storm_active
            else DUST_ACCUMULATION_PER_SOL)
    new_dust = min(1.0, state.dust_fraction + rate)
    return PanelCleanerState(
        panel_area_m2=state.panel_area_m2,
        dust_fraction=new_dust,
        panel_wear=state.panel_wear,
        total_cleaning_cycles=state.total_cleaning_cycles,
        total_energy_spent_wh=state.total_energy_spent_wh,
        total_energy_gained_wh=state.total_energy_gained_wh,
        total_dust_removed=state.total_dust_removed,
        sol=state.sol + 1,
    )


def clean_panels(
    state: PanelCleanerState,
    method: CleaningMethod = CleaningMethod.ELECTROSTATIC,
) -> PanelCleanerState:
    """Execute one cleaning cycle.  Returns new state."""
    eff = cleaning_efficiency(method, state.dust_fraction, state.panel_wear)
    dust_removed = state.dust_fraction * eff
    new_dust = max(0.0, state.dust_fraction - dust_removed)
    new_wear = min(MAX_WEAR, state.panel_wear + wear_per_cycle(method))
    energy_cost = cleaning_energy_wh(method, state.panel_area_m2)
    energy_gain = net_energy_gain_wh(
        method, state.panel_area_m2,
        state.dust_fraction, new_dust, state.panel_wear,
    )

    return PanelCleanerState(
        panel_area_m2=state.panel_area_m2,
        dust_fraction=new_dust,
        panel_wear=new_wear,
        total_cleaning_cycles=state.total_cleaning_cycles + 1,
        total_energy_spent_wh=state.total_energy_spent_wh + energy_cost,
        total_energy_gained_wh=state.total_energy_gained_wh + max(0.0, energy_gain),
        total_dust_removed=state.total_dust_removed + dust_removed,
        sol=state.sol,
    )


def tick(
    state: PanelCleanerState,
    dust_storm_active: bool = False,
    auto_clean: bool = True,
    preferred_method: CleaningMethod = CleaningMethod.ELECTROSTATIC,
) -> PanelCleanerState:
    """Advance the cleaner by one sol.

    1. Accumulate dust.
    2. If auto_clean and cleaning is worthwhile, clean.
    3. Return new state.
    """
    state = accumulate_dust(state, dust_storm_active=dust_storm_active)

    if auto_clean and should_clean(
        state.dust_fraction, state.panel_wear,
        state.panel_area_m2, preferred_method,
    ):
        state = clean_panels(state, method=preferred_method)

    return state


def simulate(
    n_sols: int,
    panel_area_m2: float = 100.0,
    dust_storm_start: int = -1,
    dust_storm_duration: int = 0,
    method: CleaningMethod = CleaningMethod.ELECTROSTATIC,
    auto_clean: bool = True,
) -> list[PanelCleanerState]:
    """Run a multi-sol simulation.  Returns list of states per sol."""
    state = create_cleaner(panel_area_m2=panel_area_m2)
    history: list[PanelCleanerState] = [state]

    for sol in range(n_sols):
        storm = (dust_storm_start <= sol
                 < dust_storm_start + dust_storm_duration)
        state = tick(
            state,
            dust_storm_active=storm,
            auto_clean=auto_clean,
            preferred_method=method,
        )
        history.append(state)

    return history


def to_dict(state: PanelCleanerState) -> dict[str, Any]:
    """Serialize state to a JSON-safe dict."""
    return {
        "panel_area_m2": state.panel_area_m2,
        "dust_fraction": round(state.dust_fraction, 6),
        "panel_wear": round(state.panel_wear, 6),
        "total_cleaning_cycles": state.total_cleaning_cycles,
        "total_energy_spent_wh": round(state.total_energy_spent_wh, 2),
        "total_energy_gained_wh": round(state.total_energy_gained_wh, 2),
        "total_dust_removed": round(state.total_dust_removed, 6),
        "sol": state.sol,
        "current_power_wh": round(
            power_output_wh(state.panel_area_m2, state.dust_fraction,
                            state.panel_wear),
            2,
        ),
    }
