"""perchlorate_scrubber.py — Mars Perchlorate Remediation System.

Models three complementary perchlorate (ClO₄⁻) removal pathways for
making Mars regolith safe for agriculture and habitat use.

Pathways modelled
-----------------
* **Thermal decomposition** (primary) — Heat regolith above 400 °C.
  NaClO₄ → NaCl + 2 O₂.  Energy-intensive but liberates useful oxygen
  as a byproduct.  At 450–500 °C achieves > 95 % removal.

* **Iron reduction** (chemical) — Zero-valent iron (ZVI) reduces ClO₄⁻
  to Cl⁻ in aqueous solution.  Lower energy than thermal, but consumes
  iron feedstock and water.  Mars regolith is ~18 % Fe₂O₃ so iron can
  be sourced locally.

* **UV photocatalysis** (supplementary) — TiO₂ catalyst + Mars UV
  radiation breaks perchlorates in solution.  Slow but uses free
  sunlight.  Mars receives 2–3× Earth UV flux (no ozone layer).

Physical references
-------------------
- Phoenix lander (2008): 0.4–0.6 wt % perchlorate in soil
- Curiosity SAM: 0.5–1.0 wt % in Gale Crater
- NaClO₄ (122.44 g/mol) → NaCl (58.44 g/mol) + 2 O₂ (64.0 g/mol)
- Human toxicity: thyroid disruption > 15 µg/L in drinking water
- Plant growth inhibition: > 100 ppm perchlorate in soil
- ZVI reaction: ~0.5 kg Fe consumed per kg ClO₄⁻ reduced

Conservation laws
-----------------
Thermal decomposition mass balance per mol NaClO₄ destroyed::

    122.44 g → 58.44 g NaCl + 2 × 32.0 g O₂   (mass conserved)

The only mass leaving the solid phase is O₂ gas from thermal
decomposition.  Therefore::

    regolith_in = clean_soil_out + o2_released

One tick = one sol.  Mass in kg, energy in kWh, concentration in ppm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Physical constants ────────────────────────────────────────────────

# Molar masses (g/mol)
PERCHLORATE_MOLAR_MASS = 99.45       # ClO₄⁻ ion
NACLO4_MOLAR_MASS = 122.44           # NaClO₄
NACL_MOLAR_MASS = 58.44              # NaCl
O2_MOLAR_MASS = 32.0                 # O₂

# Thermal decomposition parameters
THERMAL_DECOMP_TEMP_C = 450.0        # target operating temperature (°C)
THERMAL_DECOMP_MIN_C = 400.0         # minimum for any decomposition (°C)
MARS_AMBIENT_TEMP_C = -60.0          # Mars surface average (°C)
REGOLITH_SPECIFIC_HEAT = 0.84        # kJ/(kg·°C), Mars basalt
HEATING_EFFICIENCY = 0.75            # thermal system efficiency

# Sigmoid parameters for decomposition curve
SIGMOID_K = 0.07                     # steepness
MAX_DECOMP_FRACTION = 0.99           # asymptotic maximum

# Safety thresholds
SAFE_SOIL_PERCHLORATE_PPM = 100.0    # maximum for agriculture
SAFE_WATER_PERCHLORATE_UG_L = 15.0   # maximum for drinking water

# Regolith properties
MARS_REGOLITH_PERCHLORATE_FRACTION = 0.007  # mass fraction (Phoenix avg)

# Iron reduction (chemical path)
IRON_PER_KG_PERCHLORATE = 0.5        # kg Fe consumed per kg ClO₄⁻ reduced
WATER_PER_KG_REGOLITH = 2.0          # L water for washing / treatment
IRON_MAX_EFFICIENCY = 0.85           # maximum removal efficiency via iron

# Processing capacity
MAX_BATCH_KG = 500.0                 # kg regolith per sol

# UV photocatalysis (supplementary)
UV_EFFICIENCY_FACTOR = 0.15          # fraction of remaining removed by UV

# Equipment degradation
EQUIPMENT_DEGRADATION_PER_SOL = 0.0003   # health loss per sol
CRITICAL_HEALTH = 0.1                    # CRITICAL alert threshold
WARNING_HEALTH = 0.3                     # WARNING / quality-degradation threshold
CRITICAL_THROUGHPUT_FACTOR = 0.3         # throughput multiplier when CRITICAL

# Chemical path energy (kWh per kg regolith for pumping / mixing)
CHEMICAL_ENERGY_PER_KG = 0.0003          # 0.3 kWh per tonne


# ── State ─────────────────────────────────────────────────────────────

@dataclass
class PerchlorateState:
    """Mutable state of the perchlorate scrubber across sols.

    Tracks equipment configuration, consumable reserves, and cumulative
    production totals.  All masses in kg, energy in kWh, volumes in L.
    """

    batch_capacity_kg: float = 200.0
    thermal_unit_count: int = 1
    iron_reserve_kg: float = 50.0
    water_budget_L: float = 100.0
    regolith_processed_kg: float = 0.0
    perchlorate_destroyed_kg: float = 0.0
    o2_liberated_kg: float = 0.0
    clean_soil_produced_kg: float = 0.0
    salt_byproduct_kg: float = 0.0
    total_energy_kwh: float = 0.0
    sols_running: int = 0
    equipment_health: float = 1.0
    alert: str = "NOMINAL"

    def __post_init__(self) -> None:
        """Clamp fields to physically valid ranges."""
        self.batch_capacity_kg = max(0.0, self.batch_capacity_kg)
        self.thermal_unit_count = max(0, self.thermal_unit_count)
        self.iron_reserve_kg = max(0.0, self.iron_reserve_kg)
        self.water_budget_L = max(0.0, self.water_budget_L)
        self.regolith_processed_kg = max(0.0, self.regolith_processed_kg)
        self.perchlorate_destroyed_kg = max(0.0, self.perchlorate_destroyed_kg)
        self.o2_liberated_kg = max(0.0, self.o2_liberated_kg)
        self.clean_soil_produced_kg = max(0.0, self.clean_soil_produced_kg)
        self.salt_byproduct_kg = max(0.0, self.salt_byproduct_kg)
        self.total_energy_kwh = max(0.0, self.total_energy_kwh)
        self.sols_running = max(0, self.sols_running)
        self.equipment_health = max(0.0, min(1.0, self.equipment_health))


@dataclass
class ScrubberTickResult:
    """Result of one sol of perchlorate scrubbing.

    Contains per-tick deltas (not cumulative) for every output stream
    plus diagnostic fields like pathway fractions and alert status.
    """

    regolith_in_kg: float = 0.0
    perchlorate_removed_kg: float = 0.0
    clean_soil_out_kg: float = 0.0
    o2_released_kg: float = 0.0
    salt_produced_kg: float = 0.0
    energy_used_kwh: float = 0.0
    thermal_fraction: float = 0.0
    chemical_fraction: float = 0.0
    soil_perchlorate_ppm: float = 7000.0
    water_used_L: float = 0.0
    iron_consumed_kg: float = 0.0
    alert: str = "NOMINAL"


# ── Pure physics functions ────────────────────────────────────────────

def thermal_energy_kwh(mass_kg: float, start_temp_c: float,
                       target_temp_c: float) -> float:
    """Energy required to heat *mass_kg* of regolith, in kWh.

    Uses specific heat of Mars basalt (0.84 kJ/(kg·°C)) and accounts
    for heating-system efficiency losses.  Returns 0 for non-positive
    mass or when the target temperature is at or below start.
    """
    mass_kg = max(0.0, mass_kg)
    delta_t = max(0.0, target_temp_c - start_temp_c)
    if mass_kg == 0.0 or delta_t == 0.0:
        return 0.0
    energy_kj = mass_kg * REGOLITH_SPECIFIC_HEAT * delta_t
    energy_kwh = energy_kj / 3600.0
    return energy_kwh / HEATING_EFFICIENCY


def thermal_decomp_fraction(temp_c: float) -> float:
    """Fraction of perchlorate thermally decomposed at *temp_c*.

    Returns 0 below 400 °C, ramps via sigmoid to ~0.50 at 450 °C,
    ~0.96 at 500 °C, and approaches 0.99 at 600 °C+.
    """
    if temp_c < THERMAL_DECOMP_MIN_C:
        return 0.0
    exponent = -SIGMOID_K * (temp_c - THERMAL_DECOMP_TEMP_C)
    exponent = max(-50.0, min(50.0, exponent))
    return MAX_DECOMP_FRACTION / (1.0 + math.exp(exponent))


def perchlorate_mass_kg(regolith_kg: float,
                        concentration: float = MARS_REGOLITH_PERCHLORATE_FRACTION,
                        ) -> float:
    """Mass of perchlorate contained in *regolith_kg* at *concentration*."""
    return max(0.0, regolith_kg) * max(0.0, min(1.0, concentration))


def o2_from_perchlorate_kg(perchlorate_kg: float) -> float:
    """O₂ liberated by thermal decomposition of perchlorate (kg).

    Stoichiometry: NaClO₄ → NaCl + 2 O₂
    O₂ mass = perchlorate_mass × (2 × 32.0 / 122.44).
    """
    return max(0.0, perchlorate_kg) * (2.0 * O2_MOLAR_MASS / NACLO4_MOLAR_MASS)


def salt_from_perchlorate_kg(perchlorate_kg: float) -> float:
    """NaCl produced from perchlorate decomposition (kg).

    Stoichiometry: NaClO₄ → NaCl + 2 O₂
    NaCl mass = perchlorate_mass × (58.44 / 122.44).
    """
    return max(0.0, perchlorate_kg) * (NACL_MOLAR_MASS / NACLO4_MOLAR_MASS)


def iron_reduction_rate(iron_kg: float, water_L: float,
                        regolith_kg: float) -> float:
    """Fraction of perchlorate removable via iron reduction.

    Limited by iron stock, available water for slurry, and the
    inherent efficiency ceiling of the ZVI process (~85 %).
    Returns 0 when any input is non-positive.
    """
    if iron_kg <= 0.0 or water_L <= 0.0 or regolith_kg <= 0.0:
        return 0.0
    perchlorate_kg = regolith_kg * MARS_REGOLITH_PERCHLORATE_FRACTION
    if perchlorate_kg <= 0.0:
        return 0.0

    iron_capacity_kg = iron_kg / IRON_PER_KG_PERCHLORATE
    iron_fraction = min(1.0, iron_capacity_kg / perchlorate_kg)

    water_needed = regolith_kg * WATER_PER_KG_REGOLITH
    water_fraction = min(1.0, water_L / water_needed)

    return min(iron_fraction, water_fraction) * IRON_MAX_EFFICIENCY


def residual_perchlorate_ppm(initial_concentration: float,
                             fraction_removed: float) -> float:
    """Residual perchlorate in treated soil (ppm).

    Args:
        initial_concentration: mass fraction (e.g. 0.007 for 0.7 %).
        fraction_removed: 0–1 fraction of perchlorate destroyed.
    """
    initial_concentration = max(0.0, initial_concentration)
    fraction_removed = max(0.0, min(1.0, fraction_removed))
    initial_ppm = initial_concentration * 1_000_000.0
    return initial_ppm * (1.0 - fraction_removed)


def assess_alert(soil_ppm: float, equipment_health: float) -> str:
    """Classify system alert level.

    Returns ``'CRITICAL'``, ``'WARNING'``, or ``'NOMINAL'`` based on
    equipment health and output soil quality.
    """
    if equipment_health < CRITICAL_HEALTH:
        return "CRITICAL"
    if equipment_health < WARNING_HEALTH:
        return "WARNING"
    if soil_ppm > SAFE_SOIL_PERCHLORATE_PPM * 50:
        return "WARNING"
    return "NOMINAL"


# ── Tick function ─────────────────────────────────────────────────────

def tick_perchlorate(
    state: PerchlorateState,
    power_available_kwh: float = 50.0,
    regolith_input_kg: float = 200.0,
    input_perchlorate_fraction: float = MARS_REGOLITH_PERCHLORATE_FRACTION,
    ambient_temp_c: float = MARS_AMBIENT_TEMP_C,
    use_thermal: bool = True,
    use_chemical: bool = True,
) -> Tuple[PerchlorateState, ScrubberTickResult]:
    """Advance the scrubber system by one sol.

    Processes *regolith_input_kg* of Mars regolith through the enabled
    remediation pathways (thermal, chemical/iron, UV supplementary) and
    returns the updated state together with a per-sol result snapshot.

    The UV pathway is always active (uses sunlight, no power cost) and
    operates on whatever perchlorate remains after thermal and chemical
    treatment.

    Args:
        state: current scrubber state (mutated in place **and** returned).
        power_available_kwh: electrical power budget for this sol.
        regolith_input_kg: raw regolith to process.
        input_perchlorate_fraction: ClO₄⁻ mass fraction in input.
        ambient_temp_c: starting temperature of regolith.
        use_thermal: enable thermal decomposition pathway.
        use_chemical: enable iron-reduction pathway.

    Returns:
        Tuple of (updated_state, tick_result).
    """
    # ── 0. Clamp inputs ──────────────────────────────────────────────
    power_available_kwh = max(0.0, power_available_kwh)
    regolith_input_kg = max(0.0, regolith_input_kg)
    input_perchlorate_fraction = max(0.0, min(1.0, input_perchlorate_fraction))

    # ── 1. Cap regolith by capacity and equipment health ─────────────
    effective_capacity = state.batch_capacity_kg * max(1, state.thermal_unit_count)
    if state.equipment_health < CRITICAL_HEALTH:
        effective_capacity *= CRITICAL_THROUGHPUT_FACTOR
    regolith_kg = min(regolith_input_kg, effective_capacity, MAX_BATCH_KG)

    # ── 2. Perchlorate mass in this batch ────────────────────────────
    perchlorate_kg = perchlorate_mass_kg(regolith_kg, input_perchlorate_fraction)

    # ── 3. Thermal decomposition path ────────────────────────────────
    thermal_removed_kg = 0.0
    energy_used = 0.0

    if use_thermal and perchlorate_kg > 0.0 and state.thermal_unit_count > 0:
        energy_needed = thermal_energy_kwh(
            regolith_kg, ambient_temp_c, THERMAL_DECOMP_TEMP_C,
        )
        if energy_needed > 0.0:
            energy_scale = min(1.0, power_available_kwh / energy_needed)
        else:
            energy_scale = 1.0

        actual_temp = ambient_temp_c + (
            THERMAL_DECOMP_TEMP_C - ambient_temp_c
        ) * energy_scale
        decomp_frac = thermal_decomp_fraction(actual_temp)

        # Degraded equipment reduces output quality
        if state.equipment_health < WARNING_HEALTH:
            decomp_frac *= state.equipment_health / WARNING_HEALTH

        thermal_removed_kg = perchlorate_kg * decomp_frac
        energy_used = energy_needed * energy_scale

    remaining_power = max(0.0, power_available_kwh - energy_used)
    remaining_perchlorate = perchlorate_kg - thermal_removed_kg

    # ── 4. Chemical / iron-reduction path ────────────────────────────
    chemical_removed_kg = 0.0
    iron_consumed = 0.0
    water_used = 0.0

    if use_chemical and remaining_perchlorate > 0.0:
        chem_frac = iron_reduction_rate(
            state.iron_reserve_kg, state.water_budget_L, regolith_kg,
        )
        if chem_frac > 0.0:
            chemical_removed_kg = remaining_perchlorate * chem_frac
            iron_consumed = chemical_removed_kg * IRON_PER_KG_PERCHLORATE
            water_used = min(
                state.water_budget_L,
                regolith_kg * WATER_PER_KG_REGOLITH,
            )
            chem_energy = CHEMICAL_ENERGY_PER_KG * regolith_kg
            energy_used += min(remaining_power, chem_energy)

    # ── 5. UV supplementary (always active, uses sunlight) ───────────
    uv_remaining = perchlorate_kg - thermal_removed_kg - chemical_removed_kg
    uv_removed_kg = 0.0
    if uv_remaining > 0.0:
        uv_removed_kg = uv_remaining * UV_EFFICIENCY_FACTOR

    # ── 6. Total removal — cap at 99 % ──────────────────────────────
    total_removed_kg = thermal_removed_kg + chemical_removed_kg + uv_removed_kg

    if perchlorate_kg > 0.0:
        total_fraction = total_removed_kg / perchlorate_kg
        if total_fraction > 0.99:
            scale = 0.99 / total_fraction
            thermal_removed_kg *= scale
            chemical_removed_kg *= scale
            uv_removed_kg *= scale
            total_removed_kg = perchlorate_kg * 0.99
            iron_consumed = chemical_removed_kg * IRON_PER_KG_PERCHLORATE
            total_fraction = 0.99
    else:
        total_fraction = 0.0

    # ── 7. Compute outputs ───────────────────────────────────────────
    # O₂ only from thermally-decomposed perchlorate
    o2_kg = o2_from_perchlorate_kg(thermal_removed_kg)
    # Salt from all destroyed perchlorate (all pathways yield Cl⁻)
    salt_kg = salt_from_perchlorate_kg(total_removed_kg)
    # Clean soil: input mass minus O₂ that escaped as gas
    clean_soil_kg = regolith_kg - o2_kg

    # Output soil perchlorate level
    output_ppm = residual_perchlorate_ppm(
        input_perchlorate_fraction, total_fraction,
    )

    # Pathway contribution fractions
    if total_removed_kg > 0.0:
        t_frac = thermal_removed_kg / total_removed_kg
        c_frac = (chemical_removed_kg + uv_removed_kg) / total_removed_kg
    else:
        t_frac = 0.0
        c_frac = 0.0

    # ── 8. Update state ──────────────────────────────────────────────
    state.regolith_processed_kg += regolith_kg
    state.perchlorate_destroyed_kg += total_removed_kg
    state.o2_liberated_kg += o2_kg
    state.clean_soil_produced_kg += clean_soil_kg
    state.salt_byproduct_kg += salt_kg
    state.total_energy_kwh += energy_used
    state.iron_reserve_kg = max(0.0, state.iron_reserve_kg - iron_consumed)
    state.water_budget_L = max(0.0, state.water_budget_L - water_used * 0.05)
    state.sols_running += 1
    state.equipment_health = max(
        0.0, state.equipment_health - EQUIPMENT_DEGRADATION_PER_SOL,
    )
    state.alert = assess_alert(output_ppm, state.equipment_health)

    # ── 9. Build result ──────────────────────────────────────────────
    result = ScrubberTickResult(
        regolith_in_kg=regolith_kg,
        perchlorate_removed_kg=total_removed_kg,
        clean_soil_out_kg=clean_soil_kg,
        o2_released_kg=o2_kg,
        salt_produced_kg=salt_kg,
        energy_used_kwh=energy_used,
        thermal_fraction=t_frac,
        chemical_fraction=c_frac,
        soil_perchlorate_ppm=output_ppm,
        water_used_L=water_used,
        iron_consumed_kg=iron_consumed,
        alert=state.alert,
    )
    return state, result


# ── Factory ───────────────────────────────────────────────────────────

def create_scrubber(batch_capacity_kg: float = 200.0,
                    thermal_units: int = 1) -> PerchlorateState:
    """Create a fresh perchlorate scrubber with default consumables.

    Args:
        batch_capacity_kg: regolith capacity per batch (kg).
        thermal_units: number of thermal reactor units.

    Returns:
        Initialised :class:`PerchlorateState`.
    """
    return PerchlorateState(
        batch_capacity_kg=max(0.0, batch_capacity_kg),
        thermal_unit_count=max(0, thermal_units),
    )
