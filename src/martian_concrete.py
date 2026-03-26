"""martian_concrete.py -- Sulfur Concrete Production for Mars Construction.

The colony can mine regolith, smelt ore, fabricate parts -- but it
cannot BUILD.  No bricks, no walls, no foundations.  Every habitat is
a prefab tin can shipped from Earth at $50,000/kg.

Sulfur concrete changes everything.  Mars regolith is ~5% sulfur by
mass.  Heat sulfur past its 115°C melting point, mix with sieved
regolith aggregate, pour into molds, let it set in Mars's -63°C
ambient.  The result: 35-50 MPa compressive strength -- comparable to
Portland cement.  No water required.  Fully recyclable by remelting.

This is not science fiction.  NASA's 2016 study (Wan et al.) and ESA's
2019 ISRU roadmap both identify sulfur concrete as the primary
candidate for Mars construction.  It is the first building material
that can be manufactured entirely from Martian soil.

Physics modelled
----------------
* Sulfur thermodynamics -- Enthalpy of fusion 54 kJ/kg (1.73 kJ/mol,
  M = 32.06 g/mol).  Melting point 115.2°C (388.4 K).  Working temp
  120-140°C for optimal viscosity.  Above 160°C sulfur polymerises and
  viscosity spikes -- unusable.

* Regolith processing -- Raw regolith sieved to <2mm aggregate.
  Yield depends on particle size distribution: ~70% passes 2mm sieve
  for typical Mars basaltic regolith.

* Mix design -- Optimal sulfur:aggregate ratio is 35:65 by mass.
  Too little sulfur (<25%): insufficient binder, crumbles.
  Too much sulfur (>45%): brittle, shrinkage cracks.

* Compressive strength -- Peaks ~45 MPa at 35% sulfur, drops
  parabolically away from optimum.  Porosity reduces strength
  exponentially.

* Cooling/setting -- Newton's law of cooling from pour temp to
  Mars ambient.  Setting time proportional to block volume^(2/3).
  Mars ambient (210K) means rapid setting -- advantage over Earth.

* Thermal cycling -- Mars diurnal swing ±40K degrades strength
  ~0.1% per sol via microcracking.  Annual fatigue life ~2000 sols
  before structural replacement needed.

* Energy budget -- Heating energy = mass_sulfur * Cp * dT + Hfusion.
  Electric kiln powered from colony grid.  ~150 kJ per kg of sulfur
  processed (heating from 210K to 400K).

* Production rate -- One mixer batch: 200 kg concrete per 2-hour
  cycle.  Limited by kiln capacity and power availability.

Conservation laws: mass_in = mass_out, energy >= 0, strength >= 0,
sulfur_fraction in [0, 1], temperature >= Mars ambient.

One tick = one sol.  Mass in kg, energy in kJ, strength in MPa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Physical constants ──────────────────────────────────────────────────

MARS_AMBIENT_K = 210.0
SULFUR_MELTING_K = 388.4
SULFUR_WORKING_K = 403.0           # 130°C — optimal viscosity
SULFUR_MAX_WORKING_K = 433.0       # 160°C — polymerisation limit
SULFUR_ENTHALPY_FUSION_KJ_KG = 54.0
SULFUR_SPECIFIC_HEAT_KJ_KG_K = 0.71
SULFUR_DENSITY_KG_M3 = 2070.0
REGOLITH_SULFUR_FRACTION = 0.05    # ~5% sulfur in Mars regolith
REGOLITH_SIEVE_YIELD = 0.70        # 70% passes <2mm sieve
REGOLITH_SPECIFIC_HEAT_KJ_KG_K = 0.80
REGOLITH_DENSITY_KG_M3 = 1500.0

OPTIMAL_SULFUR_RATIO = 0.35
MIN_SULFUR_RATIO = 0.20
MAX_SULFUR_RATIO = 0.50
PEAK_STRENGTH_MPA = 45.0

CONCRETE_DENSITY_KG_M3 = 2200.0
BLOCK_STANDARD_KG = 20.0           # standard block mass
BLOCK_VOLUME_M3 = BLOCK_STANDARD_KG / CONCRETE_DENSITY_KG_M3

KILN_CAPACITY_KG = 50.0            # sulfur per batch
MIXER_CAPACITY_KG = 200.0          # total concrete per batch
BATCH_CYCLE_HOURS = 2.0
HOURS_PER_SOL = 24.66
BATCHES_PER_SOL = math.floor(HOURS_PER_SOL / BATCH_CYCLE_HOURS)

COOLING_COEFFICIENT = 0.002        # Newton cooling constant (1/s)
THERMAL_CYCLE_DEGRADATION = 0.001  # strength loss fraction per sol
FATIGUE_LIFE_SOLS = 2000           # typical structural life

KILN_EFFICIENCY = 0.80             # thermal efficiency
MIXER_POWER_KW = 5.0               # mechanical mixing power
MIXER_TIME_HOURS = 0.5             # mixing time per batch


# ── Pure physics functions ──────────────────────────────────────────────

def heating_energy_kj(mass_kg: float, start_k: float, target_k: float,
                      specific_heat: float, enthalpy_fusion: float = 0.0,
                      melting_point_k: float = 0.0) -> float:
    """Energy to heat material, including phase change if crossed."""
    if mass_kg <= 0.0 or target_k <= start_k:
        return 0.0
    energy = mass_kg * specific_heat * (target_k - start_k)
    if 0.0 < melting_point_k <= target_k and start_k < melting_point_k:
        energy += mass_kg * enthalpy_fusion
    return energy


def sulfur_heating_energy_kj(mass_kg: float,
                             ambient_k: float = MARS_AMBIENT_K) -> float:
    """Total energy to melt and heat sulfur to working temperature."""
    return heating_energy_kj(
        mass_kg, ambient_k, SULFUR_WORKING_K,
        SULFUR_SPECIFIC_HEAT_KJ_KG_K, SULFUR_ENTHALPY_FUSION_KJ_KG,
        SULFUR_MELTING_K,
    )


def aggregate_from_regolith_kg(raw_regolith_kg: float) -> float:
    """Mass of sieved aggregate from raw regolith (<2mm fraction)."""
    if raw_regolith_kg <= 0.0:
        return 0.0
    return raw_regolith_kg * REGOLITH_SIEVE_YIELD


def sulfur_from_regolith_kg(raw_regolith_kg: float) -> float:
    """Mass of extractable sulfur from raw regolith."""
    if raw_regolith_kg <= 0.0:
        return 0.0
    return raw_regolith_kg * REGOLITH_SULFUR_FRACTION


def concrete_strength_mpa(sulfur_fraction: float,
                          porosity: float = 0.05) -> float:
    """Compressive strength as a function of mix design.

    Parabolic curve centred on optimal ratio.  Porosity reduces
    strength exponentially.  Returns 0 outside valid range.
    """
    if sulfur_fraction < MIN_SULFUR_RATIO or sulfur_fraction > MAX_SULFUR_RATIO:
        return 0.0
    porosity = max(0.0, min(porosity, 1.0))
    deviation = (sulfur_fraction - OPTIMAL_SULFUR_RATIO) / (
        MAX_SULFUR_RATIO - MIN_SULFUR_RATIO
    )
    strength = PEAK_STRENGTH_MPA * (1.0 - 4.0 * deviation * deviation)
    strength *= math.exp(-3.0 * porosity)
    return max(0.0, strength)


def cooling_time_seconds(volume_m3: float, pour_k: float,
                         ambient_k: float = MARS_AMBIENT_K,
                         target_k: float = 0.0) -> float:
    """Time to cool a block to target temperature (Newton's law).

    Default target: ambient + 10 K (set point).
    Time scales with volume^(2/3) (surface area / volume ratio).
    """
    if volume_m3 <= 0.0:
        return 0.0
    if target_k <= 0.0:
        target_k = ambient_k + 10.0
    if pour_k <= target_k:
        return 0.0
    dt_initial = pour_k - ambient_k
    dt_final = target_k - ambient_k
    if dt_initial <= 0.0 or dt_final <= 0.0:
        return 0.0
    char_length = volume_m3 ** (1.0 / 3.0)
    effective_coeff = COOLING_COEFFICIENT / max(char_length, 0.001)
    return math.log(dt_initial / dt_final) / effective_coeff


def thermal_fatigue_strength(initial_mpa: float, sols: int,
                             cycle_amplitude_k: float = 40.0) -> float:
    """Strength after thermal cycling degradation."""
    if initial_mpa <= 0.0 or sols <= 0:
        return max(0.0, initial_mpa)
    rate = THERMAL_CYCLE_DEGRADATION * (cycle_amplitude_k / 40.0)
    return initial_mpa * math.exp(-rate * sols)


def blocks_from_concrete_kg(concrete_kg: float) -> int:
    """Number of standard blocks from a mass of concrete."""
    if concrete_kg <= 0.0:
        return 0
    return int(concrete_kg / BLOCK_STANDARD_KG)


def regolith_needed_kg(target_concrete_kg: float,
                       sulfur_ratio: float = OPTIMAL_SULFUR_RATIO) -> float:
    """Raw regolith required for a target concrete mass.

    We need both sulfur (extracted from regolith) and aggregate (sieved
    from regolith).  They may come from different batches.
    """
    if target_concrete_kg <= 0.0:
        return 0.0
    sulfur_needed = target_concrete_kg * sulfur_ratio
    aggregate_needed = target_concrete_kg * (1.0 - sulfur_ratio)
    regolith_for_sulfur = sulfur_needed / max(REGOLITH_SULFUR_FRACTION, 1e-9)
    regolith_for_aggregate = aggregate_needed / max(REGOLITH_SIEVE_YIELD, 1e-9)
    return regolith_for_sulfur + regolith_for_aggregate


def batch_energy_kj(sulfur_kg: float, aggregate_kg: float,
                    ambient_k: float = MARS_AMBIENT_K) -> float:
    """Total energy for one batch: heating + mixing."""
    heat = sulfur_heating_energy_kj(sulfur_kg, ambient_k) / max(KILN_EFFICIENCY, 0.01)
    mix = MIXER_POWER_KW * MIXER_TIME_HOURS * 3600.0  # kW*s = kJ
    return heat + mix


# ── State dataclasses ───────────────────────────────────────────────────

@dataclass
class PlantState:
    """State of the sulfur concrete production plant."""
    sol: int = 0
    sulfur_stockpile_kg: float = 0.0
    aggregate_stockpile_kg: float = 0.0
    concrete_produced_kg: float = 0.0
    blocks_produced: int = 0
    blocks_placed: int = 0
    total_energy_consumed_kj: float = 0.0
    total_regolith_processed_kg: float = 0.0
    avg_strength_mpa: float = 0.0
    oldest_block_sol: int = 0
    kiln_temp_k: float = MARS_AMBIENT_K
    plant_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sol": self.sol,
            "sulfur_stockpile_kg": round(self.sulfur_stockpile_kg, 2),
            "aggregate_stockpile_kg": round(self.aggregate_stockpile_kg, 2),
            "concrete_produced_kg": round(self.concrete_produced_kg, 2),
            "blocks_produced": self.blocks_produced,
            "blocks_placed": self.blocks_placed,
            "total_energy_consumed_kj": round(self.total_energy_consumed_kj, 2),
            "total_regolith_processed_kg": round(self.total_regolith_processed_kg, 2),
            "avg_strength_mpa": round(self.avg_strength_mpa, 2),
            "oldest_block_sol": self.oldest_block_sol,
            "kiln_temp_k": round(self.kiln_temp_k, 2),
            "plant_active": self.plant_active,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlantState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TickResult:
    """Result of one sol of concrete production."""
    sol: int = 0
    batches_run: int = 0
    concrete_produced_kg: float = 0.0
    blocks_produced: int = 0
    sulfur_consumed_kg: float = 0.0
    aggregate_consumed_kg: float = 0.0
    energy_consumed_kj: float = 0.0
    batch_strength_mpa: float = 0.0
    regolith_intake_kg: float = 0.0
    cooling_time_s: float = 0.0
    plant_active: bool = True


# ── Factory functions ───────────────────────────────────────────────────

def create_plant(initial_regolith_kg: float = 0.0) -> PlantState:
    """Create a new concrete production plant.

    Optionally pre-load with processed regolith (split into sulfur
    and aggregate stockpiles).
    """
    state = PlantState()
    if initial_regolith_kg > 0.0:
        state.sulfur_stockpile_kg = sulfur_from_regolith_kg(initial_regolith_kg)
        state.aggregate_stockpile_kg = aggregate_from_regolith_kg(initial_regolith_kg)
        state.total_regolith_processed_kg = initial_regolith_kg
    return state


# ── Tick engine ─────────────────────────────────────────────────────────

def tick(state: PlantState,
         regolith_delivery_kg: float = 100.0,
         power_available_kj: float = float("inf"),
         sulfur_ratio: float = OPTIMAL_SULFUR_RATIO,
         ambient_k: float = MARS_AMBIENT_K,
         porosity: float = 0.05) -> TickResult:
    """Advance the concrete plant by one sol.

    Process: receive regolith → extract sulfur + sieve aggregate →
    run as many batches as stockpiles and power allow → pour blocks.
    """
    state.sol += 1
    result = TickResult(sol=state.sol, plant_active=state.plant_active)

    if not state.plant_active:
        return result

    # Clamp sulfur ratio to valid range
    sulfur_ratio = max(MIN_SULFUR_RATIO, min(sulfur_ratio, MAX_SULFUR_RATIO))

    # 1. Receive and process raw regolith
    if regolith_delivery_kg > 0.0:
        new_sulfur = sulfur_from_regolith_kg(regolith_delivery_kg)
        new_aggregate = aggregate_from_regolith_kg(regolith_delivery_kg)
        state.sulfur_stockpile_kg += new_sulfur
        state.aggregate_stockpile_kg += new_aggregate
        state.total_regolith_processed_kg += regolith_delivery_kg
        result.regolith_intake_kg = regolith_delivery_kg

    # 2. Run production batches
    energy_remaining = power_available_kj
    total_concrete = 0.0
    total_sulfur_used = 0.0
    total_agg_used = 0.0
    total_energy = 0.0

    for _ in range(BATCHES_PER_SOL):
        # How much concrete can this batch produce?
        sulfur_for_batch = min(state.sulfur_stockpile_kg,
                               MIXER_CAPACITY_KG * sulfur_ratio)
        agg_for_batch = min(state.aggregate_stockpile_kg,
                            MIXER_CAPACITY_KG * (1.0 - sulfur_ratio))

        # Scale batch to limiting resource
        if sulfur_ratio > 0:
            max_by_sulfur = sulfur_for_batch / sulfur_ratio
        else:
            max_by_sulfur = 0.0
        if (1.0 - sulfur_ratio) > 0:
            max_by_agg = agg_for_batch / (1.0 - sulfur_ratio)
        else:
            max_by_agg = 0.0
        batch_concrete = min(max_by_sulfur, max_by_agg, MIXER_CAPACITY_KG)

        if batch_concrete < BLOCK_STANDARD_KG:
            break  # Not enough for even one block

        sulfur_used = batch_concrete * sulfur_ratio
        agg_used = batch_concrete * (1.0 - sulfur_ratio)
        energy = batch_energy_kj(sulfur_used, agg_used, ambient_k)

        if energy > energy_remaining:
            break  # Not enough power

        # Commit the batch
        state.sulfur_stockpile_kg -= sulfur_used
        state.aggregate_stockpile_kg -= agg_used
        energy_remaining -= energy
        total_concrete += batch_concrete
        total_sulfur_used += sulfur_used
        total_agg_used += agg_used
        total_energy += energy
        result.batches_run += 1

    # 3. Pour blocks
    new_blocks = blocks_from_concrete_kg(total_concrete)
    strength = concrete_strength_mpa(sulfur_ratio, porosity)

    state.concrete_produced_kg += total_concrete
    state.blocks_produced += new_blocks
    state.total_energy_consumed_kj += total_energy

    # Running average strength
    if state.blocks_produced > 0:
        prev_total = state.avg_strength_mpa * (state.blocks_produced - new_blocks)
        state.avg_strength_mpa = (
            (prev_total + strength * new_blocks) / state.blocks_produced
        )

    if state.oldest_block_sol == 0 and new_blocks > 0:
        state.oldest_block_sol = state.sol

    # 4. Kiln temperature (cools toward ambient between batches)
    if result.batches_run > 0:
        state.kiln_temp_k = SULFUR_WORKING_K
    else:
        # Cool toward ambient
        dt = state.kiln_temp_k - ambient_k
        state.kiln_temp_k = ambient_k + dt * math.exp(
            -COOLING_COEFFICIENT * 3600.0 * HOURS_PER_SOL
        )

    # 5. Cooling time for today's blocks
    if new_blocks > 0:
        result.cooling_time_s = cooling_time_seconds(
            BLOCK_VOLUME_M3, SULFUR_WORKING_K, ambient_k
        )

    # Fill result
    result.concrete_produced_kg = total_concrete
    result.blocks_produced = new_blocks
    result.sulfur_consumed_kg = total_sulfur_used
    result.aggregate_consumed_kg = total_agg_used
    result.energy_consumed_kj = total_energy
    result.batch_strength_mpa = strength

    # Ensure non-negative stockpiles (floating point guard)
    state.sulfur_stockpile_kg = max(0.0, state.sulfur_stockpile_kg)
    state.aggregate_stockpile_kg = max(0.0, state.aggregate_stockpile_kg)

    return result


def run_simulation(state: PlantState, sols: int = 100,
                   regolith_per_sol_kg: float = 100.0,
                   power_per_sol_kj: float = float("inf"),
                   ambient_k: float = MARS_AMBIENT_K) -> list[TickResult]:
    """Run the concrete plant for multiple sols."""
    return [
        tick(state, regolith_delivery_kg=regolith_per_sol_kg,
             power_available_kj=power_per_sol_kj, ambient_k=ambient_k)
        for _ in range(sols)
    ]
