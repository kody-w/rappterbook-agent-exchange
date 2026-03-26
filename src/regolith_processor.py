"""
regolith_processor.py — Mars regolith excavation and processing model.

Models the ISRU pipeline that turns raw Mars dirt into construction
materials, radiation shielding aggregate, and greenhouse soil amendment.

This is the foundational industry on Mars.  Before you build a second
habitat, before you expand the greenhouse, before you upgrade the rad
shield — you process regolith.  Everything on Mars starts with dirt.

Physical references:
  - Mars regolith bulk density: ~1500 kg/m³ (Viking, Phoenix)
  - Regolith composition: 43% SiO₂, 18% Fe₂O₃, 6% Al₂O₃ (MER Mössbauer)
  - Iron oxide content: 14-18% by mass (gives Mars its red color)
  - Sintering temperature for regolith bricks: 1000-1100°C
  - Microwave sintering energy: ~2.5 kWh/kg (NASA KSC experiments)
  - Compressive strength of sintered bricks: 3-20 MPa (depends on process)
  - Particle size: 1 μm-1 mm, mean ~100 μm (Viking, Spirit)
  - Perchlorate content: 0.5-1.0% by mass (Phoenix) — toxic, must be washed

One tick = one sol.  Mass in kg, energy in kWh, volume in m³.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

REGOLITH_DENSITY_KG_M3 = 1500.0        # bulk density of loose regolith
IRON_OXIDE_FRACTION = 0.17             # Fe₂O₃ mass fraction (MER average)
SILICA_FRACTION = 0.43                  # SiO₂ mass fraction
PERCHLORATE_FRACTION = 0.007            # ClO₄⁻ mass fraction (Phoenix avg)
WATER_WASH_RATIO = 2.0                  # liters water per kg regolith to wash perchlorates

# Energy costs (kWh per kg of output)
EXCAVATION_KWH_PER_M3 = 5.0            # bucket-wheel or scoop excavation
SIEVING_KWH_PER_M3 = 1.0               # mechanical vibration sieving
SINTERING_KWH_PER_KG = 2.5             # microwave sintering (NASA KSC)
IRON_EXTRACTION_KWH_PER_KG = 4.0       # hydrogen reduction of Fe₂O₃
WASHING_KWH_PER_M3 = 0.5               # water circulation for perchlorate removal

# Sintering parameters
SINTER_TEMP_C = 1050.0                  # target sintering temperature
SINTER_TEMP_MIN_C = 900.0              # minimum viable sintering temp
SINTER_TEMP_MAX_C = 1200.0             # above this, excessive energy waste
BRICK_STRENGTH_MPA_BASE = 10.0         # compressive strength at optimal temp
BRICK_MASS_KG = 12.0                   # standard brick mass
BRICK_VOLUME_M3 = 0.008                # standard brick: 0.2 x 0.1 x 0.4 m

# Efficiency factors
COLD_EXCAVATION_PENALTY = 0.60         # excavation efficiency at -120°C
WARM_EXCAVATION_BONUS = 1.0            # excavation efficiency at 0°C
DUST_STORM_HALT_THRESHOLD = 0.7        # stop outdoor excavation above this opacity
EQUIPMENT_WEAR_PER_SOL = 0.0002        # 0.02% degradation per sol from abrasion


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegolithStockpile:
    """Excavated and processed regolith inventory.

    raw_kg: unprocessed regolith awaiting sieving/washing
    sieved_kg: sieved regolith (particle-size sorted, not washed)
    washed_kg: perchlorate-free regolith (safe for greenhouse use)
    bricks: count of sintered construction bricks
    iron_kg: extracted metallic iron
    """
    raw_kg: float = 0.0
    sieved_kg: float = 0.0
    washed_kg: float = 0.0
    bricks: int = 0
    iron_kg: float = 0.0

    def __post_init__(self) -> None:
        self.raw_kg = max(0.0, self.raw_kg)
        self.sieved_kg = max(0.0, self.sieved_kg)
        self.washed_kg = max(0.0, self.washed_kg)
        self.bricks = max(0, self.bricks)
        self.iron_kg = max(0.0, self.iron_kg)

    def total_processed_kg(self) -> float:
        """Total mass of all processed materials (sieved + washed + bricks + iron)."""
        return (
            self.sieved_kg
            + self.washed_kg
            + self.bricks * BRICK_MASS_KG
            + self.iron_kg
        )


@dataclass
class ProcessorState:
    """State of the regolith processing facility.

    equipment_condition: health of excavation/processing gear [0, 1]
    sinter_furnace_temp_c: current furnace temperature
    water_available_liters: water budget for perchlorate washing
    """
    equipment_condition: float = 1.0
    sinter_furnace_temp_c: float = 20.0
    water_available_liters: float = 0.0

    def __post_init__(self) -> None:
        self.equipment_condition = max(0.0, min(1.0, self.equipment_condition))
        self.sinter_furnace_temp_c = max(-120.0, self.sinter_furnace_temp_c)
        self.water_available_liters = max(0.0, self.water_available_liters)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def excavation_efficiency(temp_c: float, dust_opacity: float) -> float:
    """Outdoor excavation efficiency based on temperature and dust.

    Returns 0.0 if dust storm halts operations.
    Linear ramp from COLD_EXCAVATION_PENALTY (-120°C) to
    WARM_EXCAVATION_BONUS (0°C).
    """
    if dust_opacity >= DUST_STORM_HALT_THRESHOLD:
        return 0.0
    t = max(-120.0, min(0.0, temp_c))
    frac = (t + 120.0) / 120.0
    temp_eff = COLD_EXCAVATION_PENALTY + frac * (WARM_EXCAVATION_BONUS - COLD_EXCAVATION_PENALTY)
    dust_penalty = 1.0 - 0.3 * min(1.0, dust_opacity / DUST_STORM_HALT_THRESHOLD)
    return temp_eff * dust_penalty


def sintering_quality(furnace_temp_c: float) -> float:
    """Brick quality factor [0, 1] based on furnace temperature.

    Below SINTER_TEMP_MIN_C: no sintering occurs (returns 0).
    Optimal at SINTER_TEMP_C: returns 1.0.
    Above SINTER_TEMP_MAX_C: energy waste but quality plateaus at 1.0.
    """
    if furnace_temp_c < SINTER_TEMP_MIN_C:
        return 0.0
    if furnace_temp_c > SINTER_TEMP_MAX_C:
        return 1.0
    # Ramp from 0 to 1 between min and optimal
    if furnace_temp_c <= SINTER_TEMP_C:
        return (furnace_temp_c - SINTER_TEMP_MIN_C) / (SINTER_TEMP_C - SINTER_TEMP_MIN_C)
    # Slight bonus zone above optimal (still 1.0)
    return 1.0


def brick_strength_mpa(quality: float) -> float:
    """Compressive strength of a sintered brick given quality factor.

    Low quality (0.3): ~3 MPa (barely holds together)
    Optimal quality (1.0): 10 MPa (good construction material)
    """
    return BRICK_STRENGTH_MPA_BASE * max(0.0, min(1.0, quality))


def perchlorate_removal_efficiency(water_ratio: float) -> float:
    """Fraction of perchlorates removed given water-to-regolith ratio.

    At WATER_WASH_RATIO (2.0 L/kg), removes 95% of perchlorates.
    Below 0.5 L/kg, negligible removal.
    Diminishing returns above 3.0 L/kg.
    """
    if water_ratio <= 0.0:
        return 0.0
    if water_ratio < 0.5:
        return water_ratio / 0.5 * 0.1
    if water_ratio >= 3.0:
        return 0.98
    # Sigmoid-like ramp from 0.1 to 0.95
    frac = (water_ratio - 0.5) / (WATER_WASH_RATIO - 0.5)
    return 0.1 + 0.85 * min(1.0, frac)


# ---------------------------------------------------------------------------
# Per-sol processing functions
# ---------------------------------------------------------------------------

def excavate_sol(
    power_kwh: float,
    temp_c: float,
    dust_opacity: float,
    equipment_condition: float,
) -> tuple[float, float]:
    """Excavate raw regolith for one sol.

    Args:
        power_kwh: power budget for excavation
        temp_c: surface temperature
        dust_opacity: current dust opacity [0, 1]
        equipment_condition: gear health [0, 1]

    Returns:
        (regolith_kg, power_consumed_kwh)

    Conservation: never consumes more power than allocated.
    """
    if power_kwh <= 0.0 or equipment_condition <= 0.0:
        return (0.0, 0.0)

    cond = max(0.0, min(1.0, equipment_condition))
    eff = excavation_efficiency(temp_c, dust_opacity) * cond

    if eff <= 0.0:
        return (0.0, 0.0)

    effective_power = power_kwh * eff
    volume_m3 = effective_power / EXCAVATION_KWH_PER_M3
    regolith_kg = volume_m3 * REGOLITH_DENSITY_KG_M3

    return (round(regolith_kg, 4), round(min(power_kwh, power_kwh), 4))


def sieve_sol(
    raw_kg: float,
    power_kwh: float,
) -> tuple[float, float, float]:
    """Sieve raw regolith into sorted particle sizes.

    Args:
        raw_kg: available raw regolith
        power_kwh: power budget for sieving

    Returns:
        (sieved_kg, raw_consumed_kg, power_consumed_kwh)
    """
    if raw_kg <= 0.0 or power_kwh <= 0.0:
        return (0.0, 0.0, 0.0)

    # Volume we can sieve given power
    volume_capacity_m3 = power_kwh / SIEVING_KWH_PER_M3
    mass_capacity_kg = volume_capacity_m3 * REGOLITH_DENSITY_KG_M3

    processed_kg = min(raw_kg, mass_capacity_kg)
    # ~5% loss to ultra-fine dust that escapes sieving
    sieved_kg = processed_kg * 0.95

    power_used = power_kwh * (processed_kg / mass_capacity_kg) if mass_capacity_kg > 0 else 0.0

    return (round(sieved_kg, 4), round(processed_kg, 4), round(min(power_used, power_kwh), 4))


def wash_sol(
    sieved_kg: float,
    water_liters: float,
    power_kwh: float,
) -> tuple[float, float, float, float]:
    """Wash sieved regolith to remove perchlorates.

    Args:
        sieved_kg: available sieved regolith
        water_liters: water budget for washing
        power_kwh: power for water circulation pumps

    Returns:
        (washed_kg, sieved_consumed_kg, water_consumed_liters, power_consumed_kwh)
    """
    if sieved_kg <= 0.0 or water_liters <= 0.0 or power_kwh <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)

    # Power-limited throughput
    volume_capacity_m3 = power_kwh / WASHING_KWH_PER_M3
    mass_power_limit_kg = volume_capacity_m3 * REGOLITH_DENSITY_KG_M3

    # Water-limited throughput
    mass_water_limit_kg = water_liters / WATER_WASH_RATIO

    processed_kg = min(sieved_kg, mass_power_limit_kg, mass_water_limit_kg)

    if processed_kg <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)

    water_used = processed_kg * WATER_WASH_RATIO
    # 80% of wash water is recoverable (reclaimed for reuse)
    water_consumed = water_used * 0.2

    # Perchlorate removal at the achieved water ratio
    actual_ratio = water_used / processed_kg if processed_kg > 0 else 0.0
    removal_eff = perchlorate_removal_efficiency(actual_ratio)

    # Only count as "washed" if >90% perchlorates removed
    washed_kg = processed_kg if removal_eff >= 0.90 else processed_kg * removal_eff

    power_used = power_kwh * (processed_kg / mass_power_limit_kg) if mass_power_limit_kg > 0 else 0.0

    return (
        round(washed_kg, 4),
        round(processed_kg, 4),
        round(min(water_consumed, water_liters), 4),
        round(min(power_used, power_kwh), 4),
    )


def sinter_bricks_sol(
    sieved_kg: float,
    power_kwh: float,
    furnace_temp_c: float,
) -> tuple[int, float, float, float]:
    """Sinter regolith into construction bricks.

    Args:
        sieved_kg: available sieved (or washed) regolith
        power_kwh: power budget for sintering
        furnace_temp_c: current furnace temperature

    Returns:
        (bricks_produced, quality_factor, sieved_consumed_kg, power_consumed_kwh)
    """
    quality = sintering_quality(furnace_temp_c)
    if quality <= 0.0 or sieved_kg <= 0.0 or power_kwh <= 0.0:
        return (0, 0.0, 0.0, 0.0)

    # Power-limited brick production
    mass_from_power = power_kwh / SINTERING_KWH_PER_KG
    # Material-limited
    mass_available = sieved_kg

    mass_used = min(mass_from_power, mass_available)
    bricks = int(mass_used / BRICK_MASS_KG)

    if bricks <= 0:
        return (0, quality, 0.0, 0.0)

    actual_mass = bricks * BRICK_MASS_KG
    power_used = actual_mass * SINTERING_KWH_PER_KG

    return (
        bricks,
        round(quality, 4),
        round(actual_mass, 4),
        round(min(power_used, power_kwh), 4),
    )


def extract_iron_sol(
    sieved_kg: float,
    power_kwh: float,
) -> tuple[float, float, float]:
    """Extract metallic iron from regolith via hydrogen reduction of Fe₂O₃.

    Fe₂O₃ + 3H₂ → 2Fe + 3H₂O (exothermic but needs activation energy).

    Args:
        sieved_kg: available sieved regolith
        power_kwh: power budget for extraction

    Returns:
        (iron_kg, sieved_consumed_kg, power_consumed_kwh)
    """
    if sieved_kg <= 0.0 or power_kwh <= 0.0:
        return (0.0, 0.0, 0.0)

    # Iron available in regolith (Fe₂O₃ is 70% iron by mass)
    fe2o3_kg = sieved_kg * IRON_OXIDE_FRACTION
    potential_iron_kg = fe2o3_kg * 0.70  # Fe from Fe₂O₃

    # Power-limited extraction
    iron_from_power = power_kwh / IRON_EXTRACTION_KWH_PER_KG
    iron_kg = min(potential_iron_kg, iron_from_power)

    if iron_kg <= 0.0:
        return (0.0, 0.0, 0.0)

    # Regolith consumed: all of it (iron is extracted, silicates remain as slag)
    regolith_consumed = iron_kg / (IRON_OXIDE_FRACTION * 0.70)
    regolith_consumed = min(regolith_consumed, sieved_kg)
    power_used = iron_kg * IRON_EXTRACTION_KWH_PER_KG

    return (
        round(iron_kg, 4),
        round(regolith_consumed, 4),
        round(min(power_used, power_kwh), 4),
    )


# ---------------------------------------------------------------------------
# Integrated per-sol tick
# ---------------------------------------------------------------------------

def tick_regolith(
    power_budget_kwh: float,
    temp_c: float,
    dust_opacity: float,
    water_liters: float,
    stockpile: RegolithStockpile,
    state: ProcessorState,
    allocation: dict[str, float] | None = None,
) -> dict:
    """Run one sol of regolith processing.

    Allocates power budget across operations.  Default allocation:
      40% excavation, 10% sieving, 15% washing, 25% sintering, 10% iron

    Args:
        power_budget_kwh: total power allocated to regolith ops
        temp_c: surface temperature
        dust_opacity: dust opacity [0, 1]
        water_liters: water available for washing
        stockpile: current material inventory (mutated in place)
        state: processor facility state (mutated in place)
        allocation: optional power allocation fractions

    Returns:
        dict with per-sol production summary
    """
    alloc = allocation or {
        "excavation": 0.40,
        "sieving": 0.10,
        "washing": 0.15,
        "sintering": 0.25,
        "iron": 0.10,
    }

    # Normalize allocation
    total_alloc = sum(alloc.values())
    if total_alloc <= 0:
        total_alloc = 1.0

    pw_exc = power_budget_kwh * alloc.get("excavation", 0.0) / total_alloc
    pw_sieve = power_budget_kwh * alloc.get("sieving", 0.0) / total_alloc
    pw_wash = power_budget_kwh * alloc.get("washing", 0.0) / total_alloc
    pw_sinter = power_budget_kwh * alloc.get("sintering", 0.0) / total_alloc
    pw_iron = power_budget_kwh * alloc.get("iron", 0.0) / total_alloc

    # --- Excavation ---
    raw_gained, pw_exc_used = excavate_sol(
        pw_exc, temp_c, dust_opacity, state.equipment_condition
    )
    stockpile.raw_kg += raw_gained

    # --- Sieving ---
    sieved_gained, raw_consumed, pw_sieve_used = sieve_sol(
        stockpile.raw_kg, pw_sieve
    )
    stockpile.raw_kg -= raw_consumed
    stockpile.sieved_kg += sieved_gained

    # --- Washing (uses sieved material + water) ---
    washed_gained, sieved_wash_consumed, water_used, pw_wash_used = wash_sol(
        stockpile.sieved_kg, water_liters, pw_wash
    )
    stockpile.sieved_kg -= sieved_wash_consumed
    stockpile.washed_kg += washed_gained
    state.water_available_liters = max(0.0, state.water_available_liters - water_used)

    # --- Sintering (uses sieved material) ---
    bricks_made, quality, sieved_sinter_consumed, pw_sinter_used = sinter_bricks_sol(
        stockpile.sieved_kg, pw_sinter, state.sinter_furnace_temp_c
    )
    stockpile.sieved_kg -= sieved_sinter_consumed
    stockpile.bricks += bricks_made

    # --- Iron extraction (uses sieved material) ---
    iron_gained, sieved_iron_consumed, pw_iron_used = extract_iron_sol(
        stockpile.sieved_kg, pw_iron
    )
    stockpile.sieved_kg -= sieved_iron_consumed
    stockpile.iron_kg += iron_gained

    # --- Equipment wear ---
    state.equipment_condition = max(
        0.0,
        state.equipment_condition - EQUIPMENT_WEAR_PER_SOL,
    )

    total_power_used = pw_exc_used + pw_sieve_used + pw_wash_used + pw_sinter_used + pw_iron_used

    return {
        "excavated_kg": round(raw_gained, 2),
        "sieved_kg": round(sieved_gained, 2),
        "washed_kg": round(washed_gained, 2),
        "bricks": bricks_made,
        "brick_quality": round(quality, 4),
        "brick_strength_mpa": round(brick_strength_mpa(quality), 2),
        "iron_kg": round(iron_gained, 4),
        "power_consumed_kwh": round(total_power_used, 2),
        "water_consumed_liters": round(water_used, 2),
        "equipment_condition": round(state.equipment_condition, 4),
        "stockpile": {
            "raw_kg": round(stockpile.raw_kg, 2),
            "sieved_kg": round(stockpile.sieved_kg, 2),
            "washed_kg": round(stockpile.washed_kg, 2),
            "bricks": stockpile.bricks,
            "iron_kg": round(stockpile.iron_kg, 4),
        },
    }
